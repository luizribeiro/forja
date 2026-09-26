use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, Layout, Tensor, required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandBuffer, MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent,
};

use crate::storage::MetalBackend;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    name: String,
    constants: Vec<(u32, u32)>,
}

struct CompletionState {
    result: Option<Result<(), BackendError>>,
}

pub(super) struct InFlightBuffer {
    pub(super) _raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// SAFETY: Metal buffer resources support concurrent retain and release, and this wrapper never
// exposes CPU access to their contents.
unsafe impl Send for InFlightBuffer {}

// SAFETY: The wrapper only keeps a Metal buffer alive and provides no access to its contents.
unsafe impl Sync for InFlightBuffer {}

pub(super) struct InFlightEvent {
    pub(super) raw: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// SAFETY: Shared events are designed for cross-thread signaling and waiting, and the wrapper
// exposes only the thread-safe wait operation.
unsafe impl Send for InFlightEvent {}

// SAFETY: Concurrent waits do not mutate the retained shared event through Rust references.
unsafe impl Sync for InFlightEvent {}

pub(super) struct InFlightResidency {
    pub(super) _raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: The residency set is committed before submission and remains immutable while shared
// across completion and waiting threads.
unsafe impl Send for InFlightResidency {}

// SAFETY: The wrapper exposes no operations on the retained residency set.
unsafe impl Sync for InFlightResidency {}

pub(super) struct Completion {
    state: Mutex<CompletionState>,
    ready: Condvar,
    event: InFlightEvent,
    _buffers: Vec<InFlightBuffer>,
    _residency: InFlightResidency,
}

impl Completion {
    #[expect(dead_code)]
    pub(super) fn new(
        buffers: Vec<InFlightBuffer>,
        event: InFlightEvent,
        residency: InFlightResidency,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(CompletionState { result: None }),
            ready: Condvar::new(),
            event,
            _buffers: buffers,
            _residency: residency,
        })
    }

    #[expect(dead_code)]
    pub(super) fn finish(&self, result: Result<(), BackendError>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.result = Some(result);
        self.ready.notify_all();
    }

    pub(super) fn wait(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        if !self
            .event
            .raw
            .waitUntilSignaledValue_timeoutMS(1, timeout_millis(timeout))
        {
            return Err(BackendError::ExecutionFailed);
        }
        self.wait_for_feedback(timeout.saturating_sub(started.elapsed()))
    }

    fn wait_for_feedback(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(result) = state.result {
                return result;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(BackendError::ExecutionFailed);
            }
            let (next, wait) = self
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && state.result.is_none() {
                return Err(BackendError::ExecutionFailed);
            }
        }
    }
}

fn timeout_millis(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX)
}

impl MetalBackend {
    pub(super) fn encode_commands(&self, commands: CommandList) -> Result<bool, BackendError> {
        let barriers = required_barriers(&commands);
        let dispatches = commands.into_dispatches();
        let tensors = dispatches
            .iter()
            .flat_map(|dispatch| {
                dispatch
                    .inputs()
                    .iter()
                    .chain(std::iter::once(dispatch.output()))
            })
            .cloned()
            .collect::<Vec<_>>();
        for tensor in &tensors {
            self.validate(tensor)?;
        }
        if dispatches.is_empty() {
            return Ok(false);
        }
        let command_buffer = self.begin_command_buffer()?;
        let temporaries = self.encode_dispatches(&command_buffer, &dispatches, &barriers)?;
        let _residency = self.make_resident(&command_buffer, &tensors, &temporaries)?;
        command_buffer.endCommandBuffer();
        Ok(true)
    }

    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        dispatches: &[Dispatch],
        barriers: &[bool],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::{
            MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandEncoder,
            MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLSize, MTLStages,
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(BackendError::ExecutionFailed)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("hold", &[])?;
        encoder.setComputePipelineState(&pipeline);
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(1);
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut temporaries = Vec::with_capacity(dispatches.len().saturating_mul(3));
        for (dispatch, &barrier) in dispatches.iter().zip(barriers) {
            if barrier {
                encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            }
            for tensor in dispatch
                .inputs()
                .iter()
                .chain(std::iter::once(dispatch.output()))
            {
                let _code = dtype_code(tensor.layout().dtype())?;
                temporaries.push(self.layout_buffer(tensor.layout())?);
            }
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            let output = buffers.get(dispatch.output())?;
            // SAFETY: The descriptor created one buffer slot, index zero is in range, and the
            // registered buffer remains live while the command buffer is encoded.
            unsafe {
                table.setAddress_atIndex(output.raw.gpuAddress(), 0);
            }
            drop(buffers);
            encoder.setArgumentTable(Some(&table));
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        Ok(temporaries)
    }

    fn layout_buffer(
        &self,
        layout: &Layout,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, BackendError> {
        use objc2_metal::MTLResourceOptions;

        let bytes = encode_layout(layout)?;
        let buffer = self
            .device
            .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
            .ok_or(BackendError::AllocationFailed)?;
        // SAFETY: `buffer` is a live shared allocation of exactly `bytes.len()` bytes and the
        // source and destination do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>().as_ptr(),
                bytes.len(),
            );
        }
        Ok(buffer)
    }

    fn begin_command_buffer(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CommandBuffer>>, BackendError> {
        let allocator = self
            .device
            .newCommandAllocator()
            .ok_or(BackendError::ExecutionFailed)?;
        let command_buffer = self
            .device
            .newCommandBuffer()
            .ok_or(BackendError::ExecutionFailed)?;
        command_buffer.beginCommandBufferWithAllocator(&allocator);
        Ok(command_buffer)
    }

    fn make_resident(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        tensors: &[Tensor],
        temporaries: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, BackendError> {
        let descriptor = MTLResidencySetDescriptor::new();
        let residency = self
            .device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut seen = HashSet::<BufferId>::new();
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for tensor in tensors.iter().filter(|tensor| seen.insert(tensor.buffer())) {
            let buffer = buffers.get(tensor)?;
            let buffer: &ProtocolObject<dyn MTLBuffer> = &buffer.raw;
            let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
            residency.addAllocation(allocation);
        }
        drop(buffers);
        for buffer in temporaries {
            let buffer: &ProtocolObject<dyn MTLBuffer> = buffer;
            let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
            residency.addAllocation(allocation);
        }
        residency.commit();
        command_buffer.useResidencySet(&residency);
        Ok(residency)
    }
}

const fn dtype_code(dtype: DType) -> Result<u32, BackendError> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::I32 | DType::U32 => Err(BackendError::InvalidInput),
    }
}

fn encode_layout(layout: &Layout) -> Result<[u8; 112], BackendError> {
    let mut bytes = [0_u8; 112];
    bytes[0..8].copy_from_slice(&layout.offset().to_ne_bytes());
    let rank = u32::try_from(layout.shape().len()).map_err(|_| BackendError::InvalidInput)?;
    bytes[8..12].copy_from_slice(&rank.to_ne_bytes());
    for (axis, &extent) in layout.shape().iter().enumerate() {
        let start = 12_usize
            .checked_add(axis.checked_mul(4).ok_or(BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)?;
        bytes[start..start + 4].copy_from_slice(&extent.to_ne_bytes());
    }
    let element_count =
        u32::try_from(layout.element_count()).map_err(|_| BackendError::InvalidInput)?;
    bytes[44..48].copy_from_slice(&element_count.to_ne_bytes());
    for (axis, &stride) in layout.strides().iter().enumerate() {
        let start = 48_usize
            .checked_add(axis.checked_mul(8).ok_or(BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)?;
        bytes[start..start + 8].copy_from_slice(&stride.to_ne_bytes());
    }
    Ok(bytes)
}

pub(super) struct PipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    pub(super) fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        source: &str,
    ) -> Result<Self, BackendError> {
        let source = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            device: device.clone(),
            library,
            pipelines: HashMap::new(),
        })
    }

    pub(super) fn get(
        &mut self,
        name: &str,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let key = PipelineKey {
            name: name.to_owned(),
            constants: constants.to_vec(),
        };
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let values = MTLFunctionConstantValues::new();
        for &(index, value) in constants {
            // SAFETY: `value` is live for the call, its type matches `MTLDataType::UInt`, and
            // the caller supplies indices declared by the selected embedded kernel.
            unsafe {
                values.setConstantValue_type_atIndex(
                    NonNull::from(&value).cast::<c_void>(),
                    MTLDataType::UInt,
                    usize::try_from(index).map_err(|_| BackendError::ExecutionFailed)?,
                );
            }
        }
        let name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&name, &values)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use forja_core::{Backend, CommandList, DType, Op, Submission};

    use super::*;

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
    }

    #[test]
    fn encoded_dispatch_reports_execution_failure_without_submission() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7]).unwrap();
        let right = backend.alloc(DType::F32, &[7]).unwrap();
        let output = backend.alloc(DType::F32, &[7]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::SiluMul, &[&left, &right], &output)
            .unwrap();
        assert_eq!(
            backend.submit(commands).unwrap().wait(),
            Err(BackendError::ExecutionFailed)
        );
    }

    #[test]
    fn submillisecond_gpu_timeouts_do_not_round_up() {
        let backend = MetalBackend::with_gpu_timeout(Duration::from_micros(1)).unwrap();
        assert_eq!(backend.gpu_timeout, Duration::from_micros(1));
        assert_eq!(timeout_millis(backend.gpu_timeout), 0);
    }
}
