use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, Instant},
};

use block2::RcBlock;
use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, Layout, Op, Submission, Tensor,
    required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions, MTLAllocation,
    MTLBuffer, MTLComputePipelineState, MTLDataType, MTLDevice, MTLEvent,
    MTLFunctionConstantValues, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent, MTLSharedEventListener,
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
    commit: CommitRetention,
}

type FeedbackHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTL4CommitFeedback>>)>;

struct CommitRetention {
    _handler: FeedbackHandler,
    options: Retained<MTL4CommitOptions>,
}

// SAFETY: The commit options are immutable after registration, and the block captures only a
// thread-safe weak completion reference. Objective-C blocks and objects may be retained and
// released from Metal callback queues.
unsafe impl Send for CommitRetention {}

// SAFETY: Shared access can only retain the immutable commit objects; callback state is protected
// by the completion mutex.
unsafe impl Sync for CommitRetention {}

impl Completion {
    pub(super) fn new(
        buffers: Vec<InFlightBuffer>,
        event: InFlightEvent,
        residency: InFlightResidency,
    ) -> Arc<Self> {
        Arc::new_cyclic(|completion: &Weak<Self>| {
            let callback_completion = completion.clone();
            let handler: FeedbackHandler = RcBlock::new(
                move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    // SAFETY: Metal supplies a live, non-null feedback object for this call.
                    let feedback = unsafe { feedback.as_ref() };
                    let result = if feedback.error().is_some() {
                        Err(BackendError::ExecutionFailed)
                    } else {
                        Ok(())
                    };
                    if let Some(completion) = callback_completion.upgrade() {
                        completion.finish(result);
                    }
                },
            );
            let options = MTL4CommitOptions::new();
            // SAFETY: `handler` is a live heap block. This completion owns both the block and the
            // options, and the in-flight tracker retains the completion until the queue event and
            // feedback callback have both completed.
            unsafe {
                options.addFeedbackHandler(RcBlock::as_ptr(&handler));
            }
            Self {
                state: Mutex::new(CompletionState { result: None }),
                ready: Condvar::new(),
                event,
                _buffers: buffers,
                _residency: residency,
                commit: CommitRetention {
                    _handler: handler,
                    options,
                },
            }
        })
    }

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

    fn wait_unbounded(&self) {
        let _signaled = self.event.raw.waitUntilSignaledValue_timeoutMS(1, u64::MAX);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
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

type NotificationHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;

struct InFlightCompletion {
    completion: Arc<Completion>,
    _listener: Retained<MTLSharedEventListener>,
    _notification: NotificationHandler,
}

// SAFETY: The listener is thread-safe, and the immutable notification block captures only
// thread-safe `Arc` values. Objective-C blocks may be retained and released on the listener queue.
unsafe impl Send for InFlightCompletion {}

// SAFETY: Shared access only retains the listener, block, and completion; their mutable state is
// protected by mutexes.
unsafe impl Sync for InFlightCompletion {}

pub(super) struct InFlightTracker {
    completions: Mutex<Vec<InFlightCompletion>>,
}

impl InFlightTracker {
    pub(super) const fn new() -> Self {
        Self {
            completions: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn track(
        self: &Arc<Self>,
        completion: &Arc<Completion>,
        listener: &Retained<MTLSharedEventListener>,
    ) -> Result<(), BackendError> {
        let pending = Arc::clone(completion);
        let owner = Arc::clone(self);
        let notification: NotificationHandler = RcBlock::new(move |_event, _value| {
            let active_completion = Arc::clone(&pending);
            let active_owner = Arc::clone(&owner);
            active_completion.wait_unbounded();
            active_owner.remove(&active_completion);
        });
        self.completions
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .push(InFlightCompletion {
                completion: Arc::clone(completion),
                _listener: listener.clone(),
                _notification: notification.clone(),
            });
        // SAFETY: The tracker stores the live heap block and listener before registration. The
        // block owns `Arc`s to both the completion and tracker, and clones them before removal, so
        // its captures remain valid even if the backend is dropped before Metal invokes it.
        unsafe {
            completion.event.raw.notifyListener_atValue_block(
                listener,
                1,
                RcBlock::as_ptr(&notification),
            );
        }
        Ok(())
    }

    fn remove(&self, completion: &Arc<Completion>) {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|candidate| !Arc::ptr_eq(&candidate.completion, completion));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

fn timeout_millis(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX)
}

/// Completion state for a Metal command buffer.
pub struct MetalSubmission {
    completion: Arc<Completion>,
    timeout: Duration,
}

impl Submission for MetalSubmission {
    fn wait(self) -> Result<(), BackendError> {
        self.completion.wait(self.timeout)
    }
}

impl MetalBackend {
    pub(super) fn submit_commands(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
        let barriers = required_barriers(&commands);
        let dispatches = commands.into_dispatches();
        if dispatches
            .iter()
            .any(|dispatch| !matches!(dispatch.op(), Op::Copy | Op::Add | Op::SiluMul))
        {
            return Err(BackendError::InvalidInput);
        }
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
        let command_buffer = self.begin_command_buffer()?;
        let temporaries = self.encode_dispatches(&command_buffer, &dispatches, &barriers)?;
        let residency = self.make_resident(&command_buffer, &tensors, &temporaries)?;
        command_buffer.endCommandBuffer();
        self.commit(&command_buffer, &tensors, &temporaries, residency)
    }

    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        dispatches: &[Dispatch],
        barriers: &[bool],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::{
            MTL4ArgumentTableDescriptor, MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages,
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(BackendError::ExecutionFailed)?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(6);
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
            let kernel = match dispatch.op() {
                Op::Copy
                    if dispatch.inputs()[0].layout().is_contiguous()
                        && dispatch.output().layout().is_contiguous() =>
                {
                    "copy_contiguous"
                }
                Op::Copy => "copy_strided",
                Op::Add
                    if dispatch
                        .inputs()
                        .iter()
                        .chain(std::iter::once(dispatch.output()))
                        .all(|tensor| tensor.layout().is_contiguous()) =>
                {
                    "add_contiguous"
                }
                Op::Add => "add_strided",
                Op::SiluMul => "silu_mul",
                _ => return Err(BackendError::InvalidInput),
            };
            temporaries.extend(self.encode_elementwise(&encoder, &table, dispatch, kernel)?);
        }
        encoder.endEncoding();
        Ok(temporaries)
    }

    fn encode_elementwise(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        kernel: &str,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::{MTL4ArgumentTable, MTL4ComputeCommandEncoder, MTLSize};

        let operands = dispatch
            .inputs()
            .iter()
            .chain(std::iter::once(dispatch.output()))
            .collect::<Vec<_>>();
        let constants = dispatch
            .inputs()
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                Ok((
                    u32::try_from(index).map_err(|_| BackendError::InvalidInput)?,
                    dtype_code(tensor.layout().dtype()),
                ))
            })
            .chain(std::iter::once(Ok((
                2,
                dtype_code(dispatch.output().layout().dtype()),
            ))))
            .collect::<Result<Vec<_>, BackendError>>()?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        encoder.setComputePipelineState(&pipeline);
        let layouts = operands
            .iter()
            .map(|tensor| self.layout_buffer(tensor.layout()))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in operands.iter().enumerate() {
            let buffer = buffers.get(tensor)?;
            // SAFETY: The argument table has room for every operand and layout address, and all
            // resources are retained through command-buffer completion.
            unsafe {
                table.setAddress_atIndex(buffer.raw.gpuAddress(), index);
                table.setAddress_atIndex(layouts[index].gpuAddress(), index + operands.len());
            }
        }
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let thread_count = usize::try_from(dispatch.output().layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: thread_count.div_ceil(group_width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        Ok(layouts)
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

    fn retain_tensors(
        &self,
        tensors: &[Tensor],
        event: InFlightEvent,
        temporaries: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        residency: InFlightResidency,
    ) -> Result<Arc<Completion>, BackendError> {
        let mut seen = HashSet::<BufferId>::new();
        let unique = tensors
            .iter()
            .filter(|tensor| seen.insert(tensor.buffer()))
            .collect::<Vec<_>>();
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut retained = Vec::with_capacity(unique.len().saturating_add(temporaries.len()));
        for tensor in &unique {
            let buffer = buffers.get_mut(tensor)?;
            buffer.wait_pending(self.gpu_timeout)?;
            retained.push(InFlightBuffer {
                _raw: buffer.raw.clone(),
            });
        }
        retained.extend(
            temporaries
                .iter()
                .cloned()
                .map(|raw| InFlightBuffer { _raw: raw }),
        );
        let completion = Completion::new(retained, event, residency);
        for tensor in unique {
            buffers.get_mut(tensor)?.track(&completion);
        }
        Ok(completion)
    }

    fn commit(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        tensors: &[Tensor],
        temporaries: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    ) -> Result<MetalSubmission, BackendError> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or(BackendError::ExecutionFailed)?;
        let completion = self.retain_tensors(
            tensors,
            InFlightEvent { raw: event.clone() },
            temporaries,
            InFlightResidency { _raw: residency },
        )?;
        self.in_flight.track(&completion, &self.event_listener)?;
        let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = command_buffer;
        let mut command_buffers = [NonNull::from(command_buffer_ref)];
        // SAFETY: The pointer names one live command buffer and the count matches the array.
        unsafe {
            self.queue.commit_count_options(
                NonNull::from(&mut command_buffers[0]),
                command_buffers.len(),
                &completion.commit.options,
            );
        }
        let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &event;
        let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
        self.queue.signalEvent_value(event, 1);
        Ok(MetalSubmission {
            completion,
            timeout: self.gpu_timeout,
        })
    }
}

const fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
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
    use std::time::{Duration, Instant};

    use forja_core::{Backend, CommandList, DType, Op, Slice, Submission};
    use forja_cpu::CpuBackend;
    use forja_testing::{TensorSpec, assert_backends_agree};
    use objc2_metal::{MTL4CommandQueue, MTLEvent};

    use super::*;

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
    }

    #[test]
    fn shared_event_listener_reaps_many_command_buffers() {
        let backend = MetalBackend::new().unwrap();
        let queue = backend.device.newMTL4CommandQueue().unwrap();
        let descriptor = MTLResidencySetDescriptor::new();
        let residency = backend
            .device
            .newResidencySetWithDescriptor_error(&descriptor)
            .unwrap();
        residency.commit();

        for _ in 0..1000 {
            let command_buffer = backend.begin_command_buffer().unwrap();
            command_buffer.endCommandBuffer();
            let event = backend.device.newSharedEvent().unwrap();
            let completion = Completion::new(
                Vec::new(),
                InFlightEvent { raw: event.clone() },
                InFlightResidency {
                    _raw: residency.clone(),
                },
            );
            completion.finish(Ok(()));
            backend
                .in_flight
                .track(&completion, &backend.event_listener)
                .unwrap();
            let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = &command_buffer;
            let mut command_buffers = [NonNull::from(command_buffer_ref)];
            // SAFETY: The pointer names one live command buffer and the count matches the array.
            unsafe {
                queue.commit_count(
                    NonNull::from(&mut command_buffers[0]),
                    command_buffers.len(),
                );
            }
            let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &event;
            let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
            queue.signalEvent_value(event, 1);
        }

        let started = Instant::now();
        while backend.in_flight.len() != 0 {
            assert!(started.elapsed() < backend.gpu_timeout);
            std::thread::yield_now();
        }
    }

    #[test]
    fn metal_empty_command_list_completes() {
        let backend = MetalBackend::new().unwrap();
        backend.submit(CommandList::new()).unwrap().wait().unwrap();
    }

    #[test]
    fn dropped_submissions_retain_released_buffers() {
        let backend = MetalBackend::new().unwrap();
        for _ in 0..100 {
            let gate = backend.alloc(DType::F32, &[4097]).unwrap();
            let up = backend.alloc(DType::F32, &[4097]).unwrap();
            let output = backend.alloc(DType::F32, &[4097]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::SiluMul, &[&gate, &up], &output)
                .unwrap();
            let submission = backend.submit(commands).unwrap();
            drop(submission);
            backend.release(&gate).unwrap();
            backend.release(&up).unwrap();
            backend.release(&output).unwrap();
        }
        drop(backend);
    }

    #[test]
    fn timed_out_work_never_exposes_an_active_output() {
        let backend = MetalBackend::with_gpu_timeout(Duration::from_micros(1)).unwrap();
        let shape = [1024, 4097];
        let gate = backend.alloc(DType::F32, &shape).unwrap();
        let up = backend.alloc(DType::F32, &shape).unwrap();
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::SiluMul, &[&gate, &up], &output)
            .unwrap();

        assert_eq!(
            backend.submit(commands).unwrap().wait(),
            Err(BackendError::ExecutionFailed)
        );
        match backend.read(&output) {
            Ok(bytes) => assert_eq!(bytes.len(), 1024 * 4097 * 4),
            Err(error) => assert_eq!(error, BackendError::ExecutionFailed),
        }
    }

    #[test]
    fn metal_silu_mul_matches_cpu_for_dtypes_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for shape in [&[7, 3072][..], &[1, 3072], &[33, 4097]] {
                let inputs = [
                    TensorSpec::contiguous(dtype, shape),
                    TensorSpec::contiguous(dtype, shape),
                ];
                let output = TensorSpec::contiguous(dtype, shape);
                assert_backends_agree(&reference, &candidate, Op::SiluMul, &inputs, &output)
                    .unwrap();
            }
        }

        let permuted = [
            TensorSpec::permuted(DType::F32, &[3072, 7], &[1, 0]),
            TensorSpec::contiguous(DType::F32, &[7, 3072]),
        ];
        assert_backends_agree(
            &reference,
            &candidate,
            Op::SiluMul,
            &permuted,
            &TensorSpec::contiguous(DType::F32, &[7, 3072]),
        )
        .unwrap();

        let mixed = [
            TensorSpec::contiguous(DType::F16, &[7, 3072]),
            TensorSpec::contiguous(DType::BF16, &[7, 3072]),
        ];
        assert_backends_agree(
            &reference,
            &candidate,
            Op::SiluMul,
            &mixed,
            &TensorSpec::contiguous(DType::F32, &[7, 3072]),
        )
        .unwrap();
    }

    #[test]
    fn metal_copy_matches_cpu_for_casts_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let floats = [DType::F32, DType::F16, DType::BF16];
        for input_dtype in floats {
            for output_dtype in floats {
                for width in [1, 7, 33, 4097] {
                    assert_backends_agree(
                        &reference,
                        &candidate,
                        Op::Copy,
                        &[TensorSpec::contiguous(input_dtype, &[width])],
                        &TensorSpec::contiguous(output_dtype, &[width]),
                    )
                    .unwrap();
                }
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Copy,
                    &[TensorSpec::permuted(input_dtype, &[7, 33], &[1, 0])],
                    &TensorSpec::contiguous(output_dtype, &[33, 7]),
                )
                .unwrap();
            }
        }
        for dtype in [DType::I32, DType::U32] {
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Copy,
                &[TensorSpec::permuted(dtype, &[7, 33], &[1, 0])],
                &TensorSpec::contiguous(dtype, &[33, 7]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_copy_writes_a_kv_cache_slice() {
        let slices = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(37, 1, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Copy,
            &[TensorSpec::contiguous(DType::F16, &[8, 1, 128])],
            &TensorSpec::sliced(DType::F16, &[8, 4096, 128], &slices),
        )
        .unwrap();
    }

    #[test]
    fn metal_add_matches_cpu_for_dtypes_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097] {
                let input = TensorSpec::contiguous(dtype, &[width]);
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Add,
                    &[input.clone(), input],
                    &TensorSpec::contiguous(dtype, &[width]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Add,
                &[
                    TensorSpec::permuted(dtype, &[7, 33], &[1, 0]),
                    TensorSpec::broadcast(dtype, &[1, 7], &[33, 7]),
                ],
                &TensorSpec::contiguous(dtype, &[33, 7]),
            )
            .unwrap();
        }
    }
}
