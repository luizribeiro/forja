use std::{
    ptr, slice,
    sync::{Arc, Mutex, Weak},
    time::Duration,
};

use crate::encoding::{Completion, InFlightTracker, MetalSubmission, PipelineCache};
use forja_core::{
    AllocationRegistry, Backend, BackendError, CommandList, DType, Layout, Tensor, ViewOp,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTL4CommandQueue, MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
    MTLResourceOptions, MTLSharedEventListener,
};

pub(super) struct MetalBuffer {
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    pending: Vec<Weak<Completion>>,
}

impl MetalBuffer {
    fn write(&mut self, range: std::ops::Range<usize>, source: &[u8]) {
        // SAFETY: `raw` is a live shared-storage buffer of `len` bytes, the registry grants
        // exclusive CPU access, and the validated range has exactly `source.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                source.as_ptr(),
                self.raw.contents().cast::<u8>().as_ptr().add(range.start),
                source.len(),
            );
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `raw` is retained for the returned borrow and Metal guarantees `contents()`
        // points to all `len` bytes of a shared-storage buffer.
        unsafe { slice::from_raw_parts(self.raw.contents().cast::<u8>().as_ptr(), self.len) }
    }

    pub(super) fn wait_pending(&mut self, timeout: Duration) -> Result<(), BackendError> {
        let mut result = Ok(());
        for pending in std::mem::take(&mut self.pending) {
            let Some(completion) = pending.upgrade() else {
                continue;
            };
            if completion.wait(timeout).is_err() {
                result = Err(BackendError::ExecutionFailed);
                self.pending.push(Arc::downgrade(&completion));
            }
        }
        result
    }

    pub(super) fn track(&mut self, completion: &Arc<Completion>) {
        self.pending.push(Arc::downgrade(completion));
    }
}

/// A Metal 4 backend using shared unified-memory buffers.
pub struct MetalBackend {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pub(super) buffers: Mutex<AllocationRegistry<MetalBuffer>>,
    pub(super) pipelines: Mutex<PipelineCache>,
    pub(super) in_flight: Arc<InFlightTracker>,
    pub(super) event_listener: Retained<MTLSharedEventListener>,
    pub(super) gpu_timeout: Duration,
}

impl MetalBackend {
    /// Creates a backend on the system default Metal 4 device.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_gpu_timeout(Duration::from_secs(10))
    }

    /// Creates a backend with a deadline for each wait on submitted GPU work.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_gpu_timeout(gpu_timeout: Duration) -> Result<Self, BackendError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(BackendError::ExecutionFailed)?;
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(BackendError::ExecutionFailed);
        }
        let queue = device
            .newMTL4CommandQueue()
            .ok_or(BackendError::ExecutionFailed)?;
        let source = concat!(
            include_str!("elementwise.metal"),
            "\n",
            include_str!("kernels.metal")
        );
        let pipelines = PipelineCache::new(&device, source)?;
        let event_listener = MTLSharedEventListener::new();
        Ok(Self {
            device,
            queue,
            buffers: Mutex::new(AllocationRegistry::new()),
            pipelines: Mutex::new(pipelines),
            in_flight: Arc::new(InFlightTracker::new()),
            event_listener,
            gpu_timeout,
        })
    }

    pub(super) fn validate(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(tensor)
            .map(|_| ())
    }
}

impl Backend for MetalBackend {
    type Submission = MetalSubmission;

    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        let byte_len = element_count(shape)?
            .checked_mul(dtype.byte_size())
            .ok_or(BackendError::AllocationFailed)?;
        let len = usize::try_from(byte_len).map_err(|_| BackendError::AllocationFailed)?;
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(BackendError::AllocationFailed)?;
        let buffer = MetalBuffer {
            raw,
            len,
            pending: Vec::new(),
        };
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let id = buffers.insert(buffer, byte_len)?;
        let layout = Layout::contiguous(dtype, 0, shape.to_vec(), byte_len)
            .map_err(|_| BackendError::InvalidInput)?;
        Tensor::new(id, layout).map_err(|_| BackendError::InvalidInput)
    }

    fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        self.validate(tensor)?;
        let layout = match op {
            ViewOp::Slice(spec) => tensor.layout().slice(&spec),
            ViewOp::Reshape(shape) => tensor.layout().reshape(&shape),
            ViewOp::Permute(axes) => tensor.layout().permute(&axes),
            ViewOp::Broadcast(shape) => tensor.layout().broadcast(&shape),
        }
        .map_err(|_| BackendError::InvalidInput)?;
        Tensor::new(tensor.buffer(), layout).map_err(|_| BackendError::InvalidInput)
    }

    fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        if !tensor.layout().is_contiguous() || bytes.len() != logical_byte_len(tensor.layout())? {
            return Err(BackendError::InvalidInput);
        }
        let range = tensor.layout().byte_span();
        let range = usize::try_from(range.start).map_err(|_| BackendError::InvalidInput)?
            ..usize::try_from(range.end).map_err(|_| BackendError::InvalidInput)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_mut(tensor)?;
        buffer.wait_pending(self.gpu_timeout)?;
        buffer.write(range, bytes);
        Ok(())
    }

    fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_mut(tensor)?;
        buffer.wait_pending(self.gpu_timeout)?;
        gather(buffer.bytes(), tensor.layout())
    }

    fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        buffers.get_mut(tensor)?.wait_pending(self.gpu_timeout)?;
        buffers.remove(tensor)?;
        Ok(())
    }

    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        self.submit_commands(commands)
    }
}

fn element_count(shape: &[u32]) -> Result<u64, BackendError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1_u64, |count, &extent| {
        count
            .checked_mul(u64::from(extent))
            .ok_or(BackendError::AllocationFailed)
    })
}

fn logical_byte_len(layout: &Layout) -> Result<usize, BackendError> {
    let bytes = layout
        .element_count()
        .checked_mul(layout.dtype().byte_size())
        .ok_or(BackendError::InvalidInput)?;
    usize::try_from(bytes).map_err(|_| BackendError::InvalidInput)
}

fn element_offsets(layout: &Layout) -> impl Iterator<Item = u64> + '_ {
    (0..layout.element_count()).map(|mut linear| {
        let mut offset = layout.offset();
        for (&extent, &stride) in layout.shape().iter().zip(layout.strides()).rev() {
            offset += (linear % u64::from(extent)) * stride;
            linear /= u64::from(extent);
        }
        offset
    })
}

fn gather(source: &[u8], layout: &Layout) -> Result<Vec<u8>, BackendError> {
    let width =
        usize::try_from(layout.dtype().byte_size()).map_err(|_| BackendError::InvalidInput)?;
    let mut result = Vec::with_capacity(logical_byte_len(layout)?);
    for offset in element_offsets(layout) {
        let start = usize::try_from(
            offset
                .checked_mul(layout.dtype().byte_size())
                .ok_or(BackendError::InvalidInput)?,
        )
        .map_err(|_| BackendError::InvalidInput)?;
        result.extend_from_slice(
            source
                .get(start..start + width)
                .ok_or(BackendError::InvalidInput)?,
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_storage_round_trips_bytes() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let bytes = (0_u8..=u8::MAX)
            .cycle()
            .take(7 * 33 * 4)
            .collect::<Vec<_>>();
        backend.write(&tensor, &bytes).unwrap();
        assert_eq!(backend.read(&tensor).unwrap(), bytes);
    }

    #[test]
    fn metal_storage_reads_a_permuted_view() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[2, 3]).unwrap();
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&tensor, &values).unwrap();
        let view = backend.view(&tensor, ViewOp::Permute(vec![1, 0])).unwrap();
        let expected = [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(backend.read(&view).unwrap(), expected);
    }

    #[test]
    fn metal_storage_rejects_use_after_release() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[7]).unwrap();
        let view = backend.view(&tensor, ViewOp::Reshape(vec![1, 7])).unwrap();
        backend.release(&tensor).unwrap();
        assert_eq!(backend.read(&view), Err(BackendError::InvalidInput));
    }
}
