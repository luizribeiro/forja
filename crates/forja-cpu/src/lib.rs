//! Reference CPU execution for Forja operations.

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use forja_core::{BackendError, BufferId, DType, Layout, Tensor, ViewOp};

static NEXT_BACKEND: AtomicU64 = AtomicU64::new(1);

/// A straightforward, single-process reference backend.
#[derive(Debug)]
pub struct CpuBackend {
    id: u64,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    next_allocation: u64,
    buffers: HashMap<BufferId, Vec<u8>>,
}

impl State {
    fn validate(&self, tensor: &Tensor) -> Result<(), BackendError> {
        let bytes = self
            .buffers
            .get(&tensor.buffer())
            .ok_or(BackendError::InvalidInput)?;
        if u64::try_from(bytes.len()).ok() != Some(tensor.layout().buffer_len()) {
            return Err(BackendError::InvalidInput);
        }
        Ok(())
    }
}

impl CpuBackend {
    /// Creates an independent backend instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: NEXT_BACKEND.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(State::default()),
        }
    }

    /// Allocates a contiguous tensor initialized to zero.
    ///
    /// # Errors
    ///
    /// Returns an allocation error if its byte size overflows or memory is unavailable.
    pub fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        let elements = element_count(shape)?;
        let byte_len = elements
            .checked_mul(dtype.byte_size())
            .ok_or(BackendError::AllocationFailed)?;
        let len = usize::try_from(byte_len).map_err(|_| BackendError::AllocationFailed)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| BackendError::AllocationFailed)?;
        bytes.resize(len, 0);
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let allocation = state.next_allocation;
        state.next_allocation = allocation
            .checked_add(1)
            .ok_or(BackendError::AllocationFailed)?;
        let buffer = BufferId::new(self.id, allocation, byte_len);
        state.buffers.insert(buffer, bytes);
        let layout = Layout::contiguous(dtype, 0, shape.to_vec(), byte_len)
            .map_err(|_| BackendError::InvalidInput)?;
        Tensor::new(buffer, layout).map_err(|_| BackendError::InvalidInput)
    }

    /// Applies a validated metadata-only view operation.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation or invalid view.
    pub fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        self.state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .validate(tensor)?;
        let layout = match op {
            ViewOp::Slice(spec) => tensor.layout().slice(&spec),
            ViewOp::Reshape(shape) => tensor.layout().reshape(&shape),
            ViewOp::Permute(axes) => tensor.layout().permute(&axes),
            ViewOp::Broadcast(shape) => tensor.layout().broadcast(&shape),
        }
        .map_err(|_| BackendError::InvalidInput)?;
        Tensor::new(tensor.buffer(), layout).map_err(|_| BackendError::InvalidInput)
    }

    /// Writes contiguous logical tensor bytes.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation, non-contiguous view, or wrong byte count.
    pub fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        if !tensor.layout().is_contiguous() || bytes.len() != logical_byte_len(tensor.layout())? {
            return Err(BackendError::InvalidInput);
        }
        let range = tensor.layout().byte_span();
        let range = usize::try_from(range.start).map_err(|_| BackendError::InvalidInput)?
            ..usize::try_from(range.end).map_err(|_| BackendError::InvalidInput)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        state.validate(tensor)?;
        state
            .buffers
            .get_mut(&tensor.buffer())
            .and_then(|target| target.get_mut(range))
            .ok_or(BackendError::InvalidInput)?
            .copy_from_slice(bytes);
        Ok(())
    }

    /// Gathers a tensor view into contiguous logical bytes.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation.
    pub fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        let state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        state.validate(tensor)?;
        let source = state
            .buffers
            .get(&tensor.buffer())
            .ok_or(BackendError::InvalidInput)?;
        gather(source, tensor.layout())
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
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
    fn rejects_unknown_foreign_and_wrong_length_tensors() {
        let backend = CpuBackend::new();
        let real = backend.alloc(DType::F32, &[1]).unwrap();
        let backend_id = real.buffer().backend();
        let wrong_length = Tensor::new(
            BufferId::new(backend_id, 0, 8),
            Layout::contiguous(DType::F32, 0, vec![2], 8).unwrap(),
        )
        .unwrap();
        let unknown = Tensor::new(
            BufferId::new(backend_id, 99, 4),
            Layout::contiguous(DType::F32, 0, vec![1], 4).unwrap(),
        )
        .unwrap();
        let foreign = CpuBackend::new().alloc(DType::F32, &[1]).unwrap();
        assert_eq!(backend.read(&wrong_length), Err(BackendError::InvalidInput));
        assert_eq!(
            backend.view(&unknown, ViewOp::Reshape(vec![1])),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(backend.read(&foreign), Err(BackendError::InvalidInput));
    }

    #[test]
    fn rejects_invalid_writes_and_allocation_overflow() {
        let backend = CpuBackend::new();
        let tensor = backend.alloc(DType::F32, &[2, 3]).unwrap();
        let permuted = backend.view(&tensor, ViewOp::Permute(vec![1, 0])).unwrap();
        assert_eq!(
            backend.write(&permuted, &[0; 24]),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            backend.write(&tensor, &[0; 4]),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            backend.alloc(DType::F32, &[u32::MAX, u32::MAX]),
            Err(BackendError::AllocationFailed)
        );
    }
}
