//! Reference CPU execution for Forja operations.

use std::{
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use forja_core::{
    Backend, BackendError, BufferId, CommandList, DType, Layout, Op, Submission, Tensor, ViewOp,
};
use half::{bf16, f16};

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

    fn validate(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .validate(tensor)
    }

    fn execute_copy(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
        let values = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        self.write_output(output, &values)
    }

    fn execute_binary(
        &self,
        inputs: &[Tensor],
        output: &Tensor,
        function: impl Fn(f32, f32) -> f32,
    ) -> Result<(), BackendError> {
        let left = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let right = decode(&self.read(&inputs[1])?, inputs[1].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let values = left
            .into_iter()
            .zip(right)
            .map(|(a, b)| function(a, b))
            .collect::<Vec<_>>();
        self.write_output(output, &values)
    }

    fn write_output(&self, output: &Tensor, values: &[f32]) -> Result<(), BackendError> {
        let converted =
            encode(values, output.layout().dtype()).ok_or(BackendError::ExecutionFailed)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let target = state
            .buffers
            .get_mut(&output.buffer())
            .ok_or(BackendError::InvalidInput)?;
        scatter(target, output.layout(), &converted)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// An already-complete CPU submission.
#[derive(Debug)]
pub struct CpuSubmission(Result<(), BackendError>);

impl Submission for CpuSubmission {
    fn wait(self) -> Result<(), BackendError> {
        self.0
    }
}

impl Backend for CpuBackend {
    type Submission = CpuSubmission;

    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        CpuBackend::alloc(self, dtype, shape)
    }

    fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        CpuBackend::view(self, tensor, op)
    }

    fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        CpuBackend::write(self, tensor, bytes)
    }

    fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        CpuBackend::read(self, tensor)
    }

    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        let dispatches = commands.into_dispatches();
        for dispatch in &dispatches {
            self.validate(dispatch.output())?;
            for input in dispatch.inputs() {
                self.validate(input)?;
            }
        }
        let result = dispatches
            .into_iter()
            .try_for_each(|dispatch| match dispatch.op() {
                Op::Copy => self.execute_copy(dispatch.inputs(), dispatch.output()),
                Op::Add => self.execute_binary(dispatch.inputs(), dispatch.output(), |a, b| a + b),
            });
        Ok(CpuSubmission(result))
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

fn scatter(target: &mut [u8], layout: &Layout, source: &[u8]) -> Result<(), BackendError> {
    let width =
        usize::try_from(layout.dtype().byte_size()).map_err(|_| BackendError::InvalidInput)?;
    for (index, offset) in element_offsets(layout).enumerate() {
        let start = usize::try_from(
            offset
                .checked_mul(layout.dtype().byte_size())
                .ok_or(BackendError::InvalidInput)?,
        )
        .map_err(|_| BackendError::InvalidInput)?;
        let source_start = index.checked_mul(width).ok_or(BackendError::InvalidInput)?;
        target
            .get_mut(start..start + width)
            .ok_or(BackendError::InvalidInput)?
            .copy_from_slice(
                source
                    .get(source_start..source_start + width)
                    .ok_or(BackendError::InvalidInput)?,
            );
    }
    Ok(())
}

fn decode(source: &[u8], input: DType) -> Option<Vec<f32>> {
    let input_width = usize::try_from(input.byte_size()).ok()?;
    source
        .chunks_exact(input_width)
        .map(|bytes| match input {
            DType::F32 => bytes.try_into().ok().map(f32::from_le_bytes),
            DType::F16 => bytes
                .try_into()
                .ok()
                .map(f16::from_le_bytes)
                .map(f16::to_f32),
            DType::BF16 => bytes
                .try_into()
                .ok()
                .map(bf16::from_le_bytes)
                .map(bf16::to_f32),
            DType::I32 | DType::U32 => None,
        })
        .collect()
}

fn encode(source: &[f32], output: DType) -> Option<Vec<u8>> {
    let mut result = Vec::with_capacity(source.len() * usize::try_from(output.byte_size()).ok()?);
    for &value in source {
        match output {
            DType::F32 => result.extend_from_slice(&value.to_le_bytes()),
            DType::F16 => result.extend_from_slice(&f16::from_f32(value).to_le_bytes()),
            DType::BF16 => result.extend_from_slice(&bf16::from_f32(value).to_le_bytes()),
            DType::I32 | DType::U32 => return None,
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::Op;

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

    #[test]
    fn copies_permuted_qwen_heads_to_contiguous_bf16() {
        let backend = CpuBackend::new();
        let source = backend.alloc(DType::F32, &[7, 16, 128]).unwrap();
        let values = (0_u16..14_336)
            .flat_map(|value| f32::from(value).to_le_bytes())
            .collect::<Vec<_>>();
        backend.write(&source, &values).unwrap();
        let permuted = backend
            .view(&source, ViewOp::Permute(vec![1, 0, 2]))
            .unwrap();
        let output = backend.alloc(DType::BF16, &[16, 7, 128]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&permuted], &output).unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let actual = backend.read(&output).unwrap();
        let expected = (0_u16..16)
            .flat_map(|head| {
                (0_u16..7).flat_map(move |sequence| {
                    (0_u16..128).flat_map(move |column| {
                        let index = (sequence * 16 + head) * 128 + column;
                        bf16::from_f32(f32::from(index)).to_le_bytes()
                    })
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_submission_with_a_foreign_tensor() {
        let backend = CpuBackend::new();
        let foreign = CpuBackend::new().alloc(DType::F32, &[1]).unwrap();
        let output = backend.alloc(DType::F32, &[1]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&foreign], &output).unwrap();
        assert!(matches!(
            backend.submit(commands),
            Err(BackendError::InvalidInput)
        ));
    }

    #[test]
    fn adds_a_broadcast_operand() {
        let backend = CpuBackend::new();
        let left = backend.alloc(DType::F32, &[2, 3]).unwrap();
        backend
            .write(&left, &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        let row = backend.alloc(DType::F32, &[1, 3]).unwrap();
        backend
            .write(&row, &f32_bytes(&[10.0, 20.0, 30.0]))
            .unwrap();
        let right = backend.view(&row, ViewOp::Broadcast(vec![2, 3])).unwrap();
        let output = backend.alloc(DType::F32, &[2, 3]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Add, &[&left, &right], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        assert_eq!(
            backend.read(&output).unwrap(),
            f32_bytes(&[11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
        );
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}
