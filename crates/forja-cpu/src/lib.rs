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

    fn execute_rms_norm(
        &self,
        inputs: &[Tensor],
        output: &Tensor,
        eps: f32,
    ) -> Result<(), BackendError> {
        let values = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let weights = decode(&self.read(&inputs[1])?, inputs[1].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let mut normalized = Vec::with_capacity(values.len());
        for row in values.chunks_exact(weights.len()) {
            let sum = row.iter().fold(0.0, |sum, value| sum + value * value);
            let count = row.iter().fold(0.0, |count, _| count + 1.0);
            let scale = (sum / count + eps).sqrt().recip();
            normalized.extend(
                row.iter()
                    .zip(&weights)
                    .map(|(value, weight)| value * scale * weight),
            );
        }
        self.write_output(output, &normalized)
    }

    fn execute_softmax(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
        let values = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let width = execution_usize(
            inputs[0]
                .layout()
                .shape()
                .last()
                .copied()
                .ok_or(BackendError::ExecutionFailed)?,
        )?;
        let mut probabilities = Vec::with_capacity(values.len());
        for row in values.chunks_exact(width) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if maximum == f32::NEG_INFINITY {
                probabilities.resize(probabilities.len() + width, 0.0);
                continue;
            }
            let start = probabilities.len();
            probabilities.extend(row.iter().map(|value| (value - maximum).exp()));
            let sum = probabilities[start..].iter().sum::<f32>();
            probabilities[start..]
                .iter_mut()
                .for_each(|value| *value /= sum);
        }
        self.write_output(output, &probabilities)
    }

    #[allow(clippy::cast_precision_loss)]
    fn execute_rope(
        &self,
        inputs: &[Tensor],
        output: &Tensor,
        theta: f32,
    ) -> Result<(), BackendError> {
        let mut values = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let positions = decode_u32(&self.read(&inputs[1])?).ok_or(BackendError::ExecutionFailed)?;
        let shape = inputs[0].layout().shape();
        let width = execution_usize(shape[2])?;
        let row_width = checked_product(execution_usize(shape[1])?, width)?;
        let half = width / 2;
        let inv_freq = (0..half)
            .map(|index| theta.powf(2.0 * index as f32 / width as f32).recip())
            .collect::<Vec<_>>();
        for (row, position) in values.chunks_exact_mut(row_width).zip(positions) {
            for head in row.chunks_exact_mut(width) {
                for (index, &frequency) in inv_freq.iter().enumerate() {
                    let angle = position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let first = head[index];
                    let second = head[index + half];
                    head[index] = first * cos - second * sin;
                    head[index + half] = second * cos + first * sin;
                }
            }
        }
        self.write_output(output, &values)
    }

    fn execute_embed(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
        let table = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let ids = decode_u32(&self.read(&inputs[1])?).ok_or(BackendError::ExecutionFailed)?;
        let shape = inputs[0].layout().shape();
        let vocab = execution_usize(shape[0])?;
        let width = execution_usize(shape[1])?;
        let capacity = execution_usize(output.layout().element_count())?;
        let mut values = Vec::with_capacity(capacity);
        let mut first_bad = None;
        for id in ids {
            if let Ok(row) = execution_usize(id)
                && row < vocab
            {
                let start = row
                    .checked_mul(width)
                    .ok_or(BackendError::ExecutionFailed)?;
                let end = start
                    .checked_add(width)
                    .ok_or(BackendError::ExecutionFailed)?;
                values
                    .extend_from_slice(table.get(start..end).ok_or(BackendError::ExecutionFailed)?);
            } else {
                let end = values
                    .len()
                    .checked_add(width)
                    .ok_or(BackendError::ExecutionFailed)?;
                values.resize(end, 0.0);
                first_bad.get_or_insert(id);
            }
        }
        self.write_output(output, &values)?;
        first_bad.map_or(Ok(()), |index| Err(BackendError::IndexOutOfRange { index }))
    }

    fn execute_matmul(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
        let left = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let right = decode(&self.read(&inputs[1])?, inputs[1].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let left_shape = inputs[0].layout().shape();
        let right_shape = inputs[1].layout().shape();
        let rank = left_shape.len();
        let rows = execution_usize(left_shape[rank - 2])?;
        let inner = execution_usize(left_shape[rank - 1])?;
        let columns = execution_usize(right_shape[rank - 1])?;
        let left_batch = checked_product(rows, inner)?;
        let right_batch = checked_product(inner, columns)?;
        let capacity = execution_usize(output.layout().element_count())?;
        let mut values = Vec::with_capacity(capacity);
        for (left_batch, right_batch) in left
            .chunks_exact(left_batch)
            .zip(right.chunks_exact(right_batch))
        {
            for row in left_batch.chunks_exact(inner) {
                for column in 0..columns {
                    values.push(
                        row.iter()
                            .zip(right_batch.chunks_exact(columns))
                            .fold(0.0, |sum, (left, b_row)| sum + left * b_row[column]),
                    );
                }
            }
        }
        self.write_output(output, &values)
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
                Op::SiluMul => {
                    self.execute_binary(dispatch.inputs(), dispatch.output(), |gate, up| {
                        gate / (1.0 + (-gate).exp()) * up
                    })
                }
                Op::RmsNorm { eps } => {
                    self.execute_rms_norm(dispatch.inputs(), dispatch.output(), eps)
                }
                Op::Softmax => self.execute_softmax(dispatch.inputs(), dispatch.output()),
                Op::Rope { theta } => {
                    self.execute_rope(dispatch.inputs(), dispatch.output(), theta)
                }
                Op::Embed => self.execute_embed(dispatch.inputs(), dispatch.output()),
                Op::Matmul => self.execute_matmul(dispatch.inputs(), dispatch.output()),
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

fn checked_product(left: usize, right: usize) -> Result<usize, BackendError> {
    left.checked_mul(right).ok_or(BackendError::ExecutionFailed)
}

fn execution_usize<T>(value: T) -> Result<usize, BackendError>
where
    usize: TryFrom<T>,
{
    usize::try_from(value).map_err(|_| BackendError::ExecutionFailed)
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

fn decode_u32(source: &[u8]) -> Option<Vec<u32>> {
    let (words, remainder) = source.as_chunks::<4>();
    remainder.is_empty().then(|| {
        words
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect()
    })
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

    #[test]
    fn computes_silu_mul_into_bf16() {
        let backend = CpuBackend::new();
        let gate = backend.alloc(DType::F32, &[5]).unwrap();
        backend
            .write(&gate, &f32_bytes(&[-10.0, -1.0, 0.0, 1.0, 10.0]))
            .unwrap();
        let up = backend.alloc(DType::F32, &[5]).unwrap();
        backend.write(&up, &f32_bytes(&[1.0; 5])).unwrap();
        let output = backend.alloc(DType::BF16, &[5]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::SiluMul, &[&gate, &up], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let expected = [-0.000_453_978_7, -0.268_941_43, 0.0, 0.731_058_6, 9.999_546]
            .into_iter()
            .flat_map(|value| bf16::from_f32(value).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(backend.read(&output).unwrap(), expected);
    }

    #[test]
    fn normalizes_qwen_hidden_rows_with_epsilon() {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[3, 1024]).unwrap();
        let mut values = vec![2.0; 1024];
        values.extend(vec![-4.0; 1024]);
        values.extend((0..512).flat_map(|_| [3.0, 4.0]));
        backend.write(&input, &f32_bytes(&values)).unwrap();
        let weight = backend.alloc(DType::F32, &[1024]).unwrap();
        backend
            .write(&weight, &f32_bytes(&vec![0.5; 1024]))
            .unwrap();
        let output = backend.alloc(DType::F32, &[3, 1024]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::RmsNorm { eps: 0.25 }, &[&input, &weight], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let mut expected = vec![1.0 / 4.25_f32.sqrt(); 1024];
        expected.extend(vec![-2.0 / 16.25_f32.sqrt(); 1024]);
        expected.extend((0..512).flat_map(|_| [1.5 / 12.75_f32.sqrt(), 2.0 / 12.75_f32.sqrt()]));
        let actual = decode(&backend.read(&output).unwrap(), DType::F32).unwrap();
        assert_relative(&actual, &expected, 1e-5);
    }

    #[test]
    fn computes_masked_softmax_and_zeros_an_all_masked_row() {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[2, 4]).unwrap();
        backend
            .write(
                &input,
                &f32_bytes(&[
                    1.0,
                    f32::NEG_INFINITY,
                    3.0,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ]),
            )
            .unwrap();
        let output = backend.alloc(DType::F32, &[2, 4]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Softmax, &[&input], &output).unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let low = (-2.0_f32).exp() / (1.0 + (-2.0_f32).exp());
        let high = 1.0 / (1.0 + (-2.0_f32).exp());
        let actual = decode(&backend.read(&output).unwrap(), DType::F32).unwrap();
        assert_relative(&actual, &[low, 0.0, high, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-5);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let actual = run_rope(&[1.0, -2.0, 3.0, -4.0], &[0], &[1, 1, 4], 10_000.0);
        assert_eq!(actual, [1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn rope_rotates_split_halves() {
        let actual = run_rope(&[1.0, 2.0, 3.0, 4.0], &[1], &[1, 1, 4], 100.0);
        let expected = [
            1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
            2.0 * 0.1_f32.cos() - 4.0 * 0.1_f32.sin(),
            3.0 * 1.0_f32.cos() + 1.0_f32.sin(),
            4.0 * 0.1_f32.cos() + 2.0 * 0.1_f32.sin(),
        ];
        assert_relative(&actual, &expected, 1e-5);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn rope_matches_independent_qwen_shape_computation() {
        let values = (0_u16..14_336)
            .map(|index| f32::from(index % 33) / 16.0 - 1.0)
            .collect::<Vec<_>>();
        let positions = (0_u32..7).collect::<Vec<_>>();
        let actual = run_rope(&values, &positions, &[7, 16, 128], 1e6);
        let mut expected = values.clone();
        for (row, &position) in expected
            .as_chunks_mut::<{ 16 * 128 }>()
            .0
            .iter_mut()
            .zip(&positions)
        {
            for head in row.as_chunks_mut::<128>().0 {
                for index in 0..64 {
                    let angle = f64::from(position) / 1e6_f64.powf(2.0 * index as f64 / 128.0);
                    let first = f64::from(head[index]);
                    let second = f64::from(head[index + 64]);
                    head[index] = (first * angle.cos() - second * angle.sin()) as f32;
                    head[index + 64] = (second * angle.cos() + first * angle.sin()) as f32;
                }
            }
        }
        assert_relative(&actual, &expected, 1e-5);
    }

    #[test]
    fn embeds_rows_near_both_ends_of_a_qwen_width_table() {
        let backend = CpuBackend::new();
        let table = backend.alloc(DType::F32, &[1000, 1024]).unwrap();
        let values = (0_u16..251)
            .cycle()
            .take(1000 * 1024)
            .map(f32::from)
            .collect::<Vec<_>>();
        backend.write(&table, &f32_bytes(&values)).unwrap();
        let ids = backend.alloc(DType::U32, &[4]).unwrap();
        backend.write(&ids, &u32_bytes(&[0, 1, 998, 999])).unwrap();
        let output = backend.alloc(DType::F32, &[4, 1024]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Embed, &[&table, &ids], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let actual = decode(&backend.read(&output).unwrap(), DType::F32).unwrap();
        let expected = [0, 1, 998, 999]
            .into_iter()
            .flat_map(|row| values[row * 1024..(row + 1) * 1024].iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn embed_zeros_bad_rows_and_reports_the_first_bad_id() {
        let backend = CpuBackend::new();
        let table = backend.alloc(DType::F32, &[3, 2]).unwrap();
        backend
            .write(&table, &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        let ids = backend.alloc(DType::U32, &[4]).unwrap();
        backend.write(&ids, &u32_bytes(&[2, 99, 0, 100])).unwrap();
        let output = backend.alloc(DType::F32, &[4, 2]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Embed, &[&table, &ids], &output)
            .unwrap();
        let error = backend.submit(commands).unwrap().wait();
        assert_eq!(error, Err(BackendError::IndexOutOfRange { index: 99 }));
        let actual = decode(&backend.read(&output).unwrap(), DType::F32).unwrap();
        assert_eq!(actual, [5.0, 6.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn multiplies_by_a_permuted_qwen_weight() {
        let backend = CpuBackend::new();
        let x = (0_u16..33)
            .cycle()
            .take(5 * 1024)
            .map(|value| (f32::from(value) - 16.0) / 16.0)
            .collect::<Vec<_>>();
        let weights = (0_u16..251)
            .cycle()
            .take(3072 * 1024)
            .map(|value| (f32::from(value) - 125.0) / 125.0)
            .collect::<Vec<_>>();
        let input = backend.alloc(DType::F32, &[5, 1024]).unwrap();
        backend.write(&input, &f32_bytes(&x)).unwrap();
        let stored = backend.alloc(DType::F32, &[3072, 1024]).unwrap();
        backend.write(&stored, &f32_bytes(&weights)).unwrap();
        let weight = backend.view(&stored, ViewOp::Permute(vec![1, 0])).unwrap();
        let output = backend.alloc(DType::F32, &[5, 3072]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&input, &weight], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        let actual = decode(&backend.read(&output).unwrap(), DType::F32).unwrap();
        let mut expected = Vec::with_capacity(5 * 3072);
        for row in 0..5 {
            for column in 0..3072 {
                let value = (0..1024)
                    .map(|inner| {
                        f64::from(x[row * 1024 + inner]) * f64::from(weights[column * 1024 + inner])
                    })
                    .sum::<f64>();
                expected.push(value as f32);
            }
        }
        assert_relative(&actual, &expected, 1e-5);
    }

    #[test]
    fn matmul_handles_unit_contraction_and_row_dimensions() {
        let backend = CpuBackend::new();
        let actual = run_matmul(&backend, &[2.0, 3.0], &[4.0, 5.0, 6.0], &[2, 1], &[1, 3]);
        assert_eq!(actual, [8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
        let actual = run_matmul(
            &backend,
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[2, 1, 2],
            &[2, 2, 1],
        );
        assert_eq!(actual, [17.0, 53.0]);
    }

    fn run_matmul(
        backend: &CpuBackend,
        a: &[f32],
        b: &[f32],
        a_shape: &[u32],
        b_shape: &[u32],
    ) -> Vec<f32> {
        let left = backend.alloc(DType::F32, a_shape).unwrap();
        backend.write(&left, &f32_bytes(a)).unwrap();
        let right = backend.alloc(DType::F32, b_shape).unwrap();
        backend.write(&right, &f32_bytes(b)).unwrap();
        let rank = a_shape.len();
        let mut shape = a_shape[..rank - 2].to_vec();
        shape.extend([a_shape[rank - 2], b_shape[rank - 1]]);
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&left, &right], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        decode(&backend.read(&output).unwrap(), DType::F32).unwrap()
    }

    fn run_rope(values: &[f32], positions: &[u32], shape: &[u32], theta: f32) -> Vec<f32> {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, shape).unwrap();
        backend.write(&input, &f32_bytes(values)).unwrap();
        let position_tensor = backend.alloc(DType::U32, &[shape[0]]).unwrap();
        backend
            .write(&position_tensor, &u32_bytes(positions))
            .unwrap();
        let output = backend.alloc(DType::F32, shape).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Rope { theta }, &[&input, &position_tensor], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        decode(&backend.read(&output).unwrap(), DType::F32).unwrap()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn assert_relative(actual: &[f32], expected: &[f32], tolerance: f32) {
        let (error, reference) = actual.iter().zip(expected).fold(
            (0.0, 0.0),
            |(error, reference), (actual, expected)| {
                (
                    error + (actual - expected).powi(2),
                    reference + expected.powi(2),
                )
            },
        );
        assert!(error.sqrt() / reference.sqrt() <= tolerance);
    }
}
