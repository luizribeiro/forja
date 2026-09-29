//! Reference CPU execution for Forja operations.

pub mod interpreter;

use std::{sync::Mutex, time::Duration};

use forja_core::{
    AllocationRegistry, Backend, BackendError, BufferId, CommandList, DType, Layout, MappedRegion,
    Op, Submission, Tensor, ViewOp,
    program::{BoundProgram, KernelSignature, ValidatedProgram},
};
use half::{bf16, f16};

/// A straightforward, single-process reference backend.
#[derive(Debug)]
pub struct CpuBackend {
    buffers: Mutex<AllocationRegistry<CpuBuffer>>,
}

#[derive(Debug)]
enum CpuBuffer {
    Owned(Vec<u8>),
    Mapped(MappedRegion),
}

impl CpuBuffer {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped(region) => region.bytes(),
        }
    }

    fn bytes_mut(&mut self) -> Result<&mut [u8], BackendError> {
        match self {
            Self::Owned(bytes) => Ok(bytes),
            Self::Mapped(_) => Err(BackendError::InvalidInput),
        }
    }
}

impl CpuBackend {
    /// Creates an independent backend instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(AllocationRegistry::new()),
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
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let layout = Layout::contiguous(dtype, 0, shape.to_vec(), byte_len)
            .map_err(|_| BackendError::InvalidInput)?;
        let buffer = buffers.insert(CpuBuffer::Owned(bytes), byte_len)?;
        buffers.tensor(buffer, layout)
    }

    /// Applies a validated metadata-only view operation.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation or invalid view.
    pub fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let layout = match op {
            ViewOp::Slice(spec) => tensor.layout().slice(&spec),
            ViewOp::Reshape(shape) => tensor.layout().reshape(&shape),
            ViewOp::Permute(axes) => tensor.layout().permute(&axes),
            ViewOp::Broadcast(shape) => tensor.layout().broadcast(&shape),
        }
        .map_err(|_| BackendError::InvalidInput)?;
        buffers.view(tensor, layout)
    }

    /// Writes contiguous logical tensor bytes.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation, non-contiguous view, or wrong byte count.
    pub fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        if !tensor.is_writable()
            || !tensor.layout().is_contiguous()
            || bytes.len() != logical_byte_len(tensor.layout())?
        {
            return Err(BackendError::InvalidInput);
        }
        let range = tensor.layout().byte_span();
        let range = usize::try_from(range.start).map_err(|_| BackendError::InvalidInput)?
            ..usize::try_from(range.end).map_err(|_| BackendError::InvalidInput)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        buffers
            .get_mut(tensor)?
            .bytes_mut()?
            .get_mut(range)
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
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let source = buffers.get(tensor)?.bytes();
        gather(source, tensor.layout())
    }

    /// Releases a tensor's allocation and invalidates all of its views.
    ///
    /// # Errors
    ///
    /// Returns invalid input for an unknown allocation.
    pub fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .remove(tensor)?;
        Ok(())
    }

    fn validate(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(tensor)
            .map(|_| ())
    }

    fn execute_copy(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
        if matches!(inputs[0].layout().dtype(), DType::I32 | DType::U32) {
            let bytes = self.read(&inputs[0])?;
            let mut buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            return scatter(
                buffers.get_mut(output)?.bytes_mut()?,
                output.layout(),
                &bytes,
            );
        }
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

    fn execute_program(&self, program: &BoundProgram) -> Result<(), BackendError> {
        enum DecodedInput {
            F32(Vec<f32>),
            U32(Vec<u32>),
        }

        let decoded = program
            .inputs()
            .iter()
            .map(|input| match input.layout().dtype() {
                DType::F32 | DType::F16 | DType::BF16 => {
                    decode(&self.read(input)?, input.layout().dtype())
                        .map(DecodedInput::F32)
                        .ok_or(BackendError::ExecutionFailed)
                }
                DType::U32 => decode_u32(&self.read(input)?)
                    .map(DecodedInput::U32)
                    .ok_or(BackendError::ExecutionFailed),
                DType::I32 => Err(BackendError::ExecutionFailed),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = decoded
            .iter()
            .map(|input| match input {
                DecodedInput::F32(values) => interpreter::Input::F32(values),
                DecodedInput::U32(values) => interpreter::Input::U32(values),
            })
            .collect::<Vec<_>>();
        let shape = program
            .outputs()
            .first()
            .map(Tensor::layout)
            .map(Layout::shape)
            .ok_or(BackendError::ExecutionFailed)?;
        let values = interpreter::interpret(program.program(), shape, &inputs)?;
        program
            .outputs()
            .iter()
            .zip(values)
            .try_for_each(|(output, values)| self.write_output(output, &values))
    }

    fn write_output(&self, output: &Tensor, values: &[f32]) -> Result<(), BackendError> {
        let converted =
            encode(values, output.layout().dtype()).ok_or(BackendError::ExecutionFailed)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let target = buffers.get_mut(output)?.bytes_mut()?;
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
            let start = probabilities.len();
            probabilities.extend_from_slice(row);
            softmax(&mut probabilities[start..]);
        }
        self.write_output(output, &probabilities)
    }

    fn execute_argmax(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), BackendError> {
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
        let indices = values
            .chunks_exact(width)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .and_then(|(index, _)| u32::try_from(index).ok())
                    .ok_or(BackendError::ExecutionFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = indices
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let target = buffers.get_mut(output)?.bytes_mut()?;
        scatter(target, output.layout(), &bytes)
    }

    fn execute_sample(
        &self,
        inputs: &[Tensor],
        output: &Tensor,
        position: u32,
    ) -> Result<(), BackendError> {
        let values = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let words = decode_u32(&self.read(&inputs[1])?).ok_or(BackendError::ExecutionFailed)?;
        let parameters = SamplingParameters::from_words(&words)?;
        let width = execution_usize(
            inputs[0]
                .layout()
                .shape()
                .last()
                .copied()
                .ok_or(BackendError::ExecutionFailed)?,
        )?;
        let indices = values
            .chunks_exact(width)
            .map(|row| sample_row(row, parameters, position))
            .collect::<Result<Vec<_>, _>>()?;
        let bytes = indices
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let target = buffers.get_mut(output)?.bytes_mut()?;
        scatter(target, output.layout(), &bytes)
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

    fn execute_sdpa(
        &self,
        inputs: &[Tensor],
        output: &Tensor,
        scale: f32,
        causal: bool,
        q_start: u32,
    ) -> Result<(), BackendError> {
        let query = decode(&self.read(&inputs[0])?, inputs[0].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let key = decode(&self.read(&inputs[1])?, inputs[1].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let value = decode(&self.read(&inputs[2])?, inputs[2].layout().dtype())
            .ok_or(BackendError::ExecutionFailed)?;
        let q_shape = inputs[0].layout().shape();
        let k_shape = inputs[1].layout().shape();
        let v_shape = inputs[2].layout().shape();
        let q_heads = execution_usize(q_shape[0])?;
        let q_len = execution_usize(q_shape[1])?;
        let width = execution_usize(q_shape[2])?;
        let kv_heads = execution_usize(k_shape[0])?;
        let kv_len = execution_usize(k_shape[1])?;
        let value_width = execution_usize(v_shape[2])?;
        let q_head_len = checked_product(q_len, width)?;
        let k_head_len = checked_product(kv_len, width)?;
        let v_head_len = checked_product(kv_len, value_width)?;
        let group = q_heads / kv_heads;
        let q_start = execution_usize(q_start)?;
        let mut result = Vec::new();
        for (head_index, q_head) in query.chunks_exact(q_head_len).enumerate() {
            let kv_head = head_index / group;
            let key_rows = key
                .chunks_exact(k_head_len)
                .nth(kv_head)
                .ok_or(BackendError::ExecutionFailed)?;
            let value_rows = value
                .chunks_exact(v_head_len)
                .nth(kv_head)
                .ok_or(BackendError::ExecutionFailed)?;
            for (q_index, q_row) in q_head.chunks_exact(width).enumerate() {
                let attended = if causal {
                    q_start
                        .checked_add(q_index)
                        .and_then(|position| position.checked_add(1))
                        .ok_or(BackendError::ExecutionFailed)?
                } else {
                    kv_len
                };
                let mut scores = key_rows
                    .chunks_exact(width)
                    .take(attended)
                    .map(|k_row| {
                        q_row.iter().zip(k_row).fold(0.0, |sum, (q, k)| sum + q * k) * scale
                    })
                    .collect::<Vec<_>>();
                softmax(&mut scores);
                for column in 0..value_width {
                    result.push(
                        scores
                            .iter()
                            .zip(value_rows.chunks_exact(value_width))
                            .fold(0.0, |sum, (weight, row)| sum + weight * row[column]),
                    );
                }
            }
        }
        self.write_output(output, &result)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// An already-complete CPU submission.
#[derive(Debug)]
pub struct CpuSubmission {
    result: Result<(), BackendError>,
    wall_time: Duration,
}

impl Submission for CpuSubmission {
    fn wait(&self) -> Result<(), BackendError> {
        self.result
    }

    fn gpu_time(&self) -> Option<Duration> {
        Some(self.wall_time)
    }
}

impl Backend for CpuBackend {
    type Submission = CpuSubmission;
    type ProgramHandle = ();

    fn prepare_program(
        &self,
        _program: &ValidatedProgram,
        _signature: &KernelSignature,
    ) -> Result<Self::ProgramHandle, BackendError> {
        Ok(())
    }

    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        CpuBackend::alloc(self, dtype, shape)
    }

    fn import_readonly(&self, bytes: MappedRegion) -> Result<BufferId, BackendError> {
        let byte_len = u64::try_from(bytes.len()).map_err(|_| BackendError::AllocationFailed)?;
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .insert_read_only(CpuBuffer::Mapped(bytes), byte_len)
    }

    fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .tensor(buffer, layout)
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

    fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
        CpuBackend::release(self, tensor)
    }

    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        let started = std::time::Instant::now();
        let retained_tensors_validated = commands.retained_tensors_validated();
        let dispatches = commands.into_dispatches();
        if !retained_tensors_validated {
            for dispatch in &dispatches {
                for output in dispatch.outputs() {
                    self.validate(output)?;
                }
                for input in dispatch.inputs() {
                    self.validate(input)?;
                }
            }
        }
        let result = dispatches
            .into_iter()
            .try_for_each(|dispatch| match dispatch.op() {
                Op::Program(_) => dispatch
                    .bound_program()
                    .ok_or(BackendError::ExecutionFailed)
                    .and_then(|program| self.execute_program(program)),
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
                Op::Argmax => self.execute_argmax(dispatch.inputs(), dispatch.output()),
                Op::Sample { position } => {
                    self.execute_sample(dispatch.inputs(), dispatch.output(), position)
                }
                Op::Rope { theta } => {
                    self.execute_rope(dispatch.inputs(), dispatch.output(), theta)
                }
                Op::Embed => self.execute_embed(dispatch.inputs(), dispatch.output()),
                Op::Matmul => self.execute_matmul(dispatch.inputs(), dispatch.output()),
                Op::Sdpa {
                    scale,
                    causal,
                    q_start,
                } => {
                    self.execute_sdpa(dispatch.inputs(), dispatch.output(), scale, causal, q_start)
                }
            });
        Ok(CpuSubmission {
            result,
            wall_time: started.elapsed(),
        })
    }
}

#[derive(Clone, Copy)]
struct SamplingParameters {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: u64,
}

impl SamplingParameters {
    fn from_words(words: &[u32]) -> Result<Self, BackendError> {
        let [temperature, top_k, top_p, seed_low, seed_high] = words else {
            return Err(BackendError::InvalidInput);
        };
        let temperature = f32::from_bits(*temperature);
        let top_p = f32::from_bits(*top_p);
        if !temperature.is_finite()
            || temperature < 0.0
            || !top_p.is_finite()
            || !(0.0..=1.0).contains(&top_p)
            || top_p == 0.0
        {
            return Err(BackendError::InvalidInput);
        }
        Ok(Self {
            temperature,
            top_k: *top_k,
            top_p,
            seed: u64::from(*seed_low) | (u64::from(*seed_high) << 32),
        })
    }
}

fn sample_row(
    row: &[f32],
    parameters: SamplingParameters,
    position: u32,
) -> Result<u32, BackendError> {
    if parameters.temperature == 0.0 || parameters.top_k == 1 {
        return row
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .and_then(|(index, _)| u32::try_from(index).ok())
            .ok_or(BackendError::ExecutionFailed);
    }
    let mut allowed = (0..row.len()).collect::<Vec<_>>();
    allowed.sort_unstable_by(|&left, &right| {
        row[right]
            .total_cmp(&row[left])
            .then_with(|| right.cmp(&left))
    });
    let requested = usize::try_from(parameters.top_k).map_err(|_| BackendError::InvalidInput)?;
    let cap = if parameters.top_p < 1.0 {
        if requested == 0 {
            1024
        } else {
            requested.min(1024)
        }
    } else if requested == 0 {
        allowed.len()
    } else {
        requested
    };
    allowed.truncate(cap.min(allowed.len()));
    if parameters.top_p < 1.0 {
        truncate_top_p(row, &mut allowed, parameters)?;
    }
    allowed
        .into_iter()
        .max_by(|&left, &right| {
            sampled_score(row[left], parameters, position, left)
                .total_cmp(&sampled_score(row[right], parameters, position, right))
                .then_with(|| left.cmp(&right))
        })
        .and_then(|index| u32::try_from(index).ok())
        .ok_or(BackendError::ExecutionFailed)
}

fn truncate_top_p(
    row: &[f32],
    allowed: &mut Vec<usize>,
    parameters: SamplingParameters,
) -> Result<(), BackendError> {
    let maximum = *allowed.first().ok_or(BackendError::ExecutionFailed)?;
    let weights = allowed
        .iter()
        .map(|&index| ((row[index] - row[maximum]) / parameters.temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total.is_finite() && total > 0.0 {
        let mut cumulative = 0.0;
        let keep = weights
            .iter()
            .position(|weight| {
                cumulative += weight / total;
                cumulative >= parameters.top_p
            })
            .map_or(weights.len(), |index| index + 1);
        allowed.truncate(keep.max(1));
    } else {
        allowed.truncate(1);
    }
    Ok(())
}

fn sampled_score(logit: f32, parameters: SamplingParameters, position: u32, index: usize) -> f32 {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let counter = parameters.seed ^ (u64::from(position) << 32) ^ index;
    let random = splitmix64(counter);
    let mantissa = u32::try_from(random >> 40).unwrap_or(u32::MAX);
    #[allow(clippy::cast_precision_loss)]
    let uniform = (mantissa as f32 + 0.5) * (1.0 / 16_777_216.0);
    logit / parameters.temperature - (-uniform.ln()).ln()
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if maximum == f32::NEG_INFINITY {
        values.fill(0.0);
        return;
    }
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
    }
    let sum = values.iter().sum::<f32>();
    for value in values {
        *value /= sum;
    }
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
    use forja_core::{
        Affine, GraphLimits, GraphTemplate, Op, ParamSpace, Slice, SymbolicLayout, TemplateTensor,
        program::{BinOp, Inst, KernelSignature, Program, ProgramKind, ValueType, prepare_program},
    };
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn replays_graphs_through_the_prevalidated_path() {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[4]).unwrap();
        let output_base = backend.alloc(DType::F32, &[4]).unwrap();
        backend
            .write(&input, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let space = ParamSpace::new(std::iter::once(1..=4).collect()).unwrap();
        let extent = Affine::parameter(0, 0, 1);
        let input = TemplateTensor::symbolic(
            input.clone(),
            SymbolicLayout::new(input.layout().clone(), space.clone())
                .slice(0, 0.into(), extent, 1)
                .unwrap(),
        )
        .unwrap();
        let output = TemplateTensor::symbolic(
            output_base.clone(),
            SymbolicLayout::new(output_base.layout().clone(), space.clone())
                .slice(0, 0.into(), extent, 1)
                .unwrap(),
        )
        .unwrap();
        let mut graph = GraphTemplate::new(space, GraphLimits::default());
        graph.dispatch(Op::Copy, &[&input], &output).unwrap();
        let graph = backend.prepare_graph(graph).unwrap();

        backend.replay(&graph, vec![3]).unwrap().wait().unwrap();

        assert_eq!(
            backend.read(&output_base).unwrap(),
            f32_bytes(&[1.0, 2.0, 3.0, 0.0])
        );
    }

    #[test]
    fn appending_to_an_instantiated_graph_restores_tensor_validation() {
        let backend = CpuBackend::new();
        let graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        let values = graph.values(Vec::new()).unwrap();
        let mut commands = graph.instantiate(&values).unwrap();
        let foreign = CpuBackend::new();
        let input = foreign.alloc(DType::F32, &[1]).unwrap();
        let output = foreign.alloc(DType::F32, &[1]).unwrap();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();

        assert!(matches!(
            backend.submit(commands),
            Err(BackendError::InvalidInput)
        ));
    }

    #[test]
    fn rejects_foreign_tensors() {
        let backend = CpuBackend::new();
        let foreign = CpuBackend::new().alloc(DType::F32, &[1]).unwrap();
        assert_eq!(backend.read(&foreign), Err(BackendError::InvalidInput));
    }

    #[test]
    fn executes_bound_programs_with_multiple_outputs() {
        let backend = CpuBackend::new();
        let values = backend.alloc(DType::F32, &[2]).unwrap();
        backend.write(&values, &f32_bytes(&[1.5, -2.0])).unwrap();
        let indices = backend.alloc(DType::U32, &[2]).unwrap();
        backend
            .write(
                &indices,
                &[2_u32.to_le_bytes(), 5_u32.to_le_bytes()].concat(),
            )
            .unwrap();
        let copied = backend.alloc(DType::F16, &[2]).unwrap();
        let summed = backend.alloc(DType::F32, &[2]).unwrap();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Cast(ValueType::F32, 1),
                Inst::Binary(BinOp::Add, 0, 2),
            ],
            outputs: vec![(0, 0), (1, 3)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(
            &backend,
            program,
            KernelSignature::new(
                1,
                vec![DType::F32, DType::U32],
                vec![DType::F16, DType::F32],
                0,
            ),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&values, &indices], &[&copied, &summed])
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();

        assert_eq!(
            backend.read(&copied).unwrap(),
            [
                f16::from_f32(1.5).to_le_bytes(),
                f16::from_f32(-2.0).to_le_bytes()
            ]
            .concat()
        );
        assert_eq!(backend.read(&summed).unwrap(), f32_bytes(&[3.5, 3.0]));
    }

    #[test]
    fn prepared_program_can_be_dispatched_repeatedly() {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        backend
            .write(&input, &f32_bytes(&vec![1.5; 7 * 1024]))
            .unwrap();
        let expected = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        let actual = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Const(2.0),
                Inst::Binary(BinOp::Add, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(
            &backend,
            program.clone(),
            KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&input], &[&expected])
            .unwrap();
        commands
            .dispatch_kernel(&prepared, &[&input], &[&actual])
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();

        assert_eq!(
            backend.read(&actual).unwrap(),
            backend.read(&expected).unwrap()
        );
    }

    #[test]
    fn prepared_dispatch_refuses_dtype_and_rank_mismatches() {
        let backend = CpuBackend::new();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(
            &backend,
            program,
            KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let input = backend.alloc(DType::F16, &[7, 1024]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        assert_eq!(
            CommandList::new().dispatch_kernel(&prepared, &[&input], &[&output]),
            Err(forja_core::OpError::ProgramSignature)
        );

        let input = backend.alloc(DType::F32, &[1024]).unwrap();
        let output = backend.alloc(DType::F32, &[1024]).unwrap();
        assert_eq!(
            CommandList::new().dispatch_kernel(&prepared, &[&input], &[&output]),
            Err(forja_core::OpError::ProgramSignature)
        );
    }

    #[test]
    fn imports_a_file_mapping_without_copying() {
        let path = std::env::temp_dir().join(format!(
            "forja-cpu-mapping-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let expected = (0_u32..4).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        fs::write(&path, &expected).unwrap();
        let region = MappedRegion::map(&fs::File::open(&path).unwrap()).unwrap();
        let mapped_pointer = region.as_ptr();
        let backend = CpuBackend::new();
        let buffer = backend.import_readonly(region).unwrap();
        fs::remove_file(path).unwrap();
        let tensor = backend
            .tensor(
                buffer,
                Layout::contiguous(DType::U32, 0, vec![4], 16).unwrap(),
            )
            .unwrap();

        let buffers = backend.buffers.lock().unwrap();
        assert_eq!(
            buffers.get(&tensor).unwrap().bytes().as_ptr(),
            mapped_pointer
        );
        drop(buffers);
        assert_eq!(backend.read(&tensor).unwrap(), expected);
    }

    #[test]
    fn rejects_every_view_after_releasing_its_allocation() {
        let backend = CpuBackend::new();
        let tensor = backend.alloc(DType::F32, &[2, 3]).unwrap();
        let view = backend.view(&tensor, ViewOp::Permute(vec![1, 0])).unwrap();
        backend.release(&view).unwrap();

        assert_eq!(backend.read(&tensor), Err(BackendError::InvalidInput));
        assert_eq!(backend.read(&view), Err(BackendError::InvalidInput));
        assert_eq!(backend.release(&tensor), Err(BackendError::InvalidInput));
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
    fn read_only_views_refuse_writes() {
        let backend = CpuBackend::new();
        let path = std::env::temp_dir().join(format!(
            "forja-cpu-read-only-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, [0; 24]).unwrap();
        let buffer = backend
            .import_readonly(MappedRegion::map(&fs::File::open(&path).unwrap()).unwrap())
            .unwrap();
        fs::remove_file(path).unwrap();
        let read_only = backend
            .tensor(
                buffer,
                Layout::contiguous(DType::F32, 0, vec![2, 3], 24).unwrap(),
            )
            .unwrap();
        let view = backend
            .view(&read_only, ViewOp::Reshape(vec![3, 2]))
            .unwrap();

        assert!(!view.is_writable());
        assert_eq!(
            backend.write(&read_only, &[0; 24]),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            backend.write(&view, &[0; 24]),
            Err(BackendError::InvalidInput)
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
    fn greedy_sampling_matches_argmax_for_edge_values_and_widths() {
        for width in [1_u32, 7, 33, 4097, 151_936] {
            let mut row = (0..width)
                .map(|index| f32::from(u16::try_from(index % 19).unwrap()) - 9.0)
                .collect::<Vec<_>>();
            if width >= 7 {
                row[..7].copy_from_slice(&[
                    f32::NEG_INFINITY,
                    -0.0,
                    0.0,
                    f32::INFINITY,
                    f32::NAN,
                    f32::NAN,
                    f32::INFINITY,
                ]);
            }
            let expected = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| u32::try_from(index).unwrap())
                .unwrap();
            for parameters in [
                sampling_parameters(0.0, 0, 1.0, 7),
                sampling_parameters(0.7, 1, 0.9, 7),
            ] {
                assert_eq!(sample_row(&row, parameters, 33).unwrap(), expected);
            }
        }
    }

    #[test]
    fn sampled_tokens_are_counter_deterministic() {
        let row = [0.1, -0.3, 1.2, 0.7, 0.0, 0.9, -1.0];
        let parameters = sampling_parameters(0.7, 0, 1.0, 0xfeed_beef_dead_cafe);
        let first = (0..128)
            .map(|position| sample_row(&row, parameters, position).unwrap())
            .collect::<Vec<_>>();
        let second = (0..128)
            .map(|position| sample_row(&row, parameters, position).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(first, second);
        assert!(first.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn top_k_and_top_p_keep_only_the_exact_allowed_set() {
        let row = [10.0, 9.0, 8.0, 1.0, 0.0, -1.0, -2.0];
        let top_k = sampling_parameters(1.0, 3, 1.0, 11);
        let top_p = sampling_parameters(1.0, 0, 0.7, 11);
        let combined = sampling_parameters(1.0, 2, 0.999, 11);
        for position in 0..4096 {
            assert!(sample_row(&row, top_k, position).unwrap() <= 2);
            assert!(sample_row(&row, top_p, position).unwrap() <= 1);
            assert!(sample_row(&row, combined, position).unwrap() <= 1);
        }
    }

    #[test]
    fn sampling_distributions_match_known_probabilities() {
        assert_chi_squared(
            &[0.0, 0.0, 0.0, 0.0],
            sampling_parameters(1.0, 0, 1.0, 91),
            &[0.25; 4],
        );
        assert_chi_squared(
            &[2.0, 1.0, 0.0],
            sampling_parameters(1.0, 2, 1.0, 92),
            &[0.731_058_6, 0.268_941_4, 0.0],
        );
        assert_chi_squared(
            &[2.0, 1.0, 0.0],
            sampling_parameters(1.0, 0, 0.8, 93),
            &[0.731_058_6, 0.268_941_4, 0.0],
        );
        assert_chi_squared(
            &[2.0, 1.0, 0.0, -1.0],
            sampling_parameters(1.0, 3, 0.95, 94),
            &[0.665_240_94, 0.244_728_48, 0.090_030_57, 0.0],
        );
    }

    #[test]
    fn sample_refuses_invalid_live_parameters() {
        let backend = CpuBackend::new();
        let logits = backend.alloc(DType::F32, &[7]).unwrap();
        backend.write(&logits, &f32_bytes(&[0.0; 7])).unwrap();
        let params = backend.alloc(DType::U32, &[5]).unwrap();
        backend
            .write(
                &params,
                &u32_bytes(&[f32::NAN.to_bits(), 0, 1.0_f32.to_bits(), 0, 0]),
            )
            .unwrap();
        let output = backend.alloc(DType::U32, &[]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Sample { position: 0 }, &[&logits, &params], &output)
            .unwrap();
        assert_eq!(
            backend.submit(commands).unwrap().wait(),
            Err(BackendError::InvalidInput)
        );
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

    #[test]
    fn sdpa_matches_a_hand_checked_two_key_case() {
        let backend = CpuBackend::new();
        let q = f32_tensor(&backend, &[1, 1, 2], &[1.0, 0.0]);
        let k = f32_tensor(&backend, &[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v = f32_tensor(&backend, &[1, 2, 1], &[10.0, 20.0]);
        let actual = submit_sdpa(&backend, &q, &k, &v, (1.0, false, 0));
        let exponential = 1.0_f32.exp();
        assert_relative(
            &actual,
            &[(10.0 * exponential + 20.0) / (exponential + 1.0)],
            1e-5,
        );
    }

    #[test]
    fn sdpa_groups_qwen_heads_over_strided_cache_views() {
        let backend = CpuBackend::new();
        let keys = backend.alloc(DType::F32, &[8, 4096, 128]).unwrap();
        let values = backend.alloc(DType::F32, &[8, 4096, 128]).unwrap();
        let stored = (0_u16..8)
            .flat_map(|head| std::iter::repeat_n(f32::from(head), 4096 * 128))
            .collect::<Vec<_>>();
        backend.write(&values, &f32_bytes(&stored)).unwrap();
        let slice = vec![
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(0, 7, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        let keys = backend.view(&keys, ViewOp::Slice(slice.clone())).unwrap();
        let values = backend.view(&values, ViewOp::Slice(slice)).unwrap();
        assert!(!keys.layout().is_contiguous() && !values.layout().is_contiguous());
        let query = backend.alloc(DType::F32, &[16, 1, 128]).unwrap();
        let actual = submit_sdpa(
            &backend,
            &query,
            &keys,
            &values,
            (128.0_f32.sqrt().recip(), true, 6),
        );
        let expected = (0_u16..8)
            .flat_map(|head| std::iter::repeat_n(f32::from(head), 2 * 128))
            .collect::<Vec<_>>();
        assert_relative(&actual, &expected, 1e-5);
    }

    #[test]
    fn sdpa_decode_equals_the_last_prefill_row() {
        let backend = CpuBackend::new();
        let key = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let query = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = f32_tensor(&backend, &[1, 4, 2], &key);
        let v = f32_tensor(&backend, &[1, 4, 2], &value);
        let q = f32_tensor(&backend, &[1, 4, 2], &query);
        let prefill = submit_sdpa(&backend, &q, &k, &v, (0.5, true, 0));
        let q = f32_tensor(&backend, &[1, 1, 2], &query[6..]);
        let decode = submit_sdpa(&backend, &q, &k, &v, (0.5, true, 3));
        assert_eq!(decode, prefill[6..]);
    }

    fn f32_tensor(backend: &CpuBackend, shape: &[u32], values: &[f32]) -> Tensor {
        let tensor = backend.alloc(DType::F32, shape).unwrap();
        backend.write(&tensor, &f32_bytes(values)).unwrap();
        tensor
    }

    fn submit_sdpa(
        backend: &CpuBackend,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        (scale, causal, q_start): (f32, bool, u32),
    ) -> Vec<f32> {
        let q_shape = query.layout().shape();
        let v_shape = value.layout().shape();
        let output = backend
            .alloc(DType::F32, &[q_shape[0], q_shape[1], v_shape[2]])
            .unwrap();
        let mut commands = CommandList::new();
        let op = Op::Sdpa {
            scale,
            causal,
            q_start,
        };
        commands
            .dispatch(op, &[query, key, value], &output)
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        decode(&backend.read(&output).unwrap(), DType::F32).unwrap()
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

    fn sampling_parameters(
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> SamplingParameters {
        SamplingParameters {
            temperature,
            top_k,
            top_p,
            seed,
        }
    }

    fn assert_chi_squared(row: &[f32], parameters: SamplingParameters, expected: &[f32]) {
        let trials = 20_000_u32;
        let mut counts = vec![0_u32; row.len()];
        for position in 0..trials {
            let token = sample_row(row, parameters, position).unwrap();
            counts[usize::try_from(token).unwrap()] += 1;
        }
        let statistic = counts
            .iter()
            .zip(expected)
            .filter(|(_, probability)| **probability > 0.0)
            .map(|(&count, &probability)| {
                let expected_count = probability * f32::from(u16::try_from(trials).unwrap());
                let difference = f32::from(u16::try_from(count).unwrap()) - expected_count;
                difference * difference / expected_count
            })
            .sum::<f32>();
        assert!(statistic < 16.3, "chi-squared statistic was {statistic}");
        for (&count, &probability) in counts.iter().zip(expected) {
            if probability == 0.0 {
                assert_eq!(count, 0);
            }
        }
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
