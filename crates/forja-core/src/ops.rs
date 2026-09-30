use std::{
    error::Error,
    fmt,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::program::{
    BindError, BoundProgram, KernelSignature, PreparedProgram, ProgramHash, bind_program,
};
use crate::{
    DType, Layout, QuantizedMatrix, QuantizedMatrixError, QuantizedMatrixPart, byte_ranges_overlap,
    is_injective,
};

/// An opaque allocation identity unique to one backend instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferId {
    backend: u64,
    allocation: u64,
    byte_len: u64,
}

impl BufferId {
    /// Creates an allocation identity for use by a backend implementation.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(backend: u64, allocation: u64, byte_len: u64) -> Self {
        Self {
            backend,
            allocation,
            byte_len,
        }
    }

    /// Returns the backend instance identity.
    #[doc(hidden)]
    #[must_use]
    pub const fn backend(self) -> u64 {
        self.backend
    }

    /// Returns the backend-local allocation identity.
    #[doc(hidden)]
    #[must_use]
    pub const fn allocation(self) -> u64 {
        self.allocation
    }

    /// Returns the registered allocation length.
    #[doc(hidden)]
    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }
}

/// A reason tensor construction failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    /// The layout was validated against a different buffer length.
    BufferLengthMismatch,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("layout and buffer lengths differ")
    }
}

impl Error for TensorError {}

/// A validated tensor view into a backend allocation.
///
/// Backends re-check the allocation identity and length whenever they receive
/// a tensor. Guests only reach tensors through resource handles maintained by
/// the host, rather than constructing this type directly.
///
/// ```compile_fail
/// use forja_core::{BufferId, DType, Layout, Tensor};
///
/// let layout = Layout::contiguous(DType::F32, 0, vec![1], 4)?;
/// let _ = Tensor::new(BufferId::new(1, 1, 4), layout)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tensor {
    buffer: BufferId,
    layout: Layout,
    writable: bool,
}

impl Tensor {
    pub(crate) fn from_allocation(
        buffer: BufferId,
        layout: Layout,
        writable: bool,
    ) -> Result<Self, TensorError> {
        if buffer.byte_len != layout.buffer_len() {
            return Err(TensorError::BufferLengthMismatch);
        }
        Ok(Self {
            buffer,
            layout,
            writable,
        })
    }

    /// Returns the allocation identity.
    #[must_use]
    pub const fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// Returns the validated layout.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Reports whether this tensor may be written.
    #[must_use]
    pub const fn is_writable(&self) -> bool {
        self.writable
    }
}

/// A trusted operation recorded in a command list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Executes a validated scalar program with this content identity.
    Program(ProgramHash),
    /// Copies and optionally casts one tensor.
    Copy,
    /// Adds two tensors elementwise.
    Add,
    /// Applies `SiLU` to a gate and multiplies it by an up tensor.
    SiluMul,
    /// Normalizes rows by their root mean square and applies a weight.
    RmsNorm {
        /// The nonnegative stabilizer added before the square root.
        eps: f32,
    },
    /// Applies stable softmax over the last axis.
    ///
    /// A row containing only negative infinity produces all zeros.
    Softmax,
    /// Selects the greatest element of each row under IEEE total order.
    ///
    /// Equal greatest values select the highest index.
    Argmax,
    /// Selects the greatest elements of each row under IEEE total order.
    ///
    /// Results are sorted descending, with lower indices first for equal values.
    TopK {
        /// Number of elements selected from each row.
        k: u32,
    },
    /// Samples one index from each row using counter-based randomness.
    ///
    /// A seed reproduces tokens for the same forja build, backend, and kernel variant. Tokens are
    /// not guaranteed to match across builds, backends, or variants.
    Sample {
        /// Absolute sequence position mixed into the random counter.
        position: u32,
    },
    /// Applies half-split rotary position embeddings.
    Rope {
        /// The positive finite frequency base.
        theta: f32,
    },
    /// Gathers embedding rows by token id.
    Embed,
    /// Multiplies rank-two or rank-three matrices.
    Matmul,
    /// Multiplies a matrix by affine-quantized transposed weights.
    QuantMatmul {
        /// Number of bits in each unsigned quantized value.
        bits: u8,
        /// Number of input elements sharing one scale and bias.
        group_size: u32,
    },
    /// Computes grouped-query scaled dot-product attention.
    Sdpa {
        /// The score multiplier.
        scale: f32,
        /// Whether keys after each query position are masked.
        causal: bool,
        /// The absolute position of the first query.
        q_start: u32,
    },
}

/// An operand named by an operation validation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    /// The input at this zero-based position.
    Input(usize),
    /// The output tensor.
    Output,
}

/// A reason an operation could not be recorded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpError {
    /// Scalar programs must be recorded with [`CommandList::dispatch_kernel`].
    ProgramRequiresBinding,
    /// Tensor views do not satisfy a scalar program's signature.
    ProgramBinding(BindError),
    /// Tensor types or rank differ from a prepared program's signature.
    ProgramSignature,
    /// The operation received the wrong number of inputs.
    Arity {
        /// The required count.
        expected: usize,
        /// The received count.
        actual: usize,
    },
    /// The operation received the wrong number of outputs.
    OutputArity {
        /// The required count.
        expected: usize,
        /// The received count.
        actual: usize,
    },
    /// An operand has a scalar type unsupported by the operation.
    DType {
        /// The offending operand.
        operand: Operand,
        /// Its scalar type.
        dtype: DType,
    },
    /// An operand shape differs from the required shape.
    Shape {
        /// The offending operand.
        operand: Operand,
    },
    /// The output maps multiple logical elements to the same storage.
    NonInjectiveOutput,
    /// The output tensor does not permit writes.
    ReadOnlyOutput,
    /// An operand contains no logical elements.
    EmptyOperand {
        /// The empty operand.
        operand: Operand,
    },
    /// An input and output may touch the same bytes.
    Aliasing {
        /// The overlapping input position.
        input: usize,
    },
    /// Two outputs may touch the same bytes.
    OutputAliasing {
        /// The first overlapping output position.
        first: usize,
        /// The second overlapping output position.
        second: usize,
    },
    /// An RMS normalization epsilon is negative or non-finite.
    InvalidEpsilon,
    /// A rotary frequency base is non-positive or non-finite.
    InvalidTheta,
    /// An attention scale is non-finite.
    InvalidScale,
    /// Quantization parameters are outside the supported set.
    InvalidQuantization,
    /// A top-k count is zero, exceeds the row width, or exceeds the supported limit.
    InvalidTopK,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid operation: {self:?}")
    }
}

impl Error for OpError {}

/// A dispatch whose signature and aliasing have been validated.
#[derive(Clone, Debug)]
pub struct Dispatch {
    op: Op,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    program: Option<ProgramDispatch>,
}

#[derive(Clone, Debug)]
struct ProgramDispatch {
    bound: BoundProgram,
    prepared: Arc<PreparedProgram>,
}

impl Dispatch {
    pub(crate) fn new(op: Op, inputs: &[&Tensor], output: &Tensor) -> Result<Self, OpError> {
        Self::new_many(op, inputs, &[output])
    }

    pub(crate) fn new_many(
        op: Op,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<Self, OpError> {
        check_common(inputs, outputs)?;
        let expected_outputs = if matches!(op, Op::TopK { .. }) { 2 } else { 1 };
        if outputs.len() != expected_outputs {
            return Err(OpError::OutputArity {
                expected: expected_outputs,
                actual: outputs.len(),
            });
        }
        let output = outputs[0];
        match op {
            Op::Program(_) => return Err(OpError::ProgramRequiresBinding),
            Op::Copy => check_copy(inputs, output)?,
            Op::Add | Op::SiluMul => check_binary(inputs, output)?,
            Op::RmsNorm { eps } => check_rms_norm(inputs, output, eps)?,
            Op::Softmax => check_softmax(inputs, output)?,
            Op::Argmax => check_argmax(inputs, output)?,
            Op::TopK { k } => check_top_k(inputs, outputs, k)?,
            Op::Sample { .. } => check_sample(inputs, output)?,
            Op::Rope { theta } => check_rope(inputs, output, theta)?,
            Op::Embed => check_embed(inputs, output)?,
            Op::Matmul => check_matmul(inputs, output)?,
            Op::QuantMatmul { bits, group_size } => {
                check_quant_matmul(inputs, output, bits, group_size)?;
            }
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => check_sdpa(inputs, output, scale, causal, q_start)?,
        }
        Ok(Self {
            op,
            inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
            outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
            program: None,
        })
    }

    pub(crate) fn kernel(
        program: &Arc<PreparedProgram>,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<Self, OpError> {
        outputs
            .first()
            .ok_or(OpError::ProgramBinding(BindError::NoOutputs))?;
        check_common(inputs, outputs)?;
        let bound =
            bind_program(program.validated(), inputs, outputs).map_err(OpError::ProgramBinding)?;
        check_kernel_signature(&bound, program.signature())?;
        bound
            .outputs()
            .first()
            .ok_or(OpError::ProgramBinding(BindError::NoOutputs))?;
        Ok(Self {
            op: Op::Program(program.validated().content_hash()),
            inputs: bound.inputs().to_vec(),
            outputs: bound.outputs().to_vec(),
            program: Some(ProgramDispatch {
                bound,
                prepared: Arc::clone(program),
            }),
        })
    }

    /// Returns the operation configuration.
    #[must_use]
    pub const fn op(&self) -> Op {
        self.op
    }
    /// Returns the input tensors.
    #[must_use]
    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
    /// Returns the output tensor.
    #[must_use]
    pub fn output(&self) -> &Tensor {
        &self.outputs[0]
    }

    /// Returns every output tensor in slot order.
    #[must_use]
    pub fn outputs(&self) -> &[Tensor] {
        &self.outputs
    }

    /// Returns the bound scalar program, when this is a program dispatch.
    #[must_use]
    pub const fn bound_program(&self) -> Option<&BoundProgram> {
        match &self.program {
            Some(program) => Some(&program.bound),
            None => None,
        }
    }

    /// Returns retained preparation state for a prepared-program dispatch.
    #[must_use]
    pub fn prepared_program(&self) -> Option<&Arc<PreparedProgram>> {
        self.program.as_ref().map(|program| &program.prepared)
    }
}

/// An ordered list of validated backend work.
#[derive(Clone, Debug, Default)]
pub struct CommandList {
    dispatches: Vec<Dispatch>,
    validation: ValidationState,
    program_recording: Duration,
    program_dispatches: u64,
}

#[derive(Clone, Debug, Default)]
enum ValidationState {
    #[default]
    Fresh,
    Prevalidated(Vec<bool>),
}

impl CommandList {
    /// Creates an empty command list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dispatches: Vec::new(),
            validation: ValidationState::Fresh,
            program_recording: Duration::ZERO,
            program_dispatches: 0,
        }
    }

    /// Returns the number of recorded dispatches.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dispatches.len()
    }

    /// Reports whether no dispatches have been recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }

    /// Validates and records a dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`] for an invalid signature or unsafe aliasing.
    pub fn dispatch(&mut self, op: Op, inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
        let dispatch = Dispatch::new(op, inputs, output)?;
        self.push_and_reset_validation(dispatch);
        Ok(())
    }

    /// Validates and records a trusted operation with multiple outputs.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`] for an invalid signature or unsafe aliasing.
    pub fn dispatch_many(
        &mut self,
        op: Op,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), OpError> {
        let dispatch = Dispatch::new_many(op, inputs, outputs)?;
        self.push_and_reset_validation(dispatch);
        Ok(())
    }

    /// Binds, validates, and records a prepared scalar-program dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::ProgramBinding`] for invalid tensor views and
    /// [`OpError::ProgramSignature`] when their types or rank differ from the prepared signature.
    pub fn dispatch_kernel(
        &mut self,
        program: &Arc<PreparedProgram>,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), OpError> {
        let started = Instant::now();
        let dispatch = Dispatch::kernel(program, inputs, outputs)?;
        self.push_and_reset_validation(dispatch);
        self.record_program(started);
        Ok(())
    }

    pub(crate) fn push_and_reset_validation(&mut self, dispatch: Dispatch) {
        self.validation = ValidationState::Fresh;
        self.dispatches.push(dispatch);
    }

    pub(crate) fn set_precomputed_barriers(&mut self, barriers: Vec<bool>) {
        self.validation = ValidationState::Prevalidated(barriers);
    }

    /// Reports whether every tensor registry entry is retained by a validated graph.
    #[doc(hidden)]
    #[must_use]
    pub const fn retained_tensors_validated(&self) -> bool {
        matches!(self.validation, ValidationState::Prevalidated(_))
    }

    fn record_program(&mut self, started: Instant) {
        self.program_recording = self.program_recording.saturating_add(started.elapsed());
        self.program_dispatches = self.program_dispatches.saturating_add(1);
    }

    /// Returns scalar-program recording count and host time.
    #[doc(hidden)]
    #[must_use]
    pub const fn program_recording(&self) -> crate::ProfileCount {
        crate::ProfileCount {
            count: self.program_dispatches,
            time: self.program_recording,
        }
    }

    /// Consumes the list into its validated dispatches.
    #[must_use]
    pub fn into_dispatches(self) -> Vec<Dispatch> {
        self.dispatches
    }

    /// Returns the most recently validated dispatch.
    #[doc(hidden)]
    #[must_use]
    pub fn last_dispatch(&self) -> Option<&Dispatch> {
        self.dispatches.last()
    }
}

fn check_kernel_signature(
    program: &BoundProgram,
    signature: &KernelSignature,
) -> Result<(), OpError> {
    let rank = program.outputs()[0].layout().shape().len();
    let inputs_match = program
        .inputs()
        .iter()
        .map(|tensor| tensor.layout().dtype())
        .eq(signature.input_dtypes().iter().copied());
    let outputs_match = program
        .outputs()
        .iter()
        .map(|tensor| tensor.layout().dtype())
        .eq(signature.output_dtypes().iter().copied());
    if rank == usize::from(signature.rank()) && inputs_match && outputs_match {
        Ok(())
    } else {
        Err(OpError::ProgramSignature)
    }
}

/// Reports whether each dispatch needs a barrier before it.
#[must_use]
pub fn required_barriers(commands: &CommandList) -> Vec<bool> {
    if let ValidationState::Prevalidated(barriers) = &commands.validation {
        return barriers.clone();
    }
    barriers_for_accesses(commands.dispatches.iter().map(dispatch_accesses))
}

#[derive(Clone, Debug)]
pub(crate) struct BufferAccess {
    pub(crate) buffer: BufferId,
    region: AccessRegion,
    pub(crate) writes: bool,
}

#[derive(Clone, Debug)]
enum AccessRegion {
    Empty,
    Exact(Layout),
    Hull(Range<u64>),
}

pub(crate) fn dispatch_accesses(dispatch: &Dispatch) -> Vec<BufferAccess> {
    dispatch
        .inputs
        .iter()
        .map(|tensor| BufferAccess::new(tensor, false))
        .chain(
            dispatch
                .outputs()
                .iter()
                .map(|tensor| BufferAccess::new(tensor, true)),
        )
        .collect()
}

pub(crate) fn barriers_for_accesses(
    dispatches: impl IntoIterator<Item = Vec<BufferAccess>>,
) -> Vec<bool> {
    let mut prior = Vec::<BufferAccess>::new();
    dispatches
        .into_iter()
        .map(|current| {
            let needs_barrier = current
                .iter()
                .any(|access| prior.iter().any(|candidate| access.conflicts(candidate)));
            if needs_barrier {
                prior.clear();
            }
            prior.extend(current);
            needs_barrier
        })
        .collect()
}

pub(crate) fn barriers_bounded_by(dispatches: &[Vec<BufferAccess>], upper: &[bool]) -> Vec<bool> {
    let mut barriers = vec![false; dispatches.len()];
    for (current_index, current) in dispatches.iter().enumerate() {
        for (prior_index, prior) in dispatches[..current_index].iter().enumerate() {
            if current
                .iter()
                .any(|access| prior.iter().any(|candidate| access.conflicts(candidate)))
            {
                let candidates = &upper[prior_index + 1..=current_index];
                if let Some(offset) = candidates.iter().position(|&candidate| candidate) {
                    barriers[prior_index + 1 + offset] = true;
                } else {
                    barriers[current_index] = true;
                }
            }
        }
    }
    barriers
}

impl BufferAccess {
    pub(crate) fn new(tensor: &Tensor, writes: bool) -> Self {
        let region = if tensor.layout().element_count() == 0 {
            AccessRegion::Empty
        } else {
            AccessRegion::Exact(tensor.layout().clone())
        };
        Self {
            buffer: tensor.buffer(),
            region,
            writes,
        }
    }

    pub(crate) fn hull(buffer: BufferId, bytes: Option<Range<u64>>, writes: bool) -> Self {
        Self {
            buffer,
            region: bytes.map_or(AccessRegion::Empty, AccessRegion::Hull),
            writes,
        }
    }

    pub(crate) fn as_hull(&self) -> Self {
        let bytes = match &self.region {
            AccessRegion::Empty => None,
            region => Some(region.byte_span()),
        };
        Self::hull(self.buffer, bytes, self.writes)
    }

    pub(crate) fn conflicts(&self, prior: &Self) -> bool {
        (self.writes || prior.writes)
            && self.buffer == prior.buffer
            && match (&self.region, &prior.region) {
                (AccessRegion::Exact(current), AccessRegion::Exact(other)) => {
                    byte_ranges_overlap(current, other)
                }
                (AccessRegion::Empty, _) | (_, AccessRegion::Empty) => false,
                (current, other) => {
                    let current = current.byte_span();
                    let other = other.byte_span();
                    current.start < other.end && other.start < current.end
                }
            }
    }
}

impl AccessRegion {
    fn byte_span(&self) -> Range<u64> {
        match self {
            Self::Exact(layout) => layout.byte_span(),
            Self::Hull(bytes) => bytes.clone(),
            Self::Empty => 0..0,
        }
    }
}

fn check_common(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), OpError> {
    for (input, tensor) in inputs.iter().enumerate() {
        if tensor.layout.element_count() == 0 {
            return Err(OpError::EmptyOperand {
                operand: Operand::Input(input),
            });
        }
    }
    if outputs.is_empty() {
        return Err(OpError::OutputArity {
            expected: 1,
            actual: 0,
        });
    }
    for output in outputs {
        if output.layout.element_count() == 0 {
            return Err(OpError::EmptyOperand {
                operand: Operand::Output,
            });
        }
        if !output.is_writable() {
            return Err(OpError::ReadOnlyOutput);
        }
        if !is_injective(output.layout()) {
            return Err(OpError::NonInjectiveOutput);
        }
        for (input, tensor) in inputs.iter().enumerate() {
            if tensor.buffer == output.buffer
                && byte_ranges_overlap(tensor.layout(), output.layout())
            {
                return Err(OpError::Aliasing { input });
            }
        }
    }
    for (first, output) in outputs.iter().enumerate() {
        for (second, candidate) in outputs.iter().enumerate().skip(first + 1) {
            if output.buffer == candidate.buffer
                && byte_ranges_overlap(output.layout(), candidate.layout())
            {
                return Err(OpError::OutputAliasing { first, second });
            }
        }
    }
    Ok(())
}

fn check_copy(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    let input_dtype = inputs[0].layout.dtype();
    let output_dtype = output.layout.dtype();
    let float_cast = matches!(input_dtype, DType::F32 | DType::F16 | DType::BF16)
        && matches!(output_dtype, DType::F32 | DType::F16 | DType::BF16);
    let integer_copy =
        matches!(input_dtype, DType::I32 | DType::U32) && input_dtype == output_dtype;
    if !float_cast && !integer_copy {
        return Err(OpError::DType {
            operand: Operand::Input(0),
            dtype: input_dtype,
        });
    }
    check_shape(output, inputs[0], Operand::Output)?;
    Ok(())
}

fn check_binary(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    check_shape(inputs[1], inputs[0], Operand::Input(1))?;
    check_shape(output, inputs[0], Operand::Output)
}

fn check_shape(tensor: &Tensor, expected: &Tensor, operand: Operand) -> Result<(), OpError> {
    if tensor.layout.shape() != expected.layout.shape() {
        return Err(OpError::Shape { operand });
    }
    Ok(())
}

fn check_rms_norm(inputs: &[&Tensor], output: &Tensor, eps: f32) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OpError::InvalidEpsilon);
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    let Some(&width) = inputs[0].layout.shape().last() else {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    };
    if inputs[1].layout.shape() != [width] {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    check_shape(output, inputs[0], Operand::Output)
}

fn check_softmax(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(output, Operand::Output)?;
    if inputs[0].layout.shape().last().is_none() {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    }
    check_shape(output, inputs[0], Operand::Output)
}

fn check_argmax(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    if output.layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: output.layout.dtype(),
        });
    }
    let Some((_, rows)) = inputs[0].layout.shape().split_last() else {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    };
    if output.layout.shape() != rows {
        return Err(OpError::Shape {
            operand: Operand::Output,
        });
    }
    Ok(())
}

fn check_top_k(inputs: &[&Tensor], outputs: &[&Tensor], k: u32) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    if outputs[0].layout.dtype() != inputs[0].layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: outputs[0].layout.dtype(),
        });
    }
    if outputs[1].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: outputs[1].layout.dtype(),
        });
    }
    let Some((&width, rows)) = inputs[0].layout.shape().split_last() else {
        return Err(shape_error(Operand::Input(0)));
    };
    if k == 0 || k > 64 || k > width {
        return Err(OpError::InvalidTopK);
    }
    let expected = rows.iter().copied().chain([k]).collect::<Vec<_>>();
    if outputs
        .iter()
        .any(|output| output.layout.shape() != expected)
    {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

fn check_sample(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    if inputs[1].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(1),
            dtype: inputs[1].layout.dtype(),
        });
    }
    if inputs[1].layout.shape() != [5] {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    check_argmax(&inputs[..1], output)
}

fn check_rope(inputs: &[&Tensor], output: &Tensor, theta: f32) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(OpError::InvalidTheta);
    }
    check_float(inputs[0], Operand::Input(0))?;
    if inputs[1].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(1),
            dtype: inputs[1].layout.dtype(),
        });
    }
    check_float(output, Operand::Output)?;
    let shape = inputs[0].layout.shape();
    if shape.len() != 3 || !shape[2].is_multiple_of(2) {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    }
    if inputs[1].layout.shape() != [shape[0]] {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    check_shape(output, inputs[0], Operand::Output)
}

fn check_embed(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    if inputs[1].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(1),
            dtype: inputs[1].layout.dtype(),
        });
    }
    check_float(output, Operand::Output)?;
    let table = inputs[0].layout.shape();
    let ids = inputs[1].layout.shape();
    if table.len() != 2 {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    }
    if ids.len() != 1 {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    if output.layout.shape() != [ids[0], table[1]] {
        return Err(OpError::Shape {
            operand: Operand::Output,
        });
    }
    Ok(())
}

fn check_matmul(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    let a = inputs[0].layout.shape();
    let b = inputs[1].layout.shape();
    if !matches!(a.len(), 2 | 3) {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    }
    let batch_axes = a.len() - 2;
    if b.len() != a.len()
        || b[..batch_axes] != a[..batch_axes]
        || b[batch_axes] != a[batch_axes + 1]
    {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    let mut expected = a[..batch_axes].to_vec();
    expected.extend([a[batch_axes], b[batch_axes + 1]]);
    if output.layout.shape() != expected {
        return Err(OpError::Shape {
            operand: Operand::Output,
        });
    }
    Ok(())
}

fn check_quant_matmul(
    inputs: &[&Tensor],
    output: &Tensor,
    bits: u8,
    group_size: u32,
) -> Result<(), OpError> {
    if inputs.len() != 4 {
        return Err(OpError::Arity {
            expected: 4,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(output, Operand::Output)?;
    if output.layout.dtype() != inputs[0].layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: output.layout.dtype(),
        });
    }
    let [m, inner]: [u32; 2] = inputs[0]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(0)))?;
    let [out, _packed_width]: [u32; 2] = inputs[1]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(1)))?;
    QuantizedMatrix::new(
        out,
        inner,
        bits,
        group_size,
        inputs[1].layout.clone(),
        inputs[2].layout.clone(),
        inputs[3].layout.clone(),
    )
    .map_err(|error| match error {
        QuantizedMatrixError::UnsupportedBitWidth | QuantizedMatrixError::UnsupportedGroupSize => {
            OpError::InvalidQuantization
        }
        QuantizedMatrixError::ColumnsNotPackable | QuantizedMatrixError::ColumnsNotGrouped => {
            shape_error(Operand::Input(0))
        }
        QuantizedMatrixError::DTypeMismatch { part } => match part {
            QuantizedMatrixPart::Packed => OpError::DType {
                operand: Operand::Input(1),
                dtype: inputs[1].layout.dtype(),
            },
            QuantizedMatrixPart::Scales => OpError::DType {
                operand: Operand::Input(2),
                dtype: inputs[2].layout.dtype(),
            },
            QuantizedMatrixPart::Biases => OpError::DType {
                operand: Operand::Input(3),
                dtype: inputs[3].layout.dtype(),
            },
        },
        QuantizedMatrixError::ShapeMismatch { part } => match part {
            QuantizedMatrixPart::Packed => shape_error(Operand::Input(1)),
            QuantizedMatrixPart::Scales => shape_error(Operand::Input(2)),
            QuantizedMatrixPart::Biases => shape_error(Operand::Input(3)),
        },
    })?;
    if output.layout.shape() != [m, out] {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

/// Counts the floating-point operations in a rank-2 or rank-3 matrix multiply.
#[must_use]
pub fn matmul_flops(left: &[u32], right: &[u32]) -> Option<u64> {
    let batch = if left.len() == 3 {
        u64::from(*left.first()?)
    } else {
        1
    };
    let rank = left.len();
    batch
        .checked_mul(u64::from(*left.get(rank.checked_sub(2)?)?))?
        .checked_mul(u64::from(*left.last()?))?
        .checked_mul(u64::from(*right.last()?))?
        .checked_mul(2)
}

/// Counts the multiply-add operations in an affine quantized matrix multiply.
#[must_use]
pub fn quant_matmul_flops(input: &[u32], packed: &[u32]) -> Option<u64> {
    let [rows, inner]: [u32; 2] = input.try_into().ok()?;
    let [columns, _]: [u32; 2] = packed.try_into().ok()?;
    u64::from(rows)
        .checked_mul(u64::from(inner))?
        .checked_mul(u64::from(columns))?
        .checked_mul(2)
}

/// Counts the floating-point operations in scaled dot-product attention.
#[must_use]
pub fn sdpa_flops(query: &[u32], key: &[u32], value: &[u32]) -> Option<u64> {
    u64::from(*query.first()?)
        .checked_mul(u64::from(*query.get(1)?))?
        .checked_mul(u64::from(*key.get(1)?))?
        .checked_mul(u64::from(*query.get(2)?).checked_add(u64::from(*value.get(2)?))?)?
        .checked_mul(2)
}

fn check_sdpa(
    inputs: &[&Tensor],
    output: &Tensor,
    scale: f32,
    causal: bool,
    q_start: u32,
) -> Result<(), OpError> {
    if inputs.len() != 3 {
        return Err(OpError::Arity {
            expected: 3,
            actual: inputs.len(),
        });
    }
    for (index, input) in inputs.iter().enumerate() {
        check_float(input, Operand::Input(index))?;
    }
    check_float(output, Operand::Output)?;
    if !scale.is_finite() {
        return Err(OpError::InvalidScale);
    }
    let [hq, sq, d] = shape3(inputs[0], Operand::Input(0))?;
    let [hkv, skv, kd] = shape3(inputs[1], Operand::Input(1))?;
    if d != kd || !hq.is_multiple_of(hkv) {
        return Err(shape_error(Operand::Input(1)));
    }
    let [vh, vs, dv] = shape3(inputs[2], Operand::Input(2))?;
    if vh != hkv || vs != skv {
        return Err(shape_error(Operand::Input(2)));
    }
    if causal && q_start.checked_add(sq).is_none_or(|end| end > skv) {
        return Err(shape_error(Operand::Input(0)));
    }
    if output.layout.shape() != [hq, sq, dv] {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

const fn shape_error(operand: Operand) -> OpError {
    OpError::Shape { operand }
}

fn shape3(tensor: &Tensor, operand: Operand) -> Result<[u32; 3], OpError> {
    tensor
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(operand))
}

fn check_float(tensor: &Tensor, operand: Operand) -> Result<(), OpError> {
    match tensor.layout.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        dtype => Err(OpError::DType { operand, dtype }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(buffer: u64, dtype: DType, shape: &[u32], strides: &[u64]) -> Tensor {
        let bytes = u64::from(shape.iter().product::<u32>()) * dtype.byte_size();
        Tensor::from_allocation(
            BufferId::new(1, buffer, bytes),
            Layout::new(dtype, 0, shape.to_vec(), strides.to_vec(), bytes).unwrap(),
            true,
        )
        .unwrap()
    }

    fn slice(buffer: u64, start: u64, len: u32) -> Tensor {
        Tensor::from_allocation(
            BufferId::new(1, buffer, 64),
            Layout::new(DType::F32, start, vec![len], vec![1], 64).unwrap(),
            true,
        )
        .unwrap()
    }

    fn lane(buffer: u64, offset: u64) -> Tensor {
        Tensor::from_allocation(
            BufferId::new(1, buffer, 32),
            Layout::new(DType::F32, offset, vec![4, 1], vec![2, 1], 32).unwrap(),
            true,
        )
        .unwrap()
    }

    #[test]
    fn independent_dispatches_need_no_barriers() {
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Add,
                &[&slice(1, 0, 4), &slice(2, 0, 4)],
                &slice(3, 0, 4),
            )
            .unwrap();
        commands
            .dispatch(
                Op::Add,
                &[&slice(4, 0, 4), &slice(5, 0, 4)],
                &slice(6, 0, 4),
            )
            .unwrap();

        assert_eq!(required_barriers(&commands), [false, false]);
    }

    #[test]
    fn bounded_barriers_fall_back_at_an_uncovered_dependency() {
        let buffer = BufferId::new(1, 1, 16);
        let accesses = [
            vec![BufferAccess::hull(buffer, Some(0..16), true)],
            vec![BufferAccess::hull(buffer, Some(0..16), false)],
        ];

        assert_eq!(
            barriers_bounded_by(&accesses, &[false, false]),
            [false, true]
        );
    }

    #[test]
    fn disjoint_strided_regions_need_no_barrier() {
        let source = tensor(2, DType::F32, &[4, 1], &[1, 1]);
        let sink = tensor(3, DType::F32, &[4, 1], &[1, 1]);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&source], &lane(1, 0))
            .unwrap();
        commands.dispatch(Op::Copy, &[&lane(1, 1)], &sink).unwrap();

        assert_eq!(required_barriers(&commands), [false, false]);
    }

    #[test]
    fn dependency_chain_needs_a_barrier_at_each_link() {
        let a = slice(1, 0, 4);
        let b = slice(2, 0, 4);
        let first = slice(3, 0, 4);
        let second = slice(4, 0, 4);
        let third = slice(5, 0, 4);
        let mut commands = CommandList::new();
        commands.dispatch(Op::Add, &[&a, &b], &first).unwrap();
        commands.dispatch(Op::Add, &[&first, &b], &second).unwrap();
        commands.dispatch(Op::Add, &[&second, &b], &third).unwrap();

        assert_eq!(required_barriers(&commands), [false, true, true]);
    }

    #[test]
    fn kv_cache_write_after_read_needs_a_barrier() {
        let cache_read = slice(1, 4, 4);
        let cache_write = slice(1, 6, 4);
        let source = slice(2, 0, 4);
        let first_output = slice(3, 0, 4);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&cache_read], &first_output)
            .unwrap();
        commands
            .dispatch(Op::Copy, &[&source], &cache_write)
            .unwrap();

        assert_eq!(required_barriers(&commands), [false, true]);
    }

    #[test]
    fn disjoint_slices_of_one_buffer_need_no_barrier() {
        let first_half = slice(1, 0, 4);
        let second_half = slice(1, 8, 4);
        let source = slice(2, 0, 4);
        let output = slice(3, 0, 4);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&source], &first_half)
            .unwrap();
        commands
            .dispatch(Op::Copy, &[&second_half], &output)
            .unwrap();

        assert_eq!(required_barriers(&commands), [false, false]);
    }

    #[test]
    fn copy_rejects_each_invalid_signature() {
        let input = tensor(1, DType::F32, &[2], &[1]);
        let output = tensor(2, DType::F32, &[2], &[0]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &output),
            Err(OpError::NonInjectiveOutput)
        );
        let overlap = Tensor::from_allocation(
            input.buffer(),
            Layout::contiguous(DType::F32, 0, vec![2], 8).unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &overlap),
            Err(OpError::Aliasing { input: 0 })
        );
        let integer = tensor(3, DType::I32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&integer], &input),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::I32
            })
        );
        let wrong_shape = tensor(4, DType::F32, &[1], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &wrong_shape),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn dispatch_rejects_a_read_only_output() {
        let input = tensor(1, DType::F32, &[2], &[1]);
        let writable = tensor(2, DType::F32, &[2], &[1]);
        let output =
            Tensor::from_allocation(writable.buffer(), writable.layout().clone(), false).unwrap();

        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &output),
            Err(OpError::ReadOnlyOutput)
        );
    }

    #[test]
    fn add_rejects_invalid_signatures() {
        let input = tensor(1, DType::F32, &[2], &[1]);
        let integer = tensor(2, DType::U32, &[2], &[1]);
        let wrong_shape = tensor(3, DType::F32, &[1], &[1]);
        let output = tensor(4, DType::F32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input, &integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input, &wrong_shape], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
    }

    #[test]
    fn silu_mul_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[2], &[1]);
        let integer = tensor(2, DType::I32, &[2], &[1]);
        let output = tensor(3, DType::BF16, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::SiluMul, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::SiluMul, &[&input, &integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::I32
            })
        );
    }

    #[test]
    fn rms_norm_rejects_invalid_signatures() {
        let input = tensor(1, DType::F32, &[3, 2], &[2, 1]);
        let weight = tensor(2, DType::F32, &[3], &[1]);
        let output = tensor(3, DType::F32, &[3, 2], &[2, 1]);
        let integer = tensor(4, DType::I32, &[3, 2], &[2, 1]);
        let wrong_output = tensor(5, DType::F32, &[3, 1], &[1, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: -1.0 }, &[&input, &weight], &output),
            Err(OpError::InvalidEpsilon)
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&input, &weight], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&integer, &weight], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::I32
            })
        );
        let good_weight = tensor(6, DType::F32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::RmsNorm { eps: 0.0 },
                &[&input, &good_weight],
                &wrong_output
            ),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn softmax_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[2], &[1]);
        let integer = tensor(2, DType::U32, &[2], &[1]);
        let output = tensor(3, DType::F32, &[2], &[1]);
        let scalar = tensor(4, DType::F32, &[], &[]);
        let scalar_output = tensor(5, DType::F32, &[], &[]);
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[], &output),
            Err(OpError::Arity {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&scalar], &scalar_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&input], &scalar),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn argmax_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[3, 7], &[7, 1]);
        let integer = tensor(2, DType::U32, &[3, 7], &[7, 1]);
        let output = tensor(3, DType::U32, &[3], &[1]);
        let float_output = tensor(4, DType::F32, &[3], &[1]);
        let wrong_output = tensor(5, DType::U32, &[1], &[1]);
        let scalar = tensor(6, DType::F32, &[], &[]);
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax, &[], &output),
            Err(OpError::Arity {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax, &[&integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax, &[&input], &float_output),
            Err(OpError::DType {
                operand: Operand::Output,
                dtype: DType::F32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax, &[&input], &wrong_output),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax, &[&scalar], &wrong_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
    }

    #[test]
    fn top_k_checks_count_outputs_and_shapes() {
        let input = tensor(1, DType::F16, &[3, 7], &[7, 1]);
        let values = tensor(2, DType::F16, &[3, 7], &[7, 1]);
        let indices = tensor(3, DType::U32, &[3, 7], &[7, 1]);
        assert!(
            CommandList::new()
                .dispatch_many(Op::TopK { k: 7 }, &[&input], &[&values, &indices])
                .is_ok()
        );
        assert_eq!(
            CommandList::new().dispatch_many(Op::TopK { k: 0 }, &[&input], &[&values, &indices]),
            Err(OpError::InvalidTopK)
        );
        assert_eq!(
            CommandList::new().dispatch_many(Op::TopK { k: 8 }, &[&input], &[&values, &indices]),
            Err(OpError::InvalidTopK)
        );
        assert_eq!(
            CommandList::new().dispatch_many(Op::TopK { k: 7 }, &[&input], &[&values]),
            Err(OpError::OutputArity {
                expected: 2,
                actual: 1
            })
        );
        let wrong_indices = tensor(4, DType::U32, &[3, 1], &[1, 1]);
        assert_eq!(
            CommandList::new().dispatch_many(
                Op::TopK { k: 7 },
                &[&input],
                &[&values, &wrong_indices]
            ),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn selection_rejects_empty_rows_before_dispatch() {
        let logits = tensor(1, DType::F32, &[1, 0], &[0, 1]);
        let params = tensor(2, DType::U32, &[5], &[1]);
        let output = tensor(3, DType::U32, &[1], &[1]);

        for op in [Op::Argmax, Op::Sample { position: 9 }] {
            let inputs = match op {
                Op::Sample { .. } => vec![&logits, &params],
                _ => vec![&logits],
            };
            assert_eq!(
                CommandList::new().dispatch(op, &inputs, &output),
                Err(OpError::EmptyOperand {
                    operand: Operand::Input(0)
                })
            );
        }
    }

    #[test]
    fn sample_requires_logits_five_parameter_words_and_row_outputs() {
        let logits = tensor(1, DType::BF16, &[3, 7], &[7, 1]);
        let params = tensor(2, DType::U32, &[5], &[1]);
        let short_params = tensor(3, DType::U32, &[4], &[1]);
        let output = tensor(4, DType::U32, &[3], &[1]);
        let float_output = tensor(5, DType::F32, &[3], &[1]);

        assert!(
            CommandList::new()
                .dispatch(Op::Sample { position: 9 }, &[&logits, &params], &output)
                .is_ok()
        );
        assert_eq!(
            CommandList::new().dispatch(
                Op::Sample { position: 9 },
                &[&logits, &short_params],
                &output
            ),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(
                Op::Sample { position: 9 },
                &[&logits, &params],
                &float_output
            ),
            Err(OpError::DType {
                operand: Operand::Output,
                dtype: DType::F32
            })
        );
    }

    #[test]
    fn rope_rejects_invalid_signatures() {
        let odd = tensor(1, DType::F32, &[7, 16, 127], &[2032, 127, 1]);
        let positions = tensor(2, DType::U32, &[7], &[1]);
        let odd_output = tensor(3, DType::F32, &[7, 16, 127], &[2032, 127, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Rope { theta: 1e6 }, &[&odd, &positions], &odd_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        let input = tensor(4, DType::F32, &[7, 16, 128], &[2048, 128, 1]);
        let output = tensor(5, DType::F32, &[7, 16, 128], &[2048, 128, 1]);
        let short_positions = tensor(6, DType::U32, &[6], &[1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::Rope { theta: 1e6 },
                &[&input, &short_positions],
                &output
            ),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let float_positions = tensor(7, DType::F32, &[7], &[1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::Rope { theta: 1e6 },
                &[&input, &float_positions],
                &output
            ),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::F32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Rope { theta: 0.0 }, &[&input, &positions], &output),
            Err(OpError::InvalidTheta)
        );
        let empty = tensor(8, DType::F32, &[7, 0, 128], &[0, 128, 1]);
        let empty_output = tensor(9, DType::F32, &[7, 0, 128], &[0, 128, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::Rope { theta: 1e6 },
                &[&empty, &positions],
                &empty_output
            ),
            Err(OpError::EmptyOperand {
                operand: Operand::Input(0)
            })
        );
    }

    #[test]
    fn embed_rejects_invalid_signatures() {
        let table = tensor(1, DType::F32, &[1000, 1024], &[1024, 1]);
        let ids = tensor(2, DType::U32, &[3], &[1]);
        let wrong_output = tensor(3, DType::F32, &[3, 1023], &[1023, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Embed, &[&table, &ids], &wrong_output),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
        let float_ids = tensor(4, DType::F32, &[3], &[1]);
        let output = tensor(5, DType::F32, &[3, 1024], &[1024, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Embed, &[&table, &float_ids], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::F32
            })
        );
    }

    #[test]
    fn matmul_rejects_invalid_shapes() {
        let a = tensor(1, DType::F32, &[2, 3, 4], &[12, 4, 1]);
        let b = tensor(2, DType::F32, &[2, 3, 5], &[15, 5, 1]);
        let output = tensor(3, DType::F32, &[2, 3, 5], &[15, 5, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Matmul, &[&a, &b], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let rank_four = tensor(4, DType::F32, &[2, 2, 3, 4], &[24, 12, 4, 1]);
        let rank_four_b = tensor(5, DType::F32, &[2, 2, 4, 5], &[40, 20, 5, 1]);
        let rank_four_output = tensor(6, DType::F32, &[2, 2, 3, 5], &[30, 15, 5, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Matmul, &[&rank_four, &rank_four_b], &rank_four_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        let wrong_batch = tensor(7, DType::F32, &[3, 4, 5], &[20, 5, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Matmul, &[&a, &wrong_batch], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
    }

    #[test]
    fn quant_matmul_accepts_model_shapes_and_odd_rows() {
        let shapes = [
            (2048, 4096, 4),
            (2048, 512, 4),
            (2048, 512, 4),
            (4096, 2048, 4),
            (2048, 768, 4),
            (768, 2048, 4),
            (2048, 128, 8),
        ];
        let mut buffer = 100;
        for rows in [1, 7, 33] {
            for (inner, out, bits) in shapes {
                let packed_width = inner * bits / 32;
                let groups = inner / 64;
                let input = tensor(buffer, DType::BF16, &[rows, inner], &[inner.into(), 1]);
                let packed = tensor(
                    buffer + 1,
                    DType::U32,
                    &[out, packed_width],
                    &[packed_width.into(), 1],
                );
                let scales = tensor(buffer + 2, DType::BF16, &[out, groups], &[groups.into(), 1]);
                let biases = tensor(buffer + 3, DType::BF16, &[out, groups], &[groups.into(), 1]);
                let output = tensor(buffer + 4, DType::BF16, &[rows, out], &[out.into(), 1]);
                let op = Op::QuantMatmul {
                    bits: u8::try_from(bits).unwrap(),
                    group_size: 64,
                };
                CommandList::new()
                    .dispatch(op, &[&input, &packed, &scales, &biases], &output)
                    .unwrap();
                buffer += 5;
            }
        }
    }

    #[test]
    fn quant_matmul_rejects_invalid_packing_and_groups() {
        let input = tensor(1, DType::F32, &[7, 64], &[64, 1]);
        let packed = tensor(2, DType::U32, &[3, 8], &[8, 1]);
        let scales = tensor(3, DType::F16, &[3, 1], &[1, 1]);
        let biases = tensor(4, DType::F16, &[3, 1], &[1, 1]);
        let output = tensor(5, DType::F32, &[7, 3], &[3, 1]);
        let op = Op::QuantMatmul {
            bits: 4,
            group_size: 64,
        };

        let bad_packed = tensor(6, DType::U32, &[3, 7], &[7, 1]);
        assert_eq!(
            CommandList::new().dispatch(op, &[&input, &bad_packed, &scales, &biases], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let bad_inner = tensor(7, DType::F32, &[7, 63], &[63, 1]);
        assert_eq!(
            CommandList::new().dispatch(op, &[&bad_inner, &packed, &scales, &biases], &output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        let bad_scales = tensor(8, DType::F16, &[3, 2], &[2, 1]);
        assert_eq!(
            CommandList::new().dispatch(op, &[&input, &packed, &bad_scales, &biases], &output),
            Err(OpError::Shape {
                operand: Operand::Input(2)
            })
        );
        let wrong_dtype = tensor(9, DType::F32, &[3, 8], &[8, 1]);
        assert_eq!(
            CommandList::new().dispatch(op, &[&input, &wrong_dtype, &scales, &biases], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::F32
            })
        );
        let mismatched_biases = tensor(10, DType::BF16, &[3, 1], &[1, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                op,
                &[&input, &packed, &scales, &mismatched_biases],
                &output
            ),
            Err(OpError::DType {
                operand: Operand::Input(3),
                dtype: DType::BF16
            })
        );
        assert_eq!(
            CommandList::new().dispatch(
                Op::QuantMatmul {
                    bits: 2,
                    group_size: 64
                },
                &[&input, &packed, &scales, &biases],
                &output
            ),
            Err(OpError::InvalidQuantization)
        );
    }

    #[test]
    fn sdpa_rejects_invalid_signatures() {
        let q = tensor(1, DType::F32, &[4, 2, 3], &[6, 3, 1]);
        let k = tensor(2, DType::F32, &[2, 3, 3], &[9, 3, 1]);
        let v = tensor(3, DType::F32, &[2, 3, 2], &[6, 2, 1]);
        let output = tensor(4, DType::F32, &[4, 2, 2], &[4, 2, 1]);
        let noncausal = Op::Sdpa {
            scale: 1.0,
            causal: false,
            q_start: 0,
        };
        assert_eq!(
            CommandList::new().dispatch(noncausal, &[&q, &k], &output),
            Err(OpError::Arity {
                expected: 3,
                actual: 2
            })
        );
        let bad_groups = tensor(5, DType::F32, &[3, 2, 3], &[6, 3, 1]);
        let bad_groups_output = tensor(6, DType::F32, &[3, 2, 2], &[4, 2, 1]);
        assert_eq!(
            CommandList::new().dispatch(noncausal, &[&bad_groups, &k, &v], &bad_groups_output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let wrong_d = tensor(7, DType::F32, &[2, 3, 4], &[12, 4, 1]);
        assert_eq!(
            CommandList::new().dispatch(noncausal, &[&q, &wrong_d, &v], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let wrong_length = tensor(8, DType::F32, &[2, 2, 2], &[4, 2, 1]);
        assert_eq!(
            CommandList::new().dispatch(noncausal, &[&q, &k, &wrong_length], &output),
            Err(OpError::Shape {
                operand: Operand::Input(2)
            })
        );
        let overflow = Op::Sdpa {
            scale: 1.0,
            causal: true,
            q_start: u32::MAX,
        };
        assert_eq!(
            CommandList::new().dispatch(overflow, &[&q, &k, &v], &output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
    }
}
