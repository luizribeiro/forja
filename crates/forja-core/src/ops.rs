use std::{
    any::Any,
    error::Error,
    fmt,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::program::{
    BindError, BoundProgram, KernelSignature, PreparedEpilogue, PreparedProgram, ProgramHash,
    bind_program,
};
use crate::{
    DType, Layout, LayoutError, QuantizedMatrix, QuantizedMatrixError, QuantizedMatrixPart,
    byte_ranges_overlap, is_injective,
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

    pub(crate) fn rebase(&self, buffer: BufferId, byte_offset: u64) -> Result<Self, LayoutError> {
        let layout = self.layout.rebase(byte_offset, buffer.byte_len())?;
        Self::from_allocation(buffer, layout, self.writable)
            .map_err(|_| LayoutError::ArithmeticOverflow)
    }
}

/// A trusted operation recorded in a command list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Executes a validated scalar program with this content identity.
    Program(ProgramHash),
    /// Copies and optionally casts one tensor.
    Copy,
    /// Normalizes rows by their root mean square and applies a weight.
    RmsNorm {
        /// The nonnegative stabilizer added before the square root.
        eps: f32,
    },
    /// Selects the greatest element of each row under IEEE total order.
    ///
    /// Equal greatest values select the highest index.
    Argmax {
        /// Maximum number of row elements reduced by one partial workgroup.
        chunk_width: u32,
    },
    /// Selects the greatest elements of each row under IEEE total order.
    ///
    /// Results are sorted descending, with lower indices first for equal values.
    TopK {
        /// Number of elements selected from each row.
        k: u32,
        /// Whether selected logits are softmax-normalized over the selected set.
        normalize: bool,
        /// Maximum number of row elements selected by one partial workgroup.
        chunk_width: u32,
    },
    /// Samples one index from each row using counter-based randomness.
    ///
    /// Non-finite logits retain their IEEE total order and bypass temperature and random-noise
    /// arithmetic.
    ///
    /// A seed reproduces tokens for the same forja build, backend, and kernel variant. Tokens are
    /// not guaranteed to match across builds, backends, or variants.
    Sample {
        /// Absolute sequence position mixed into the random counter.
        position: u32,
        /// Maximum number of row elements reduced by one partial workgroup.
        chunk_width: u32,
        /// Largest top-k count handled by the bounded fast path.
        fast_top_k: u32,
        /// Largest partial-workgroup count handled by the bounded fast path.
        fast_max_chunks: u32,
        /// Number of bounded rejection-sampling attempts.
        rejection_rounds: u32,
    },
    /// Samples one permitted index from each row using counter-based randomness.
    ///
    /// Non-finite logits retain their IEEE total order and bypass temperature and random-noise
    /// arithmetic.
    SampleMasked {
        /// Absolute sequence position mixed into the random counter.
        position: u32,
        /// Maximum number of row elements reduced by one partial workgroup.
        chunk_width: u32,
        /// Largest top-k count handled by the bounded fast path.
        fast_top_k: u32,
        /// Largest partial-workgroup count handled by the bounded fast path.
        fast_max_chunks: u32,
        /// Number of bounded rejection-sampling attempts.
        rejection_rounds: u32,
    },
    /// Applies half-split rotary position embeddings.
    Rope {
        /// The positive finite frequency base.
        theta: f32,
    },
    /// Gathers embedding rows by token id.
    Embed,
    /// Gathers and dequantizes affine-quantized embedding rows by token id.
    QuantEmbed {
        /// Number of bits in each unsigned quantized value.
        bits: u8,
        /// Number of embedding elements sharing one scale and bias.
        group_size: u32,
    },
    /// Multiplies rank-two or rank-three matrices.
    Matmul,
    /// Multiplies rows by dense expert matrices selected at run time.
    GatherMatmul,
    /// Multiplies a matrix by affine-quantized transposed weights.
    QuantMatmul {
        /// Number of bits in each unsigned quantized value.
        bits: u8,
        /// Number of input elements sharing one scale and bias.
        group_size: u32,
    },
    /// Multiplies rows by one or two affine-quantized expert projections selected at run time.
    GatherQuantMatmul {
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
    /// Rebinding a validated tensor layout to another allocation failed.
    Layout(LayoutError),
    /// Scalar programs must be recorded with [`CommandList::dispatch_kernel`].
    ProgramRequiresBinding,
    /// Tensor views do not satisfy a scalar program's signature.
    ProgramBinding(BindError),
    /// Tensor types or rank differ from a prepared program's signature.
    ProgramSignature,
    /// The primitive family or bound tensors do not match an epilogue signature.
    EpilogueSignature,
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
    /// An engine-selected kernel parameter is outside its documented safety bounds.
    InvalidKernelParameter,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid operation: {self:?}")
    }
}

impl Error for OpError {}

/// Eight candidates per lane bound the sampling kernels' thread-local selection arrays.
const MAX_SAMPLING_CHUNK_WIDTH: u32 = 2_048;
/// Twenty ranks bound the sampling kernels' threadgroup ranking arrays.
const MAX_SAMPLING_FAST_TOP_K: u32 = 20;
/// Eight candidates per thread bound a fast merge to 2,048 partial candidates.
const MAX_SAMPLING_FAST_PARTIALS: u32 = 2_048;
/// Eight attempts cap the command and argument resources recorded for one sampling dispatch.
const MAX_SAMPLING_REJECTION_ROUNDS: u32 = 8;

fn check_reduction_parameters(chunk_width: u32) -> Result<(), OpError> {
    if chunk_width == 0 {
        return Err(OpError::InvalidKernelParameter);
    }
    Ok(())
}

fn check_sampling_parameters(
    chunk_width: u32,
    fast_top_k: u32,
    fast_max_chunks: u32,
    rejection_rounds: u32,
) -> Result<(), OpError> {
    let fast_partials = fast_top_k.checked_mul(fast_max_chunks);
    if chunk_width == 0
        || chunk_width > MAX_SAMPLING_CHUNK_WIDTH
        || fast_top_k == 0
        || fast_top_k > MAX_SAMPLING_FAST_TOP_K
        || fast_max_chunks == 0
        || fast_partials.is_none_or(|count| count > MAX_SAMPLING_FAST_PARTIALS)
        || rejection_rounds == 0
        || rejection_rounds > MAX_SAMPLING_REJECTION_ROUNDS
    {
        return Err(OpError::InvalidKernelParameter);
    }
    Ok(())
}

/// A dispatch whose signature and aliasing have been validated.
#[derive(Clone)]
pub struct Dispatch {
    op: Op,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    program: Option<ProgramDispatch>,
    epilogue: Option<EpilogueDispatch>,
    backend_data: Option<BackendDispatchData>,
}

/// Type-erased backend selection retained with a validated dispatch.
#[doc(hidden)]
#[derive(Clone)]
pub struct BackendDispatchData(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for BackendDispatchData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BackendDispatchData")
    }
}

impl BackendDispatchData {
    /// Erases backend-owned dispatch data while retaining its concrete type.
    #[must_use]
    pub fn new<T: Any + Send + Sync>(data: T) -> Self {
        Self(Arc::new(data))
    }

    /// Returns the retained data when its concrete type matches.
    #[must_use]
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

impl fmt::Debug for Dispatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Dispatch")
            .field("op", &self.op)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("program", &self.program)
            .field("epilogue", &self.epilogue)
            .field("backend_data", &self.backend_data)
            .finish()
    }
}

#[derive(Clone, Debug)]
struct ProgramDispatch {
    bound: BoundProgram,
    prepared: Arc<PreparedProgram>,
}

#[derive(Clone, Debug)]
struct EpilogueDispatch {
    prepared: Arc<PreparedEpilogue>,
    primitive_input_count: usize,
}

impl Dispatch {
    pub(crate) fn rebase_tensors(
        &self,
        mut rebase: impl FnMut(&Tensor) -> Result<Tensor, LayoutError>,
    ) -> Result<Self, OpError> {
        let inputs = self
            .inputs
            .iter()
            .map(&mut rebase)
            .collect::<Result<Vec<_>, _>>()
            .map_err(OpError::Layout)?;
        let outputs = self
            .outputs
            .iter()
            .map(rebase)
            .collect::<Result<Vec<_>, _>>()
            .map_err(OpError::Layout)?;
        let input_refs = inputs.iter().collect::<Vec<_>>();
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let mut rebound = if let Some(program) = self.prepared_program() {
            Self::kernel(program, &input_refs, &output_refs)?
        } else if let Some(epilogue) = self.prepared_epilogue() {
            let primitive_count = self.primitive_input_count();
            Self::with_epilogue(
                self.op,
                &input_refs[..primitive_count],
                output_refs[0],
                epilogue,
                &input_refs[primitive_count..],
            )?
        } else {
            Self::new_many(self.op, &input_refs, &output_refs)?
        };
        if let Some(data) = self.backend_data.clone() {
            rebound.set_backend_data(data);
        }
        Ok(rebound)
    }
    pub(crate) fn new(op: Op, inputs: &[&Tensor], output: &Tensor) -> Result<Self, OpError> {
        Self::new_many(op, inputs, &[output])
    }

    pub(crate) fn new_many(
        op: Op,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<Self, OpError> {
        check_common(inputs, outputs)?;
        let expected_outputs = match op {
            Op::TopK { .. } => 2,
            _ => 1,
        };
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
            Op::RmsNorm { eps } => check_rms_norm(inputs, output, eps)?,
            Op::Argmax { chunk_width } => {
                check_reduction_parameters(chunk_width)?;
                check_argmax(inputs, output)?;
            }
            Op::TopK { k, chunk_width, .. } => {
                check_reduction_parameters(chunk_width)?;
                check_top_k(inputs, outputs, k)?;
            }
            Op::Sample {
                chunk_width,
                fast_top_k,
                fast_max_chunks,
                rejection_rounds,
                ..
            } => {
                check_sampling_parameters(
                    chunk_width,
                    fast_top_k,
                    fast_max_chunks,
                    rejection_rounds,
                )?;
                check_sample(inputs, output)?;
            }
            Op::SampleMasked {
                chunk_width,
                fast_top_k,
                fast_max_chunks,
                rejection_rounds,
                ..
            } => {
                check_sampling_parameters(
                    chunk_width,
                    fast_top_k,
                    fast_max_chunks,
                    rejection_rounds,
                )?;
                check_sample_masked(inputs, output)?;
            }
            Op::Rope { theta } => check_rope(inputs, output, theta)?,
            Op::Embed => check_embed(inputs, output)?,
            Op::QuantEmbed { bits, group_size } => {
                check_quant_embed(inputs, output, bits, group_size)?;
            }
            Op::Matmul => check_matmul(inputs, output)?,
            Op::GatherMatmul => check_gather_matmul(inputs, output)?,
            Op::QuantMatmul { bits, group_size } => {
                check_quant_matmul(inputs, output, bits, group_size)?;
            }
            Op::GatherQuantMatmul { bits, group_size } => {
                check_gather_quant_matmul(inputs, output, bits, group_size)?;
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
            epilogue: None,
            backend_data: None,
        })
    }

    pub(crate) fn with_epilogue(
        op: Op,
        inputs: &[&Tensor],
        output: &Tensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&Tensor],
    ) -> Result<Self, OpError> {
        let mut dispatch = Self::new(op, inputs, output)?;
        let signature = epilogue.signature();
        let accumulator_count = match op {
            Op::GatherQuantMatmul { .. } if inputs.len() == 8 => 2,
            _ => 1,
        };
        if signature.accumulator_count() != accumulator_count
            || usize::from(signature.logical_rank()) != output.layout().shape().len()
            || signature.output_dtype() != output.layout().dtype()
            || !matches!(
                op,
                Op::Matmul | Op::QuantMatmul { .. } | Op::GatherQuantMatmul { .. }
            )
            || extras.len() != signature.extra_dtypes().len()
            || extras
                .iter()
                .zip(signature.extra_dtypes())
                .any(|(extra, &dtype)| {
                    extra.layout().dtype() != dtype
                        || extra.layout().shape() != output.layout().shape()
                })
        {
            return Err(OpError::EpilogueSignature);
        }
        let mut all_inputs = inputs.to_vec();
        all_inputs.extend_from_slice(extras);
        check_common(&all_inputs, &[output])?;
        dispatch
            .inputs
            .extend(extras.iter().map(|tensor| (*tensor).clone()));
        dispatch.epilogue = Some(EpilogueDispatch {
            prepared: Arc::clone(epilogue),
            primitive_input_count: inputs.len(),
        });
        Ok(dispatch)
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
            epilogue: None,
            backend_data: None,
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

    /// Returns retained epilogue preparation state, when present.
    #[must_use]
    pub fn prepared_epilogue(&self) -> Option<&Arc<PreparedEpilogue>> {
        self.epilogue.as_ref().map(|epilogue| &epilogue.prepared)
    }

    /// Returns physical primitive operands without the epilogue's extra views.
    #[must_use]
    pub fn primitive_inputs(&self) -> &[Tensor] {
        &self.inputs[..self.primitive_input_count()]
    }

    fn primitive_input_count(&self) -> usize {
        self.epilogue
            .as_ref()
            .map_or(self.inputs.len(), |epilogue| epilogue.primitive_input_count)
    }

    /// Attaches backend-owned selection data to this validated dispatch.
    #[doc(hidden)]
    #[must_use]
    pub fn with_backend_data(mut self, data: BackendDispatchData) -> Self {
        self.backend_data = Some(data);
        self
    }

    /// Replaces backend-owned selection data after validation.
    #[doc(hidden)]
    pub fn set_backend_data(&mut self, data: BackendDispatchData) {
        self.backend_data = Some(data);
    }

    /// Returns backend-owned selection data when its concrete type matches.
    #[doc(hidden)]
    #[must_use]
    pub fn backend_data<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.backend_data.as_ref()?.get()
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

    /// Records a dispatch carrying a backend-owned validated selection.
    #[doc(hidden)]
    pub fn dispatch_many_with_backend_data(
        &mut self,
        op: Op,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        data: BackendDispatchData,
    ) -> Result<(), OpError> {
        let dispatch = Dispatch::new_many(op, inputs, outputs)?.with_backend_data(data);
        self.push_and_reset_validation(dispatch);
        Ok(())
    }

    /// Validates and records a primitive with a guest-authored scalar epilogue.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`] for an invalid primitive or epilogue binding.
    pub fn dispatch_epilogue(
        &mut self,
        op: Op,
        inputs: &[&Tensor],
        output: &Tensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&Tensor],
    ) -> Result<(), OpError> {
        let dispatch = Dispatch::with_epilogue(op, inputs, output, epilogue, extras)?;
        self.push_and_reset_validation(dispatch);
        Ok(())
    }

    /// Records a dispatch that was already validated through this command boundary.
    #[doc(hidden)]
    pub fn record_validated(&mut self, dispatch: Dispatch) {
        self.push_and_reset_validation(dispatch);
    }

    /// Attaches backend-owned selection data to the most recent dispatch.
    #[doc(hidden)]
    pub fn set_last_backend_data(&mut self, data: BackendDispatchData) -> bool {
        self.dispatches.last_mut().is_some_and(|dispatch| {
            dispatch.set_backend_data(data);
            true
        })
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

    /// Returns the validated dispatch sequence.
    #[doc(hidden)]
    #[must_use]
    pub fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
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
    if k == 0 || k > width {
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

fn check_sample_masked(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 3 {
        return Err(OpError::Arity {
            expected: 3,
            actual: inputs.len(),
        });
    }
    check_sample(&inputs[..2], output)?;
    let width = inputs[0]
        .layout
        .shape()
        .last()
        .copied()
        .ok_or_else(|| shape_error(Operand::Input(0)))?;
    if inputs[2].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(2),
            dtype: inputs[2].layout.dtype(),
        });
    }
    if inputs[2].layout.shape() != [width.div_ceil(32)] || !inputs[2].layout.is_contiguous() {
        return Err(shape_error(Operand::Input(2)));
    }
    Ok(())
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

fn check_quant_embed(
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
    if inputs[3].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(3),
            dtype: inputs[3].layout.dtype(),
        });
    }
    check_float(output, Operand::Output)?;
    let [vocab, packed_width]: [u32; 2] = inputs[0]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(0)))?;
    let values_per_word = 32_u32
        .checked_div(u32::from(bits))
        .filter(|_| matches!(bits, 4 | 8))
        .ok_or(OpError::InvalidQuantization)?;
    let hidden = packed_width
        .checked_mul(values_per_word)
        .ok_or(OpError::InvalidQuantization)?;
    QuantizedMatrix::new(
        vocab,
        hidden,
        bits,
        group_size,
        inputs[0].layout.clone(),
        inputs[1].layout.clone(),
        inputs[2].layout.clone(),
    )
    .map_err(|error| quantized_matrix_error(error, inputs))?;
    let [tokens]: [u32; 1] = inputs[3]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(3)))?;
    if output.layout.shape() != [tokens, hidden] {
        return Err(shape_error(Operand::Output));
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

fn check_gather_matmul(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 3 {
        return Err(OpError::Arity {
            expected: 3,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    if output.layout.dtype() != inputs[0].layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: output.layout.dtype(),
        });
    }
    if inputs[2].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(2),
            dtype: inputs[2].layout.dtype(),
        });
    }
    let [rows, inner]: [u32; 2] = inputs[0]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(0)))?;
    let [_, weight_inner, columns]: [u32; 3] = inputs[1]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(1)))?;
    if weight_inner != inner {
        return Err(shape_error(Operand::Input(1)));
    }
    let [index_rows, routes]: [u32; 2] = inputs[2]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(2)))?;
    if index_rows != rows {
        return Err(shape_error(Operand::Input(2)));
    }
    if output.layout.shape() != [rows, routes, columns] {
        return Err(shape_error(Operand::Output));
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
    .map_err(|error| quantized_matrix_error(error, inputs))?;
    if output.layout.shape() != [m, out] {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

fn check_gather_quant_matmul(
    inputs: &[&Tensor],
    output: &Tensor,
    bits: u8,
    group_size: u32,
) -> Result<(), OpError> {
    if !matches!(inputs.len(), 5 | 6 | 8) {
        return Err(OpError::Arity {
            expected: 5,
            actual: inputs.len(),
        });
    }
    if inputs.len() == 6 {
        return check_gather_quant_weighted_sum(inputs, output, bits, group_size);
    }
    let index_position = inputs.len() - 1;
    check_gather_quant_projection(
        inputs,
        output,
        bits,
        group_size,
        [0, 1, 2, 3, index_position],
    )?;
    if inputs.len() == 8 {
        check_gather_quant_projection(inputs, output, bits, group_size, [0, 4, 5, 6, 7])?;
        for (second, first) in [(4, 1), (5, 2), (6, 3)] {
            if inputs[second].layout.dtype() != inputs[first].layout.dtype() {
                return Err(OpError::DType {
                    operand: Operand::Input(second),
                    dtype: inputs[second].layout.dtype(),
                });
            }
            if inputs[second].layout.shape() != inputs[first].layout.shape() {
                return Err(shape_error(Operand::Input(second)));
            }
        }
    }
    Ok(())
}

fn check_gather_quant_weighted_sum(
    inputs: &[&Tensor],
    output: &Tensor,
    bits: u8,
    group_size: u32,
) -> Result<(), OpError> {
    for position in [0, 5] {
        check_float(inputs[position], Operand::Input(position))?;
        if inputs[position].layout.dtype() != inputs[0].layout.dtype() {
            return Err(OpError::DType {
                operand: Operand::Input(position),
                dtype: inputs[position].layout.dtype(),
            });
        }
    }
    check_float(output, Operand::Output)?;
    if output.layout.dtype() != inputs[0].layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: output.layout.dtype(),
        });
    }
    if inputs[4].layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(4),
            dtype: inputs[4].layout.dtype(),
        });
    }
    let [rows, routes, inner]: [u32; 3] = inputs[0]
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(0)))?;
    if routes == 0 {
        return Err(shape_error(Operand::Input(0)));
    }
    let (_, columns) =
        check_quantized_expert_weights(inputs, bits, group_size, [0, 1, 2, 3], inner)?;
    for position in [4, 5] {
        if inputs[position].layout.shape() != [rows, routes] {
            return Err(shape_error(Operand::Input(position)));
        }
    }
    if output.layout.shape() != [rows, columns] {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

fn check_gather_quant_projection(
    inputs: &[&Tensor],
    output: &Tensor,
    bits: u8,
    group_size: u32,
    positions: [usize; 5],
) -> Result<(), OpError> {
    let [
        input_position,
        packed_position,
        scale_position,
        bias_position,
        index_position,
    ] = positions;
    let input = inputs[input_position];
    let index_input = inputs[index_position];
    check_float(input, Operand::Input(input_position))?;
    check_float(output, Operand::Output)?;
    if output.layout.dtype() != input.layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Output,
            dtype: output.layout.dtype(),
        });
    }
    if index_input.layout.dtype() != DType::U32 {
        return Err(OpError::DType {
            operand: Operand::Input(index_position),
            dtype: index_input.layout.dtype(),
        });
    }
    let [rows, inner]: [u32; 2] = input
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(input_position)))?;
    let (_, out) = check_quantized_expert_weights(
        inputs,
        bits,
        group_size,
        [
            input_position,
            packed_position,
            scale_position,
            bias_position,
        ],
        inner,
    )?;
    let [index_rows, k]: [u32; 2] = index_input
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(index_position)))?;
    if index_rows != rows {
        return Err(shape_error(Operand::Input(index_position)));
    }
    if output.layout.shape() != [rows, k, out] {
        return Err(shape_error(Operand::Output));
    }
    Ok(())
}

fn check_quantized_expert_weights(
    inputs: &[&Tensor],
    bits: u8,
    group_size: u32,
    positions: [usize; 4],
    inner: u32,
) -> Result<(u32, u32), OpError> {
    let [
        input_position,
        packed_position,
        scale_position,
        bias_position,
    ] = positions;
    let packed_input = inputs[packed_position];
    let scale_input = inputs[scale_position];
    let bias_input = inputs[bias_position];
    let [experts, out, _]: [u32; 3] = packed_input
        .layout
        .shape()
        .try_into()
        .map_err(|_| shape_error(Operand::Input(packed_position)))?;
    for (position, part) in [
        (packed_position, packed_input),
        (scale_position, scale_input),
        (bias_position, bias_input),
    ] {
        if part.layout.shape().first() != Some(&experts) {
            return Err(shape_error(Operand::Input(position)));
        }
    }
    let packed = expert_matrix_layout(packed_input.layout())
        .map_err(|()| shape_error(Operand::Input(packed_position)))?;
    let scales = expert_matrix_layout(scale_input.layout())
        .map_err(|()| shape_error(Operand::Input(scale_position)))?;
    let biases = expert_matrix_layout(bias_input.layout())
        .map_err(|()| shape_error(Operand::Input(bias_position)))?;
    QuantizedMatrix::new(out, inner, bits, group_size, packed, scales, biases).map_err(
        |error| {
            quantized_matrix_error_at(
                error,
                inputs,
                [
                    input_position,
                    packed_position,
                    scale_position,
                    bias_position,
                ],
            )
        },
    )?;
    Ok((experts, out))
}

fn expert_matrix_layout(layout: &Layout) -> Result<Layout, ()> {
    let [_, rows, columns]: [u32; 3] = layout.shape().try_into().map_err(|_| ())?;
    Layout::new(
        layout.dtype(),
        layout.offset(),
        vec![rows, columns],
        layout.strides()[1..].to_vec(),
        layout.buffer_len(),
    )
    .map_err(|_| ())
}

fn quantized_matrix_error(error: QuantizedMatrixError, inputs: &[&Tensor]) -> OpError {
    quantized_matrix_error_at(error, inputs, [0, 1, 2, 3])
}

fn quantized_matrix_error_at(
    error: QuantizedMatrixError,
    inputs: &[&Tensor],
    positions: [usize; 4],
) -> OpError {
    match error {
        QuantizedMatrixError::UnsupportedBitWidth | QuantizedMatrixError::UnsupportedGroupSize => {
            OpError::InvalidQuantization
        }
        QuantizedMatrixError::ColumnsNotPackable | QuantizedMatrixError::ColumnsNotGrouped => {
            shape_error(Operand::Input(positions[0]))
        }
        QuantizedMatrixError::DTypeMismatch { part } => {
            let input = positions[quantized_part_input(part)];
            OpError::DType {
                operand: Operand::Input(input),
                dtype: inputs[input].layout.dtype(),
            }
        }
        QuantizedMatrixError::ShapeMismatch { part } => {
            shape_error(Operand::Input(positions[quantized_part_input(part)]))
        }
    }
}

const fn quantized_part_input(part: QuantizedMatrixPart) -> usize {
    match part {
        QuantizedMatrixPart::Packed => 1,
        QuantizedMatrixPart::Scales => 2,
        QuantizedMatrixPart::Biases => 3,
    }
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

/// Counts the floating-point operations in gathered dense matrix multiplication.
#[must_use]
pub fn gather_matmul_flops(input: &[u32], weights: &[u32], indices: &[u32]) -> Option<u64> {
    let [rows, inner]: [u32; 2] = input.try_into().ok()?;
    let [_, weight_inner, columns]: [u32; 3] = weights.try_into().ok()?;
    let [index_rows, routes]: [u32; 2] = indices.try_into().ok()?;
    (rows == index_rows && inner == weight_inner).then_some(())?;
    u64::from(rows)
        .checked_mul(u64::from(routes))?
        .checked_mul(u64::from(inner))?
        .checked_mul(u64::from(columns))?
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

/// Counts the multiply-add operations in gathered affine quantized matrix multiplication.
#[must_use]
pub fn gather_quant_matmul_flops(input: &[u32], packed: &[u32], indices: &[u32]) -> Option<u64> {
    let [rows, inner]: [u32; 2] = input.try_into().ok()?;
    let [_, columns, _]: [u32; 3] = packed.try_into().ok()?;
    let [index_rows, k]: [u32; 2] = indices.try_into().ok()?;
    (rows == index_rows).then_some(())?;
    u64::from(rows)
        .checked_mul(u64::from(k))?
        .checked_mul(u64::from(inner))?
        .checked_mul(u64::from(columns))?
        .checked_mul(2)
}

/// Counts a generic gathered quantized dispatch, including paired projections or weighted sum.
#[must_use]
pub fn gather_quant_matmul_dispatch_flops(
    input: &[u32],
    packed: &[u32],
    indices: &[u32],
    input_count: usize,
) -> Option<u64> {
    match input_count {
        5 => gather_quant_matmul_flops(input, packed, indices),
        6 => gather_quant_matmul_weighted_flops(input, packed, indices),
        8 => gather_quant_matmul_flops(input, packed, indices)?.checked_mul(2),
        _ => None,
    }
}

fn gather_quant_matmul_weighted_flops(
    input: &[u32],
    packed: &[u32],
    indices: &[u32],
) -> Option<u64> {
    let [rows, routes, inner]: [u32; 3] = input.try_into().ok()?;
    let [_, columns, _]: [u32; 3] = packed.try_into().ok()?;
    let [index_rows, index_routes]: [u32; 2] = indices.try_into().ok()?;
    (rows == index_rows && routes == index_routes).then_some(())?;
    u64::from(rows)
        .checked_mul(u64::from(routes))?
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
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    if inputs[2].layout.dtype() != inputs[1].layout.dtype() {
        return Err(OpError::DType {
            operand: Operand::Input(2),
            dtype: inputs[2].layout.dtype(),
        });
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
    use crate::program::{BinOp, EpilogueSignature, Inst, Program, ProgramKind, prepare_epilogue};

    fn argmax() -> Op {
        Op::Argmax { chunk_width: 2_048 }
    }

    fn sample(position: u32) -> Op {
        Op::Sample {
            position,
            chunk_width: 2_048,
            fast_top_k: 20,
            fast_max_chunks: 102,
            rejection_rounds: 2,
        }
    }

    fn sample_masked(position: u32) -> Op {
        Op::SampleMasked {
            position,
            chunk_width: 2_048,
            fast_top_k: 20,
            fast_max_chunks: 102,
            rejection_rounds: 2,
        }
    }

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

    fn bias_epilogue(rank: u8, dtype: DType) -> Arc<PreparedEpilogue> {
        prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Binary(BinOp::Add, 0, 1),
                ],
                outputs: vec![(0, 2)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, rank, vec![dtype], dtype),
        )
        .unwrap()
    }

    fn paired_epilogue(rank: u8, dtype: DType) -> Arc<PreparedEpilogue> {
        prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Binary(BinOp::Mul, 0, 1),
                ],
                outputs: vec![(0, 2)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(2, rank, vec![], dtype),
        )
        .unwrap()
    }

    #[test]
    fn independent_dispatches_need_no_barriers() {
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&slice(1, 0, 4)], &slice(3, 0, 4))
            .unwrap();
        commands
            .dispatch(Op::Copy, &[&slice(4, 0, 4)], &slice(6, 0, 4))
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
        let first = slice(3, 0, 4);
        let second = slice(4, 0, 4);
        let third = slice(5, 0, 4);
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&a], &first).unwrap();
        commands.dispatch(Op::Copy, &[&first], &second).unwrap();
        commands.dispatch(Op::Copy, &[&second], &third).unwrap();

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
    fn argmax_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[3, 7], &[7, 1]);
        let integer = tensor(2, DType::U32, &[3, 7], &[7, 1]);
        let output = tensor(3, DType::U32, &[3], &[1]);
        let float_output = tensor(4, DType::F32, &[3], &[1]);
        let wrong_output = tensor(5, DType::U32, &[1], &[1]);
        let scalar = tensor(6, DType::F32, &[], &[]);
        assert_eq!(
            CommandList::new().dispatch(argmax(), &[], &output),
            Err(OpError::Arity {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            CommandList::new().dispatch(argmax(), &[&integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(argmax(), &[&input], &float_output),
            Err(OpError::DType {
                operand: Operand::Output,
                dtype: DType::F32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(argmax(), &[&input], &wrong_output),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
        assert_eq!(
            CommandList::new().dispatch(argmax(), &[&scalar], &wrong_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Argmax { chunk_width: 0 }, &[&input], &output),
            Err(OpError::InvalidKernelParameter)
        );
    }

    #[test]
    fn top_k_checks_count_outputs_and_shapes() {
        let input = tensor(1, DType::F16, &[3, 7], &[7, 1]);
        let values = tensor(2, DType::F16, &[3, 7], &[7, 1]);
        let indices = tensor(3, DType::U32, &[3, 7], &[7, 1]);
        assert!(
            CommandList::new()
                .dispatch_many(
                    Op::TopK {
                        k: 7,
                        normalize: false,
                        chunk_width: 2_048,
                    },
                    &[&input],
                    &[&values, &indices],
                )
                .is_ok()
        );
        let wide_input = tensor(4, DType::F16, &[1, 65], &[65, 1]);
        let wide_values = tensor(5, DType::F16, &[1, 65], &[65, 1]);
        let wide_indices = tensor(6, DType::U32, &[1, 65], &[65, 1]);
        assert!(
            CommandList::new()
                .dispatch_many(
                    Op::TopK {
                        k: 65,
                        normalize: false,
                        chunk_width: 2_048,
                    },
                    &[&wide_input],
                    &[&wide_values, &wide_indices],
                )
                .is_ok()
        );
        assert_eq!(
            CommandList::new().dispatch_many(
                Op::TopK {
                    k: 0,
                    normalize: false,
                    chunk_width: 2_048,
                },
                &[&input],
                &[&values, &indices],
            ),
            Err(OpError::InvalidTopK)
        );
        assert_eq!(
            CommandList::new().dispatch_many(
                Op::TopK {
                    k: 8,
                    normalize: false,
                    chunk_width: 2_048,
                },
                &[&input],
                &[&values, &indices],
            ),
            Err(OpError::InvalidTopK)
        );
        assert_eq!(
            CommandList::new().dispatch_many(
                Op::TopK {
                    k: 7,
                    normalize: false,
                    chunk_width: 2_048,
                },
                &[&input],
                &[&values],
            ),
            Err(OpError::OutputArity {
                expected: 2,
                actual: 1
            })
        );
        let wrong_indices = tensor(7, DType::U32, &[3, 1], &[1, 1]);
        assert_eq!(
            CommandList::new().dispatch_many(
                Op::TopK {
                    k: 7,
                    normalize: false,
                    chunk_width: 2_048,
                },
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

        for op in [argmax(), sample(9)] {
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
                .dispatch(sample(9), &[&logits, &params], &output)
                .is_ok()
        );
        assert_eq!(
            CommandList::new().dispatch(sample(9), &[&logits, &short_params], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(sample(9), &[&logits, &params], &float_output),
            Err(OpError::DType {
                operand: Operand::Output,
                dtype: DType::F32
            })
        );
    }

    #[test]
    fn masked_sample_requires_one_contiguous_mask_bit_per_logit() {
        let logits = tensor(1, DType::F32, &[3, 33], &[33, 1]);
        let params = tensor(2, DType::U32, &[5], &[1]);
        let mask = tensor(3, DType::U32, &[2], &[1]);
        let short_mask = tensor(4, DType::U32, &[1], &[1]);
        let strided_mask = Tensor::from_allocation(
            BufferId::new(1, 5, 12),
            Layout::new(DType::U32, 0, vec![2], vec![2], 12).unwrap(),
            true,
        )
        .unwrap();
        let output = tensor(6, DType::U32, &[3], &[1]);

        assert!(
            CommandList::new()
                .dispatch(sample_masked(9), &[&logits, &params, &mask], &output)
                .is_ok()
        );
        for invalid in [&short_mask, &strided_mask] {
            assert_eq!(
                CommandList::new().dispatch(
                    sample_masked(9),
                    &[&logits, &params, invalid],
                    &output
                ),
                Err(OpError::Shape {
                    operand: Operand::Input(2)
                })
            );
        }
    }

    #[test]
    fn sampling_parameters_enforce_kernel_capacities() {
        let logits = tensor(1, DType::F32, &[1, 33], &[33, 1]);
        let params = tensor(2, DType::U32, &[5], &[1]);
        let output = tensor(3, DType::U32, &[1], &[1]);
        for op in [
            Op::Sample {
                position: 0,
                chunk_width: 2_049,
                fast_top_k: 20,
                fast_max_chunks: 102,
                rejection_rounds: 2,
            },
            Op::Sample {
                position: 0,
                chunk_width: 2_048,
                fast_top_k: 21,
                fast_max_chunks: 1,
                rejection_rounds: 2,
            },
            Op::Sample {
                position: 0,
                chunk_width: 2_048,
                fast_top_k: 20,
                fast_max_chunks: 103,
                rejection_rounds: 2,
            },
            Op::Sample {
                position: 0,
                chunk_width: 2_048,
                fast_top_k: 20,
                fast_max_chunks: 102,
                rejection_rounds: 0,
            },
            Op::Sample {
                position: 0,
                chunk_width: 2_048,
                fast_top_k: 20,
                fast_max_chunks: 102,
                rejection_rounds: 9,
            },
        ] {
            assert_eq!(
                CommandList::new().dispatch(op, &[&logits, &params], &output),
                Err(OpError::InvalidKernelParameter)
            );
        }
        assert_eq!(
            CommandList::new().dispatch(
                Op::SampleMasked {
                    position: 0,
                    chunk_width: 2_048,
                    fast_top_k: 20,
                    fast_max_chunks: 102,
                    rejection_rounds: 0,
                },
                &[&logits, &params, &tensor(4, DType::U32, &[2], &[1])],
                &output,
            ),
            Err(OpError::InvalidKernelParameter)
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
    fn quantized_embed_rejects_invalid_signatures() {
        let packed = tensor(1, DType::U32, &[33, 128], &[128, 1]);
        let scales = tensor(2, DType::BF16, &[33, 16], &[16, 1]);
        let biases = tensor(3, DType::BF16, &[33, 16], &[16, 1]);
        let ids = tensor(4, DType::U32, &[7], &[1]);
        let output = tensor(5, DType::F32, &[7, 1024], &[1024, 1]);
        let op = Op::QuantEmbed {
            bits: 4,
            group_size: 64,
        };
        CommandList::new()
            .dispatch(op, &[&packed, &scales, &biases, &ids], &output)
            .unwrap();
        let wrong_output = tensor(6, DType::F32, &[7, 1023], &[1023, 1]);
        assert_eq!(
            CommandList::new().dispatch(op, &[&packed, &scales, &biases, &ids], &wrong_output,),
            Err(OpError::Shape {
                operand: Operand::Output
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
    fn matmul_epilogue_retains_extra_views_as_inputs() {
        let left = tensor(20, DType::F32, &[7, 33], &[33, 1]);
        let right = tensor(21, DType::F32, &[33, 17], &[17, 1]);
        let output = tensor(22, DType::F32, &[7, 17], &[17, 1]);
        let bias = tensor(23, DType::F32, &[7, 17], &[0, 1]);
        let epilogue = bias_epilogue(2, DType::F32);
        let mut commands = CommandList::new();

        commands
            .dispatch_epilogue(Op::Matmul, &[&left, &right], &output, &epilogue, &[&bias])
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        assert_eq!(
            dispatch.inputs(),
            [left.clone(), right.clone(), bias.clone()]
        );
        assert_eq!(dispatch.primitive_inputs(), [left, right]);
        assert!(dispatch.prepared_epilogue().is_some());

        let source = tensor(24, DType::F32, &[7, 17], &[17, 1]);
        let bias_write = Tensor::from_allocation(
            bias.buffer(),
            Layout::new(DType::F32, 0, vec![7, 17], vec![17, 1], 476).unwrap(),
            true,
        )
        .unwrap();
        commands
            .dispatch(Op::Copy, &[&source], &bias_write)
            .unwrap();
        assert_eq!(required_barriers(&commands), [false, true]);
    }

    #[test]
    fn epilogue_binding_refuses_mismatched_views_and_aliases() {
        let left = tensor(30, DType::F32, &[7, 33], &[33, 1]);
        let right = tensor(31, DType::F32, &[33, 17], &[17, 1]);
        let output = tensor(32, DType::F32, &[7, 17], &[17, 1]);
        let epilogue = bias_epilogue(2, DType::F32);
        let wrong_shape = tensor(33, DType::F32, &[1, 17], &[0, 1]);
        assert_eq!(
            CommandList::new().dispatch_epilogue(
                Op::Matmul,
                &[&left, &right],
                &output,
                &epilogue,
                &[&wrong_shape],
            ),
            Err(OpError::EpilogueSignature)
        );

        let aliased =
            Tensor::from_allocation(output.buffer(), output.layout().clone(), true).unwrap();
        assert_eq!(
            CommandList::new().dispatch_epilogue(
                Op::Matmul,
                &[&left, &right],
                &output,
                &epilogue,
                &[&aliased],
            ),
            Err(OpError::Aliasing { input: 2 })
        );
        assert_eq!(
            CommandList::new().dispatch_epilogue(
                Op::Copy,
                &[&output],
                &tensor(34, DType::F32, &[7, 17], &[17, 1]),
                &epilogue,
                &[&output],
            ),
            Err(OpError::EpilogueSignature)
        );
    }

    #[test]
    fn gathered_matmul_checks_expert_and_index_shapes() {
        let input = tensor(1, DType::BF16, &[7, 33], &[33, 1]);
        let weights = tensor(2, DType::F16, &[5, 33, 17], &[561, 17, 1]);
        let indices = tensor(3, DType::U32, &[7, 3], &[3, 1]);
        let output = tensor(4, DType::BF16, &[7, 3, 17], &[51, 17, 1]);
        assert!(
            CommandList::new()
                .dispatch(Op::GatherMatmul, &[&input, &weights, &indices], &output)
                .is_ok()
        );
        let wrong_weights = tensor(5, DType::F16, &[5, 32, 17], &[544, 17, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::GatherMatmul,
                &[&input, &wrong_weights, &indices],
                &output,
            ),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        let wrong_indices = tensor(6, DType::U32, &[6, 3], &[3, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::GatherMatmul,
                &[&input, &weights, &wrong_indices],
                &output,
            ),
            Err(OpError::Shape {
                operand: Operand::Input(2)
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
    fn gathered_quant_matmul_checks_stacked_shapes_and_indices() {
        let input = tensor(1, DType::BF16, &[7, 64], &[64, 1]);
        let packed = tensor(2, DType::U32, &[5, 33, 8], &[264, 8, 1]);
        let scales = tensor(3, DType::F16, &[5, 33, 1], &[33, 1, 1]);
        let biases = tensor(4, DType::F16, &[5, 33, 1], &[33, 1, 1]);
        let indices = tensor(5, DType::U32, &[7, 3], &[3, 1]);
        let output = tensor(6, DType::BF16, &[7, 3, 33], &[99, 33, 1]);
        let op = Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        };
        assert!(
            CommandList::new()
                .dispatch(op, &[&input, &packed, &scales, &biases, &indices], &output,)
                .is_ok()
        );
        assert!(
            CommandList::new()
                .dispatch_epilogue(
                    op,
                    &[
                        &input, &packed, &scales, &biases, &packed, &scales, &biases, &indices,
                    ],
                    &output,
                    &paired_epilogue(3, DType::BF16),
                    &[],
                )
                .is_ok()
        );
        assert_eq!(
            CommandList::new().dispatch(
                op,
                &[
                    &input, &packed, &scales, &biases, &packed, &scales, &biases, &packed, &scales,
                    &biases, &indices,
                ],
                &output,
            ),
            Err(OpError::Arity {
                expected: 5,
                actual: 11,
            })
        );
        let wrong_indices = tensor(7, DType::U32, &[6, 3], &[3, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                op,
                &[&input, &packed, &scales, &biases, &wrong_indices],
                &output,
            ),
            Err(OpError::Shape {
                operand: Operand::Input(4)
            })
        );
        let wrong_experts = tensor(8, DType::F16, &[4, 33, 1], &[33, 1, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                op,
                &[&input, &packed, &wrong_experts, &biases, &indices],
                &output,
            ),
            Err(OpError::Shape {
                operand: Operand::Input(2)
            })
        );
        let fused = Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        };
        assert!(
            CommandList::new()
                .dispatch(
                    fused,
                    &[
                        &input, &packed, &scales, &biases, &packed, &scales, &biases, &indices,
                    ],
                    &output,
                )
                .is_ok()
        );
        let wrong_up = tensor(9, DType::U32, &[5, 32, 8], &[256, 8, 1]);
        assert_eq!(
            CommandList::new().dispatch(
                fused,
                &[
                    &input, &packed, &scales, &biases, &wrong_up, &scales, &biases, &indices,
                ],
                &output,
            ),
            Err(OpError::Shape {
                operand: Operand::Input(5)
            })
        );
    }

    #[test]
    fn gathered_quant_work_counts_generic_dispatch_modes() {
        let packed = [5, 33, 8];
        let indices = [7, 3];

        assert_eq!(
            gather_quant_matmul_dispatch_flops(&[7, 64], &packed, &indices, 5),
            Some(88_704)
        );
        assert_eq!(
            gather_quant_matmul_dispatch_flops(&[7, 64], &packed, &indices, 8),
            Some(177_408)
        );
        assert_eq!(
            gather_quant_matmul_dispatch_flops(&[7, 3, 64], &packed, &indices, 6),
            Some(88_704)
        );
        assert_eq!(
            gather_quant_matmul_dispatch_flops(&[7, 64], &packed, &indices, 7),
            None
        );
    }

    #[test]
    fn gathered_quant_matmul_accepts_weighted_sum_with_residual_epilogue() {
        let routed = tensor(1, DType::BF16, &[7, 3, 64], &[192, 64, 1]);
        let packed = tensor(2, DType::U32, &[5, 33, 8], &[264, 8, 1]);
        let scales = tensor(3, DType::F16, &[5, 33, 1], &[33, 1, 1]);
        let biases = tensor(4, DType::F16, &[5, 33, 1], &[33, 1, 1]);
        let indices = tensor(5, DType::U32, &[7, 3], &[3, 1]);
        let weights = tensor(6, DType::BF16, &[7, 3], &[3, 1]);
        let output = tensor(7, DType::BF16, &[7, 33], &[33, 1]);
        let residual = tensor(8, DType::BF16, &[7, 33], &[33, 1]);
        assert!(
            CommandList::new()
                .dispatch_epilogue(
                    Op::GatherQuantMatmul {
                        bits: 4,
                        group_size: 64,
                    },
                    &[&routed, &packed, &scales, &biases, &indices, &weights],
                    &output,
                    &bias_epilogue(2, DType::BF16),
                    &[&residual],
                )
                .is_ok()
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

    #[test]
    fn sdpa_rejects_mismatched_or_unsupported_cache_dtypes() {
        let q = tensor(1, DType::F32, &[4, 2, 3], &[6, 3, 1]);
        let output = tensor(2, DType::F32, &[4, 2, 2], &[4, 2, 1]);
        let op = Op::Sdpa {
            scale: 1.0,
            causal: false,
            q_start: 0,
        };
        for (key_dtype, value_dtype, expected) in [
            (
                DType::F16,
                DType::BF16,
                OpError::DType {
                    operand: Operand::Input(2),
                    dtype: DType::BF16,
                },
            ),
            (
                DType::F32,
                DType::F16,
                OpError::DType {
                    operand: Operand::Input(2),
                    dtype: DType::F16,
                },
            ),
            (
                DType::U32,
                DType::U32,
                OpError::DType {
                    operand: Operand::Input(1),
                    dtype: DType::U32,
                },
            ),
        ] {
            let key = tensor(3, key_dtype, &[2, 3, 3], &[9, 3, 1]);
            let value = tensor(4, value_dtype, &[2, 3, 2], &[6, 2, 1]);
            assert_eq!(
                CommandList::new().dispatch(op, &[&q, &key, &value], &output),
                Err(expected)
            );
        }
    }
}
