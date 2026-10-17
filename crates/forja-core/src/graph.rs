use std::{
    any::Any,
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    sync::Arc,
};

use crate::{
    Affine, BackendDispatchData, ByteHull, CommandList, Dispatch, Op, OpError, ParamError,
    ParamSpace, ParamValues, SymbolicLayout, SymbolicLayoutError, Tensor, TensorError,
    byte_ranges_overlap, gather_matmul_flops, gather_quant_matmul_dispatch_flops, matmul_flops,
    ops::{BufferAccess, barriers_bounded_by, barriers_for_accesses, dispatch_accesses},
    program::{BindError, Inst, PreparedEpilogue, PreparedProgram, ProgramKind},
    quant_matmul_flops, sdpa_flops,
};

/// A reason graph-template construction or instantiation failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// Parameter evaluation failed.
    Parameter(ParamError),
    /// A symbolic layout could not be instantiated.
    SymbolicLayout(SymbolicLayoutError),
    /// A tensor and symbolic layout name incompatible allocations.
    Tensor(TensorError),
    /// A tensor layout could not be rebound into planned storage.
    Layout(crate::LayoutError),
    /// Concrete operation validation failed.
    Operation(OpError),
    /// A symbolic tensor belongs to another parameter space.
    ParameterSpaceMismatch,
    /// The graph contains more dispatches than its configured limit.
    DispatchLimit,
    /// A concrete operand exceeds the configured element limit.
    TensorElementsLimit,
    /// A dispatch exceeds the configured work limit.
    WorkLimit,
    /// Static graph memory planning exceeded 64-bit address arithmetic.
    MemoryPlanningOverflow,
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid graph template: {self:?}")
    }
}

impl Error for GraphError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::SymbolicLayout(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::Layout(error) => Some(error),
            Self::Operation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ParamError> for GraphError {
    fn from(error: ParamError) -> Self {
        Self::Parameter(error)
    }
}

impl From<SymbolicLayoutError> for GraphError {
    fn from(error: SymbolicLayoutError) -> Self {
        Self::SymbolicLayout(error)
    }
}

impl From<TensorError> for GraphError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<crate::LayoutError> for GraphError {
    fn from(error: crate::LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<OpError> for GraphError {
    fn from(error: OpError) -> Self {
        Self::Operation(error)
    }
}

fn generic_gather_quant_flops(dispatch: &Dispatch) -> Option<u64> {
    let inputs = dispatch.primitive_inputs();
    let indices = if inputs.len() == 8 { 7 } else { 4 };
    inputs
        .first()
        .zip(inputs.get(1))
        .zip(inputs.get(indices))
        .and_then(|((input, packed), indices)| {
            gather_quant_matmul_dispatch_flops(
                input.layout().shape(),
                packed.layout().shape(),
                indices.layout().shape(),
                inputs.len(),
            )
        })
}

/// Resource bounds applied to graph dispatches at creation and instantiation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphLimits {
    dispatches: usize,
    tensor_elements: u64,
    work_per_dispatch: u64,
}

impl GraphLimits {
    /// Creates graph limits for dispatch count, operand size, and per-dispatch work.
    #[must_use]
    pub const fn new(dispatches: usize, tensor_elements: u64, work_per_dispatch: u64) -> Self {
        Self {
            dispatches,
            tensor_elements,
            work_per_dispatch,
        }
    }

    fn check_dispatch(&self, dispatch: &Dispatch) -> Result<(), GraphError> {
        let mut work = 0_u64;
        for tensor in dispatch.inputs().iter().chain(dispatch.outputs()) {
            work = work
                .checked_add(self.tensor_work(tensor)?)
                .ok_or(GraphError::WorkLimit)?;
        }
        let flops = match dispatch.op() {
            Op::Matmul => dispatch
                .inputs()
                .first()
                .zip(dispatch.inputs().get(1))
                .and_then(|(left, right)| {
                    matmul_flops(left.layout().shape(), right.layout().shape())
                }),
            Op::GatherMatmul => dispatch
                .inputs()
                .first()
                .zip(dispatch.inputs().get(1))
                .zip(dispatch.inputs().get(2))
                .and_then(|((input, weights), indices)| {
                    gather_matmul_flops(
                        input.layout().shape(),
                        weights.layout().shape(),
                        indices.layout().shape(),
                    )
                }),
            Op::QuantMatmul { .. } => dispatch
                .inputs()
                .first()
                .zip(dispatch.inputs().get(1))
                .and_then(|(input, packed)| {
                    quant_matmul_flops(input.layout().shape(), packed.layout().shape())
                }),
            Op::GatherQuantMatmul { .. } => generic_gather_quant_flops(dispatch),
            Op::Sdpa { .. } => dispatch
                .inputs()
                .first()
                .zip(dispatch.inputs().get(1))
                .zip(dispatch.inputs().get(2))
                .and_then(|((query, key), value)| {
                    sdpa_flops(
                        query.layout().shape(),
                        key.layout().shape(),
                        value.layout().shape(),
                    )
                }),
            _ => Some(0),
        }
        .ok_or(GraphError::WorkLimit)?;
        if work
            .checked_add(flops)
            .is_none_or(|total| total > self.work_per_dispatch)
        {
            return Err(GraphError::WorkLimit);
        }
        Ok(())
    }

    fn check_program(&self, dispatch: &Dispatch) -> Result<(), GraphError> {
        let program = dispatch
            .prepared_program()
            .ok_or(GraphError::WorkLimit)?
            .validated();
        let elements = self.tensor_work(dispatch.output())?;
        let mut work = 0_u64;
        for tensor in dispatch.inputs().iter().chain(dispatch.outputs()) {
            work = work
                .checked_add(self.tensor_work(tensor)?)
                .ok_or(GraphError::WorkLimit)?;
        }
        let reduction_passes = if program.program().kind == ProgramKind::Row {
            program
                .program()
                .insts
                .iter()
                .filter(|inst| matches!(inst, Inst::Reduce(_, _)))
                .count()
        } else {
            0
        };
        let passes = program
            .program()
            .insts
            .len()
            .checked_add(reduction_passes)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(GraphError::WorkLimit)?;
        if passes
            .checked_mul(elements)
            .and_then(|program_work| work.checked_add(program_work))
            .is_none_or(|total| total > self.work_per_dispatch)
        {
            return Err(GraphError::WorkLimit);
        }
        Ok(())
    }

    fn tensor_work(&self, tensor: &Tensor) -> Result<u64, GraphError> {
        let elements = tensor.layout().element_count();
        if elements > self.tensor_elements {
            return Err(GraphError::TensorElementsLimit);
        }
        Ok(elements)
    }
}

impl Default for GraphLimits {
    fn default() -> Self {
        Self::new(usize::MAX, u64::MAX, u64::MAX)
    }
}

/// A concrete or parameterized tensor retained by a graph template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemplateTensor {
    /// A tensor whose layout is identical for every instantiation.
    Concrete(Tensor),
    /// A tensor whose layout is instantiated from graph parameters.
    Symbolic {
        /// The allocation and access rights inherited from the base tensor.
        base: Tensor,
        /// The checked symbolic layout recipe.
        layout: SymbolicLayout,
    },
}

impl TemplateTensor {
    /// Creates a symbolic tensor while preserving the base tensor's allocation and access rights.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] when the layout was built for a different allocation length.
    pub fn symbolic(base: Tensor, layout: SymbolicLayout) -> Result<Self, TensorError> {
        if base.buffer().byte_len() != layout.base().buffer_len() {
            return Err(TensorError::BufferLengthMismatch);
        }
        Ok(Self::Symbolic { base, layout })
    }

    fn space(&self) -> Option<&ParamSpace> {
        match self {
            Self::Concrete(_) => None,
            Self::Symbolic { layout, .. } => Some(layout.space()),
        }
    }

    fn is_parameter_dependent(&self) -> bool {
        matches!(self, Self::Symbolic { .. })
    }

    fn instantiate(&self, values: &ParamValues) -> Result<Tensor, GraphError> {
        match self {
            Self::Concrete(tensor) => Ok(tensor.clone()),
            Self::Symbolic { base, layout } => Ok(Tensor::from_allocation(
                base.buffer(),
                layout.instantiate(values)?,
                base.is_writable(),
            )?),
        }
    }

    fn buffer(&self) -> crate::BufferId {
        match self {
            Self::Concrete(tensor) | Self::Symbolic { base: tensor, .. } => tensor.buffer(),
        }
    }

    fn byte_hull(&self) -> Result<ByteHull, GraphError> {
        match self {
            Self::Concrete(tensor) => Ok(ByteHull::from_layouts([tensor.layout()])),
            Self::Symbolic { layout, .. } => Ok(layout.byte_hull()?),
        }
    }

    fn access(&self, writes: bool) -> Result<BufferAccess, GraphError> {
        match self {
            Self::Concrete(tensor) => Ok(BufferAccess::new(tensor, writes)),
            Self::Symbolic { .. } => Ok(BufferAccess::hull(
                self.buffer(),
                self.byte_hull()?.byte_span(),
                writes,
            )),
        }
    }

    fn rebase(
        &self,
        placements: &HashMap<crate::BufferId, u64>,
        arena: crate::BufferId,
    ) -> Result<Self, GraphError> {
        let Some(&offset) = placements.get(&self.buffer()) else {
            return Ok(self.clone());
        };
        match self {
            Self::Concrete(tensor) => Ok(Self::Concrete(tensor.rebase(arena, offset)?)),
            Self::Symbolic { base, layout } => Ok(Self::Symbolic {
                base: base.rebase(arena, offset)?,
                layout: layout.rebase(offset, arena.byte_len())?,
            }),
        }
    }

    fn hull_access(&self, writes: bool) -> Result<BufferAccess, GraphError> {
        Ok(BufferAccess::hull(
            self.buffer(),
            self.byte_hull()?.byte_span(),
            writes,
        ))
    }
}

impl From<Tensor> for TemplateTensor {
    fn from(tensor: Tensor) -> Self {
        Self::Concrete(tensor)
    }
}

/// An operation configuration recorded in a graph template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TemplateOp {
    /// An operation whose configuration is independent of graph parameters.
    Static(Op),
    /// Grouped-query scaled dot-product attention with a parameterized query position.
    Sdpa {
        /// The score multiplier.
        scale: f32,
        /// Whether keys after each query position are masked.
        causal: bool,
        /// The absolute position of the first query.
        q_start: Affine,
    },
    /// Sampling with a parameterized absolute sequence position.
    Sample {
        /// Absolute sequence position mixed into the random counter.
        position: Affine,
    },
    /// Masked sampling with a parameterized absolute sequence position.
    SampleMasked {
        /// Absolute sequence position mixed into the random counter.
        position: Affine,
    },
}

impl TemplateOp {
    /// Creates an attention configuration whose query position may depend on one parameter.
    #[must_use]
    pub const fn sdpa(scale: f32, causal: bool, q_start: Affine) -> Self {
        Self::Sdpa {
            scale,
            causal,
            q_start,
        }
    }

    /// Creates a sampling configuration whose position may depend on one parameter.
    #[must_use]
    pub const fn sample(position: Affine) -> Self {
        Self::Sample { position }
    }

    /// Creates a masked sampling configuration whose position may depend on one parameter.
    #[must_use]
    pub const fn sample_masked(position: Affine) -> Self {
        Self::SampleMasked { position }
    }

    /// Resolves the operation to a concrete configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when an affine position cannot be evaluated.
    pub fn instantiate(self, values: &ParamValues) -> Result<Op, ParamError> {
        match self {
            Self::Static(op) => Ok(op),
            Self::Sdpa {
                scale,
                causal,
                q_start,
            } => Ok(Op::Sdpa {
                scale,
                causal,
                q_start: q_start.evaluate(values)?,
            }),
            Self::Sample { position } => Ok(Op::Sample {
                position: position.evaluate(values)?,
            }),
            Self::SampleMasked { position } => Ok(Op::SampleMasked {
                position: position.evaluate(values)?,
            }),
        }
    }

    /// Reports whether the configuration depends on graph parameters.
    #[must_use]
    pub const fn is_parameter_dependent(self) -> bool {
        match self {
            Self::Static(_) => false,
            Self::Sdpa { q_start, .. } => !q_start.is_constant(),
            Self::Sample { position } | Self::SampleMasked { position } => !position.is_constant(),
        }
    }
}

impl From<Op> for TemplateOp {
    fn from(op: Op) -> Self {
        match op {
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => Self::sdpa(scale, causal, Affine::constant(q_start)),
            Op::Sample { position } => Self::sample(Affine::constant(position)),
            Op::SampleMasked { position } => Self::sample_masked(Affine::constant(position)),
            _ => Self::Static(op),
        }
    }
}

#[derive(Clone, Debug)]
enum DynamicDispatch {
    Operation {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
    Program {
        program: Arc<PreparedProgram>,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
    Epilogue {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
        epilogue: Arc<PreparedEpilogue>,
        primitive_input_count: usize,
    },
}

impl DynamicDispatch {
    fn is_parameter_dependent(&self) -> bool {
        match self {
            Self::Operation {
                op,
                inputs,
                outputs,
            }
            | Self::Epilogue {
                op,
                inputs,
                outputs,
                ..
            } => {
                op.is_parameter_dependent()
                    || inputs.iter().any(TemplateTensor::is_parameter_dependent)
                    || outputs.iter().any(TemplateTensor::is_parameter_dependent)
            }
            Self::Program {
                inputs, outputs, ..
            } => inputs
                .iter()
                .chain(outputs)
                .any(TemplateTensor::is_parameter_dependent),
        }
    }

    fn uses_only(&self, space: &ParamSpace) -> bool {
        self.tensors()
            .all(|tensor| tensor.space().is_none_or(|candidate| candidate == space))
    }

    fn check_hull_aliasing(&self) -> Result<(), GraphError> {
        match self {
            Self::Operation {
                inputs, outputs, ..
            }
            | Self::Epilogue {
                inputs, outputs, ..
            } => {
                for (input, tensor) in inputs.iter().enumerate() {
                    for output in outputs {
                        if hulls_overlap(tensor, output)? {
                            return Err(OpError::Aliasing { input }.into());
                        }
                    }
                }
                for (first, output) in outputs.iter().enumerate() {
                    for (second, candidate) in outputs.iter().enumerate().skip(first + 1) {
                        if hulls_overlap(output, candidate)? {
                            return Err(OpError::OutputAliasing { first, second }.into());
                        }
                    }
                }
            }
            Self::Program {
                inputs, outputs, ..
            } => {
                for (input, tensor) in inputs.iter().enumerate() {
                    for (output, candidate) in outputs.iter().enumerate() {
                        if hulls_overlap(tensor, candidate)? {
                            return Err(OpError::ProgramBinding(BindError::InputOutputAliasing {
                                input,
                                output,
                            })
                            .into());
                        }
                    }
                }
                for (first, output) in outputs.iter().enumerate() {
                    for (second, candidate) in outputs.iter().enumerate().skip(first + 1) {
                        if hulls_overlap(output, candidate)? {
                            return Err(OpError::ProgramBinding(BindError::OutputAliasing {
                                first,
                                second,
                            })
                            .into());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn accesses(&self) -> Result<Vec<BufferAccess>, GraphError> {
        self.collect_accesses(TemplateTensor::access)
    }

    fn hull_accesses(&self) -> Result<Vec<BufferAccess>, GraphError> {
        self.collect_accesses(TemplateTensor::hull_access)
    }

    fn collect_accesses(
        &self,
        access: fn(&TemplateTensor, bool) -> Result<BufferAccess, GraphError>,
    ) -> Result<Vec<BufferAccess>, GraphError> {
        let (inputs, outputs) = match self {
            Self::Operation {
                inputs, outputs, ..
            }
            | Self::Epilogue {
                inputs, outputs, ..
            }
            | Self::Program {
                inputs, outputs, ..
            } => (inputs.as_slice(), outputs.as_slice()),
        };
        inputs
            .iter()
            .map(|tensor| access(tensor, false))
            .chain(outputs.iter().map(|tensor| access(tensor, true)))
            .collect()
    }

    fn tensors(&self) -> impl Iterator<Item = &TemplateTensor> {
        let (inputs, outputs) = match self {
            Self::Operation {
                inputs, outputs, ..
            }
            | Self::Epilogue {
                inputs, outputs, ..
            }
            | Self::Program {
                inputs, outputs, ..
            } => (inputs.as_slice(), outputs.as_slice()),
        };
        inputs.iter().chain(outputs)
    }

    fn instantiate(
        &self,
        values: &ParamValues,
        limits: GraphLimits,
    ) -> Result<Dispatch, GraphError> {
        match self {
            Self::Operation {
                op,
                inputs,
                outputs,
            } => {
                let inputs = instantiate_tensors(inputs, values)?;
                let outputs = instantiate_tensors(outputs, values)?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                let output_refs = outputs.iter().collect::<Vec<_>>();
                let dispatch =
                    Dispatch::new_many(op.instantiate(values)?, &input_refs, &output_refs)?;
                limits.check_dispatch(&dispatch)?;
                Ok(dispatch)
            }
            Self::Program {
                program,
                inputs,
                outputs,
            } => {
                let inputs = instantiate_tensors(inputs, values)?;
                let outputs = instantiate_tensors(outputs, values)?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                let output_refs = outputs.iter().collect::<Vec<_>>();
                let dispatch = Dispatch::kernel(program, &input_refs, &output_refs)?;
                limits.check_program(&dispatch)?;
                Ok(dispatch)
            }
            Self::Epilogue {
                op,
                inputs,
                outputs,
                epilogue,
                primitive_input_count,
            } => {
                let inputs = instantiate_tensors(inputs, values)?;
                let outputs = instantiate_tensors(outputs, values)?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                let output = outputs.first().ok_or(OpError::OutputArity {
                    expected: 1,
                    actual: 0,
                })?;
                let dispatch = Dispatch::with_epilogue(
                    op.instantiate(values)?,
                    &input_refs[..*primitive_input_count],
                    output,
                    epilogue,
                    &input_refs[*primitive_input_count..],
                )?;
                limits.check_dispatch(&dispatch)?;
                Ok(dispatch)
            }
        }
    }
}

#[derive(Clone, Debug)]
enum TemplateDispatch {
    Static(Box<Dispatch>),
    Dynamic {
        dispatch: DynamicDispatch,
        backend_data: Option<BackendDispatchData>,
    },
    Rule {
        dispatch: DynamicDispatch,
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    },
}

impl TemplateDispatch {
    fn rebase(
        &self,
        placements: &HashMap<crate::BufferId, u64>,
        arena: crate::BufferId,
    ) -> Result<Self, GraphError> {
        let tensor = |tensor: &TemplateTensor| tensor.rebase(placements, arena);
        let dynamic = |dispatch: &DynamicDispatch| -> Result<DynamicDispatch, GraphError> {
            Ok(match dispatch {
                DynamicDispatch::Operation {
                    op,
                    inputs,
                    outputs,
                } => DynamicDispatch::Operation {
                    op: *op,
                    inputs: inputs.iter().map(tensor).collect::<Result<_, _>>()?,
                    outputs: outputs.iter().map(tensor).collect::<Result<_, _>>()?,
                },
                DynamicDispatch::Program {
                    program,
                    inputs,
                    outputs,
                } => DynamicDispatch::Program {
                    program: Arc::clone(program),
                    inputs: inputs.iter().map(tensor).collect::<Result<_, _>>()?,
                    outputs: outputs.iter().map(tensor).collect::<Result<_, _>>()?,
                },
                DynamicDispatch::Epilogue {
                    op,
                    inputs,
                    outputs,
                    epilogue,
                    primitive_input_count,
                } => DynamicDispatch::Epilogue {
                    op: *op,
                    inputs: inputs.iter().map(tensor).collect::<Result<_, _>>()?,
                    outputs: outputs.iter().map(tensor).collect::<Result<_, _>>()?,
                    epilogue: Arc::clone(epilogue),
                    primitive_input_count: *primitive_input_count,
                },
            })
        };
        Ok(match self {
            Self::Static(dispatch) => {
                Self::Static(Box::new(dispatch.rebase_tensors(|tensor| {
                    placements.get(&tensor.buffer()).map_or_else(
                        || Ok(tensor.clone()),
                        |&offset| tensor.rebase(arena, offset),
                    )
                })?))
            }
            Self::Dynamic {
                dispatch,
                backend_data,
            } => Self::Dynamic {
                dispatch: dynamic(dispatch)?,
                backend_data: backend_data.clone(),
            },
            Self::Rule {
                dispatch,
                parameter,
                arms,
            } => Self::Rule {
                dispatch: dynamic(dispatch)?,
                parameter: *parameter,
                arms: arms.clone(),
            },
        })
    }
}

/// Static storage assignment for graph-private tensor allocations.
///
/// A placement only names tensors whose guest handles were gone at capture and whose complete
/// lifetime is inside the graph. Rebinding those allocations cannot expose aliasing to guest code:
/// the guest has no resource handle with which to read, write, or create another view of them.
/// Placements share bytes only when their inclusive dispatch lifetimes do not overlap and an
/// already-required graph barrier separates the two lifetimes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphMemoryPlan {
    placements: HashMap<crate::BufferId, u64>,
    arena_bytes: u64,
    transient_bytes: u64,
    peak_live_bytes: u64,
}

impl GraphMemoryPlan {
    /// Returns the storage required by one execution arena.
    #[must_use]
    pub const fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    /// Returns the sum of the original graph-private allocation lengths.
    #[must_use]
    pub const fn transient_bytes(&self) -> u64 {
        self.transient_bytes
    }

    /// Returns the largest simultaneous live set before placement constraints.
    #[must_use]
    pub const fn peak_live_bytes(&self) -> u64 {
        self.peak_live_bytes
    }

    /// Reports whether the allocation is graph-private and placed in the arena.
    #[must_use]
    pub fn contains(&self, buffer: crate::BufferId) -> bool {
        self.placements.contains_key(&buffer)
    }

    /// Returns the assigned byte offset for a graph-private allocation.
    #[must_use]
    pub fn offset(&self, buffer: crate::BufferId) -> Option<u64> {
        self.placements.get(&buffer).copied()
    }
}

/// A validated command sequence over one parameter space.
#[derive(Clone, Debug)]
pub struct GraphTemplate {
    space: ParamSpace,
    limits: GraphLimits,
    dispatches: Vec<TemplateDispatch>,
    barrier_accesses: Vec<Vec<BufferAccess>>,
    hull_barrier_accesses: Vec<Vec<BufferAccess>>,
    required_barriers: Vec<bool>,
    memory_arena: Option<crate::BufferId>,
}

/// A validated graph template with optional backend-owned preparation state.
#[derive(Clone)]
pub struct PreparedGraph {
    template: GraphTemplate,
    backend_state: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for PreparedGraph {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedGraph")
            .field("template", &self.template)
            .field("has_backend_state", &self.backend_state.is_some())
            .finish()
    }
}

impl PreparedGraph {
    /// Retains a graph template without backend-specific preparation.
    #[must_use]
    pub const fn new(template: GraphTemplate) -> Self {
        Self {
            template,
            backend_state: None,
        }
    }

    /// Retains a graph template and type-erased backend preparation state.
    #[doc(hidden)]
    #[must_use]
    pub fn with_backend_state<T>(template: GraphTemplate, state: T) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            template,
            backend_state: Some(Arc::new(state)),
        }
    }

    /// Returns the validated portable template.
    #[doc(hidden)]
    #[must_use]
    pub const fn template(&self) -> &GraphTemplate {
        &self.template
    }

    /// Returns the resource bounds retained by the portable template.
    #[doc(hidden)]
    #[must_use]
    pub const fn limits(&self) -> GraphLimits {
        self.template.limits()
    }

    /// Returns backend preparation state when its concrete type matches.
    #[doc(hidden)]
    #[must_use]
    pub fn backend_state<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.backend_state.as_ref()?.downcast_ref()
    }

    /// Checks raw replay values against this graph's parameter space.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when the count differs or a value is out of range.
    pub fn values(&self, values: Vec<u32>) -> Result<ParamValues, ParamError> {
        self.template.values(values)
    }
}

impl GraphTemplate {
    /// Creates an empty graph template governed by the supplied resource bounds.
    #[must_use]
    pub const fn new(space: ParamSpace, limits: GraphLimits) -> Self {
        Self {
            space,
            limits,
            dispatches: Vec::new(),
            barrier_accesses: Vec::new(),
            hull_barrier_accesses: Vec::new(),
            required_barriers: Vec::new(),
            memory_arena: None,
        }
    }

    /// Returns the number of recorded dispatches.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.dispatches.len()
    }

    /// Reports whether the template contains no dispatches.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }

    /// Returns the conservative barrier decisions computed from symbolic byte hulls.
    #[must_use]
    pub fn required_barriers(&self) -> &[bool] {
        &self.required_barriers
    }

    /// Returns one tensor handle for every allocation retained by this template.
    #[doc(hidden)]
    #[must_use]
    pub fn retained_tensors(&self) -> Vec<Tensor> {
        let mut seen = HashSet::new();
        self.dispatches
            .iter()
            .flat_map(|dispatch| match dispatch {
                TemplateDispatch::Static(dispatch) => dispatch
                    .inputs()
                    .iter()
                    .chain(dispatch.outputs())
                    .cloned()
                    .collect::<Vec<_>>(),
                TemplateDispatch::Dynamic { dispatch, .. }
                | TemplateDispatch::Rule { dispatch, .. } => dispatch
                    .tensors()
                    .map(|tensor| match tensor {
                        TemplateTensor::Concrete(tensor)
                        | TemplateTensor::Symbolic { base: tensor, .. } => tensor.clone(),
                    })
                    .collect(),
            })
            .filter(|tensor| seen.insert(tensor.buffer()))
            .collect()
    }

    /// Plans graph-private allocations by inclusive dispatch lifetime.
    ///
    /// `externally_reachable` must contain every allocation for which the guest still owns a
    /// tensor handle at graph creation. Such allocations are never placed in the arena.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::MemoryPlanningOverflow`] when byte accounting or placement exceeds
    /// 64-bit arithmetic.
    pub fn plan_memory(
        &self,
        externally_reachable: &HashSet<crate::BufferId>,
    ) -> Result<GraphMemoryPlan, GraphError> {
        plan_graph_memory(self, externally_reachable)
    }

    /// Rebinds graph-private allocations into one physical execution arena.
    ///
    /// # Errors
    ///
    /// Returns an operation or layout error if the supplied arena cannot hold every placement.
    pub fn bind_memory(&self, plan: &GraphMemoryPlan, arena: &Tensor) -> Result<Self, GraphError> {
        if arena.buffer().byte_len() < plan.arena_bytes || !arena.is_writable() {
            return Err(GraphError::MemoryPlanningOverflow);
        }
        let mut rebound = Self::new(self.space.clone(), self.limits);
        rebound.memory_arena = Some(arena.buffer());
        for dispatch in &self.dispatches {
            let dispatch = dispatch.rebase(&plan.placements, arena.buffer())?;
            let (accesses, hull_accesses) = match &dispatch {
                TemplateDispatch::Static(dispatch) => {
                    let accesses = dispatch_accesses(dispatch);
                    let hulls = accesses.iter().map(BufferAccess::as_hull).collect();
                    (accesses, hulls)
                }
                TemplateDispatch::Dynamic { dispatch, .. }
                | TemplateDispatch::Rule { dispatch, .. } => {
                    (dispatch.accesses()?, dispatch.hull_accesses()?)
                }
            };
            rebound.dispatches.push(dispatch);
            rebound.barrier_accesses.push(accesses);
            rebound.hull_barrier_accesses.push(hull_accesses);
        }
        rebound
            .required_barriers
            .clone_from(&self.required_barriers);
        Ok(rebound)
    }

    /// Returns the physical arena allocation used by this bound template.
    #[doc(hidden)]
    #[must_use]
    pub const fn memory_arena(&self) -> Option<crate::BufferId> {
        self.memory_arena
    }

    /// Returns the resource bounds retained by this template.
    #[doc(hidden)]
    #[must_use]
    pub const fn limits(&self) -> GraphLimits {
        self.limits
    }

    /// Returns the concrete dispatches whose tensors and operation never depend on parameters.
    #[doc(hidden)]
    pub fn static_dispatches(&self) -> impl Iterator<Item = (usize, &Dispatch)> {
        self.dispatches
            .iter()
            .enumerate()
            .filter_map(|(index, dispatch)| match dispatch {
                TemplateDispatch::Static(dispatch) => Some((index, dispatch.as_ref())),
                TemplateDispatch::Dynamic { .. } | TemplateDispatch::Rule { .. } => None,
            })
    }

    /// Returns indices whose concrete operation or tensor layout is instantiated per replay.
    #[doc(hidden)]
    pub fn parameter_dependent_dispatch_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.dispatches
            .iter()
            .enumerate()
            .filter_map(|(index, dispatch)| match dispatch {
                TemplateDispatch::Static(_) => None,
                TemplateDispatch::Dynamic { .. } | TemplateDispatch::Rule { .. } => Some(index),
            })
    }

    /// Instantiates one backend-preparation candidate for every fixed selection and rule arm.
    #[doc(hidden)]
    pub fn backend_dispatch_candidates(&self) -> Result<Vec<(usize, bool, Dispatch)>, GraphError> {
        let lower = self.space.values(
            self.space
                .ranges()
                .iter()
                .map(|range| *range.start())
                .collect(),
        )?;
        let mut candidates = Vec::new();
        for (index, candidate) in self.dispatches.iter().enumerate() {
            match candidate {
                TemplateDispatch::Static(dispatch) => {
                    candidates.push((index, false, dispatch.as_ref().clone()));
                }
                TemplateDispatch::Dynamic {
                    dispatch,
                    backend_data,
                } => {
                    let concrete = dispatch.instantiate(&lower, self.limits)?;
                    candidates.push((
                        index,
                        true,
                        backend_data
                            .clone()
                            .map_or(concrete.clone(), |data| concrete.with_backend_data(data)),
                    ));
                }
                TemplateDispatch::Rule {
                    dispatch,
                    parameter,
                    arms,
                } => {
                    let slot = usize::from(*parameter);
                    let mut values = lower.as_slice().to_vec();
                    let mut lo = *self.space.ranges()[slot].start();
                    for (hi, data) in arms {
                        if lo > *hi || !self.space.ranges()[slot].contains(&lo) {
                            return Err(GraphError::ParameterSpaceMismatch);
                        }
                        values[slot] = lo;
                        let concrete = dispatch
                            .instantiate(&self.space.values(values.clone())?, self.limits)?
                            .with_backend_data(data.clone());
                        candidates.push((index, true, concrete));
                        lo = hi.saturating_add(1);
                    }
                }
            }
        }
        Ok(candidates)
    }

    /// Checks raw replay values against this template's parameter space.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when the count differs or a value is out of range.
    pub fn values(&self, values: Vec<u32>) -> Result<ParamValues, ParamError> {
        self.space.values(values)
    }

    /// Validates and records one trusted operation at every parameter-space corner.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] for mixed parameter spaces, invalid concrete operations, or work
    /// beyond the configured bounds.
    pub fn dispatch(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
    ) -> Result<(), GraphError> {
        self.dispatch_many(op, inputs, &[output])
    }

    /// Validates and records one trusted multi-output operation at every parameter-space corner.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] for mixed parameter spaces, invalid concrete operations, or work
    /// beyond the configured bounds.
    pub fn dispatch_many(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
    ) -> Result<(), GraphError> {
        self.record(
            DynamicDispatch::Operation {
                op: op.into(),
                inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
                outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
            },
            None,
        )
    }

    /// Validates and records an operation with a fixed backend selection.
    #[doc(hidden)]
    pub fn dispatch_variant_many(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
        data: BackendDispatchData,
    ) -> Result<(), GraphError> {
        self.record(
            DynamicDispatch::Operation {
                op: op.into(),
                inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
                outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
            },
            Some(data),
        )
    }

    /// Validates and records an operation selected by replay parameter intervals.
    #[doc(hidden)]
    pub fn dispatch_variant_rule_many(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    ) -> Result<(), GraphError> {
        self.record_rule(
            DynamicDispatch::Operation {
                op: op.into(),
                inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
                outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
            },
            parameter,
            arms,
        )
    }

    /// Validates and records one prepared scalar-program dispatch at every corner.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] for mixed parameter spaces, invalid bindings or signatures, or work
    /// beyond the configured bounds.
    pub fn dispatch_kernel(
        &mut self,
        program: &Arc<PreparedProgram>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
    ) -> Result<(), GraphError> {
        self.record(
            DynamicDispatch::Program {
                program: Arc::clone(program),
                inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
                outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
            },
            None,
        )
    }

    /// Validates and records an epilogue-bearing primitive at every parameter-space corner.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] for mixed parameter spaces, invalid bindings, or excessive work.
    pub fn dispatch_epilogue(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&TemplateTensor],
    ) -> Result<(), GraphError> {
        self.record_epilogue(op, inputs, output, epilogue, extras, None)
    }

    /// Records an epilogue-bearing primitive with a fixed backend selection.
    #[doc(hidden)]
    pub fn dispatch_epilogue_variant(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&TemplateTensor],
        data: BackendDispatchData,
    ) -> Result<(), GraphError> {
        self.record_epilogue(op, inputs, output, epilogue, extras, Some(data))
    }

    /// Records an epilogue-bearing primitive selected by replay parameter intervals.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_epilogue_variant_rule(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&TemplateTensor],
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    ) -> Result<(), GraphError> {
        let dispatch = epilogue_dispatch(op, inputs, output, epilogue, extras);
        self.record_rule(dispatch, parameter, arms)
    }

    fn record_epilogue(
        &mut self,
        op: impl Into<TemplateOp>,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
        epilogue: &Arc<PreparedEpilogue>,
        extras: &[&TemplateTensor],
        backend_data: Option<BackendDispatchData>,
    ) -> Result<(), GraphError> {
        let dispatch = epilogue_dispatch(op, inputs, output, epilogue, extras);
        self.record(dispatch, backend_data)
    }

    /// Records a concrete dispatch already validated at its command-list boundary.
    #[doc(hidden)]
    pub fn record_validated(&mut self, dispatch: Dispatch) -> Result<(), GraphError> {
        // This relies on the host constructing the graph with its command-validation limits.
        if self.dispatches.len() >= self.limits.dispatches {
            return Err(GraphError::DispatchLimit);
        }
        let accesses = dispatch_accesses(&dispatch);
        let hull_accesses = accesses.iter().map(BufferAccess::as_hull).collect();
        self.dispatches
            .push(TemplateDispatch::Static(Box::new(dispatch)));
        self.record_accesses(accesses, hull_accesses);
        Ok(())
    }

    /// Instantiates parameter-dependent dispatches and fully validates their concrete forms.
    ///
    /// Static dispatches are reused without validation because they contain only concrete tensors
    /// and were fully validated when recorded.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError`] when values belong to another space or a parameter-dependent
    /// dispatch fails concrete layout, operation, program, or work validation.
    pub fn instantiate(&self, values: &ParamValues) -> Result<CommandList, GraphError> {
        if !self.space.contains_values(values) {
            return Err(GraphError::ParameterSpaceMismatch);
        }
        let mut commands = CommandList::new();
        for dispatch in &self.dispatches {
            let concrete = match dispatch {
                TemplateDispatch::Static(dispatch) => dispatch.as_ref().clone(),
                TemplateDispatch::Dynamic {
                    dispatch,
                    backend_data,
                } => {
                    let concrete = dispatch.instantiate(values, self.limits)?;
                    backend_data
                        .clone()
                        .map_or(concrete.clone(), |data| concrete.with_backend_data(data))
                }
                TemplateDispatch::Rule {
                    dispatch,
                    parameter,
                    arms,
                } => {
                    let value = values
                        .as_slice()
                        .get(usize::from(*parameter))
                        .ok_or(GraphError::ParameterSpaceMismatch)?;
                    let data = arms
                        .iter()
                        .find(|(hi, _)| value <= hi)
                        .map(|(_, data)| data.clone())
                        .ok_or(GraphError::ParameterSpaceMismatch)?;
                    dispatch
                        .instantiate(values, self.limits)?
                        .with_backend_data(data)
                }
            };
            commands.push_and_reset_validation(concrete);
        }
        commands.set_precomputed_barriers(self.required_barriers.clone());
        Ok(commands)
    }

    fn record(
        &mut self,
        dispatch: DynamicDispatch,
        backend_data: Option<BackendDispatchData>,
    ) -> Result<(), GraphError> {
        if self.dispatches.len() >= self.limits.dispatches {
            return Err(GraphError::DispatchLimit);
        }
        if !dispatch.uses_only(&self.space) {
            return Err(GraphError::ParameterSpaceMismatch);
        }
        dispatch.check_hull_aliasing()?;
        let accesses = dispatch.accesses()?;
        let hull_accesses = dispatch.hull_accesses()?;
        let mut corners = self.space.corners().into_iter();
        let first = corners
            .next()
            .ok_or(GraphError::ParameterSpaceMismatch)
            .and_then(|values| dispatch.instantiate(&values, self.limits))?;
        for values in corners {
            dispatch.instantiate(&values, self.limits)?;
        }
        if dispatch.is_parameter_dependent() {
            self.dispatches.push(TemplateDispatch::Dynamic {
                dispatch,
                backend_data,
            });
        } else {
            let first = backend_data.map_or(first.clone(), |data| first.with_backend_data(data));
            self.dispatches
                .push(TemplateDispatch::Static(Box::new(first)));
        }
        self.record_accesses(accesses, hull_accesses);
        Ok(())
    }

    fn record_rule(
        &mut self,
        dispatch: DynamicDispatch,
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    ) -> Result<(), GraphError> {
        if self.dispatches.len() >= self.limits.dispatches {
            return Err(GraphError::DispatchLimit);
        }
        if arms.is_empty()
            || !dispatch.uses_only(&self.space)
            || usize::from(parameter) >= self.space.ranges().len()
        {
            return Err(GraphError::ParameterSpaceMismatch);
        }
        dispatch.check_hull_aliasing()?;
        let accesses = dispatch.accesses()?;
        let hull_accesses = dispatch.hull_accesses()?;
        for values in self.space.corners() {
            dispatch.instantiate(&values, self.limits)?;
        }
        self.dispatches.push(TemplateDispatch::Rule {
            dispatch,
            parameter,
            arms,
        });
        self.record_accesses(accesses, hull_accesses);
        Ok(())
    }

    fn record_accesses(&mut self, accesses: Vec<BufferAccess>, hull_accesses: Vec<BufferAccess>) {
        self.barrier_accesses.push(accesses);
        self.hull_barrier_accesses.push(hull_accesses);
        let upper = barriers_for_accesses(self.hull_barrier_accesses.iter().cloned());
        self.required_barriers = barriers_bounded_by(&self.barrier_accesses, &upper);
    }
}

fn epilogue_dispatch(
    op: impl Into<TemplateOp>,
    inputs: &[&TemplateTensor],
    output: &TemplateTensor,
    epilogue: &Arc<PreparedEpilogue>,
    extras: &[&TemplateTensor],
) -> DynamicDispatch {
    let mut all_inputs = inputs
        .iter()
        .map(|tensor| (*tensor).clone())
        .collect::<Vec<_>>();
    all_inputs.extend(extras.iter().map(|tensor| (*tensor).clone()));
    DynamicDispatch::Epilogue {
        op: op.into(),
        inputs: all_inputs,
        outputs: vec![output.clone()],
        epilogue: Arc::clone(epilogue),
        primitive_input_count: inputs.len(),
    }
}

fn hulls_overlap(first: &TemplateTensor, second: &TemplateTensor) -> Result<bool, GraphError> {
    if first.buffer() != second.buffer() {
        return Ok(false);
    }
    if let (TemplateTensor::Concrete(first), TemplateTensor::Concrete(second)) = (first, second) {
        return Ok(byte_ranges_overlap(first.layout(), second.layout()));
    }
    Ok(first.byte_hull()?.overlaps_hull(&second.byte_hull()?))
}

fn instantiate_tensors(
    tensors: &[TemplateTensor],
    values: &ParamValues,
) -> Result<Vec<Tensor>, GraphError> {
    tensors
        .iter()
        .map(|tensor| tensor.instantiate(values))
        .collect()
}

#[derive(Clone, Copy)]
struct Lifetime {
    start: usize,
    end: usize,
    bytes: u64,
    begins_with_write: bool,
    consumed: bool,
}

fn plan_graph_memory(
    graph: &GraphTemplate,
    externally_reachable: &HashSet<crate::BufferId>,
) -> Result<GraphMemoryPlan, GraphError> {
    let mut lifetimes = HashMap::<crate::BufferId, Lifetime>::new();
    for (index, dispatch) in graph.dispatches.iter().enumerate() {
        let (inputs, outputs) = template_dispatch_buffers(dispatch);
        for buffer in inputs {
            lifetimes
                .entry(buffer)
                .and_modify(|lifetime| {
                    lifetime.end = index;
                    lifetime.consumed |= lifetime.begins_with_write;
                })
                .or_insert(Lifetime {
                    start: index,
                    end: index,
                    bytes: buffer.byte_len(),
                    begins_with_write: false,
                    consumed: false,
                });
        }
        for buffer in outputs {
            lifetimes
                .entry(buffer)
                .and_modify(|lifetime| {
                    lifetime.end = index;
                })
                .or_insert(Lifetime {
                    start: index,
                    end: index,
                    bytes: buffer.byte_len(),
                    begins_with_write: true,
                    consumed: false,
                });
        }
    }
    let mut candidates = lifetimes
        .into_iter()
        .filter(|(buffer, lifetime)| {
            !externally_reachable.contains(buffer)
                && lifetime.begins_with_write
                && lifetime.consumed
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(buffer, lifetime)| (lifetime.start, buffer.allocation()));

    let transient_bytes = candidates.iter().try_fold(0_u64, |sum, (_, lifetime)| {
        sum.checked_add(lifetime.bytes)
            .ok_or(GraphError::MemoryPlanningOverflow)
    })?;
    let peak_live_bytes = (0..graph.dispatches.len()).try_fold(0_u64, |peak, index| {
        let live = candidates.iter().try_fold(0_u64, |sum, (_, lifetime)| {
            if lifetime.start <= index && index <= lifetime.end {
                sum.checked_add(lifetime.bytes)
                    .ok_or(GraphError::MemoryPlanningOverflow)
            } else {
                Ok(sum)
            }
        })?;
        Ok::<_, GraphError>(peak.max(live))
    })?;

    let mut placed = Vec::<(Lifetime, std::ops::Range<u64>)>::new();
    let mut placements = HashMap::with_capacity(candidates.len());
    let mut arena_bytes = 0_u64;
    for (buffer, lifetime) in candidates {
        let mut conflicts = placed
            .iter()
            .filter(|(prior, _)| lifetimes_conflict(graph, *prior, lifetime))
            .map(|(_, range)| (range.start, range.end))
            .collect::<Vec<_>>();
        conflicts.sort_unstable();
        let mut offset = 0_u64;
        for (start, end) in conflicts {
            offset = align_up(offset, 16)?;
            if offset
                .checked_add(lifetime.bytes)
                .is_some_and(|candidate_end| candidate_end <= start)
            {
                break;
            }
            offset = offset.max(end);
        }
        offset = align_up(offset, 16)?;
        let end = offset
            .checked_add(lifetime.bytes)
            .ok_or(GraphError::MemoryPlanningOverflow)?;
        arena_bytes = arena_bytes.max(end);
        placements.insert(buffer, offset);
        placed.push((lifetime, offset..end));
    }
    arena_bytes = align_up(arena_bytes, 4)?;
    Ok(GraphMemoryPlan {
        placements,
        arena_bytes,
        transient_bytes,
        peak_live_bytes,
    })
}

fn template_dispatch_buffers(
    dispatch: &TemplateDispatch,
) -> (Vec<crate::BufferId>, Vec<crate::BufferId>) {
    match dispatch {
        TemplateDispatch::Static(dispatch) => (
            dispatch.inputs().iter().map(Tensor::buffer).collect(),
            dispatch.outputs().iter().map(Tensor::buffer).collect(),
        ),
        TemplateDispatch::Dynamic { dispatch, .. } | TemplateDispatch::Rule { dispatch, .. } => {
            match dispatch {
                DynamicDispatch::Operation {
                    inputs, outputs, ..
                }
                | DynamicDispatch::Epilogue {
                    inputs, outputs, ..
                }
                | DynamicDispatch::Program {
                    inputs, outputs, ..
                } => (
                    inputs.iter().map(TemplateTensor::buffer).collect(),
                    outputs.iter().map(TemplateTensor::buffer).collect(),
                ),
            }
        }
    }
}

fn lifetimes_conflict(graph: &GraphTemplate, first: Lifetime, second: Lifetime) -> bool {
    if first.end >= second.start {
        return true;
    }
    let Some(after_first) = first.end.checked_add(1) else {
        return true;
    };
    !(after_first..=second.start).any(|index| graph.required_barriers.get(index) == Some(&true))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GraphError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or(GraphError::MemoryPlanningOverflow)
}
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::{
        GraphError, GraphLimits, GraphTemplate, TemplateDispatch, TemplateOp, TemplateTensor,
    };
    use crate::{
        Affine, BackendDispatchData, BufferId, CommandList, DType, Dispatch, Layout, Op, OpError,
        Operand, ParamError, ParamSpace, ParamValues, SymbolicLayout, Tensor,
        ops::{BufferAccess, barriers_for_accesses},
        program::{
            BinOp, EpilogueSignature, Inst, KernelSignature, PreparedProgram, Program, ProgramKind,
            prepare_epilogue, prepared_for_test,
        },
        required_barriers,
    };

    fn tensor(buffer: u64, shape: &[u32]) -> Tensor {
        let bytes = shape
            .iter()
            .map(|&extent| u64::from(extent))
            .product::<u64>()
            * DType::F32.byte_size();
        let layout = Layout::contiguous(DType::F32, 0, shape.to_vec(), bytes).unwrap();
        Tensor::from_allocation(BufferId::new(1, buffer, bytes), layout, true).unwrap()
    }

    #[test]
    fn ordinary_sample_keeps_its_graph_identity() {
        let operation = Op::Sample { position: 4097 };
        let template = TemplateOp::from(operation);
        let space = ParamSpace::new(std::iter::once(0..=0).collect()).unwrap();

        assert_eq!(template, TemplateOp::sample(Affine::constant(4097)));
        assert_eq!(
            template
                .instantiate(&space.values(vec![0]).unwrap())
                .unwrap(),
            operation
        );
    }

    fn memory_graph(read_first_after_second_write: bool) -> (GraphTemplate, [Tensor; 7]) {
        let tensors = [
            tensor(1, &[4]),
            tensor(2, &[4]),
            tensor(3, &[4]),
            tensor(4, &[4]),
            tensor(5, &[4]),
            tensor(6, &[4]),
            tensor(7, &[4]),
        ];
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        graph
            .record_validated(Dispatch::new(Op::Copy, &[&tensors[0]], &tensors[1]).unwrap())
            .unwrap();
        graph
            .record_validated(Dispatch::new(Op::Copy, &[&tensors[1]], &tensors[2]).unwrap())
            .unwrap();
        graph
            .record_validated(Dispatch::new(Op::Copy, &[&tensors[2]], &tensors[3]).unwrap())
            .unwrap();
        if read_first_after_second_write {
            graph
                .record_validated(Dispatch::new(Op::Copy, &[&tensors[1]], &tensors[4]).unwrap())
                .unwrap();
        }
        graph
            .record_validated(Dispatch::new(Op::Copy, &[&tensors[3]], &tensors[5]).unwrap())
            .unwrap();
        (graph, tensors)
    }

    #[test]
    fn memory_plan_aliases_barrier_separated_transients() {
        let (graph, tensors) = memory_graph(false);
        let reachable = [
            tensors[0].buffer(),
            tensors[2].buffer(),
            tensors[5].buffer(),
        ]
        .into_iter()
        .collect();
        let plan = graph.plan_memory(&reachable).unwrap();

        assert_eq!(plan.offset(tensors[1].buffer()), Some(0));
        assert_eq!(plan.offset(tensors[3].buffer()), Some(0));
        assert_eq!(plan.transient_bytes(), 32);
        assert_eq!(plan.peak_live_bytes(), 16);
        assert!(plan.arena_bytes() >= plan.peak_live_bytes());
        assert!(plan.arena_bytes() <= plan.transient_bytes());

        let arena = tensor(8, &[4]);
        let rebound = graph.bind_memory(&plan, &arena).unwrap();
        assert_eq!(rebound.memory_arena(), Some(arena.buffer()));
        assert_eq!(rebound.required_barriers(), graph.required_barriers());
    }

    #[test]
    fn memory_plan_never_places_guest_reachable_allocations() {
        let (graph, tensors) = memory_graph(false);
        let reachable = tensors.iter().map(Tensor::buffer).collect();
        let plan = graph.plan_memory(&reachable).unwrap();
        assert_eq!(plan.arena_bytes(), 0);
        assert!(!plan.contains(tensors[1].buffer()));
    }

    #[test]
    fn memory_plan_keeps_a_value_live_through_its_last_read() {
        let (graph, tensors) = memory_graph(true);
        let reachable = [
            tensors[0].buffer(),
            tensors[2].buffer(),
            tensors[4].buffer(),
            tensors[5].buffer(),
        ]
        .into_iter()
        .collect::<HashSet<_>>();
        let plan = graph.plan_memory(&reachable).unwrap();
        assert_ne!(
            plan.offset(tensors[1].buffer()),
            plan.offset(tensors[3].buffer())
        );
        assert_eq!(plan.peak_live_bytes(), 32);
        assert!(plan.arena_bytes() >= plan.peak_live_bytes());
        assert!(plan.arena_bytes() <= plan.transient_bytes());
    }

    #[test]
    fn variant_rules_select_inclusive_boundary_arms() {
        let space = ParamSpace::new(std::iter::once(1..=33).collect()).unwrap();
        let left = TemplateTensor::from(tensor(1, &[1, 7]));
        let right = TemplateTensor::from(tensor(2, &[7, 33]));
        let output = TemplateTensor::from(tensor(3, &[1, 33]));
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch_variant_rule_many(
                Op::Matmul,
                &[&left, &right],
                &[&output],
                0,
                vec![
                    (7, BackendDispatchData::new("low")),
                    (33, BackendDispatchData::new("high")),
                ],
            )
            .unwrap();

        let at_seven = graph.instantiate(&space.values(vec![7]).unwrap()).unwrap();
        let at_eight = graph.instantiate(&space.values(vec![8]).unwrap()).unwrap();
        assert_eq!(
            at_seven.dispatches()[0].backend_data::<&str>(),
            Some(&"low")
        );
        assert_eq!(
            at_eight.dispatches()[0].backend_data::<&str>(),
            Some(&"high")
        );
    }

    fn symbolic_prefix(
        buffer: u64,
        shape: &[u32],
        space: ParamSpace,
        len: Affine,
    ) -> TemplateTensor {
        let base = tensor(buffer, shape);
        let layout = SymbolicLayout::new(base.layout().clone(), space)
            .slice(0, 0.into(), len, 1)
            .unwrap();
        TemplateTensor::symbolic(base, layout).unwrap()
    }

    fn symbolic_slice(
        buffer: u64,
        shape: &[u32],
        space: ParamSpace,
        axis: u8,
        start: Affine,
        len: Affine,
    ) -> TemplateTensor {
        let base = tensor(buffer, shape);
        let layout = SymbolicLayout::new(base.layout().clone(), space)
            .slice(axis, start, len, 1)
            .unwrap();
        TemplateTensor::symbolic(base, layout).unwrap()
    }

    #[test]
    fn epilogue_views_follow_symbolic_replay_shapes() {
        let space = ParamSpace::new(std::iter::once(0..=6).collect()).unwrap();
        let rows = Affine::parameter(0, 1, 1);
        let left = symbolic_prefix(30, &[7, 3], space.clone(), rows);
        let right = TemplateTensor::from(tensor(31, &[3, 5]));
        let output = symbolic_prefix(32, &[7, 5], space.clone(), rows);
        let extra = symbolic_prefix(33, &[7, 5], space.clone(), rows);
        let epilogue = prepare_epilogue(
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
            EpilogueSignature::new(1, 2, vec![DType::F32], DType::F32),
        )
        .unwrap();
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch_epilogue(Op::Matmul, &[&left, &right], &output, &epilogue, &[&extra])
            .unwrap();

        let replay = graph.instantiate(&space.values(vec![6]).unwrap()).unwrap();
        let dispatch = &replay.dispatches()[0];
        assert_eq!(dispatch.output().layout().shape(), &[7, 5]);
        assert_eq!(dispatch.inputs()[2].layout().shape(), &[7, 5]);
        assert!(dispatch.prepared_epilogue().is_some());
    }

    #[test]
    fn epilogue_variant_rules_preserve_bindings_and_select_arms() {
        let space = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let rows = Affine::parameter(0, 0, 1);
        let left = symbolic_prefix(40, &[7, 3], space.clone(), rows);
        let right = TemplateTensor::from(tensor(41, &[3, 5]));
        let output = symbolic_prefix(42, &[7, 5], space.clone(), rows);
        let extra = symbolic_prefix(43, &[7, 5], space.clone(), rows);
        let epilogue = prepare_epilogue(
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
            EpilogueSignature::new(1, 2, vec![DType::F32], DType::F32),
        )
        .unwrap();
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch_epilogue_variant_rule(
                Op::Matmul,
                &[&left, &right],
                &output,
                &epilogue,
                &[&extra],
                0,
                vec![
                    (1, BackendDispatchData::new("single")),
                    (7, BackendDispatchData::new("batch")),
                ],
            )
            .unwrap();

        for (rows, expected) in [(1, "single"), (7, "batch")] {
            let replay = graph
                .instantiate(&space.values(vec![rows]).unwrap())
                .unwrap();
            let dispatch = &replay.dispatches()[0];
            assert_eq!(dispatch.backend_data::<&str>(), Some(&expected));
            assert!(dispatch.prepared_epilogue().is_some());
            assert_eq!(dispatch.inputs()[2].layout().shape(), &[rows, 5]);
        }

        let candidates = graph.backend_dispatch_candidates().unwrap();
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().all(|(site, dynamic, dispatch)| {
            *site == 0 && *dynamic && dispatch.prepared_epilogue().is_some()
        }));
        assert_eq!(
            candidates
                .iter()
                .map(|(_, _, dispatch)| *dispatch.backend_data::<&str>().unwrap())
                .collect::<Vec<_>>(),
            ["single", "batch"]
        );
    }

    fn fresh_operation_succeeds(
        op: TemplateOp,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
        values: &ParamValues,
    ) -> bool {
        let Ok(inputs) = inputs
            .iter()
            .map(|tensor| tensor.instantiate(values))
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let Ok(output) = output.instantiate(values) else {
            return false;
        };
        let Ok(op) = op.instantiate(values) else {
            return false;
        };
        let input_refs = inputs.iter().collect::<Vec<_>>();
        CommandList::new()
            .dispatch(op, &input_refs, &output)
            .is_ok()
    }

    fn operation_replay_matches_fresh(
        graph: &GraphTemplate,
        space: &ParamSpace,
        parameters: std::ops::RangeInclusive<u32>,
        op: TemplateOp,
        inputs: &[&TemplateTensor],
        output: &TemplateTensor,
    ) -> bool {
        parameters
            .map(|parameter| space.values(vec![parameter]).unwrap())
            .all(|values| {
                graph.instantiate(&values).is_ok()
                    == fresh_operation_succeeds(op, inputs, output, &values)
            })
    }

    fn fresh_program_succeeds(
        program: &std::sync::Arc<PreparedProgram>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
        values: &ParamValues,
    ) -> bool {
        let Ok(inputs) = inputs
            .iter()
            .map(|tensor| tensor.instantiate(values))
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let Ok(outputs) = outputs
            .iter()
            .map(|tensor| tensor.instantiate(values))
            .collect::<Result<Vec<_>, _>>()
        else {
            return false;
        };
        let input_refs = inputs.iter().collect::<Vec<_>>();
        let output_refs = outputs.iter().collect::<Vec<_>>();
        CommandList::new()
            .dispatch_kernel(program, &input_refs, &output_refs)
            .is_ok()
    }

    fn program_replay_matches_fresh(
        graph: &GraphTemplate,
        space: &ParamSpace,
        parameters: std::ops::RangeInclusive<u32>,
        program: &std::sync::Arc<PreparedProgram>,
        inputs: &[&TemplateTensor],
        outputs: &[&TemplateTensor],
    ) -> bool {
        parameters
            .map(|parameter| space.values(vec![parameter]).unwrap())
            .all(|values| {
                graph.instantiate(&values).is_ok()
                    == fresh_program_succeeds(program, inputs, outputs, &values)
            })
    }

    fn symbolic_element(
        base: &Tensor,
        space: ParamSpace,
        offset: u32,
        scale: u32,
    ) -> TemplateTensor {
        let layout = SymbolicLayout::new(base.layout().clone(), space)
            .slice(0, Affine::parameter(0, offset, scale), 1.into(), 1)
            .unwrap();
        TemplateTensor::symbolic(base.clone(), layout).unwrap()
    }
    #[test]
    fn instantiates_affine_attention_positions_with_checked_arithmetic() {
        let space = ParamSpace::new(std::iter::once(0..=u32::MAX).collect()).unwrap();
        let op = TemplateOp::sdpa(0.125, true, Affine::parameter(0, 1, 1));

        assert_eq!(
            op.instantiate(&space.values(vec![7]).unwrap()),
            Ok(Op::Sdpa {
                scale: 0.125,
                causal: true,
                q_start: 8,
            })
        );
        assert_eq!(
            op.instantiate(&space.values(vec![u32::MAX]).unwrap()),
            Err(ParamError::ArithmeticOverflow)
        );
        assert!(op.is_parameter_dependent());
        assert!(!TemplateOp::from(Op::Copy).is_parameter_dependent());
    }

    #[test]
    fn symbolic_tensors_preserve_allocation_access() {
        let layout = Layout::contiguous(DType::F32, 0, vec![7], 28).unwrap();
        let base = Tensor::from_allocation(BufferId::new(1, 2, 28), layout.clone(), false).unwrap();
        let space = ParamSpace::new(std::iter::once(0..=6).collect()).unwrap();
        let symbolic = SymbolicLayout::new(layout, space);

        let tensor = TemplateTensor::symbolic(base.clone(), symbolic).unwrap();
        assert!(matches!(
            tensor,
            TemplateTensor::Symbolic { base: retained, .. } if retained == base
        ));
    }

    #[test]
    fn refuses_values_outside_the_template_space() {
        let space = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let foreign = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let graph = GraphTemplate::new(space.clone(), GraphLimits::default());

        assert!(matches!(
            space.values(vec![]),
            Err(ParamError::CountMismatch { .. })
        ));
        assert!(matches!(
            space.values(vec![8]),
            Err(ParamError::OutOfRange { .. })
        ));
        assert!(matches!(
            graph.instantiate(&foreign.values(vec![1]).unwrap()),
            Err(GraphError::ParameterSpaceMismatch)
        ));
    }

    #[test]
    fn checks_operation_and_program_work_bounds() {
        let input = tensor(1, &[4]);
        let output = tensor(2, &[4]);
        let copy = Dispatch::new(Op::Copy, &[&input], &output).unwrap();
        assert_eq!(
            GraphLimits::new(1, 4, 7).check_dispatch(&copy),
            Err(GraphError::WorkLimit)
        );
        assert_eq!(
            GraphLimits::new(1, 3, u64::MAX).check_dispatch(&copy),
            Err(GraphError::TensorElementsLimit)
        );
        assert!(GraphLimits::new(1, 4, 8).check_dispatch(&copy).is_ok());

        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepared_for_test(
            program,
            KernelSignature::new(1, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let program = Dispatch::kernel(&prepared, &[&input], &[&output]).unwrap();
        assert_eq!(
            GraphLimits::new(1, 4, 11).check_program(&program),
            Err(GraphError::WorkLimit)
        );
        assert!(GraphLimits::new(1, 4, 12).check_program(&program).is_ok());
    }

    #[test]
    fn validated_recording_matches_regular_recording() {
        let input = tensor(1, &[7]);
        let middle = tensor(2, &[7]);
        let output = tensor(3, &[7]);
        let space = ParamSpace::new(Vec::new()).unwrap();
        let limits = GraphLimits::new(2, 7, 21);
        let mut regular = GraphTemplate::new(space.clone(), limits);
        regular
            .dispatch(
                Op::Copy,
                &[&TemplateTensor::from(input.clone())],
                &TemplateTensor::from(middle.clone()),
            )
            .unwrap();
        regular
            .dispatch(
                Op::Copy,
                &[&TemplateTensor::from(middle.clone())],
                &TemplateTensor::from(output.clone()),
            )
            .unwrap();

        let mut validated = GraphTemplate::new(space, limits);
        validated
            .record_validated(Dispatch::new(Op::Copy, &[&input], &middle).unwrap())
            .unwrap();
        validated
            .record_validated(Dispatch::new(Op::Copy, &[&middle], &output).unwrap())
            .unwrap();

        assert_eq!(regular.required_barriers(), validated.required_barriers());
        let split = |graph: &GraphTemplate| {
            graph
                .dispatches
                .iter()
                .map(|dispatch| matches!(dispatch, TemplateDispatch::Static(_)))
                .collect::<Vec<_>>()
        };
        assert_eq!(split(&regular), split(&validated));
        assert_eq!(
            validated.record_validated(Dispatch::new(Op::Copy, &[&input], &middle).unwrap()),
            Err(GraphError::DispatchLimit)
        );
    }

    #[test]
    fn refuses_mixed_parameter_spaces() {
        let space = ParamSpace::new(std::iter::once(1..=3).collect()).unwrap();
        let other = ParamSpace::new(std::iter::once(1..=3).collect()).unwrap();
        let input = symbolic_prefix(1, &[3], other, Affine::parameter(0, 0, 1));
        let output = TemplateTensor::from(tensor(2, &[3]));
        let mut graph = GraphTemplate::new(space, GraphLimits::default());

        assert_eq!(
            graph.dispatch(Op::Copy, &[&input], &output),
            Err(GraphError::ParameterSpaceMismatch)
        );
    }

    #[test]
    fn corner_checks_refuse_empty_low_and_work_over_high() {
        let zero_space = ParamSpace::new(std::iter::once(0..=1).collect()).unwrap();
        let empty_input = symbolic_prefix(1, &[1], zero_space.clone(), Affine::parameter(0, 0, 1));
        let empty_output = symbolic_prefix(2, &[1], zero_space.clone(), Affine::parameter(0, 0, 1));
        let mut empty = GraphTemplate::new(zero_space, GraphLimits::default());
        assert_eq!(
            empty.dispatch(Op::Copy, &[&empty_input], &empty_output),
            Err(GraphError::Operation(OpError::EmptyOperand {
                operand: Operand::Input(0),
            }))
        );

        let work_space = ParamSpace::new(std::iter::once(0..=3).collect()).unwrap();
        let input = symbolic_prefix(3, &[4], work_space.clone(), Affine::parameter(0, 1, 1));
        let output = symbolic_prefix(4, &[4], work_space.clone(), Affine::parameter(0, 1, 1));
        let mut work = GraphTemplate::new(work_space, GraphLimits::new(1, 4, 7));
        assert_eq!(
            work.dispatch(Op::Copy, &[&input], &output),
            Err(GraphError::WorkLimit)
        );
    }

    #[test]
    fn corner_checks_refuse_attention_position_overflow_at_high() {
        let space = ParamSpace::new(std::iter::once(0..=u32::MAX).collect()).unwrap();
        let query = TemplateTensor::from(tensor(1, &[2, 1, 1]));
        let key = TemplateTensor::from(tensor(2, &[1, 1, 1]));
        let value = TemplateTensor::from(tensor(3, &[1, 1, 1]));
        let output = TemplateTensor::from(tensor(4, &[2, 1, 1]));
        let mut graph = GraphTemplate::new(space, GraphLimits::default());

        assert_eq!(
            graph.dispatch(
                TemplateOp::sdpa(1.0, false, Affine::parameter(0, 1, 1)),
                &[&query, &key, &value],
                &output,
            ),
            Err(GraphError::Parameter(ParamError::ArithmeticOverflow))
        );
    }

    #[test]
    fn replay_rechecks_attention_head_divisibility() {
        let space = ParamSpace::new(std::iter::once(0..=2).collect()).unwrap();
        let heads = Affine::parameter(0, 2, 1);
        let query = symbolic_prefix(1, &[4, 1, 1], space.clone(), heads);
        let key = TemplateTensor::from(tensor(2, &[2, 1, 1]));
        let value = TemplateTensor::from(tensor(3, &[2, 1, 1]));
        let output = symbolic_prefix(4, &[4, 1, 1], space.clone(), heads);
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch(
                TemplateOp::sdpa(1.0, false, 0.into()),
                &[&query, &key, &value],
                &output,
            )
            .unwrap();

        assert!(graph.instantiate(&space.values(vec![0]).unwrap()).is_ok());
        assert!(graph.instantiate(&space.values(vec![2]).unwrap()).is_ok());
        assert!(matches!(
            graph.instantiate(&space.values(vec![1]).unwrap()),
            Err(GraphError::Operation(OpError::Shape {
                operand: Operand::Input(1),
            }))
        ));
    }

    #[test]
    fn hulls_refuse_aliasing_that_appears_only_inside_the_box() {
        let space = ParamSpace::new(std::iter::once(0..=2).collect()).unwrap();
        let base = tensor(1, &[5]);
        let moving = SymbolicLayout::new(base.layout().clone(), space.clone())
            .slice(0, Affine::parameter(0, 0, 2), 1.into(), 1)
            .unwrap();
        let input = TemplateTensor::symbolic(base.clone(), moving).unwrap();
        let output = TemplateTensor::from(
            Tensor::from_allocation(
                base.buffer(),
                Layout::contiguous(DType::F32, 2, vec![1], 20).unwrap(),
                true,
            )
            .unwrap(),
        );
        let mut graph = GraphTemplate::new(space, GraphLimits::default());

        assert_eq!(
            graph.dispatch(Op::Copy, &[&input], &output),
            Err(GraphError::Operation(OpError::Aliasing { input: 0 }))
        );
    }

    #[test]
    fn prepared_programs_are_checked_at_corners() {
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepared_for_test(
            program.clone(),
            KernelSignature::new(1, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let wrong_rank = prepared_for_test(
            program,
            KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let space = ParamSpace::new(std::iter::once(0..=3).collect()).unwrap();
        let len = Affine::parameter(0, 1, 1);
        let input = symbolic_prefix(1, &[4], space.clone(), len);
        let output = symbolic_prefix(2, &[4], space.clone(), len);
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch_kernel(&prepared, &[&input], &[&output])
            .unwrap();

        let mut invalid = GraphTemplate::new(space, GraphLimits::default());
        assert_eq!(
            invalid.dispatch_kernel(&wrong_rank, &[&input], &[&output]),
            Err(GraphError::Operation(OpError::ProgramSignature))
        );
    }

    #[test]
    fn concrete_strided_program_outputs_use_exact_alias_checks() {
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0), (1, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepared_for_test(
            program,
            KernelSignature::new(3, vec![DType::F32], vec![DType::F32, DType::F32], 0),
        )
        .unwrap();
        let input = tensor(1, &[7, 16, 64]);
        let buffer = BufferId::new(1, 2, 57_344);
        let low = Tensor::from_allocation(
            buffer,
            Layout::new(DType::F32, 0, vec![7, 16, 64], vec![2_048, 128, 1], 57_344).unwrap(),
            true,
        )
        .unwrap();
        let high = Tensor::from_allocation(
            buffer,
            Layout::new(DType::F32, 64, vec![7, 16, 64], vec![2_048, 128, 1], 57_344).unwrap(),
            true,
        )
        .unwrap();
        let space = ParamSpace::new(Vec::new()).unwrap();
        let mut graph = GraphTemplate::new(space, GraphLimits::default());

        graph
            .dispatch_kernel(&prepared, &[&input.into()], &[&low.into(), &high.into()])
            .unwrap();
    }

    proptest! {
        #[test]
        fn replay_matches_fresh_validation_without_filtering(hi in 2_u32..=7) {
            let space = ParamSpace::new(std::iter::once(0..=hi).collect()).unwrap();
            let position = Affine::parameter(0, 0, 1);
            let cache_shape = [8, hi + 1, 128];
            let cache_write = symbolic_slice(
                3,
                &cache_shape,
                space.clone(),
                1,
                position,
                1.into(),
            );
            let copy_input = TemplateTensor::from(tensor(1, &[8, 1, 128]));
            let mut copy = GraphTemplate::new(space.clone(), GraphLimits::default());
            copy.dispatch(Op::Copy, &[&copy_input], &cache_write).unwrap();
            prop_assert!(operation_replay_matches_fresh(
                &copy,
                &space,
                0..=hi,
                Op::Copy.into(),
                &[&copy_input],
                &cache_write,
            ));

            let extent = Affine::parameter(0, 1, 1);
            let left = symbolic_prefix(4, &[hi + 1, 3], space.clone(), extent);
            let right = TemplateTensor::from(tensor(5, &[3, 5]));
            let product = symbolic_prefix(6, &[hi + 1, 5], space.clone(), extent);
            let mut matmul = GraphTemplate::new(space.clone(), GraphLimits::default());
            matmul.dispatch(Op::Matmul, &[&left, &right], &product).unwrap();
            prop_assert!(operation_replay_matches_fresh(
                &matmul,
                &space,
                0..=hi,
                Op::Matmul.into(),
                &[&left, &right],
                &product,
            ));

            let attention_space = ParamSpace::new(std::iter::once(0..=2).collect()).unwrap();
            let heads = Affine::parameter(0, 2, 1);
            let query = symbolic_prefix(7, &[4, 1, 1], attention_space.clone(), heads);
            let key = TemplateTensor::from(tensor(8, &[2, 1, 1]));
            let value = TemplateTensor::from(tensor(9, &[2, 1, 1]));
            let attention_output =
                symbolic_prefix(10, &[4, 1, 1], attention_space.clone(), heads);
            let attention_op = TemplateOp::sdpa(1.0, false, 0.into());
            let mut attention =
                GraphTemplate::new(attention_space.clone(), GraphLimits::default());
            attention
                .dispatch(
                    attention_op,
                    &[&query, &key, &value],
                    &attention_output,
                )
                .unwrap();
            prop_assert!(operation_replay_matches_fresh(
                &attention,
                &attention_space,
                0..=2,
                attention_op,
                &[&query, &key, &value],
                &attention_output,
            ));

            let program = Program {
                kind: ProgramKind::Map,
                insts: vec![Inst::Input(0)],
                outputs: vec![(0, 0)],
            }
            .validate()
            .unwrap();
            let prepared = prepared_for_test(
                program,
                KernelSignature::new(1, vec![DType::F32], vec![DType::F32], 0),
            )
            .unwrap();
            let program_input = symbolic_prefix(11, &[hi + 1], space.clone(), extent);
            let program_output = symbolic_prefix(12, &[hi + 1], space.clone(), extent);
            let mut program_graph = GraphTemplate::new(space.clone(), GraphLimits::default());
            program_graph
                .dispatch_kernel(&prepared, &[&program_input], &[&program_output])
                .unwrap();
            prop_assert!(program_replay_matches_fresh(
                &program_graph,
                &space,
                0..=hi,
                &prepared,
                &[&program_input],
                &[&program_output],
            ));

            let contiguity_space = ParamSpace::new(std::iter::once(1..=3).collect()).unwrap();
            let width = Affine::parameter(0, 0, 1);
            let noncontiguous_input = symbolic_slice(
                13,
                &[3, 3],
                contiguity_space.clone(),
                1,
                0.into(),
                width,
            );
            let noncontiguous_output = symbolic_slice(
                14,
                &[3, 3],
                contiguity_space.clone(),
                1,
                0.into(),
                width,
            );
            let row_program = Program {
                kind: ProgramKind::Row,
                insts: vec![Inst::Input(0)],
                outputs: vec![(0, 0)],
            }
            .validate()
            .unwrap();
            let prepared_row = prepared_for_test(
                row_program,
                KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
            )
            .unwrap();
            let mut contiguity =
                GraphTemplate::new(contiguity_space.clone(), GraphLimits::default());
            contiguity
                .dispatch_kernel(
                    &prepared_row,
                    &[&noncontiguous_input],
                    &[&noncontiguous_output],
                )
                .unwrap();
            let interior = contiguity_space.values(vec![2]).unwrap();
            prop_assert!(!noncontiguous_output
                .instantiate(&interior)
                .unwrap()
                .layout()
                .is_contiguous());
            prop_assert!(program_replay_matches_fresh(
                &contiguity,
                &contiguity_space,
                1..=3,
                &prepared_row,
                &[&noncontiguous_input],
                &[&noncontiguous_output],
            ));
        }
    }

    #[test]
    fn instantiated_lists_reuse_hull_barriers() {
        let space = ParamSpace::new(std::iter::once(0..=2).collect()).unwrap();
        let cache = tensor(1, &[5]);
        let moving = SymbolicLayout::new(cache.layout().clone(), space.clone())
            .slice(0, Affine::parameter(0, 0, 2), 1.into(), 1)
            .unwrap();
        let cache_write = TemplateTensor::symbolic(cache.clone(), moving).unwrap();
        let cache_read = TemplateTensor::from(
            Tensor::from_allocation(
                cache.buffer(),
                Layout::contiguous(DType::F32, 2, vec![1], 20).unwrap(),
                true,
            )
            .unwrap(),
        );
        let source = TemplateTensor::from(tensor(2, &[1]));
        let sink = TemplateTensor::from(tensor(3, &[1]));
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph.dispatch(Op::Copy, &[&source], &cache_write).unwrap();
        graph.dispatch(Op::Copy, &[&cache_read], &sink).unwrap();

        assert_eq!(graph.required_barriers(), [false, true]);
        for value in 0..=2 {
            let values = space.values(vec![value]).unwrap();
            let mut commands = graph.instantiate(&values).unwrap();
            assert_eq!(required_barriers(&commands), [false, true]);
            let sink = sink.instantiate(&values).unwrap();
            let next = tensor(4, &[1]);
            commands.dispatch(Op::Copy, &[&sink], &next).unwrap();
            assert_eq!(required_barriers(&commands), [false, value == 1, true]);
        }
    }

    proptest! {
        #[test]
        fn exact_graph_barriers_are_bounded_and_cover_every_instantiation(
            hi in 0_u32..=3,
            write_offset in 0_u32..=4,
            write_scale in 0_u32..=2,
            read_offset in 0_u32..=4,
            read_scale in 0_u32..=2,
        ) {
            let space = ParamSpace::new(std::iter::once(0..=hi).collect()).unwrap();
            let cache = tensor(1, &[16]);
            let cache_write = symbolic_element(
                &cache,
                space.clone(),
                write_offset,
                write_scale,
            );
            let cache_read = symbolic_element(
                &cache,
                space.clone(),
                read_offset,
                read_scale,
            );
            let lanes = Tensor::from_allocation(
                cache.buffer(),
                Layout::new(DType::F32, 0, vec![4, 1], vec![2, 1], 64).unwrap(),
                true,
            ).unwrap();
            let other_lane = TemplateTensor::from(Tensor::from_allocation(
                cache.buffer(),
                Layout::new(DType::F32, 1, vec![4, 1], vec![2, 1], 64).unwrap(),
                true,
            ).unwrap());
            let lanes = TemplateTensor::from(lanes);
            let lane_source = TemplateTensor::from(tensor(4, &[4, 1]));
            let lane_sink = TemplateTensor::from(tensor(5, &[4, 1]));
            let source = TemplateTensor::from(tensor(2, &[1]));
            let sink = TemplateTensor::from(tensor(3, &[1]));
            let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
            graph.dispatch(Op::Copy, &[&lane_source], &lanes).unwrap();
            graph.dispatch(Op::Copy, &[&other_lane], &lane_sink).unwrap();
            graph.dispatch(Op::Copy, &[&source], &cache_write).unwrap();
            graph.dispatch(Op::Copy, &[&cache_read], &sink).unwrap();
            let exact = graph.required_barriers().to_vec();
            let hull = barriers_for_accesses([
                vec![
                    BufferAccess::hull(lane_source.buffer(), lane_source.byte_hull().unwrap().byte_span(), false),
                    BufferAccess::hull(lanes.buffer(), lanes.byte_hull().unwrap().byte_span(), true),
                ],
                vec![
                    BufferAccess::hull(other_lane.buffer(), other_lane.byte_hull().unwrap().byte_span(), false),
                    BufferAccess::hull(lane_sink.buffer(), lane_sink.byte_hull().unwrap().byte_span(), true),
                ],
                vec![
                    BufferAccess::hull(source.buffer(), source.byte_hull().unwrap().byte_span(), false),
                    BufferAccess::hull(cache_write.buffer(), cache_write.byte_hull().unwrap().byte_span(), true),
                ],
                vec![
                    BufferAccess::hull(cache_read.buffer(), cache_read.byte_hull().unwrap().byte_span(), false),
                    BufferAccess::hull(sink.buffer(), sink.byte_hull().unwrap().byte_span(), true),
                ],
            ]);

            for (exact_has_barrier, hull_has_barrier) in exact.iter().zip(&hull) {
                prop_assert!(!exact_has_barrier || *hull_has_barrier);
            }

            for value in 0..=hi {
                let values = space.values(vec![value]).unwrap();
                let lane_source = lane_source.instantiate(&values).unwrap();
                let lanes = lanes.instantiate(&values).unwrap();
                let other_lane = other_lane.instantiate(&values).unwrap();
                let lane_sink = lane_sink.instantiate(&values).unwrap();
                let source = source.instantiate(&values).unwrap();
                let cache_write = cache_write.instantiate(&values).unwrap();
                let cache_read = cache_read.instantiate(&values).unwrap();
                let sink = sink.instantiate(&values).unwrap();
                let mut concrete = CommandList::new();
                concrete.dispatch(Op::Copy, &[&lane_source], &lanes).unwrap();
                concrete.dispatch(Op::Copy, &[&other_lane], &lane_sink).unwrap();
                concrete.dispatch(Op::Copy, &[&source], &cache_write).unwrap();
                concrete.dispatch(Op::Copy, &[&cache_read], &sink).unwrap();

                let dispatches = concrete.into_dispatches();
                let mut prior = Vec::new();
                for (index, dispatch) in dispatches.iter().enumerate() {
                    if exact[index] {
                        prior.clear();
                    }
                    let current = crate::ops::dispatch_accesses(dispatch);
                    for access in &current {
                        for candidate in &prior {
                            prop_assert!(!access.conflicts(candidate));
                        }
                    }
                    prior.extend(current);
                }
            }
        }
    }
}
