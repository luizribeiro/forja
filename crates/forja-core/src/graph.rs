use std::{error::Error, fmt, sync::Arc};

use crate::{
    Affine, ByteHull, CommandList, Dispatch, Op, OpError, ParamError, ParamSpace, ParamValues,
    SymbolicLayout, SymbolicLayoutError, Tensor, TensorError, byte_ranges_overlap,
    ops::{BufferAccess, barriers_for_accesses},
    program::{BindError, Inst, PreparedProgram, ProgramKind},
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

impl From<OpError> for GraphError {
    fn from(error: OpError) -> Self {
        Self::Operation(error)
    }
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
        let mut work = self.tensor_work(dispatch.output())?;
        for input in dispatch.inputs() {
            work = work
                .checked_add(self.tensor_work(input)?)
                .ok_or(GraphError::WorkLimit)?;
        }
        let flops = match dispatch.op() {
            Op::Matmul => matmul_flops(dispatch.inputs()),
            Op::Sdpa { .. } => sdpa_flops(dispatch.inputs()),
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
        Ok(BufferAccess {
            buffer: self.buffer(),
            bytes: self.byte_hull()?.byte_span(),
            writes,
        })
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
        }
    }

    /// Reports whether the configuration depends on graph parameters.
    #[must_use]
    pub const fn is_parameter_dependent(self) -> bool {
        matches!(self, Self::Sdpa { q_start, .. } if !q_start.is_constant())
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
            _ => Self::Static(op),
        }
    }
}

#[derive(Clone, Debug)]
enum DynamicDispatch {
    Operation {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        output: Box<TemplateTensor>,
    },
    Program {
        program: Arc<PreparedProgram>,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
}

impl DynamicDispatch {
    fn is_parameter_dependent(&self) -> bool {
        match self {
            Self::Operation { op, inputs, output } => {
                op.is_parameter_dependent()
                    || inputs.iter().any(TemplateTensor::is_parameter_dependent)
                    || output.is_parameter_dependent()
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
            Self::Operation { inputs, output, .. } => {
                for (input, tensor) in inputs.iter().enumerate() {
                    if hulls_overlap(tensor, output)? {
                        return Err(OpError::Aliasing { input }.into());
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
        let (inputs, outputs) = match self {
            Self::Operation { inputs, output, .. } => {
                (inputs.as_slice(), std::slice::from_ref(output.as_ref()))
            }
            Self::Program {
                inputs, outputs, ..
            } => (inputs.as_slice(), outputs.as_slice()),
        };
        inputs
            .iter()
            .map(|tensor| tensor.access(false))
            .chain(outputs.iter().map(|tensor| tensor.access(true)))
            .collect()
    }

    fn tensors(&self) -> impl Iterator<Item = &TemplateTensor> {
        let (inputs, outputs) = match self {
            Self::Operation { inputs, output, .. } => {
                (inputs.as_slice(), std::slice::from_ref(output.as_ref()))
            }
            Self::Program {
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
            Self::Operation { op, inputs, output } => {
                let inputs = instantiate_tensors(inputs, values)?;
                let output = output.instantiate(values)?;
                let input_refs = inputs.iter().collect::<Vec<_>>();
                let dispatch = Dispatch::new(op.instantiate(values)?, &input_refs, &output)?;
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
        }
    }
}

#[derive(Clone, Debug)]
enum TemplateDispatch {
    Static(Box<Dispatch>),
    Dynamic(DynamicDispatch),
}

/// A validated command sequence over one parameter space.
#[derive(Clone, Debug)]
pub struct GraphTemplate {
    space: ParamSpace,
    limits: GraphLimits,
    dispatches: Vec<TemplateDispatch>,
    barrier_accesses: Vec<Vec<BufferAccess>>,
    required_barriers: Vec<bool>,
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
            required_barriers: Vec::new(),
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
        self.record(DynamicDispatch::Operation {
            op: op.into(),
            inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
            output: Box::new(output.clone()),
        })
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
        self.record(DynamicDispatch::Program {
            program: Arc::clone(program),
            inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
            outputs: outputs.iter().map(|tensor| (*tensor).clone()).collect(),
        })
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
                TemplateDispatch::Dynamic(dispatch) => dispatch.instantiate(values, self.limits)?,
            };
            commands.push_and_reset_validation(concrete);
        }
        commands.set_precomputed_barriers(self.required_barriers.clone());
        Ok(commands)
    }

    fn record(&mut self, dispatch: DynamicDispatch) -> Result<(), GraphError> {
        if self.dispatches.len() >= self.limits.dispatches {
            return Err(GraphError::DispatchLimit);
        }
        if !dispatch.uses_only(&self.space) {
            return Err(GraphError::ParameterSpaceMismatch);
        }
        dispatch.check_hull_aliasing()?;
        let accesses = dispatch.accesses()?;
        let mut corners = self.space.corners().into_iter();
        let first = corners
            .next()
            .ok_or(GraphError::ParameterSpaceMismatch)
            .and_then(|values| dispatch.instantiate(&values, self.limits))?;
        for values in corners {
            dispatch.instantiate(&values, self.limits)?;
        }
        if dispatch.is_parameter_dependent() {
            self.dispatches.push(TemplateDispatch::Dynamic(dispatch));
        } else {
            self.dispatches
                .push(TemplateDispatch::Static(Box::new(first)));
        }
        self.barrier_accesses.push(accesses);
        self.required_barriers = barriers_for_accesses(self.barrier_accesses.iter().cloned());
        Ok(())
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
fn matmul_flops(inputs: &[Tensor]) -> Option<u64> {
    let left = inputs.first()?.layout().shape();
    let right = inputs.get(1)?.layout().shape();
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

fn sdpa_flops(inputs: &[Tensor]) -> Option<u64> {
    let query = inputs.first()?.layout().shape();
    let key = inputs.get(1)?.layout().shape();
    let value = inputs.get(2)?.layout().shape();
    u64::from(*query.first()?)
        .checked_mul(u64::from(*query.get(1)?))?
        .checked_mul(u64::from(*key.get(1)?))?
        .checked_mul(u64::from(*query.get(2)?).checked_add(u64::from(*value.get(2)?))?)?
        .checked_mul(2)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{GraphError, GraphLimits, GraphTemplate, TemplateOp, TemplateTensor};
    use crate::{
        Affine, BufferId, CommandList, DType, Dispatch, Layout, Op, OpError, Operand, ParamError,
        ParamSpace, ParamValues, SymbolicLayout, Tensor,
        program::{
            Inst, KernelSignature, PreparedProgram, Program, ProgramKind, prepared_for_test,
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

            let add_input = TemplateTensor::from(tensor(2, &[8, 1, 128]));
            let mut add = GraphTemplate::new(space.clone(), GraphLimits::default());
            add.dispatch(Op::Add, &[&copy_input, &add_input], &cache_write)
                .unwrap();
            prop_assert!(operation_replay_matches_fresh(
                &add,
                &space,
                0..=hi,
                Op::Add.into(),
                &[&copy_input, &add_input],
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
        fn hull_barriers_cover_every_concrete_instantiation(
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
            let source = TemplateTensor::from(tensor(2, &[1]));
            let sink = TemplateTensor::from(tensor(3, &[1]));
            let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
            graph.dispatch(Op::Copy, &[&source], &cache_write).unwrap();
            graph.dispatch(Op::Copy, &[&cache_read], &sink).unwrap();
            let hull = graph.required_barriers().to_vec();

            for value in 0..=hi {
                let values = space.values(vec![value]).unwrap();
                let source = source.instantiate(&values).unwrap();
                let cache_write = cache_write.instantiate(&values).unwrap();
                let cache_read = cache_read.instantiate(&values).unwrap();
                let sink = sink.instantiate(&values).unwrap();
                let mut concrete = CommandList::new();
                concrete.dispatch(Op::Copy, &[&source], &cache_write).unwrap();
                concrete.dispatch(Op::Copy, &[&cache_read], &sink).unwrap();

                for (hull_has_barrier, concrete_has_barrier) in
                    hull.iter().zip(required_barriers(&concrete))
                {
                    prop_assert!(*hull_has_barrier || !concrete_has_barrier);
                }
            }
        }
    }
}
