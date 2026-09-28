use std::{error::Error, fmt};

use crate::{
    Affine, Dispatch, Op, OpError, ParamError, ParamValues, SymbolicLayout, SymbolicLayoutError,
    Tensor, TensorError,
    program::{Inst, ProgramKind},
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    use super::{GraphError, GraphLimits, TemplateOp, TemplateTensor};
    use crate::{
        Affine, BufferId, DType, Dispatch, Layout, Op, ParamError, ParamSpace, SymbolicLayout,
        Tensor,
        program::{Inst, KernelSignature, Program, ProgramKind, prepared_for_test},
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
}
