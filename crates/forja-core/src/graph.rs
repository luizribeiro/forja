use std::{error::Error, fmt};

use crate::{
    Affine, Op, OpError, ParamError, ParamValues, SymbolicLayout, SymbolicLayoutError, Tensor,
    TensorError,
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

#[cfg(test)]
mod tests {
    use super::{TemplateOp, TemplateTensor};
    use crate::{
        Affine, BufferId, DType, Layout, Op, ParamError, ParamSpace, SymbolicLayout, Tensor,
    };

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
}
