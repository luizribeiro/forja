use crate::{Affine, Op, ParamError, ParamValues};

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
    use super::TemplateOp;
    use crate::{Affine, Op, ParamError, ParamSpace};

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
}
