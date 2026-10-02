use std::{error::Error, fmt, ops::RangeInclusive};

use crate::{
    Affine, Axis, Constraint, DType, DeviceCapability, Dimension, Layout, LayoutClass, MAX_PARAMS,
    OperationValue, ParamSpace, Relation, SymbolicLayoutError, TemplateOp, TemplateTensor,
    TensorRef, TensorSlot, ValueRef, operation_value, symbolic::LinearAffine,
};

/// A reason a constraint cannot be proven over a complete replay range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstraintProofError {
    /// The constraint is false for at least one value in the declared range.
    Violated {
        /// Canonical constraint text.
        constraint: String,
        /// Range fact that caused the refusal.
        actual: String,
    },
    /// The proof engine has no sound range proof for this atom.
    Unsupported {
        /// Canonical constraint text.
        constraint: String,
    },
    /// Checked range arithmetic overflowed.
    ArithmeticOverflow {
        /// Canonical constraint text.
        constraint: String,
    },
}

impl fmt::Display for ConstraintProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Violated { constraint, actual } => {
                write!(
                    formatter,
                    "constraint '{constraint}' failed over replay range ({actual})"
                )
            }
            Self::Unsupported { constraint } => {
                write!(formatter, "constraint '{constraint}' has no range proof")
            }
            Self::ArithmeticOverflow { constraint } => {
                write!(
                    formatter,
                    "constraint '{constraint}' range arithmetic overflowed"
                )
            }
        }
    }
}

impl Error for ConstraintProofError {}

/// One inclusive parameter interval selecting a stable variant name.
struct Expression {
    offset: u64,
    coefficients: [u64; MAX_PARAMS],
    product: Option<(u64, [u64; MAX_PARAMS])>,
}

impl Expression {
    const fn constant(value: u64) -> Self {
        Self {
            offset: value,
            coefficients: [0; MAX_PARAMS],
            product: None,
        }
    }

    fn from_affine(affine: Affine) -> Self {
        let (offset, term) = affine.parts();
        let mut expression = Self::constant(u64::from(offset));
        if let Some((parameter, coefficient)) = term
            && let Some(slot) = expression.coefficients.get_mut(usize::from(parameter))
        {
            *slot = u64::from(coefficient);
        }
        expression
    }

    const fn from_linear(affine: LinearAffine) -> Self {
        Self {
            offset: affine.offset,
            coefficients: affine.coefficients,
            product: None,
        }
    }

    fn checked_product(self, other: Self, canonical: &str) -> Result<Self, ConstraintProofError> {
        if let Some(value) = self.constant_value() {
            return other.checked_scale(value, canonical);
        }
        if let Some(value) = other.constant_value() {
            return self.checked_scale(value, canonical);
        }
        if self.product.is_some() || other.product.is_some() {
            return Err(ConstraintProofError::Unsupported {
                constraint: canonical.to_owned(),
            });
        }
        Ok(Self {
            offset: self.offset,
            coefficients: self.coefficients,
            product: Some((other.offset, other.coefficients)),
        })
    }

    fn checked_scale(mut self, scale: u64, canonical: &str) -> Result<Self, ConstraintProofError> {
        self.offset = self.offset.checked_mul(scale).ok_or_else(|| {
            ConstraintProofError::ArithmeticOverflow {
                constraint: canonical.to_owned(),
            }
        })?;
        for coefficient in &mut self.coefficients {
            *coefficient = coefficient.checked_mul(scale).ok_or_else(|| {
                ConstraintProofError::ArithmeticOverflow {
                    constraint: canonical.to_owned(),
                }
            })?;
        }
        if let Some((offset, coefficients)) = &mut self.product {
            *offset = offset.checked_mul(scale).ok_or_else(|| {
                ConstraintProofError::ArithmeticOverflow {
                    constraint: canonical.to_owned(),
                }
            })?;
            for coefficient in coefficients {
                *coefficient = coefficient.checked_mul(scale).ok_or_else(|| {
                    ConstraintProofError::ArithmeticOverflow {
                        constraint: canonical.to_owned(),
                    }
                })?;
            }
        }
        Ok(self)
    }

    fn bounds(
        self,
        ranges: &[RangeInclusive<u32>],
        canonical: &str,
    ) -> Result<(u64, u64), ConstraintProofError> {
        let first = linear_bounds(self.offset, self.coefficients, ranges, canonical)?;
        if let Some((offset, coefficients)) = self.product {
            let second = linear_bounds(offset, coefficients, ranges, canonical)?;
            Ok((
                first.0.checked_mul(second.0).ok_or_else(|| {
                    ConstraintProofError::ArithmeticOverflow {
                        constraint: canonical.to_owned(),
                    }
                })?,
                first.1.checked_mul(second.1).ok_or_else(|| {
                    ConstraintProofError::ArithmeticOverflow {
                        constraint: canonical.to_owned(),
                    }
                })?,
            ))
        } else {
            Ok(first)
        }
    }

    fn constant_value(self) -> Option<u64> {
        if self.product.is_none() && self.coefficients == [0; MAX_PARAMS] {
            Some(self.offset)
        } else {
            None
        }
    }

    fn is_multiple_of(self, divisor: u64) -> bool {
        self.product.is_none()
            && self.offset.is_multiple_of(divisor)
            && self
                .coefficients
                .iter()
                .all(|coefficient| coefficient.is_multiple_of(divisor))
    }
}

fn linear_bounds(
    offset: u64,
    coefficients: [u64; MAX_PARAMS],
    ranges: &[RangeInclusive<u32>],
    canonical: &str,
) -> Result<(u64, u64), ConstraintProofError> {
    let mut minimum = offset;
    let mut maximum = offset;
    for (index, coefficient) in coefficients.into_iter().enumerate() {
        if coefficient == 0 {
            continue;
        }
        let range = ranges
            .get(index)
            .ok_or_else(|| ConstraintProofError::Unsupported {
                constraint: canonical.to_owned(),
            })?;
        minimum = coefficient
            .checked_mul(u64::from(*range.start()))
            .and_then(|term| minimum.checked_add(term))
            .ok_or_else(|| ConstraintProofError::ArithmeticOverflow {
                constraint: canonical.to_owned(),
            })?;
        maximum = coefficient
            .checked_mul(u64::from(*range.end()))
            .and_then(|term| maximum.checked_add(term))
            .ok_or_else(|| ConstraintProofError::ArithmeticOverflow {
                constraint: canonical.to_owned(),
            })?;
    }
    Ok((minimum, maximum))
}
