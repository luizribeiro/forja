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
/// Proves constraints over the complete declared parameter box.
///
/// # Errors
///
/// Returns [`ConstraintProofError`] for the first false, overflowing, or unsupported atom.
pub fn prove_constraints(
    space: &ParamSpace,
    operation: TemplateOp,
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    capabilities: &[DeviceCapability],
    constraints: &[Constraint],
) -> Result<(), ConstraintProofError> {
    prove_with_ranges(
        operation,
        inputs,
        outputs,
        capabilities,
        constraints,
        space.ranges(),
    )
}

fn prove_with_ranges(
    operation: TemplateOp,
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    capabilities: &[DeviceCapability],
    constraints: &[Constraint],
    ranges: &[RangeInclusive<u32>],
) -> Result<(), ConstraintProofError> {
    for constraint in constraints {
        prove_atom(operation, inputs, outputs, capabilities, constraint, ranges)?;
    }
    Ok(())
}

fn prove_atom(
    operation: TemplateOp,
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    capabilities: &[DeviceCapability],
    constraint: &Constraint,
    ranges: &[RangeInclusive<u32>],
) -> Result<(), ConstraintProofError> {
    let canonical = constraint.to_string();
    let violated = |actual| ConstraintProofError::Violated {
        constraint: canonical.clone(),
        actual,
    };
    match constraint {
        Constraint::DType { tensor, allowed } => {
            let actual = tensor_facts(inputs, outputs, *tensor)?.dtype;
            if allowed.contains(&actual) {
                Ok(())
            } else {
                Err(violated(format!("{} dtype was {actual:?}", tensor.name)))
            }
        }
        Constraint::Rank { tensor, rank } => {
            let actual = tensor_facts(inputs, outputs, *tensor)?.dimensions.len();
            if actual == usize::from(*rank) {
                Ok(())
            } else {
                Err(violated(format!("{} rank was {actual}", tensor.name)))
            }
        }
        Constraint::Value { value, relation } => {
            let expression = value_expression(operation, inputs, outputs, *value, &canonical)?;
            prove_relation(expression, *relation, value.name(), ranges, &canonical)
        }
        Constraint::DimensionsEqual { left, right } => {
            let left_expression = dimension_expression(inputs, outputs, *left)?;
            let right_expression = dimension_expression(inputs, outputs, *right)?;
            if left_expression == right_expression {
                Ok(())
            } else {
                let left_bounds = left_expression.bounds(ranges, &canonical)?;
                let right_bounds = right_expression.bounds(ranges, &canonical)?;
                Err(violated(format!(
                    "{} was {}..={}, {} was {}..={}",
                    left.name,
                    left_bounds.0,
                    left_bounds.1,
                    right.name,
                    right_bounds.0,
                    right_bounds.1
                )))
            }
        }
        Constraint::Layout { tensor, class } => {
            let facts = tensor_facts(inputs, outputs, *tensor)?;
            let matches = match class {
                LayoutClass::Contiguous => facts.proves_contiguous(ranges, &canonical)?,
                LayoutClass::RowMajor => facts.proves_matrix_layout(false, ranges, &canonical)?,
                LayoutClass::ColumnMajor => facts.proves_matrix_layout(true, ranges, &canonical)?,
            };
            if matches {
                Ok(())
            } else {
                Err(violated(format!("{} is not always {class}", tensor.name)))
            }
        }
        Constraint::ByteOffsetAligned { tensor, alignment } => {
            let facts = tensor_facts(inputs, outputs, *tensor)?;
            let byte_offset = facts
                .offset
                .checked_scale(facts.dtype.byte_size(), &canonical)?;
            if *alignment != 0 && byte_offset.is_multiple_of(*alignment) {
                Ok(())
            } else {
                Err(violated(format!(
                    "{} byte offset is not always aligned to {alignment}",
                    tensor.name
                )))
            }
        }
        Constraint::Capability(capability) => {
            if capabilities.contains(capability) {
                Ok(())
            } else {
                Err(violated(format!("{capability} is unavailable")))
            }
        }
    }
}

fn value_expression(
    operation: TemplateOp,
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    value: ValueRef,
    canonical: &str,
) -> Result<Expression, ConstraintProofError> {
    match value {
        ValueRef::Dimension(dimension) => dimension_expression(inputs, outputs, dimension),
        ValueRef::Product { left, right, .. } => {
            let left = dimension_expression(inputs, outputs, left)?;
            let right = dimension_expression(inputs, outputs, right)?;
            left.checked_product(right, canonical)
        }
        ValueRef::Quotient {
            numerator,
            denominator,
            ..
        } => {
            let numerator = dimension_expression(inputs, outputs, numerator)?;
            let denominator = dimension_expression(inputs, outputs, denominator)?;
            match (numerator.constant_value(), denominator.constant_value()) {
                (Some(numerator), Some(denominator))
                    if denominator != 0 && numerator.is_multiple_of(denominator) =>
                {
                    Ok(Expression::constant(numerator / denominator))
                }
                _ => Err(ConstraintProofError::Unsupported {
                    constraint: canonical.to_owned(),
                }),
            }
        }
        ValueRef::Operation(value) => operation_expression(operation, value, canonical),
    }
}

fn operation_expression(
    operation: TemplateOp,
    value: OperationValue,
    canonical: &str,
) -> Result<Expression, ConstraintProofError> {
    let TemplateOp::Static(operation) = operation else {
        return Err(ConstraintProofError::Unsupported {
            constraint: canonical.to_owned(),
        });
    };
    let Some(value) = operation_value(operation, value) else {
        return Err(ConstraintProofError::Unsupported {
            constraint: canonical.to_owned(),
        });
    };
    Ok(Expression::constant(value))
}

fn prove_relation(
    expression: Expression,
    relation: Relation,
    name: &str,
    ranges: &[RangeInclusive<u32>],
    canonical: &str,
) -> Result<(), ConstraintProofError> {
    let (minimum, maximum) = expression.bounds(ranges, canonical)?;
    let accepted = match relation {
        Relation::Equal(expected) => minimum == expected && maximum == expected,
        Relation::Range { min, max } => minimum >= min && maximum <= max,
        Relation::OneOf(values) => expression
            .constant_value()
            .is_some_and(|value| values.contains(&value)),
        Relation::MultipleOf(_) if expression.product.is_some() => {
            return Err(ConstraintProofError::Unsupported {
                constraint: canonical.to_owned(),
            });
        }
        Relation::MultipleOf(divisor) => divisor != 0 && expression.is_multiple_of(divisor),
    };
    if accepted {
        Ok(())
    } else if matches!(relation, Relation::OneOf(_)) && expression.constant_value().is_none() {
        Err(ConstraintProofError::Unsupported {
            constraint: canonical.to_owned(),
        })
    } else {
        Err(ConstraintProofError::Violated {
            constraint: canonical.to_owned(),
            actual: format!("{name} spans {minimum}..={maximum}"),
        })
    }
}

fn dimension_expression(
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    dimension: Dimension,
) -> Result<Expression, ConstraintProofError> {
    let facts = tensor_facts(inputs, outputs, dimension.tensor)?;
    let index = match dimension.axis {
        Axis::Index(index) => usize::from(index),
        Axis::FromEnd(distance) => facts
            .dimensions
            .len()
            .checked_sub(usize::from(distance))
            .ok_or_else(|| unsupported_dimension(dimension))?,
    };
    facts
        .dimensions
        .get(index)
        .copied()
        .ok_or_else(|| unsupported_dimension(dimension))
}

fn unsupported_dimension(dimension: Dimension) -> ConstraintProofError {
    ConstraintProofError::Unsupported {
        constraint: format!("{} axis is unavailable", dimension.name),
    }
}

#[derive(Clone, Debug)]
struct TensorFacts {
    dtype: DType,
    dimensions: Vec<Expression>,
    strides: Vec<u64>,
    offset: Expression,
}

impl TensorFacts {
    fn from_layout(layout: &Layout) -> Self {
        Self {
            dtype: layout.dtype(),
            dimensions: layout
                .shape()
                .iter()
                .map(|&value| Expression::constant(u64::from(value)))
                .collect(),
            strides: layout.strides().to_vec(),
            offset: Expression::constant(layout.offset()),
        }
    }

    fn proves_contiguous(
        &self,
        ranges: &[RangeInclusive<u32>],
        canonical: &str,
    ) -> Result<bool, ConstraintProofError> {
        let mut expected = Some(1_u64);
        for (dimension, &stride) in self.dimensions.iter().zip(&self.strides).rev() {
            let (minimum, maximum) = dimension.bounds(ranges, canonical)?;
            if maximum <= 1 {
                continue;
            }
            if expected != Some(stride) {
                return Ok(false);
            }
            expected = if minimum == maximum {
                expected.and_then(|value| value.checked_mul(maximum))
            } else {
                None
            };
        }
        Ok(true)
    }

    fn proves_matrix_layout(
        &self,
        column_major: bool,
        ranges: &[RangeInclusive<u32>],
        canonical: &str,
    ) -> Result<bool, ConstraintProofError> {
        if !matches!(self.dimensions.len(), 2 | 3) {
            return Ok(false);
        }
        let rank = self.dimensions.len();
        let rows = self.dimensions[rank - 2].bounds(ranges, canonical)?.1;
        let columns = self.dimensions[rank - 1].bounds(ranges, canonical)?.1;
        let row_stride = self.strides[rank - 2];
        let column_stride = self.strides[rank - 1];
        let matrix_span = if column_major {
            columns
                .checked_sub(1)
                .and_then(|last| last.checked_mul(column_stride))
                .and_then(|span| span.checked_add(rows))
        } else {
            rows.checked_sub(1)
                .and_then(|last| last.checked_mul(row_stride))
                .and_then(|span| span.checked_add(columns))
        }
        .ok_or_else(|| ConstraintProofError::ArithmeticOverflow {
            constraint: canonical.to_owned(),
        })?;
        let matrix_matches = if column_major {
            row_stride == 1 && column_stride >= rows
        } else {
            column_stride == 1 && row_stride >= columns
        };
        let batch_matches = rank == 2
            || self.dimensions[0].bounds(ranges, canonical)?.1 <= 1
            || self.strides[0] >= matrix_span;
        Ok(matrix_matches && batch_matches)
    }
}

fn tensor_facts(
    inputs: &[&TemplateTensor],
    outputs: &[&TemplateTensor],
    tensor: TensorRef,
) -> Result<TensorFacts, ConstraintProofError> {
    let candidate = match tensor.slot {
        TensorSlot::Input(index) => inputs.get(usize::from(index)),
        TensorSlot::Output(index) => outputs.get(usize::from(index)),
    }
    .ok_or_else(|| ConstraintProofError::Unsupported {
        constraint: format!("{} tensor is unavailable", tensor.name),
    })?;
    match candidate {
        TemplateTensor::Concrete(tensor) => Ok(TensorFacts::from_layout(tensor.layout())),
        TemplateTensor::Symbolic { layout, .. } => {
            let facts = layout
                .constraint_facts()
                .map_err(|error| symbolic_proof_error(&error))?;
            Ok(TensorFacts {
                dtype: facts.dtype,
                dimensions: facts
                    .dimensions
                    .into_iter()
                    .map(Expression::from_affine)
                    .collect(),
                strides: facts.strides,
                offset: Expression::from_linear(facts.offset),
            })
        }
    }
}

fn symbolic_proof_error(error: &SymbolicLayoutError) -> ConstraintProofError {
    ConstraintProofError::Unsupported {
        constraint: format!("symbolic layout facts are unavailable: {error}"),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferId, Op, Tensor};

    const INPUT: TensorRef = TensorRef::input("input", 0);
    const OTHER: TensorRef = TensorRef::input("other", 1);
    const INPUT_WIDTH: Dimension = Dimension {
        name: "input width",
        tensor: INPUT,
        axis: Axis::Index(0),
    };
    const OTHER_WIDTH: Dimension = Dimension {
        name: "other width",
        tensor: OTHER,
        axis: Axis::Index(0),
    };
    const PRODUCT: ValueRef = ValueRef::Product {
        name: "cross product",
        left: INPUT_WIDTH,
        right: OTHER_WIDTH,
    };
    fn tensor(buffer: u64, layout: Layout) -> Tensor {
        Tensor::from_allocation(BufferId::new(7, buffer, layout.buffer_len()), layout, true)
            .unwrap()
    }

    fn symbolic_vector(
        buffer: u64,
        space: &ParamSpace,
        parameter: u8,
        capacity: u32,
    ) -> TemplateTensor {
        let base = Layout::contiguous(
            DType::F32,
            0,
            vec![capacity],
            u64::from(capacity) * DType::F32.byte_size(),
        )
        .unwrap();
        let symbolic = crate::SymbolicLayout::new(base.clone(), space.clone())
            .slice(0, 0.into(), Affine::parameter(parameter, 0, 1), 1)
            .unwrap();
        TemplateTensor::symbolic(tensor(buffer, base), symbolic).unwrap()
    }

    #[test]
    fn rejects_contiguity_that_fails_only_inside_the_range() {
        let space = ParamSpace::new(std::iter::once(0..=3).collect()).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![3, 3], 36).unwrap();
        let symbolic = crate::SymbolicLayout::new(base.clone(), space.clone())
            .slice(1, 0.into(), Affine::parameter(0, 0, 1), 1)
            .unwrap();
        let input = TemplateTensor::symbolic(tensor(1, base), symbolic).unwrap();
        let constraint = Constraint::Layout {
            tensor: INPUT,
            class: LayoutClass::Contiguous,
        };

        assert!(matches!(
            prove_constraints(
                &space,
                TemplateOp::Static(Op::Copy),
                &[&input],
                &[],
                &[],
                &[constraint]
            ),
            Err(ConstraintProofError::Violated { .. })
        ));
    }
    #[test]
    fn checked_range_arithmetic_reports_overflow() {
        let expression = Expression {
            offset: u64::MAX,
            coefficients: [1, 0, 0, 0],
            product: None,
        };
        let ranges = std::iter::once(1..=1).collect::<Vec<_>>();
        assert!(matches!(
            expression.bounds(&ranges, "overflow fixture"),
            Err(ConstraintProofError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn dynamic_quotient_without_a_range_proof_is_unsupported() {
        let space = ParamSpace::new(vec![1..=3, 1..=5]).unwrap();
        let left = symbolic_vector(1, &space, 0, 3);
        let right = symbolic_vector(2, &space, 1, 5);
        let constraint = Constraint::Value {
            value: ValueRef::Quotient {
                name: "ratio",
                numerator: INPUT_WIDTH,
                denominator: OTHER_WIDTH,
            },
            relation: Relation::Range { min: 1, max: 3 },
        };
        assert!(matches!(
            prove_constraints(
                &space,
                TemplateOp::Static(Op::Add),
                &[&left, &right],
                &[],
                &[],
                &[constraint]
            ),
            Err(ConstraintProofError::Unsupported { .. })
        ));
    }
    #[test]
    fn divisibility_of_dimension_products_is_unsupported() {
        let space = ParamSpace::new(vec![1..=3, 1..=5]).unwrap();
        let left = symbolic_vector(1, &space, 0, 3);
        let right = symbolic_vector(2, &space, 1, 5);
        let constraint = Constraint::Value {
            value: PRODUCT,
            relation: Relation::MultipleOf(2),
        };

        assert!(matches!(
            prove_constraints(
                &space,
                TemplateOp::Static(Op::Add),
                &[&left, &right],
                &[],
                &[],
                &[constraint]
            ),
            Err(ConstraintProofError::Unsupported { .. })
        ));
    }
}
