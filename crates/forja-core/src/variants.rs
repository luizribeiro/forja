use std::fmt;

use crate::{DType, Op};

/// A portable operation family that may have backend algorithm variants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OperationKind {
    /// Dense matrix multiplication.
    Matmul,
    /// Affine-quantized matrix multiplication.
    QuantMatmul,
    /// Routed affine-quantized matrix multiplication.
    GatherQuantMatmul,
    /// Routed fused affine-quantized projections with `SiLU`.
    GatherQuantSiluMul,
    /// Scaled dot-product attention.
    Sdpa,
    /// Greatest-element selection.
    TopK,
    /// An operation without selectable variants.
    Other,
}

impl OperationKind {
    /// Classifies a configured operation without inspecting its tensors.
    #[must_use]
    pub const fn of(op: Op) -> Self {
        match op {
            Op::Matmul => Self::Matmul,
            Op::QuantMatmul { .. } => Self::QuantMatmul,
            Op::GatherQuantMatmul { .. } => Self::GatherQuantMatmul,
            Op::GatherQuantSiluMul { .. } => Self::GatherQuantSiluMul,
            Op::Sdpa { .. } => Self::Sdpa,
            Op::TopK { .. } => Self::TopK,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for OperationKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Matmul => "matmul",
            Self::QuantMatmul => "quant-matmul",
            Self::GatherQuantMatmul => "gather-quant-matmul",
            Self::GatherQuantSiluMul => "gather-quant-silu-mul",
            Self::Sdpa => "sdpa",
            Self::TopK => "top-k",
            Self::Other => "other",
        })
    }
}

/// A tensor position in an operation signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorSlot {
    /// An input position.
    Input(u8),
    /// An output position.
    Output(u8),
}

/// A named tensor position used by catalog constraints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorRef {
    /// Canonical human-readable operand name.
    pub name: &'static str,
    /// Position in the operation signature.
    pub slot: TensorSlot,
}

impl TensorRef {
    /// Names an input tensor.
    #[must_use]
    pub const fn input(name: &'static str, index: u8) -> Self {
        Self {
            name,
            slot: TensorSlot::Input(index),
        }
    }

    /// Names an output tensor.
    #[must_use]
    pub const fn output(name: &'static str, index: u8) -> Self {
        Self {
            name,
            slot: TensorSlot::Output(index),
        }
    }
}

/// An axis counted from the start or end of a tensor shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    /// Zero-based axis counted from the start.
    Index(u8),
    /// One-based axis counted from the end.
    FromEnd(u8),
}

/// A named tensor dimension used by a constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dimension {
    /// Canonical dimension name.
    pub name: &'static str,
    /// Tensor containing the dimension.
    pub tensor: TensorRef,
    /// Axis containing the dimension.
    pub axis: Axis,
}

/// A scalar operation field available to constraints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationValue {
    /// Quantized element width.
    Bits,
    /// Quantization group size.
    GroupSize,
    /// Top-k selection count.
    TopK,
}

impl fmt::Display for OperationValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bits => "bits",
            Self::GroupSize => "group size",
            Self::TopK => "k",
        })
    }
}

/// Reads a catalog-visible scalar field from an operation.
#[must_use]
pub fn operation_value(op: Op, value: OperationValue) -> Option<u64> {
    match (op, value) {
        (
            Op::QuantMatmul { bits, .. }
            | Op::GatherQuantMatmul { bits, .. }
            | Op::GatherQuantSiluMul { bits, .. },
            OperationValue::Bits,
        ) => Some(u64::from(bits)),
        (
            Op::QuantMatmul { group_size, .. }
            | Op::GatherQuantMatmul { group_size, .. }
            | Op::GatherQuantSiluMul { group_size, .. },
            OperationValue::GroupSize,
        ) => Some(u64::from(group_size)),
        (Op::TopK { k, .. }, OperationValue::TopK) => Some(u64::from(k)),
        _ => None,
    }
}

/// An integer fact read from an operation and its tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueRef {
    /// A tensor extent.
    Dimension(Dimension),
    /// A checked product of two tensor extents.
    Product {
        /// Canonical product name.
        name: &'static str,
        /// Left factor.
        left: Dimension,
        /// Right factor.
        right: Dimension,
    },
    /// A checked quotient of two tensor extents.
    Quotient {
        /// Canonical quotient name.
        name: &'static str,
        /// Numerator.
        numerator: Dimension,
        /// Denominator.
        denominator: Dimension,
    },
    /// A scalar operation field.
    Operation(OperationValue),
}

impl ValueRef {
    /// Returns the stable human-readable name of this value.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dimension(dimension) => dimension.name,
            Self::Product { name, .. } | Self::Quotient { name, .. } => name,
            Self::Operation(OperationValue::Bits) => "bits",
            Self::Operation(OperationValue::GroupSize) => "group size",
            Self::Operation(OperationValue::TopK) => "k",
        }
    }
}

/// A closed integer relation used by catalog constraints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Relation {
    /// Exactly one value.
    Equal(u64),
    /// One inclusive interval.
    Range {
        /// Inclusive lower bound.
        min: u64,
        /// Inclusive upper bound.
        max: u64,
    },
    /// One of a fixed set of values.
    OneOf(&'static [u64]),
    /// Divisible by a nonzero value.
    MultipleOf(u64),
}

/// A layout property required by an implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutClass {
    /// Dense row-major storage.
    Contiguous,
    /// A matrix whose columns are adjacent.
    RowMajor,
    /// A matrix whose rows are adjacent.
    ColumnMajor,
}

impl fmt::Display for LayoutClass {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Contiguous => "contiguous",
            Self::RowMajor => "row-major",
            Self::ColumnMajor => "column-major",
        })
    }
}

/// A device or backend feature required by a variant.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceCapability {
    /// Metal 4 command submission and shader support.
    Metal4,
}

impl fmt::Display for DeviceCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Metal4 => "metal4",
        })
    }
}

/// One checkable variant constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Constraint {
    /// A tensor must have one of the listed scalar types.
    DType {
        /// Tensor being checked.
        tensor: TensorRef,
        /// Accepted scalar types.
        allowed: &'static [DType],
    },
    /// A tensor must have the given rank.
    Rank {
        /// Tensor being checked.
        tensor: TensorRef,
        /// Required rank.
        rank: u8,
    },
    /// An integer fact must satisfy a relation.
    Value {
        /// Fact being checked.
        value: ValueRef,
        /// Required relation.
        relation: Relation,
    },
    /// Two dimensions must be equal.
    DimensionsEqual {
        /// Left dimension.
        left: Dimension,
        /// Right dimension.
        right: Dimension,
    },
    /// A tensor must have a supported layout class.
    Layout {
        /// Tensor being checked.
        tensor: TensorRef,
        /// Required layout class.
        class: LayoutClass,
    },
    /// A tensor's byte offset must have the requested alignment.
    ByteOffsetAligned {
        /// Tensor being checked.
        tensor: TensorRef,
        /// Required byte alignment.
        alignment: u64,
    },
    /// A backend capability must be present.
    Capability(DeviceCapability),
}

impl fmt::Display for Constraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DType { tensor, allowed } => {
                write!(formatter, "{} dtype is one of ", tensor.name)?;
                write_joined(formatter, allowed.iter().copied().map(dtype_name), ", ")
            }
            Self::Rank { tensor, rank } => write!(formatter, "{} rank = {rank}", tensor.name),
            Self::Value { value, relation } => write_relation(formatter, value.name(), *relation),
            Self::DimensionsEqual { left, right } => {
                write!(formatter, "{} = {}", left.name, right.name)
            }
            Self::Layout { tensor, class } => write!(formatter, "{} is {class}", tensor.name),
            Self::ByteOffsetAligned { tensor, alignment } => {
                write!(
                    formatter,
                    "{} byte offset is aligned to {alignment}",
                    tensor.name
                )
            }
            Self::Capability(capability) => write!(formatter, "requires {capability}"),
        }
    }
}

fn write_relation(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    relation: Relation,
) -> fmt::Result {
    match relation {
        Relation::Equal(value) => write!(formatter, "{name} = {value}"),
        Relation::Range { min, max } => write!(formatter, "{name} is in {min}..={max}"),
        Relation::OneOf(values) => {
            write!(formatter, "{name} is one of ")?;
            write_joined(formatter, values.iter(), ", ")
        }
        Relation::MultipleOf(divisor) => write!(formatter, "{name} is a multiple of {divisor}"),
    }
}

fn write_joined<T: fmt::Display>(
    formatter: &mut fmt::Formatter<'_>,
    values: impl IntoIterator<Item = T>,
    separator: &str,
) -> fmt::Result {
    let mut first = true;
    for value in values {
        if !first {
            formatter.write_str(separator)?;
        }
        first = false;
        value.fmt(formatter)?;
    }
    Ok(())
}

const fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
    }
}

/// The initial, standard execution guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Guarantee {
    /// Identical inputs and state follow a deterministic algorithm.
    Deterministic,
}

impl fmt::Display for Guarantee {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Deterministic => "deterministic",
        })
    }
}

/// The initial registry lifecycle state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lifecycle {
    /// The variant is supported and listed.
    Active,
}

impl fmt::Display for Lifecycle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Active => "active",
        })
    }
}
