use std::{fmt, fmt::Write as _};

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

/// A backend algorithm and its machine-checkable contract.
#[derive(Clone, Copy, Debug)]
pub struct VariantDef<I> {
    /// Stable algorithm identity.
    pub name: &'static str,
    /// Portable operation computed by the algorithm.
    pub operation: OperationKind,
    /// Backend-private implementation tag.
    pub implementation: I,
    /// Capabilities required before the algorithm can be selected.
    pub availability: &'static [DeviceCapability],
    /// Additional requirements beyond the operation contract.
    pub constraints: &'static [Constraint],
    /// Execution guarantee placeholder.
    pub guarantees: Guarantee,
    /// Shapes and workloads where this algorithm is useful.
    pub use_when: &'static [&'static str],
    /// Shapes and workloads where another algorithm is preferable.
    pub avoid_when: &'static [&'static str],
    /// Implementation trade-offs and operational details.
    pub notes: &'static [&'static str],
    /// Lifecycle placeholder.
    pub lifecycle: Lifecycle,
}

/// Renders a stable Markdown catalog from registry definitions.
#[must_use]
pub fn render_catalog_markdown<I>(
    backend: &str,
    device: &str,
    capabilities: &[DeviceCapability],
    variants: &[VariantDef<I>],
) -> String {
    let mut output =
        format!("# {backend} variant catalog\n\n**Device:** {device}\n\n**Capabilities:** ");
    append_display_list(&mut output, capabilities);
    output.push('\n');
    for variant in variants {
        let _ = write!(
            output,
            "\n## {}\n\n**Name:** `{}`\n\n**Computes:** `{}`\n\n**Constraints:**\n",
            variant.name, variant.name, variant.operation
        );
        if variant.constraints.is_empty() {
            output.push_str("\n- None beyond the operation contract.\n");
        } else {
            for constraint in variant.constraints {
                let _ = write!(output, "\n- {constraint}\n");
            }
        }
        append_markdown_sentences(&mut output, "Use when", variant.use_when);
        append_markdown_sentences(&mut output, "Avoid when", variant.avoid_when);
        append_markdown_sentences(&mut output, "Notes", variant.notes);
        let _ = write!(
            output,
            "\n- Guarantee: {}.\n\n- Lifecycle: {}.\n",
            variant.guarantees, variant.lifecycle
        );
    }
    output
}

fn append_markdown_sentences(output: &mut String, heading: &str, sentences: &[&str]) {
    let _ = write!(output, "\n**{heading}:**\n");
    for sentence in sentences {
        let _ = write!(output, "\n- {sentence}\n");
    }
}

fn append_display_list<T: fmt::Display>(output: &mut String, values: &[T]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        let _ = write!(output, "{value}");
    }
}

/// Renders stable unversioned JSON from registry definitions.
#[must_use]
pub fn render_catalog_json<I>(
    backend: &str,
    device: &str,
    capabilities: &[DeviceCapability],
    variants: &[VariantDef<I>],
) -> String {
    let variants = variants
        .iter()
        .map(|variant| {
            serde_json::json!({
                "name": variant.name,
                "computes": variant.operation.to_string(),
                "constraints": variant.constraints.iter().map(json_constraint).collect::<Vec<_>>(),
                "use_when": variant.use_when,
                "avoid_when": variant.avoid_when,
                "notes": variant.notes,
                "guarantees": variant.guarantees.to_string(),
                "lifecycle": variant.lifecycle.to_string(),
            })
        })
        .collect::<Vec<_>>();
    let mut output = serde_json::json!({
        "backend": backend,
        "device": device,
        "capabilities": capabilities.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "variants": variants,
    })
    .to_string();
    output.push('\n');
    output
}

fn json_constraint(constraint: &Constraint) -> serde_json::Value {
    let mut object = serde_json::Map::from_iter([
        ("kind".to_owned(), constraint_kind(constraint).into()),
        ("id".to_owned(), constraint.to_string().into()),
    ]);
    match constraint {
        Constraint::DType { tensor, allowed } => {
            object.insert("tensor".to_owned(), tensor.name.into());
            object.insert(
                "allowed".to_owned(),
                allowed
                    .iter()
                    .copied()
                    .map(dtype_name)
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
        Constraint::Rank { tensor, rank } => {
            object.insert("tensor".to_owned(), tensor.name.into());
            object.insert("rank".to_owned(), (*rank).into());
        }
        Constraint::Value { value, relation } => {
            append_json_value_ref(&mut object, *value);
            append_json_relation(&mut object, *relation);
        }
        Constraint::DimensionsEqual { left, right } => {
            object.insert("left".to_owned(), json_dimension(*left));
            object.insert("right".to_owned(), json_dimension(*right));
        }
        Constraint::Layout { tensor, class } => {
            object.insert("tensor".to_owned(), tensor.name.into());
            object.insert("layout".to_owned(), class.to_string().into());
        }
        Constraint::ByteOffsetAligned { tensor, alignment } => {
            object.insert("tensor".to_owned(), tensor.name.into());
            object.insert("alignment".to_owned(), (*alignment).into());
        }
        Constraint::Capability(capability) => {
            object.insert("capability".to_owned(), capability.to_string().into());
        }
    }
    object.into()
}

const fn constraint_kind(constraint: &Constraint) -> &'static str {
    match constraint {
        Constraint::DType { .. } => "dtype",
        Constraint::Rank { .. } => "rank",
        Constraint::Value {
            value: ValueRef::Dimension(_),
            relation: Relation::Equal(_),
        } => "dim-equal",
        Constraint::Value {
            value: ValueRef::Dimension(_),
            relation: Relation::Range { .. },
        } => "dim-range",
        Constraint::Value {
            value: ValueRef::Dimension(_),
            relation: Relation::OneOf(_),
        } => "dim-one-of",
        Constraint::Value {
            value: ValueRef::Dimension(_),
            relation: Relation::MultipleOf(_),
        } => "dim-divisible",
        Constraint::Value {
            value: ValueRef::Product { .. },
            ..
        } => "dim-product",
        Constraint::Value {
            value: ValueRef::Quotient { .. },
            ..
        } => "dim-quotient",
        Constraint::Value {
            value: ValueRef::Operation(_),
            ..
        } => "operation-value",
        Constraint::DimensionsEqual { .. } => "dims-equal",
        Constraint::Layout { .. } => "layout",
        Constraint::ByteOffsetAligned { .. } => "offset-alignment",
        Constraint::Capability(_) => "capability",
    }
}

fn append_json_value_ref(output: &mut serde_json::Map<String, serde_json::Value>, value: ValueRef) {
    match value {
        ValueRef::Dimension(dimension) => {
            output.insert("dimension".to_owned(), json_dimension(dimension));
        }
        ValueRef::Product { name, left, right } => {
            output.insert("value".to_owned(), name.into());
            output.insert("left".to_owned(), json_dimension(left));
            output.insert("right".to_owned(), json_dimension(right));
        }
        ValueRef::Quotient {
            name,
            numerator,
            denominator,
        } => {
            output.insert("value".to_owned(), name.into());
            output.insert("numerator".to_owned(), json_dimension(numerator));
            output.insert("denominator".to_owned(), json_dimension(denominator));
        }
        ValueRef::Operation(value) => {
            output.insert("value".to_owned(), value.to_string().into());
        }
    }
}

fn json_dimension(dimension: Dimension) -> serde_json::Value {
    let mut output = serde_json::Map::from_iter([
        ("name".to_owned(), dimension.name.into()),
        ("tensor".to_owned(), dimension.tensor.name.into()),
    ]);
    match dimension.axis {
        Axis::Index(axis) => {
            output.insert("axis".to_owned(), axis.into());
        }
        Axis::FromEnd(axis) => {
            output.insert("axis_from_end".to_owned(), axis.into());
        }
    }
    output.into()
}

fn append_json_relation(
    output: &mut serde_json::Map<String, serde_json::Value>,
    relation: Relation,
) {
    match relation {
        Relation::Equal(value) => {
            output.insert("equal".to_owned(), value.into());
        }
        Relation::Range { min, max } => {
            output.insert("min".to_owned(), min.into());
            output.insert("max".to_owned(), max.into());
        }
        Relation::OneOf(values) => {
            output.insert("one_of".to_owned(), serde_json::json!(values));
        }
        Relation::MultipleOf(divisor) => {
            output.insert("multiple_of".to_owned(), divisor.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: TensorRef = TensorRef::input("input", 0);
    const WIDTH: Dimension = Dimension {
        name: "width",
        tensor: INPUT,
        axis: Axis::FromEnd(1),
    };
    const CONSTRAINTS: &[Constraint] = &[Constraint::Value {
        value: ValueRef::Dimension(WIDTH),
        relation: Relation::Range { min: 1, max: 33 },
    }];
    const VARIANTS: &[VariantDef<()>] = &[VariantDef {
        name: "test.reference",
        operation: OperationKind::TopK,
        implementation: (),
        availability: &[DeviceCapability::Metal4],
        constraints: CONSTRAINTS,
        guarantees: Guarantee::Deterministic,
        use_when: &["testing stable output"],
        avoid_when: &["production"],
        notes: &["quotes are escaped: \"yes\""],
        lifecycle: Lifecycle::Active,
    }];

    #[test]
    fn renders_stable_catalogs() {
        let markdown =
            render_catalog_markdown("metal", "Test GPU", &[DeviceCapability::Metal4], VARIANTS);
        assert!(markdown.contains("**Name:** `test.reference`"));
        assert!(markdown.contains("width is in 1..=33"));

        let json = render_catalog_json(
            "metal",
            "Test \\\"GPU",
            &[DeviceCapability::Metal4],
            VARIANTS,
        );
        assert!(json.contains("\"kind\":\"dim-range\""));
        assert!(json.contains("Test \\\\\\\"GPU"));
        assert!(!json.contains("schema_version"));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["variants"][0]["constraints"][0]["min"], 1);
        assert_eq!(value["variants"][0]["constraints"][0]["max"], 33);
        assert_eq!(
            value["variants"][0]["guarantees"],
            VARIANTS[0].guarantees.to_string()
        );
        assert_eq!(
            value["variants"][0]["lifecycle"],
            VARIANTS[0].lifecycle.to_string()
        );
        assert!(markdown.contains(&format!("Guarantee: {}", VARIANTS[0].guarantees)));
        assert!(markdown.contains(&format!("Lifecycle: {}", VARIANTS[0].lifecycle)));
    }
}
