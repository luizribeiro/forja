use std::{error::Error, fmt};

use forja_core::{
    Axis, Constraint, DeviceCapability, Dimension, Dispatch, Guarantee, Layout, LayoutClass,
    Lifecycle, OperationKind, OperationValue, Relation, TensorRef, TensorSlot, ValueRef,
    VariantDef, operation_value, render_catalog_json, render_catalog_markdown,
};

use crate::matmul::{MatrixLayout, classify};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MetalVariant {
    MatmulGemv,
    MatmulGemvTransposed,
    MatmulSteel64x64x16_2x2,
    MatmulSteel64x64x16_1x2,
    MatmulSteel64x32x32_2x2,
    MatmulSteel32x64x16_1x2,
    QuantMatmulGemv,
    QuantMatmulQ8FastGemv,
    QuantMatmulSmallM,
    QuantMatmulTiled,
    GatherQuantMatmulRouteGemv,
    GatherQuantMatmulGrouped,
    GatherQuantSiluMulRouteGemv,
    GatherQuantSiluMulGrouped,
    SdpaDecomposed,
    SdpaVectorSinglePass,
    SdpaVectorTwoPass,
    SdpaSteel,
    TopKSingleK8,
    TopKPartials,
}

/// A registry variant resolved and checked for one concrete dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedVariant {
    index: usize,
}

impl ValidatedVariant {
    /// Returns the stable algorithm name.
    #[must_use]
    pub fn name(self) -> &'static str {
        REGISTRY[self.index].name
    }

    /// Returns the operation family computed by the algorithm.
    #[must_use]
    pub fn operation(self) -> OperationKind {
        REGISTRY[self.index].operation
    }
}

/// A reason a concrete Metal algorithm variant was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariantError {
    /// No registry entry has the requested stable name.
    Unknown {
        /// Requested operation family.
        operation: OperationKind,
        /// Requested stable name.
        name: String,
    },
    /// The stable name belongs to another operation family.
    WrongOperation {
        /// Requested operation family.
        operation: OperationKind,
        /// Requested stable name.
        name: String,
        /// Operation family owning the name.
        actual: OperationKind,
    },
    /// A required capability is absent.
    Unavailable {
        /// Requested operation family.
        operation: OperationKind,
        /// Requested stable name.
        name: String,
        /// First unavailable capability.
        capability: DeviceCapability,
    },
    /// The first canonical constraint that the dispatch violates.
    Constraint {
        /// Requested operation family.
        operation: OperationKind,
        /// Requested stable name.
        name: String,
        /// Canonical constraint text.
        constraint: String,
        /// Concrete fact that caused the refusal.
        actual: String,
    },
}

impl fmt::Display for VariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { operation, name } => {
                write!(
                    formatter,
                    "metal {operation} variant '{name}': unknown variant"
                )
            }
            Self::WrongOperation {
                operation,
                name,
                actual,
            } => write!(
                formatter,
                "metal {operation} variant '{name}': computes {actual}, not {operation}"
            ),
            Self::Unavailable {
                operation,
                name,
                capability,
            } => write!(
                formatter,
                "metal {operation} variant '{name}': unavailable without {capability}"
            ),
            Self::Constraint {
                operation,
                name,
                constraint,
                actual,
            } => write!(
                formatter,
                "metal {operation} variant '{name}': constraint '{constraint}' failed ({actual})"
            ),
        }
    }
}

impl Error for VariantError {}

const METAL4: &[DeviceCapability] = &[DeviceCapability::Metal4];
const ACTIVE: Lifecycle = Lifecycle::Active;
const DETERMINISTIC: Guarantee = Guarantee::Deterministic;

const LEFT: TensorRef = TensorRef::input("left", 0);
const RIGHT: TensorRef = TensorRef::input("right", 1);
const INPUT: TensorRef = TensorRef::input("input", 0);
const PACKED: TensorRef = TensorRef::input("packed weights", 1);
const SCALES: TensorRef = TensorRef::input("scales", 2);
const BIASES: TensorRef = TensorRef::input("biases", 3);
const INDICES_MATMUL: TensorRef = TensorRef::input("route indices", 4);
const INDICES_SILU: TensorRef = TensorRef::input("route indices", 7);
const OUTPUT: TensorRef = TensorRef::output("output", 0);
const QUERY: TensorRef = TensorRef::input("query", 0);
const KEY: TensorRef = TensorRef::input("key", 1);
const VALUE: TensorRef = TensorRef::input("value", 2);

const M: Dimension = dim("m", LEFT, Axis::FromEnd(2));
const QUANT_M: Dimension = dim("m", INPUT, Axis::Index(0));
const INNER: Dimension = dim("inner", INPUT, Axis::Index(1));
const EXPERTS: Dimension = dim("experts", PACKED, Axis::Index(0));
const MATMUL_ROWS: Dimension = dim("rows", INDICES_MATMUL, Axis::Index(0));
const MATMUL_ROUTES: Dimension = dim("routes", INDICES_MATMUL, Axis::Index(1));
const SILU_ROWS: Dimension = dim("rows", INDICES_SILU, Axis::Index(0));
const SILU_ROUTES: Dimension = dim("routes", INDICES_SILU, Axis::Index(1));
const QUERY_HEADS: Dimension = dim("query heads", QUERY, Axis::Index(0));
const QUERY_LENGTH: Dimension = dim("query length", QUERY, Axis::Index(1));
const KEY_HEADS: Dimension = dim("key/value heads", KEY, Axis::Index(0));
const KEY_LENGTH: Dimension = dim("key length", KEY, Axis::Index(1));
const KEY_WIDTH: Dimension = dim("key width", KEY, Axis::Index(2));
const VALUE_WIDTH: Dimension = dim("value width", VALUE, Axis::Index(2));
const TOP_K_WIDTH: Dimension = dim("width", INPUT, Axis::FromEnd(1));

const fn dim(name: &'static str, tensor: TensorRef, axis: Axis) -> Dimension {
    Dimension { name, tensor, axis }
}

const fn value(dimension: Dimension, relation: Relation) -> Constraint {
    Constraint::Value {
        value: ValueRef::Dimension(dimension),
        relation,
    }
}

const fn operation(value: OperationValue, relation: Relation) -> Constraint {
    Constraint::Value {
        value: ValueRef::Operation(value),
        relation,
    }
}

const M_ONE: &[Constraint] = &[value(M, Relation::Equal(1))];
const M_ONE_TRANSPOSED: &[Constraint] = &[
    value(M, Relation::Equal(1)),
    Constraint::Layout {
        tensor: LEFT,
        class: LayoutClass::RowMajor,
    },
    Constraint::Layout {
        tensor: RIGHT,
        class: LayoutClass::ColumnMajor,
    },
];
const STEEL: &[Constraint] = &[value(
    M,
    Relation::Range {
        min: 2,
        max: u64::MAX,
    },
)];
const QUANT_GEMV: &[Constraint] = &[value(QUANT_M, Relation::Equal(1))];
const Q8_FAST: &[Constraint] = &[
    value(QUANT_M, Relation::Equal(1)),
    operation(OperationValue::Bits, Relation::Equal(8)),
    operation(OperationValue::GroupSize, Relation::MultipleOf(8)),
    value(INNER, Relation::MultipleOf(256)),
    Constraint::Layout {
        tensor: INPUT,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: PACKED,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: SCALES,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: BIASES,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: OUTPUT,
        class: LayoutClass::Contiguous,
    },
    Constraint::ByteOffsetAligned {
        tensor: INPUT,
        alignment: 4,
    },
    Constraint::ByteOffsetAligned {
        tensor: PACKED,
        alignment: 2,
    },
];
const SMALL_M: &[Constraint] = &[value(
    QUANT_M,
    Relation::Range {
        min: 2,
        max: u64::MAX,
    },
)];
const ACTIVATION_DTYPES: &[forja_core::DType] = &[
    forja_core::DType::F32,
    forja_core::DType::F16,
    forja_core::DType::BF16,
];
const TILED: &[Constraint] = &[
    value(
        QUANT_M,
        Relation::Range {
            min: 16,
            max: u64::MAX,
        },
    ),
    Constraint::DType {
        tensor: INPUT,
        allowed: ACTIVATION_DTYPES,
    },
    Constraint::Layout {
        tensor: INPUT,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: PACKED,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: SCALES,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: BIASES,
        class: LayoutClass::Contiguous,
    },
    Constraint::Layout {
        tensor: OUTPUT,
        class: LayoutClass::Contiguous,
    },
];

const fn grouped(rows: Dimension, routes: Dimension) -> [Constraint; 2] {
    [
        Constraint::Value {
            value: ValueRef::Product {
                name: "rows * routes",
                left: rows,
                right: routes,
            },
            relation: Relation::Range {
                min: 64,
                max: u64::MAX,
            },
        },
        value(EXPERTS, Relation::Range { min: 1, max: 256 }),
    ]
}

const GROUPED_MATMUL: &[Constraint] = &grouped(MATMUL_ROWS, MATMUL_ROUTES);
const GROUPED_SILU: &[Constraint] = &grouped(SILU_ROWS, SILU_ROUTES);
const WIDTHS: &[u64] = &[64, 128];
const VECTOR_COMMON: &[Constraint] = &[
    value(QUERY_LENGTH, Relation::Equal(1)),
    value(KEY_WIDTH, Relation::OneOf(WIDTHS)),
    Constraint::DimensionsEqual {
        left: KEY_WIDTH,
        right: VALUE_WIDTH,
    },
    Constraint::Value {
        value: ValueRef::Quotient {
            name: "grouped-head SIMD count",
            numerator: QUERY_HEADS,
            denominator: KEY_HEADS,
        },
        relation: Relation::Range { min: 1, max: 32 },
    },
    value(KEY_LENGTH, Relation::Range { min: 1, max: 1023 }),
];
const VECTOR_TWO_PASS: &[Constraint] = &[
    value(QUERY_LENGTH, Relation::Equal(1)),
    value(KEY_WIDTH, Relation::OneOf(WIDTHS)),
    Constraint::DimensionsEqual {
        left: KEY_WIDTH,
        right: VALUE_WIDTH,
    },
    Constraint::Value {
        value: ValueRef::Quotient {
            name: "grouped-head SIMD count",
            numerator: QUERY_HEADS,
            denominator: KEY_HEADS,
        },
        relation: Relation::Range { min: 1, max: 32 },
    },
    value(
        KEY_LENGTH,
        Relation::Range {
            min: 1024,
            max: 65_536,
        },
    ),
];
const SDPA_STEEL: &[Constraint] = &[
    value(
        QUERY_LENGTH,
        Relation::Range {
            min: 2,
            max: u64::MAX,
        },
    ),
    value(KEY_WIDTH, Relation::OneOf(WIDTHS)),
    Constraint::DimensionsEqual {
        left: KEY_WIDTH,
        right: VALUE_WIDTH,
    },
];
const TOP_K_SINGLE: &[Constraint] = &[
    operation(OperationValue::TopK, Relation::Equal(8)),
    value(TOP_K_WIDTH, Relation::Range { min: 8, max: 2048 }),
];

macro_rules! variant {
    ($name:literal, $operation:ident, $implementation:ident, $constraints:expr,
     $use_when:expr, $avoid_when:expr, $notes:expr) => {
        VariantDef {
            name: $name,
            operation: OperationKind::$operation,
            implementation: MetalVariant::$implementation,
            availability: METAL4,
            constraints: $constraints,
            guarantees: DETERMINISTIC,
            use_when: $use_when,
            avoid_when: $avoid_when,
            notes: $notes,
            lifecycle: ACTIVE,
        }
    };
}

pub(crate) static REGISTRY: &[VariantDef<MetalVariant>] = &[
    variant!(
        "matmul.gemv",
        Matmul,
        MatmulGemv,
        M_ONE,
        &["m = 1 with ordinary matrix layouts."],
        &["the right matrix is column-major."],
        &["Bandwidth-oriented matrix-vector multiplication."]
    ),
    variant!(
        "matmul.gemv-transposed",
        Matmul,
        MatmulGemvTransposed,
        M_ONE_TRANSPOSED,
        &["m = 1 and the right matrix is column-major."],
        &["either matrix needs layout staging."],
        &["Reads the right operand in its transposed storage order."]
    ),
    variant!(
        "matmul.steel-64x64x16-2x2",
        Matmul,
        MatmulSteel64x64x16_2x2,
        STEEL,
        &["large f32 output matrices."],
        &["m = 1 or small half-precision products."],
        &["A 64 by 64 Steel tile with 16-wide reduction blocks."]
    ),
    variant!(
        "matmul.steel-64x64x16-1x2",
        Matmul,
        MatmulSteel64x64x16_1x2,
        STEEL,
        &["large half-precision output matrices."],
        &["small products where launch overhead dominates."],
        &["A 64 by 64 Steel tile with lower thread count."]
    ),
    variant!(
        "matmul.steel-64x32x32-2x2",
        Matmul,
        MatmulSteel64x32x32_2x2,
        STEEL,
        &["half-precision NT products and smaller outputs."],
        &["wide output tiles dominate the shape."],
        &["A 64 by 32 Steel tile with 32-wide reduction blocks."]
    ),
    variant!(
        "matmul.steel-32x64x16-1x2",
        Matmul,
        MatmulSteel32x64x16_1x2,
        STEEL,
        &["wide output matrices or small f32 NT products."],
        &["tall output tiles dominate the shape."],
        &["A 32 by 64 Steel tile with 16-wide reduction blocks."]
    ),
    variant!(
        "quant-matmul.gemv",
        QuantMatmul,
        QuantMatmulGemv,
        QUANT_GEMV,
        &["single-row decode with general supported quantization."],
        &["large batches that can amortize tiled work."],
        &["General quantized matrix-vector multiplication."]
    ),
    variant!(
        "quant-matmul.q8-fast-gemv",
        QuantMatmul,
        QuantMatmulQ8FastGemv,
        Q8_FAST,
        &["aligned contiguous q8 decode with wide inner dimensions."],
        &["strided tensors or q4 weights."],
        &["Uses vectorized q8 loads."]
    ),
    variant!(
        "quant-matmul.small-m",
        QuantMatmul,
        QuantMatmulSmallM,
        SMALL_M,
        &["small batches or non-contiguous larger inputs."],
        &["contiguous batches of at least 16 rows."],
        &["The general quantized matrix-matrix fallback."]
    ),
    variant!(
        "quant-matmul.tiled",
        QuantMatmul,
        QuantMatmulTiled,
        TILED,
        &["contiguous batches of at least 16 rows."],
        &["small or strided batches."],
        &["One algorithm supports f32, f16, and bf16 activations."]
    ),
    variant!(
        "gather-quant-matmul.route-gemv",
        GatherQuantMatmul,
        GatherQuantMatmulRouteGemv,
        &[],
        &["fewer than 64 routed rows or more than 256 experts."],
        &["enough routes to amortize grouping."],
        &["Executes each routed row directly."]
    ),
    variant!(
        "gather-quant-matmul.grouped",
        GatherQuantMatmul,
        GatherQuantMatmulGrouped,
        GROUPED_MATMUL,
        &["at least 64 routed rows with at most 256 experts."],
        &["small route counts."],
        &["Sorts routes into bounded expert groups."]
    ),
    variant!(
        "gather-quant-silu-mul.route-gemv",
        GatherQuantSiluMul,
        GatherQuantSiluMulRouteGemv,
        &[],
        &["fewer than 64 routed rows or more than 256 experts."],
        &["enough routes to amortize grouping."],
        &["Executes both projections per routed row."]
    ),
    variant!(
        "gather-quant-silu-mul.grouped",
        GatherQuantSiluMul,
        GatherQuantSiluMulGrouped,
        GROUPED_SILU,
        &["at least 64 routed rows with at most 256 experts."],
        &["small route counts."],
        &["Groups routes before both quantized projections."]
    ),
    variant!(
        "sdpa.decomposed",
        Sdpa,
        SdpaDecomposed,
        &[],
        &["general attention shapes and layouts."],
        &["a specialized vector or Steel kernel is known to fit."],
        &["Broad baseline assembled from trusted operations."]
    ),
    variant!(
        "sdpa.vector-single-pass",
        Sdpa,
        SdpaVectorSinglePass,
        VECTOR_COMMON,
        &["decode with key length below 1024."],
        &["prefill or long cache lengths."],
        &["Keeps the reduction in one pass."]
    ),
    variant!(
        "sdpa.vector-two-pass",
        Sdpa,
        SdpaVectorTwoPass,
        VECTOR_TWO_PASS,
        &["decode with key length from 1024 through 65536."],
        &["prefill or short cache lengths."],
        &["Uses bounded temporary reductions."]
    ),
    variant!(
        "sdpa.steel",
        Sdpa,
        SdpaSteel,
        SDPA_STEEL,
        &["multi-token queries with supported head widths."],
        &["single-token decode."],
        &["Uses tiled matrix operations for prefill-shaped work."]
    ),
    variant!(
        "top-k.single-k8",
        TopK,
        TopKSingleK8,
        TOP_K_SINGLE,
        &["k = 8 and width at most 2048."],
        &["other k values or wider rows."],
        &["Selects and finalizes in one kernel."]
    ),
    variant!(
        "top-k.partials",
        TopK,
        TopKPartials,
        &[],
        &["the full core TopK range."],
        &["k = 8 with width at most 2048."],
        &["Reduces chunk partials in a finalize pass."]
    ),
];

pub(crate) fn resolve(
    dispatch: &Dispatch,
    name: &str,
    capabilities: &[DeviceCapability],
) -> Result<ValidatedVariant, VariantError> {
    let operation = OperationKind::of(dispatch.op());
    let Some((index, definition)) = REGISTRY
        .iter()
        .enumerate()
        .find(|(_, definition)| definition.name == name)
    else {
        return Err(VariantError::Unknown {
            operation,
            name: name.to_owned(),
        });
    };
    if definition.operation != operation {
        return Err(VariantError::WrongOperation {
            operation,
            name: name.to_owned(),
            actual: definition.operation,
        });
    }
    if let Some(&capability) = definition
        .availability
        .iter()
        .find(|capability| !capabilities.contains(capability))
    {
        return Err(VariantError::Unavailable {
            operation,
            name: name.to_owned(),
            capability,
        });
    }
    for constraint in definition.constraints {
        if let Err(actual) = check_constraint(dispatch, capabilities, constraint) {
            return Err(VariantError::Constraint {
                operation,
                name: name.to_owned(),
                constraint: constraint.to_string(),
                actual,
            });
        }
    }
    Ok(ValidatedVariant { index })
}

fn check_constraint(
    dispatch: &Dispatch,
    capabilities: &[DeviceCapability],
    constraint: &Constraint,
) -> Result<(), String> {
    match constraint {
        Constraint::DType { tensor, allowed } => {
            let dtype = tensor_layout(dispatch, *tensor)?.dtype();
            if allowed.contains(&dtype) {
                Ok(())
            } else {
                Err(format!("{} dtype was {dtype:?}", tensor.name))
            }
        }
        Constraint::Rank { tensor, rank } => {
            let actual = tensor_layout(dispatch, *tensor)?.shape().len();
            if actual == usize::from(*rank) {
                Ok(())
            } else {
                Err(format!("{} rank was {actual}", tensor.name))
            }
        }
        Constraint::Value { value, relation } => {
            let actual = concrete_value(dispatch, *value)?;
            if relation_accepts(*relation, actual) {
                Ok(())
            } else {
                Err(format!("{} was {actual}", value.name()))
            }
        }
        Constraint::DimensionsEqual { left, right } => {
            let left_value = dimension(dispatch, *left)?;
            let right_value = dimension(dispatch, *right)?;
            if left_value == right_value {
                Ok(())
            } else {
                Err(format!(
                    "{} was {left_value}, {} was {right_value}",
                    left.name, right.name
                ))
            }
        }
        Constraint::Layout { tensor, class } => {
            let layout = tensor_layout(dispatch, *tensor)?;
            if layout_matches(layout, *class) {
                Ok(())
            } else {
                Err(format!("{} layout did not match {class}", tensor.name))
            }
        }
        Constraint::ByteOffsetAligned { tensor, alignment } => {
            let layout = tensor_layout(dispatch, *tensor)?;
            let offset = layout
                .offset()
                .checked_mul(layout.dtype().byte_size())
                .ok_or_else(|| format!("{} byte offset overflowed", tensor.name))?;
            if *alignment != 0 && offset.is_multiple_of(*alignment) {
                Ok(())
            } else {
                Err(format!("{} byte offset was {offset}", tensor.name))
            }
        }
        Constraint::Capability(capability) => {
            if capabilities.contains(capability) {
                Ok(())
            } else {
                Err(format!("capability {capability} was unavailable"))
            }
        }
    }
}

fn concrete_value(dispatch: &Dispatch, value: ValueRef) -> Result<u64, String> {
    match value {
        ValueRef::Dimension(dimension_ref) => dimension(dispatch, dimension_ref),
        ValueRef::Product { name, left, right } => dimension(dispatch, left)?
            .checked_mul(dimension(dispatch, right)?)
            .ok_or_else(|| format!("{name} overflowed")),
        ValueRef::Quotient {
            name,
            numerator,
            denominator,
        } => {
            let numerator = dimension(dispatch, numerator)?;
            let denominator = dimension(dispatch, denominator)?;
            if denominator == 0 || !numerator.is_multiple_of(denominator) {
                return Err(format!("{name} was not an exact quotient"));
            }
            Ok(numerator / denominator)
        }
        ValueRef::Operation(value) => {
            operation_value(dispatch.op(), value).ok_or_else(|| format!("operation has no {value}"))
        }
    }
}

fn dimension(dispatch: &Dispatch, dimension: Dimension) -> Result<u64, String> {
    let shape = tensor_layout(dispatch, dimension.tensor)?.shape();
    let index = match dimension.axis {
        Axis::Index(index) => usize::from(index),
        Axis::FromEnd(distance) => shape
            .len()
            .checked_sub(usize::from(distance))
            .ok_or_else(|| format!("{} axis was absent", dimension.name))?,
    };
    shape
        .get(index)
        .copied()
        .map(u64::from)
        .ok_or_else(|| format!("{} axis was absent", dimension.name))
}

fn tensor_layout(dispatch: &Dispatch, tensor: TensorRef) -> Result<&Layout, String> {
    let candidate = match tensor.slot {
        TensorSlot::Input(index) => dispatch.inputs().get(usize::from(index)),
        TensorSlot::Output(index) => dispatch.outputs().get(usize::from(index)),
    };
    candidate
        .map(forja_core::Tensor::layout)
        .ok_or_else(|| format!("{} tensor was absent", tensor.name))
}

const fn relation_accepts(relation: Relation, actual: u64) -> bool {
    match relation {
        Relation::Equal(expected) => actual == expected,
        Relation::Range { min, max } => actual >= min && actual <= max,
        Relation::OneOf(values) => {
            let mut index = 0;
            while index < values.len() {
                if actual == values[index] {
                    return true;
                }
                index += 1;
            }
            false
        }
        Relation::MultipleOf(divisor) => divisor != 0 && actual.is_multiple_of(divisor),
    }
}

fn layout_matches(layout: &Layout, class: LayoutClass) -> bool {
    match class {
        LayoutClass::Contiguous => layout.is_contiguous(),
        LayoutClass::RowMajor => matches!(classify(layout), MatrixLayout::RowMajor { .. }),
        LayoutClass::ColumnMajor => matches!(classify(layout), MatrixLayout::ColumnMajor { .. }),
    }
}

pub(crate) fn markdown(device: &str) -> String {
    render_catalog_markdown("metal", device, METAL4, REGISTRY)
}

pub(crate) fn json(device: &str) -> String {
    render_catalog_json("metal", device, METAL4, REGISTRY)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use forja_core::{CommandList, DType, Op, Slice, ViewOp};
    use forja_cpu::CpuBackend;
    use proptest::prelude::*;

    use super::*;

    fn matmul_dispatch(rows: u32) -> Dispatch {
        let backend = CpuBackend::new();
        let left = backend.alloc(DType::F32, &[rows, 7]).unwrap();
        let right = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[rows, 33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&left, &right], &output)
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn matmul_case(dtype: DType, rows: u32, inner: u32, columns: u32, nt: bool) -> Dispatch {
        let backend = CpuBackend::new();
        let left = backend.alloc(dtype, &[rows, inner]).unwrap();
        let right = if nt {
            let stored = backend.alloc(dtype, &[columns, inner]).unwrap();
            backend.view(&stored, ViewOp::Permute(vec![1, 0])).unwrap()
        } else {
            backend.alloc(dtype, &[inner, columns]).unwrap()
        };
        let output = backend.alloc(dtype, &[rows, columns]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&left, &right], &output)
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn top_k_dispatch(width: u32) -> Dispatch {
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[width]).unwrap();
        let values = backend.alloc(DType::F32, &[8]).unwrap();
        let indices = backend.alloc(DType::U32, &[8]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::TopK {
                    k: 8,
                    normalize: false,
                },
                &[&input],
                &[&values, &indices],
            )
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn quant_dispatch(dtype: DType, rows: u32, inner: u32, strided: bool) -> Dispatch {
        let backend = CpuBackend::new();
        let columns = 33;
        let bits = 8;
        let group_size = 64;
        let input = if strided {
            let allocation = backend.alloc(dtype, &[rows, inner * 2]).unwrap();
            backend
                .view(
                    &allocation,
                    ViewOp::Slice(vec![
                        Slice::new(0, rows, 1).unwrap(),
                        Slice::new(0, inner, 2).unwrap(),
                    ]),
                )
                .unwrap()
        } else {
            backend.alloc(dtype, &[rows, inner]).unwrap()
        };
        let packed = backend
            .alloc(DType::U32, &[columns, inner * u32::from(bits) / 32])
            .unwrap();
        let parameter_dtype = if dtype == DType::F32 {
            DType::F16
        } else {
            dtype
        };
        let scales = backend
            .alloc(parameter_dtype, &[columns, inner / group_size])
            .unwrap();
        let biases = backend
            .alloc(parameter_dtype, &[columns, inner / group_size])
            .unwrap();
        let output = backend.alloc(dtype, &[rows, columns]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::QuantMatmul { bits, group_size },
                &[&input, &packed, &scales, &biases],
                &output,
            )
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn gathered_dispatch(silu: bool, rows: u32, routes: u32, experts: u32) -> Dispatch {
        let backend = CpuBackend::new();
        let (dtype, inner, columns, bits, group_size) = (DType::BF16, 64, 33, 4, 64);
        let input = backend.alloc(dtype, &[rows, inner]).unwrap();
        let weights = || {
            (
                backend
                    .alloc(
                        DType::U32,
                        &[experts, columns, inner * u32::from(bits) / 32],
                    )
                    .unwrap(),
                backend
                    .alloc(dtype, &[experts, columns, inner / group_size])
                    .unwrap(),
                backend
                    .alloc(dtype, &[experts, columns, inner / group_size])
                    .unwrap(),
            )
        };
        let (packed, scales, biases) = weights();
        let indices = backend.alloc(DType::U32, &[rows, routes]).unwrap();
        let output = backend.alloc(dtype, &[rows, routes, columns]).unwrap();
        let mut commands = CommandList::new();
        if silu {
            let (up_packed, up_scales, up_biases) = weights();
            commands
                .dispatch(
                    Op::GatherQuantSiluMul { bits, group_size },
                    &[
                        &input, &packed, &scales, &biases, &up_packed, &up_scales, &up_biases,
                        &indices,
                    ],
                    &output,
                )
                .unwrap();
        } else {
            commands
                .dispatch(
                    Op::GatherQuantMatmul { bits, group_size },
                    &[&input, &packed, &scales, &biases, &indices],
                    &output,
                )
                .unwrap();
        }
        commands.into_dispatches().remove(0)
    }

    fn sdpa_dispatch(
        dtype: DType,
        group: u32,
        query_length: u32,
        key_length: u32,
        width: u32,
        value_width: u32,
    ) -> Dispatch {
        let backend = CpuBackend::new();
        let kv_heads = 8;
        let query_heads = kv_heads * group;
        let query = backend
            .alloc(dtype, &[query_heads, query_length, width])
            .unwrap();
        let key = backend
            .alloc(dtype, &[kv_heads, key_length, width])
            .unwrap();
        let value = backend
            .alloc(dtype, &[kv_heads, key_length, value_width])
            .unwrap();
        let output = backend
            .alloc(dtype, &[query_heads, query_length, value_width])
            .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 1.0,
                    causal: false,
                    q_start: 0,
                },
                &[&query, &key, &value],
                &output,
            )
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn assert_registry_implies_encoder_support(dispatch: &Dispatch) {
        for definition in REGISTRY
            .iter()
            .filter(|definition| definition.operation == OperationKind::of(dispatch.op()))
        {
            if resolve(dispatch, definition.name, METAL4).is_ok() {
                assert!(
                    crate::encoding::variant_supported(dispatch, definition.implementation)
                        .unwrap(),
                    "{} accepted a dispatch its encoder gate refused",
                    definition.name
                );
            }
        }
    }

    fn float_dtype(index: usize) -> DType {
        [DType::F32, DType::F16, DType::BF16][index % 3]
    }

    #[test]
    fn registry_names_and_guidance_are_stable_and_complete() {
        let mut names = HashSet::new();
        for variant in REGISTRY {
            assert!(names.insert(variant.name), "duplicate {}", variant.name);
            assert!(variant.name.len() <= 64);
            assert!(variant.name.bytes().all(|byte| byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'-')));
            assert!(!variant.use_when.is_empty(), "{} use_when", variant.name);
            assert!(
                !variant.avoid_when.is_empty(),
                "{} avoid_when",
                variant.name
            );
            assert!(!variant.notes.is_empty(), "{} notes", variant.name);
        }
    }

    #[test]
    fn catalog_order_matches_registry_order() {
        let markdown = markdown("Test GPU");
        let json = json("Test GPU");
        let mut markdown_position = 0;
        let mut json_position = 0;
        for variant in REGISTRY {
            let next_markdown = markdown[markdown_position..].find(variant.name).unwrap();
            let next_json = json[json_position..].find(variant.name).unwrap();
            markdown_position += next_markdown + variant.name.len();
            json_position += next_json + variant.name.len();
        }
        assert!(!json.contains("schema_version"));
    }

    #[test]
    fn resolves_concrete_variants_and_reports_refusal_kinds() {
        let gemv = matmul_dispatch(1);
        let validated = resolve(&gemv, "matmul.gemv", METAL4).unwrap();
        assert_eq!(validated.name(), "matmul.gemv");
        assert_eq!(
            REGISTRY[validated.index].implementation,
            MetalVariant::MatmulGemv
        );

        assert!(matches!(
            resolve(&gemv, "matmul.missing", METAL4),
            Err(VariantError::Unknown { .. })
        ));
        assert!(matches!(
            resolve(&gemv, "top-k.partials", METAL4),
            Err(VariantError::WrongOperation { .. })
        ));
        assert!(matches!(
            resolve(&gemv, "matmul.gemv", &[]),
            Err(VariantError::Unavailable { .. })
        ));

        let matrix = matmul_dispatch(2);
        let error = resolve(&matrix, "matmul.gemv", METAL4).unwrap_err();
        assert_eq!(
            error.to_string(),
            "metal matmul variant 'matmul.gemv': constraint 'm = 1' failed (m was 2)"
        );
    }

    #[test]
    fn validates_inclusive_top_k_width_boundary() {
        assert!(resolve(&top_k_dispatch(2048), "top-k.single-k8", METAL4).is_ok());
        let error = resolve(&top_k_dispatch(2049), "top-k.single-k8", METAL4).unwrap_err();
        assert!(error.to_string().contains("width is in 8..=2048"));
        assert!(resolve(&top_k_dispatch(2049), "top-k.partials", METAL4).is_ok());
    }

    #[test]
    fn relation_boundaries_are_inclusive_and_checked() {
        assert!(relation_accepts(Relation::Range { min: 7, max: 33 }, 7));
        assert!(relation_accepts(Relation::Range { min: 7, max: 33 }, 33));
        assert!(!relation_accepts(Relation::Range { min: 7, max: 33 }, 6));
        assert!(relation_accepts(Relation::MultipleOf(8), 64));
        assert!(!relation_accepts(Relation::MultipleOf(8), 65));
        assert!(!relation_accepts(Relation::MultipleOf(0), 0));
    }

    #[test]
    fn every_variant_has_an_encoder_accepted_fixture() {
        let fixtures = [
            ("matmul.gemv", matmul_case(DType::F32, 1, 7, 33, false)),
            (
                "matmul.gemv-transposed",
                matmul_case(DType::F32, 1, 7, 33, true),
            ),
            (
                "matmul.steel-64x64x16-2x2",
                matmul_case(DType::F32, 2, 7, 33, false),
            ),
            (
                "matmul.steel-64x64x16-1x2",
                matmul_case(DType::F16, 2, 7, 33, false),
            ),
            (
                "matmul.steel-64x32x32-2x2",
                matmul_case(DType::BF16, 2, 7, 33, true),
            ),
            (
                "matmul.steel-32x64x16-1x2",
                matmul_case(DType::F16, 2, 33, 7, false),
            ),
            (
                "quant-matmul.gemv",
                quant_dispatch(DType::F32, 1, 64, false),
            ),
            (
                "quant-matmul.q8-fast-gemv",
                quant_dispatch(DType::F16, 1, 256, false),
            ),
            (
                "quant-matmul.small-m",
                quant_dispatch(DType::BF16, 7, 64, true),
            ),
            (
                "quant-matmul.tiled",
                quant_dispatch(DType::BF16, 16, 64, false),
            ),
            (
                "gather-quant-matmul.route-gemv",
                gathered_dispatch(false, 1, 1, 257),
            ),
            (
                "gather-quant-matmul.grouped",
                gathered_dispatch(false, 64, 1, 256),
            ),
            (
                "gather-quant-silu-mul.route-gemv",
                gathered_dispatch(true, 1, 1, 257),
            ),
            (
                "gather-quant-silu-mul.grouped",
                gathered_dispatch(true, 64, 1, 256),
            ),
            (
                "sdpa.decomposed",
                sdpa_dispatch(DType::F32, 1, 7, 33, 33, 7),
            ),
            (
                "sdpa.vector-single-pass",
                sdpa_dispatch(DType::F16, 2, 1, 33, 128, 128),
            ),
            (
                "sdpa.vector-two-pass",
                sdpa_dispatch(DType::BF16, 2, 1, 1024, 128, 128),
            ),
            ("sdpa.steel", sdpa_dispatch(DType::F32, 8, 2, 2, 64, 64)),
            ("top-k.single-k8", top_k_dispatch(2048)),
            ("top-k.partials", top_k_dispatch(2049)),
        ];
        assert_eq!(fixtures.len(), REGISTRY.len());
        for (name, dispatch) in fixtures {
            let definition = REGISTRY
                .iter()
                .find(|definition| definition.name == name)
                .unwrap();
            assert!(resolve(&dispatch, name, METAL4).is_ok(), "{name}");
            assert!(
                crate::encoding::variant_supported(&dispatch, definition.implementation).unwrap(),
                "{name}"
            );
        }
    }

    #[test]
    fn specialized_boundaries_match_encoder_gates() {
        for (dispatch, name, accepted) in [
            (
                quant_dispatch(DType::BF16, 15, 64, false),
                "quant-matmul.tiled",
                false,
            ),
            (
                quant_dispatch(DType::BF16, 16, 64, false),
                "quant-matmul.tiled",
                true,
            ),
            (
                quant_dispatch(DType::BF16, 16, 64, true),
                "quant-matmul.tiled",
                false,
            ),
            (
                gathered_dispatch(false, 63, 1, 256),
                "gather-quant-matmul.grouped",
                false,
            ),
            (
                gathered_dispatch(false, 64, 1, 256),
                "gather-quant-matmul.grouped",
                true,
            ),
            (
                gathered_dispatch(false, 64, 1, 257),
                "gather-quant-matmul.grouped",
                false,
            ),
            (
                sdpa_dispatch(DType::F16, 32, 1, 1023, 64, 64),
                "sdpa.vector-single-pass",
                true,
            ),
            (
                sdpa_dispatch(DType::F16, 33, 1, 1023, 64, 64),
                "sdpa.vector-single-pass",
                false,
            ),
            (
                sdpa_dispatch(DType::F16, 8, 2, 1023, 64, 64),
                "sdpa.vector-single-pass",
                false,
            ),
            (
                sdpa_dispatch(DType::F16, 8, 1, 1024, 64, 64),
                "sdpa.vector-two-pass",
                true,
            ),
            (top_k_dispatch(2048), "top-k.single-k8", true),
            (top_k_dispatch(2049), "top-k.single-k8", false),
        ] {
            assert_eq!(resolve(&dispatch, name, METAL4).is_ok(), accepted, "{name}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn accepted_registry_variants_pass_real_encoder_gates(
            family in 0_usize..5,
            dtype in 0_usize..3,
            size in prop::sample::select(vec![1_u32, 7, 16, 33]),
            odd in prop::sample::select(vec![7_u32, 33, 64, 128, 256, 1024]),
            alternate in any::<bool>(),
        ) {
            let dtype = float_dtype(dtype);
            let dispatch = match family {
                0 => matmul_case(dtype, size, odd, 33, alternate),
                1 => quant_dispatch(dtype, size, if odd < 64 { 64 } else { odd }, alternate),
                2 => gathered_dispatch(alternate, size, if alternate { 7 } else { 1 }, if odd == 256 { 257 } else { odd }),
                3 => sdpa_dispatch(dtype, (size % 32).max(1), if alternate { 1 } else { 7 }, odd.max(33), 64, 64),
                _ => top_k_dispatch(odd.max(8)),
            };
            assert_registry_implies_encoder_support(&dispatch);
        }
    }
}
