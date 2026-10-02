use forja_core::{
    Axis, Constraint, DeviceCapability, Dimension, Guarantee, LayoutClass, Lifecycle,
    OperationKind, OperationValue, Relation, TensorRef, ValueRef, VariantDef, render_catalog_json,
    render_catalog_markdown,
};

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

pub(crate) fn markdown(device: &str) -> String {
    render_catalog_markdown("metal", device, METAL4, REGISTRY)
}

pub(crate) fn json(device: &str) -> String {
    render_catalog_json("metal", device, METAL4, REGISTRY)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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
}
