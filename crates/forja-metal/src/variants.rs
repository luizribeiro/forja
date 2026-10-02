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
