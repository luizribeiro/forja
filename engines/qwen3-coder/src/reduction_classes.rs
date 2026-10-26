pub(crate) const SERVING_REDUCTION_CLASSES: &[(&str, &str)] = &[
    ("quant-matmul.gemv", "quant-matmul.row-gemv"),
    ("quant-matmul.gemv-n2", "quant-matmul.row-gemv"),
    ("quant-matmul.row-gemv", "quant-matmul.row-gemv"),
    (
        "gather-quant-matmul.route-gemv",
        "gather-quant-matmul.route-gemv",
    ),
    ("quant-matmul.small-m", "quant-matmul.small-m"),
    ("quant-matmul.q8-fast-gemv", "quant-matmul.q8-fast-gemv"),
    ("quant-matmul.q8-fast-gemv-n2", "quant-matmul.q8-fast-gemv"),
    (
        "gather-quant-matmul.grouped-small-m",
        "gather-quant-matmul.grouped-small-m",
    ),
    (
        "gather-quant-matmul.grouped-pair-small-m",
        "gather-quant-matmul.grouped-pair-small-m",
    ),
    ("quant-matmul.tiled", "quant-matmul.tiled"),
    ("gather-quant-matmul.grouped", "gather-quant-matmul.grouped"),
    (
        "gather-quant-matmul.grouped-32x32-2x2",
        "gather-quant-matmul.grouped-32x32-2x2",
    ),
    (
        "gather-quant-matmul.grouped-32x32-2x2-blocked",
        "gather-quant-matmul.grouped-32x32-2x2-blocked",
    ),
    (
        "gather-quant-matmul.grouped-pair-32x32-2x2-blocked",
        "gather-quant-matmul.grouped-pair-32x32-2x2-blocked",
    ),
    ("matmul.column-gemv", "matmul.column-gemv"),
    ("sdpa.steel", "sdpa.steel"),
    ("sdpa.steel-half", "sdpa.steel-half"),
    ("paged-sdpa.steel", "paged-sdpa.steel"),
    ("paged-sdpa.steel-half-32x16", "paged-sdpa.steel-half-32x16"),
    ("paged-sdpa.steel-half-64x16", "paged-sdpa.steel-half-64x16"),
];

pub(crate) fn serving_reduction_class(name: &str) -> Option<&'static str> {
    SERVING_REDUCTION_CLASSES
        .iter()
        .find_map(|&(variant, class)| (variant == name).then_some(class))
}
