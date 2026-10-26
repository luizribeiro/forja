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
];

pub(crate) fn serving_reduction_class(name: &str) -> Option<&'static str> {
    SERVING_REDUCTION_CLASSES
        .iter()
        .find_map(|&(variant, class)| (variant == name).then_some(class))
}
