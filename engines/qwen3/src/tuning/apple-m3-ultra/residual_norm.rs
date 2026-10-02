use forja_sdk::{DType, Result, Tuning, kernel::Kernel, nn::blocks::residual_norm_kernel};

pub(crate) const META: Tuning = Tuning {
    name: "residual-norm",
    slot: "residual-norm",
    device_family: "apple-m3-ultra",
    shape_class: "qwen3-hidden-1024",
    description: "Fuse residual addition with the following RMS normalization",
    rationale: "Avoids one materialized residual and one dispatch between decoder sublayers",
};

pub(crate) fn kernel(dtype: DType) -> Result<Kernel> {
    residual_norm_kernel(dtype, crate::RMS_EPSILON)
}
