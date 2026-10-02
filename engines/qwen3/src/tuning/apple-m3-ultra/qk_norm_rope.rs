use forja_sdk::{DType, Result, Tuning, kernel::Kernel, nn::blocks::qk_norm_rope_kernel};

pub(crate) const META: Tuning = Tuning {
    name: "qk-norm-rope",
    slot: "qk-norm-rope",
    device_family: "apple-m3-ultra",
    shape_class: "qwen3-head-128",
    description: "Fuse query/key RMS normalization with rotary embedding",
    rationale: "Reuses each head row while it is resident and removes intermediate tensors",
};

pub(crate) fn kernel(dtype: DType) -> Result<Kernel> {
    qk_norm_rope_kernel(dtype, crate::ROPE_THETA)
}
