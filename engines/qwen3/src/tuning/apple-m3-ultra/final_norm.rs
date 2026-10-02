use forja_sdk::{DType, Result, Tuning, kernel::Kernel};

pub(crate) const META: Tuning = Tuning {
    name: "final-norm",
    slot: "final-norm",
    device_family: "apple-m3-ultra",
    shape_class: "qwen3-hidden-1024",
    description: "Use a row kernel for the final RMS normalization",
    rationale: "Keeps reduction and scaling in one dispatch before the vocabulary projection",
};

pub(crate) fn kernel(dtype: DType) -> Result<Kernel> {
    final_norm_program(2, &[dtype; 2], &[dtype])
}

#[forja_sdk::kernel(row)]
fn final_norm(
    input: forja_sdk::kernel::Row,
    weight: forja_sdk::kernel::Row,
) -> forja_sdk::kernel::Row {
    input * ((input * input).row_mean() + crate::RMS_EPSILON).rsqrt() * weight
}
