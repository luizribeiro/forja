use forja_sdk::{DType, Result, Tuning, kernel::Kernel};

pub(crate) const META: Tuning = Tuning {
    name: "silu-mul",
    slot: "silu-mul",
    device_family: "apple-m3-ultra",
    shape_class: "qwen3-intermediate-3072",
    description: "Fuse SiLU activation with gated multiplication",
    rationale: "Eliminates the activation intermediate in every MLP",
};

pub(crate) fn kernel(dtype: DType) -> Result<Kernel> {
    silu_mul_program(2, &[dtype; 2], &[dtype])
}

#[forja_sdk::kernel(map)]
fn silu_mul(gate: forja_sdk::kernel::Elem, up: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    gate * gate.sigmoid() * up
}
