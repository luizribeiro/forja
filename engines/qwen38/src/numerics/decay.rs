use forja_sdk::{Result, Tensor};

#[forja_sdk::kernel(map)]
pub(crate) fn softplus(
    a: forja_sdk::kernel::Elem,
    bias: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    let x = a + bias;
    if x > 20.0 { x } else { x.exp_rounded().ln_1p() }
}
#[forja_sdk::kernel(map)]
fn coefficient(
    softplus: forja_sdk::kernel::Elem,
    log_a: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    (-log_a.exp_rounded() * softplus).exp_rounded()
}
pub(crate) fn decay(
    a: &Tensor<f32>,
    bias: &Tensor<f32>,
    log_a: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    coefficient(&softplus(a, bias)?, log_a)
}
