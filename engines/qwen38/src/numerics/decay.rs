use forja_sdk::{Result, Tensor};

pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
pub(crate) fn decay(
    a: &Tensor<f32>,
    bias: &Tensor<f32>,
    log_a: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    let values = a
        .to_vec()?
        .into_iter()
        .zip(bias.to_vec()?)
        .zip(log_a.to_vec()?)
        .map(|((a, bias), log_a)| (-log_a.exp() * softplus(a + bias)).exp())
        .collect::<Vec<_>>();
    Tensor::from_slice(&values, a.shape())
}
