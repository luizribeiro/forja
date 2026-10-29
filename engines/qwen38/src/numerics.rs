mod decay;
pub(crate) use decay::decay;
use forja_sdk::{Result, Tensor, bf16};

#[forja_sdk::kernel(map)]
fn sigmoid(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x.sigmoid()
}
#[forja_sdk::kernel(map)]
fn silu(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x * x.sigmoid()
}
#[forja_sdk::kernel(map)]
pub(crate) fn multiply(
    x: forja_sdk::kernel::Elem,
    y: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    x * y
}
#[forja_sdk::kernel(row)]
pub(crate) fn l2(x: forja_sdk::kernel::Row, scale: f32) -> forja_sdk::kernel::Row {
    x * ((x * x).row_sum() + 1.0e-6).rsqrt() * scale
}
pub(crate) fn inverse_sqrt_width(width: u32) -> Result<f32> {
    Ok(f32::from(
        u16::try_from(width).map_err(|_| forja_sdk::Error::loading("head width exceeds u16"))?,
    )
    .sqrt()
    .recip())
}
pub(crate) fn sigmoid_bf16(x: &Tensor<bf16>) -> Result<Tensor<bf16>> {
    sigmoid(x)
}
pub(crate) fn silu_bf16(x: &Tensor<bf16>) -> Result<Tensor<bf16>> {
    silu(x)
}
pub(crate) fn gated_norm(
    y: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    z: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    let ones = Tensor::constant(
        &vec![
            1.0_f32;
            usize::try_from(y.shape()[1])
                .map_err(|_| forja_sdk::Error::loading("head width exceeds address space"))?
        ],
        &[y.shape()[1]],
    )?;
    let normalized = y
        .to_dtype::<f32>()?
        .rms_norm(&ones, 1.0e-6)?
        .to_dtype::<bf16>()?;
    let weighted = multiply(&normalized, weight)?;
    multiply(&weighted.to_dtype::<f32>()?, &silu(&z.to_dtype::<f32>()?)?)?.to_dtype()
}
