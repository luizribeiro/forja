mod decay;
pub(crate) use decay::{decay, log_decay};
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
    ones: &Tensor<f32>,
) -> Result<Tensor<bf16>> {
    let normalized = y
        .to_dtype::<f32>()?
        .rms_norm(ones, 1.0e-6)?
        .to_dtype::<bf16>()?;
    let weighted = multiply(&normalized, weight)?;
    multiply(&weighted.to_dtype::<f32>()?, &silu(&z.to_dtype::<f32>()?)?)?.to_dtype()
}

#[forja_sdk::kernel(map)]
fn weighted_silu(
    weighted: forja_sdk::kernel::Elem,
    z: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    weighted * (z * z.sigmoid())
}

pub(crate) fn gated_norm_decode(
    y: &Tensor<bf16>,
    weight: &Tensor<bf16>,
    z: &Tensor<bf16>,
    ones: &Tensor<bf16>,
) -> Result<Tensor<bf16>> {
    let normalized = y.rms_norm(ones, 1.0e-6)?;
    weighted_silu(&multiply(&normalized, weight)?, z)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::{gated_norm, gated_norm_decode};
    use forja_sdk::{NativeDevice, Tensor, bf16, set_native_device};

    fn compare_gated_norm(device: NativeDevice) {
        set_native_device(device);
        for (rows, width) in [
            (1, 1),
            (7, 7),
            (33, 33),
            (48, 128),
            (192, 128),
            (384, 128),
            (7, 4097),
        ] {
            let y = Tensor::from_slice(
                &(0..rows * width * 2)
                    .map(|i| bf16::from_f64(f64::from(i % 97) / 64.0 - 0.75))
                    .collect::<Vec<_>>(),
                &[rows, width * 2],
            )
            .unwrap()
            .narrow(1, 0, width)
            .unwrap();
            let z = Tensor::from_slice(
                &(0..rows * width * 2)
                    .map(|i| {
                        bf16::from_f64(if i % 17 == 0 {
                            -40.0
                        } else if i % 19 == 0 {
                            40.0
                        } else {
                            f64::from(i % 89) / 16.0 - 2.75
                        })
                    })
                    .collect::<Vec<_>>(),
                &[rows, width * 2],
            )
            .unwrap()
            .narrow(1, 0, width)
            .unwrap();
            let weight = Tensor::from_slice(
                &(0..width)
                    .map(|i| bf16::from_f64(f64::from(i % 7) / 32.0 + 0.875))
                    .collect::<Vec<_>>(),
                &[width],
            )
            .unwrap();
            let ones = Tensor::constant(&vec![1.0_f32; usize::try_from(width).unwrap()], &[width])
                .unwrap();
            let expected = gated_norm(&y, &weight, &z, &ones)
                .unwrap()
                .to_vec()
                .unwrap();
            let actual = gated_norm_decode(&y, &weight, &z, &ones.to_dtype().unwrap())
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| a.to_bits() != b.to_bits()),
                None,
                "rows={rows} width={width}"
            );
        }
    }

    #[test]
    fn cpu_decode_gated_norm_preserves_staged_bf16_bits() {
        compare_gated_norm(NativeDevice::Cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_decode_gated_norm_preserves_staged_bf16_bits() {
        compare_gated_norm(NativeDevice::Metal);
    }
}
