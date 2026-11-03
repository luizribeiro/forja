use forja_sdk::{Result, Tensor};

#[forja_sdk::kernel(map)]
pub(crate) fn softplus(
    a: forja_sdk::kernel::Elem,
    bias: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    softplus_value(a, bias)
}

#[forja_sdk::kernel(helper)]
fn softplus_value(a: f32, bias: f32) -> f32 {
    let x = a + bias;
    if x > 20.0 { x } else { x.exp_rounded().ln_1p() }
}
#[forja_sdk::kernel(map)]
fn decode_coefficient(
    a: forja_sdk::kernel::Elem,
    bias: forja_sdk::kernel::Elem,
    log_a: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    (-log_a.exp_rounded() * softplus_value(a, bias)).exp_rounded()
}
#[forja_sdk::kernel(map)]
fn log_coefficient(
    softplus: forja_sdk::kernel::Elem,
    log_a: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    -log_a.exp_rounded() * softplus
}
pub(crate) fn log_decay(
    a: &Tensor<f32>,
    bias: &Tensor<f32>,
    log_a: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    log_coefficient(&softplus(a, bias)?, log_a)
}
pub(crate) fn decay(
    a: &Tensor<f32>,
    bias: &Tensor<f32>,
    log_a: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    decode_coefficient(a, bias, log_a)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::{decode_coefficient, softplus};
    use forja_sdk::{NativeDevice, Tensor, set_native_device};

    #[forja_sdk::kernel(map)]
    fn staged_coefficient(
        softplus: forja_sdk::kernel::Elem,
        log_a: forja_sdk::kernel::Elem,
    ) -> forja_sdk::kernel::Elem {
        (-log_a.exp_rounded() * softplus).exp_rounded()
    }

    #[test]
    fn fused_decay_matches_staged_softplus_at_threshold_and_negative_inputs() {
        set_native_device(NativeDevice::Cpu);
        let a = Tensor::from_slice(
            &[
                -40.0,
                -3.0,
                -1.0,
                0.0,
                19.9,
                20.0,
                20.1,
                30.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            &[11],
        )
        .unwrap();
        let bias = Tensor::from_slice(
            &[0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[11],
        )
        .unwrap();
        let log_a = Tensor::from_slice(
            &[0.1, -1.0, 0.5, 0.0, 0.2, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0],
            &[11],
        )
        .unwrap();
        let staged = staged_coefficient(&softplus(&a, &bias).unwrap(), &log_a).unwrap();
        let fused = decode_coefficient(&a, &bias, &log_a).unwrap();
        for (actual, expected) in fused.to_vec().unwrap().iter().zip(staged.to_vec().unwrap()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
}
