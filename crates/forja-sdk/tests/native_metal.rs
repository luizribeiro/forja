#![cfg(all(feature = "native-metal", target_os = "macos"))]

//! Metal-backed native SDK coverage.

use forja_sdk::{
    DType, NativeDevice, Slice, Tensor, set_native_device,
    target::metal::{Variant, VariantRule},
};

use crate::support;

#[forja_sdk::kernel(map)]
fn double_epilogue_value(accumulator: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    accumulator * 2.0
}

#[test]
fn metal_views_round_trip_non_contiguous_values() {
    set_native_device(NativeDevice::Metal);
    let tensor = Tensor::from_slice(&(0_u32..21).collect::<Vec<_>>(), &[3, 7]).unwrap();
    let view = tensor
        .slice(&[Slice::new(0, 3, 1).unwrap(), Slice::new(1, 3, 2).unwrap()])
        .unwrap()
        .t()
        .unwrap();

    assert_eq!(view.shape(), [3, 3]);
    assert_eq!(view.to_vec().unwrap(), [1, 8, 15, 3, 10, 17, 5, 12, 19]);
}

#[test]
fn metal_guest_rope_covers_styles_and_partial_prefixes() {
    use forja_sdk::nn::blocks::{RopeConfig, RopeStyle, rope, rope_kernel};

    set_native_device(NativeDevice::Metal);
    for (style, head_dim, rotary_dim) in [
        (RopeStyle::HalfSplit, 6, None),
        (RopeStyle::Interleaved, 7, Some(6)),
    ] {
        let config = RopeConfig {
            style,
            rotary_dim,
            theta: 10_000.0,
        };
        let values = (0..2 * head_dim)
            .map(|index| f32::from(u16::try_from(index).unwrap()) / 8.0 - 0.5)
            .collect::<Vec<_>>();
        let input = Tensor::from_slice(&values, &[2, 1, head_dim]).unwrap();
        let metal = rope(
            &rope_kernel(DType::F32, head_dim, config).unwrap(),
            &input,
            &Tensor::from_slice(&[3.0_f32, 7.0], &[2]).unwrap(),
            config,
        )
        .unwrap_or_else(|error| panic!("{style:?}: {error}"))
        .to_vec()
        .unwrap();
        let expected = match style {
            RopeStyle::HalfSplit => input
                .rope(
                    &Tensor::from_slice(&[3_u32, 7], &[2]).unwrap(),
                    config.theta,
                )
                .unwrap()
                .to_vec()
                .unwrap(),
            RopeStyle::Interleaved => interleaved_rope_reference(
                &values,
                2,
                head_dim,
                rotary_dim.unwrap(),
                &[3.0, 7.0],
                config.theta,
            ),
        };
        forja_testing::assert_f32_values_agree(&expected, &metal).unwrap();
        if let Some(rotary_dim) = rotary_dim {
            for row in 0..2 {
                let tail = usize::try_from(row * head_dim + rotary_dim).unwrap();
                assert_eq!(metal[tail].to_bits(), values[tail].to_bits());
            }
        }
    }
}

#[test]
fn metal_guest_softmax_matches_trusted_operation() {
    set_native_device(NativeDevice::Metal);
    support::guest_softmax_matches_trusted_operation();
}

#[test]
fn metal_guest_elementwise_operations_match_trusted_operations() {
    set_native_device(NativeDevice::Metal);
    support::guest_elementwise_operations_match_trusted_operations();
}

#[allow(clippy::cast_precision_loss)]
fn interleaved_rope_reference(
    values: &[f32],
    rows: u32,
    head_dim: u32,
    rotary_dim: u32,
    positions: &[f32],
    theta: f32,
) -> Vec<f32> {
    let mut output = values.to_vec();
    for row in 0..rows {
        for pair in 0..rotary_dim / 2 {
            let index = usize::try_from(row * head_dim + pair * 2).unwrap();
            let exponent = -2.0 * pair as f32 / rotary_dim as f32;
            let angle = positions[usize::try_from(row).unwrap()] * theta.powf(exponent);
            output[index] = values[index] * angle.cos() - values[index + 1] * angle.sin();
            output[index + 1] = values[index + 1] * angle.cos() + values[index] * angle.sin();
        }
    }
    output
}

#[test]
fn metal_structured_ops_match_the_cpu_backend() {
    set_native_device(NativeDevice::Metal);
    support::matmul_rope_embedding_match_cpu();
}

#[test]
fn metal_attention_and_cache_copy_match_the_cpu_backend() {
    set_native_device(NativeDevice::Metal);
    support::attention_and_cache_copy_match_cpu();
}

#[test]
fn metal_neural_network_modules_match_the_cpu_backend() {
    set_native_device(NativeDevice::Metal);
    support::neural_network_modules_match_cpu();
}

#[test]
fn explicit_metal_variant_matches_native_and_rule_syntax() {
    set_native_device(NativeDevice::Metal);
    let variant = Variant::new("matmul.gemv").unwrap();
    let left = Tensor::from_slice(&[2.0_f32], &[1, 1]).unwrap();
    let right = Tensor::from_slice(&[3.0_f32], &[1, 1]).unwrap();
    assert_eq!(
        left.matmul_with(&right, &variant)
            .unwrap()
            .to_vec()
            .unwrap(),
        [6.0]
    );

    let parameter = forja_sdk::Param::new(1..=33).unwrap();
    let low = Variant::new("sdpa.decomposed").unwrap();
    let high = Variant::new("sdpa.vector-single-pass").unwrap();
    assert!(VariantRule::new(&parameter, vec![(1..=7, low), (8..=33, high)]).is_ok());
    assert!(Variant::new("Matmul.Gemv").is_err());
}

#[test]
fn metal_epilogue_selection_is_explicit_and_rules_require_capture() {
    set_native_device(NativeDevice::Metal);
    let epilogue = double_epilogue_value_epilogue(1, 2, &[], DType::F32).unwrap();
    let left = Tensor::from_slice(&[2.0_f32], &[1, 1]).unwrap();
    let right = Tensor::from_slice(&[3.0_f32], &[1, 1]).unwrap();
    let variant = Variant::new("matmul.gemv").unwrap();
    assert_eq!(
        left.matmul_epilogue_with(&right, &epilogue, &[], &variant)
            .unwrap()
            .to_vec()
            .unwrap(),
        [12.0]
    );

    let parameter = forja_sdk::Param::new(1..=1).unwrap();
    let rule = VariantRule::new(&parameter, vec![(1..=1, variant)]).unwrap();
    let error = left
        .matmul_epilogue_with(&right, &epilogue, &[], &rule)
        .err()
        .unwrap();
    assert_eq!(
        error.to_string(),
        "Metal variant rules can only be used during capture"
    );
}
