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
