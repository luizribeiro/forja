#![cfg(feature = "native")]

//! CPU-backed native SDK coverage.

mod support;

#[test]
fn structured_ops_match_the_cpu_backend() {
    support::matmul_rope_embedding_match_cpu();
}

#[test]
fn attention_and_cache_copy_match_the_cpu_backend() {
    support::attention_and_cache_copy_match_cpu();
}

#[test]
fn neural_network_modules_match_the_cpu_backend() {
    support::neural_network_modules_match_cpu();
}

#[test]
fn refusals_and_execution_failures_surface_as_results() {
    let source = forja_sdk::Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
    let mut alias = source.narrow(0, 0, 2).unwrap();
    assert!(source.copy_into(&mut alias).is_err());

    let table = forja_sdk::Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let invalid_ids = forja_sdk::Tensor::from_slice(&[2_u32], &[1]).unwrap();
    let output = table.embedding(&invalid_ids).unwrap();
    assert!(output.to_vec().is_err());
}

#[test]
fn sampling_accepts_live_parameters_and_rejects_invalid_values() {
    let logits = forja_sdk::Tensor::from_slice(&[1.0_f32, 3.0, 3.0, 2.0], &[1, 4]).unwrap();
    let greedy =
        forja_sdk::Tensor::from_slice(&[0.0_f32.to_bits(), 0, 1.0_f32.to_bits(), 7, 0], &[5])
            .unwrap();
    assert_eq!(logits.sample(&greedy, 9).unwrap().to_vec().unwrap(), [2]);

    let invalid =
        forja_sdk::Tensor::from_slice(&[f32::NAN.to_bits(), 0, 1.0_f32.to_bits(), 7, 0], &[5])
            .unwrap();
    assert!(logits.sample(&invalid, 9).unwrap().to_vec().is_err());
}

#[test]
fn affine_quantized_matmul_routes_through_the_sdk() {
    use forja_sdk::{Tensor, bf16};

    let input = Tensor::from_slice(&[1.0_f32; 64], &[1, 64]).unwrap();
    let packed = Tensor::from_slice(&[0x1111_1111_u32; 16], &[2, 8]).unwrap();
    let scales = Tensor::from_slice(&[bf16::from_f32(0.5), bf16::from_f32(0.25)], &[2, 1]).unwrap();
    let biases = Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-1.0)], &[2, 1]).unwrap();

    let output = input
        .quant_matmul(&packed, &scales, &biases, 4, 64)
        .unwrap();
    assert_eq!(output.to_vec().unwrap(), [96.0, -48.0]);
}

#[test]
fn moe_helpers_route_and_combine_experts() {
    use forja_sdk::{Tensor, nn};

    let logits = Tensor::from_slice(&[0.0_f32, 2.0_f32.ln(), 4.0_f32.ln()], &[1, 3]).unwrap();
    let (weights, indices) = nn::moe_router(&logits, 2, true).unwrap();
    assert_eq!(indices.to_vec().unwrap(), [2, 1]);
    let weights = weights.to_vec().unwrap();
    assert!((weights[0] - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((weights[1] - 1.0 / 3.0).abs() < 1.0e-6);

    let experts = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 5.0, 6.0, 7.0], &[1, 2, 3]).unwrap();
    let route_weights = Tensor::from_slice(&[0.25_f32, 0.75], &[1, 2]).unwrap();
    assert_eq!(
        nn::moe_combine(&experts, &route_weights)
            .unwrap()
            .to_vec()
            .unwrap(),
        [4.0, 5.0, 6.0]
    );
}
