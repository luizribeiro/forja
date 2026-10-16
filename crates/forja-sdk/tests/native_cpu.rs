#![cfg(feature = "native")]

//! CPU-backed native SDK coverage.

use crate::support;

#[forja_sdk::kernel(map)]
fn paired_silu(
    first: forja_sdk::kernel::Elem,
    second: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    first * first.sigmoid() * second
}

#[forja_sdk::kernel(map)]
fn add_residual(
    accumulator: forja_sdk::kernel::Elem,
    residual: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    accumulator + residual
}

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
fn paired_gathered_projections_route_through_the_sdk() {
    use forja_sdk::{DType, Tensor, bf16};

    let input = Tensor::from_slice(&[1.0_f32; 64], &[1, 64]).unwrap();
    let first_packed = Tensor::from_slice(&[0x1111_1111_u32; 16], &[1, 2, 8]).unwrap();
    let first_scales =
        Tensor::from_slice(&[bf16::from_f32(0.5), bf16::from_f32(0.25)], &[1, 2, 1]).unwrap();
    let first_biases =
        Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-1.0)], &[1, 2, 1]).unwrap();
    let second_packed = Tensor::from_slice(&[0x2222_2222_u32; 16], &[1, 2, 8]).unwrap();
    let second_scales =
        Tensor::from_slice(&[bf16::from_f32(0.25), bf16::from_f32(0.5)], &[1, 2, 1]).unwrap();
    let second_biases =
        Tensor::from_slice(&[bf16::from_f32(0.0), bf16::from_f32(1.0)], &[1, 2, 1]).unwrap();
    let indices = Tensor::from_slice(&[0_u32], &[1, 1]).unwrap();
    let epilogue = paired_silu_epilogue(2, 3, &[], DType::F32).unwrap();
    let output = input
        .gather_quant_matmul_pair_epilogue(
            [
                (&first_packed, &first_scales, &first_biases),
                (&second_packed, &second_scales, &second_biases),
            ],
            &indices,
            4,
            64,
            &epilogue,
            &[],
        )
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = [96.0_f32, -48.0]
        .into_iter()
        .zip([32.0_f32, 128.0])
        .map(|(first, second)| first / (1.0 + (-first).exp()) * second)
        .collect::<Vec<_>>();
    for (actual, expected) in output.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
}

#[test]
fn weighted_gathered_projection_routes_through_the_sdk() {
    use forja_sdk::{DType, Tensor, bf16, kernel::TensorRef};

    let input = Tensor::from_slice(&[1.0_f32; 128], &[1, 2, 64]).unwrap();
    let packed = Tensor::from_slice(&[0x1111_1111_u32; 16], &[1, 2, 8]).unwrap();
    let scales =
        Tensor::from_slice(&[bf16::from_f32(0.5), bf16::from_f32(0.25)], &[1, 2, 1]).unwrap();
    let biases =
        Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-1.0)], &[1, 2, 1]).unwrap();
    let indices = Tensor::from_slice(&[0_u32, 0], &[1, 2]).unwrap();
    let weights = Tensor::from_slice(&[0.25_f32, 0.75], &[1, 2]).unwrap();
    let residual = Tensor::from_slice(&[3.0_f32, -4.0], &[1, 2]).unwrap();
    let epilogue = add_residual_epilogue(1, 2, &[DType::F32], DType::F32).unwrap();
    let output = input
        .gather_quant_matmul_weighted_epilogue(
            &packed,
            &scales,
            &biases,
            &indices,
            &weights,
            4,
            64,
            &epilogue,
            &[TensorRef::new(&residual).unwrap()],
        )
        .unwrap();

    assert_eq!(output.to_vec().unwrap(), [99.0, -52.0]);
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

    let (weights, indices) = nn::moe_router(&logits, 2, false).unwrap();
    assert_eq!(indices.to_vec().unwrap(), [2, 1]);
    let weights = weights.to_vec().unwrap();
    assert!((weights[0] - 4.0 / 7.0).abs() < 1.0e-6);
    assert!((weights[1] - 2.0 / 7.0).abs() < 1.0e-6);

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
