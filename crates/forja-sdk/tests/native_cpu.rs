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
fn refusals_and_execution_failures_surface_as_results() {
    let source = forja_sdk::Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
    let mut alias = source.narrow(0, 0, 2).unwrap();
    assert!(source.copy_into(&mut alias).is_err());

    let table = forja_sdk::Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let invalid_ids = forja_sdk::Tensor::from_slice(&[2_u32], &[1]).unwrap();
    let output = table.embedding(&invalid_ids).unwrap();
    assert!(output.to_vec().is_err());
}
