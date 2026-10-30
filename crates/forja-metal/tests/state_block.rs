use forja_core::{DType, DeltaBlockScratch, Op, StateOp};
use forja_testing::{TensorSpec, assert_backends_agree_many_selected, state_test_metadata};

fn filled(shape: &[u32], value: f32) -> TensorSpec {
    let count = shape.iter().map(|&item| u64::from(item)).product::<u64>();
    TensorSpec::initialized(
        DType::F32,
        shape,
        value.to_le_bytes().repeat(usize::try_from(count).unwrap()),
    )
}

fn compare(tokens: u32, heads: u32, query_heads: u32, keys: u32, values: u32, initial: f32) {
    let (access, metadata) = state_test_metadata().unwrap();
    let scratch = DeltaBlockScratch::new(tokens, heads, keys, values).unwrap();
    let mut inputs = vec![
        TensorSpec::permuted(DType::F32, &[keys, query_heads, tokens], &[2, 1, 0]),
        TensorSpec::permuted(DType::F32, &[keys, query_heads, tokens], &[2, 1, 0]),
        TensorSpec::contiguous(DType::F32, &[tokens, heads, values]),
        filled(&[tokens, heads], -0.015),
        filled(&[tokens, heads], 0.25),
        filled(&[heads, keys, values], initial),
    ];
    inputs.extend(metadata);
    assert_backends_agree_many_selected(
        &forja_cpu::CpuBackend::new(),
        &forja_metal::MetalBackend::new().unwrap(),
        Op::State(StateOp::DeltaRuleBlock(access)),
        &inputs,
        &[
            TensorSpec::contiguous(DType::F32, &[tokens, heads, values]),
            TensorSpec::contiguous(DType::F32, &[heads, keys, values]),
            TensorSpec::contiguous(DType::F32, &[scratch.elements()]),
        ],
        &[0, 1],
    )
    .unwrap();
}

#[test]
fn metal_delta_block_matches_cpu_on_odd_and_partial_blocks() {
    for tokens in [1, 7, 33, 63, 64, 65] {
        for initial in [0.0, 0.125] {
            compare(tokens, 6, 2, 7, 33, initial);
        }
    }
}
