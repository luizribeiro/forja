use forja_core::{DType, Op, StateOp};
use forja_testing::{TensorSpec, assert_backends_agree_many, state_test_metadata};

#[test]
fn metal_delta_rule_matches_cpu_with_odd_and_grouped_heads() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for (heads, q_heads, keys, width) in [
        (6, 2, 1, 7),
        (6, 2, 7, 33),
        (6, 2, 33, 1),
        (48, 16, 128, 128),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (access, metadata) = state_test_metadata().unwrap();
            let mut inputs = vec![
                TensorSpec::permuted(DType::F32, &[keys, q_heads], &[1, 0]),
                TensorSpec::permuted(DType::F32, &[keys, q_heads], &[1, 0]),
                TensorSpec::contiguous(dtype, &[heads, width]),
                TensorSpec::contiguous(DType::F32, &[heads]),
                TensorSpec::contiguous(dtype, &[heads]),
                TensorSpec::contiguous(DType::F32, &[heads, keys, width]),
            ];
            inputs.extend(metadata);
            assert_backends_agree_many(
                &cpu,
                &metal,
                Op::State(StateOp::DeltaRule(access)),
                &inputs,
                &[
                    TensorSpec::contiguous(dtype, &[heads, width]),
                    TensorSpec::contiguous(DType::F32, &[heads, keys, width]),
                ],
            )
            .unwrap();
        }
    }
}
