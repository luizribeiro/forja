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

#[test]
fn metal_causal_conv_matches_cpu_with_strides_and_history_tails() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for (tokens, channels, taps) in [(1, 1, 2), (7, 33, 4), (33, 7, 7), (1, 4097, 4)] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (access, metadata) = state_test_metadata().unwrap();
            let mut inputs = vec![
                TensorSpec::permuted(dtype, &[channels, tokens], &[1, 0]),
                TensorSpec::permuted(dtype, &[channels, taps], &[1, 0]),
                TensorSpec::contiguous(dtype, &[taps - 1, channels]),
            ];
            inputs.extend(metadata);
            assert_backends_agree_many(
                &cpu,
                &metal,
                Op::State(StateOp::CausalDepthwiseConv(access)),
                &inputs,
                &[
                    TensorSpec::permuted(dtype, &[channels, tokens], &[1, 0]),
                    TensorSpec::contiguous(dtype, &[taps - 1, channels]),
                ],
            )
            .unwrap();
        }
    }
}

#[test]
fn metal_causal_conv_pins_ordered_fused_tap_bits() {
    use forja_core::Backend;
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    let (access, metadata) = state_test_metadata().unwrap();
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let mut inputs = vec![
        TensorSpec::initialized(DType::F32, &[1, 1], bytes(&[0.0])),
        TensorSpec::initialized(
            DType::F32,
            &[4, 1],
            bytes(&[1.0, 1.0 - f32::EPSILON, 0.0, 0.0]),
        ),
        TensorSpec::initialized(DType::F32, &[3, 1], bytes(&[-1.0, 1.0 + f32::EPSILON, 0.0])),
    ];
    inputs.extend(metadata);
    let mut output = None;
    forja_testing::assert_backends_agree_many_selected_with(
        &cpu,
        &metal,
        Op::State(StateOp::CausalDepthwiseConv(access)),
        &inputs,
        &[
            TensorSpec::contiguous(DType::F32, &[1, 1]),
            TensorSpec::contiguous(DType::F32, &[3, 1]),
        ],
        &[0, 1],
        |commands| {
            output = Some(commands.last_dispatch().unwrap().output().clone());
        },
    )
    .unwrap();
    assert_eq!(
        metal.read(&output.unwrap()).unwrap(),
        0xa880_0000_u32.to_le_bytes()
    );
}
