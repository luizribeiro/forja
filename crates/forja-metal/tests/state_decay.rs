use forja_core::{DType, Op, StateOp};
use forja_testing::{TensorSpec, assert_backends_agree_many, state_test_metadata};

#[test]
fn metal_delta_rule_matches_cpu_for_zero_state_and_decay_underflow() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    let initialized = |shape: &[u32], value: f32| {
        let count = shape.iter().map(|&n| u64::from(n)).product::<u64>();
        TensorSpec::initialized(
            DType::F32,
            shape,
            value.to_le_bytes().repeat(usize::try_from(count).unwrap()),
        )
    };
    for (initial_state, decay, beta) in [
        (0., 1., 0.),
        (1e-30, 1e-20, 0.),
        (1., 0., 0.),
        (0., 1., 0.25),
        (1e-30, 1e-20, 0.25),
        (1., 0., 0.25),
    ] {
        let (access, metadata) = state_test_metadata().unwrap();
        let mut inputs = vec![
            TensorSpec::contiguous(DType::F32, &[2, 128]),
            TensorSpec::contiguous(DType::F32, &[2, 128]),
            TensorSpec::contiguous(DType::F32, &[6, 33]),
            initialized(&[6], decay),
            initialized(&[6], beta),
            initialized(&[6, 128, 33], initial_state),
        ];
        inputs.extend(metadata);
        assert_backends_agree_many(
            &cpu,
            &metal,
            Op::State(StateOp::DeltaRule(access)),
            &inputs,
            &[
                TensorSpec::contiguous(DType::F32, &[6, 33]),
                TensorSpec::contiguous(DType::F32, &[6, 128, 33]),
            ],
        )
        .unwrap();
    }
}
