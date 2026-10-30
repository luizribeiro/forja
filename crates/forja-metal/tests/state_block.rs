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

#[test]
fn metal_delta_block_profile_accounts_for_four_stages_once() {
    use forja_core::{Backend, CommandList, KernelCost, Submission};
    use forja_testing::{DeterministicValues, generated_tensor_bytes};
    let backend = forja_metal::MetalBackend::new().unwrap();
    let (access, metadata) = state_test_metadata().unwrap();
    let shapes: &[&[u32]] = &[
        &[7, 1, 7],
        &[7, 1, 7],
        &[7, 3, 33],
        &[7, 3],
        &[7, 3],
        &[3, 7, 33],
    ];
    let mut inputs = shapes
        .iter()
        .map(|shape| backend.alloc(DType::F32, shape).unwrap())
        .collect::<Vec<_>>();
    for spec in metadata {
        let tensor = backend
            .alloc(spec.dtype(), spec.allocation_shape())
            .unwrap();
        backend
            .write(
                &tensor,
                &generated_tensor_bytes(&spec, &mut DeterministicValues::new(1)).unwrap(),
            )
            .unwrap();
        inputs.push(tensor);
    }
    let output = backend.alloc(DType::F32, &[7, 3, 33]).unwrap();
    let state = backend.alloc(DType::F32, &[3, 7, 33]).unwrap();
    let scratch = backend
        .alloc(
            DType::F32,
            &[DeltaBlockScratch::new(7, 3, 7, 33).unwrap().elements()],
        )
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::DeltaRuleBlock(access)),
            &inputs.iter().collect::<Vec<_>>(),
            &[&output, &state, &scratch],
        )
        .unwrap();
    let expected = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    let submission = backend.submit_profiled(commands).unwrap();
    submission.wait().unwrap();
    let profile = submission.profile().unwrap();
    let dispatch = &profile.per_dispatch[0];
    assert_eq!(dispatch.physical_dispatches, 4);
    assert_eq!(dispatch.internal.len(), 4);
    assert_eq!(
        dispatch
            .internal
            .iter()
            .map(|stage| stage.cost.unwrap().read_bytes)
            .sum::<u64>(),
        expected.read_bytes
    );
    assert_eq!(
        dispatch
            .internal
            .iter()
            .map(|stage| stage.cost.unwrap().write_bytes)
            .sum::<u64>(),
        expected.write_bytes
    );
    let duration = dispatch
        .internal
        .iter()
        .map(|stage| stage.gpu_time)
        .sum::<std::time::Duration>();
    assert!(duration.abs_diff(dispatch.gpu_time) <= std::time::Duration::from_nanos(2));
}
