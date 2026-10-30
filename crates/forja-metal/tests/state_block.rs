use forja_core::{DType, DeltaBlockScratch, Op, StateOp};
use forja_testing::{TensorSpec, assert_backends_agree_many_selected_with, state_test_metadata};

fn filled(shape: &[u32], value: f32) -> TensorSpec {
    let count = shape.iter().map(|&item| u64::from(item)).product::<u64>();
    TensorSpec::initialized(
        DType::F32,
        shape,
        value.to_le_bytes().repeat(usize::try_from(count).unwrap()),
    )
}

fn compare(
    tokens: u32,
    heads: u32,
    query_heads: u32,
    keys: u32,
    values: u32,
    initial: f32,
    underflow: bool,
) {
    compare_using(
        [tokens, heads, query_heads, keys, values],
        initial,
        underflow,
        None,
    );
}

fn compare_using(shape: [u32; 5], initial: f32, underflow: bool, variant: Option<&str>) {
    let [tokens, heads, query_heads, keys, values] = shape;
    let (access, metadata) = state_test_metadata().unwrap();
    let scratch = DeltaBlockScratch::new(tokens, heads, keys, values).unwrap();
    let mut inputs = vec![
        TensorSpec::permuted(DType::F32, &[keys, query_heads, tokens], &[2, 1, 0]),
        TensorSpec::permuted(DType::F32, &[keys, query_heads, tokens], &[2, 1, 0]),
        TensorSpec::contiguous(DType::F32, &[tokens, heads, values]),
        TensorSpec::initialized(
            DType::F32,
            &[tokens, heads],
            (0..tokens * heads)
                .flat_map(|index| {
                    (if underflow && index / heads % 17 == 9 {
                        f32::NEG_INFINITY
                    } else {
                        -0.015_f32
                    })
                    .to_le_bytes()
                })
                .collect(),
        ),
        filled(&[tokens, heads], 0.25),
        filled(&[heads, keys, values], initial),
    ];
    inputs.extend(metadata);
    compare_specs(
        Op::State(StateOp::DeltaRuleBlock(access)),
        &inputs,
        &[
            TensorSpec::contiguous(DType::F32, &[tokens, heads, values]),
            TensorSpec::contiguous(DType::F32, &[heads, keys, values]),
            TensorSpec::contiguous(DType::F32, &[scratch.elements()]),
        ],
        variant,
    );
}

fn compare_specs(op: Op, inputs: &[TensorSpec], outputs: &[TensorSpec], variant: Option<&str>) {
    let metal = forja_metal::MetalBackend::new().unwrap();
    assert_backends_agree_many_selected_with(
        &forja_cpu::CpuBackend::new(),
        &metal,
        op,
        inputs,
        outputs,
        &[0, 1],
        |commands| {
            if let Some(name) = variant {
                let pick = metal
                    .validate_variant(commands.last_dispatch().unwrap(), name)
                    .unwrap();
                assert!(commands.set_last_backend_data(forja_core::BackendDispatchData::new(pick)));
            }
        },
    )
    .unwrap();
}

#[test]
fn metal_mma_delta_blocks_mask_future_nan_values_before_output_arithmetic() {
    for nan_row in [1, 7, 8, 31, 32, 63] {
        let (access, metadata) = state_test_metadata().unwrap();
        let mut inputs = vec![
            filled(&[64, 1, 7], 0.125),
            filled(&[64, 1, 7], 0.125),
            TensorSpec::initialized(
                DType::F32,
                &[64, 3, 33],
                (0..64 * 3 * 33)
                    .flat_map(|index| {
                        (if index / (3 * 33) == nan_row {
                            f32::NAN
                        } else {
                            1.0
                        })
                        .to_le_bytes()
                    })
                    .collect(),
            ),
            filled(&[64, 3], -0.015),
            filled(&[64, 3], 0.25),
            filled(&[3, 7, 33], 0.125),
        ];
        inputs.extend(metadata);
        let scratch = DeltaBlockScratch::new(64, 3, 7, 33).unwrap();
        let outputs = [
            TensorSpec::contiguous(DType::F32, &[64, 3, 33]),
            TensorSpec::contiguous(DType::F32, &[3, 7, 33]),
            TensorSpec::contiguous(DType::F32, &[scratch.elements()]),
        ];
        for name in [
            "delta-rule-block.wy64-mma-v16",
            "delta-rule-block.wy64-mma-v32",
        ] {
            compare_specs(
                Op::State(StateOp::DeltaRuleBlock(access)),
                &inputs,
                &outputs,
                Some(name),
            );
        }
    }
}

#[test]
fn metal_mma_delta_blocks_match_cpu_across_tails_and_underflow() {
    for name in [
        "delta-rule-block.wy64-mma-v16",
        "delta-rule-block.wy64-mma-v32",
    ] {
        for tokens in [1, 7, 33, 63, 64, 65, 4097] {
            for initial in [0.0, 0.125] {
                compare_using([tokens, 3, 1, 7, 33], initial, true, Some(name));
            }
        }
        compare_using([65, 6, 2, 128, 128], 0.125, false, Some(name));
    }
}

#[test]
fn metal_delta_block_matches_cpu_on_odd_and_partial_blocks() {
    for tokens in [1, 7, 33, 63, 64, 65] {
        for initial in [0.0, 0.125] {
            compare(tokens, 6, 2, 7, 33, initial, false);
        }
    }
}

#[test]
fn metal_delta_block_survives_underflow_and_realistic_head_widths() {
    compare(65, 6, 2, 7, 33, 0.125, true);
    compare(7, 6, 2, 128, 128, 0.125, false);
}

#[test]
fn metal_delta_block_preserves_a_long_canonical_scan() {
    compare(4097, 1, 1, 7, 7, 0.125, true);
}

#[test]
fn metal_delta_block_profile_accounts_for_four_stages_once() {
    for variant in [
        None,
        Some("delta-rule-block.wy64-mma-v16"),
        Some("delta-rule-block.wy64-mma-v32"),
    ] {
        assert_block_profile(variant);
    }
}

fn assert_block_profile(variant: Option<&str>) {
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
    if let Some(name) = variant {
        let pick = backend
            .validate_variant(commands.last_dispatch().unwrap(), name)
            .unwrap();
        assert!(commands.set_last_backend_data(forja_core::BackendDispatchData::new(pick)));
    }
    let submission = backend.submit_profiled(commands).unwrap();
    submission.wait().unwrap();
    let profile = submission.profile().unwrap();
    let dispatch = &profile.per_dispatch[0];
    assert_eq!(dispatch.physical_dispatches, 4);
    assert_eq!(dispatch.internal.len(), 4);
    let sum = dispatch
        .internal
        .iter()
        .try_fold(KernelCost::default(), |sum, stage| {
            sum.checked_add(stage.cost.unwrap())
        })
        .unwrap();
    assert_eq!(sum, dispatch.cost);
    assert_eq!(sum.flops, expected.flops);
    assert_eq!(sum.state_read_bytes, 3 * 7 * 33 * 4);
    assert_eq!(sum.state_write_bytes, sum.state_read_bytes);
    if variant.is_some() {
        assert_eq!(
            dispatch.internal[1].cost.unwrap().scratch_read_bytes,
            4096 * 4 * 3 * 2 + 21 * 7 * 4
        );
        assert_ne!(sum.read_bytes, expected.read_bytes);
        assert!(dispatch.padded_flops[0] > sum.flops);
        assert_eq!(dispatch.padded_flops[0], dispatch.padded_flops[1]);
    } else {
        assert_eq!(sum, expected);
    }
    let duration = dispatch
        .internal
        .iter()
        .map(|stage| stage.gpu_time)
        .sum::<std::time::Duration>();
    assert!(duration.abs_diff(dispatch.gpu_time) <= std::time::Duration::from_nanos(2));
}
