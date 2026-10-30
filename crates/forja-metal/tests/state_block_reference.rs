use forja_core::{
    Backend, BackendDispatchData, CommandList, DType, DeltaBlockScratch, Op, StateOp, Submission,
};
use forja_testing::{
    DeterministicValues, assert_outputs_agree, generated_tensor_bytes, state_test_metadata,
};

#[test]
fn metal_multi_block_wy_matches_token_recurrence_at_realistic_widths() {
    for tokens in [65, 4097] {
        for initial in [0.0, 0.125] {
            for underflow in [false, true] {
                compare(tokens, initial, underflow);
            }
        }
    }
}

fn compare(tokens: u32, initial: f32, underflow: bool) {
    let metal = forja_metal::MetalBackend::new().unwrap();
    let count = usize::try_from(tokens).unwrap();
    let mut rng = DeterministicValues::new(73);
    let query = (0..count * 128)
        .map(|_| rng.next_f32() * 0.125)
        .collect::<Vec<_>>();
    let key = normalized_keys(&query);
    let value = (0..count * 128).map(|_| rng.next_f32()).collect::<Vec<_>>();
    let decay = (0..count)
        .map(|row| {
            if underflow && row % 97 == 93 {
                f32::NEG_INFINITY
            } else if underflow && row % 71 == 69 {
                -1000.0
            } else {
                -0.015
            }
        })
        .collect::<Vec<_>>();
    let beta = vec![0.25; count];
    let state = vec![initial; 128 * 128];
    let (expected_output, expected_state) = forja_testing::delta_rule::recurrent(
        &query,
        &key,
        &value,
        &decay,
        &beta,
        &state,
        [128, 128],
    );
    let mut inputs = Vec::new();
    for (shape, values) in [
        (&[tokens, 1, 128][..], &query),
        (&[tokens, 1, 128], &key),
        (&[tokens, 1, 128], &value),
        (&[tokens, 1], &decay),
        (&[tokens, 1], &beta),
        (&[1, 128, 128], &state),
    ] {
        let tensor = metal.alloc(DType::F32, shape).unwrap();
        metal
            .write(
                &tensor,
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        inputs.push(tensor);
    }
    let (access, metadata) = state_test_metadata().unwrap();
    for spec in metadata {
        let tensor = metal.alloc(spec.dtype(), spec.allocation_shape()).unwrap();
        metal
            .write(&tensor, &generated_tensor_bytes(&spec, &mut rng).unwrap())
            .unwrap();
        inputs.push(tensor);
    }
    let output = metal.alloc(DType::F32, &[tokens, 1, 128]).unwrap();
    let next = metal.alloc(DType::F32, &[1, 128, 128]).unwrap();
    let scratch = metal
        .alloc(
            DType::F32,
            &[DeltaBlockScratch::new(tokens, 1, 128, 128)
                .unwrap()
                .elements()],
        )
        .unwrap();
    for variant in [
        "delta-rule-block.wy64-fixed",
        "delta-rule-block.wy64-mma-v16",
        "delta-rule-block.wy64-mma-v32",
    ] {
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::State(StateOp::DeltaRuleBlock(access)),
                &inputs.iter().collect::<Vec<_>>(),
                &[&output, &next, &scratch],
            )
            .unwrap();
        let pick = metal
            .validate_variant(commands.last_dispatch().unwrap(), variant)
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(pick)));
        metal.submit(commands).unwrap().wait().unwrap();
        for (expected, tensor) in [(&expected_output, &output), (&expected_state, &next)] {
            let actual = metal.read(tensor).unwrap();
            let expected = expected
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
        }
    }
}

fn normalized_keys(query: &[f32]) -> Vec<f32> {
    let mut key = query.to_vec();
    for row in key.chunks_mut(128) {
        let norm = (row.iter().map(|value| value * value).sum::<f32>() + 1e-6).sqrt();
        for value in row {
            *value /= norm;
        }
    }
    key
}
