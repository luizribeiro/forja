use forja_core::{Backend, BackendDispatchData, CommandList, DType, Op, StateOp, Submission};
use forja_testing::{DeterministicValues, TensorSpec, generated_tensor_bytes, state_test_metadata};

#[test]
fn metal_authenticated_state_lanes_match_singles_with_inactive_rows() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for delta in [false, true] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for lanes in [1, 2, 4, 7, 8] {
                compare(&backend, delta, dtype, lanes, false);
            }
        }
    }
}

#[test]
fn metal_authenticated_lane_batch_refuses_stale_generation() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for delta in [false, true] {
        compare(&backend, delta, DType::BF16, 4, true);
    }
}

fn compare(
    backend: &forja_metal::MetalBackend,
    delta: bool,
    dtype: DType,
    lanes: usize,
    corrupt: bool,
) {
    let (access, specs) = specifications(delta, dtype);
    let op = Op::State(if delta {
        StateOp::DeltaRule(access)
    } else {
        StateOp::CausalDepthwiseConv(access)
    });
    let name = if delta {
        "delta-rule.simd-lanes"
    } else {
        "causal-depthwise-conv.direct-lanes"
    };
    let mut batch = CommandList::new();
    let mut results = Vec::new();
    for lane in 0..lanes {
        let inputs = lane_inputs(backend, &specs, lane, corrupt);
        let output = backend
            .alloc(dtype, if delta { &[48, 128] } else { &[1, 4097] })
            .unwrap();
        let state = backend
            .alloc(
                if delta { DType::F32 } else { dtype },
                if delta { &[48, 128, 128] } else { &[3, 4097] },
            )
            .unwrap();
        let single_output = backend.alloc(dtype, output.layout().shape()).unwrap();
        let single_state = backend
            .alloc(state.layout().dtype(), state.layout().shape())
            .unwrap();
        for tensor in [&output, &state, &single_output, &single_state] {
            backend
                .write(
                    tensor,
                    &vec![
                        0x3f;
                        usize::try_from(
                            tensor.layout().element_count() * tensor.layout().dtype().byte_size()
                        )
                        .unwrap()
                    ],
                )
                .unwrap();
        }
        if !corrupt {
            let mut single = CommandList::new();
            single
                .dispatch_many(
                    op,
                    &inputs.iter().collect::<Vec<_>>(),
                    &[&single_output, &single_state],
                )
                .unwrap();
            backend.submit(single).unwrap().wait().unwrap();
        }
        batch
            .dispatch_many(op, &inputs.iter().collect::<Vec<_>>(), &[&output, &state])
            .unwrap();
        let selected = backend
            .validate_variant(batch.last_dispatch().unwrap(), name)
            .unwrap();
        assert!(batch.set_last_backend_data(BackendDispatchData::new(selected)));
        results.push((output, state, single_output, single_state));
    }
    let submission = backend.submit_profiled(batch).unwrap();
    if corrupt {
        assert!(submission.wait().is_err());
        return;
    }
    submission.wait().unwrap();
    let profile = submission
        .profile()
        .unwrap_or_else(|| panic!("missing profile delta={delta} dtype={dtype:?} lanes={lanes}"));
    assert_eq!(
        profile
            .per_dispatch
            .iter()
            .map(|p| p.physical_dispatches)
            .sum::<u64>(),
        1
    );
    for (output, state, expected, expected_state) in results {
        assert_eq!(
            backend.read(&output).unwrap(),
            backend.read(&expected).unwrap()
        );
        assert_eq!(
            backend.read(&state).unwrap(),
            backend.read(&expected_state).unwrap()
        );
    }
}

fn specifications(delta: bool, dtype: DType) -> (forja_core::StateAccess, Vec<TensorSpec>) {
    let (access, metadata) = state_test_metadata().unwrap();
    let mut specs = if delta {
        vec![
            TensorSpec::permuted(DType::F32, &[128, 16], &[1, 0]),
            TensorSpec::permuted(DType::F32, &[128, 16], &[1, 0]),
            TensorSpec::contiguous(dtype, &[48, 128]),
            TensorSpec::contiguous(DType::F32, &[48]),
            TensorSpec::contiguous(dtype, &[48]),
            TensorSpec::contiguous(DType::F32, &[48, 128, 128]),
        ]
    } else {
        vec![
            TensorSpec::contiguous(dtype, &[1, 4097]),
            TensorSpec::permuted(dtype, &[4097, 4], &[1, 0]),
            TensorSpec::contiguous(dtype, &[3, 4097]),
        ]
    };
    specs.extend(metadata);
    (access, specs)
}

fn lane_inputs(
    backend: &forja_metal::MetalBackend,
    specs: &[TensorSpec],
    lane: usize,
    corrupt: bool,
) -> Vec<forja_core::Tensor> {
    let mut values = DeterministicValues::new(0x6a09_e667 + u64::try_from(lane).unwrap());
    let inputs = specs
        .iter()
        .map(|spec| {
            let tensor = backend
                .alloc(spec.dtype(), spec.allocation_shape())
                .unwrap();
            backend
                .write(&tensor, &generated_tensor_bytes(spec, &mut values).unwrap())
                .unwrap();
            spec.views().iter().fold(tensor, |tensor, view| {
                backend.view(&tensor, view.clone()).unwrap()
            })
        })
        .collect::<Vec<_>>();
    let ordinary = inputs.len() - 3;
    if lane == 1 && !corrupt {
        backend
            .write(&inputs[ordinary + 2], &0_u32.to_le_bytes())
            .unwrap();
    }
    if lane == 2 && corrupt {
        backend
            .write(&inputs[ordinary + 1], &0_u32.to_le_bytes().repeat(3))
            .unwrap();
    }
    inputs
}
