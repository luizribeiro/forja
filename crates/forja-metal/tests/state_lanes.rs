use forja_core::{Backend, BackendDispatchData, CommandList, DType, Op, StateOp, Submission};
use forja_testing::{DeterministicValues, TensorSpec, generated_tensor_bytes};

#[derive(Clone, Copy, Debug)]
enum Geometry {
    Delta([u32; 4]),
    Conv([u32; 3]),
}

#[test]
fn metal_authenticated_state_lanes_match_singles_with_inactive_rows() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for geometry in [
        Geometry::Delta([48, 16, 128, 128]),
        Geometry::Conv([1, 10240, 4]),
        Geometry::Conv([1, 4097, 4]),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for lanes in [1, 2, 4, 7, 8] {
                compare(&backend, geometry, dtype, lanes, false);
            }
        }
    }
}

#[test]
fn metal_authenticated_lane_batch_refuses_stale_generation() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for geometry in [
        Geometry::Delta([48, 16, 128, 128]),
        Geometry::Conv([1, 4097, 4]),
    ] {
        compare(&backend, geometry, DType::BF16, 4, true);
    }
}

#[test]
fn metal_lane_batches_preserve_odd_keys_values_and_history_tails() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for geometry in [
        Geometry::Delta([6, 2, 7, 33]),
        Geometry::Delta([6, 2, 33, 7]),
        Geometry::Delta([1, 1, 1, 1]),
        Geometry::Conv([7, 33, 4]),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            compare(&backend, geometry, dtype, 4, false);
        }
    }
}

fn compare(
    backend: &forja_metal::MetalBackend,
    geometry: Geometry,
    dtype: DType,
    lanes: usize,
    corrupt: bool,
) {
    let delta = matches!(geometry, Geometry::Delta(_));
    let specs = specifications(geometry, dtype);
    let name = if delta {
        "delta-rule.simd-lanes"
    } else {
        "causal-depthwise-conv.direct-lanes"
    };
    let mut batch = CommandList::new();
    let mut results = Vec::new();
    for lane in 0..lanes {
        let (access, inputs) = lane_inputs(backend, &specs, lane, corrupt);
        let op = Op::State(if delta {
            StateOp::DeltaRule(access)
        } else {
            StateOp::CausalDepthwiseConv(access)
        });
        let output = backend
            .alloc(dtype, inputs[if delta { 2 } else { 0 }].layout().shape())
            .unwrap();
        let previous = &inputs[inputs.len() - 4];
        let state = backend
            .alloc(previous.layout().dtype(), previous.layout().shape())
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

fn specifications(geometry: Geometry, dtype: DType) -> Vec<TensorSpec> {
    match geometry {
        Geometry::Delta([heads, q_heads, keys, values]) => vec![
            TensorSpec::permuted(DType::F32, &[keys, q_heads], &[1, 0]),
            TensorSpec::permuted(DType::F32, &[keys, q_heads], &[1, 0]),
            TensorSpec::contiguous(dtype, &[heads, values]),
            TensorSpec::contiguous(DType::F32, &[heads]),
            TensorSpec::contiguous(dtype, &[heads]),
            TensorSpec::contiguous(DType::F32, &[heads, keys, values]),
        ],
        Geometry::Conv([tokens, channels, taps]) => vec![
            TensorSpec::contiguous(dtype, &[tokens, channels]),
            TensorSpec::permuted(dtype, &[channels, taps], &[1, 0]),
            TensorSpec::contiguous(dtype, &[taps - 1, channels]),
        ],
    }
}

fn lane_inputs(
    backend: &forja_metal::MetalBackend,
    specs: &[TensorSpec],
    lane: usize,
    corrupt: bool,
) -> (forja_core::StateAccess, Vec<forja_core::Tensor>) {
    let (access, metadata) = lane_metadata(lane);
    let mut specs = specs.to_vec();
    specs.extend(metadata);
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
            .write(
                &inputs[ordinary + 1],
                &0_u32.to_le_bytes().repeat(
                    usize::try_from(inputs[ordinary + 1].layout().element_count()).unwrap(),
                ),
            )
            .unwrap();
    }
    (access, inputs)
}

fn lane_metadata(lane: usize) -> (forja_core::StateAccess, [TensorSpec; 3]) {
    use forja_core::{StateAccess, StateLayout, StatePlane, StatePool};
    let pool = StatePool::new(
        StateLayout::new(vec![StatePlane::new(DType::F32, vec![1]).unwrap()]).unwrap(),
        8,
        4096,
    )
    .unwrap();
    let mut leases = (0..=lane)
        .map(|_| pool.reserve().unwrap())
        .collect::<Vec<_>>();
    let sequence = leases.last_mut().unwrap();
    let transaction = sequence.begin(1).unwrap();
    let slot = sequence.bind(&transaction).unwrap();
    let table = slot.table();
    let access = StateAccess::from_slot(&slot).unwrap();
    assert_eq!(usize::try_from(access.source()).unwrap() / 3, lane);
    (
        access,
        [
            TensorSpec::initialized(
                DType::U32,
                &[2, 2],
                table
                    .entries()
                    .iter()
                    .flat_map(|entry| [entry.page(), entry.generation()])
                    .flat_map(u32::to_le_bytes)
                    .collect(),
            ),
            TensorSpec::initialized(
                DType::U32,
                &[24],
                table
                    .generations()
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            TensorSpec::initialized(DType::U32, &[1], 1_u32.to_le_bytes().to_vec()),
        ],
    )
}
