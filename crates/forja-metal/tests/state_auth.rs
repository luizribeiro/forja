use forja_core::{Backend, CommandList, DType, Op, StateOp, Submission};
use forja_testing::{DeterministicValues, generated_tensor_bytes, state_test_metadata};

#[test]
fn metal_state_authentication_preserves_canaries_and_ignores_poisoned_lanes() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for kind in 0..5 {
        let delta = kind != 0;
        let block = kind >= 2;
        let (access, metadata) = state_test_metadata().unwrap();
        let shapes = input_shapes(kind);
        let mut inputs = shapes
            .iter()
            .map(|shape| backend.alloc(DType::F32, shape).unwrap())
            .collect::<Vec<_>>();
        for spec in &metadata {
            let tensor = backend
                .alloc(spec.dtype(), spec.allocation_shape())
                .unwrap();
            backend
                .write(
                    &tensor,
                    &generated_tensor_bytes(spec, &mut DeterministicValues::new(1)).unwrap(),
                )
                .unwrap();
            inputs.push(tensor);
        }
        let output_shape = if block {
            vec![7, 1, 33]
        } else if delta {
            vec![1, 33]
        } else {
            vec![7, 33]
        };
        let y = backend.alloc(DType::F32, &output_shape).unwrap();
        let next = backend.alloc(DType::F32, shapes[shapes.len() - 1]).unwrap();
        let before_y = vec![0x5a; usize::try_from(y.layout().element_count() * 4).unwrap()];
        let before_state = vec![0x5a; usize::try_from(next.layout().element_count() * 4).unwrap()];
        backend.write(&y, &before_y).unwrap();
        backend.write(&next, &before_state).unwrap();
        let scratch = block.then(|| {
            backend
                .alloc(
                    DType::F32,
                    &[forja_core::DeltaBlockScratch::new(7, 1, 7, 33)
                        .unwrap()
                        .elements()],
                )
                .unwrap()
        });
        let before_scratch = scratch.as_ref().map(|tensor| {
            vec![0x5a; usize::try_from(tensor.layout().element_count() * 4).unwrap()]
        });
        if let Some(tensor) = &scratch {
            backend
                .write(tensor, before_scratch.as_ref().unwrap())
                .unwrap();
        }
        let mut outputs = vec![&y, &next];
        outputs.extend(scratch.as_ref());
        let assert_private_canaries = || {
            assert_eq!(backend.read(&next).unwrap(), before_state);
            if let Some(tensor) = &scratch {
                assert_eq!(
                    &backend.read(tensor).unwrap(),
                    before_scratch.as_ref().unwrap()
                );
            }
        };
        let metadata_start = shapes.len();
        backend
            .write(&inputs[metadata_start + 1], &[0; 12])
            .unwrap();
        assert!(run(&backend, access, kind, &inputs, &outputs).is_err());
        assert_eq!(backend.read(&y).unwrap(), before_y);
        assert_private_canaries();
        backend.write(&inputs[metadata_start], &[0xff; 16]).unwrap();
        assert!(run(&backend, access, kind, &inputs, &outputs).is_err());
        assert_eq!(backend.read(&y).unwrap(), before_y);
        assert_private_canaries();
        backend.write(&inputs[metadata_start + 2], &[0; 4]).unwrap();
        run(&backend, access, kind, &inputs, &outputs).unwrap();
        assert_eq!(backend.read(&y).unwrap(), vec![0; before_y.len()]);
        assert_private_canaries();
    }
}

fn input_shapes(kind: u32) -> &'static [&'static [u32]] {
    let block = kind >= 2;
    let delta = kind != 0;
    if block {
        &[
            &[7, 1, 7],
            &[7, 1, 7],
            &[7, 1, 33],
            &[7, 1],
            &[7, 1],
            &[1, 7, 33],
        ]
    } else if delta {
        &[&[1, 7], &[1, 7], &[1, 33], &[1], &[1], &[1, 7, 33]]
    } else {
        &[&[7, 33], &[4, 33], &[3, 33]]
    }
}

fn run(
    backend: &forja_metal::MetalBackend,
    access: forja_core::StateAccess,
    kind: u32,
    inputs: &[forja_core::Tensor],
    outputs: &[&forja_core::Tensor],
) -> Result<(), forja_core::BackendError> {
    let block = kind >= 2;
    let delta = kind != 0;
    let operation = if block {
        StateOp::DeltaRuleBlock(access)
    } else if delta {
        StateOp::DeltaRule(access)
    } else {
        StateOp::CausalDepthwiseConv(access)
    };
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(operation),
            &inputs.iter().collect::<Vec<_>>(),
            outputs,
        )
        .unwrap();
    let name = match kind {
        3 => Some("delta-rule-block.wy64-mma-v16"),
        4 => Some("delta-rule-block.wy64-mma-v32"),
        _ => None,
    };
    if let Some(name) = name {
        let pick = backend
            .validate_variant(commands.last_dispatch().unwrap(), name)
            .unwrap();
        assert!(commands.set_last_backend_data(forja_core::BackendDispatchData::new(pick)));
    }
    backend.submit(commands).unwrap().wait()
}
