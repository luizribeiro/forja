use forja_core::{Backend, CommandList, DType, Op, StateOp, Submission};
use forja_testing::{DeterministicValues, generated_tensor_bytes, state_test_metadata};

#[test]
fn metal_state_authentication_preserves_canaries_and_ignores_poisoned_lanes() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for delta in [false, true] {
        let (access, metadata) = state_test_metadata().unwrap();
        let shapes: &[&[u32]] = if delta {
            &[&[1, 7], &[1, 7], &[1, 33], &[1], &[1], &[1, 7, 33]]
        } else {
            &[&[7, 33], &[4, 33], &[3, 33]]
        };
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
        let output_shape = if delta { vec![1, 33] } else { vec![7, 33] };
        let y = backend.alloc(DType::F32, &output_shape).unwrap();
        let next = backend.alloc(DType::F32, shapes[shapes.len() - 1]).unwrap();
        let before_y = vec![0x5a; usize::try_from(y.layout().element_count() * 4).unwrap()];
        let before_state = vec![0x5a; usize::try_from(next.layout().element_count() * 4).unwrap()];
        backend.write(&y, &before_y).unwrap();
        backend.write(&next, &before_state).unwrap();
        let metadata_start = shapes.len();
        let run = || {
            let operation = if delta {
                StateOp::DeltaRule(access)
            } else {
                StateOp::CausalDepthwiseConv(access)
            };
            let mut commands = CommandList::new();
            commands
                .dispatch_many(
                    Op::State(operation),
                    &inputs.iter().collect::<Vec<_>>(),
                    &[&y, &next],
                )
                .unwrap();
            backend.submit(commands).unwrap().wait()
        };
        backend
            .write(&inputs[metadata_start + 1], &[0; 12])
            .unwrap();
        assert!(run().is_err());
        assert_eq!(backend.read(&y).unwrap(), before_y);
        assert_eq!(backend.read(&next).unwrap(), before_state);
        backend.write(&inputs[metadata_start], &[0xff; 16]).unwrap();
        assert!(run().is_err());
        assert_eq!(backend.read(&y).unwrap(), before_y);
        assert_eq!(backend.read(&next).unwrap(), before_state);
        backend.write(&inputs[metadata_start + 2], &[0; 4]).unwrap();
        run().unwrap();
        assert_eq!(backend.read(&y).unwrap(), vec![0; before_y.len()]);
        assert_eq!(backend.read(&next).unwrap(), before_state);
    }
}
