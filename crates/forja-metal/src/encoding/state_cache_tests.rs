use super::reusable_dispatch;
use forja_core::{CommandList, DType, Op, StateOp};

#[test]
fn state_replay_requires_fresh_authentication_flags() {
    let backend = forja_cpu::CpuBackend::new();
    let (access, _) = forja_testing::state_test_metadata().unwrap();
    let shapes = [
        (DType::F32, vec![1, 1]),
        (DType::F32, vec![2, 1]),
        (DType::F32, vec![1, 1]),
        (DType::U32, vec![2, 2]),
        (DType::U32, vec![3]),
        (DType::U32, vec![1]),
    ];
    let inputs = shapes
        .iter()
        .map(|(dtype, shape)| backend.alloc(*dtype, shape).unwrap())
        .collect::<Vec<_>>();
    let y = backend.alloc(DType::F32, &[1, 1]).unwrap();
    let next = backend.alloc(DType::F32, &[1, 1]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::CausalDepthwiseConv(access)),
            &inputs.iter().collect::<Vec<_>>(),
            &[&y, &next],
        )
        .unwrap();
    assert!(!reusable_dispatch(commands.last_dispatch().unwrap()));
}

#[test]
fn protected_replay_bindings_replace_both_banks_without_new_parameters() {
    use forja_core::{Backend, GraphLimits, GraphTemplate, ParamSpace, ReplayBindings, Submission};
    let backend = crate::MetalBackend::new().unwrap();
    let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
    let right = backend.alloc(DType::F32, &[7, 33]).unwrap();
    let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
    let other_output = backend.alloc(DType::F32, &[7, 33]).unwrap();
    backend
        .write(&left, &1_f32.to_le_bytes().repeat(231))
        .unwrap();
    backend
        .write(&right, &2_f32.to_le_bytes().repeat(231))
        .unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&left], &output).unwrap();
    let mut template =
        GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
    template
        .record_validated(commands.into_dispatches().remove(0))
        .unwrap();
    let graph = backend.prepare_graph(template).unwrap();
    let mut bindings = ReplayBindings::default();
    bindings.buffer(left.buffer(), right.buffer()).unwrap();
    bindings
        .buffer(output.buffer(), other_output.buffer())
        .unwrap();
    for profiled in [false, true] {
        let owner = std::sync::Arc::new(());
        let released = std::sync::Arc::downgrade(&owner);
        backend
            .replay_bound_retained(
                &graph,
                Vec::new(),
                &bindings,
                vec![forja_core::CompletionResource::new(owner)],
                profiled,
            )
            .unwrap()
            .wait_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert!(released.upgrade().is_none());
        assert_eq!(
            backend.read(&other_output).unwrap(),
            2_f32.to_le_bytes().repeat(231)
        );
        backend
            .replay(&graph, Vec::new())
            .unwrap()
            .wait_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert_eq!(
            backend.read(&output).unwrap(),
            1_f32.to_le_bytes().repeat(231)
        );
    }
}

#[test]
fn drained_failed_state_outputs_retire_without_authorizing_reads() {
    use forja_core::{Backend, BackendError, Submission};
    let backend = crate::MetalBackend::new().unwrap();
    let (access, _) = forja_testing::state_test_metadata().unwrap();
    let inputs = [
        (DType::F32, vec![1, 7]),
        (DType::F32, vec![2, 7]),
        (DType::F32, vec![1, 7]),
        (DType::U32, vec![2, 2]),
        (DType::U32, vec![3]),
        (DType::U32, vec![1]),
    ]
    .into_iter()
    .map(|(dtype, shape)| backend.alloc(dtype, &shape).unwrap())
    .collect::<Vec<_>>();
    backend
        .write(&inputs[3], &u32::MAX.to_le_bytes().repeat(4))
        .unwrap();
    backend.write(&inputs[5], &1_u32.to_le_bytes()).unwrap();
    let output = backend.alloc(DType::F32, &[1, 7]).unwrap();
    let next = backend.alloc(DType::F32, &[1, 7]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::CausalDepthwiseConv(access)),
            &inputs.iter().collect::<Vec<_>>(),
            &[&output, &next],
        )
        .unwrap();
    let submission = backend.submit(commands).unwrap();
    let failure = submission
        .wait_timeout(std::time::Duration::from_secs(10))
        .unwrap_err();
    assert!(matches!(failure, BackendError::DeviceErrorFlag { .. }));
    let middle = backend
        .tensor(
            output.buffer(),
            forja_core::Layout::new(DType::F32, 1, vec![5], vec![1], 28).unwrap(),
        )
        .unwrap();
    backend
        .write(&middle, &3_f32.to_le_bytes().repeat(5))
        .unwrap();
    assert_eq!(
        backend.read_replay_output(&middle).unwrap(),
        3_f32.to_le_bytes().repeat(5)
    );
    assert_eq!(backend.read(&output), Err(failure));
    let head = backend
        .tensor(
            output.buffer(),
            forja_core::Layout::new(DType::F32, 0, vec![1], vec![1], 28).unwrap(),
        )
        .unwrap();
    let tail = backend
        .tensor(
            output.buffer(),
            forja_core::Layout::new(DType::F32, 6, vec![1], vec![1], 28).unwrap(),
        )
        .unwrap();
    for edge in [&head, &tail] {
        assert_eq!(backend.read_replay_output(edge), Err(failure));
    }
    backend
        .write(&output, &2_f32.to_le_bytes().repeat(7))
        .unwrap();
    assert_eq!(
        backend.read(&output).unwrap(),
        2_f32.to_le_bytes().repeat(7)
    );
    assert_eq!(backend.read(&next), Err(failure));
    for tensor in [&output, &next] {
        backend.release(tensor).unwrap();
        assert!(
            backend
                .tensor(tensor.buffer(), tensor.layout().clone())
                .is_err()
        );
    }
    assert_eq!(
        submission.wait_timeout(std::time::Duration::from_secs(10)),
        Err(failure)
    );
}
