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
