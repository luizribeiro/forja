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
