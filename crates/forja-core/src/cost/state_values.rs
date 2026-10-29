use super::tests::tensor;

use crate::{
    CommandList, DType, KernelCost, Op, StateAccess, StateLayout, StateOp, StatePlane, StatePool,
};

#[test]
fn delta_contract_refuses_alias_and_shape_errors_and_costs_state_traffic() {
    let pool = StatePool::new(
        StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7, 33]).unwrap()]).unwrap(),
        1,
        16384,
    )
    .unwrap();
    let mut sequence = pool.reserve().unwrap();
    let tx = sequence.begin(1).unwrap();
    let access = StateAccess::from_slot(&sequence.bind(&tx).unwrap()).unwrap();
    let shapes: &[&[u32]] = &[
        &[1, 7],
        &[1, 7],
        &[3, 33],
        &[3],
        &[3],
        &[3, 7, 33],
        &[2, 2],
        &[3],
        &[1],
    ];
    let inputs: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(i, shape)| {
            tensor(
                i as u64 + 1,
                if i >= 6 { DType::U32 } else { DType::F32 },
                shape,
            )
        })
        .collect();
    let refs: Vec<_> = inputs.iter().collect();
    let y = tensor(10, DType::F32, &[3, 33]);
    let state = tensor(11, DType::F32, &[3, 7, 33]);
    let op = Op::State(StateOp::DeltaRule(access));
    let mut commands = CommandList::new();
    commands.dispatch_many(op, &refs, &[&y, &state]).unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(cost.flops, 7 * 3 * 7 * 33 + 2 * 3 * 33);
    assert_eq!(cost.read_bytes, 3280);
    assert_eq!(cost.write_bytes, (3 * 33 + 3 * 7 * 33) * 4);
    assert_eq!(cost.scalar_ops, 8);
    assert!(
        commands
            .dispatch_many(op, &refs, &[&y, &inputs[5]])
            .is_err()
    );
    assert!(
        commands
            .dispatch_many(op, &refs[..8], &[&y, &state])
            .is_err()
    );
    assert!(commands.dispatch_many(op, &refs, &[&state, &y]).is_err());
}
