use super::tests::tensor;

use crate::{
    CommandList, DType, KernelCost, Op, StateAccess, StateLayout, StateOp, StatePlane, StatePool,
    Tensor,
};

fn delta_fixture(block: bool) -> (StateAccess, Vec<Tensor>, [Tensor; 2]) {
    let pool = StatePool::new(
        StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7, 33]).unwrap()]).unwrap(),
        1,
        16384,
    )
    .unwrap();
    let mut sequence = pool.reserve().unwrap();
    let tx = sequence.begin(1).unwrap();
    let access = StateAccess::from_slot(&sequence.bind(&tx).unwrap()).unwrap();
    let query = if block { vec![7, 1, 7] } else { vec![1, 7] };
    let value = if block { vec![7, 3, 33] } else { vec![3, 33] };
    let gate = if block { vec![7, 3] } else { vec![3] };
    let shapes: &[&[u32]] = &[
        &query,
        &query,
        &value,
        &gate,
        &gate,
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
    let y = tensor(10, DType::F32, &value);
    let state = tensor(11, DType::F32, &[3, 7, 33]);
    (access, inputs, [y, state])
}

#[test]
fn delta_contract_refuses_alias_and_shape_errors_and_costs_state_traffic() {
    let (access, inputs, [y, state]) = delta_fixture(false);
    let refs: Vec<_> = inputs.iter().collect();
    let op = Op::State(StateOp::DeltaRule(access));
    let mut commands = CommandList::new();
    commands.dispatch_many(op, &refs, &[&y, &state]).unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(cost.flops, 7 * 3 * 7 * 33 + 2 * 3 * 33);
    assert_eq!(cost.read_bytes, 3280);
    assert_eq!(cost.write_bytes, (3 * 33 + 3 * 7 * 33) * 4);
    assert_eq!(cost.scalar_ops, 8);
    assert_eq!(
        (cost.state_read_bytes, cost.state_write_bytes),
        (2772, 2772)
    );
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

#[test]
fn block_contract_refuses_bad_grouping_scratch_and_aliases() {
    let (access, inputs, [y, state]) = delta_fixture(true);
    let mut refs: Vec<_> = inputs.iter().collect();
    let layout = crate::DeltaBlockScratch::new(7, 3, 7, 33).unwrap();
    let scratch = tensor(12, DType::F32, &[layout.elements()]);
    let op = Op::State(StateOp::DeltaRuleBlock(access));
    CommandList::new()
        .dispatch_many(op, &refs, &[&y, &state, &scratch])
        .unwrap();
    assert!(
        CommandList::new()
            .dispatch_many(op, &refs, &[&y, &state])
            .is_err()
    );
    assert!(
        CommandList::new()
            .dispatch_many(op, &refs, &[&y, &state, &y])
            .is_err()
    );
    let wrong_scratch = tensor(13, DType::F32, &[layout.elements() - 1]);
    assert!(
        CommandList::new()
            .dispatch_many(op, &refs, &[&y, &state, &wrong_scratch])
            .is_err()
    );
    let wrong_query = tensor(14, DType::F32, &[7, 2, 7]);
    refs[0] = &wrong_query;
    assert!(
        CommandList::new()
            .dispatch_many(op, &refs, &[&y, &state, &scratch])
            .is_err()
    );
}

#[test]
fn block_cost_counts_each_stage_once_with_scratch_and_special_functions() {
    let (access, inputs, [y, state]) = delta_fixture(true);
    let scratch = tensor(
        12,
        DType::F32,
        &[crate::DeltaBlockScratch::new(7, 3, 7, 33)
            .unwrap()
            .elements()],
    );
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::DeltaRuleBlock(access)),
            &inputs.iter().collect::<Vec<_>>(),
            &[&y, &state, &scratch],
        )
        .unwrap();
    let dispatch = commands.last_dispatch().unwrap();
    let stages = KernelCost::delta_block_stages(dispatch).unwrap();
    let total = KernelCost::of(dispatch).unwrap();
    assert_eq!(
        total.read_bytes,
        stages.iter().map(|stage| stage.read_bytes).sum::<u64>()
    );
    assert_eq!(
        total.write_bytes,
        stages.iter().map(|stage| stage.write_bytes).sum::<u64>()
    );
    assert_eq!(total.sfu_ops, stages[0].sfu_ops);
    assert!(total.sfu_ops > 0);
    assert!(total.scratch_read_bytes > 0);
    assert_eq!(stages[0].scratch_write_bytes, 3 * 4096 * 8 + 3 * 7 * 8);
    assert_eq!(total.state_read_bytes, 3 * 7 * 33 * 4);
    assert_eq!(total.state_write_bytes, total.state_read_bytes);
}

#[test]
fn convolution_counts_history_tap_reads_and_boundary_carry() {
    let (access, metadata, _) = delta_fixture(false);
    for tokens in [1, 7, 33] {
        let x = tensor(1, DType::F32, &[tokens, 7]);
        let weights = tensor(2, DType::F32, &[4, 7]);
        let history = tensor(3, DType::F32, &[3, 7]);
        let y = tensor(10, DType::F32, &[tokens, 7]);
        let next = tensor(11, DType::F32, &[3, 7]);
        let mut inputs = vec![&x, &weights, &history];
        inputs.extend(metadata[6..].iter());
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::State(StateOp::CausalDepthwiseConv(access)),
                &inputs,
                &[&y, &next],
            )
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        assert_eq!(cost.state_read_bytes, if tokens == 1 { 140 } else { 168 });
        assert_eq!(cost.state_write_bytes, 84);
        assert_eq!(cost.flops, 2 * u64::from(tokens) * 7 * 4);
    }
}
