use forja_core::{Dispatch, KernelCost};

fn group_bound(cost: KernelCost, groups: u64, factor: u64) -> Option<KernelCost> {
    let scale = |value: u64| Some(value.checked_mul(factor.min(groups))?.div_ceil(groups));
    Some(KernelCost {
        flops: scale(cost.flops)?,
        read_bytes: scale(cost.read_bytes)?,
        write_bytes: scale(cost.write_bytes)?,
        scalar_ops: scale(cost.scalar_ops)?,
        sfu_ops: scale(cost.sfu_ops)?,
        metadata_read_bytes: scale(cost.metadata_read_bytes)?,
        scratch_read_bytes: scale(cost.scratch_read_bytes)?,
        scratch_write_bytes: scale(cost.scratch_write_bytes)?,
        state_read_bytes: scale(cost.state_read_bytes)?,
        state_write_bytes: scale(cost.state_write_bytes)?,
        ..KernelCost::default()
    })
}

fn carry_factor(values: u64, panel: Option<u32>) -> u64 {
    if panel.is_some_and(|panel| values.is_multiple_of(u64::from(panel))) {
        1
    } else {
        2
    }
}

pub(super) fn costs(dispatch: &Dispatch) -> Option<(KernelCost, KernelCost)> {
    let (panel, groups) = crate::encoding::block_schedule(dispatch)?;
    let mut stages = panel
        .and_then(|panel| crate::profile_cost::state_block::stages(dispatch, panel))
        .or_else(|| KernelCost::delta_block_stages(dispatch))?;
    if let Some(panel) = panel {
        for (cost, padded) in
            stages
                .iter_mut()
                .zip(crate::profile_cost::state_block::padded_stages(
                    dispatch, panel,
                )?)
        {
            cost.flops = cost.flops.max(padded);
        }
    }
    let block = u64::try_from(forja_core::DeltaBlockScratch::TOKENS).ok()?;
    let rows = u64::from(*dispatch.inputs().first()?.layout().shape().first()?).min(block);
    let values = u64::from(*dispatch.inputs().get(5)?.layout().shape().get(2)?);
    let factors = [
        8_u64.checked_mul(block.div_ceil(rows))?,
        4,
        carry_factor(values, panel),
        8,
    ];
    let mut total = KernelCost::default();
    let mut critical = KernelCost::default();
    for (stage, ((mut cost, groups), factor)) in
        stages.into_iter().zip(groups).zip(factors).enumerate()
    {
        if groups == 0 {
            continue;
        }
        if panel.is_none() {
            cost.scalar_ops = cost.scalar_ops.checked_add(cost.flops)?;
            cost.flops = 0;
        }
        // Carry operands and all scratch cells are f32; partial value panels keep their bound.
        cost = super::address_elements(
            dispatch,
            cost,
            if stage == 2 && panel.is_some() { 4 } else { 2 },
        )?;
        total = total.checked_add(cost)?;
        critical = critical.checked_add(group_bound(cost, groups, factor)?)?;
    }
    Some((total, critical))
}

#[cfg(test)]
pub(super) fn test_dispatch(
    cpu: &forja_cpu::CpuBackend,
    tokens: u32,
    keys: u32,
    values: u32,
) -> Dispatch {
    use forja_core::{CommandList, DType, Op, StateOp};
    let access = forja_testing::state_test_metadata().unwrap().0;
    let inputs = [
        (DType::F32, vec![tokens, 1, keys]),
        (DType::F32, vec![tokens, 1, keys]),
        (DType::F32, vec![tokens, 1, values]),
        (DType::F32, vec![tokens, 1]),
        (DType::F32, vec![tokens, 1]),
        (DType::F32, vec![1, keys, values]),
        (DType::U32, vec![2, 2]),
        (DType::U32, vec![3]),
        (DType::U32, vec![1]),
    ]
    .map(|(dtype, shape)| cpu.alloc(dtype, &shape).unwrap());
    let layout = forja_core::DeltaBlockScratch::new(tokens, 1, keys, values).unwrap();
    let output = cpu.alloc(DType::F32, &[tokens, 1, values]).unwrap();
    let next = cpu.alloc(DType::F32, &[1, keys, values]).unwrap();
    let scratch = cpu.alloc(DType::F32, &[layout.elements()]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::DeltaRuleBlock(access)),
            &inputs.iter().collect::<Vec<_>>(),
            &[&output, &next, &scratch],
        )
        .unwrap();
    commands.into_dispatches().remove(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn carry_panels_bound_a_walk_including_partial_columns() {
        for values in [1_u64, 7, 33, 128, 4097] {
            for panel in [16_u32, 32] {
                let work = (0..values.div_ceil(u64::from(panel)))
                    .map(|group| {
                        let start = group * u64::from(panel);
                        (start..(start + u64::from(panel)).min(values))
                            .map(|_| 128 * 64 * 7)
                            .sum::<u64>()
                            + 28 * u64::from(panel)
                    })
                    .collect::<Vec<_>>();
                let bound = group_bound(
                    KernelCost {
                        scalar_ops: work.iter().sum(),
                        ..KernelCost::default()
                    },
                    u64::try_from(work.len()).unwrap(),
                    carry_factor(values, Some(panel)),
                )
                .unwrap();
                assert!(bound.scalar_ops >= *work.iter().max().unwrap());
            }
        }
    }

    #[test]
    fn gram_group_bounds_cover_a_walk_of_the_triangular_stage() {
        for rows in [1_u64, 7, 33, 64] {
            for keys in [1_u64, 7, 128] {
                let mut groups = [0_u64; 32];
                for index in 0..4096_u64 {
                    let (row, column) = (index / 64, index % 64);
                    let work = if row < rows && column <= row {
                        4 * keys + row - column + if column == 0 { row + 1 } else { 0 }
                    } else {
                        1
                    };
                    groups[usize::try_from(index / 128).unwrap()] += work;
                }
                let total = KernelCost {
                    scalar_ops: groups.iter().sum(),
                    ..KernelCost::default()
                };
                let bound = group_bound(total, 32, 8 * 64_u64.div_ceil(rows)).unwrap();
                assert!(bound.scalar_ops >= *groups.iter().max().unwrap());
            }
        }
    }
    #[test]
    fn block_groups_share_the_encoder_geometry_and_scratch_is_affine() {
        let cpu = forja_cpu::CpuBackend::new();
        let dispatch = test_dispatch(&cpu, 64, 128, 33);
        assert_eq!(
            crate::encoding::block_schedule(&dispatch).unwrap().1,
            [32, 2, 1, 17]
        );
        let raw = KernelCost {
            read_bytes: 4096,
            ..KernelCost::default()
        };
        let operand = super::super::addressing(&dispatch, raw).unwrap();
        let scratch = super::super::addressing(
            &dispatch,
            KernelCost {
                scratch_read_bytes: 4096,
                ..raw
            },
        )
        .unwrap();
        assert!(scratch.scalar_ops < operand.scalar_ops);
        assert!(costs(&dispatch).unwrap().1.scalar_ops > 0);
    }
}
