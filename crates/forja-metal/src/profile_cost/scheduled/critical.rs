use forja_core::{Dispatch, KernelCost, Op, StateOp, program::ProgramKind};

pub(crate) fn of(dispatch: &Dispatch) -> Option<KernelCost> {
    if matches!(
        dispatch.op(),
        Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. }
    ) {
        return super::attention::costs(
            dispatch,
            crate::encoding::AttentionSchedule::selected(dispatch)?,
        )
        .map(|(_, critical)| critical);
    }
    if matches!(dispatch.op(), Op::State(StateOp::DeltaRuleBlock(_))) {
        return super::state::costs(dispatch).map(|(_, critical)| critical);
    }
    let total = super::of(dispatch)?;
    if let Some(cost) = matrix_tile(dispatch, total) {
        return Some(cost);
    }
    let output = dispatch.output().layout();
    let width = u64::from(output.shape().last().copied().unwrap_or(1)).max(1);
    let elements = output.element_count().max(1);
    let row = matches!(
        dispatch.op(),
        Op::RmsNorm { .. }
            | Op::Argmax { .. }
            | Op::TopK { .. }
            | Op::Sample { .. }
            | Op::SampleMasked { .. }
    ) || dispatch
        .bound_program()
        .is_some_and(|p| p.program().program().kind == ProgramKind::Row);
    let groups = match dispatch.op() {
        Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. } => {
            let shape = dispatch.inputs().first()?.layout().shape();
            u64::from(*shape.first()?).checked_mul(u64::from(*shape.get(1)?))?
        }
        Op::State(StateOp::DeltaRule(_) | StateOp::DeltaRuleBlock(_)) => {
            // Prefix stages are shared by the value columns.
            let state = dispatch.inputs().get(5)?.layout().shape();
            u64::from(*state.first()?)
        }
        _ if row => elements.checked_div(width)?,
        _ => elements,
    }
    .max(1);
    let mut cost = KernelCost {
        read_bytes: total.read_bytes.div_ceil(groups),
        write_bytes: total.write_bytes.div_ceil(groups),
        scalar_ops: total.scalar_ops.checked_add(total.flops)?.div_ceil(groups),
        sfu_ops: total.sfu_ops.div_ceil(groups),
        metadata_read_bytes: total.metadata_read_bytes,
        ..KernelCost::default()
    };
    cost.read_bytes = cost.read_bytes.checked_add(total.metadata_read_bytes)?;
    if row {
        cost.scalar_ops = cost.scalar_ops.checked_add(4096)?;
    }
    if matches!(dispatch.op(), Op::RmsNorm { .. }) {
        cost.read_bytes = cost.read_bytes.max(width.checked_mul(8)?);
    }
    match dispatch.op() {
        Op::Matmul | Op::GatherMatmul | Op::QuantMatmul { .. } | Op::GatherQuantMatmul { .. } => {
            let inputs = dispatch.primitive_inputs();
            let inner = u64::from(*inputs.first()?.layout().shape().last()?);
            let quantized = matches!(
                dispatch.op(),
                Op::QuantMatmul { .. } | Op::GatherQuantMatmul { .. }
            );
            let weight = if quantized {
                4_u64
                    .checked_add(inputs.get(2)?.layout().dtype().byte_size())?
                    .checked_add(inputs.get(3)?.layout().dtype().byte_size())?
            } else {
                inputs.get(1)?.layout().dtype().byte_size()
            };
            cost.read_bytes = cost.read_bytes.max(
                inner.checked_mul(
                    inputs
                        .first()?
                        .layout()
                        .dtype()
                        .byte_size()
                        .checked_add(weight)?,
                )?,
            );
            cost.scalar_ops =
                cost.scalar_ops
                    .max(inner.checked_mul(if quantized { 10 } else { 2 })?);
        }
        _ => {}
    }
    // Route planning is a prefix before output work; its one-group stages cannot be averaged
    // across the much larger matrix grid.
    cost.checked_add(super::route_setup(dispatch)?)
}

fn matrix_tile(dispatch: &Dispatch, total: KernelCost) -> Option<KernelCost> {
    if !matches!(dispatch.op(), Op::Matmul | Op::QuantMatmul { .. }) {
        return None;
    }
    let tile = if matches!(
        crate::encoding::selected_variant(dispatch),
        Some(crate::variants::MetalVariant::MatmulGemvTransposed)
    ) {
        [1, 32, 1]
    } else {
        crate::variants::selected_cost_model(dispatch).tile
    };
    let [rows, columns] = dispatch.output().layout().shape() else {
        return None;
    };
    if tile[0] == 0 || tile[1] == 0 {
        return None;
    }
    let groups = u64::from(*rows)
        .div_ceil(u64::from(tile[0]))
        .checked_mul(u64::from(*columns).div_ceil(u64::from(tile[1])))?;
    Some(KernelCost {
        read_bytes: total.read_bytes.div_ceil(groups),
        write_bytes: total.write_bytes.div_ceil(groups),
        flops: total.flops.div_ceil(groups),
        scalar_ops: total.scalar_ops.div_ceil(groups),
        sfu_ops: total.sfu_ops.div_ceil(groups),
        ..KernelCost::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType, StateStep};
    use forja_cpu::CpuBackend;

    #[test]
    fn tiled_matrix_critical_work_retains_a_complete_cooperative_tile() {
        use forja_core::{BackendDispatchData, DeviceCapability, ViewOp};
        let cpu = CpuBackend::new();
        for (rows, columns, inner, transposed) in [
            (7, 33, 4097, false),
            (512, 192, 576, false),
            (512, 2048, 1024, false),
            (1, 33, 4097, true),
            (1, 1024, 2048, true),
        ] {
            let left = cpu.alloc(DType::F32, &[rows, inner]).unwrap();
            let right = if transposed {
                let packed = cpu.alloc(DType::F32, &[columns, inner]).unwrap();
                cpu.view(&packed, ViewOp::Permute(vec![1, 0])).unwrap()
            } else {
                cpu.alloc(DType::F32, &[inner, columns]).unwrap()
            };
            let output = cpu.alloc(DType::F32, &[rows, columns]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::Matmul, &[&left, &right], &output)
                .unwrap();
            let selected = crate::variants::resolve(
                commands.last_dispatch().unwrap(),
                if transposed {
                    "matmul.gemv-transposed"
                } else {
                    "matmul.steel-32x64x16-1x2"
                },
                &[DeviceCapability::Metal4],
            )
            .unwrap();
            commands.set_last_backend_data(BackendDispatchData::new(selected));
            let dispatch = commands.last_dispatch().unwrap();
            let total = super::super::of(dispatch).unwrap();
            let critical = of(dispatch).unwrap();
            let groups = if transposed {
                u64::from(columns).div_ceil(32)
            } else {
                u64::from(rows).div_ceil(32) * u64::from(columns).div_ceil(64)
            };
            assert_eq!(critical.flops, total.flops.div_ceil(groups));
            assert_eq!(critical.read_bytes, total.read_bytes.div_ceil(groups));
        }
    }

    #[test]
    fn a_single_row_path_cannot_repeat_the_whole_dispatch_address_work() {
        let cpu = CpuBackend::new();
        for width in [1, 7, 33, 1024, 4097] {
            let input = cpu.alloc(DType::F32, &[1, width]).unwrap();
            let weight = cpu.alloc(DType::F32, &[width]).unwrap();
            let output = cpu.alloc(DType::F32, &[1, width]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::RmsNorm { eps: 1e-5 }, &[&input, &weight], &output)
                .unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let total = super::super::of(dispatch).unwrap();
            let critical = of(dispatch).unwrap();
            assert!(critical.scalar_ops <= total.scalar_ops + 4096);
        }
    }

    #[test]
    fn one_convolution_output_retains_the_entire_tap_chain() {
        let cpu = CpuBackend::new();
        let access: StateStep = forja_testing::state_test_metadata().unwrap().0;
        for taps in [7, 33, 4097] {
            for channels in [1, 7] {
                let inputs = [
                    (DType::F32, vec![1, channels]),
                    (DType::F32, vec![taps, channels]),
                    (DType::F32, vec![taps - 1, channels]),
                    (DType::U32, vec![2, 2]),
                    (DType::U32, vec![3]),
                    (DType::U32, vec![1]),
                ]
                .map(|(dtype, shape)| cpu.alloc(dtype, &shape).unwrap());
                let output = cpu.alloc(DType::F32, &[1, channels]).unwrap();
                let next = cpu.alloc(DType::F32, &[taps - 1, channels]).unwrap();
                let mut list = CommandList::new();
                list.dispatch_many(
                    Op::State(StateOp::CausalDepthwiseConv(access)),
                    &inputs.iter().collect::<Vec<_>>(),
                    &[&output, &next],
                )
                .unwrap();
                let cost = of(list.last_dispatch().unwrap()).unwrap();
                assert!(cost.scalar_ops >= u64::from(taps) * 2);
                assert!(cost.read_bytes >= u64::from(taps) * 8);
            }
        }
    }

    #[test]
    fn cooperative_rows_retain_reduction_width_and_key_visits() {
        let cpu = CpuBackend::new();
        for width in [1, 7, 33, 4097] {
            let input = cpu.alloc(DType::F32, &[7, width]).unwrap();
            let weight = cpu.alloc(DType::F32, &[width]).unwrap();
            let output = cpu.alloc(DType::F32, &[7, width]).unwrap();
            let mut list = CommandList::new();
            list.dispatch(Op::RmsNorm { eps: 1e-5 }, &[&input, &weight], &output)
                .unwrap();
            let cost = of(list.last_dispatch().unwrap()).unwrap();
            assert!(cost.read_bytes >= u64::from(width) * 8);
            assert!(cost.scalar_ops >= u64::from(width) * 2);
        }
        let query = cpu.alloc(DType::F32, &[16, 7, 128]).unwrap();
        let cache = cpu.alloc(DType::F32, &[8, 4097, 128]).unwrap();
        let output = cpu.alloc(DType::F32, &[16, 7, 128]).unwrap();
        let mut list = CommandList::new();
        list.dispatch(
            Op::Sdpa {
                scale: 1.0,
                causal: true,
                q_start: 0,
            },
            &[&query, &cache, &cache],
            &output,
        )
        .unwrap();
        let cost = of(list.last_dispatch().unwrap()).unwrap();
        assert!(cost.read_bytes >= 4097 * 128 * 8);
        assert!(cost.sfu_ops >= 4097);
    }

    #[test]
    fn state_prefix_work_is_not_averaged_across_value_columns() {
        let cpu = CpuBackend::new();
        let mut previous = KernelCost::default();
        for values in [1, 7, 33] {
            let dispatch = super::super::state::test_dispatch(&cpu, 64, 7, values);
            let cost = of(&dispatch).unwrap();
            assert!(cost.scalar_ops >= previous.scalar_ops);
            assert!(cost.read_bytes >= previous.read_bytes);
            previous = cost;
        }
    }

    #[test]
    fn matrix_serial_work_does_not_shrink_with_the_output_grid() {
        let cpu = CpuBackend::new();
        for (rows, columns) in [(1, 1), (7, 33), (33, 7)] {
            let input = cpu.alloc(DType::F32, &[rows, 4097]).unwrap();
            let weights = cpu.alloc(DType::F32, &[4097, columns]).unwrap();
            let output = cpu.alloc(DType::F32, &[rows, columns]).unwrap();
            let mut list = CommandList::new();
            list.dispatch(Op::Matmul, &[&input, &weights], &output)
                .unwrap();
            let cost = of(list.last_dispatch().unwrap()).unwrap();
            assert!(cost.scalar_ops >= 4097 * 2);
            assert!(cost.read_bytes >= 4097 * 8);
        }
    }
}
