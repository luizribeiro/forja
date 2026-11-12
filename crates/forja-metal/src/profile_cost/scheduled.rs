use forja_core::{Dispatch, KernelCost, Op};
mod attention;
pub(crate) use attention::costs as attention_cost;
mod critical;
pub(crate) use critical::of as critical_path;
mod row_program;
mod sampling;
#[cfg(test)]
mod scratch_address_tests;
mod state;

pub(crate) fn of(dispatch: &Dispatch) -> Option<KernelCost> {
    let mut cost = super::of(dispatch)?;
    cost = cost
        .checked_add(route_setup(dispatch)?)?
        .checked_add(epilogue_calls(dispatch)?)?;
    if matches!(
        dispatch.op(),
        Op::State(forja_core::StateOp::DeltaRuleBlock(_))
    ) {
        return state::costs(dispatch).map(|(total, _)| total);
    }
    if matches!(dispatch.op(), Op::Sample { .. } | Op::SampleMasked { .. }) {
        return addressing(dispatch, sampling::of(dispatch, cost)?);
    }
    if matches!(dispatch.op(), Op::Program(_)) {
        return addressing(dispatch, row_program::of(dispatch, cost)?);
    }
    if matches!(
        dispatch.op(),
        Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. }
    ) {
        return attention::costs(
            dispatch,
            crate::encoding::AttentionSchedule::selected(dispatch)?,
        )
        .map(|(total, _)| total);
    }
    let model = crate::variants::selected_cost_model(dispatch);
    let [rows, columns, _] = model.tile.map(u64::from);
    if rows == 0 {
        cost.scalar_ops = cost.scalar_ops.checked_add(cost.flops)?;
        cost.flops = 0;
        return scalar_issue_work(dispatch, addressing(dispatch, cost)?);
    }
    let address_base = addressing(dispatch, cost)?
        .scalar_ops
        .checked_sub(cost.scalar_ops)?;
    let flops = *model.padded_flops(dispatch, cost)?.get(1)?;
    let input_values = flops.checked_div(2)?.checked_div(columns)?;
    let weight_values = flops.checked_div(2)?.checked_div(rows)?;
    let input = dispatch.primitive_inputs();
    let input_bytes = input_values.checked_mul(input.first()?.layout().dtype().byte_size())?;
    let weight_bytes = match dispatch.op() {
        Op::QuantMatmul { bits, .. } | Op::GatherQuantMatmul { bits, .. } => {
            let words = weight_values.div_ceil(32 / u64::from(bits));
            let parameters = words
                .checked_mul(2)?
                .checked_mul(input.get(2)?.layout().dtype().byte_size())?;
            cost.scalar_ops = cost
                .scalar_ops
                .checked_sub(cost.flops)?
                .checked_add(weight_values.checked_mul(8)?)?;
            words.checked_mul(4)?.checked_add(parameters)?
        }
        Op::Matmul => weight_values.checked_mul(input.get(1)?.layout().dtype().byte_size())?,
        _ => return None,
    };
    cost.read_bytes = cost
        .read_bytes
        .checked_add(input_bytes)?
        .checked_add(weight_bytes)?;
    cost.flops = flops;
    if matches!(
        crate::encoding::selected_variant(dispatch),
        Some(
            crate::variants::MetalVariant::QuantMatmulTiled
                | crate::variants::MetalVariant::QuantMatmulTiledK64
        )
    ) {
        cost.scalar_ops =
            cost.scalar_ops
                .checked_add(address_base)?
                .checked_add(quantized_loader_addresses(
                    dispatch,
                    model.tile,
                    input_values,
                    weight_values,
                )?)?;
        return Some(cost);
    }
    scalar_issue_work(dispatch, addressing(dispatch, cost)?)
}

fn quantized_loader_addresses(
    dispatch: &Dispatch,
    tile: [u32; 3],
    inputs: u64,
    weights: u64,
) -> Option<u64> {
    let Op::QuantMatmul { bits, .. } = dispatch.op() else {
        return None;
    };
    let layout = dispatch.inputs().first()?.layout();
    let shape = layout.shape();
    let rows = u64::from(*shape.first()?);
    let inner = u64::from(*shape.get(1)?);
    let vector = 16_u64.checked_div(layout.dtype().byte_size())?;
    let wide = rows.checked_rem(u64::from(tile[0]))? == 0
        && inner.checked_rem(u64::from(tile[2]))? == 0
        && forja_core::ByteAlignment::of(layout.byte_span().start).supports(16)
        && layout.strides().first()?.checked_rem(vector)? == 0
        && *layout.strides().get(1)? == 1;
    let input_groups = inputs.div_ceil(if wide { vector } else { 1 });
    let words = weights.div_ceil(32_u64.checked_div(u64::from(bits))?);
    // Each packed word loads three affine addresses before decoding its values.
    input_groups
        .checked_add(words.checked_mul(3)?)?
        .checked_add(dispatch.output().layout().element_count())?
        .checked_mul(32)
}

fn scalar_issue_work(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    if let Some(groups) = scalar_matrix_groups(dispatch) {
        let groups = groups.max(1);
        let slots = groups.div_ceil(32).checked_mul(32)?;
        cost.scalar_ops = cost
            .scalar_ops
            .checked_add(cost.flops)?
            .checked_mul(slots)?
            .div_ceil(groups);
        cost.sfu_ops = cost.sfu_ops.checked_mul(slots)?.div_ceil(groups);
        let shape = dispatch.output().layout().shape();
        let grid = u64::from(*shape.first()?)
            .div_ceil(8)
            .checked_mul(u64::from(*shape.get(1)?).div_ceil(8))?;
        cost.scalar_ops = cost
            .scalar_ops
            .checked_add(grid.checked_mul(8 * 8 * 32 * 32)?)?;
        cost.flops = 0;
    }
    Some(cost)
}

pub(crate) fn scalar_matrix_groups(dispatch: &Dispatch) -> Option<u64> {
    let Op::QuantMatmul { group_size, .. } = dispatch.op() else {
        return None;
    };
    if !matches!(
        crate::encoding::selected_variant(dispatch),
        Some(
            crate::variants::MetalVariant::QuantMatmulSmallM
                | crate::variants::MetalVariant::QuantMatmulPortable
        )
    ) {
        return None;
    }
    u64::from(*dispatch.inputs().first()?.layout().shape().last()?)
        .checked_div(u64::from(group_size))
}

pub(super) fn addressing(dispatch: &Dispatch, cost: KernelCost) -> Option<KernelCost> {
    address_elements(dispatch, cost, 2)
}

pub(super) fn address_elements(
    dispatch: &Dispatch,
    mut cost: KernelCost,
    bytes_per_value: u64,
) -> Option<KernelCost> {
    let layouts = dispatch
        .inputs()
        .iter()
        .chain(dispatch.outputs())
        .map(forja_core::Tensor::layout);
    let affine = matches!(
        dispatch.op(),
        Op::Matmul
            | Op::QuantMatmul { .. }
            | Op::GatherMatmul
            | Op::GatherQuantMatmul { .. }
            | Op::Sdpa { .. }
            | Op::PagedSdpa { .. }
            | Op::PagedSdpaAppend { .. }
    );
    let generic = !affine
        && (matches!(
            dispatch.op(),
            Op::Program(_) | Op::RmsNorm { .. } | Op::State(_)
        ) || layouts.clone().any(|layout| !layout.is_contiguous()));
    let weight = if generic {
        u64::try_from(
            layouts
                .map(|layout| layout.shape().len())
                .max()
                .unwrap_or(0),
        )
        .ok()?
        .checked_mul(128)?
    } else if affine {
        32
    } else {
        0
    };
    let scratch = cost
        .scratch_read_bytes
        .checked_add(cost.scratch_write_bytes)?;
    let bytes = cost
        .read_bytes
        .checked_add(cost.write_bytes)?
        .checked_sub(scratch)?;
    let (scratch_width, scratch_weight) = if matches!(
        dispatch.op(),
        Op::State(forja_core::StateOp::DeltaRuleBlock(_))
    ) {
        (4, 4)
    } else {
        (2, 32)
    };
    cost.scalar_ops = cost
        .scalar_ops
        .checked_add(bytes.div_ceil(bytes_per_value).checked_mul(weight)?)?
        .checked_add(
            scratch
                .div_ceil(scratch_width)
                .checked_mul(weight.min(scratch_weight))?,
        )?;
    Some(cost)
}

pub(crate) fn launches(dispatch: &Dispatch) -> Option<u64> {
    let grouped = crate::encoding::internal_gather_dispatch_names(dispatch);
    if !grouped.is_empty() {
        return u64::try_from(grouped.len()).ok();
    }
    Some(match dispatch.op() {
        Op::Sample {
            rejection_rounds, ..
        } if !crate::encoding::parallel_sampling(dispatch) => {
            u64::from(rejection_rounds).checked_mul(4)?.checked_add(7)?
        }
        Op::Sample { .. } | Op::SampleMasked { .. } | Op::Argmax { .. } | Op::TopK { .. } => 2,
        Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. } => u64::from(
            dispatch
                .inputs()
                .first()?
                .layout()
                .shape()
                .first()
                .copied()?,
        )
        .checked_mul(12)?
        .checked_add(16)?,
        Op::State(_) | Op::Matmul | Op::QuantMatmul { .. } => 4,
        Op::Program(_)
        | Op::Copy
        | Op::PagedCopy { .. }
        | Op::RmsNorm { .. }
        | Op::Rope { .. }
        | Op::Embed
        | Op::QuantEmbed { .. }
        | Op::GatherMatmul
        | Op::GatherQuantMatmul { .. } => 1,
    })
}

fn epilogue_calls(dispatch: &Dispatch) -> Option<KernelCost> {
    let mut cost = KernelCost::default();
    let Some(epilogue) = dispatch.prepared_epilogue() else {
        return Some(cost);
    };
    let calls = dispatch.output().layout().element_count();
    for &inst in &epilogue.validated().program().insts {
        let (scalar, sfu) = crate::scalar_codegen::instruction_work(inst);
        cost.scalar_ops = cost.scalar_ops.checked_add(calls.checked_mul(scalar)?)?;
        cost.sfu_ops = cost.sfu_ops.checked_add(calls.checked_mul(sfu)?)?;
        if let forja_core::program::Inst::Input(slot) = inst
            && let Some(extra) =
                slot.checked_sub(u32::from(epilogue.signature().accumulator_count()))
        {
            let dtype = epilogue
                .signature()
                .extra_dtypes()
                .get(usize::try_from(extra).ok()?)?;
            cost.read_bytes = cost
                .read_bytes
                .checked_add(calls.checked_mul(dtype.byte_size())?)?;
            cost.scalar_ops = cost.scalar_ops.checked_add(
                calls
                    .checked_mul(u64::from(epilogue.signature().logical_rank()))?
                    .checked_mul(128)?,
            )?;
        }
    }
    Some(cost)
}

pub(super) fn route_setup(dispatch: &Dispatch) -> Option<KernelCost> {
    let mut setup = KernelCost::default();
    for &kernel in crate::encoding::internal_gather_dispatch_names(dispatch)
        .iter()
        .take(3)
    {
        setup = setup.checked_add(super::internal(dispatch, kernel, None)?)?;
    }
    setup.scalar_ops = setup
        .scalar_ops
        .checked_add(setup.read_bytes.checked_div(4)?.checked_mul(8)?)?
        .checked_add(setup.write_bytes.checked_div(4)?.checked_mul(4)?)?;
    Some(setup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{BackendDispatchData, CommandList, DType, DeviceCapability};
    use forja_cpu::CpuBackend;

    #[test]
    fn metal_sampling_launch_bound_covers_encoded_rejection_rounds() {
        use forja_core::{Backend, Submission};
        let backend = crate::MetalBackend::new().unwrap();
        let logits = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let controls = backend.alloc(DType::U32, &[5]).unwrap();
        backend
            .write(
                &controls,
                &[0_u32, 1, 1.0_f32.to_bits(), 7, 0]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = backend.alloc(DType::U32, &[7]).unwrap();
        for rounds in [1, 4, 8] {
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::Sample {
                        position: 0,
                        chunk_width: 2048,
                        fast_top_k: 8,
                        fast_max_chunks: 32,
                        rejection_rounds: rounds,
                    },
                    &[&logits, &controls],
                    &output,
                )
                .unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let bound = launches(dispatch).unwrap();
            let cost = of(dispatch).unwrap();
            assert!(cost.scalar_ops >= dispatch.work().unwrap());
            assert!(cost.sfu_ops > super::super::of(dispatch).unwrap().sfu_ops);
            let submission = backend.submit_profiled(commands).unwrap();
            submission.wait().unwrap();
            let actual = submission.profile().unwrap().per_dispatch[0].physical_dispatches;
            assert!(actual > 1 && actual <= bound);
        }
    }

    #[test]
    fn scalar_epilogues_price_casts_selects_and_special_functions() {
        use forja_core::program::{
            BinOp, EpilogueSignature, Inst, Program, ProgramKind, UnOp, ValueType, prepare_epilogue,
        };
        let backend = CpuBackend::new();
        let input = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let weight = backend.alloc(DType::F32, &[33, 7]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 7]).unwrap();
        let epilogue = prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Const(0.0),
                    Inst::Binary(BinOp::Lt, 0, 1),
                    Inst::Select(2, 0, 1),
                    Inst::Cast(ValueType::F32, 3),
                    Inst::Unary(UnOp::Log1p, 4),
                ],
                outputs: vec![(0, 5)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, 2, vec![], DType::F32),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_epilogue(Op::Matmul, &[&input, &weight], &output, &epilogue, &[])
            .unwrap();
        let cost = epilogue_calls(commands.last_dispatch().unwrap()).unwrap();
        assert!(cost.scalar_ops >= 49 * 5);
        assert!(cost.sfu_ops >= 49);
        assert_eq!(cost.read_bytes, 0);
        assert!(
            crate::scalar_codegen::instruction_work(Inst::Binary(BinOp::Div, 0, 1)).0
                > crate::scalar_codegen::instruction_work(Inst::Binary(BinOp::Add, 0, 1)).0
        );
    }

    #[test]
    fn scalar_matrix_work_covers_the_padded_simd_issue_walk() {
        let backend = CpuBackend::new();
        for groups in [1, 3, 7, 31, 32, 33, 65] {
            let inner = groups * 64;
            let input = backend.alloc(DType::F32, &[8, inner]).unwrap();
            let packed = backend.alloc(DType::U32, &[8, inner / 8]).unwrap();
            let scales = backend.alloc(DType::BF16, &[8, groups]).unwrap();
            let biases = backend.alloc(DType::BF16, &[8, groups]).unwrap();
            let output = backend.alloc(DType::F32, &[8, 8]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::QuantMatmul {
                        bits: 4,
                        group_size: 64,
                    },
                    &[&input, &packed, &scales, &biases],
                    &output,
                )
                .unwrap();
            let selected = crate::variants::resolve(
                commands.last_dispatch().unwrap(),
                "quant-matmul.small-m",
                &[DeviceCapability::Metal4],
            )
            .unwrap();
            commands.set_last_backend_data(BackendDispatchData::new(selected));
            let dispatch = commands.last_dispatch().unwrap();
            let mut issue_slots = 0_u64;
            for _ in (0..groups).step_by(32) {
                for _ in 0..32 {
                    issue_slots += 1;
                }
            }
            let cost = of(dispatch).unwrap();
            assert_eq!(cost.flops, 0);
            assert_eq!(scalar_matrix_groups(dispatch), Some(u64::from(groups)));
            assert!(cost.scalar_ops >= issue_slots * 8 * 8 * 64 * 2);
        }
    }

    #[test]
    fn tiled_loader_prices_guarded_scalar_accesses_for_shifted_or_strided_views() {
        use forja_core::{Slice, ViewOp};
        let cpu = CpuBackend::new();
        for (dtype, padding, shift) in [
            (DType::F32, 0_u32, 0_u32),
            (DType::BF16, 0, 0),
            (DType::F32, 1, 0),
            (DType::BF16, 1, 1),
        ] {
            let storage = cpu.alloc(dtype, &[32, 1024 + padding]).unwrap();
            let input = cpu
                .view(
                    &storage,
                    ViewOp::Slice(vec![
                        Slice::new(0, 32, 1).unwrap(),
                        Slice::new(shift, 1024, 1).unwrap(),
                    ]),
                )
                .unwrap();
            let packed = cpu.alloc(DType::U32, &[33, 128]).unwrap();
            let scale = cpu.alloc(DType::BF16, &[33, 16]).unwrap();
            let output = cpu.alloc(dtype, &[32, 33]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::QuantMatmul {
                        bits: 4,
                        group_size: 64,
                    },
                    &[&input, &packed, &scale, &scale],
                    &output,
                )
                .unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let mut calls = 0;
            for _ in 0..2 {
                for _ in 0..16 {
                    for _ in 0..128 {
                        let values = 16;
                        calls += if padding == 0 {
                            values / (16 / dtype.byte_size())
                        } else {
                            values
                        };
                        calls += 32 * 64 / 8 / 128 * 3;
                    }
                }
            }
            calls += 32 * 33;
            assert_eq!(
                quantized_loader_addresses(dispatch, [32, 32, 64], 2 * 32 * 1024, 2 * 32 * 1024),
                Some(calls * 32)
            );
        }
    }

    #[test]
    fn quantized_schedule_counts_decode_calls_per_tile_and_trip() {
        let backend = CpuBackend::new();
        for rows in [16, 32, 33, 64, 65] {
            for columns in [1, 7, 33] {
                let input = backend.alloc(DType::F32, &[rows, 1024]).unwrap();
                let packed = backend.alloc(DType::U32, &[columns, 128]).unwrap();
                let scales = backend.alloc(DType::BF16, &[columns, 16]).unwrap();
                let biases = backend.alloc(DType::BF16, &[columns, 16]).unwrap();
                let output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
                let mut commands = CommandList::new();
                commands
                    .dispatch(
                        Op::QuantMatmul {
                            bits: 4,
                            group_size: 64,
                        },
                        &[&input, &packed, &scales, &biases],
                        &output,
                    )
                    .unwrap();
                let selected = crate::variants::resolve(
                    commands.last_dispatch().unwrap(),
                    "quant-matmul.tiled",
                    &[DeviceCapability::Metal4],
                )
                .unwrap();
                commands.set_last_backend_data(BackendDispatchData::new(selected));
                let dispatch = commands.last_dispatch().unwrap();
                let mut decoded = 0_u64;
                let mut input_loads = 0_u64;
                for _ in (0..rows).step_by(32) {
                    for _ in (0..columns).step_by(32) {
                        for _ in (0..1024).step_by(32) {
                            for _ in 0..32 {
                                for _ in 0..32 {
                                    decoded += 1;
                                    input_loads += 1;
                                }
                            }
                        }
                    }
                }
                let expected_addresses = (input_loads / if rows % 32 == 0 { 4 } else { 1 }
                    + decoded / 8 * 3
                    + u64::from(rows) * u64::from(columns))
                    * 32;
                assert_eq!(
                    quantized_loader_addresses(dispatch, [32, 32, 32], input_loads, decoded),
                    Some(expected_addresses)
                );
                assert!(
                    quantized_loader_addresses(dispatch, [0, 32, 32], input_loads, decoded)
                        .is_none()
                );
                let logical = super::super::of(dispatch).unwrap();
                let scheduled = of(dispatch).unwrap();
                assert_eq!(scheduled.flops, decoded * 32 * 2);
                assert!(scheduled.scalar_ops >= decoded * 8);
                assert_eq!(
                    scheduled.read_bytes,
                    logical.read_bytes + input_loads * 4 + decoded
                );
            }
        }
    }
}
