use forja_core::{Dispatch, KernelCost, Op};
mod critical;
pub(crate) use critical::of as critical_path;
mod row_program;

pub(crate) fn of(dispatch: &Dispatch) -> Option<KernelCost> {
    let mut cost = super::of(dispatch)?;
    cost = cost
        .checked_add(route_setup(dispatch)?)?
        .checked_add(epilogue_calls(dispatch)?)?;
    if matches!(dispatch.op(), Op::Sample { .. } | Op::SampleMasked { .. }) {
        let visits = dispatch.work()?;
        cost.scalar_ops = cost.scalar_ops.max(visits.checked_mul(32)?);
        cost.sfu_ops = cost.sfu_ops.max(visits.checked_mul(4)?);
        cost.read_bytes = cost
            .read_bytes
            .checked_add(visits.checked_mul(dispatch.inputs()[0].layout().dtype().byte_size())?)?;
    }
    if matches!(dispatch.op(), Op::Program(_)) {
        return addressing(dispatch, row_program::of(dispatch, cost)?);
    }
    let model = crate::variants::selected_cost_model(dispatch);
    let [rows, columns, _] = model.tile.map(u64::from);
    if rows == 0 {
        cost.scalar_ops = cost.scalar_ops.checked_add(cost.flops)?;
        cost.flops = 0;
        return addressing(dispatch, cost);
    }
    let flops = model.padded_flops(dispatch, cost)?[1];
    let input_values = flops.checked_div(2)?.checked_div(columns)?;
    let weight_values = flops.checked_div(2)?.checked_div(rows)?;
    let input = dispatch.primitive_inputs();
    let input_bytes = input_values.checked_mul(input[0].layout().dtype().byte_size())?;
    let weight_bytes = match dispatch.op() {
        Op::QuantMatmul { bits, .. } | Op::GatherQuantMatmul { bits, .. } => {
            let words = weight_values.div_ceil(32 / u64::from(bits));
            let parameters = words
                .checked_mul(2)?
                .checked_mul(input[2].layout().dtype().byte_size())?;
            cost.scalar_ops = cost
                .scalar_ops
                .checked_sub(cost.flops)?
                .checked_add(weight_values.checked_mul(8)?)?;
            words.checked_mul(4)?.checked_add(parameters)?
        }
        Op::Matmul => weight_values.checked_mul(input[1].layout().dtype().byte_size())?,
        _ => return None,
    };
    cost.read_bytes = cost
        .read_bytes
        .checked_add(input_bytes)?
        .checked_add(weight_bytes)?;
    cost.flops = flops;
    addressing(dispatch, cost)
}

pub(super) fn addressing(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
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
    let visits = cost.read_bytes.checked_add(cost.write_bytes)?.div_ceil(2);
    cost.scalar_ops = cost.scalar_ops.checked_add(visits.checked_mul(weight)?)?;
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
        Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. } => {
            u64::from(dispatch.inputs()[0].layout().shape()[0])
                .checked_mul(12)?
                .checked_add(16)?
        }
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
    fn quantized_schedule_counts_decode_calls_per_tile_and_trip() {
        let backend = CpuBackend::new();
        for rows in [16, 33, 65] {
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
