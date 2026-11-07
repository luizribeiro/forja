use forja_core::{
    Dispatch, KernelCost,
    program::{Inst, ProgramKind},
};

pub(super) fn of(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    let bound = dispatch.bound_program()?;
    let program = bound.program().program();
    let elements = dispatch.output().layout().element_count();
    let width = u64::from(
        dispatch
            .output()
            .layout()
            .shape()
            .last()
            .copied()
            .unwrap_or(1),
    );
    let rows = elements.checked_div(width)?;
    let reductions = u64::try_from(
        program
            .insts
            .iter()
            .filter(|inst| matches!(inst, Inst::Reduce(..)))
            .count(),
    )
    .ok()?;
    let mut passes = reductions.checked_add(1)?;
    let resident = if program.kind == ProgramKind::Row
        && width <= u64::from(crate::map_codegen::REGISTER_RESIDENT_WIDTH)
    {
        rows.checked_mul(width.div_ceil(32))?.checked_mul(32)?
    } else {
        0
    };
    cost.scalar_ops = 0;
    cost.sfu_ops = 0;
    for &inst in &program.insts {
        if matches!(inst, Inst::Reduce(..)) {
            passes = passes.checked_sub(1)?;
            let combines = rows.checked_mul(width.checked_add(1024)?)?.checked_mul(4)?;
            cost.scalar_ops = cost.scalar_ops.checked_add(combines)?;
            continue;
        }
        if matches!(inst, Inst::Const(_) | Inst::Index(_) | Inst::Extent(_)) {
            continue;
        }
        let calls = elements.checked_mul(passes)?.max(resident);
        let (scalar_ops, sfu_ops) = crate::scalar_codegen::instruction_work(inst);
        cost.scalar_ops = cost
            .scalar_ops
            .checked_add(calls.checked_mul(scalar_ops)?)?;
        cost.sfu_ops = cost.sfu_ops.checked_add(calls.checked_mul(sfu_ops)?)?;
        if let Inst::Input(slot) = inst {
            let dtype = dispatch
                .inputs()
                .get(usize::try_from(slot).ok()?)?
                .layout()
                .dtype();
            cost.read_bytes = cost
                .read_bytes
                .checked_add(calls.checked_mul(dtype.byte_size())?)?;
        }
    }
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{
        CommandList, DType,
        program::{BinOp, KernelSignature, Program, RedOp, UnOp, prepare_program},
    };
    use forja_cpu::CpuBackend;

    #[test]
    fn scheduled_map_addresses_cover_a_walk_of_every_operand_axis() {
        let backend = CpuBackend::new();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Add, 0, 1),
            ],
            outputs: vec![(0, 2)],
        };
        let prepared = prepare_program(
            &backend,
            program.validate().unwrap(),
            KernelSignature::new(3, vec![DType::F32; 2], vec![DType::F32], 0),
        )
        .unwrap();
        let input = backend.alloc(DType::F32, &[1, 7, 33]).unwrap();
        let extra = backend.alloc(DType::F32, &[1, 7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 7, 33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&input, &extra], &[&output])
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        let cost = super::super::of(dispatch).unwrap();
        let mut visited = 0_u64;
        for _ in 0..output.layout().element_count() {
            for tensor in dispatch.inputs().iter().chain(dispatch.outputs()) {
                for _ in tensor.layout().shape() {
                    visited += 2;
                }
            }
        }
        assert!(cost.scalar_ops >= visited);
        assert!(cost.scalar_ops > output.layout().element_count() * 128);
    }

    #[test]
    fn scalar_map_work_keeps_rank_zero_index_domain() {
        let backend = CpuBackend::new();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0), Inst::Unary(UnOp::Log1p, 0)],
            outputs: vec![(0, 1)],
        };
        let prepared = prepare_program(
            &backend,
            program.validate().unwrap(),
            KernelSignature::new(0, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let input = backend.alloc(DType::F32, &[]).unwrap();
        let output = backend.alloc(DType::F32, &[]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&input], &[&output])
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        let cost = of(dispatch, super::super::super::of(dispatch).unwrap()).unwrap();
        assert!(cost.sfu_ops >= 1);
        assert!(cost.scalar_ops >= 2);
        assert_eq!(cost.read_bytes, 8);
    }

    #[test]
    fn row_program_bound_covers_resident_and_rereading_walks() {
        let backend = CpuBackend::new();
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Unary(UnOp::Exp, 0),
                Inst::Reduce(RedOp::Sum, 1),
                Inst::Unary(UnOp::Rsqrt, 2),
                Inst::Binary(BinOp::Mul, 0, 3),
            ],
            outputs: vec![(0, 4)],
        };
        let prepared = prepare_program(
            &backend,
            program.validate().unwrap(),
            KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        for width in [1_u32, 7, 33, 1024, 4097] {
            let input = backend.alloc(DType::F32, &[7, width]).unwrap();
            let output = backend.alloc(DType::F32, &[7, width]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch_kernel(&prepared, &[&input], &[&output])
                .unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let cost = of(dispatch, super::super::super::of(dispatch).unwrap()).unwrap();
            let mut rereading = 0_u64;
            for end in [2, program.insts.len()] {
                for _ in 0..7 {
                    for _ in 0..width {
                        for inst in &program.insts[..end] {
                            rereading += u64::from(inst.is_special_function());
                        }
                    }
                }
            }
            assert!(cost.sfu_ops >= rereading);
            if width <= crate::map_codegen::REGISTER_RESIDENT_WIDTH {
                let mut resident = 0;
                for _ in 0..7 {
                    for _ in 0..width.next_multiple_of(32) {
                        for inst in &program.insts {
                            resident += u64::from(inst.is_special_function());
                        }
                    }
                }
                assert!(cost.sfu_ops >= resident);
            }
            assert!(cost.scalar_ops >= cost.sfu_ops);
            assert!(cost.read_bytes >= 7 * u64::from(width) * 4 * 3);
        }
    }
}
