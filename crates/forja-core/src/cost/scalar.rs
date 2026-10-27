use crate::{
    Dispatch, KernelCost, Op,
    program::{Inst, Program, ProgramKind},
};

pub(super) fn program_ops(program: &Program, elements: u64, width: u64) -> Option<u64> {
    let rows = elements.checked_div(width)?;
    let mut uniform = Vec::with_capacity(program.insts.len());
    let mut operations = 0_u64;
    for inst in &program.insts {
        let is_uniform = match *inst {
            Inst::Const(_) | Inst::Extent(_) | Inst::Reduce(_, _) => true,
            Inst::Unary(_, arg) | Inst::Cast(_, arg) => *uniform.get(usize::try_from(arg).ok()?)?,
            Inst::Binary(_, left, right) => {
                *uniform.get(usize::try_from(left).ok()?)?
                    && *uniform.get(usize::try_from(right).ok()?)?
            }
            Inst::Select(condition, left, right) => [condition, left, right].iter().all(|&arg| {
                usize::try_from(arg)
                    .ok()
                    .and_then(|index| uniform.get(index))
                    .copied()
                    == Some(true)
            }),
            Inst::Input(_) | Inst::Index(_) => false,
        };
        let count = match inst {
            Inst::Reduce(_, _) => rows.checked_mul(width.checked_sub(1)?)?,
            Inst::Unary(_, _) | Inst::Binary(_, _, _) => {
                if program.kind == ProgramKind::Row && is_uniform {
                    rows
                } else {
                    elements
                }
            }
            Inst::Input(_)
            | Inst::Index(_)
            | Inst::Const(_)
            | Inst::Extent(_)
            | Inst::Cast(_, _)
            | Inst::Select(_, _, _) => 0,
        };
        operations = operations.checked_add(count)?;
        uniform.push(is_uniform);
    }
    Some(operations)
}

pub(super) fn count(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    let input = dispatch.primitive_inputs();
    let elements = dispatch.output().layout().element_count();
    cost.scalar_ops = match dispatch.op() {
        Op::Program(_) => {
            let program = dispatch.bound_program()?.program().program();
            let width = u64::from(
                dispatch
                    .output()
                    .layout()
                    .shape()
                    .last()
                    .copied()
                    .unwrap_or(1),
            );
            program_ops(program, elements, width)?
        }
        Op::RmsNorm { .. } => elements.checked_mul(4)?.checked_add(
            elements
                .checked_div(u64::from(*input[0].layout().shape().last()?))?
                .checked_mul(3)?,
        )?,
        Op::Rope { .. } => elements.checked_mul(3)?,
        Op::Argmax { .. } => input[0].layout().element_count().checked_sub(elements)?,
        Op::TopK { k, normalize, .. } => input[0]
            .layout()
            .element_count()
            .checked_mul(u64::from(k))?
            .checked_add(if normalize {
                elements.checked_mul(3)?
            } else {
                0
            })?,
        Op::Sample { .. } | Op::SampleMasked { .. } => {
            input[0].layout().element_count().checked_mul(4)?
        }
        Op::Copy
        | Op::PagedCopy { .. }
        | Op::Embed
        | Op::QuantEmbed { .. }
        | Op::Matmul
        | Op::GatherMatmul
        | Op::QuantMatmul { .. }
        | Op::GatherQuantMatmul { .. }
        | Op::Sdpa { .. }
        | Op::PagedSdpa { .. }
        | Op::PagedSdpaAppend { .. } => cost.scalar_ops,
    };
    if let Some(epilogue) = dispatch.prepared_epilogue() {
        cost.scalar_ops = cost.scalar_ops.checked_add(program_ops(
            epilogue.validated().program(),
            elements,
            1,
        )?)?;
    }
    if matches!(
        dispatch.op(),
        Op::QuantMatmul { .. } | Op::GatherQuantMatmul { .. }
    ) {
        cost.scalar_ops = cost.scalar_ops.checked_add(cost.flops)?;
    }
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{BinOp, RedOp, UnOp};

    #[test]
    fn scalar_map_counts_its_single_element() {
        use crate::{
            DType, Dispatch,
            cost::tests::tensor,
            program::{KernelSignature, prepared_for_test},
        };
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0), Inst::Unary(UnOp::Exp, 0)],
            outputs: vec![(0, 1)],
        }
        .validate()
        .unwrap();
        let prepared = prepared_for_test(
            program,
            KernelSignature::new(0, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let input = tensor(1, DType::F32, &[]);
        let output = tensor(2, DType::F32, &[]);
        let dispatch = Dispatch::kernel(&prepared, &[&input], &[&output]).unwrap();
        let cost = KernelCost::of(&dispatch).unwrap();
        assert_eq!(cost.scalar_ops, 1);
        assert_eq!((cost.read_bytes, cost.write_bytes), (4, 4));
    }

    #[test]
    fn row_reductions_count_uniform_arithmetic_once_per_row() {
        let mut program = Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Binary(BinOp::Mul, 0, 0),
                Inst::Reduce(RedOp::Sum, 1),
                Inst::Unary(UnOp::Rsqrt, 2),
                Inst::Binary(BinOp::Mul, 0, 3),
            ],
            outputs: vec![(0, 4)],
        };
        assert_eq!(
            program_ops(&program, 7 * 33, 33),
            Some(7 * (33 + 32 + 1 + 33))
        );
        program.kind = ProgramKind::Map;
        program.insts = vec![Inst::Input(0), Inst::Unary(UnOp::Exp, 0)];
        assert_eq!(program_ops(&program, 4097, 4097), Some(4097));
        assert_eq!(program_ops(&program, 4097, 0), None);
    }
}
