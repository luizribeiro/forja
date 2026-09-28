//! NaN-sensitive generated-program semantics.

use forja_core::{
    DType,
    program::{BinOp, Inst, Program, ProgramKind, UnOp},
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_program_backends_agree, program::ProgramCase};

const COMPARISONS: [BinOp; 6] = [
    BinOp::Lt,
    BinOp::Le,
    BinOp::Eq,
    BinOp::Ne,
    BinOp::Ge,
    BinOp::Gt,
];

#[test]
fn map_nan_semantics_are_context_independent() {
    assert_nan_semantics(ProgramKind::Map, &[7]);
}

#[test]
fn resident_row_nan_semantics_are_context_independent() {
    assert_nan_semantics(ProgramKind::Row, &[1, 33]);
}

#[test]
fn rereading_row_nan_semantics_are_context_independent() {
    assert_nan_semantics(ProgramKind::Row, &[1, 1025]);
}

fn assert_nan_semantics(kind: ProgramKind, shape: &[u32]) {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for op in COMPARISONS {
        for operands in [(0, 1), (1, 0), (0, 0)] {
            assert_case(&cpu, &metal, comparison_program(kind, op, operands), shape);
        }
    }
    for op in [BinOp::Min, BinOp::Max] {
        for operands in [(0, 1), (1, 0)] {
            assert_case(&cpu, &metal, extreme_program(kind, op, operands), shape);
        }
        assert_case(&cpu, &metal, large_extreme_program(kind, op), shape);
    }
    assert_case(&cpu, &metal, large_comparison_program(kind), shape);
}

fn assert_case(cpu: &CpuBackend, metal: &MetalBackend, program: Program, shape: &[u32]) {
    let elements = shape.iter().copied().map(u64::from).product::<u64>();
    let input = (0..elements)
        .flat_map(|_| f32::NAN.to_le_bytes())
        .collect::<Vec<_>>();
    let outputs = [DType::F32, DType::F16, DType::BF16]
        .map(|dtype| TensorSpec::contiguous(dtype, shape))
        .to_vec();
    let case = ProgramCase::new(
        program,
        shape.to_vec(),
        vec![TensorSpec::initialized(DType::F32, shape, input)],
        outputs,
    );
    assert_program_backends_agree(cpu, metal, &case).unwrap();
}

fn comparison_program(kind: ProgramKind, op: BinOp, operands: (u32, u32)) -> Program {
    Program {
        kind,
        insts: vec![
            Inst::Input(0),
            Inst::Const(0.0),
            Inst::Const(1.0),
            Inst::Const(2.0),
            Inst::Binary(op, operands.0, operands.1),
            Inst::Select(4, 2, 3),
        ],
        outputs: vec![(0, 5), (1, 5), (2, 5)],
    }
}

fn extreme_program(kind: ProgramKind, op: BinOp, operands: (u32, u32)) -> Program {
    Program {
        kind,
        insts: vec![
            Inst::Input(0),
            Inst::Const(0.0),
            Inst::Binary(op, operands.0, operands.1),
        ],
        outputs: vec![(0, 2), (1, 2), (2, 2)],
    }
}

fn large_comparison_program(kind: ProgramKind) -> Program {
    let mut insts = vec![
        Inst::Input(0),
        Inst::Const(0.0),
        Inst::Const(1.0),
        Inst::Const(2.0),
    ];
    let mut selected = Vec::with_capacity(18);
    for op in COMPARISONS {
        for operands in [(0, 1), (1, 0), (0, 0)] {
            let comparison = u32::try_from(insts.len()).unwrap();
            insts.push(Inst::Binary(op, operands.0, operands.1));
            selected.push(u32::try_from(insts.len()).unwrap());
            insts.push(Inst::Select(comparison, 2, 3));
        }
    }
    let mut total = selected[0];
    for value in &selected[1..] {
        let next = u32::try_from(insts.len()).unwrap();
        insts.push(Inst::Binary(BinOp::Add, total, *value));
        total = next;
    }
    insts.push(Inst::Binary(BinOp::Min, 0, 1));
    insts.push(Inst::Binary(BinOp::Max, 0, 1));
    while insts.len() < 64 {
        let next = u32::try_from(insts.len()).unwrap();
        insts.push(Inst::Unary(UnOp::Neg, total));
        total = next;
    }
    Program {
        kind,
        insts,
        outputs: vec![(0, total), (1, total), (2, total)],
    }
}

fn large_extreme_program(kind: ProgramKind, op: BinOp) -> Program {
    let mut insts = vec![Inst::Input(0), Inst::Const(0.0), Inst::Binary(op, 0, 1)];
    let mut value = 2;
    while insts.len() < 64 {
        let next = u32::try_from(insts.len()).unwrap();
        insts.push(Inst::Unary(UnOp::Neg, value));
        value = next;
    }
    Program {
        kind,
        insts,
        outputs: vec![(0, value), (1, value), (2, value)],
    }
}
