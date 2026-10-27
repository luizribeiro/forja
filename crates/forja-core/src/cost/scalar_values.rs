use super::{KernelCost, tests::tensor};
use crate::{
    CommandList, DType, Op, Tensor,
    program::{EpilogueSignature, Inst, Program, ProgramKind, UnOp, prepare_epilogue},
};

fn cost(op: Op, inputs: &[&Tensor], outputs: &[&Tensor]) -> KernelCost {
    let mut commands = CommandList::new();
    commands.dispatch_many(op, inputs, outputs).unwrap();
    KernelCost::of(commands.last_dispatch().unwrap()).unwrap()
}

#[test]
fn norm_and_rope_scalar_work_matches_hand_counts() {
    let input = tensor(1, DType::F32, &[7, 33]);
    let weights = tensor(2, DType::F32, &[33]);
    let output = tensor(3, DType::F32, &[7, 33]);
    assert_eq!(
        cost(Op::RmsNorm { eps: 1e-5 }, &[&input, &weights], &[&output]).scalar_ops,
        945
    );
    let input = tensor(4, DType::BF16, &[7, 16, 128]);
    let positions = tensor(5, DType::U32, &[7]);
    let output = tensor(6, DType::BF16, &[7, 16, 128]);
    assert_eq!(
        cost(
            Op::Rope { theta: 10_000.0 },
            &[&input, &positions],
            &[&output]
        )
        .scalar_ops,
        43_008
    );
}

#[test]
fn selection_scalar_work_includes_optional_top_k_normalization() {
    let logits = tensor(1, DType::F32, &[7, 33]);
    let argmax = tensor(2, DType::U32, &[7]);
    assert_eq!(
        cost(Op::Argmax { chunk_width: 2048 }, &[&logits], &[&argmax]).scalar_ops,
        224
    );
    let values = tensor(3, DType::F32, &[7, 8]);
    let indices = tensor(4, DType::U32, &[7, 8]);
    for (normalize, expected) in [(false, 1848), (true, 2016)] {
        assert_eq!(
            cost(
                Op::TopK {
                    k: 8,
                    normalize,
                    chunk_width: 2048
                },
                &[&logits],
                &[&values, &indices]
            )
            .scalar_ops,
            expected
        );
    }
}

#[test]
fn sampling_counts_logits_and_reads_the_mask_separately() {
    let logits = tensor(1, DType::F32, &[7, 33]);
    let controls = tensor(2, DType::U32, &[5]);
    let mask = tensor(3, DType::U32, &[2]);
    let output = tensor(4, DType::U32, &[7]);
    let operations = [
        Op::Sample {
            position: 7,
            chunk_width: 2048,
            fast_top_k: 8,
            fast_max_chunks: 32,
            rejection_rounds: 4,
        },
        Op::SampleMasked {
            position: 7,
            chunk_width: 2048,
            fast_top_k: 8,
            fast_max_chunks: 32,
            rejection_rounds: 4,
        },
    ];
    for (op, inputs, reads) in [
        (operations[0], vec![&logits, &controls], 944),
        (operations[1], vec![&logits, &controls, &mask], 952),
    ] {
        let cost = cost(op, &inputs, &[&output]);
        assert_eq!(cost.scalar_ops, 924);
        assert_eq!((cost.read_bytes, cost.write_bytes), (reads, 28));
    }
}

#[test]
fn fused_epilogue_counts_scalar_work_separately_from_matrix_flops() {
    let input = tensor(1, DType::F32, &[7, 33]);
    let weights = tensor(2, DType::F32, &[33, 7]);
    let output = tensor(3, DType::F32, &[7, 7]);
    let epilogue = prepare_epilogue(
        Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0), Inst::Unary(UnOp::Exp, 0)],
            outputs: vec![(0, 1)],
        }
        .validate()
        .unwrap(),
        EpilogueSignature::new(1, 2, vec![], DType::F32),
    )
    .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_epilogue(Op::Matmul, &[&input, &weights], &output, &epilogue, &[])
        .unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(cost.scalar_ops, 49);
    assert_eq!(cost.flops, 3234);
}
