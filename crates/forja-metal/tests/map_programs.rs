//! Map-program GPU differential tests.

mod common;

use std::time::Duration;

use forja_core::{
    Backend, CommandList, DType, Op, Slice, Submission, Tensor, ViewOp,
    program::{BinOp, Inst, Program, ProgramKind, UnOp, ValidatedProgram},
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{
    DeterministicValues, assert_outputs_agree, assert_program_backends_agree,
    program::map_programs,
    representative::{half_split_rope, residual_add, silu_mul},
};
use proptest::{
    prelude::*,
    test_runner::{RngSeed, TestCaseError},
};

use common::{GROSS_REGRESSION_RATIO, comparative_gpu_times, report_intervals};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        rng_seed: RngSeed::Fixed(0x510e_527f_ade6_82d1),
        ..ProptestConfig::default()
    })]

    #[test]
    fn metal_map_programs_match_the_interpreter(case in map_programs(64)) {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let report = assert_program_backends_agree(&cpu, &metal, &case)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        report_intervals(report);
    }
}

#[test]
fn transcendentals_stay_within_dtype_tolerances() {
    let program = transcendental_program().validate().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let mut values = DeterministicValues::new(0xa54f_f53a_5f1d_36f1);
        let bytes = (0..4097)
            .flat_map(|_| values.next_f32().to_le_bytes())
            .collect::<Vec<_>>();
        let expected = run_program(&cpu, &program, dtype, &bytes);
        let actual = run_program(&metal, &program, dtype, &bytes);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_outputs_agree(dtype, expected, &actual).unwrap();
        }
    }
}

#[test]
fn metal_map_preserves_scalar_corner_semantics() {
    let program = Program {
        kind: ProgramKind::Map,
        insts: vec![
            Inst::Input(0),
            Inst::Input(1),
            Inst::Binary(BinOp::Min, 0, 1),
            Inst::Binary(BinOp::Max, 0, 1),
            Inst::Cast(forja_core::program::ValueType::U32, 0),
            Inst::Cast(forja_core::program::ValueType::F32, 4),
            Inst::Input(2),
            Inst::Binary(BinOp::Add, 6, 6),
            Inst::Cast(forja_core::program::ValueType::F32, 7),
        ],
        outputs: vec![(0, 2), (1, 3), (2, 5), (3, 8)],
    }
    .validate()
    .unwrap();
    let floats = [
        f32::NAN,
        -1.0,
        0.0,
        1.0,
        f32::INFINITY,
        4_294_967_296.0,
        4_294_967_040.0,
    ];
    let others = [1.0, f32::NAN, 2.0, -2.0, 3.0, 0.0, 5.0];
    let integers = [u32::MAX, 1, 2, u32::MAX - 1, 17, 1 << 31, 0];
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let expected = run_corner_program(&cpu, &program, &floats, &others, &integers);
    let actual = run_corner_program(&metal, &program, &floats, &others, &integers);
    for (expected, actual) in expected.iter().zip(actual) {
        for (expected, actual) in expected
            .as_chunks::<4>()
            .0
            .iter()
            .zip(actual.as_chunks::<4>().0)
        {
            let expected = f32::from_le_bytes(*expected);
            let actual = f32::from_le_bytes(*actual);
            assert!(
                expected.to_bits() == actual.to_bits() || expected.is_nan() && actual.is_nan(),
                "expected {expected:?}, got {actual:?}"
            );
        }
    }
}

#[test]
fn metal_map_preserves_nan_dependent_control_flow() {
    let program = Program {
        kind: ProgramKind::Map,
        insts: vec![
            Inst::Input(0),
            Inst::Const(1.0),
            Inst::Binary(BinOp::Min, 0, 1),
            Inst::Binary(BinOp::Max, 2, 1),
            Inst::Binary(BinOp::Eq, 3, 3),
            Inst::Const(17.0),
            Inst::Const(29.0),
            Inst::Select(4, 5, 6),
        ],
        outputs: vec![(0, 7)],
    }
    .validate()
    .unwrap();
    let backend = MetalBackend::new().unwrap();
    let input = backend.alloc(DType::F32, &[1]).unwrap();
    backend.write(&input, &f32::NAN.to_le_bytes()).unwrap();
    let output = backend.alloc(DType::F32, &[1]).unwrap();
    let prepared =
        forja_testing::prepare_program(&backend, &program, &[&input], &[&output]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, &[&input], &[&output])
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    let actual = f32::from_le_bytes(backend.read(&output).unwrap().try_into().unwrap());
    assert_eq!(actual.to_bits(), 29.0_f32.to_bits());
}

#[test]
fn metal_transcendentals_remain_defined_for_finite_inputs() {
    let program = Program {
        kind: ProgramKind::Map,
        insts: vec![
            Inst::Const(58.5625),
            Inst::Unary(UnOp::Tanh, 0),
            Inst::Unary(UnOp::Log, 1),
        ],
        outputs: vec![(0, 2)],
    }
    .validate()
    .unwrap();
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let expected = cpu.alloc(DType::BF16, &[1]).unwrap();
    let actual = metal.alloc(DType::BF16, &[1]).unwrap();
    let cpu_program = forja_testing::prepare_program(&cpu, &program, &[], &[&expected]).unwrap();
    let mut cpu_commands = CommandList::new();
    cpu_commands
        .dispatch_kernel(&cpu_program, &[], &[&expected])
        .unwrap();
    cpu.submit(cpu_commands).unwrap().wait().unwrap();
    let metal_program = forja_testing::prepare_program(&metal, &program, &[], &[&actual]).unwrap();
    let mut metal_commands = CommandList::new();
    metal_commands
        .dispatch_kernel(&metal_program, &[], &[&actual])
        .unwrap();
    metal.submit(metal_commands).unwrap().wait().unwrap();
    assert_outputs_agree(
        DType::BF16,
        &cpu.read(&expected).unwrap(),
        &metal.read(&actual).unwrap(),
    )
    .unwrap();
}

#[test]
fn metal_transcendentals_preserve_numeric_classes() {
    let program = Program {
        kind: ProgramKind::Map,
        insts: vec![
            Inst::Input(0),
            Inst::Unary(UnOp::Log, 0),
            Inst::Unary(UnOp::Sin, 1),
            Inst::Unary(UnOp::Sqrt, 0),
            Inst::Unary(UnOp::Rsqrt, 0),
        ],
        outputs: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
    }
    .validate()
    .unwrap();
    let values = [
        -1.0,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let bytes = values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let expected = run_small_program(&cpu, &program, &bytes, values.len());
    let actual = run_small_program(&metal, &program, &bytes, values.len());
    for (expected, actual) in expected.iter().zip(actual) {
        assert_outputs_agree(DType::F32, expected, &actual).unwrap();
    }
}

fn run_corner_program<B: Backend>(
    backend: &B,
    program: &ValidatedProgram,
    floats: &[f32],
    others: &[f32],
    integers: &[u32],
) -> Vec<Vec<u8>> {
    let shape = [u32::try_from(floats.len()).unwrap()];
    let inputs = [
        backend.alloc(DType::F32, &shape).unwrap(),
        backend.alloc(DType::F32, &shape).unwrap(),
        backend.alloc(DType::U32, &shape).unwrap(),
    ];
    backend
        .write(
            &inputs[0],
            &floats
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    backend
        .write(
            &inputs[1],
            &others
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    backend
        .write(
            &inputs[2],
            &integers
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let outputs = (0..4)
        .map(|_| backend.alloc(DType::F32, &shape).unwrap())
        .collect::<Vec<_>>();
    let input_refs = inputs.iter().collect::<Vec<_>>();
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let prepared =
        forja_testing::prepare_program(backend, program, &input_refs, &output_refs).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, &input_refs, &output_refs)
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    outputs
        .iter()
        .map(|output| backend.read(output).unwrap())
        .collect()
}

fn transcendental_program() -> Program {
    Program {
        kind: ProgramKind::Map,
        insts: vec![
            Inst::Input(0),
            Inst::Unary(UnOp::Abs, 0),
            Inst::Const(1.25),
            Inst::Binary(BinOp::Add, 1, 2),
            Inst::Unary(UnOp::Exp, 0),
            Inst::Unary(UnOp::Log, 3),
            Inst::Unary(UnOp::Sqrt, 3),
            Inst::Unary(UnOp::Rsqrt, 3),
            Inst::Unary(UnOp::Sin, 0),
            Inst::Unary(UnOp::Cos, 0),
            Inst::Unary(UnOp::Tanh, 0),
            Inst::Const(1.5),
            Inst::Binary(BinOp::Pow, 3, 11),
            Inst::Binary(BinOp::Add, 4, 5),
            Inst::Binary(BinOp::Add, 6, 7),
            Inst::Binary(BinOp::Add, 8, 9),
            Inst::Binary(BinOp::Add, 10, 12),
        ],
        outputs: vec![(0, 13), (1, 14), (2, 15), (3, 16)],
    }
}

fn run_program<B: Backend>(
    backend: &B,
    program: &ValidatedProgram,
    dtype: DType,
    input_bytes: &[u8],
) -> Vec<Vec<u8>> {
    let input = backend.alloc(DType::F32, &[4097]).unwrap();
    backend.write(&input, input_bytes).unwrap();
    let outputs = (0..4)
        .map(|_| backend.alloc(dtype, &[4097]).unwrap())
        .collect::<Vec<_>>();
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let prepared =
        forja_testing::prepare_program(backend, program, &[&input], &output_refs).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, &[&input], &output_refs)
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    outputs
        .iter()
        .map(|output| backend.read(output).unwrap())
        .collect()
}

fn run_small_program<B: Backend>(
    backend: &B,
    program: &ValidatedProgram,
    input_bytes: &[u8],
    len: usize,
) -> Vec<Vec<u8>> {
    let shape = [u32::try_from(len).unwrap()];
    let input = backend.alloc(DType::F32, &shape).unwrap();
    backend.write(&input, input_bytes).unwrap();
    let outputs = (0..4)
        .map(|_| backend.alloc(DType::F32, &shape).unwrap())
        .collect::<Vec<_>>();
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let prepared =
        forja_testing::prepare_program(backend, program, &[&input], &output_refs).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, &[&input], &output_refs)
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    outputs
        .iter()
        .map(|output| backend.read(output).unwrap())
        .collect()
}

#[test]
fn representative_maps_match_without_gross_kernel_regression() {
    let backend = MetalBackend::new().unwrap();
    let timings = [
        binary_timing(
            &backend,
            "residual-add",
            Op::Add,
            &residual_add(),
            &[128, 1024],
            1,
        ),
        binary_timing(
            &backend,
            "silu-mul",
            Op::SiluMul,
            &silu_mul(),
            &[33, 3072],
            2,
        ),
        rope_timing(&backend),
    ];
    eprintln!("operation,trusted_ns,program_ns,ratio");
    let mut failures = Vec::new();
    for timing in timings {
        let ratio = timing.program.as_secs_f64() / timing.trusted.as_secs_f64();
        eprintln!(
            "{},{},{},{ratio:.3}",
            timing.name,
            timing.trusted.as_nanos(),
            timing.program.as_nanos()
        );
        if ratio > GROSS_REGRESSION_RATIO {
            failures.push(format!("{}={ratio:.3}", timing.name));
        }
    }
    assert!(
        failures.is_empty(),
        "map ratios exceeded {GROSS_REGRESSION_RATIO}: {failures:?}"
    );
}

struct Timing {
    name: &'static str,
    trusted: Duration,
    program: Duration,
}

fn binary_timing(
    backend: &MetalBackend,
    name: &'static str,
    op: Op,
    source: &Program,
    shape: &[u32],
    seed: u64,
) -> Timing {
    let left = initialized_f32(backend, shape, seed);
    let right = initialized_f32(backend, shape, seed + 1);
    let trusted_output = backend.alloc(DType::F32, shape).unwrap();
    let program_output = backend.alloc(DType::F32, shape).unwrap();
    let program = source.validate().unwrap();
    let _prepared =
        forja_testing::prepare_program(backend, &program, &[&left, &right], &[&program_output])
            .unwrap();
    run(
        backend,
        trusted_commands(op, &[&left, &right], &trusted_output),
    );
    run(
        backend,
        program_commands(backend, &program, &[&left, &right], &[&program_output]),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&trusted_output).unwrap(),
        &backend.read(&program_output).unwrap(),
    )
    .unwrap();
    let (trusted, program) = comparative_gpu_times(
        backend,
        || trusted_commands(op, &[&left, &right], &trusted_output),
        || program_commands(backend, &program, &[&left, &right], &[&program_output]),
    );
    Timing {
        name,
        trusted,
        program,
    }
}

fn rope_timing(backend: &MetalBackend) -> Timing {
    let values = initialized_f32(backend, &[33, 16, 128], 4);
    let positions = backend.alloc(DType::U32, &[33]).unwrap();
    backend
        .write(
            &positions,
            &(0_u32..33).flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
        )
        .unwrap();
    let reshaped = backend
        .view(&values, ViewOp::Reshape(vec![33, 16, 2, 64]))
        .unwrap();
    let first = half(backend, &reshaped, 0);
    let second = half(backend, &reshaped, 1);
    let trusted_output = backend.alloc(DType::F32, &[33, 16, 128]).unwrap();
    let trusted_reshaped = backend
        .view(&trusted_output, ViewOp::Reshape(vec![33, 16, 2, 64]))
        .unwrap();
    let expected_first = half(backend, &trusted_reshaped, 0);
    let expected_second = half(backend, &trusted_reshaped, 1);
    let program_first = backend.alloc(DType::F32, &[33, 16, 1, 64]).unwrap();
    let program_second = backend.alloc(DType::F32, &[33, 16, 1, 64]).unwrap();
    let program = half_split_rope().validate().unwrap();
    let _prepared = forja_testing::prepare_program(
        backend,
        &program,
        &[&first, &second],
        &[&program_first, &program_second],
    )
    .unwrap();
    run(
        backend,
        trusted_commands(
            Op::Rope { theta: 1_000_000.0 },
            &[&values, &positions],
            &trusted_output,
        ),
    );
    run(
        backend,
        program_commands(
            backend,
            &program,
            &[&first, &second],
            &[&program_first, &program_second],
        ),
    );
    for (expected, actual) in [
        (&expected_first, &program_first),
        (&expected_second, &program_second),
    ] {
        assert_outputs_agree(
            DType::F32,
            &backend.read(expected).unwrap(),
            &backend.read(actual).unwrap(),
        )
        .unwrap();
    }
    let (trusted, program) = comparative_gpu_times(
        backend,
        || {
            trusted_commands(
                Op::Rope { theta: 1_000_000.0 },
                &[&values, &positions],
                &trusted_output,
            )
        },
        || {
            program_commands(
                backend,
                &program,
                &[&first, &second],
                &[&program_first, &program_second],
            )
        },
    );
    Timing {
        name: "rope",
        trusted,
        program,
    }
}

fn initialized_f32(backend: &MetalBackend, shape: &[u32], seed: u64) -> Tensor {
    let tensor = backend.alloc(DType::F32, shape).unwrap();
    let mut values = DeterministicValues::new(seed);
    let bytes = (0..tensor.layout().element_count())
        .flat_map(|_| values.next_f32().to_le_bytes())
        .collect::<Vec<_>>();
    backend.write(&tensor, &bytes).unwrap();
    tensor
}

fn half(backend: &MetalBackend, tensor: &Tensor, index: u32) -> Tensor {
    backend
        .view(
            tensor,
            ViewOp::Slice(vec![
                Slice::new(0, 33, 1).unwrap(),
                Slice::new(0, 16, 1).unwrap(),
                Slice::new(index, 1, 1).unwrap(),
                Slice::new(0, 64, 1).unwrap(),
            ]),
        )
        .unwrap()
}

fn trusted_commands(op: Op, inputs: &[&Tensor], output: &Tensor) -> CommandList {
    let mut commands = CommandList::new();
    commands.dispatch(op, inputs, output).unwrap();
    commands
}

fn program_commands(
    backend: &MetalBackend,
    program: &ValidatedProgram,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
) -> CommandList {
    let prepared = forja_testing::prepare_program(backend, program, inputs, outputs).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, inputs, outputs)
        .unwrap();
    commands
}

fn run(backend: &MetalBackend, commands: CommandList) {
    backend.submit(commands).unwrap().wait().unwrap();
}
