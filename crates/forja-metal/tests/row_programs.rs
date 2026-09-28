//! Row-program GPU differential and performance tests.

mod common;

use std::time::Duration;

use forja_core::{
    Backend, CommandList, DType, Op, Submission, Tensor, ViewOp,
    program::{Inst, Program, ProgramKind, RedOp, ValidatedProgram},
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{
    DeterministicValues, assert_outputs_agree, assert_program_backends_agree,
    program::{ProgramCase, row_programs, stable_row_programs},
    representative::{residual_rms_norm, rms_norm, softmax},
};
use proptest::{prelude::*, test_runner::TestCaseError};

use common::median_gpu_time;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn metal_row_programs_match_the_interpreter(case in row_programs(64)) {
        compare_program(&case)?;
    }

    #[test]
    fn stable_metal_row_programs_match_the_interpreter(case in stable_row_programs(64)) {
        compare_program(&case)?;
    }
}

fn compare_program(case: &ProgramCase) -> Result<(), TestCaseError> {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().map_err(|error| TestCaseError::fail(error.to_string()))?;
    assert_program_backends_agree(&cpu, &metal, case)
        .map_err(|error| TestCaseError::fail(error.to_string()))
}

#[test]
fn row_reductions_propagate_nan_and_handle_singletons() {
    let program = Program {
        kind: ProgramKind::Row,
        insts: vec![
            Inst::Input(0),
            Inst::Reduce(RedOp::Sum, 0),
            Inst::Reduce(RedOp::Max, 0),
            Inst::Reduce(RedOp::Min, 0),
        ],
        outputs: vec![(0, 1), (1, 2), (2, 3)],
    }
    .validate()
    .unwrap();
    for (shape, values) in [
        (vec![2, 1], vec![3.0, f32::NAN]),
        (
            vec![2, 7],
            vec![
                f32::NAN,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ],
        ),
    ] {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let expected = run_with_values(&cpu, &program, &shape, &values);
        let actual = run_with_values(&metal, &program, &shape, &values);
        for (expected, actual) in expected.iter().zip(&actual) {
            assert_outputs_agree(DType::F32, expected, actual).unwrap();
        }
        let width = usize::try_from(*shape.last().unwrap()).unwrap();
        for output in &actual {
            for value in output.as_chunks::<4>().0.iter().skip(width) {
                assert!(f32::from_le_bytes(*value).is_nan());
            }
        }
    }
}

#[test]
fn representative_rows_match_and_meet_kernel_time() {
    let backend = MetalBackend::new().unwrap();
    let timings = [rms_timing(&backend), softmax_timing(&backend)];
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
        if ratio > 1.05 {
            failures.push(format!("{}={ratio:.3}", timing.name));
        }
    }
    assert!(
        failures.is_empty(),
        "row ratios exceeded 1.05: {failures:?}"
    );
}

#[test]
fn fused_residual_norm_matches_and_beats_two_dispatches() {
    let backend = MetalBackend::new().unwrap();
    let shape = [128, 1024];
    let residual = initialized_f32(&backend, &shape, 7);
    let update = initialized_f32(&backend, &shape, 8);
    let weight = initialized_f32(&backend, &[1024], 9);
    let broadcast_weight = backend
        .view(&weight, ViewOp::Broadcast(shape.to_vec()))
        .unwrap();
    let intermediate = backend.alloc(DType::F32, &shape).unwrap();
    let trusted_output = backend.alloc(DType::F32, &shape).unwrap();
    let program_output = backend.alloc(DType::F32, &shape).unwrap();
    let program = residual_rms_norm(1.0e-6).validate().unwrap();
    run(
        &backend,
        residual_norm_commands(&residual, &update, &weight, &intermediate, &trusted_output),
    );
    run(
        &backend,
        program_commands(
            &program,
            &[&residual, &update, &broadcast_weight],
            &[&program_output],
        ),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&trusted_output).unwrap(),
        &backend.read(&program_output).unwrap(),
    )
    .unwrap();
    let unfused = median_gpu_time(&backend, || {
        residual_norm_commands(&residual, &update, &weight, &intermediate, &trusted_output)
    });
    let fused = median_gpu_time(&backend, || {
        program_commands(
            &program,
            &[&residual, &update, &broadcast_weight],
            &[&program_output],
        )
    });
    let ratio = fused.as_secs_f64() / unfused.as_secs_f64();
    eprintln!("operation,unfused_ns,fused_ns,ratio");
    eprintln!(
        "residual-rms-norm,{},{},{ratio:.3}",
        unfused.as_nanos(),
        fused.as_nanos()
    );
    assert!(ratio <= 1.05, "fused ratio exceeded 1.05: {ratio:.3}");
}

struct Timing {
    name: &'static str,
    trusted: Duration,
    program: Duration,
}

fn rms_timing(backend: &MetalBackend) -> Timing {
    let shape = [128, 1024];
    let input = initialized_f32(backend, &shape, 1);
    let weight = initialized_f32(backend, &[1024], 2);
    let broadcast_weight = backend
        .view(&weight, ViewOp::Broadcast(shape.to_vec()))
        .unwrap();
    let expected = backend.alloc(DType::F32, &shape).unwrap();
    let actual = backend.alloc(DType::F32, &shape).unwrap();
    let program = rms_norm(1.0e-6).validate().unwrap();
    run(
        backend,
        trusted_commands(Op::RmsNorm { eps: 1.0e-6 }, &[&input, &weight], &expected),
    );
    run(
        backend,
        program_commands(&program, &[&input, &broadcast_weight], &[&actual]),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&expected).unwrap(),
        &backend.read(&actual).unwrap(),
    )
    .unwrap();
    Timing {
        name: "rms-norm",
        trusted: median_gpu_time(backend, || {
            trusted_commands(Op::RmsNorm { eps: 1.0e-6 }, &[&input, &weight], &expected)
        }),
        program: median_gpu_time(backend, || {
            program_commands(&program, &[&input, &broadcast_weight], &[&actual])
        }),
    }
}

fn softmax_timing(backend: &MetalBackend) -> Timing {
    let shape = [128, 1024];
    let input = initialized_f32(backend, &shape, 3);
    let expected = backend.alloc(DType::F32, &shape).unwrap();
    let actual = backend.alloc(DType::F32, &shape).unwrap();
    let program = softmax().validate().unwrap();
    run(backend, trusted_commands(Op::Softmax, &[&input], &expected));
    run(backend, program_commands(&program, &[&input], &[&actual]));
    assert_outputs_agree(
        DType::F32,
        &backend.read(&expected).unwrap(),
        &backend.read(&actual).unwrap(),
    )
    .unwrap();
    Timing {
        name: "softmax",
        trusted: median_gpu_time(backend, || {
            trusted_commands(Op::Softmax, &[&input], &expected)
        }),
        program: median_gpu_time(backend, || {
            program_commands(&program, &[&input], &[&actual])
        }),
    }
}

fn run_with_values<B: Backend>(
    backend: &B,
    program: &ValidatedProgram,
    shape: &[u32],
    values: &[f32],
) -> Vec<Vec<u8>> {
    let input = backend.alloc(DType::F32, shape).unwrap();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    backend.write(&input, &bytes).unwrap();
    let outputs = (0..3)
        .map(|_| backend.alloc(DType::F32, shape).unwrap())
        .collect::<Vec<_>>();
    run(
        backend,
        program_commands(program, &[&input], &outputs.iter().collect::<Vec<_>>()),
    );
    outputs
        .iter()
        .map(|output| backend.read(output).unwrap())
        .collect()
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

fn trusted_commands(op: Op, inputs: &[&Tensor], output: &Tensor) -> CommandList {
    let mut commands = CommandList::new();
    commands.dispatch(op, inputs, output).unwrap();
    commands
}

fn program_commands(
    program: &ValidatedProgram,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
) -> CommandList {
    let mut commands = CommandList::new();
    commands.dispatch_program(program, inputs, outputs).unwrap();
    commands
}

fn residual_norm_commands(
    residual: &Tensor,
    update: &Tensor,
    weight: &Tensor,
    intermediate: &Tensor,
    output: &Tensor,
) -> CommandList {
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::Add, &[residual, update], intermediate)
        .unwrap();
    commands
        .dispatch(Op::RmsNorm { eps: 1.0e-6 }, &[intermediate, weight], output)
        .unwrap();
    commands
}

fn run<B: Backend>(backend: &B, commands: CommandList) {
    backend.submit(commands).unwrap().wait().unwrap();
}
