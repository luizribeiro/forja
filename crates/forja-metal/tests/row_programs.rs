//! Row-program GPU differential and performance tests.

mod common;

use std::time::Duration;

use forja_core::{
    Backend, CommandList, DType, Op, Slice, Submission, Tensor, ViewOp,
    program::{BinOp, Inst, Program, ProgramKind, RedOp, UnOp, ValidatedProgram},
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{
    DeterministicValues, TensorSpec, assert_outputs_agree, assert_program_backends_agree,
    program::{ProgramCase, row_programs, stable_row_programs},
    program_interval_report,
    representative::{qk_norm_rope, residual_rms_norm, rms_norm, softmax},
};
use proptest::{
    prelude::*,
    test_runner::{FileFailurePersistence, RngSeed, TestCaseError},
};

use common::{comparative_gpu_times, report_intervals};

proptest! {
    #![proptest_config(general_row_program_config())]

    #[test]
    fn metal_row_programs_match_the_interpreter(case in row_programs(64)) {
        compare_program(&case)?;
    }
}

proptest! {
    #![proptest_config(stable_row_program_config())]

    #[test]
    fn stable_metal_row_programs_match_the_interpreter(case in stable_row_programs(64)) {
        compare_program(&case)?;
    }
}

/// Random programs can be intrinsically ill-conditioned, so up to 10% may be
/// discarded while every program still faces the unchanged 1% element guard.
fn general_row_program_config() -> ProptestConfig {
    row_program_config(4, 8)
}

fn stable_row_program_config() -> ProptestConfig {
    row_program_config(2, 4)
}

fn row_program_config(normal_reject_limit: u32, exploration_reject_limit: u32) -> ProptestConfig {
    let explore = std::env::var_os("FORJA_PROPTEST_EXPLORE").is_some();
    let failure_file = if explore {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/row_programs-exploration.txt"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/row_programs.txt"
        )
    };
    ProptestConfig {
        cases: if explore { 64 } else { 32 },
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(failure_file))),
        rng_seed: if explore {
            RngSeed::Random
        } else {
            RngSeed::Fixed(0x6a09_e667_f3bc_c909)
        },
        max_global_rejects: if explore {
            exploration_reject_limit
        } else {
            normal_reject_limit
        },
        ..ProptestConfig::default()
    }
}

fn compare_program(case: &ProgramCase) -> Result<(), TestCaseError> {
    let preflight =
        program_interval_report(case).map_err(|error| TestCaseError::fail(error.to_string()))?;
    if !intervals_are_judgeable(preflight) {
        report_intervals(preflight);
        prop_assume!(false);
    }
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().map_err(|error| TestCaseError::fail(error.to_string()))?;
    let report = assert_program_backends_agree(&cpu, &metal, case)
        .map_err(|error| TestCaseError::fail(error.to_string()))?;
    report_intervals(report);
    Ok(())
}

const fn intervals_are_judgeable(report: forja_testing::IntervalReport) -> bool {
    report.vacuous_elements.saturating_mul(100) < report.total_elements
}

#[test]
fn interval_preflight_rejects_one_percent_vacuity() {
    let report = forja_testing::IntervalReport {
        vacuous_elements: 1,
        total_elements: 100,
        ..forja_testing::IntervalReport::default()
    };
    assert!(!intervals_are_judgeable(report));

    let report = forja_testing::IntervalReport {
        total_elements: 101,
        ..report
    };
    assert!(intervals_are_judgeable(report));
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
fn wide_constant_sum_uses_reduction_interval() {
    let value = 0.558_304_9_f32;
    let case = ProgramCase::new(
        Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Const(value), Inst::Reduce(RedOp::Sum, 0)],
            outputs: vec![(0, 1)],
        },
        vec![1, 4097],
        Vec::new(),
        vec![TensorSpec::contiguous(DType::F32, &[1, 4097])],
    );
    let report =
        assert_program_backends_agree(&CpuBackend::new(), &MetalBackend::new().unwrap(), &case)
            .unwrap();
    assert_eq!(report.reduction_steps, 4096);
    assert_eq!(report.vacuous_elements, 0);
    assert!(report.max_relative_width < 2.0e-3);
}

#[test]
fn ambiguous_tanh_predicate_accepts_either_branch() {
    let mut values = vec![1.0_f32; 1024];
    values[252] = 0.000_701_427_46;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let case = ProgramCase::new(
        Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Const(-2.0),
                Inst::Unary(UnOp::Tanh, 0),
                Inst::Binary(BinOp::Le, 0, 2),
                Inst::Select(3, 1, 0),
            ],
            outputs: vec![(0, 4)],
        },
        vec![1, 1024],
        vec![TensorSpec::initialized(DType::F32, &[1, 1024], bytes)],
        vec![TensorSpec::contiguous(DType::F16, &[1, 1024])],
    );
    let report =
        assert_program_backends_agree(&CpuBackend::new(), &MetalBackend::new().unwrap(), &case)
            .unwrap();
    assert_eq!(report.ambiguous_predicates, 1);
    assert_eq!(report.ambiguous_selects, 1);
    assert_eq!(report.vacuous_elements, 1);
    assert!(report.median_relative_width < 2.0e-2);
}

#[test]
fn extrema_cancellation_stays_inside_a_narrow_interval() {
    let values = [
        0.632_068_16_f32,
        0.989_548_9,
        -0.582_683_56,
        -0.281_108_38,
        -0.811_542_5,
        0.304_452_9,
        0.845_899_1,
        -0.888_599_16,
        0.172_425_27,
        -0.914_864_06,
        0.697_305_2,
        0.642_967,
        -0.986_603_74,
        -0.073_560_24,
        -0.227_872_37,
        0.885_685_2,
        -0.525_104_76,
        0.363_495_35,
        -0.673_333_9,
        -0.938_154_94,
        -0.894_051_55,
        0.055_157_66,
        -0.300_585_75,
        0.651_407_5,
        -0.306_084_16,
        -0.405_201_9,
        -0.951_931_95,
        -0.013_886_929,
        -0.681_762_7,
        -0.436_078_07,
        -0.125_971_56,
        0.106_915_236,
        0.486_760_38,
    ];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let case = ProgramCase::new(
        Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Unary(UnOp::Sin, 0),
                Inst::Reduce(RedOp::Max, 1),
                Inst::Binary(BinOp::Add, 1, 2),
                Inst::Reduce(RedOp::Min, 3),
            ],
            outputs: vec![(0, 4)],
        },
        vec![1, 33],
        vec![TensorSpec::initialized(DType::F32, &[1, 33], bytes)],
        vec![TensorSpec::contiguous(DType::F32, &[1, 33])],
    );
    let report =
        assert_program_backends_agree(&CpuBackend::new(), &MetalBackend::new().unwrap(), &case)
            .unwrap();
    assert_eq!(report.reduction_steps, 64);
    assert_eq!(report.vacuous_elements, 0);
    assert!(report.max_relative_width < 1.0e-3);
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
    let program_residual = backend.alloc(DType::F32, &shape).unwrap();
    let program_output = backend.alloc(DType::F32, &shape).unwrap();
    let program = residual_rms_norm(1.0e-6).validate().unwrap();
    let _prepared = forja_testing::prepare_program(
        &backend,
        &program,
        &[&residual, &update, &broadcast_weight],
        &[&program_residual, &program_output],
    )
    .unwrap();
    run(
        &backend,
        residual_norm_commands(&residual, &update, &weight, &intermediate, &trusted_output),
    );
    run(
        &backend,
        program_commands(
            &backend,
            &program,
            &[&residual, &update, &broadcast_weight],
            &[&program_residual, &program_output],
        ),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&trusted_output).unwrap(),
        &backend.read(&program_output).unwrap(),
    )
    .unwrap();
    let (unfused, fused) = comparative_gpu_times(
        &backend,
        || residual_norm_commands(&residual, &update, &weight, &intermediate, &trusted_output),
        || {
            program_commands(
                &backend,
                &program,
                &[&residual, &update, &broadcast_weight],
                &[&program_residual, &program_output],
            )
        },
    );
    let ratio = fused.as_secs_f64() / unfused.as_secs_f64();
    eprintln!("operation,unfused_ns,fused_ns,ratio");
    eprintln!(
        "residual-rms-norm,{},{},{ratio:.3}",
        unfused.as_nanos(),
        fused.as_nanos()
    );
    assert!(ratio <= 1.05, "fused ratio exceeded 1.05: {ratio:.3}");
}

#[test]
fn fused_qk_norm_rope_matches_and_beats_two_dispatches() {
    let backend = MetalBackend::new().unwrap();
    let values = initialized_f32(&backend, &[8, 16, 128], 10);
    let weight = initialized_f32(&backend, &[128], 11);
    let positions = position_tensor(&backend, DType::U32);
    let program_positions = position_tensor(&backend, DType::F32);
    let program_positions = backend
        .view(&program_positions, ViewOp::Reshape(vec![8, 1, 1]))
        .unwrap();
    let program_positions = backend
        .view(&program_positions, ViewOp::Broadcast(vec![8, 16, 64]))
        .unwrap();
    let first = qk_half(&backend, &values, 0);
    let second = qk_half(&backend, &values, 64);
    let first_weight = qk_weight(&backend, &weight, 0);
    let second_weight = qk_weight(&backend, &weight, 64);
    let normalized = backend.alloc(DType::F32, &[8, 16, 128]).unwrap();
    let expected = backend.alloc(DType::F32, &[8, 16, 128]).unwrap();
    let actual = backend.alloc(DType::F32, &[8, 16, 128]).unwrap();
    let actual_first = qk_half(&backend, &actual, 0);
    let actual_second = qk_half(&backend, &actual, 64);
    let program = qk_norm_rope(1.0e-6, 1_000_000.0).validate().unwrap();
    let _prepared = forja_testing::prepare_program(
        &backend,
        &program,
        &[
            &first,
            &second,
            &first_weight,
            &second_weight,
            &program_positions,
        ],
        &[&actual_first, &actual_second],
    )
    .unwrap();
    let trusted = || qk_norm_rope_commands(&values, &weight, &positions, &normalized, &expected);
    let fused = || {
        program_commands(
            &backend,
            &program,
            &[
                &first,
                &second,
                &first_weight,
                &second_weight,
                &program_positions,
            ],
            &[&actual_first, &actual_second],
        )
    };
    run(&backend, trusted());
    run(&backend, fused());
    assert_outputs_agree(
        DType::F32,
        &backend.read(&expected).unwrap(),
        &backend.read(&actual).unwrap(),
    )
    .unwrap();
    let (unfused, fused) = comparative_gpu_times(&backend, trusted, fused);
    let ratio = fused.as_secs_f64() / unfused.as_secs_f64();
    eprintln!("operation,unfused_ns,fused_ns,ratio");
    eprintln!(
        "qk-norm-rope,{},{},{ratio:.3}",
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
    let _prepared =
        forja_testing::prepare_program(backend, &program, &[&input, &broadcast_weight], &[&actual])
            .unwrap();
    run(
        backend,
        trusted_commands(Op::RmsNorm { eps: 1.0e-6 }, &[&input, &weight], &expected),
    );
    run(
        backend,
        program_commands(backend, &program, &[&input, &broadcast_weight], &[&actual]),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&expected).unwrap(),
        &backend.read(&actual).unwrap(),
    )
    .unwrap();
    let (trusted, program) = comparative_gpu_times(
        backend,
        || trusted_commands(Op::RmsNorm { eps: 1.0e-6 }, &[&input, &weight], &expected),
        || program_commands(backend, &program, &[&input, &broadcast_weight], &[&actual]),
    );
    Timing {
        name: "rms-norm",
        trusted,
        program,
    }
}

fn softmax_timing(backend: &MetalBackend) -> Timing {
    let shape = [128, 1024];
    let input = initialized_f32(backend, &shape, 3);
    let expected = backend.alloc(DType::F32, &shape).unwrap();
    let actual = backend.alloc(DType::F32, &shape).unwrap();
    let program = softmax().validate().unwrap();
    let _prepared =
        forja_testing::prepare_program(backend, &program, &[&input], &[&actual]).unwrap();
    run(backend, trusted_commands(Op::Softmax, &[&input], &expected));
    run(
        backend,
        program_commands(backend, &program, &[&input], &[&actual]),
    );
    assert_outputs_agree(
        DType::F32,
        &backend.read(&expected).unwrap(),
        &backend.read(&actual).unwrap(),
    )
    .unwrap();
    let (trusted, program) = comparative_gpu_times(
        backend,
        || trusted_commands(Op::Softmax, &[&input], &expected),
        || program_commands(backend, &program, &[&input], &[&actual]),
    );
    Timing {
        name: "softmax",
        trusted,
        program,
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
        program_commands(
            backend,
            program,
            &[&input],
            &outputs.iter().collect::<Vec<_>>(),
        ),
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

fn program_commands<B: Backend>(
    backend: &B,
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

fn qk_norm_rope_commands(
    values: &Tensor,
    weight: &Tensor,
    positions: &Tensor,
    normalized: &Tensor,
    output: &Tensor,
) -> CommandList {
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::RmsNorm { eps: 1.0e-6 }, &[values, weight], normalized)
        .unwrap();
    commands
        .dispatch(
            Op::Rope { theta: 1_000_000.0 },
            &[normalized, positions],
            output,
        )
        .unwrap();
    commands
}

fn position_tensor(backend: &MetalBackend, dtype: DType) -> Tensor {
    let tensor = backend.alloc(dtype, &[8]).unwrap();
    let bytes = match dtype {
        DType::F32 => (0_u16..8)
            .flat_map(|value| f32::from(value).to_le_bytes())
            .collect::<Vec<_>>(),
        DType::U32 => (0_u32..8).flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
        _ => panic!("unsupported position dtype"),
    };
    backend.write(&tensor, &bytes).unwrap();
    tensor
}

fn qk_half(backend: &MetalBackend, tensor: &Tensor, start: u32) -> Tensor {
    backend
        .view(
            tensor,
            ViewOp::Slice(vec![
                Slice::new(0, 8, 1).unwrap(),
                Slice::new(0, 16, 1).unwrap(),
                Slice::new(start, 64, 1).unwrap(),
            ]),
        )
        .unwrap()
}

fn qk_weight(backend: &MetalBackend, tensor: &Tensor, start: u32) -> Tensor {
    let weight = backend
        .view(
            tensor,
            ViewOp::Slice(vec![Slice::new(start, 64, 1).unwrap()]),
        )
        .unwrap();
    backend
        .view(&weight, ViewOp::Broadcast(vec![8, 16, 64]))
        .unwrap()
}

fn run<B: Backend>(backend: &B, commands: CommandList) {
    backend.submit(commands).unwrap().wait().unwrap();
}
