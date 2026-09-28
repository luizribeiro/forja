//! Component running fused residual addition and RMS normalization.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "program-smoke",
});

use forja_sdk::{
    DType, Tensor,
    program::{Kernel as SdkKernel, Program, ReduceOp},
};
use l9o::gpu::compute::{Binop, Dtype, Inst, Kernel, KernelSignature, ProgramKind, ProgramSource};

struct Component;

impl Guest for Component {
    async fn run() -> Result<Vec<f32>, String> {
        fused().map_err(|error| error.to_string())
    }

    async fn churn_program_cache() -> Result<(), String> {
        run_distinct_programs(20).map_err(|error| error.to_string())
    }

    async fn exercise_explicit_kernel() -> Result<(), String> {
        exercise_explicit_kernel().map_err(|error| error.to_string())
    }

    async fn exhaust_shared_quota() -> Result<(), String> {
        let kernels = (0_u16..4)
            .map(explicit_kernel)
            .collect::<Result<Vec<_>, _>>()?;
        let result = run_distinct_programs(5).map_err(|error| error.to_string());
        if kernels.len() != 4 {
            return Err("explicit kernel retention failed".to_owned());
        }
        result
    }
}

fn exercise_explicit_kernel() -> forja_sdk::Result<()> {
    let program = Program::map();
    program.output(0, program.input(0) * 2.0);
    let kernel = SdkKernel::new(&program, 2, &[DType::F32], &[DType::F32])?;
    let wrong_rank = Tensor::from_slice(&[1.0_f32; 7], &[7])?;
    let mismatch = wrong_rank
        .run_kernel(&kernel, &[])
        .err()
        .ok_or_else(|| forja_sdk::Error::loading("kernel accepted the wrong rank"))?;
    if mismatch.to_string() != "kernel signature expects rank 2, but tensor has rank 1" {
        return Err(forja_sdk::Error::loading(format!(
            "kernel returned an unclear signature error: {mismatch}"
        )));
    }
    for shape in [[1_u32, 7], [7, 33], [33, 1]] {
        let count = usize::try_from(shape[0] * shape[1])
            .map_err(|_| forja_sdk::Error::loading("test tensor is too large"))?;
        let input = Tensor::from_slice(&vec![3.0_f32; count], &shape)?;
        let output = input
            .run_kernel(&kernel, &[])?
            .pop()
            .ok_or_else(|| forja_sdk::Error::loading("kernel produced no output"))?;
        if output.to_vec()? != vec![6.0; count] {
            return Err(forja_sdk::Error::loading(
                "reused kernel produced wrong values",
            ));
        }
    }
    let input = Tensor::from_slice(&[4.0_f32; 7], &[1, 7])?;
    let pending = input
        .run_kernel(&kernel, &[])?
        .pop()
        .ok_or_else(|| forja_sdk::Error::loading("kernel produced no pending output"))?;
    drop(kernel);
    if pending.to_vec()? != [8.0; 7] {
        return Err(forja_sdk::Error::loading(
            "dropped kernel did not complete pending work",
        ));
    }
    Ok(())
}

fn run_distinct_programs(count: u16) -> forja_sdk::Result<()> {
    let input = Tensor::from_slice(&[2.0_f32], &[1])?;
    for value in 0..count {
        let program = Program::map();
        program.output(0, program.input(0) + program.constant(f32::from(value)));
        let output = input
            .run_program(&program, &[])?
            .pop()
            .ok_or_else(|| forja_sdk::Error::loading("program produced no output"))?;
        let values = output.to_vec()?;
        let actual = values
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("program output was empty"))?;
        if actual.to_bits() != (2.0 + f32::from(value)).to_bits() {
            return Err(forja_sdk::Error::loading(
                "cached program produced a wrong value",
            ));
        }
    }
    Ok(())
}

fn explicit_kernel(value: u16) -> Result<Kernel, String> {
    Kernel::create(
        &ProgramSource {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Const(f32::from(value)),
                Inst::Binary((Binop::Add, 0, 1)),
            ],
            outputs: vec![(0, 2)],
        },
        &KernelSignature {
            rank: 1,
            inputs: vec![Dtype::F32],
            outputs: vec![Dtype::F32],
            scalars: 0,
        },
    )
    .map_err(|error| format!("explicit kernel creation returned {error:?}"))
}

fn fused() -> forja_sdk::Result<Vec<f32>> {
    let residual = Tensor::from_slice(&values(0.03125), &[7, 1024])?;
    let update = Tensor::from_slice(&values(-0.015_625), &[7, 1024])?;
    let weight_values = (0_u16..1024)
        .map(|index| 0.5 + f32::from(index) / 2048.0)
        .collect::<Vec<_>>();
    let weight = Tensor::from_slice(&weight_values, &[1024])?.broadcast_as(&[7, 1024])?;
    let program = Program::row();
    let sum = program.input(0) + program.input(1);
    let square_sum = program.reduce(ReduceOp::Sum, sum * sum);
    let inverse_rms = (square_sum / program.extent(-1) + 1.0e-6).rsqrt();
    program.output(0, sum * inverse_rms * program.input(2));
    residual
        .run_program(&program, &[&update, &weight])?
        .remove(0)
        .to_vec()
}

fn values(scale: f32) -> Vec<f32> {
    (0_u16..7 * 1024)
        .map(|index| (f32::from(index % 257) - 128.0) * scale)
        .collect()
}

export!(Component);
