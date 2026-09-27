//! Component running fused residual addition and RMS normalization.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "program-smoke",
});

use forja_sdk::{
    Tensor,
    program::{Program, ReduceOp},
};

struct Component;

impl Guest for Component {
    async fn run() -> Result<Vec<f32>, String> {
        fused().map_err(|error| error.to_string())
    }
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
