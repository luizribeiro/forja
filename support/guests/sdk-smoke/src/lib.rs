//! Component running a Qwen-shaped attention block through the guest SDK.

#![allow(
    clippy::too_many_arguments,
    reason = "wit-bindgen lowers each exported list into a pointer-length pair"
)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "sdk-smoke",
});

use forja_sdk::{Tensor, nn};

struct Component;

impl Guest for Component {
    async fn run(
        input: Vec<f32>,
        norm_weight: Vec<f32>,
        q_weight: Vec<f32>,
        k_weight: Vec<f32>,
        v_weight: Vec<f32>,
        output_weight: Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        attention(
            &input,
            &norm_weight,
            &q_weight,
            &k_weight,
            &v_weight,
            &output_weight,
        )
        .map_err(|error| error.to_string())
    }
}

fn attention(
    input: &[f32],
    norm_weight: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    output_weight: &[f32],
) -> forja_sdk::Result<Vec<f32>> {
    let input = Tensor::from_slice(input, &[7, 1024])?;
    let norm = nn::RmsNorm::new(Tensor::from_slice(norm_weight, &[1024])?, 1.0e-6);
    let hidden = norm.forward(&input)?;
    let positions = Tensor::from_slice(&[0_u32, 1, 2, 3, 4, 5, 6], &[7])?;

    let query = nn::Linear::new(Tensor::from_slice(q_weight, &[2048, 1024])?)
        .forward(&hidden)?
        .reshape(&[7, 16, 128])?
        .rope(&positions, 1_000_000.0)?
        .permute(&[1, 0, 2])?;
    let key = nn::Linear::new(Tensor::from_slice(k_weight, &[1024, 1024])?)
        .forward(&hidden)?
        .reshape(&[7, 8, 128])?
        .rope(&positions, 1_000_000.0)?
        .permute(&[1, 0, 2])?;
    let value = nn::Linear::new(Tensor::from_slice(v_weight, &[1024, 1024])?)
        .forward(&hidden)?
        .reshape(&[7, 8, 128])?
        .permute(&[1, 0, 2])?;
    let attended = nn::ops::sdpa(&query, &key, &value, 128.0_f32.sqrt().recip(), true, 0)?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[7, 2048])?;
    let projected =
        nn::Linear::new(Tensor::from_slice(output_weight, &[1024, 2048])?).forward(&attended)?;
    (&input + &projected)?.to_vec()
}

export!(Component);
