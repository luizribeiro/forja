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

    async fn capture_replay(value: u32) -> Result<Vec<u32>, String> {
        capture_replay(value).map_err(|error| error.to_string())
    }

    async fn capture_refusals() -> Result<Vec<String>, String> {
        capture_refusals().map_err(|error| error.to_string())
    }
}

fn capture_replay(value: u32) -> forja_sdk::Result<Vec<u32>> {
    let values = (10_u32..17).collect::<Vec<_>>();
    let source = Tensor::constant(&values, &[7])?;
    let replay_output = Tensor::zeros(&[7])?;
    let lazy_output = Tensor::zeros(&[7])?;
    let position = forja_sdk::Param::new(0..=6)?;
    let graph = forja_sdk::capture(&[&position], || {
        let end = (position.at(6) + 1)?;
        let input = source.narrow(0, 0, end.clone())?;
        let mut output = replay_output.narrow(0, 0, end)?;
        input.copy_into(&mut output)
    })?;
    graph.replay(&[value])?;
    let end = value
        .checked_add(1)
        .ok_or_else(|| forja_sdk::Error::loading("lazy prefix length overflowed"))?;
    let input = source.narrow(0, 0, end)?;
    let mut output = lazy_output.narrow(0, 0, end)?;
    input.copy_into(&mut output)?;
    let lazy = lazy_output.to_vec()?;
    let replay = replay_output.to_vec()?;
    if replay != lazy {
        return Err(forja_sdk::Error::loading(
            "captured prefix differs from lazy execution",
        ));
    }
    Ok(replay)
}

fn capture_refusals() -> forja_sdk::Result<Vec<String>> {
    let source = Tensor::constant(&(0_u32..7).collect::<Vec<_>>(), &[7])?;
    let position = forja_sdk::Param::new(0..=6)?;
    let mut errors = Vec::new();
    errors.push(
        forja_sdk::capture(&[&position], || Tensor::from_slice(&[1_u32], &[1]))
            .err()
            .ok_or_else(|| forja_sdk::Error::loading("from_slice was accepted"))?
            .to_string(),
    );
    errors.push(
        forja_sdk::capture(&[&position], || {
            source.write(&(0_u32..7).collect::<Vec<_>>())?;
            Ok(())
        })
        .err()
        .ok_or_else(|| forja_sdk::Error::loading("write was accepted"))?
        .to_string(),
    );
    errors.push(
        forja_sdk::capture(&[&position], || source.to_vec())
            .err()
            .ok_or_else(|| forja_sdk::Error::loading("to_vec was accepted"))?
            .to_string(),
    );
    errors.push(
        forja_sdk::capture(&[&position], || {
            forja_sdk::eval()?;
            Ok(())
        })
        .err()
        .ok_or_else(|| forja_sdk::Error::loading("eval was accepted"))?
        .to_string(),
    );
    errors.push(
        forja_sdk::capture(&[&position], || {
            source
                .narrow(0, 0, (position.at(2) + 1)?)?
                .broadcast_as(&[7])
        })
        .err()
        .ok_or_else(|| forja_sdk::Error::loading("symbolic broadcast was accepted"))?
        .to_string(),
    );
    Ok(errors)
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
