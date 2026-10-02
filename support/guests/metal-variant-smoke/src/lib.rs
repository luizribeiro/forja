//! Component importing the optional Metal variant interface.

#![allow(clippy::same_length_and_capacity, clippy::unused_async_trait_impl)]

wit_bindgen::generate!({
    inline: "package smoke:metal; world smoke {
        export run: async func() -> result<_, string>;
        export run-bare: async func() -> result<_, string>;
    }",
});

use forja_sdk::{
    Param, Tensor, capture, eval,
    target::metal::{Variant, VariantRule},
};

struct Component;

impl Guest for Component {
    async fn run() -> Result<(), String> {
        exercise_sdk().map_err(|error| error.to_string())
    }

    async fn run_bare() -> Result<(), String> {
        let left = Tensor::from_slice(&[2.0_f32], &[1, 1]).map_err(|error| error.to_string())?;
        let right = Tensor::from_slice(&[3.0_f32], &[1, 1]).map_err(|error| error.to_string())?;
        left.matmul(&right)
            .and_then(|_| eval())
            .map_err(|error| error.to_string())
    }
}

fn exercise_sdk() -> forja_sdk::Result<()> {
    if Variant::new("Matmul.Gemv").is_ok() {
        return Err(forja_sdk::Error::loading(
            "invalid variant name was accepted",
        ));
    }
    let gemv = Variant::new("matmul.gemv")?;
    let parameter = Param::new(1..=33)?;
    let low = Variant::new("sdpa.decomposed")?;
    let high = Variant::new("sdpa.vector-single-pass")?;
    VariantRule::new(&parameter, vec![(1..=7, low), (8..=33, high)])?;
    let left = Tensor::from_slice(&[2.0_f32], &[1, 1])?;
    let right = Tensor::from_slice(&[3.0_f32], &[1, 1])?;
    let output = left.matmul_with(&right, &gemv)?;
    eval()?;
    if output.to_vec()? != [6.0] {
        return Err(forja_sdk::Error::loading(
            "pinned SDK matmul produced the wrong value",
        ));
    }
    let replay_parameter = Param::new(32..=33)?;
    let first = Variant::new("matmul.steel-64x64x16-2x2")?;
    let second = Variant::new("matmul.steel-32x64x16-1x2")?;
    let rule = VariantRule::new(&replay_parameter, vec![(32..=32, first), (33..=33, second)])?;
    let left = Tensor::<f32>::zeros(&[7, 33])?;
    let right = Tensor::<f32>::zeros(&[33, 7])?;
    let graph = capture(&[&replay_parameter], || {
        let inner = replay_parameter.at(32);
        left.narrow(1, 0, inner.clone())?
            .matmul_with(&right.narrow(0, 0, inner)?, &rule)
    })?;
    graph.replay(&[32])?;
    graph.replay(&[33])?;
    Ok(())
}

export!(Component);
