//! Native Qwen3 engine checks against independent transformer fixtures.

use std::{env, error::Error, path::PathBuf};

use forja_sdk::{Engine, Tensor, Weights};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, PromptFixture, normwise_relative_error,
};
use qwen3::Qwen3;

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn first_layer_matches_transformers() -> Result<(), Box<dyn Error>> {
    let (fixture, mut engine, tokens) = single_token_case()?;
    let actual = engine.first_layer(&tokens)?.to_vec()?;
    let expected = fixture
        .hidden_state(1)
        .ok_or("layer 1 fixture is missing")?
        .values();
    let error = normwise_relative_error(expected, &actual)?;
    assert!(
        error <= BF16_HIDDEN_STATE_TOLERANCE,
        "layer 1 relative error {error}"
    );
    Ok(())
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn all_layers_match_transformers() -> Result<(), Box<dyn Error>> {
    let (fixture, mut engine, tokens) = single_token_case()?;
    let output = engine.step(forja_sdk::StepInput {
        tokens,
        start_pos: 0,
        taps: true,
    })?;
    assert_eq!(output.taps.len(), 28);
    for (index, tap) in output.taps.iter().enumerate() {
        let actual = tap.to_vec()?;
        let expected = fixture
            .hidden_state(index + 1)
            .ok_or("layer fixture is missing")?
            .values();
        let error = normwise_relative_error(expected, &actual)?;
        assert!(
            error <= BF16_HIDDEN_STATE_TOLERANCE,
            "layer {} relative error {error}",
            index + 1
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn chunked_prefill_matches_single_pass_and_transformers() -> Result<(), Box<dyn Error>> {
    let (fixture, mut engine, tokens) = prompt_case("short-english", 7)?;
    let single = engine.step(forja_sdk::StepInput {
        tokens,
        start_pos: 0,
        taps: true,
    })?;
    let single_logits = single.logits.to_vec()?;
    let single_taps = single
        .taps
        .iter()
        .map(Tensor::to_vec)
        .collect::<Result<Vec<_>, _>>()?;
    drop(engine);

    let (_, mut engine, first) = prompt_case("short-english", 3)?;
    engine.step(forja_sdk::StepInput {
        tokens: first,
        start_pos: 0,
        taps: false,
    })?;
    let suffix = fixture.prompt_ids()[3..7]
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let chunked = engine.step(forja_sdk::StepInput {
        tokens: Tensor::from_slice(&suffix, &[4])?,
        start_pos: 3,
        taps: true,
    })?;

    assert_close("chunked logits", &single_logits, &chunked.logits.to_vec()?)?;
    for (index, (single, chunked)) in single_taps.iter().zip(&chunked.taps).enumerate() {
        let chunked = chunked.to_vec()?;
        assert_close(
            &format!("layer {} single pass", index + 1),
            &single[3 * 1_024..7 * 1_024],
            &chunked,
        )?;
        let golden = fixture
            .hidden_state(index + 1)
            .ok_or("layer fixture is missing")?;
        assert_close(
            &format!("layer {} transformers", index + 1),
            &golden.values()[3 * 1_024..7 * 1_024],
            &chunked,
        )?;
    }
    Ok(())
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    let error = normwise_relative_error(expected, actual)?;
    assert!(
        error <= BF16_HIDDEN_STATE_TOLERANCE,
        "{label} relative error {error}"
    );
    Ok(())
}

fn single_token_case() -> Result<(PromptFixture, Qwen3, Tensor<u32>), Box<dyn Error>> {
    prompt_case("single-token", 1)
}

fn prompt_case(
    name: &str,
    token_count: usize,
) -> Result<(PromptFixture, Qwen3, Tensor<u32>), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let fixtures = FixtureDirectory::open(root.join("golden/qwen3-0.6b"))?;
    let fixture = fixtures
        .prompt(name)
        .ok_or("prompt fixture is missing")?
        .clone();
    let tokens = fixture
        .prompt_ids()
        .get(..token_count)
        .ok_or("prompt fixture has too few tokens")?
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = Weights::open(root.join("Qwen3-0.6B/model.safetensors"))?;
    let engine = Qwen3::load(&weights)?;
    let length = u32::try_from(tokens.len())?;
    Ok((fixture, engine, Tensor::from_slice(&tokens, &[length])?))
}
