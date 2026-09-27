//! Native Qwen3 engine checks against independent transformer fixtures.

use std::{env, error::Error, path::PathBuf};

use forja_sdk::{Engine, Tensor, Weights};
use golden_fixtures::{BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, normwise_relative_error};
use qwen3::Qwen3;

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn first_layer_matches_transformers() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let fixtures = FixtureDirectory::open(root.join("golden/qwen3-0.6b"))?;
    let fixture = fixtures
        .prompt("single-token")
        .ok_or("single-token fixture is missing")?;
    let tokens = fixture
        .prompt_ids()
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = Weights::open(root.join("Qwen3-0.6B/model.safetensors"))?;
    let mut engine = Qwen3::load(&weights)?;
    let tokens = Tensor::from_slice(&tokens, &[u32::try_from(tokens.len())?])?;
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
