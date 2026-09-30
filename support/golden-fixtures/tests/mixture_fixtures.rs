//! Validation of generated mixture-model transformer fixtures.

use std::{env, error::Error, path::PathBuf};

use golden_fixtures::FixtureDirectory;

const OLMOE_GENERATE: &str = "uv run --project support/golden support/golden/olmoe.py";
const QWEN_GENERATE: &str = "uv run --project support/golden support/golden/qwen3_coder.py";
const QWEN_QUANTIZED_GENERATE: &str =
    "uv run --project support/golden support/golden/qwen3_coder.py --same-bytes";

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn mixture_fixtures_are_self_consistent() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let olmoe_path = required_fixture(
        &root,
        "golden/OLMoE-1B-7B-0924",
        OLMOE_GENERATE,
        "OLMoE-1B-7B-0924",
    )?;
    let qwen_path = required_fixture(
        &root,
        "golden/Qwen3-Coder-30B-A3B-Instruct",
        QWEN_GENERATE,
        "Qwen3-Coder-30B-A3B-Instruct",
    )?;
    let qwen_quantized_path = required_fixture(
        &root,
        "golden/Qwen3-Coder-30B-A3B-Instruct-4bit",
        QWEN_QUANTIZED_GENERATE,
        "Qwen3-Coder-30B-A3B-Instruct-4bit",
    )?;

    let olmoe = FixtureDirectory::open(olmoe_path)?;
    assert_eq!(olmoe.num_hidden_layers(), None);
    validate(&olmoe, 3, 16, 2_048, 64);

    let qwen = FixtureDirectory::open(qwen_path)?;
    assert_eq!(qwen.num_hidden_layers(), Some(4));
    validate(&qwen, 2, 4, 2_048, 128);
    assert!(qwen.require_complete_model_outputs().is_err());

    let qwen_quantized = FixtureDirectory::open(qwen_quantized_path)?;
    assert_eq!(qwen_quantized.num_hidden_layers(), None);
    validate(&qwen_quantized, 2, 48, 2_048, 128);
    qwen_quantized.require_complete_model_outputs()?;
    Ok(())
}

fn required_fixture(
    root: &std::path::Path,
    relative: &str,
    script: &str,
    model: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let path = root.join(relative);
    if !path.join("manifest.json").is_file() {
        return Err(format!(
            "{model} golden fixtures are missing; generate them with {script} --model \"$FORJA_MODELS/{model}\" --out \"$FORJA_MODELS/{relative}\""
        )
        .into());
    }
    Ok(path)
}

fn validate(
    fixtures: &FixtureDirectory,
    prompts: usize,
    layers: usize,
    hidden_size: usize,
    experts: usize,
) {
    assert_eq!(fixtures.prompts().len(), prompts);
    for prompt in fixtures.prompts() {
        assert_eq!(prompt.hidden_states().len(), layers + 1);
        for hidden_state in prompt.hidden_states() {
            assert_eq!(
                hidden_state.shape(),
                [prompt.prompt_ids().len(), hidden_size]
            );
        }
        assert_eq!(
            prompt.router_logits().unwrap().shape(),
            [layers, prompt.prompt_ids().len(), experts]
        );
    }
}
