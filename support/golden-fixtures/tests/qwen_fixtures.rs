//! Validation of the generated Qwen3 transformer fixtures.

use std::{env, error::Error, path::PathBuf};

use golden_fixtures::FixtureDirectory;

const GENERATE_COMMAND: &str = "nix develop -c sh -c 'uv run --project support/golden \
    support/golden/generate.py --model \"$FORJA_MODELS/Qwen3-0.6B\" \
    --out \"$FORJA_MODELS/golden/qwen3-0.6b\"'";

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn qwen_fixtures_are_self_consistent() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let fixture_path = root.join("golden/qwen3-0.6b");
    if !fixture_path.join("manifest.json").is_file() {
        return Err(format!(
            "Qwen3 golden fixtures are missing; generate them with: {GENERATE_COMMAND}"
        )
        .into());
    }
    let fixtures = FixtureDirectory::open(fixture_path)?;
    assert_eq!(fixtures.prompts().len(), 4);
    for prompt in fixtures.prompts() {
        assert_eq!(prompt.hidden_states().len(), 29);
        for hidden_state in prompt.hidden_states() {
            assert_eq!(hidden_state.shape(), [prompt.prompt_ids().len(), 1024]);
        }
        assert_eq!(prompt.prompt_logits().shape(), [151_936]);
        assert_eq!(prompt.greedy_tokens().len(), 32);
        assert_eq!(prompt.greedy_step_logits().shape(), [32, 151_936]);
        for (logits, &token) in prompt
            .greedy_step_logits()
            .values()
            .as_chunks::<151_936>()
            .0
            .iter()
            .zip(prompt.greedy_tokens())
        {
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .ok_or("step logits are empty")?;
            assert_eq!(i64::try_from(argmax)?, token);
        }
    }
    Ok(())
}
