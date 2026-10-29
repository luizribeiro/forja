use super::{Error, execution::Execution};
use forja_sdk::EngineLoadConfig;
use golden_fixtures::{
    FixtureDirectory, PromptFixture, mean_logit_kl_divergence, normwise_relative_error,
};
use qwen38::{Config, Model};

pub(super) fn tiny_load() -> EngineLoadConfig {
    EngineLoadConfig {
        max_context: Some(64),
        kv_cache: forja_sdk::KvCacheFormat::F16,
        replay: false,
        ..EngineLoadConfig::default()
    }
}
pub(super) async fn verify(
    model: &mut impl Execution,
    fixture: &PromptFixture,
) -> Result<(), Box<dyn Error>> {
    let ids = fixture
        .prompt_ids()
        .iter()
        .map(|&id| u32::try_from(id))
        .collect::<Result<Vec<_>, _>>()?;
    let prompt = u32::try_from(ids.len())?;
    let mut output = model.forward(&ids, 0, true).await?;
    assert_eq!(output.taps.len(), fixture.hidden_states().len() - 1);
    for (layer, tap) in output.taps.iter().enumerate() {
        let error = normwise_relative_error(
            fixture
                .hidden_state(layer + 1)
                .ok_or("missing layer")?
                .values(),
            tap,
        )?;
        assert!(
            error <= 5.0e-2,
            "layer {} relative error {error}",
            layer + 1
        );
    }
    let mut logits = Vec::new();
    let mut matches = 0;
    for (index, &expected) in fixture.greedy_tokens().iter().enumerate() {
        let row = &output.logits;
        matches += usize::from(top1(row) == Some(usize::try_from(expected)?));
        logits.extend_from_slice(row);
        if index + 1 < fixture.greedy_tokens().len() {
            output = model
                .forward(
                    &[u32::try_from(expected)?],
                    prompt + u32::try_from(index)?,
                    false,
                )
                .await?;
        }
    }
    let reference = fixture.greedy_step_logits().values();
    let kl = mean_logit_kl_divergence(reference, &logits, fixture.prompt_logits().values().len())?;
    super::agree(reference, &logits)?;
    eprintln!("{}: mean KL {kl}, top-1 {matches}", fixture.name());
    assert!(kl <= 1.0e-2, "mean KL {kl}");
    assert!(
        matches * 100 >= fixture.greedy_tokens().len() * 95,
        "top-1 {matches}"
    );
    Ok(())
}
fn top1(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })
        .map(|(index, _)| index)
}
#[tokio::test]
async fn full_tiny_model_matches_teacher_forced_transformers() -> Result<(), Box<dyn Error>> {
    let weights = forja_sdk::Weights::open(super::fixture("model.safetensors"))?;
    let fixtures = FixtureDirectory::open(super::fixture(""))?;
    let mut model = Model::load(&weights, &tiny_load(), Config::tiny())?;
    let fixture = fixtures.prompt("tiny").ok_or("missing fixture")?;
    verify(&mut model, fixture).await
}
