use super::{Error, Weights, execution::Execution, fixture, verification};
use forja_host::BatchPhase;
fn model(weights: &Weights<'_>) -> forja_sdk::Result<qwen38::Model> {
    let mut load = verification::tiny_load();
    load.max_context = Some(512);
    qwen38::Model::load(weights, &load, qwen38::Config::tiny())
}
#[tokio::test]
async fn canonical_prompt_groups_and_reset_preserve_bits() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut reused = model(&weights)?;
    for count in [1, 7, 33, 63, 64, 65, 129] {
        let tokens = (0..count).map(|id| id % 33).collect::<Vec<_>>();
        let mut whole = model(&weights)?;
        let expected = Execution::forward(&mut whole, &tokens, 0, false, BatchPhase::Prompt)
            .await?
            .logits;
        for grouping in [64, 512, 2048] {
            let mut chunks = model(&weights)?;
            let mut actual = Vec::new();
            for start in (0..tokens.len()).step_by(grouping) {
                let end = (start + grouping).min(tokens.len());
                actual = Execution::forward(
                    &mut chunks,
                    &tokens[start..end],
                    u32::try_from(start)?,
                    false,
                    BatchPhase::Prompt,
                )
                .await?
                .logits;
            }
            assert_eq!(expected, actual, "count={count} grouping={grouping}");
        }
        assert_eq!(
            expected,
            Execution::forward(&mut reused, &tokens, 0, false, BatchPhase::Prompt)
                .await?
                .logits
        );
    }
    Ok(())
}
