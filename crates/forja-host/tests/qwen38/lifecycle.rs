use super::{Error, Weights, execution::Execution, fixture, verification};
fn model(weights: &Weights<'_>) -> forja_sdk::Result<qwen38::Model> {
    qwen38::Model::load(weights, &verification::tiny_load(), qwen38::Config::tiny())
}
#[tokio::test]
async fn prompt_chunks_and_new_sequences_preserve_tokenwise_results() -> Result<(), Box<dyn Error>>
{
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut reused = model(&weights)?;
    for count in [1, 7, 33] {
        let tokens = (0..count).map(|id| id % 33).collect::<Vec<_>>();
        let mut whole = model(&weights)?;
        let expected = Execution::forward(&mut whole, &tokens, 0, false)
            .await?
            .logits;
        let mut chunks = model(&weights)?;
        let first = Execution::forward(&mut chunks, &tokens[..1], 0, false).await?;
        let actual = if count == 1 {
            first.logits
        } else {
            Execution::forward(&mut chunks, &tokens[1..], 1, false)
                .await?
                .logits
        };
        assert_eq!(expected, actual);
        assert_eq!(
            expected,
            Execution::forward(&mut reused, &tokens, 0, false)
                .await?
                .logits
        );
    }
    Ok(())
}
