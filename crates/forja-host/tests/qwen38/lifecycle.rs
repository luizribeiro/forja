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

#[test]
fn prompt_and_decode_refuse_ambiguous_or_split_token_spans() -> Result<(), Box<dyn Error>> {
    use forja_sdk::{DecodeInput, Engine, SamplingParams, Tensor};
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut model = model(&weights)?;
    let one = Tensor::from_slice(&[1_u32], &[1])?;
    let pair = Tensor::from_slice(&[1_u32, 2], &[2])?;
    model.forward(&one, 0, false)?;
    assert!(model.forward(&one, 1, false).is_err());
    assert!(model.decode_token(&pair, 1, false).is_err());
    let mut load = verification::tiny_load();
    load.max_context = Some(128);
    let mut engine = qwen38::Qwen38::with_config(&weights, &load, qwen38::Config::tiny())?;
    engine.decode(DecodeInput {
        phase: forja_sdk::BatchPhase::Prompt,
        tokens: Some(Tensor::from_slice(&[1_u32; 64], &[64])?),
        start_pos: 0,
        sampling: SamplingParams::default(),
    })?;
    assert!(
        engine
            .decode(DecodeInput {
                phase: forja_sdk::BatchPhase::Prompt,
                tokens: None,
                start_pos: 64,
                sampling: SamplingParams::default()
            })
            .is_err()
    );
    Ok(())
}

#[test]
fn eager_cache_precisions_agree_at_canonical_boundaries() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut load = verification::tiny_load();
    load.max_context = Some(128);
    let mut reduced = qwen38::Model::load(&weights, &load, qwen38::Config::tiny())?;
    load.kv_cache = forja_sdk::KvCacheFormat::F32;
    let mut reference = qwen38::Model::load(&weights, &load, qwen38::Config::tiny())?;
    for position in 0..=64 {
        let token = forja_sdk::Tensor::from_slice(&[position % 33], &[1])?;
        let expected = reference
            .decode_token(&token, position, false)?
            .logits
            .to_vec()?;
        let actual = reduced
            .decode_token(&token, position, false)?
            .logits
            .to_vec()?;
        if [0, 63, 64].contains(&position) {
            let error = golden_fixtures::normwise_relative_error(&expected, &actual)?;
            assert!(error <= 1e-2, "position={position} error={error}");
        }
    }
    load.kv_cache = forja_sdk::KvCacheFormat::Bf16;
    assert!(qwen38::Model::load(&weights, &load, qwen38::Config::tiny()).is_err());
    Ok(())
}
