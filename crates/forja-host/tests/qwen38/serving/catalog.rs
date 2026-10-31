use super::{Error, fixture, runner_with_context, verification};
use forja_host::{BatchPhase, EngineDecode, SamplingParams};
use forja_sdk::{DecodeInput, Engine, Tensor, Weights};

#[tokio::test]
async fn prompt_catalog_replays_base_shapes_and_replaces_one_tail() -> Result<(), Box<dyn Error>> {
    let mut runner = runner_with_context(forja_cpu::CpuBackend::new(), true, 2048).await?;
    assert_eq!(runner.metrics().live_graphs, 12);
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut load = verification::tiny_load();
    load.max_context = Some(2048);
    let mut native = qwen38::Qwen38::with_config(&weights, &load, qwen38::Config::tiny())?;
    for rows in [7, 33, 7, 1, 64, 512, 2048] {
        let tokens = (0..rows).map(|position| position % 33).collect::<Vec<_>>();
        let expected = native.decode(DecodeInput {
            phase: forja_sdk::BatchPhase::Prompt,
            tokens: Some(Tensor::from_slice(&tokens, &[rows])?),
            start_pos: 0,
            sampling: forja_sdk::SamplingParams::default(),
        })?;
        let actual = runner
            .decode(EngineDecode {
                phase: BatchPhase::Prompt,
                tokens: Some(tokens),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })
            .await??;
        assert_eq!(
            runner.read(&actual.logits).await?,
            expected
                .logits
                .to_vec()?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            runner.read(&actual.token).await?,
            expected
                .token
                .to_vec()?
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(runner.metrics().live_graphs, 13);
    }
    assert_eq!(runner.metrics().prepared_graphs, 16);
    Ok(())
}
