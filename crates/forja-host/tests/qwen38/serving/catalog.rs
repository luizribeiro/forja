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

#[tokio::test]
async fn prompt_forward_reuses_capture_without_exposing_combined_feedback()
-> Result<(), Box<dyn Error>> {
    let mut runner = runner_with_context(forja_cpu::CpuBackend::new(), true, 128).await?;
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut load = verification::tiny_load();
    load.max_context = Some(128);
    let mut native = qwen38::Model::load(&weights, &load, qwen38::Config::tiny())?;
    for phase in [BatchPhase::Prompt, BatchPhase::Decode] {
        let prompt = Tensor::from_slice(&[1_u32; 64], &[64])?;
        native.forward(&prompt, 0, false)?;
        runner
            .step(forja_host::EngineStep {
                phase: BatchPhase::Prompt,
                tokens: vec![1; 64],
                start_pos: 0,
                taps: false,
            })
            .await??;
        assert!(
            runner
                .decode(EngineDecode {
                    phase: BatchPhase::Decode,
                    tokens: None,
                    start_pos: 64,
                    sampling: SamplingParams::default()
                })
                .await?
                .is_err()
        );
        assert_eq!(
            runner
                .checkpoint_state_lane(0)?
                .ok_or("missing carry")?
                .position()
                .tokens,
            64
        );
        let token = Tensor::from_slice(&[3_u32], &[1])?;
        let expected = match phase {
            BatchPhase::Prompt => native.forward(&token, 64, false)?,
            BatchPhase::Decode => native.decode_token(&token, 64, false)?,
        };
        let logits = runner
            .begin_decode(forja_host::EngineBeginDecode {
                phase,
                tokens: Some(vec![3]),
                start_pos: 64,
            })
            .await??;
        assert_eq!(
            runner.read(&logits).await?,
            expected
                .logits
                .to_vec()?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(runner.checkpoint_state_lane(0).is_err());
        let selected = runner
            .finish_decode(
                &logits,
                64,
                SamplingParams::default(),
                forja_host::PackedMask::new(&[1, 0]),
            )
            .await??;
        assert_eq!(runner.read(&selected.token).await?, 0_u32.to_le_bytes());
        assert_eq!(
            runner
                .checkpoint_state_lane(0)?
                .ok_or("missing carry")?
                .position()
                .tokens,
            65
        );
        assert_eq!(runner.metrics().live_graphs, 11);
    }
    Ok(())
}
