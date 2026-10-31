use super::{Error, fixture, verification};
use forja_host::{BatchPhase, EngineDecode, EngineRunner, SamplingParams};
use forja_sdk::{DecodeInput, Engine, Tensor, Weights};

pub(super) async fn runner<B: forja_core::Backend + Send + Sync + 'static>(
    backend: B,
    replay: bool,
) -> Result<EngineRunner<B>, Box<dyn Error>> {
    let mut runner = EngineRunner::new(
        test_guests::qwen38_tiny(),
        backend,
        forja_host::Limits::new(1 << 30, 8, 1 << 30, 20_000, 1 << 27),
        fixture("model.safetensors"),
    )
    .await?;
    runner
        .load_with_selections(
            None,
            &forja_host::EngineLoadConfig {
                max_context: Some(64),
                replay,
                paged_cache: true,
                kv_cache: forja_host::KvCacheFormat::F16,
                ..Default::default()
            },
        )
        .await??;
    assert!(!runner.supports_prefix_reuse());
    Ok(runner)
}

#[tokio::test]
async fn serving_feedback_matches_explicit_eager_tokens() -> Result<(), Box<dyn Error>> {
    let mut runner = runner(forja_cpu::CpuBackend::new(), false).await?;
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut native =
        qwen38::Qwen38::with_config(&weights, &verification::tiny_load(), qwen38::Config::tiny())?;
    for _ in 0..2 {
        let mut previous = Vec::new();
        for step in 0..5 {
            let (position, phase, tokens) = if step == 0 {
                (0, BatchPhase::Prompt, vec![1, 7, 32])
            } else {
                (step + 2, BatchPhase::Decode, previous)
            };
            let expected = native.decode(DecodeInput {
                phase: if step == 0 {
                    forja_sdk::BatchPhase::Prompt
                } else {
                    forja_sdk::BatchPhase::Decode
                },
                tokens: Some(Tensor::from_slice(
                    &tokens,
                    &[u32::try_from(tokens.len())?],
                )?),
                start_pos: position,
                sampling: forja_sdk::SamplingParams::default(),
            })?;
            let actual = runner
                .decode(EngineDecode {
                    phase,
                    tokens: (step == 0).then_some(tokens),
                    start_pos: position,
                    sampling: SamplingParams::default(),
                })
                .await??;
            previous = expected.token.to_vec()?;
            assert_eq!(
                runner.read(&actual.token).await?,
                previous
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                runner.read(&actual.logits).await?,
                expected
                    .logits
                    .to_vec()?
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn constrained_selection_and_retained_rewind_preserve_hybrid_bits()
-> Result<(), Box<dyn Error>> {
    let mut runner = runner(forja_cpu::CpuBackend::new(), false).await?;
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut native =
        qwen38::Model::load(&weights, &verification::tiny_load(), qwen38::Config::tiny())?;
    native.forward(&Tensor::from_slice(&[1, 7, 32], &[3])?, 0, false)?;
    runner
        .decode(EngineDecode {
            phase: BatchPhase::Prompt,
            tokens: Some(vec![1, 7, 32]),
            start_pos: 0,
            sampling: SamplingParams::default(),
        })
        .await??;
    let expected = native.decode_token(&Tensor::from_slice(&[3], &[1])?, 3, false)?;
    let logits = runner
        .begin_decode(forja_host::EngineBeginDecode {
            phase: BatchPhase::Decode,
            tokens: Some(vec![3]),
            start_pos: 3,
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
    assert!(
        runner
            .finish_decode(&logits, 3, SamplingParams::default(), &[33])
            .await?
            .is_err()
    );
    assert!(runner.checkpoint_state_lane(0).is_err());
    let selected = runner
        .finish_decode(&logits, 3, SamplingParams::default(), &[0])
        .await??;
    assert_eq!(runner.read(&selected.token).await?, 0_u32.to_le_bytes());
    let position = runner
        .checkpoint_state_lane(0)?
        .ok_or("missing state checkpoint")?
        .position();
    assert_eq!(
        (
            position.tokens,
            position.sampler,
            position.grammar,
            position.parser
        ),
        (4, 3, 4, 4)
    );
    let expected = native
        .decode_token(&Tensor::from_slice(&[0], &[1])?, 4, false)?
        .logits
        .to_vec()?
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for repeat in 0..2 {
        if repeat == 1 {
            runner.rewind_decode(4).await??;
        }
        let decoded = runner
            .decode(EngineDecode {
                phase: BatchPhase::Decode,
                tokens: None,
                start_pos: 4,
                sampling: SamplingParams::default(),
            })
            .await??;
        assert_eq!(runner.read(&decoded.logits).await?, expected);
        let token = runner.read(&decoded.token).await?;
        if repeat == 0 {
            selected = token;
        } else {
            assert_eq!(selected, token);
        }
    }
    Ok(())
}

#[path = "serving/batches.rs"]
mod batches;
