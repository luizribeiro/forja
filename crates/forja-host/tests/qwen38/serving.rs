use super::{Error, fixture, verification};
use forja_host::{BatchPhase, EngineDecode, EngineRunner, SamplingParams};
use forja_sdk::{DecodeInput, Engine, Tensor, Weights};

pub(super) async fn runner<B: forja_core::Backend + Send + Sync + 'static>(
    backend: B,
    replay: bool,
) -> Result<EngineRunner<B>, Box<dyn Error>> {
    runner_with_context(backend, replay, 64).await
}

pub(super) async fn runner_with_context<B: forja_core::Backend + Send + Sync + 'static>(
    backend: B,
    replay: bool,
    context: u32,
) -> Result<EngineRunner<B>, Box<dyn Error>> {
    runner_with_storage(backend, replay, context, forja_host::KvCacheFormat::F16).await
}

async fn runner_with_storage<B: forja_core::Backend + Send + Sync + 'static>(
    backend: B,
    replay: bool,
    context: u32,
    storage: forja_host::KvCacheFormat,
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
                max_context: Some(context),
                replay,
                paged_cache: true,
                kv_cache: storage,
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

async fn constrained_selection_case(replay: bool) -> Result<(), Box<dyn Error>> {
    let mut runner = runner(forja_cpu::CpuBackend::new(), replay).await?;
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
            .finish_decode(
                &logits,
                3,
                SamplingParams::default(),
                forja_host::PackedMask::new(&[0, 2])
            )
            .await?
            .is_err()
    );
    assert!(runner.checkpoint_state_lane(0).is_err());
    let selected = runner
        .finish_decode(
            &logits,
            3,
            SamplingParams::default(),
            forja_host::PackedMask::new(&[0, 1]),
        )
        .await??;
    assert_eq!(runner.read(&selected.token).await?, 32_u32.to_le_bytes());
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
        .decode_token(&Tensor::from_slice(&[32], &[1])?, 4, false)?
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

#[tokio::test]
async fn constrained_selection_and_retained_rewind_preserve_hybrid_bits()
-> Result<(), Box<dyn Error>> {
    for replay in [false, true] {
        constrained_selection_case(replay).await?;
    }
    Ok(())
}

#[path = "serving/catalog.rs"]
mod catalog;

#[path = "serving/prefix.rs"]
pub(super) mod prefix;

#[tokio::test]
async fn paged_cache_precisions_match_eager_at_page_boundaries() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    for replay in [false, true] {
        let mut reduced_bytes: Option<u64> = None;
        for (host_storage, guest_storage) in [
            (
                forja_host::KvCacheFormat::F16,
                forja_sdk::KvCacheFormat::F16,
            ),
            (
                forja_host::KvCacheFormat::F32,
                forja_sdk::KvCacheFormat::F32,
            ),
        ] {
            let mut runner =
                runner_with_storage(forja_cpu::CpuBackend::new(), replay, 128, host_storage)
                    .await?;
            let info = runner
                .describe()
                .await?
                .serving
                .ok_or("missing serving geometry")?;
            let pages =
                u64::from(128_u32.div_ceil(info.page_tokens)) + u64::from(info.active_lanes);
            let generations = pages.checked_mul(4).ok_or("generation byte overflow")?;
            let bytes = runner
                .paged_memory_usage()?
                .ok_or("missing pool")?
                .slab_bytes
                .checked_sub(generations)
                .ok_or("slab excludes generations")?;
            assert!(bytes > 0);
            if let Some(reduced) = reduced_bytes {
                assert_eq!(bytes, reduced.checked_mul(2).unwrap());
            } else {
                reduced_bytes = Some(bytes);
            }
            let mut load = verification::tiny_load();
            load.max_context = Some(128);
            load.kv_cache = guest_storage;
            let mut eager = qwen38::Qwen38::with_config(&weights, &load, qwen38::Config::tiny())?;
            for position in 0..=64 {
                let tokens = vec![position % 33];
                let phase = if position == 0 {
                    BatchPhase::Prompt
                } else {
                    BatchPhase::Decode
                };
                let expected = eager
                    .decode(DecodeInput {
                        phase: if position == 0 {
                            forja_sdk::BatchPhase::Prompt
                        } else {
                            forja_sdk::BatchPhase::Decode
                        },
                        tokens: Some(Tensor::from_slice(&tokens, &[1])?),
                        start_pos: position,
                        sampling: forja_sdk::SamplingParams::default(),
                    })?
                    .logits
                    .to_vec()?;
                let actual = runner
                    .decode(EngineDecode {
                        phase,
                        tokens: Some(tokens),
                        start_pos: position,
                        sampling: SamplingParams::default(),
                    })
                    .await??;
                if [0, 63, 64].contains(&position) {
                    let bytes = runner.read(&actual.logits).await?;
                    let (words, remainder) = bytes.as_chunks::<4>();
                    assert_eq!(remainder, &[] as &[u8]);
                    let values = words
                        .iter()
                        .copied()
                        .map(f32::from_le_bytes)
                        .collect::<Vec<_>>();
                    let error = golden_fixtures::normwise_relative_error(&expected, &values)?;
                    assert!(error <= 1e-2, "position={position} error={error}");
                }
            }
        }
    }
    Ok(())
}

#[path = "serving/state_picks.rs"]
mod state_picks;
