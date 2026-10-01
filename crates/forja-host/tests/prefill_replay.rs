//! Chunked prefill differential checks across the bundled engines.

use std::{env, error::Error, path::Path, path::PathBuf, time::Duration};

use forja_core::Backend;
use forja_host::{EngineDecode, EngineRunner, EngineStep, Limits, SamplingParams};
use forja_testing::normwise_relative_error;
use golden_fixtures::{BF16_HIDDEN_STATE_TOLERANCE, decode_f32_le};

const LENGTHS: [u32; 9] = [1, 7, 16, 33, 511, 512, 513, 1_500, 4_000];
const LIMITS: Limits = Limits::new(
    48 * 1024 * 1024 * 1024,
    4,
    1_000_000_000,
    20_000,
    1024 * 1024 * 1024,
)
.with_command_limits(4_096, u64::MAX)
.with_guest_call_timeout(Duration::from_secs(300))
.with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn cpu_small_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::qwen3_bf16(),
        test_guests::qwen3_bf16_no_replay(),
        &model_root()?.join("Qwen3-0.6B/model.safetensors"),
        None,
        qwen3::VOCAB,
        &[1, 7],
        forja_cpu::CpuBackend::new(),
        forja_cpu::CpuBackend::new(),
        forja_cpu::CpuBackend::new(),
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_qwen3_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::qwen3_bf16(),
        test_guests::qwen3_bf16_no_replay(),
        &model_root()?.join("Qwen3-0.6B/model.safetensors"),
        None,
        qwen3::VOCAB,
        &LENGTHS,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_olmoe_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::olmoe(),
        test_guests::olmoe_no_replay(),
        &model_root()?.join("OLMoE-1B-7B-0924/model.safetensors.index.json"),
        None,
        olmoe::VOCAB,
        &LENGTHS,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_qwen3_coder_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::qwen3_coder(),
        test_guests::qwen3_coder_no_replay(),
        &model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json"),
        Some(4),
        qwen3_coder::VOCAB,
        &LENGTHS,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn compare_prefill<B>(
    replay_component: &Path,
    lazy_component: &Path,
    weights: &Path,
    layers: Option<u32>,
    vocab: u32,
    lengths: &[u32],
    replay_backend: B,
    lazy_backend: B,
    reference_backend: B,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let mut replay = EngineRunner::new(replay_component, replay_backend, LIMITS, weights).await?;
    let mut lazy = EngineRunner::new(lazy_component, lazy_backend, LIMITS, weights).await?;
    let mut reference =
        EngineRunner::new(lazy_component, reference_backend, LIMITS, weights).await?;
    replay.load_with_config(layers).await??;
    lazy.load_with_config(layers).await??;
    reference.load_with_config(layers).await??;
    for &length in lengths {
        let tokens = (0..length)
            .map(|index| (index.wrapping_mul(7_919).wrapping_add(17)) % vocab)
            .collect::<Vec<_>>();
        let replay_prefill = replay
            .decode(EngineDecode {
                tokens: Some(tokens.clone()),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })
            .await??;
        let lazy_prefill = lazy
            .decode(EngineDecode {
                tokens: Some(tokens.clone()),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })
            .await??;
        let (replay_prefill_logits, replay_token) =
            read_decode_output(&mut replay, &replay_prefill).await?;
        let (lazy_prefill_logits, lazy_token) =
            read_decode_output(&mut lazy, &lazy_prefill).await?;
        assert_eq!(
            replay_prefill_logits, lazy_prefill_logits,
            "prefill logits differed after {length} tokens"
        );
        assert_eq!(
            replay_token, lazy_token,
            "prefill token differed after {length} tokens"
        );

        reference
            .step(EngineStep {
                tokens: tokens.clone(),
                start_pos: 0,
                taps: true,
            })
            .await??;
        let continuation_token = u32::from_le_bytes(replay_token.as_slice().try_into()?);
        let reference_next = reference
            .step(EngineStep {
                tokens: vec![continuation_token],
                start_pos: length,
                taps: true,
            })
            .await??;
        let reference_logits = reference.read(&reference_next.logits).await?;

        let replay_next = replay
            .decode(EngineDecode {
                tokens: None,
                start_pos: length,
                sampling: SamplingParams::default(),
            })
            .await??;
        let lazy_next = lazy
            .decode(EngineDecode {
                tokens: None,
                start_pos: length,
                sampling: SamplingParams::default(),
            })
            .await??;
        let (replay_logits, replay_token) = read_decode_output(&mut replay, &replay_next).await?;
        let (lazy_logits, lazy_token) = read_decode_output(&mut lazy, &lazy_next).await?;
        assert_eq!(
            replay_logits, lazy_logits,
            "continuation logits differed after {length} tokens"
        );
        assert_eq!(
            replay_token, lazy_token,
            "continuation token differed after {length} tokens"
        );
        let error = normwise_relative_error(
            &decode_f32_le(&reference_logits)?,
            &decode_f32_le(&replay_logits)?,
        );
        assert!(
            error <= BF16_HIDDEN_STATE_TOLERANCE,
            "continuation after {length} tokens had relative error {error}"
        );
    }
    Ok(())
}

async fn read_decode_output<B>(
    runner: &mut EngineRunner<B>,
    output: &forja_host::EngineDecodeOutput,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    Ok((
        runner.read(&output.logits).await?,
        runner.read(&output.token).await?,
    ))
}

fn model_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env::var("FORJA_MODELS")?))
}
