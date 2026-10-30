//! Chunked prefill differential checks across the bundled engines.

use crate::common;

use std::{env, error::Error, fs, path::Path, path::PathBuf, time::Duration};

use forja_core::Backend;
use forja_host::{
    EngineDecode, EngineLoadConfig, EngineRunner, EngineStep, Limits, SamplingParams,
};
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
const REPLAY_ARENA_BUDGET: u64 = 4 * 1024 * 1024;
const REPLAY_TEST_LIMITS: Limits =
    Limits::new(REPLAY_ARENA_BUDGET, 2, 32_768, 128, REPLAY_ARENA_BUDGET)
        .with_command_limits(16, 1_000_000)
        .with_guest_call_timeout(Duration::from_secs(60))
        .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(600));

#[tokio::test]
async fn chunked_prefill_replay_bounds_memory_and_evicts_tails() -> Result<(), Box<dyn Error>> {
    let weights = empty_weight_file("prefill-replay")?;
    let mut replay = EngineRunner::new(
        test_guests::prefill_replay_smoke(),
        forja_cpu::CpuBackend::new(),
        REPLAY_TEST_LIMITS,
        &weights,
    )
    .await?;
    let mut lazy = EngineRunner::new(
        test_guests::prefill_replay_smoke(),
        forja_cpu::CpuBackend::new(),
        REPLAY_TEST_LIMITS,
        &weights,
    )
    .await?;
    replay.load().await??;
    lazy.load_with_selections(
        None,
        &EngineLoadConfig {
            replay: false,
            ..EngineLoadConfig::default()
        },
    )
    .await??;

    let tokens = patterned_tokens(32_760);
    let replay_output = replay
        .step(EngineStep {
            phase: forja_host::BatchPhase::Prompt,
            tokens: tokens.clone(),
            start_pos: 0,
            taps: true,
        })
        .await??;
    let lazy_output = lazy
        .step(EngineStep {
            phase: forja_host::BatchPhase::Prompt,
            tokens: tokens.clone(),
            start_pos: 0,
            taps: true,
        })
        .await??;
    assert_eq!(
        forward_count(&mut replay, &replay_output).await?,
        count_bits(3)
    );
    assert_eq!(
        forward_count(&mut lazy, &lazy_output).await?,
        count_bits(143)
    );
    let replay_cache = decode_f32_le(&replay.read(&replay_output.taps[0]).await?)?;
    let lazy_cache = decode_f32_le(&lazy.read(&lazy_output.taps[0]).await?)?;
    assert_eq!(replay_cache, lazy_cache);
    assert!(
        replay_cache
            .iter()
            .zip(tokens)
            .all(|(&cached, token)| cached.to_bits()
                == f32::from(u16::try_from(token).unwrap()).to_bits())
    );
    assert_eq!(replay.metrics().live_graphs, 2);
    assert!(
        replay
            .memory_usage()
            .accounted_bytes()
            .is_some_and(|bytes| bytes <= REPLAY_ARENA_BUDGET)
    );

    assert_eq!(run_stub_prefill(&mut replay, 32_760).await?, count_bits(4));
    assert_eq!(run_stub_prefill(&mut replay, 32_760).await?, count_bits(4));
    assert_eq!(replay.metrics().live_graphs, 3);

    let mut calls = 4_u16;
    for length in 30..=37 {
        calls += 1;
        assert_eq!(
            run_stub_prefill(&mut replay, length).await?,
            count_bits(calls)
        );
        calls += 1;
        assert_eq!(
            run_stub_prefill(&mut replay, length).await?,
            count_bits(calls)
        );
        assert!(replay.metrics().live_graphs <= 9);
    }
    assert_eq!(replay.metrics().live_graphs, 9);
    calls += 1;
    assert_eq!(run_stub_prefill(&mut replay, 8).await?, count_bits(calls));
    calls += 1;
    assert_eq!(run_stub_prefill(&mut replay, 8).await?, count_bits(calls));
    assert_eq!(replay.metrics().live_graphs, 9);

    drop((replay, lazy));
    fs::remove_file(weights)?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn metal_chunked_prefill_replay_matches_a_small_lazy_shape() -> Result<(), Box<dyn Error>> {
    let weights = empty_weight_file("prefill-replay-metal")?;
    let mut runner = EngineRunner::new(
        test_guests::prefill_replay_smoke(),
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
        REPLAY_TEST_LIMITS,
        &weights,
    )
    .await?;
    runner.load().await??;
    let tokens = patterned_tokens(2_081);
    let output = runner
        .step(EngineStep {
            phase: forja_host::BatchPhase::Prompt,
            tokens: tokens.clone(),
            start_pos: 0,
            taps: true,
        })
        .await??;
    assert_eq!(forward_count(&mut runner, &output).await?, count_bits(3));
    let cache = decode_f32_le(&runner.read(&output.taps[0]).await?)?;
    assert!(
        cache
            .iter()
            .zip(tokens)
            .all(|(&cached, token)| cached.to_bits()
                == f32::from(u16::try_from(token).unwrap()).to_bits())
    );
    assert_eq!(runner.metrics().live_graphs, 2);
    assert!(
        runner
            .memory_usage()
            .accounted_bytes()
            .is_some_and(|bytes| bytes <= REPLAY_ARENA_BUDGET)
    );
    drop(runner);
    fs::remove_file(weights)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn cpu_small_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::qwen3_bf16(),
        test_guests::qwen3_bf16(),
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
        test_guests::qwen3_bf16(),
        &model_root()?.join("Qwen3-0.6B/model.safetensors"),
        None,
        qwen3::VOCAB,
        &LENGTHS,
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_olmoe_prefill_replay_matches_lazy() -> Result<(), Box<dyn Error>> {
    compare_olmoe_prefill(&[1, 7, 16, 33]).await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_olmoe_prefill_replay_matches_lazy_at_chunk_boundaries() -> Result<(), Box<dyn Error>>
{
    compare_olmoe_prefill(&[511, 512, 513]).await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_olmoe_prefill_replay_matches_lazy_at_long_context() -> Result<(), Box<dyn Error>> {
    compare_olmoe_prefill(&[1_500]).await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_olmoe_prefill_replay_matches_lazy_near_context_limit() -> Result<(), Box<dyn Error>>
{
    compare_olmoe_prefill(&[4_000]).await
}

#[cfg(target_os = "macos")]
async fn compare_olmoe_prefill(lengths: &[u32]) -> Result<(), Box<dyn Error>> {
    compare_prefill(
        test_guests::olmoe(),
        test_guests::olmoe(),
        &model_root()?.join("OLMoE-1B-7B-0924/model.safetensors.index.json"),
        None,
        olmoe::VOCAB,
        lengths,
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
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
        test_guests::qwen3_coder(),
        &model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json"),
        Some(4),
        qwen3_coder::VOCAB,
        &LENGTHS,
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_qwen3_coder_reuses_prefill_graph_across_offsets() -> Result<(), Box<dyn Error>> {
    let component = test_guests::qwen3_coder();
    let weights =
        model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json");
    let load_config = common::load_config(component)?;
    let mut lazy_config = load_config.clone();
    lazy_config.replay = false;
    let mut replay = EngineRunner::new(
        component,
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
        LIMITS,
        &weights,
    )
    .await?;
    let mut lazy = EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        LIMITS,
        &weights,
    )
    .await?;
    replay.load_with_selections(Some(1), &load_config).await??;
    lazy.load_with_selections(Some(1), &lazy_config).await??;

    for offset in [0, 512, 4_096, 32_256] {
        let tokens = (0..512)
            .map(|index| (index + offset) % qwen3_coder::VOCAB)
            .collect::<Vec<_>>();
        let input = EngineDecode {
            phase: forja_host::BatchPhase::Prompt,
            tokens: Some(tokens),
            start_pos: offset,
            sampling: SamplingParams::default(),
        };
        let replay_output = replay.decode(input.clone()).await??;
        let lazy_output = lazy.decode(input).await??;
        assert_eq!(
            read_decode_output(&mut replay, &replay_output).await?,
            read_decode_output(&mut lazy, &lazy_output).await?,
            "prefill differed at offset {offset}"
        );
        assert_eq!(replay.metrics().live_graphs, 1);
    }
    assert!(replay.metrics().live_graphs <= 16);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_qwen3_coder_bounds_exact_prefill_tail_graphs() -> Result<(), Box<dyn Error>> {
    let component = test_guests::qwen3_coder();
    let weights =
        model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json");
    let load_config = common::load_config(component)?;
    let mut lazy_config = load_config.clone();
    lazy_config.replay = false;
    let mut replay = EngineRunner::new(
        component,
        forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
        LIMITS,
        &weights,
    )
    .await?;
    let mut lazy = EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        LIMITS,
        &weights,
    )
    .await?;
    replay.load_with_selections(Some(1), &load_config).await??;
    lazy.load_with_selections(Some(1), &lazy_config).await??;

    let lengths = (2_u32..=9)
        .chain(2..=9)
        .chain(10..=121)
        .chain(2..=73)
        .collect::<Vec<_>>();
    assert_eq!(lengths.len(), 200);
    assert_eq!(
        lengths
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        120
    );
    let mut peak_graphs = 0;
    for length in lengths {
        let tokens = (0..length)
            .map(|index| (index.wrapping_mul(7_919).wrapping_add(17)) % qwen3_coder::VOCAB)
            .collect::<Vec<_>>();
        let input = EngineDecode {
            phase: forja_host::BatchPhase::Prompt,
            tokens: Some(tokens),
            start_pos: 0,
            sampling: SamplingParams::default(),
        };
        let replay_output = replay.decode(input.clone()).await??;
        let lazy_output = lazy.decode(input).await??;
        assert_eq!(
            read_decode_output(&mut replay, &replay_output).await?,
            read_decode_output(&mut lazy, &lazy_output).await?,
            "prefill differed after {length} tokens"
        );
        peak_graphs = peak_graphs.max(replay.metrics().live_graphs);
        assert!(peak_graphs <= 8);
    }
    assert_eq!(peak_graphs, 8);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
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
    let load_config = common::load_config(replay_component)?;
    let mut lazy_load_config = load_config.clone();
    lazy_load_config.replay = false;
    let mut replay = EngineRunner::new(replay_component, replay_backend, LIMITS, weights).await?;
    let mut lazy = EngineRunner::new(lazy_component, lazy_backend, LIMITS, weights).await?;
    let mut reference =
        EngineRunner::new(lazy_component, reference_backend, LIMITS, weights).await?;
    replay.load_with_selections(layers, &load_config).await??;
    lazy.load_with_selections(layers, &lazy_load_config)
        .await??;
    reference
        .load_with_selections(layers, &load_config)
        .await??;
    for &length in lengths {
        let tokens = (0..length)
            .map(|index| (index.wrapping_mul(7_919).wrapping_add(17)) % vocab)
            .collect::<Vec<_>>();
        let replay_prefill = replay
            .decode(EngineDecode {
                phase: forja_host::BatchPhase::Prompt,
                tokens: Some(tokens.clone()),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })
            .await??;
        let lazy_prefill = lazy
            .decode(EngineDecode {
                phase: forja_host::BatchPhase::Prompt,
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
                phase: forja_host::BatchPhase::Prompt,
                tokens: tokens.clone(),
                start_pos: 0,
                taps: true,
            })
            .await??;
        let continuation_token = u32::from_le_bytes(replay_token.as_slice().try_into()?);
        let reference_next = reference
            .step(EngineStep {
                phase: forja_host::BatchPhase::Prompt,
                tokens: vec![continuation_token],
                start_pos: length,
                taps: true,
            })
            .await??;
        let reference_logits = reference.read(&reference_next.logits).await?;

        let replay_next = replay
            .decode(EngineDecode {
                phase: forja_host::BatchPhase::Decode,
                tokens: None,
                start_pos: length,
                sampling: SamplingParams::default(),
            })
            .await??;
        let lazy_next = lazy
            .decode(EngineDecode {
                phase: forja_host::BatchPhase::Decode,
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

async fn run_stub_prefill<B>(
    runner: &mut EngineRunner<B>,
    length: u32,
) -> Result<u32, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let output = runner
        .step(EngineStep {
            phase: forja_host::BatchPhase::Prompt,
            tokens: patterned_tokens(length),
            start_pos: 0,
            taps: false,
        })
        .await??;
    forward_count(runner, &output).await
}

async fn forward_count<B>(
    runner: &mut EngineRunner<B>,
    output: &forja_host::EngineOutput,
) -> Result<u32, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let logits = decode_f32_le(&runner.read(&output.logits).await?)?;
    Ok(logits[0].to_bits())
}

fn count_bits(count: u16) -> u32 {
    f32::from(count).to_bits()
}

fn patterned_tokens(length: u32) -> Vec<u32> {
    (0..length).map(|position| position % 4).collect()
}

fn empty_weight_file(label: &str) -> Result<PathBuf, Box<dyn Error>> {
    let mut header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend_from_slice(&0_f32.to_le_bytes());
    let path = env::temp_dir().join(format!("forja-{label}-{}", std::process::id()));
    fs::write(&path, bytes)?;
    Ok(path)
}

fn model_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env::var("FORJA_MODELS")?))
}
