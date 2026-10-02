//! Qwen3-Coder quantized engine checks against independent transformer fixtures.

mod common;

use std::{env, error::Error, path::PathBuf, time::Duration};

use forja_host::{EngineDecode, EngineRunner, EngineStep, Limits, SamplingParams};
use golden_fixtures::{FixtureDirectory, decode_f32_le, normwise_relative_error};

const LIMITS: Limits = Limits::new(
    24 * 1024 * 1024 * 1024,
    4,
    1_000_000_000,
    20_000,
    1024 * 1024 * 1024,
)
.with_command_limits(4_096, u64::MAX)
.with_guest_call_timeout(Duration::from_secs(300))
.with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));
const FULL_LAYERS: u32 = 48;
const REPLAY_LAYERS: u32 = 4;
const HIDDEN_TOLERANCE: f64 = 1.0e-5;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn hidden_states_and_routing_match_same_byte_reference() -> Result<(), Box<dyn Error>> {
    let root = model_root()?;
    let fixtures = FixtureDirectory::open(root.join("golden/Qwen3-Coder-30B-A3B-Instruct-4bit"))?;
    let fixture = fixtures
        .prompt("python")
        .ok_or("python fixture is missing")?;
    let tokens = fixture
        .prompt_ids()
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let mut runner = runner(test_guests::qwen3_coder()).await?;
    let info = runner.describe().await?;
    assert_eq!(info.vocab, qwen3_coder::VOCAB);
    assert!(
        info.tap_layers
            .starts_with(&(1..=FULL_LAYERS).collect::<Vec<_>>())
    );
    assert!(
        info.router_layers
            .starts_with(&(1..=FULL_LAYERS).collect::<Vec<_>>())
    );
    runner
        .load_with_selections(Some(FULL_LAYERS), &load_config()?)
        .await??;
    let output = runner
        .step(EngineStep {
            tokens,
            start_pos: 0,
            taps: true,
        })
        .await??;

    for (index, tap) in output.taps.iter().enumerate() {
        let actual = decode_f32_le(&runner.read(tap).await?)?;
        let expected = fixture
            .hidden_state(index + 1)
            .ok_or("hidden-state fixture is missing")?;
        let error = normwise_relative_error(expected.values(), &actual)?;
        eprintln!(
            "Qwen3-Coder same-byte layer {} error {error:.8e}",
            index + 1
        );
        assert!(
            error <= HIDDEN_TOLERANCE,
            "layer {} relative error {error}",
            index + 1
        );
    }
    let expected = fixture.router_logits().ok_or("router fixture is missing")?;
    let row_width = usize::try_from(qwen3_coder::EXPERTS)?;
    for (layer, router) in output.router_logits.iter().enumerate() {
        let actual = decode_f32_le(&runner.read(router).await?)?;
        let start = layer
            .checked_mul(fixture.prompt_ids().len())
            .and_then(|rows| rows.checked_mul(row_width))
            .ok_or("router fixture offset overflowed")?;
        let end = start
            .checked_add(actual.len())
            .ok_or("router fixture offset overflowed")?;
        let reference = expected
            .values()
            .get(start..end)
            .ok_or("router fixture is incomplete")?;
        let error = normwise_relative_error(reference, &actual)?;
        eprintln!(
            "Qwen3-Coder same-byte router layer {} error {error:.8e}",
            layer + 1
        );
        assert!(
            error <= HIDDEN_TOLERANCE,
            "router layer {} relative error {error}",
            layer + 1
        );
        assert_selected_sets(
            reference,
            &actual,
            row_width,
            usize::try_from(qwen3_coder::TOP_K)?,
        )?;
    }

    let float_fixtures = FixtureDirectory::open(root.join("golden/Qwen3-Coder-30B-A3B-Instruct"))?;
    let float_fixture = float_fixtures
        .prompt("python")
        .ok_or("float-weight python fixture is missing")?;
    for (index, tap) in output
        .taps
        .iter()
        .take(usize::try_from(REPLAY_LAYERS - 1)?)
        .enumerate()
    {
        let actual = decode_f32_le(&runner.read(tap).await?)?;
        let expected = float_fixture
            .hidden_state(index + 1)
            .ok_or("float-weight hidden-state fixture is missing")?;
        let error = normwise_relative_error(expected.values(), &actual)?;
        eprintln!(
            "Qwen3-Coder float-weight layer {} error {error:.8e}",
            index + 1
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn replay_matches_lazy_greedy_and_sampled_decode() -> Result<(), Box<dyn Error>> {
    let replay = decode_outputs(true).await?;
    let lazy = decode_outputs(false).await?;
    assert_eq!(replay, lazy);
    Ok(())
}

#[cfg(target_os = "macos")]
async fn decode_outputs(replay: bool) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Box<dyn Error>> {
    let mut runner = runner_with_graph_replay(forja_metal::MetalGraphReplay::Tier2).await?;
    let mut config = load_config()?;
    config.replay = replay;
    runner
        .load_with_selections(Some(REPLAY_LAYERS), &config)
        .await??;
    let greedy = SamplingParams::default();
    let sampled = SamplingParams {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.9,
        seed: 17,
    };
    let inputs = [
        EngineDecode {
            tokens: Some(vec![17, 33, 4097]),
            start_pos: 0,
            sampling: greedy,
        },
        EngineDecode {
            tokens: None,
            start_pos: 3,
            sampling: greedy,
        },
        EngineDecode {
            tokens: None,
            start_pos: 4,
            sampling: sampled,
        },
    ];
    let mut outputs = Vec::new();
    for input in inputs {
        let output = runner.decode(input).await??;
        outputs.push((
            runner.read(&output.token).await?,
            runner.read(&output.logits).await?,
        ));
    }
    Ok(outputs)
}

#[cfg(target_os = "macos")]
async fn runner(
    component: &std::path::Path,
) -> Result<EngineRunner<forja_metal::MetalBackend>, Box<dyn Error>> {
    Ok(EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        LIMITS,
        model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json"),
    )
    .await?)
}

#[cfg(target_os = "macos")]
async fn runner_with_graph_replay(
    graph_replay: forja_metal::MetalGraphReplay,
) -> Result<EngineRunner<forja_metal::MetalBackend>, Box<dyn Error>> {
    Ok(EngineRunner::new(
        test_guests::qwen3_coder(),
        forja_metal::MetalBackend::with_graph_replay(graph_replay)?,
        LIMITS,
        model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json"),
    )
    .await?)
}

fn load_config() -> Result<forja_host::EngineLoadConfig, Box<dyn Error>> {
    common::load_config(test_guests::qwen3_coder())
}

fn assert_selected_sets(
    expected: &[f32],
    actual: &[f32],
    width: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    if expected.len() != actual.len() || !expected.len().is_multiple_of(width) {
        return Err("router logits have incompatible shapes".into());
    }
    for (row, (expected, actual)) in expected
        .chunks_exact(width)
        .zip(actual.chunks_exact(width))
        .enumerate()
    {
        let expected = top_k_set(expected, k);
        let actual = top_k_set(actual, k);
        if expected != actual {
            return Err(
                format!("router row {row} selected {actual:?}, expected {expected:?}").into(),
            );
        }
    }
    Ok(())
}

fn top_k_set(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(k);
    indices.sort_unstable();
    indices
}

fn model_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(
        env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?,
    ))
}
