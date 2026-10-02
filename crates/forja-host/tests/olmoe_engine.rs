//! `OLMoE` engine checks against independent transformer fixtures.

mod common;

use std::{env, error::Error, path::PathBuf, time::Duration};

use forja_host::{EngineRunner, EngineStep, Limits};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, decode_f32_le, normwise_relative_error,
};

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
const ROUTER_NEAR_TIE: f32 = 1.0e-3;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn prompt_hidden_states_and_routing_match_transformers() -> Result<(), Box<dyn Error>> {
    let root = model_root()?;
    let fixtures = FixtureDirectory::open(root.join("golden/OLMoE-1B-7B-0924"))?;
    let fixture = fixtures
        .prompt("short-english")
        .ok_or("short-english fixture is missing")?;
    let tokens = fixture
        .prompt_ids()
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let mut runner = EngineRunner::new(
        test_guests::olmoe(),
        forja_metal::MetalBackend::new()?,
        LIMITS,
        root.join("OLMoE-1B-7B-0924/model.safetensors.index.json"),
    )
    .await?;
    let info = runner.describe().await?;
    assert_eq!(info.vocab, olmoe::VOCAB);
    assert_eq!(info.tap_layers, (1..=16).collect::<Vec<_>>());
    assert_eq!(info.router_layers, (1..=16).collect::<Vec<_>>());
    runner
        .load_with_selections(None, &common::load_config(test_guests::olmoe())?)
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
        assert!(
            error <= BF16_HIDDEN_STATE_TOLERANCE,
            "layer {} relative error {error}",
            index + 1
        );
    }
    let expected = fixture.router_logits().ok_or("router fixture is missing")?;
    let row_width = usize::try_from(olmoe::EXPERTS)?;
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
        assert!(
            error <= BF16_HIDDEN_STATE_TOLERANCE,
            "router layer {} relative error {error}",
            layer + 1
        );
        assert_selected_sets(
            reference,
            &actual,
            row_width,
            usize::try_from(olmoe::TOP_K)?,
        )?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn replay_matches_lazy_decode() -> Result<(), Box<dyn Error>> {
    let weights = model_root()?.join("OLMoE-1B-7B-0924/model.safetensors.index.json");
    let replay = decode_logits(test_guests::olmoe(), &weights).await?;
    let lazy = decode_logits(test_guests::olmoe_no_replay(), &weights).await?;
    assert_eq!(replay, lazy);
    Ok(())
}

#[cfg(target_os = "macos")]
async fn decode_logits(
    component: &std::path::Path,
    weights: &std::path::Path,
) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    let mut runner = EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        LIMITS,
        weights,
    )
    .await?;
    runner
        .load_with_selections(None, &common::load_config(test_guests::olmoe())?)
        .await??;
    runner
        .step(EngineStep {
            tokens: (0..8).collect(),
            start_pos: 0,
            taps: false,
        })
        .await??;
    let mut logits = Vec::new();
    for step in 0..4 {
        let output = runner
            .step(EngineStep {
                tokens: vec![(step * 7_919 + 17) % olmoe::VOCAB],
                start_pos: 8 + step,
                taps: false,
            })
            .await??;
        logits.push(runner.read(&output.logits).await?);
    }
    Ok(logits)
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
        let expected_set = top_k_set(expected, k);
        let actual_set = top_k_set(actual, k);
        if expected_set != actual_set && !near_tie_swap(expected, &expected_set, &actual_set) {
            return Err(format!(
                "router row {row} selected {actual_set:?}, expected {expected_set:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn near_tie_swap(reference: &[f32], expected: &[usize], actual: &[usize]) -> bool {
    let changed = expected
        .iter()
        .filter(|index| !actual.contains(index))
        .chain(actual.iter().filter(|index| !expected.contains(index)))
        .map(|&index| reference[index])
        .collect::<Vec<_>>();
    let minimum = changed.iter().copied().reduce(f32::min);
    let maximum = changed.iter().copied().reduce(f32::max);
    matches!((minimum, maximum), (Some(minimum), Some(maximum)) if maximum - minimum <= ROUTER_NEAR_TIE)
}

fn top_k_set(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(k);
    indices.sort_unstable();
    indices
}

fn model_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(env::var("FORJA_MODELS")?))
}
