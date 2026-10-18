//! Qwen3-Coder quantized engine checks against independent transformer fixtures.

use crate::common;

use std::{
    env,
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};

use forja_core::Op;
use forja_host::{
    EngineBeginDecode, EngineDecode, EngineRunner, EngineStep, Limits, SamplingParams,
};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, decode_f32_le, normwise_relative_error,
};

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
const SERVING_TOKENS: u32 = 128;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_hidden_states_and_routing_match_same_byte_reference() -> Result<(), Box<dyn Error>> {
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
            error <= BF16_HIDDEN_STATE_TOLERANCE,
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
            error <= BF16_HIDDEN_STATE_TOLERANCE,
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
async fn metal_replay_matches_lazy_greedy_and_sampled_decode() -> Result<(), Box<dyn Error>> {
    let replay = decode_outputs(true).await?;
    let lazy = decode_outputs(false).await?;
    assert_eq!(replay, lazy);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_rewind_matches_decode_without_speculation() -> Result<(), Box<dyn Error>> {
    let mut runner = runner_with_graph_replay(forja_metal::MetalGraphReplay::Tier2).await?;
    runner
        .load_with_selections(Some(REPLAY_LAYERS), &load_config()?)
        .await??;
    let expected = constrained_continuation(&mut runner, false).await?;
    let rewound = constrained_continuation(&mut runner, true).await?;
    assert_eq!(rewound, expected);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and FORJA_PERFORMANCE_GATE"]
async fn metal_constrained_decode_meets_latency_gate() -> Result<(), Box<dyn Error>> {
    if env::var_os("FORJA_PERFORMANCE_GATE").is_none() {
        return Ok(());
    }
    let mut runner = runner_with_graph_replay(forja_metal::MetalGraphReplay::Tier2).await?;
    runner.load_with_selections(None, &load_config()?).await??;
    timed_continuation(&mut runner, false, 8).await?;
    timed_continuation(&mut runner, true, 8).await?;

    let ordinary_a = timed_continuation(&mut runner, false, SERVING_TOKENS).await?;
    let constrained_b = timed_continuation(&mut runner, true, SERVING_TOKENS).await?;
    let constrained_b2 = timed_continuation(&mut runner, true, SERVING_TOKENS).await?;
    let ordinary_a2 = timed_continuation(&mut runner, false, SERVING_TOKENS).await?;
    assert_eq!(ordinary_a.tokens, constrained_b.tokens);
    assert_eq!(ordinary_a.tokens, constrained_b2.tokens);
    assert_eq!(ordinary_a.tokens, ordinary_a2.tokens);

    let ordinary = [ordinary_a, ordinary_a2];
    let constrained = [constrained_b, constrained_b2];
    let ordinary_itl = ordinary.iter().flat_map(|run| run.itl.iter().copied());
    let constrained_itl = constrained.iter().flat_map(|run| run.itl.iter().copied());
    let ordinary_percentiles = percentiles(ordinary_itl);
    let constrained_percentiles = percentiles(constrained_itl);
    let deltas = [
        constrained_percentiles[0].saturating_sub(ordinary_percentiles[0]),
        constrained_percentiles[1].saturating_sub(ordinary_percentiles[1]),
        constrained_percentiles[2].saturating_sub(ordinary_percentiles[2]),
    ];
    eprintln!(
        "30B ordinary ITL p50/p90/p99 {ordinary_percentiles:?}; constrained {constrained_percentiles:?}; deltas {deltas:?}"
    );
    eprintln!(
        "30B slowest ordinary {:?}/{:?}; constrained {:?}/{:?}",
        ordinary[0].slowest(),
        ordinary[1].slowest(),
        constrained[0].slowest(),
        constrained[1].slowest()
    );
    assert!(ordinary_percentiles[0] <= Duration::from_millis(11));
    assert!(deltas[0] <= Duration::from_millis(1));
    assert!(deltas[1] <= Duration::from_micros(1_500));
    assert!(deltas[2] <= Duration::from_millis(2));

    let ordinary_idle: Duration = ordinary.iter().map(TimedRun::gpu_idle).sum();
    let constrained_idle: Duration = constrained.iter().map(TimedRun::gpu_idle).sum();
    let ordinary_ttft = std::array::from_fn::<_, 2, _>(|index| ordinary[index].ttft);
    let constrained_ttft = std::array::from_fn::<_, 2, _>(|index| constrained[index].ttft);
    let ordinary_e2e = std::array::from_fn::<_, 2, _>(|index| ordinary[index].wall);
    let constrained_e2e = std::array::from_fn::<_, 2, _>(|index| constrained[index].wall);
    let (begin, gap, masked_gpu) = profile_masked_sample(&mut runner).await?;
    let rewind = profile_rewind(&mut runner).await?;
    eprintln!(
        "30B TTFT ordinary {ordinary_ttft:?}, constrained {constrained_ttft:?}; E2E ordinary {ordinary_e2e:?}, constrained {constrained_e2e:?}; GPU idle ordinary {ordinary_idle:?}, constrained {constrained_idle:?}; mask upload {} bytes; begin {begin:?}; begin-to-finish gap {gap:?}; masked Sample GPU {masked_gpu:?}; rewind {rewind:?}",
        qwen3_coder::VOCAB / 8
    );
    Ok(())
}

#[cfg(target_os = "macos")]
struct TimedRun {
    tokens: Vec<u32>,
    itl: Vec<Duration>,
    ttft: Duration,
    wall: Duration,
    gpu: Duration,
}

#[cfg(target_os = "macos")]
impl TimedRun {
    fn gpu_idle(&self) -> Duration {
        self.wall.saturating_sub(self.gpu)
    }

    fn slowest(&self) -> Vec<(usize, Duration)> {
        let mut indexed = self.itl.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_unstable_by_key(|(_, duration)| *duration);
        indexed.into_iter().rev().take(8).collect()
    }
}

#[cfg(target_os = "macos")]
async fn timed_continuation(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    constrained: bool,
    count: u32,
) -> Result<TimedRun, Box<dyn Error>> {
    let sampling = SamplingParams::default();
    let prompt = vec![17, 33, 4097];
    let before = runner.metrics();
    let started = Instant::now();
    let first = runner
        .enqueue_decode(EngineDecode {
            tokens: Some(prompt.clone()),
            start_pos: 0,
            sampling,
        })
        .await??;
    let allow = vec![u32::MAX; usize::try_from(qwen3_coder::VOCAB / 32)?];
    let mut active = Some(first);
    let mut tokens = Vec::with_capacity(usize::try_from(count)?);
    let mut arrivals = Vec::with_capacity(usize::try_from(count)?);
    let mut position = u32::try_from(prompt.len())?;
    while u32::try_from(tokens.len())? < count {
        let needed = u32::try_from(tokens.len())?.saturating_add(1) < count;
        let queued = if needed {
            Some(
                runner
                    .enqueue_decode(EngineDecode {
                        tokens: None,
                        start_pos: position,
                        sampling,
                    })
                    .await??,
            )
        } else {
            None
        };
        let output = active
            .take()
            .ok_or("timed continuation has no active decode")?;
        let token = if constrained {
            let token = read_token(&runner.read_queued_token_retained(&output).await?)?;
            let word = usize::try_from(token / 32)?;
            if allow
                .get(word)
                .is_none_or(|bits| bits & (1_u32 << (token % 32)) == 0)
            {
                return Err("speculative token is not allowed".into());
            }
            runner.release_queued_decode(output)?;
            token
        } else {
            read_token(&runner.read_queued_token(output).await?)?
        };
        tokens.push(token);
        arrivals.push(started.elapsed());
        if let Some(output) = queued {
            active = Some(output);
        }
        if needed {
            position = position
                .checked_add(1)
                .ok_or("decode position overflowed")?;
        }
    }
    runner.wait_for_queued_submissions().await?;
    let wall = started.elapsed();
    let after = runner.metrics();
    let ttft = arrivals
        .first()
        .copied()
        .ok_or("timed continuation generated no tokens")?;
    let itl = arrivals
        .windows(2)
        .map(|times| times[1].saturating_sub(times[0]))
        .collect();
    Ok(TimedRun {
        tokens,
        itl,
        ttft,
        wall,
        gpu: after.gpu_time.saturating_sub(before.gpu_time),
    })
}

#[cfg(target_os = "macos")]
fn percentiles(values: impl Iterator<Item = Duration>) -> [Duration; 3] {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    [50, 90, 99].map(|percentile| {
        let rank = values.len().saturating_mul(percentile).saturating_add(99) / 100;
        values[rank.saturating_sub(1).min(values.len().saturating_sub(1))]
    })
}

#[cfg(target_os = "macos")]
async fn profile_masked_sample(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
) -> Result<(Duration, Duration, Duration), Box<dyn Error>> {
    let prompt = vec![17, 33, 4097];
    let first = runner
        .decode(EngineDecode {
            tokens: Some(prompt.clone()),
            start_pos: 0,
            sampling: SamplingParams::default(),
        })
        .await??;
    runner.read(&first.token).await?;
    let position = u32::try_from(prompt.len())?;
    runner.set_profiling(true);
    let logits = runner
        .begin_decode(EngineBeginDecode {
            tokens: None,
            start_pos: position,
        })
        .await??;
    let begin = runner
        .take_profile()
        .ok_or("constrained forward was not profiled")?
        .wall_time;
    let allow = vec![u32::MAX; usize::try_from(qwen3_coder::VOCAB / 32)?];
    let gap_started = Instant::now();
    let gap = gap_started.elapsed();
    let output = runner
        .finish_decode(&logits, position, SamplingParams::default(), &allow)
        .await??;
    let profile = runner
        .take_profile()
        .ok_or("masked selection was not profiled")?;
    runner.set_profiling(false);
    runner.read_queued_token(output).await?;
    let masked_gpu = profile
        .submission
        .into_iter()
        .flat_map(|submission| submission.per_dispatch)
        .find(|dispatch| matches!(dispatch.op, Op::SampleMasked { .. }))
        .map(|dispatch| dispatch.gpu_time)
        .ok_or("masked Sample dispatch is missing")?;
    Ok((begin, gap, masked_gpu))
}

#[cfg(target_os = "macos")]
async fn profile_rewind(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
) -> Result<Duration, Box<dyn Error>> {
    let prompt = vec![17, 33, 4097];
    let position = u32::try_from(prompt.len())?;
    let first = runner
        .enqueue_decode(EngineDecode {
            tokens: Some(prompt),
            start_pos: 0,
            sampling: SamplingParams::default(),
        })
        .await??;
    runner.read_queued_token(first).await?;
    let speculative = runner
        .enqueue_decode(EngineDecode {
            tokens: None,
            start_pos: position,
            sampling: SamplingParams::default(),
        })
        .await??;
    runner.discard_queued_decode(speculative).await?;
    runner.set_profiling(true);
    runner.rewind_decode(position).await??;
    let rewind = runner
        .take_profile()
        .ok_or("rewind was not profiled")?
        .wall_time;
    runner.set_profiling(false);
    Ok(rewind)
}

#[cfg(target_os = "macos")]
async fn constrained_continuation(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    speculate: bool,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let sampling = SamplingParams {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.9,
        seed: 17,
    };
    let prompt = vec![17, 33, 4097];
    let first = runner
        .enqueue_decode(EngineDecode {
            tokens: Some(prompt.clone()),
            start_pos: 0,
            sampling,
        })
        .await??;
    let mut tokens = vec![read_token(&runner.read_queued_token(first).await?)?];
    let first_position = u32::try_from(prompt.len())?;
    if speculate {
        let output = runner
            .enqueue_decode(EngineDecode {
                tokens: None,
                start_pos: first_position,
                sampling,
            })
            .await??;
        runner.discard_queued_decode(output).await?;
        runner.rewind_decode(first_position).await??;
    }
    let allow = vec![u32::MAX; usize::try_from(qwen3_coder::VOCAB / 32)?];
    let end = first_position
        .checked_add(2)
        .ok_or("decode position overflowed")?;
    for position in first_position..end {
        let logits = runner
            .begin_decode(EngineBeginDecode {
                tokens: None,
                start_pos: position,
            })
            .await??;
        let output = runner
            .finish_decode(&logits, position, sampling, &allow)
            .await??;
        tokens.push(read_token(&runner.read_queued_token(output).await?)?);
    }
    Ok(tokens)
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

fn read_token(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    Ok(u32::from_le_bytes(bytes.try_into()?))
}

fn model_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(
        env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?,
    ))
}
