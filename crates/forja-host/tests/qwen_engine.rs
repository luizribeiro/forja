//! Native Qwen3 engine checks against independent transformer fixtures.

mod common;

use std::{collections::VecDeque, env, error::Error, path::PathBuf, time::Duration};

use forja_core::Backend;
use forja_host::{
    EngineDecode, EngineRunner, EngineStep, Limits, SamplingParams, bindings::l9o::gpu::compute,
};
use forja_sdk::{Engine, Tensor, Weights};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, PromptFixture, decode_f32_le,
    normwise_relative_error,
};
use qwen3::Qwen3;

const REPLAY_LIMITS: Limits = Limits::new(
    8 * 1024 * 1024 * 1024,
    4,
    1_000_000_000,
    20_000,
    1024 * 1024 * 1024,
)
.with_command_limits(4_096, u64::MAX)
.with_guest_call_timeout(Duration::from_secs(300))
.with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn first_layer_matches_transformers() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let (fixture, mut engine, tokens) = single_token_case()?;
    let actual = engine.first_layer(&tokens)?.to_vec()?;
    let expected = fixture
        .hidden_state(1)
        .ok_or("layer 1 fixture is missing")?
        .values();
    let error = normwise_relative_error(expected, &actual)?;
    assert!(
        error <= BF16_HIDDEN_STATE_TOLERANCE,
        "layer 1 relative error {error}"
    );
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn all_layers_match_transformers() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let (fixture, mut engine, tokens) = single_token_case()?;
    let output = engine.step(forja_sdk::StepInput {
        tokens,
        start_pos: 0,
        taps: true,
    })?;
    assert_eq!(output.taps.len(), 28);
    for (index, tap) in output.taps.iter().enumerate() {
        let actual = tap.to_vec()?;
        let expected = fixture
            .hidden_state(index + 1)
            .ok_or("layer fixture is missing")?
            .values();
        let error = normwise_relative_error(expected, &actual)?;
        assert!(
            error <= BF16_HIDDEN_STATE_TOLERANCE,
            "layer {} relative error {error}",
            index + 1
        );
    }
    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn chunked_prefill_matches_single_pass_and_transformers() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let (fixture, mut engine, tokens) = prompt_case("short-english", 7)?;
    let single = engine.step(forja_sdk::StepInput {
        tokens,
        start_pos: 0,
        taps: true,
    })?;
    let single_logits = single.logits.to_vec()?;
    let single_taps = single
        .taps
        .iter()
        .map(Tensor::to_vec)
        .collect::<Result<Vec<_>, _>>()?;
    drop(engine);

    let (_, mut engine, first) = prompt_case("short-english", 3)?;
    engine.step(forja_sdk::StepInput {
        tokens: first,
        start_pos: 0,
        taps: false,
    })?;
    let suffix = fixture.prompt_ids()[3..7]
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let chunked = engine.step(forja_sdk::StepInput {
        tokens: Tensor::from_slice(&suffix, &[4])?,
        start_pos: 3,
        taps: true,
    })?;

    assert_close("chunked logits", &single_logits, &chunked.logits.to_vec()?)?;
    for (index, (single, chunked)) in single_taps.iter().zip(&chunked.taps).enumerate() {
        let chunked = chunked.to_vec()?;
        assert_close(
            &format!("layer {} single pass", index + 1),
            &single[3 * 1_024..7 * 1_024],
            &chunked,
        )?;
        let golden = fixture
            .hidden_state(index + 1)
            .ok_or("layer fixture is missing")?;
        assert_close(
            &format!("layer {} transformers", index + 1),
            &golden.values()[3 * 1_024..7 * 1_024],
            &chunked,
        )?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_replay_matches_lazy_decode() -> Result<(), Box<dyn Error>> {
    compare_replay(
        vec![
            (
                "Metal Tier 1",
                forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier1)?,
            ),
            (
                "Metal Tier 2",
                forja_metal::MetalBackend::with_graph_replay(forja_metal::MetalGraphReplay::Tier2)?,
            ),
        ],
        forja_metal::MetalBackend::new()?,
        128,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_greedy_selection_matches_host_argmax() -> Result<(), Box<dyn Error>> {
    compare_greedy_selection(
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        test_guests::qwen3_bf16(),
        128,
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_sampled_selection_matches_sequential() -> Result<(), Box<dyn Error>> {
    compare_pipelined_selection(
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        test_guests::qwen3_bf16_no_replay(),
        test_guests::qwen3_bf16(),
        128,
        SamplingParams {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            seed: 0xfeed_beef_dead_cafe,
        },
    )
    .await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_greedy_graph_matches_sample_at_zero_temperature() -> Result<(), Box<dyn Error>> {
    compare_pipelined_selection(
        forja_metal::MetalBackend::new()?,
        forja_metal::MetalBackend::new()?,
        test_guests::qwen3_bf16_no_replay(),
        test_guests::qwen3_bf16(),
        128,
        SamplingParams::default(),
    )
    .await
}

async fn compare_pipelined_selection<B>(
    sequential_backend: B,
    pipelined_backend: B,
    sequential_component: &std::path::Path,
    pipelined_component: &std::path::Path,
    token_count: u32,
    sampling: SamplingParams,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let weights = root.join("Qwen3-0.6B/model.safetensors");
    let mut sequential = EngineRunner::new(
        sequential_component,
        sequential_backend,
        REPLAY_LIMITS,
        &weights,
    )
    .await?;
    let mut pipelined = EngineRunner::new(
        pipelined_component,
        pipelined_backend,
        REPLAY_LIMITS,
        weights,
    )
    .await?;
    let config = common::load_config(test_guests::qwen3_bf16())?;
    sequential.load_with_selections(None, &config).await??;
    pipelined.load_with_selections(None, &config).await??;
    let prompt = (0_u32..8).collect::<Vec<_>>();
    let expected = selected_tokens(&mut sequential, &prompt, token_count, false, None, sampling)
        .await
        .map_err(|error| format!("sequential decode failed: {error}"))?;
    let actual = selected_tokens(&mut pipelined, &prompt, token_count, true, None, sampling)
        .await
        .map_err(|error| format!("pipelined decode failed: {error}"))?;
    assert_eq!(actual, expected);
    let eos = *expected.get(3).ok_or("sequential decode was incomplete")?;
    let stopped = selected_tokens(
        &mut pipelined,
        &prompt,
        token_count,
        true,
        Some(eos),
        sampling,
    )
    .await
    .map_err(|error| format!("EOS decode failed: {error}"))?;
    assert_eq!(stopped, expected[..=3]);
    let next_position = u32::try_from(prompt.len())?
        .checked_add(u32::try_from(stopped.len())?.saturating_sub(1))
        .ok_or("decode position overflowed")?;
    let resumed = pipelined
        .enqueue_decode(EngineDecode {
            tokens: None,
            start_pos: next_position,
            sampling: SamplingParams::default(),
        })
        .await?;
    assert!(matches!(resumed, Err(compute::Error::OpSignature(_))));
    Ok(())
}

async fn selected_tokens<B>(
    runner: &mut EngineRunner<B>,
    prompt: &[u32],
    token_count: u32,
    pipelined: bool,
    eos: Option<u32>,
    sampling: SamplingParams,
) -> Result<Vec<u32>, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    if token_count == 0 {
        return Ok(Vec::new());
    }
    let input = EngineDecode {
        tokens: Some(prompt.to_vec()),
        start_pos: 0,
        sampling,
    };
    let mut tokens = Vec::with_capacity(usize::try_from(token_count)?);
    if pipelined {
        let output = runner
            .enqueue_decode(input)
            .await?
            .map_err(|error| format!("prefill enqueue failed: {error:?}"))?;
        tokens.push(read_token(
            &runner
                .read_queued_token(output)
                .await
                .map_err(|error| format!("prefill read failed: {error:?}"))?,
        )?);
        if tokens.last() == eos.as_ref() {
            return Ok(tokens);
        }
        let replay_base = runner.metrics().submissions;
        let mut consumed = 0_u64;
        let mut replay_count = 0_u32;
        let mut outputs = VecDeque::with_capacity(2);
        let mut position = u32::try_from(prompt.len())?;
        let end = position
            .checked_add(token_count.saturating_sub(1))
            .ok_or("decode position overflowed")?;
        while position < end || !outputs.is_empty() {
            while position < end && outputs.len() < 2 {
                runner.set_profiling(replay_count > 0);
                let output = runner
                    .enqueue_decode(EngineDecode {
                        tokens: None,
                        start_pos: position,
                        sampling,
                    })
                    .await?
                    .map_err(|error| format!("decode enqueue at {position} failed: {error:?}"))?;
                if replay_count > 0 {
                    let profile = runner
                        .take_profile()
                        .ok_or("profiled replay did not produce a profile")?;
                    assert_no_guest_buffer_io(&profile, position);
                }
                runner.set_profiling(false);
                outputs.push_back(output);
                replay_count = next_replay_count(replay_count)?;
                position = position
                    .checked_add(1)
                    .ok_or("decode position overflowed")?;
            }
            let output = outputs.pop_front().ok_or("decode queue is empty")?;
            tokens.push(read_token(
                &runner
                    .read_queued_token(output)
                    .await
                    .map_err(|error| format!("decode read failed: {error:?}"))?,
            )?);
            consumed = consumed
                .checked_add(1)
                .ok_or("submission count overflowed")?;
            let expected = replay_base
                .checked_add(consumed)
                .ok_or("submission count overflowed")?;
            runner.wait_for_submissions(expected).await?;
            if tokens.last() == eos.as_ref() {
                if outputs.is_empty() {
                    return Err("EOS did not leave a speculative replay queued".into());
                }
                for output in outputs.drain(..) {
                    runner.discard_queued_decode(output).await?;
                }
                break;
            }
        }
    } else {
        let output = runner.decode(input).await??;
        tokens.push(read_token(&runner.read(&output.token).await?)?);
        for offset in 0..token_count.saturating_sub(1) {
            let position = u32::try_from(prompt.len())?
                .checked_add(offset)
                .ok_or("decode position overflowed")?;
            let output = runner
                .decode(EngineDecode {
                    tokens: None,
                    start_pos: position,
                    sampling,
                })
                .await??;
            tokens.push(read_token(&runner.read(&output.token).await?)?);
        }
    }
    Ok(tokens)
}

fn assert_no_guest_buffer_io(profile: &forja_host::EngineStepProfile, position: u32) {
    assert_eq!(
        profile.imports.write.count, 0,
        "replay at {position} issued a guest tensor write"
    );
    assert_eq!(
        profile.imports.read.count, 0,
        "replay at {position} issued a guest tensor read"
    );
}

fn next_replay_count(count: u32) -> Result<u32, Box<dyn Error>> {
    count
        .checked_add(1)
        .ok_or_else(|| "replay count overflowed".into())
}

async fn compare_greedy_selection<B>(
    host_backend: B,
    selected_backend: B,
    component: &std::path::Path,
    token_count: u32,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let weights = root.join("Qwen3-0.6B/model.safetensors");
    let mut host = EngineRunner::new(component, host_backend, REPLAY_LIMITS, &weights).await?;
    let mut selected =
        EngineRunner::new(component, selected_backend, REPLAY_LIMITS, weights).await?;
    let config = common::load_config(test_guests::qwen3_bf16())?;
    host.load_with_selections(None, &config).await??;
    selected.load_with_selections(None, &config).await??;
    let prompt = (0_u32..8).collect::<Vec<_>>();
    let host_output = host
        .step(EngineStep {
            tokens: prompt.clone(),
            start_pos: 0,
            taps: false,
        })
        .await??;
    let selected_output = selected
        .decode(EngineDecode {
            tokens: Some(prompt),
            start_pos: 0,
            sampling: SamplingParams::default(),
        })
        .await??;
    let mut host_token = host_argmax(&host.read(&host_output.logits).await?)?;
    let mut selected_token = read_token(&selected.read(&selected_output.token).await?)?;
    assert_eq!(selected_token, host_token, "prefill selection differed");

    for offset in 1..token_count {
        let position = 7_u32
            .checked_add(offset)
            .ok_or("decode position overflowed")?;
        let replacement =
            (offset == token_count / 2).then(|| selected_token.wrapping_add(1) % qwen3::VOCAB);
        let input_token = replacement.unwrap_or(host_token);
        let host_output = host
            .step(EngineStep {
                tokens: vec![input_token],
                start_pos: position,
                taps: false,
            })
            .await??;
        let selected_output = selected
            .decode(EngineDecode {
                tokens: replacement.map(|token| vec![token]),
                start_pos: position,
                sampling: SamplingParams::default(),
            })
            .await??;
        host_token = host_argmax(&host.read(&host_output.logits).await?)?;
        selected_token = read_token(&selected.read(&selected_output.token).await?)?;
        assert_eq!(
            selected_token, host_token,
            "selection differed at {position}"
        );
    }
    let position = 7_u32
        .checked_add(token_count)
        .ok_or("decode position overflowed")?;
    let replacement = selected_token.wrapping_add(1) % qwen3::VOCAB;
    let host_output = host
        .step(EngineStep {
            tokens: vec![replacement],
            start_pos: position,
            taps: false,
        })
        .await??;
    let selected_output = selected
        .decode(EngineDecode {
            tokens: Some(vec![replacement]),
            start_pos: position,
            sampling: SamplingParams::default(),
        })
        .await??;
    assert_eq!(
        read_token(&selected.read(&selected_output.token).await?)?,
        host_argmax(&host.read(&host_output.logits).await?)?,
        "replacement token was not honored at {position}"
    );
    Ok(())
}

fn host_argmax(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    let values = decode_f32_le(bytes)?;
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| u32::try_from(index))
        .transpose()?
        .ok_or_else(|| "cannot select from empty logits".into())
}

fn read_token(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    Ok(u32::from_le_bytes(bytes.try_into()?))
}

async fn compare_replay<B>(
    replay_backends: Vec<(&'static str, B)>,
    lazy_backend: B,
    token_count: u32,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let weights = root.join("Qwen3-0.6B/model.safetensors");
    let mut replays = Vec::with_capacity(replay_backends.len());
    for (label, backend) in replay_backends {
        replays.push((
            label,
            EngineRunner::new(test_guests::qwen3_bf16(), backend, REPLAY_LIMITS, &weights).await?,
        ));
    }
    let mut lazy = EngineRunner::new(
        test_guests::qwen3_bf16_no_replay(),
        lazy_backend,
        REPLAY_LIMITS,
        weights,
    )
    .await?;
    let config = common::load_config(test_guests::qwen3_bf16())?;
    for (_, replay) in &mut replays {
        replay.load_with_selections(None, &config).await??;
    }
    lazy.load_with_selections(None, &config).await??;
    let prompt = (0_u32..8).collect::<Vec<_>>();
    for (_, replay) in &mut replays {
        replay
            .step(EngineStep {
                tokens: prompt.clone(),
                start_pos: 0,
                taps: false,
            })
            .await??;
    }
    lazy.step(EngineStep {
        tokens: prompt,
        start_pos: 0,
        taps: false,
    })
    .await??;

    for step in 0..token_count {
        let start_pos = 8 + step;
        let token = (step * 7_919 + 17) % qwen3::VOCAB;
        let lazy_output = lazy
            .step(EngineStep {
                tokens: vec![token],
                start_pos,
                taps: false,
            })
            .await??;
        let lazy_logits = lazy.read(&lazy_output.logits).await?;
        for (label, replay) in &mut replays {
            let replay_output = replay
                .step(EngineStep {
                    tokens: vec![token],
                    start_pos,
                    taps: false,
                })
                .await??;
            assert_eq!(
                replay.read(&replay_output.logits).await?,
                lazy_logits,
                "{label} decode logits differed at position {start_pos}"
            );
        }
    }
    Ok(())
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    let error = normwise_relative_error(expected, actual)?;
    assert!(
        error <= BF16_HIDDEN_STATE_TOLERANCE,
        "{label} relative error {error}"
    );
    Ok(())
}

fn single_token_case() -> Result<(PromptFixture, Qwen3, Tensor<u32>), Box<dyn Error>> {
    prompt_case("single-token", 1)
}

fn prompt_case(
    name: &str,
    token_count: usize,
) -> Result<(PromptFixture, Qwen3, Tensor<u32>), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let fixtures = FixtureDirectory::open(root.join("golden/qwen3-0.6b"))?;
    let fixture = fixtures
        .prompt(name)
        .ok_or("prompt fixture is missing")?
        .clone();
    let tokens = fixture
        .prompt_ids()
        .get(..token_count)
        .ok_or("prompt fixture has too few tokens")?
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = Weights::open(root.join("Qwen3-0.6B/model.safetensors"))?;
    let engine = Qwen3::load(&weights, native_load_config()?)?;
    let length = u32::try_from(tokens.len())?;
    Ok((fixture, engine, Tensor::from_slice(&tokens, &[length])?))
}

fn native_load_config() -> Result<forja_sdk::EngineLoadConfig, Box<dyn Error>> {
    let config = common::load_config(test_guests::qwen3())?;
    Ok(forja_sdk::EngineLoadConfig {
        num_hidden_layers: None,
        tunings: config.tunings,
        fixed_variant_picks: config
            .fixed_variant_picks
            .into_iter()
            .map(|pick| forja_sdk::FixedVariantPick {
                site: pick.site,
                name: pick.name,
            })
            .collect(),
        variant_rule_picks: config
            .variant_rule_picks
            .into_iter()
            .map(|pick| forja_sdk::VariantRulePick {
                site: pick.site,
                parameter: pick.parameter,
                arms: pick
                    .arms
                    .into_iter()
                    .map(|arm| forja_sdk::VariantPickArm {
                        lo: arm.lo,
                        hi: arm.hi,
                        name: arm.name,
                    })
                    .collect(),
            })
            .collect(),
    })
}
