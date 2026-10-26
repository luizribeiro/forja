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
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchSelect,
    EngineBatchSelectLane, EngineBeginDecode, EngineDecode, EngineRunner, EngineStep, Limits,
    PagedPrefix, SamplingParams,
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
async fn metal_prompt_chunk_schedules_preserve_logits_and_tokens() -> Result<(), Box<dyn Error>> {
    let limits = Limits::new(
        64 * 1024 * 1024 * 1024,
        4,
        1_000_000_000,
        20_000,
        1024 * 1024 * 1024,
    )
    .with_command_limits(4_096, u64::MAX)
    .with_guest_call_timeout(Duration::from_secs(300))
    .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));
    let mut runner = runner_with_limits(test_guests::qwen3_coder(), limits).await?;
    runner
        .load_with_selections(Some(FULL_LAYERS), &load_config()?)
        .await??;
    let tokens = (0..2_752)
        .map(|position| 100 + position % 997)
        .collect::<Vec<_>>();
    let small = chunked_prompt_outputs(&mut runner, &tokens, 512).await?;
    let large = chunked_prompt_outputs(&mut runner, &tokens, 2_048).await?;
    for (index, (small, large)) in small.iter().zip(&large).enumerate() {
        assert!(small.0 == large.0, "logits differ at output {index}");
        assert_eq!(small.1, large.1, "tokens differ at output {index}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn chunked_prompt_outputs(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    allowance: u32,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Box<dyn Error>> {
    runner.prepare_paged_lane(0, None, 0)?;
    runner.prepare_paged_lane(0, None, u64::try_from(tokens.len())? + 4)?;
    let mut outputs = Vec::new();
    let mut next = None;
    let chunking = forja_host::PromptChunking::new(std::num::NonZeroU32::new(512).unwrap());
    let mut position = 0;
    for work in tokens.chunks(usize::try_from(allowance)?) {
        for part in chunking.pieces(work) {
            let output = runner
                .batch_step(EngineBatch {
                    shape: BatchShape {
                        phase: BatchPhase::Prompt,
                        bucket: 1,
                        tokens: 512,
                        selection: BatchSelection::Combined,
                    },
                    lanes: vec![Some(EngineBatchLane {
                        slot: 0,
                        tokens: Some(part.to_vec()),
                        start_pos: position,
                        sampling: SamplingParams::default(),
                    })],
                })
                .await??;
            let logits = runner.read(&output.logits).await?;
            let selected = runner.read(&output.tokens).await?;
            next = Some((logits, selected));
            position += u32::try_from(part.len())?;
        }
    }
    outputs.push(next.ok_or("empty prompt")?);
    for offset in 0..3 {
        let selected = batch_tokens(&outputs.last().ok_or("missing output")?.1)?[0];
        let output = runner
            .batch_step(EngineBatch {
                shape: BatchShape {
                    phase: BatchPhase::Decode,
                    bucket: 1,
                    tokens: 1,
                    selection: BatchSelection::Combined,
                },
                lanes: vec![Some(EngineBatchLane {
                    slot: 0,
                    tokens: Some(vec![selected]),
                    start_pos: u32::try_from(tokens.len())? + offset,
                    sampling: SamplingParams::default(),
                })],
            })
            .await??;
        outputs.push((
            runner.read(&output.logits).await?,
            runner.read(&output.tokens).await?,
        ));
    }
    Ok(outputs)
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_batch_tokens_match_single_lanes_exactly() -> Result<(), Box<dyn Error>> {
    let mut runner = runner(test_guests::qwen3_coder()).await?;
    let mut config = load_config()?;
    config.paged_cache = true;
    runner.load_with_selections(Some(1), &config).await??;
    let sampling_modes = [
        SamplingParams::default(),
        SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            seed: 17,
        },
    ];
    for sampling in sampling_modes {
        for bucket in [2, 4, 8] {
            for lane in 0..8 {
                runner.prepare_paged_lane(lane, None, 0)?;
            }
            let prefixes = (0..bucket)
                .map(|lane| 1 + (lane * 3) % 15)
                .collect::<Vec<_>>();
            let inputs = (0..bucket).map(|lane| 17 + lane).collect::<Vec<_>>();
            let mut singles = Vec::new();
            for (&prefix, &token) in prefixes.iter().zip(&inputs) {
                let retained = batch_prefix(&mut runner, prefix).await?;
                runner.prepare_paged_lane(0, Some(retained), 64)?;
                singles.push(
                    batch_decode(&mut runner, 1, &[Some((prefix, token))], sampling).await?[0],
                );
            }
            let mut retained = Vec::new();
            for &prefix in &prefixes {
                retained.push(batch_prefix(&mut runner, prefix).await?);
            }
            for (lane, prefix) in retained.into_iter().enumerate() {
                runner.prepare_paged_lane(u32::try_from(lane)?, Some(prefix), 64)?;
            }
            let lanes = prefixes
                .iter()
                .copied()
                .zip(inputs.iter().copied())
                .map(Some)
                .collect::<Vec<_>>();
            let batched = batch_decode(&mut runner, bucket, &lanes, sampling).await?;
            assert_eq!(batched, singles);
        }
    }
    for lane in 0..8 {
        runner.prepare_paged_lane(lane, None, 0)?;
    }
    let active = [(0_usize, 7_u32, 31_u32), (3, 10, 37), (7, 13, 41)];
    let mut singles = Vec::new();
    let mut prefixes = Vec::new();
    for &(_, position, token) in &active {
        let prefix = batch_prefix(&mut runner, position).await?;
        runner.prepare_paged_lane(0, Some(prefix), 64)?;
        singles.push(
            batch_decode(
                &mut runner,
                1,
                &[Some((position, token))],
                SamplingParams::default(),
            )
            .await?[0],
        );
        prefixes.push(batch_prefix(&mut runner, position).await?);
    }
    let mut lanes = vec![None; 8];
    for ((lane, position, token), prefix) in active.into_iter().zip(prefixes) {
        runner.prepare_paged_lane(u32::try_from(lane)?, Some(prefix), 64)?;
        lanes[lane] = Some((position, token));
    }
    let padded = batch_decode(&mut runner, 8, &lanes, SamplingParams::default()).await?;
    assert_eq!(&[padded[0], padded[3], padded[7]], singles.as_slice());
    assert!(
        padded
            .iter()
            .enumerate()
            .all(|(lane, &token)| lanes[lane].is_some() || token == 0)
    );
    assert_lane_mask_changes_replay().await?;
    assert_batch_graph_modes().await?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn assert_lane_mask_changes_replay() -> Result<(), Box<dyn Error>> {
    let patterns = [
        [(0_usize, 1_u32, 31_u32), (3, 4, 37), (7, 7, 41)],
        [(1, 2, 43), (4, 5, 47), (6, 8, 53)],
        [(0, 3, 59), (2, 6, 61), (7, 9, 67)],
        [(1, 4, 71), (3, 7, 73), (5, 10, 79)],
        [(0, 5, 83), (4, 8, 89), (7, 11, 97)],
    ];
    let mut singles = runner(test_guests::qwen3_coder()).await?;
    let mut config = load_config()?;
    config.paged_cache = true;
    singles.load_with_selections(Some(1), &config).await??;
    let mut expected = Vec::new();
    for pattern in patterns {
        let mut tokens = Vec::new();
        for &(_, position, token) in &pattern {
            let prefix = batch_prefix(&mut singles, position).await?;
            singles.prepare_paged_lane(0, Some(prefix), 64)?;
            tokens.push(
                batch_decode(
                    &mut singles,
                    1,
                    &[Some((position, token))],
                    SamplingParams::default(),
                )
                .await?[0],
            );
        }
        expected.push(tokens);
    }

    let mut batched = runner(test_guests::qwen3_coder()).await?;
    batched.load_with_selections(Some(1), &config).await??;
    let mut prefixes = Vec::new();
    for pattern in patterns {
        let mut retained = Vec::new();
        for &(_, position, _) in &pattern {
            retained.push(batch_prefix(&mut batched, position).await?);
        }
        prefixes.push(retained);
    }
    let prepared_before = batched.metrics().prepared_graphs;
    for ((pattern, retained), expected) in patterns.into_iter().zip(prefixes).zip(expected) {
        for lane in 0..8 {
            batched.prepare_paged_lane(lane, None, 0)?;
        }
        let mut lanes = vec![None; 8];
        for ((lane, position, token), prefix) in pattern.into_iter().zip(retained) {
            batched.prepare_paged_lane(u32::try_from(lane)?, Some(prefix), 64)?;
            lanes[lane] = Some((position, token));
        }
        let actual = batch_decode(&mut batched, 8, &lanes, SamplingParams::default()).await?;
        assert_eq!(
            pattern.map(|(lane, _, _)| actual[lane]),
            expected.as_slice()
        );
        assert!(
            actual
                .iter()
                .enumerate()
                .all(|(lane, &token)| lanes[lane].is_some() || token == 0)
        );
    }
    assert_eq!(batched.metrics().prepared_graphs, prepared_before + 2);
    Ok(())
}

#[cfg(target_os = "macos")]
async fn assert_batch_graph_modes() -> Result<(), Box<dyn Error>> {
    let mut runner = runner(test_guests::qwen3_coder()).await?;
    let mut config = load_config()?;
    config.paged_cache = true;
    runner.load_with_selections(Some(1), &config).await??;
    let prefixes = [
        batch_prefix(&mut runner, 1).await?,
        batch_prefix(&mut runner, 4).await?,
    ];
    for (lane, prefix) in prefixes.into_iter().enumerate() {
        runner.prepare_paged_lane(u32::try_from(lane)?, Some(prefix), 64)?;
    }
    let shape = BatchShape {
        phase: BatchPhase::Decode,
        bucket: 2,
        tokens: 1,
        selection: BatchSelection::Forward,
    };
    let lanes = [(1, 31), (4, 37)].map(|(start_pos, token)| {
        Some(EngineBatchLane {
            slot: start_pos / 4,
            tokens: Some(vec![token]),
            start_pos,
            sampling: SamplingParams::default(),
        })
    });
    let logits = runner
        .batch_forward(EngineBatch {
            shape,
            lanes: lanes.into(),
        })
        .await??;
    let sampling = SamplingParams {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.9,
        seed: 17,
    };
    let selected = runner
        .batch_select(
            &logits,
            EngineBatchSelect {
                shape: BatchShape {
                    selection: BatchSelection::Masked,
                    ..shape
                },
                lanes: [(1, 101), (4, 102)]
                    .map(|(start_pos, token)| {
                        Some(EngineBatchSelectLane {
                            start_pos,
                            sampling,
                            allow: vec![token],
                        })
                    })
                    .into(),
            },
        )
        .await??;
    assert_eq!(
        batch_tokens(&runner.read(&selected.tokens).await?)?,
        [101, 102]
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_batch_graph_catalog_stays_within_limit() -> Result<(), Box<dyn Error>> {
    let mut runner = runner(test_guests::qwen3_coder()).await?;
    let mut config = load_config()?;
    config.paged_cache = true;
    runner.load_with_selections(Some(1), &config).await??;

    populate_single_replay_catalog(&mut runner).await?;

    for bucket in [1, 2, 4, 8] {
        let prefix = batch_prefix(&mut runner, 1).await?;
        runner.prepare_paged_lane(0, Some(prefix), 64)?;
        let mut lanes = vec![None; usize::try_from(bucket)?];
        lanes[0] = Some((1, 31));
        batch_decode(&mut runner, bucket, &lanes, SamplingParams::default()).await?;
        lanes[0] = Some((2, 37));
        batch_decode(&mut runner, bucket, &lanes, SamplingParams::default()).await?;

        let prefix = batch_prefix(&mut runner, 1).await?;
        runner.prepare_paged_lane(0, Some(prefix), 64)?;
        for (position, token, allowed) in [(1, 41, 47), (2, 43, 53)] {
            let shape = BatchShape {
                phase: BatchPhase::Decode,
                bucket,
                tokens: 1,
                selection: BatchSelection::Forward,
            };
            let mut lanes = vec![None; usize::try_from(bucket)?];
            lanes[0] = Some(EngineBatchLane {
                slot: 0,
                tokens: Some(vec![token]),
                start_pos: position,
                sampling: SamplingParams::default(),
            });
            let logits = runner.batch_forward(EngineBatch { shape, lanes }).await??;
            let mut lanes = vec![None; usize::try_from(bucket)?];
            lanes[0] = Some(EngineBatchSelectLane {
                start_pos: position,
                sampling: SamplingParams::default(),
                allow: vec![allowed],
            });
            let selected = runner
                .batch_select(
                    &logits,
                    EngineBatchSelect {
                        shape: BatchShape {
                            selection: BatchSelection::Masked,
                            ..shape
                        },
                        lanes,
                    },
                )
                .await??;
            assert_eq!(
                batch_tokens(&runner.read(&selected.tokens).await?)?[0],
                allowed
            );
        }
    }

    assert_eq!(runner.metrics().live_graphs, 2 + 2 + 4 * 3);
    assert_eq!(runner.metrics().prepared_graphs, 2 + 2 + 4 * 3);
    Ok(())
}

#[cfg(target_os = "macos")]
async fn populate_single_replay_catalog(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
) -> Result<(), Box<dyn Error>> {
    let mut position = 0;
    for sequence in [9, 512, 2_048, 4_000] {
        let output = runner
            .step(EngineStep {
                tokens: vec![17; sequence],
                start_pos: position,
                taps: false,
            })
            .await??;
        runner.read(&output.logits).await?;
        position += u32::try_from(sequence)?;
    }
    assert_eq!(runner.metrics().live_graphs, 2);
    let output = runner
        .step(EngineStep {
            tokens: vec![17],
            start_pos: position,
            taps: false,
        })
        .await??;
    runner.read(&output.logits).await?;
    position += 1;
    assert_eq!(runner.metrics().live_graphs, 3);
    for sampling in [
        SamplingParams::default(),
        SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            seed: 17,
        },
    ] {
        let output = runner
            .decode(EngineDecode {
                tokens: None,
                start_pos: position,
                sampling,
            })
            .await??;
        runner.read(&output.token).await?;
        position += 1;
    }
    assert_eq!(runner.metrics().live_graphs, 3);
    let logits = runner
        .begin_decode(EngineBeginDecode {
            tokens: Some(vec![17]),
            start_pos: position,
        })
        .await??;
    let allow = vec![u32::MAX; usize::try_from(qwen3_coder::VOCAB / 32)?];
    let output = runner
        .finish_decode(&logits, position, SamplingParams::default(), &allow)
        .await??;
    runner.read_queued_token(output).await?;
    assert_eq!(runner.metrics().live_graphs, 2 + 2);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_first_paged_layer_matches_same_byte_reference() -> Result<(), Box<dyn Error>> {
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
    let mut config = load_config()?;
    config.paged_cache = true;
    runner.load_with_selections(Some(1), &config).await??;
    let output = runner
        .step(EngineStep {
            tokens,
            start_pos: 0,
            taps: true,
        })
        .await??;
    let actual = decode_f32_le(&runner.read(&output.router_logits[0]).await?)?;
    let expected = fixture.router_logits().ok_or("router fixture is missing")?;
    let reference = expected
        .values()
        .get(..actual.len())
        .ok_or("router fixture is incomplete")?;
    let error = normwise_relative_error(reference, &actual)?;
    assert!(
        error <= BF16_HIDDEN_STATE_TOLERANCE,
        "relative error {error}"
    );
    assert_selected_sets(
        reference,
        &actual,
        usize::try_from(qwen3_coder::EXPERTS)?,
        usize::try_from(qwen3_coder::TOP_K)?,
    )?;
    Ok(())
}

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
    let replay = decode_outputs(false, true, REPLAY_LAYERS).await?;
    let lazy = decode_outputs(false, false, REPLAY_LAYERS).await?;
    assert_eq!(replay, lazy);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_paged_replay_matches_lazy_decode_at_one_layer() -> Result<(), Box<dyn Error>> {
    assert_paged_replay_matches_lazy(1).await
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_paged_replay_matches_lazy_decode_at_all_layers() -> Result<(), Box<dyn Error>> {
    assert_paged_replay_matches_lazy(FULL_LAYERS).await
}

#[cfg(target_os = "macos")]
async fn assert_paged_replay_matches_lazy(layers: u32) -> Result<(), Box<dyn Error>> {
    let replay = decode_outputs(true, true, layers).await?;
    let lazy = decode_outputs(true, false, layers).await?;
    for (step, (replay, lazy)) in replay.iter().zip(&lazy).enumerate() {
        assert_eq!(replay.0, lazy.0, "token differs at step {step}");
        assert_eq!(replay.1, lazy.1, "logits differ at step {step}");
    }
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
async fn decode_outputs(
    paged_cache: bool,
    replay: bool,
    layers: u32,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>, Box<dyn Error>> {
    let mut runner = runner_with_graph_replay(forja_metal::MetalGraphReplay::Tier2).await?;
    let mut config = load_config()?;
    config.paged_cache = paged_cache;
    config.replay = replay;
    runner.load_with_selections(Some(layers), &config).await??;
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
    runner_with_limits(component, LIMITS).await
}

#[cfg(target_os = "macos")]
async fn runner_with_limits(
    component: &std::path::Path,
    limits: Limits,
) -> Result<EngineRunner<forja_metal::MetalBackend>, Box<dyn Error>> {
    Ok(EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        limits,
        model_root()?.join("Qwen3-Coder-30B-A3B-Instruct-4bit/model.safetensors.index.json"),
    )
    .await?)
}

#[cfg(target_os = "macos")]
async fn batch_prefix(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    length: u32,
) -> Result<PagedPrefix, Box<dyn Error>> {
    runner.prepare_paged_lane(0, None, 0)?;
    runner.prepare_paged_lane(0, None, 64)?;
    let tokens = (0..length).map(|position| 100 + position).collect();
    let output = runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: 16,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(tokens),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    runner.read(&output.tokens).await?;
    let prefix = runner
        .paged_prefix()?
        .ok_or_else(|| Box::<dyn Error>::from("paged prefix is unavailable"))?;
    runner.prepare_paged_lane(0, None, 0)?;
    Ok(prefix)
}

#[cfg(target_os = "macos")]
async fn batch_decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    bucket: u32,
    lanes: &[Option<(u32, u32)>],
    sampling: SamplingParams,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let output = runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                bucket,
                tokens: 1,
                selection: BatchSelection::Combined,
            },
            lanes: lanes
                .iter()
                .enumerate()
                .map(|(slot, lane)| {
                    lane.map(|(start_pos, token)| EngineBatchLane {
                        slot: u32::try_from(slot).unwrap(),
                        tokens: Some(vec![token]),
                        start_pos,
                        sampling,
                    })
                })
                .collect(),
        })
        .await??;
    batch_tokens(&runner.read(&output.tokens).await?)
}

#[cfg(target_os = "macos")]
fn batch_tokens(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("batch token output is not u32-aligned".into());
    }
    Ok(words.iter().copied().map(u32::from_le_bytes).collect())
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
