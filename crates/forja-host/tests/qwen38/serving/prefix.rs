use super::{Error, runner_with_context};
use forja_core::{Backend, CanonicalPrefixSchedule};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineRunner,
    HybridPrefixStore, SamplingParams, SnapshotAllowance,
};
use std::{num::NonZeroU32, sync::atomic::AtomicBool};

enum Evaluation {
    Replay,
    SavePrompt,
    SaveContext,
}

async fn evaluate<B: Backend + Send + Sync + 'static>(
    runner: &mut EngineRunner<B>,
    cache: &HybridPrefixStore<B>,
    tokens: &[u32],
    reuse: bool,
    lane: u32,
    mode: Evaluation,
) -> Result<(Vec<u8>, [u8; 32], u64), Box<dyn Error>> {
    let namespace = cache.namespace();
    let schedule = CanonicalPrefixSchedule::new(namespace.block(), NonZeroU32::new(4096).unwrap())?;
    let allowance = SnapshotAllowance::new(2 << 30, None)?;
    let hit = if reuse {
        cache.lookup(namespace, tokens)?
    } else {
        None
    };
    let complete = matches!(mode, Evaluation::SaveContext);
    let prompt = if complete {
        runner.canonical_context(namespace, tokens.to_vec())?
    } else {
        runner.canonical_prompt(namespace, tokens.to_vec())?
    };
    runner.prepare_paged_lane(lane, None, 0)?;
    let mut position = runner
        .prepare_canonical_lane(lane, prompt, hit.as_ref(), u64::try_from(tokens.len())? + 1)
        .map_err(|error| format!("admit len={} reuse={reuse}: {error}", tokens.len()))?;
    let cached = position;
    let mut logits = Vec::new();
    while position < tokens.len() as u64 {
        let count = if complete {
            u32::try_from((tokens.len() as u64 - position).min(u64::from(namespace.block().get())))?
        } else {
            schedule.limit(
                position,
                tokens.len() as u64,
                (tokens.len() as u64 - position).min(2048) as u32,
            )?
        };
        let count = runner.prompt_piece_limit(u32::try_from(position)?, count)?;
        let start = usize::try_from(position)?;
        let end = start + usize::try_from(count)?;
        let output = runner
            .batch_step(EngineBatch {
                shape: BatchShape {
                    phase: BatchPhase::Prompt,
                    bucket: 1,
                    tokens: [64, 512, 2048]
                        .into_iter()
                        .find(|&size| size >= count)
                        .unwrap(),
                    selection: BatchSelection::Combined,
                },
                lanes: vec![Some(EngineBatchLane {
                    slot: lane,
                    tokens: Some(tokens[start..end].to_vec()),
                    start_pos: u32::try_from(position)?,
                    sampling: SamplingParams::default(),
                })],
            })
            .await?
            .map_err(|error| format!("piece len={} start={position}: {error}", tokens.len()))?;
        logits = runner.read(&output.logits).await?;
        runner.read_queued_batch_tokens(output).await?;
        position += u64::from(count);
        let boundary = if complete {
            position == tokens.len() as u64
        } else {
            schedule.contains(position, tokens.len() as u64)
        };
        if !matches!(mode, Evaluation::Replay) && boundary {
            let entry = runner
                .capture_canonical_prefix(lane, allowance)
                .map_err(|error| format!("capture len={} end={position}: {error}", tokens.len()))?;
            cache.publish(entry, &AtomicBool::new(false))?;
        }
    }
    let (position, digest) = runner.state_lane_fingerprint(lane)?;
    assert_eq!(position.tokens, tokens.len() as u64);
    Ok((logits, digest, cached))
}

pub(crate) async fn assert_boundaries<B: Backend + Send + Sync + 'static>(
    mut runner: EngineRunner<B>,
    lengths: &[u32],
) -> Result<(), Box<dyn Error>> {
    let namespace = runner.canonical_prefix_namespace([7; 32], &["test-fixed"])?;
    for &length in lengths {
        let cache = runner
            .create_canonical_prefix_store(namespace, SnapshotAllowance::new(2 << 30, None)?)?;
        let tokens = (0..length).map(|i| i % 33).collect::<Vec<_>>();
        let cold = evaluate(
            &mut runner,
            &cache,
            &tokens,
            false,
            0,
            Evaluation::SavePrompt,
        )
        .await?;
        assert_eq!(
            cache.usage()?.entries,
            if length <= 64 {
                0
            } else if length == 4161 {
                2
            } else {
                1
            }
        );
        let hit = evaluate(&mut runner, &cache, &tokens, true, 0, Evaluation::Replay).await?;
        assert_eq!(cold.0, hit.0, "logits at {length}");
        assert_eq!(cold.1, hit.1, "state at {length}");
        assert_eq!(hit.2, u64::from((length - 1) / 64 * 64));
    }
    Ok(())
}

#[tokio::test]
async fn canonical_prefix_replays_tail_with_identical_logits_and_state()
-> Result<(), Box<dyn Error>> {
    let runner = runner_with_context(forja_cpu::CpuBackend::new(), false, 8192).await?;
    assert_boundaries(runner, &[63, 64, 65, 4096, 4097, 4161]).await
}

pub(crate) async fn assert_shared_parent<B: Backend + Send + Sync + 'static>(
    mut runner: EngineRunner<B>,
    parent_tokens: u32,
) -> Result<(), Box<dyn Error>> {
    let namespace = runner.canonical_prefix_namespace([8; 32], &["test-fixed"])?;
    let cache =
        runner.create_canonical_prefix_store(namespace, SnapshotAllowance::new(2 << 30, None)?)?;
    let parent = (0..parent_tokens).map(|i| i % 33).collect::<Vec<_>>();
    let original = evaluate(
        &mut runner,
        &cache,
        &parent,
        false,
        0,
        Evaluation::SavePrompt,
    )
    .await?;
    let mut children = Vec::new();
    for lane in [1, 2] {
        let tokens = (0..parent_tokens + 64)
            .map(|i| (i + if i >= parent_tokens - 1 { lane } else { 0 }) % 33)
            .collect::<Vec<_>>();
        let cold = evaluate(&mut runner, &cache, &tokens, false, 0, Evaluation::Replay).await?;
        let hit = evaluate(&mut runner, &cache, &tokens, true, lane, Evaluation::Replay).await?;
        assert_eq!(hit.2, u64::from(parent_tokens - 1));
        assert_eq!(cold.0, hit.0);
        assert_eq!(cold.1, hit.1);
        children.push((lane, hit.1));
    }
    for (lane, digest) in children {
        assert_eq!(runner.state_lane_fingerprint(lane)?.1, digest);
    }
    let again = evaluate(&mut runner, &cache, &parent, true, 0, Evaluation::Replay).await?;
    assert_eq!(again.2, u64::from(parent_tokens - 1));
    assert_eq!(original.0, again.0);
    assert_eq!(original.1, again.1);
    Ok(())
}

#[tokio::test]
async fn shared_canonical_parent_survives_two_live_branch_writers() -> Result<(), Box<dyn Error>> {
    let runner = runner_with_context(forja_cpu::CpuBackend::new(), true, 8192).await?;
    assert_shared_parent(runner, 65).await
}

#[tokio::test]
async fn canonical_branches_replay_across_the_sparse_snapshot_interval()
-> Result<(), Box<dyn Error>> {
    let runner = runner_with_context(forja_cpu::CpuBackend::new(), true, 8192).await?;
    assert_shared_parent(runner, 4097).await
}

pub(crate) async fn assert_recomputed_reply<B: Backend + Send + Sync + 'static>(
    mut runner: EngineRunner<B>,
) -> Result<(), Box<dyn Error>> {
    let namespace = runner.canonical_prefix_namespace([9; 32], &["test-fixed"])?;
    let allowance = SnapshotAllowance::new(2 << 30, None)?;
    let cache = runner.create_canonical_prefix_store(namespace, allowance)?;
    let tokens = (0..129).map(|i| i % 33).collect::<Vec<_>>();
    evaluate(
        &mut runner,
        &cache,
        &tokens[..127],
        false,
        0,
        Evaluation::SavePrompt,
    )
    .await?;
    let cold = evaluate(&mut runner, &cache, &tokens, false, 1, Evaluation::Replay).await?;
    let generated = runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                bucket: 1,
                tokens: 1,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(vec![tokens[127]]),
                start_pos: 127,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    runner.read_queued_batch_tokens(generated).await?;
    assert!(runner.capture_canonical_prefix(0, allowance).is_err());
    let recomputed = evaluate(
        &mut runner,
        &cache,
        &tokens[..128],
        true,
        0,
        Evaluation::SaveContext,
    )
    .await?;
    assert_eq!(recomputed.2, 64);
    let hit = evaluate(&mut runner, &cache, &tokens, true, 0, Evaluation::Replay).await?;
    assert_eq!(hit.2, 128);
    assert_eq!(cold.0, hit.0);
    assert_eq!(cold.1, hit.1);
    Ok(())
}

#[tokio::test]
async fn recomputed_reply_preserves_cold_logits_and_every_state_plane() -> Result<(), Box<dyn Error>>
{
    assert_recomputed_reply(runner_with_context(forja_cpu::CpuBackend::new(), true, 512).await?)
        .await
}
