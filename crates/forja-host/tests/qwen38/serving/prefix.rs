use super::{Error, runner_with_context};
use forja_core::{Backend, CanonicalPrefixSchedule};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineRunner,
    HybridPrefixStore, SamplingParams, SnapshotAllowance,
};
use std::{num::NonZeroU32, sync::atomic::AtomicBool};

async fn evaluate<B: Backend + Send + Sync + 'static>(
    runner: &mut EngineRunner<B>,
    cache: &HybridPrefixStore<B>,
    tokens: &[u32],
    reuse: bool,
) -> Result<(Vec<u8>, [u8; 32], u64), Box<dyn Error>> {
    let namespace = cache.namespace();
    let schedule = CanonicalPrefixSchedule::new(namespace.block(), NonZeroU32::new(4096).unwrap())?;
    let allowance = SnapshotAllowance::new(1 << 28, None)?;
    let hit = if reuse {
        cache.lookup(namespace, tokens)?
    } else {
        None
    };
    let prompt = runner.canonical_prompt(namespace, tokens.to_vec())?;
    runner.prepare_paged_lane(0, None, 0)?;
    let mut position = runner.prepare_canonical_lane(0, prompt, hit.as_ref(), 8192)?;
    let cached = position;
    let mut logits = Vec::new();
    while position < tokens.len() as u64 {
        let count = schedule.limit(
            position,
            tokens.len() as u64,
            (tokens.len() as u64 - position).min(2048) as u32,
        )?;
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
                    slot: 0,
                    tokens: Some(tokens[start..end].to_vec()),
                    start_pos: u32::try_from(position)?,
                    sampling: SamplingParams::default(),
                })],
            })
            .await??;
        logits = runner.read(&output.logits).await?;
        runner.read(&output.tokens).await?;
        position += u64::from(count);
        if schedule.contains(position, tokens.len() as u64) {
            let entry = runner.capture_canonical_prefix(0, allowance)?;
            cache.publish(entry, &AtomicBool::new(false))?;
        }
    }
    let (position, digest) = runner.state_lane_fingerprint(0)?;
    assert_eq!(position.tokens, tokens.len() as u64);
    Ok((logits, digest, cached))
}

#[tokio::test]
async fn canonical_prefix_replays_tail_with_identical_logits_and_state()
-> Result<(), Box<dyn Error>> {
    let mut runner = runner_with_context(forja_cpu::CpuBackend::new(), false, 8192).await?;
    let namespace = runner.canonical_prefix_namespace([7; 32], &["test-fixed"])?;
    let cache =
        runner.create_canonical_prefix_store(namespace, SnapshotAllowance::new(1 << 28, None)?)?;
    let tokens = (0..65).map(|i| i % 33).collect::<Vec<_>>();
    let cold = evaluate(&mut runner, &cache, &tokens, false).await?;
    let hit = evaluate(&mut runner, &cache, &tokens, true).await?;
    assert_eq!(cold.0, hit.0);
    assert_eq!(cold.1, hit.1);
    assert_eq!(hit.2, 64);
    Ok(())
}
