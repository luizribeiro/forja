use crate::common::verify_runner as runner;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineLaneCheckpoint,
    EngineLoadConfig, SamplingParams,
};

#[tokio::test]
async fn verify_declarations_refuse_unpaged_loading_before_guest_execution() -> wasmtime::Result<()>
{
    let mut runner = runner().await?;
    assert!(!runner.try_reserve_draft_lane(0)?);
    assert!(runner.try_release_draft_lane(0)?);
    let error = runner.load().await?.unwrap_err();
    assert!(
        matches!(error, forja_host::bindings::l9o::gpu::compute::Error::Layout(message) if message == "verify shapes require paged serving storage")
    );
    Ok(())
}

#[tokio::test]
async fn combined_batch_entry_reserves_verify_rows_until_host_acceptance() -> wasmtime::Result<()> {
    let mut runner = runner().await?;
    runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                paged_cache: true,
                max_context: Some(32),
                ..Default::default()
            },
        )
        .await??;
    runner.prepare_paged_lane(0, None, 16)?;
    assert!(runner.try_reserve_draft_lane(0)?);
    let mut checkpoint =
        EngineLaneCheckpoint::new(0, 1, (), 16, runner.checkpoint_paged_lane(0)?.unwrap());
    checkpoint.state_version = runner.checkpoint_state_lane(0)?;
    let output = runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Verify,
                bucket: 1,
                tokens: 3,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(vec![1, 2, 3]),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    let samples = runner.read_queued_batch_tokens(output).await?;
    assert_eq!(
        samples
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(u32::from_le_bytes)
            .collect::<Vec<_>>(),
        [2, 3, 0]
    );
    assert!(runner.checkpoint_state_lane(0).is_err());
    assert_eq!(runner.paged_lane_prefix(0)?.unwrap().logical_len()?, 3);
    runner.rollback_lane(0, checkpoint)?;
    assert_eq!(
        runner.checkpoint_state_lane(0)?.unwrap().position().tokens,
        0
    );
    assert_eq!(runner.paged_lane_prefix(0)?.unwrap().logical_len()?, 0);
    assert!(runner.try_release_draft_lane(0)?);
    Ok(())
}
