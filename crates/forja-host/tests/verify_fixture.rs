use crate::common::verify_runner;
use forja_host::{
    BatchPhase, BatchSelection, EngineBatch, EngineBatchLane, EngineLoadConfig, SamplingParams,
};

#[tokio::test]
async fn protected_sampling_fixture_declares_verify_rows_and_runs_plain_input()
-> wasmtime::Result<()> {
    let mut runner = verify_runner().await?;
    let info = runner.describe().await?;
    let shapes = &info.serving.as_ref().unwrap().batch_shapes;
    assert_eq!(
        shapes
            .iter()
            .filter(|shape| shape.phase == BatchPhase::Verify)
            .map(|shape| shape.tokens)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    let prompt = *shapes
        .iter()
        .find(|shape| {
            shape.phase == BatchPhase::Prompt && shape.selection == BatchSelection::Combined
        })
        .unwrap();
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
    runner.prepare_paged_lane(0, None, 8)?;
    let output = runner
        .batch_step(EngineBatch {
            shape: prompt,
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(vec![1; 7]),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    assert_eq!(
        runner.read_queued_batch_tokens(output).await?,
        2_u32.to_le_bytes()
    );
    assert_eq!(runner.paged_lane_prefix(0)?.unwrap().logical_len()?, 7);
    assert_eq!(
        runner.checkpoint_state_lane(0)?.unwrap().position().tokens,
        7
    );
    Ok(())
}
