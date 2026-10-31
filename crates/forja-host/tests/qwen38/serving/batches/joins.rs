use super::*;

async fn admit(runner: &mut EngineRunner<CpuBackend>, lane: u32) -> Result<(), Box<dyn Error>> {
    runner.prepare_paged_lane(lane, None, 0)?;
    runner.prepare_paged_lane(lane, None, 64)?;
    runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: 64,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: lane,
                tokens: Some(prompt(lane)),
                start_pos: 0,
                sampling: sampling(lane, true),
            })],
        })
        .await??;
    Ok(())
}
async fn advance(
    runner: &mut EngineRunner<CpuBackend>,
    bucket: u32,
    positions: &[(u32, u32)],
) -> Result<EngineBatchOutput, Box<dyn Error>> {
    let mut lanes = vec![None; usize::try_from(bucket)?];
    for &(lane, position) in positions {
        lanes[usize::try_from(lane)?] = Some(EngineBatchLane {
            slot: lane,
            tokens: None,
            start_pos: position,
            sampling: sampling(lane, true),
        });
    }
    Ok(runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                bucket,
                tokens: 1,
                selection: BatchSelection::Combined,
            },
            lanes,
        })
        .await??)
}
async fn membership_case(bucket: u32, replay: bool) -> Result<(), Box<dyn Error>> {
    let mut reference = runner(CpuBackend::new(), false).await?;
    let last = bucket - 1;
    let first_trace = single(&mut reference, 0, true).await?;
    let last_trace = single(&mut reference, last, true).await?;
    let mut actual = runner(CpuBackend::new(), replay).await?;
    admit(&mut actual, 0).await?;
    admit(&mut actual, last).await?;
    let position = actual
        .checkpoint_state_lane(0)?
        .ok_or("missing carry")?
        .position();
    assert!(matches!(
        actual.prepare_paged_lane(0, None, u64::MAX),
        Err(forja_host::bindings::l9o::gpu::compute::Error::Quota(_))
    ));
    assert_eq!(
        actual
            .checkpoint_state_lane(0)?
            .ok_or("missing carry")?
            .position(),
        position
    );
    advance(&mut actual, bucket, &[(0, 7), (last, 7)]).await?;
    actual.prepare_paged_lane(last, None, 0)?;
    let output = advance(&mut actual, bucket, &[(0, 8)]).await?;
    assert_eq!(
        &actual.read(&output.logits).await?[..132],
        first_trace[2].logits
    );
    admit(&mut actual, last).await?;
    let output = advance(&mut actual, bucket, &[(0, 9), (last, 7)]).await?;
    let logits = actual.read(&output.logits).await?;
    let tokens = actual.read(&output.tokens).await?;
    for (lane, row) in [
        (0, &first_trace[3]),
        (usize::try_from(last)?, &last_trace[1]),
    ] {
        assert_eq!(&logits[lane * 132..(lane + 1) * 132], row.logits);
        assert_eq!(&tokens[lane * 4..(lane + 1) * 4], row.token);
    }
    Ok(())
}
#[tokio::test]
async fn leaving_and_rejoining_lanes_preserve_peers_after_quota_refusal()
-> Result<(), Box<dyn Error>> {
    for bucket in [2, 4, 8] {
        for replay in [false, true] {
            membership_case(bucket, replay).await?;
        }
    }
    Ok(())
}
