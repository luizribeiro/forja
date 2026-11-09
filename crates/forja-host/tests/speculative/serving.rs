use forja_cpu::CpuBackend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineLoadConfig,
    EngineRunner, SamplingParams,
    speculative::{VerifyAcceptance, VerifyBudget, select_verify_shape},
};

async fn fixture() -> Result<EngineRunner<CpuBackend>, Box<dyn std::error::Error>> {
    let mut runner = crate::common::verify_runner().await?;
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
    Ok(runner)
}

#[tokio::test]
async fn verify_batches_publish_each_prefix_before_host_staged_plain_feedback()
-> Result<(), Box<dyn std::error::Error>> {
    for rows in [2_u32, 3] {
        for accepted in 0..rows {
            let mut runner = fixture().await?;
            runner.prepare_paged_lane(1, None, 16)?;
            assert!(runner.try_reserve_draft_lane(1)?);
            let shape = BatchShape {
                phase: BatchPhase::Verify,
                bucket: 1,
                tokens: rows,
                selection: BatchSelection::Combined,
            };
            let mut drafts = vec![2, 3];
            drafts.truncate(usize::try_from(rows - 1)?);
            if accepted < rows - 1 {
                drafts[usize::try_from(accepted)?] = 1;
            }
            let fed = select_verify_shape(
                &[shape],
                &drafts,
                VerifyBudget::new(0, 2, 7, 32, 16).ok_or("budget missing")?,
            )
            .ok_or("shape missing")?;
            let mut tokens = vec![1];
            tokens.extend_from_slice(fed.tokens());
            let output = runner
                .enqueue_batch_step(EngineBatch {
                    shape,
                    lanes: vec![Some(EngineBatchLane {
                        slot: 1,
                        tokens: Some(tokens),
                        start_pos: 0,
                        sampling: SamplingParams::default(),
                    })],
                })
                .await??;
            let bytes = runner.read(&output.tokens).await?;
            let samples: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect();
            let decision = VerifyAcceptance::new(fed, &samples, 4)?
                .scan(|_| Ok::<_, std::convert::Infallible>(None))?;
            assert_eq!(decision.accepted(), accepted);
            let pending = decision.pending();
            runner.publish_verified_batch(output, &decision)?;
            assert_eq!(
                runner
                    .paged_lane_prefix(1)?
                    .ok_or("prefix missing")?
                    .logical_len()?,
                u64::from(accepted) + 1
            );

            for step in 1..=2 {
                let output = runner
                    .enqueue_batch_step(EngineBatch {
                        shape: BatchShape {
                            phase: BatchPhase::Decode,
                            bucket: 2,
                            tokens: 1,
                            selection: BatchSelection::Combined,
                        },
                        lanes: vec![
                            None,
                            Some(EngineBatchLane {
                                slot: 1,
                                tokens: (step == 1).then(|| vec![pending]),
                                start_pos: accepted + step,
                                sampling: SamplingParams::default(),
                            }),
                        ],
                    })
                    .await??;
                let bytes = runner.read_queued_batch_tokens(output).await?;
                assert_eq!(
                    u32::from_le_bytes(bytes[4..8].try_into()?),
                    (pending + step) % 4
                );
            }
            assert!(runner.try_release_draft_lane(1)?);
        }
    }
    Ok(())
}
