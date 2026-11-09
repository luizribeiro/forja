use super::{Error, runner};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineDecode,
    SamplingParams,
    speculative::{VerifyAcceptance, VerifyBudget, select_verify_shape},
};

#[tokio::test]
async fn verify_rows_match_decode_logits_samples_and_each_state_plane() -> Result<(), Box<dyn Error>>
{
    for replay in [false, true] {
        let mut runner = runner(forja_cpu::CpuBackend::new(), replay).await?;
        let sampling = SamplingParams {
            temperature: 0.7,
            seed: 33,
            ..Default::default()
        };
        let mut expected = Vec::new();
        for (start_pos, token) in [7, 8, 3].into_iter().enumerate() {
            let output = runner
                .decode(EngineDecode {
                    phase: BatchPhase::Decode,
                    tokens: Some(vec![token]),
                    start_pos: u32::try_from(start_pos)?,
                    sampling,
                })
                .await??;
            expected.push((
                runner.read(&output.logits).await?,
                runner.read(&output.token).await?,
                runner.state_lane_fingerprint(0)?.1,
            ));
        }
        for rows in 2..=qwen38::Qwen38::VERIFY_ROWS {
            runner.prepare_paged_lane(7, None, 0)?;
            runner.prepare_paged_lane(7, None, 64)?;
            assert!(runner.try_reserve_draft_lane(7)?);
            let shape = BatchShape {
                phase: BatchPhase::Verify,
                bucket: 1,
                tokens: rows,
                selection: BatchSelection::Combined,
            };
            let drafts = [8, 3];
            let fed = select_verify_shape(
                &[shape],
                &drafts,
                VerifyBudget::new(0, rows - 1, 7, 64, 64).unwrap(),
            )
            .unwrap();
            let output = runner
                .enqueue_batch_step(EngineBatch {
                    shape,
                    lanes: vec![Some(EngineBatchLane {
                        slot: 7,
                        tokens: Some([7, 8, 3][..usize::try_from(rows)?].to_vec()),
                        start_pos: 0,
                        sampling,
                    })],
                })
                .await??;
            let logits = runner.read(&output.logits).await?;
            let selected = runner.read(&output.tokens).await?;
            let samples = selected
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect::<Vec<_>>();
            for row in 0..usize::try_from(rows)? {
                assert_eq!(&logits[row * 132..(row + 1) * 132], expected[row].0);
                assert_eq!(&selected[row * 4..(row + 1) * 4], expected[row].1);
            }
            let decision = VerifyAcceptance::new(fed, &samples, 33)?
                .scan(|_| Ok::<_, std::convert::Infallible>(None))?;
            let accepted = decision.accepted();
            runner.publish_verified_batch(output, &decision)?;
            assert_eq!(
                runner.state_lane_fingerprint(7)?.1,
                expected[usize::try_from(accepted)?].2
            );
            assert_eq!(
                runner.paged_lane_prefix(7)?.unwrap().logical_len()?,
                u64::from(accepted) + 1
            );
            assert!(runner.try_release_draft_lane(7)?);
        }
    }
    Ok(())
}

#[tokio::test]
async fn recording_quota_controls_verify_declarations_and_plain_decode_fallback()
-> Result<(), Box<dyn Error>> {
    for limit in [4096, 8191, 8192] {
        let enabled = limit >= 8192;
        let mut runner = forja_host::EngineRunner::new(
            test_guests::qwen38_tiny(),
            forja_cpu::CpuBackend::new(),
            forja_host::Limits::new(1 << 30, 8, 1 << 30, 20_000, 1 << 27)
                .with_command_limits(limit, forja_core::DEFAULT_DISPATCH_WORK_LIMIT),
            crate::qwen38::fixture("model.safetensors"),
        )
        .await?;
        let declared = runner.describe().await?.serving.unwrap().batch_shapes;
        assert_eq!(
            declared
                .iter()
                .filter(|shape| shape.phase == BatchPhase::Verify)
                .map(|shape| shape.tokens)
                .collect::<Vec<_>>(),
            if enabled { vec![2, 3] } else { vec![] }
        );
        runner
            .load_with_selections(
                None,
                &forja_host::EngineLoadConfig {
                    max_context: Some(64),
                    paged_cache: true,
                    kv_cache: forja_host::KvCacheFormat::F16,
                    ..Default::default()
                },
            )
            .await??;
        assert_eq!(runner.metrics().live_graphs, if enabled { 12 } else { 10 });
        let output = runner
            .decode(EngineDecode {
                phase: BatchPhase::Prompt,
                tokens: Some(vec![1, 7, 32]),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })
            .await??;
        let token = u32::from_le_bytes(runner.read(&output.token).await?.try_into().unwrap());
        if !enabled {
            assert!(
                runner
                    .enqueue_batch_step(EngineBatch {
                        shape: BatchShape {
                            phase: BatchPhase::Verify,
                            bucket: 1,
                            tokens: 3,
                            selection: BatchSelection::Combined
                        },
                        lanes: vec![Some(EngineBatchLane {
                            slot: 0,
                            tokens: Some(vec![token; 3]),
                            start_pos: 3,
                            sampling: SamplingParams::default()
                        })],
                    })
                    .await?
                    .is_err()
            );
            assert_eq!(
                runner.checkpoint_state_lane(0)?.unwrap().position().tokens,
                3
            );
        }
        let output = runner
            .decode(EngineDecode {
                phase: BatchPhase::Decode,
                tokens: Some(vec![token]),
                start_pos: 3,
                sampling: SamplingParams::default(),
            })
            .await??;
        assert_eq!(runner.read(&output.token).await?.len(), 4);
    }
    Ok(())
}
