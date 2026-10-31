use super::{Error, Runner, load_with};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineDecode,
    SamplingParams,
};

struct Row {
    logits: Vec<u8>,
    token: Vec<u8>,
}
fn prompt(lane: u32) -> Vec<u32> {
    (0..7).map(|position| 100 + position + lane).collect()
}
fn sampling(lane: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_k: 7,
        top_p: 0.9,
        seed: 0x1_0000_0000 + u64::from(lane),
    }
}
async fn singles(runner: &mut Runner, bucket: u32) -> Result<Vec<Vec<Row>>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for lane in 0..bucket {
        let mut trace = Vec::new();
        for step in 0..3 {
            let output = runner
                .decode(EngineDecode {
                    phase: if step == 0 {
                        BatchPhase::Prompt
                    } else {
                        BatchPhase::Decode
                    },
                    tokens: (step == 0).then(|| prompt(lane)),
                    start_pos: if step == 0 { 0 } else { 6 + step },
                    sampling: sampling(lane),
                })
                .await??;
            trace.push(Row {
                logits: runner.read(&output.logits).await?,
                token: runner.read(&output.token).await?,
            });
        }
        rows.push(trace);
    }
    Ok(rows)
}
async fn compare(bucket: u32) -> Result<(), Box<dyn Error>> {
    let (mut reference, _) = load_with(false, Some(64)).await?;
    let expected = singles(&mut reference, bucket).await?;
    drop(reference);
    let (mut runner, _) = load_with(true, Some(64)).await?;
    assert_eq!(runner.metrics().live_graphs, 10);
    for lane in 0..bucket {
        runner
            .prepare_paged_lane(lane, None, 0)
            .map_err(|e| format!("release lane {lane}: {e:?}"))?;
        runner
            .prepare_paged_lane(lane, None, 64)
            .map_err(|e| format!("reserve lane {lane}: {e:?}"))?;
        let output = runner
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
                    sampling: sampling(lane),
                })],
            })
            .await
            .map_err(|error| format!("prompt outer lane={lane}: {error:?}"))?
            .map_err(|error| format!("prompt lane={lane}: {error:?}"))?;
        let row = &expected[usize::try_from(lane)?][0];
        assert_eq!(runner.read(&output.logits).await?, row.logits);
        assert_eq!(runner.read(&output.tokens).await?, row.token);
    }
    for step in 1..3 {
        let output = runner
            .batch_step(EngineBatch {
                shape: BatchShape {
                    phase: BatchPhase::Decode,
                    bucket,
                    tokens: 1,
                    selection: BatchSelection::Combined,
                },
                lanes: (0..bucket)
                    .map(|lane| {
                        Some(EngineBatchLane {
                            slot: lane,
                            tokens: None,
                            start_pos: 6 + step,
                            sampling: sampling(lane),
                        })
                    })
                    .collect(),
            })
            .await?
            .map_err(|error| format!("decode bucket={bucket} step={step}: {error:?}"))?;
        let logits = runner.read(&output.logits).await?;
        let tokens = runner.read(&output.tokens).await?;
        for (lane, trace) in expected.iter().enumerate() {
            let row = &trace[usize::try_from(step)?];
            assert_eq!(
                &logits[lane * row.logits.len()..(lane + 1) * row.logits.len()],
                row.logits
            );
            assert_eq!(&tokens[lane * 4..(lane + 1) * 4], row.token);
        }
    }
    assert_eq!(runner.metrics().live_graphs, 11);
    Ok(())
}
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_replayed_buckets_match_independent_single_lanes() -> Result<(), Box<dyn Error>>
{
    for bucket in [1, 2, 4] {
        compare(bucket).await?;
    }
    Ok(())
}
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_eager_bucket_eight_matches_independent_single_lanes()
-> Result<(), Box<dyn Error>> {
    compare(8).await
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_full_catalog_replays_under_the_live_quota() -> Result<(), Box<dyn Error>> {
    let (mut runner, _) = load_with(true, None).await?;
    assert_eq!(runner.metrics().live_graphs, 12);
    let mut expected = None;
    for _ in 0..2 {
        let mut rows = Vec::new();
        for step in 0..2 {
            let output = runner
                .decode(EngineDecode {
                    phase: if step == 0 {
                        BatchPhase::Prompt
                    } else {
                        BatchPhase::Decode
                    },
                    tokens: (step == 0).then(|| prompt(0)),
                    start_pos: if step == 0 { 0 } else { 7 },
                    sampling: SamplingParams::default(),
                })
                .await??;
            rows.push((
                runner.read(&output.logits).await?,
                runner.read(&output.token).await?,
            ));
        }
        if let Some(expected) = &expected {
            assert_eq!(&rows, expected);
        } else {
            expected = Some(rows);
        }
        assert_eq!(runner.metrics().live_graphs, 13);
        assert!(runner.metrics().live_bytes <= 48 * 1024 * 1024 * 1024);
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_long_prompt_replays_within_production_submission_limits()
-> Result<(), Box<dyn Error>> {
    let (mut runner, _) = load_with(true, None).await?;
    for start in [0, 2048] {
        runner
            .step(forja_host::EngineStep {
                phase: BatchPhase::Prompt,
                tokens: (0..2048).map(|row| 100 + row % 7).collect(),
                start_pos: start,
                taps: false,
            })
            .await??;
    }
    for (phase, tokens, start_pos) in [
        (BatchPhase::Prompt, Some(vec![100]), 4096),
        (BatchPhase::Decode, None, 4097),
    ] {
        let output = runner
            .decode(EngineDecode {
                phase,
                tokens,
                start_pos,
                sampling: SamplingParams::default(),
            })
            .await??;
        assert_eq!(runner.read(&output.token).await?.len(), 4);
        assert!(
            runner
                .read(&output.logits)
                .await?
                .as_chunks::<4>()
                .0
                .iter()
                .all(|bytes| f32::from_le_bytes(*bytes).is_finite())
        );
    }
    assert_eq!(
        runner
            .checkpoint_state_lane(0)?
            .ok_or("missing long prompt state")?
            .position()
            .tokens,
        4098
    );
    assert_eq!(runner.metrics().live_graphs, 13);
    Ok(())
}
