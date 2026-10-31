use super::{Error, runner};
use forja_cpu::CpuBackend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchOutput,
    EngineDecode, EngineRunner, SamplingParams,
};

struct Row {
    logits: Vec<u8>,
    token: Vec<u8>,
}
fn prompt(lane: u32) -> Vec<u32> {
    (0..7).map(|position| (position + lane) % 33).collect()
}
fn sampling(lane: u32, stochastic: bool) -> SamplingParams {
    if stochastic {
        SamplingParams {
            temperature: 0.7,
            top_k: 7,
            top_p: 0.9,
            seed: 0x1_0000_0000 + u64::from(lane),
        }
    } else {
        SamplingParams::default()
    }
}

async fn single(
    runner: &mut EngineRunner<CpuBackend>,
    lane: u32,
    stochastic: bool,
) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for step in 0..4 {
        let output = runner
            .decode(EngineDecode {
                phase: if step == 0 {
                    BatchPhase::Prompt
                } else {
                    BatchPhase::Decode
                },
                tokens: (step == 0).then(|| prompt(lane)),
                start_pos: if step == 0 { 0 } else { 6 + step },
                sampling: sampling(lane, stochastic),
            })
            .await??;
        rows.push(Row {
            logits: runner.read(&output.logits).await?,
            token: runner.read(&output.token).await?,
        });
    }
    Ok(rows)
}

async fn assert_rows(
    runner: &mut EngineRunner<CpuBackend>,
    output: &EngineBatchOutput,
    expected: &[Vec<Row>],
    step: usize,
) -> Result<(), Box<dyn Error>> {
    let logits = runner.read(&output.logits).await?;
    let tokens = runner.read(&output.tokens).await?;
    for (lane, expected) in expected.iter().enumerate() {
        assert_eq!(&logits[lane * 132..(lane + 1) * 132], expected[step].logits);
        assert_eq!(&tokens[lane * 4..(lane + 1) * 4], expected[step].token);
    }
    Ok(())
}

async fn batch(bucket: u32, stochastic: bool, expected: &[Vec<Row>]) -> Result<(), Box<dyn Error>> {
    let mut runner = runner(CpuBackend::new(), false).await?;
    for lane in 0..bucket {
        runner.prepare_paged_lane(lane, None, 64)?;
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
                    sampling: sampling(lane, stochastic),
                })],
            })
            .await??;
        assert_rows(
            &mut runner,
            &output,
            std::slice::from_ref(&expected[usize::try_from(lane)?]),
            0,
        )
        .await?;
    }
    for step in 1..4 {
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
                            sampling: sampling(lane, stochastic),
                        })
                    })
                    .collect(),
            })
            .await??;
        assert_rows(&mut runner, &output, expected, usize::try_from(step)?).await?;
    }
    Ok(())
}

#[tokio::test]
async fn decode_buckets_match_single_lane_tokens_and_logits() -> Result<(), Box<dyn Error>> {
    for stochastic in [false, true] {
        let mut reference = runner(CpuBackend::new(), false).await?;
        let mut expected = Vec::new();
        for lane in 0..8 {
            expected.push(single(&mut reference, lane, stochastic).await?);
        }
        for bucket in [1, 2, 4, 8] {
            batch(bucket, stochastic, &expected[..usize::try_from(bucket)?]).await?;
        }
    }
    Ok(())
}
