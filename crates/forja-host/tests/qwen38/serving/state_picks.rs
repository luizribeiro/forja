use super::{Error, fixture};
use forja_core::{Op, StateOp};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineDecode,
    EngineLoadConfig, EngineRunner, FixedVariantPick, SamplingParams,
};

#[derive(Debug, Eq, PartialEq)]
struct Row {
    logits: Vec<u8>,
    tokens: Vec<u8>,
}

async fn runner(picked: bool) -> Result<EngineRunner<forja_metal::MetalBackend>, Box<dyn Error>> {
    let mut runner = EngineRunner::new(
        test_guests::qwen38_tiny_metal(),
        forja_metal::MetalBackend::new()?,
        forja_host::Limits::new(1 << 30, 8, 1 << 30, 20_000, 1 << 27),
        fixture("model.safetensors"),
    )
    .await?;
    let mut picks = [
        ("quant", "quant-matmul.portable"),
        ("quant-decode", "quant-matmul.portable"),
        ("quant-decode-batch", "quant-matmul.portable"),
        ("quant-decode-batch-wide", "quant-matmul.portable"),
        ("attention", "sdpa.decomposed"),
        ("sample", "sample.parallel"),
        ("delta-prefill", "delta-rule-block.wy64-fixed"),
        ("paged-attention.prefill", "paged-sdpa.decomposed"),
        ("paged-attention.decode", "paged-sdpa.decomposed"),
    ]
    .to_vec();
    if picked {
        picks.extend([
            ("delta-decode", "delta-rule.simd-linear-state-lanes"),
            ("conv-decode", "causal-depthwise-conv.direct-lanes"),
        ]);
    }
    runner
        .load_with_selections(
            // The fixture's attention width has no paged Metal variant.
            Some(3),
            &EngineLoadConfig {
                max_context: Some(64),
                replay: false,
                paged_cache: true,
                kv_cache: forja_host::KvCacheFormat::F16,
                fixed_variant_picks: picks
                    .into_iter()
                    .map(|(site, name)| FixedVariantPick {
                        site: site.into(),
                        name: name.into(),
                    })
                    .collect(),
                ..Default::default()
            },
        )
        .await??;
    Ok(runner)
}

async fn trace(
    bucket: u32,
    picked: bool,
    batch: bool,
    first: u32,
) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut runner = runner(picked).await?;
    for lane in 0..bucket {
        runner.prepare_paged_lane(lane, None, 64)?;
    }
    runner.set_profiling(true);
    let mut rows = Vec::new();
    for step in 0..3 {
        let tokens = (0..bucket)
            .map(|row| (first + row + step * 3) % 33)
            .collect::<Vec<_>>();
        let (logits, selected) = if batch {
            let output = runner
                .batch_step(EngineBatch {
                    shape: BatchShape {
                        phase: BatchPhase::Decode,
                        bucket,
                        tokens: 1,
                        selection: BatchSelection::Combined,
                    },
                    lanes: tokens
                        .into_iter()
                        .enumerate()
                        .map(|(lane, token)| {
                            Ok(Some(EngineBatchLane {
                                slot: u32::try_from(lane)?,
                                tokens: Some(vec![token]),
                                start_pos: step,
                                sampling: SamplingParams::default(),
                            }))
                        })
                        .collect::<Result<_, Box<dyn Error>>>()?,
                })
                .await??;
            (output.logits, output.tokens)
        } else {
            let output = runner
                .decode(EngineDecode {
                    phase: BatchPhase::Decode,
                    tokens: Some(tokens),
                    start_pos: step,
                    sampling: SamplingParams::default(),
                })
                .await??;
            (output.logits, output.token)
        };
        rows.push(Row {
            logits: runner.read(&logits).await?,
            tokens: runner.read(&selected).await?,
        });
        let profile = runner.take_profile().ok_or("missing decode profile")?;
        assert_state_picks(&profile, picked);
    }
    Ok(rows)
}

#[tokio::test]
async fn metal_single_lane_state_picks_preserve_previous_decode_bits() -> Result<(), Box<dyn Error>>
{
    for batch in [false, true] {
        assert_eq!(
            trace(1, false, batch, 7).await?,
            trace(1, true, batch, 7).await?
        );
    }
    Ok(())
}

fn assert_state_picks(profile: &forja_host::EngineStepProfile, picked: bool) {
    let dispatches = profile
        .submissions
        .iter()
        .flat_map(|submission| &submission.per_dispatch)
        .collect::<Vec<_>>();
    if picked {
        for (operation, name) in [
            (true, "delta-rule.simd-linear-state-lanes"),
            (false, "causal-depthwise-conv.direct-lanes"),
        ] {
            let states = dispatches
                .iter()
                .filter(|d| {
                    matches!(d.op, Op::State(StateOp::DeltaRule(_))) == operation
                        && matches!(d.op, Op::State(_))
                })
                .collect::<Vec<_>>();
            assert_ne!(states.len(), 0);
            assert!(states.iter().all(|d| d.variant == Some(name)));
        }
    }
}
