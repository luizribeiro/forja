use super::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineRunner, Error,
    PagedPrefix, SamplingParams,
};

type Runner = EngineRunner<forja_metal::MetalBackend>;

#[derive(Clone)]
pub(super) enum Prefix {
    Paged(PagedPrefix),
    Cold(Vec<u32>, BatchShape),
}

impl Prefix {
    pub(super) async fn prepare(
        runner: &mut Runner,
        serving: &forja_host::ServingInfo,
        tokens: Vec<u32>,
        reserved: u64,
    ) -> Result<Self, Box<dyn Error>> {
        let shape = serving
            .batch_shapes
            .iter()
            .filter(|shape| {
                shape.phase == BatchPhase::Prompt
                    && shape.bucket == 1
                    && shape.selection == BatchSelection::Combined
                    && shape.tokens <= 512
            })
            .max_by_key(|shape| shape.tokens)
            .copied()
            .ok_or("batch benchmark has no bounded prompt shape")?;
        if !runner.supports_prefix_reuse() {
            return Ok(Self::Cold(tokens, shape));
        }
        prefill(runner, 0, &tokens, shape, reserved).await?;
        Ok(Self::Paged(
            runner
                .paged_prefix()?
                .ok_or("batch benchmark did not produce a paged prefix")?,
        ))
    }

    pub(super) async fn restore(
        &self,
        runner: &mut Runner,
        lane: u32,
        tokens: u64,
        reserved: u64,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Paged(prefix) => {
                let prefix = prefix.fork_reserved(tokens, reserved)?;
                runner.prepare_paged_lane(lane, None, 0)?;
                runner.prepare_paged_lane(lane, Some(prefix), reserved)?;
                Ok(())
            }
            Self::Cold(prompt, shape) => prefill(runner, lane, prompt, *shape, reserved).await,
        }
    }
}

async fn prefill(
    runner: &mut Runner,
    lane: u32,
    tokens: &[u32],
    shape: BatchShape,
    reserved: u64,
) -> Result<(), Box<dyn Error>> {
    runner.prepare_paged_lane(lane, None, 0)?;
    runner.prepare_paged_lane(lane, None, reserved)?;
    for (index, chunk) in tokens.chunks(usize::try_from(shape.tokens)?).enumerate() {
        let start = u32::try_from(index)?
            .checked_mul(shape.tokens)
            .ok_or("batch prompt offset overflow")?;
        let output = runner
            .batch_step(EngineBatch {
                shape,
                lanes: vec![Some(EngineBatchLane {
                    slot: lane,
                    tokens: Some(chunk.to_vec()),
                    start_pos: start,
                    sampling: SamplingParams::default(),
                })],
            })
            .await?
            .map_err(|error| format!("batch benchmark prefix failed: {error:?}"))?;
        runner.read(&output.tokens).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_batch_prefix_rebuilds_carry_for_every_lane_and_repetition()
    -> Result<(), Box<dyn Error>> {
        let models = std::env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?;
        let component = test_guests::qwen38();
        let crate::args::Command::Bench(options) = crate::args::parse([
            "--isolated".to_owned(),
            "bench".to_owned(),
            "--engine".to_owned(),
            component.display().to_string(),
            "--model".to_owned(),
            std::path::Path::new(&models)
                .join("Qwen3.8-27B-4bit")
                .display()
                .to_string(),
        ])?
        else {
            return Err("expected benchmark options".into());
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(async {
                let (mut runner, info, _) =
                    super::super::load_benchmark_runner(&options, &options.points[0], component)
                        .await?;
                let serving = info.serving.as_ref().ok_or("missing serving shapes")?;
                let tokens = super::super::synthetic_tokens(33, info.vocab);
                let prefix = Prefix::prepare(&mut runner, serving, tokens, 36).await?;
                assert!(matches!(prefix, Prefix::Cold(..)));
                let mut expected = None;
                for _ in 0..2 {
                    for lane in 0..2 {
                        prefix.restore(&mut runner, lane, 33, 36).await?;
                    }
                    let output = runner
                        .batch_step(EngineBatch {
                            shape: BatchShape {
                                phase: BatchPhase::Decode,
                                bucket: 2,
                                tokens: 1,
                                selection: BatchSelection::Combined,
                            },
                            lanes: (0..2)
                                .map(|slot| {
                                    Some(EngineBatchLane {
                                        slot,
                                        tokens: Some(vec![42]),
                                        start_pos: 33,
                                        sampling: SamplingParams::default(),
                                    })
                                })
                                .collect(),
                        })
                        .await?
                        .map_err(|error| format!("batch prefix decode failed: {error:?}"))?;
                    let actual = runner.read(&output.logits).await?;
                    let (first, second) = actual.split_at(actual.len() / 2);
                    assert!(
                        first == second,
                        "identical prompt lanes have different logits"
                    );
                    if let Some(expected) = &expected {
                        assert!(
                            actual == *expected,
                            "restored carry differs across repetitions"
                        );
                    }
                    expected = Some(actual);
                }
                super::super::prepare_benchmark_runner(&mut runner, 36)?;
                super::super::prepare_benchmark_runner(&mut runner, 36)?;
                Ok::<_, Box<dyn Error>>(())
            })
    }
}
