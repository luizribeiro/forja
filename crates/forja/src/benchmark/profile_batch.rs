use super::{Profile, Sample, TokenProfile, sample};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineInfo, EngineRunner,
    SamplingParams,
};
use std::{
    error::Error,
    time::{Duration, Instant},
};

pub(super) async fn measure(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    info: &EngineInfo,
    tokens: &[u32],
    options: &Profile,
    device: String,
) -> Result<TokenProfile, Box<dyn Error>> {
    let serving = info
        .serving
        .as_ref()
        .ok_or("batch profile requires serving")?;
    let shape = |phase, bucket| {
        serving
            .batch_shapes
            .iter()
            .filter(|shape| {
                shape.phase == phase
                    && shape.bucket == bucket
                    && shape.selection == BatchSelection::Combined
            })
            .max_by_key(|shape| shape.tokens)
            .copied()
    };
    let prompt = shape(BatchPhase::Prompt, 1).ok_or("no prompt shape")?;
    let decode = shape(BatchPhase::Decode, options.batch_size).ok_or("undeclared decode bucket")?;
    for _ in 0..options.warmups {
        phase(runner, tokens, options, [prompt, decode], false).await?;
    }
    let baseline = phase(runner, tokens, options, [prompt, decode], false).await?;
    let profiled = phase(runner, tokens, options, [prompt, decode], true).await?;
    Ok(TokenProfile {
        device,
        baseline_wall: Duration::from_secs_f64(baseline.wall_seconds),
        baseline_gpu: Duration::from_secs_f64(baseline.gpu_seconds),
        profiled_wall: Duration::from_secs_f64(profiled.wall_seconds),
        profiled_gpu: Duration::from_secs_f64(profiled.gpu_seconds),
        cache_setup_wall: Duration::ZERO,
        step: runner
            .take_profile()
            .ok_or("batch profile has no timing detail")?,
    })
}

async fn phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    options: &Profile,
    shapes: [BatchShape; 2],
    profiling: bool,
) -> Result<Sample, Box<dyn Error>> {
    let [prompt, decode] = shapes;
    runner.set_profiling(false);
    let reserved = u64::try_from(options.context)?
        .checked_add(1)
        .ok_or("reservation overflow")?;
    for lane in 0..options.batch_size {
        runner.prepare_paged_lane(lane, None, 0)?;
        runner.prepare_paged_lane(lane, None, reserved)?;
        for (index, chunk) in tokens.chunks(usize::try_from(prompt.tokens)?).enumerate() {
            let start_pos = u32::try_from(index)?
                .checked_mul(prompt.tokens)
                .ok_or("prompt offset overflow")?;
            let output = runner
                .batch_step(EngineBatch {
                    shape: prompt,
                    lanes: vec![Some(EngineBatchLane {
                        slot: lane,
                        tokens: Some(chunk.to_vec()),
                        start_pos,
                        sampling: SamplingParams::default(),
                    })],
                })
                .await?
                .map_err(|e| format!("profile prompt failed: {e:?}"))?;
            runner.read(&output.tokens).await?;
        }
    }
    let before = runner.metrics();
    runner.set_profiling(profiling);
    let started = Instant::now();
    let output = runner
        .batch_step(EngineBatch {
            shape: decode,
            lanes: (0..options.batch_size)
                .map(|slot| {
                    Ok(Some(EngineBatchLane {
                        slot,
                        tokens: None,
                        start_pos: u32::try_from(tokens.len())?,
                        sampling: SamplingParams::default(),
                    }))
                })
                .collect::<Result<_, std::num::TryFromIntError>>()?,
        })
        .await?;
    runner.set_profiling(false);
    let output = output.map_err(|e| format!("profile decode failed: {e:?}"))?;
    runner.read(&output.tokens).await?;
    sample(before, runner.metrics(), started.elapsed())
}
