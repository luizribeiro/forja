use super::{
    DecodeOptions, EngineInfo, EngineRunner, Error, Instant, Profile, Sample, TokenProfile, sample,
};
use forja_host::{
    BatchPhase, EngineBatch, EngineBatchLane,
    speculative::{VerifyAcceptance, VerifyBudget, select_verify_shape},
};

pub(super) async fn measure(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    info: &EngineInfo,
    tokens: &[u32],
    options: &Profile,
    decode: DecodeOptions,
    device: String,
) -> Result<TokenProfile, Box<dyn Error>> {
    let rows = u32::try_from(options.step_tokens.ok_or("missing verify rows")?)?;
    let shapes = &info
        .serving
        .as_ref()
        .ok_or("verify requires serving")?
        .batch_shapes;
    let shape = *shapes
        .iter()
        .find(|shape| shape.phase == BatchPhase::Verify && shape.tokens == rows)
        .ok_or("undeclared verify shape; check dispatches_per_list")?;
    for _ in 0..options.warmups {
        phase(runner, tokens, options, decode, shape, info.vocab, false).await?;
    }
    let baseline = phase(runner, tokens, options, decode, shape, info.vocab, false).await?;
    let profiled = phase(runner, tokens, options, decode, shape, info.vocab, true).await?;
    Ok(TokenProfile {
        device,
        baseline_wall: std::time::Duration::from_secs_f64(baseline.wall_seconds),
        baseline_gpu: std::time::Duration::from_secs_f64(baseline.gpu_seconds),
        profiled_wall: std::time::Duration::from_secs_f64(profiled.wall_seconds),
        profiled_gpu: std::time::Duration::from_secs_f64(profiled.gpu_seconds),
        cache_setup_wall: std::time::Duration::ZERO,
        step: runner
            .take_profile()
            .ok_or("verify profile has no timing detail")?,
    })
}

async fn phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    options: &Profile,
    decode: DecodeOptions,
    shape: forja_host::BatchShape,
    vocab: u32,
    profiling: bool,
) -> Result<Sample, Box<dyn Error>> {
    runner.set_profiling(false);
    let token = super::prepare_profile_context(
        runner,
        tokens,
        options.context,
        decode,
        usize::try_from(shape.tokens)?,
    )
    .await?;
    if !runner.try_reserve_draft_lane(0)? {
        return Err("verify draft storage is unavailable".into());
    }
    let position = u32::try_from(options.context)?;
    let end = u64::from(position)
        .checked_add(u64::from(shape.tokens))
        .ok_or("verify extent overflow")?;
    let proposals = vec![0; usize::try_from(shape.tokens - 1)?];
    let budget = VerifyBudget::new(
        u64::from(position),
        shape.tokens - 1,
        u64::from(shape.tokens),
        end,
        end,
    )
    .ok_or("verify budget is empty")?;
    let fed =
        select_verify_shape(&[shape], &proposals, budget).ok_or("verify shape does not fit")?;
    let mut input = vec![token];
    input.extend_from_slice(fed.tokens());
    runner.set_profiling(profiling);
    let before = runner.metrics();
    let started = Instant::now();
    let output = runner
        .enqueue_batch_step(EngineBatch {
            shape,
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(input),
                start_pos: position,
                sampling: decode.sampling,
            })],
        })
        .await?
        .map_err(|e| format!("profile verify failed: {e:?}"))?;
    let bytes = runner.read(&output.tokens).await?;
    let measured = sample(before, runner.metrics(), started.elapsed())?;
    runner.set_profiling(false);
    let samples = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(u32::from_le_bytes)
        .collect::<Vec<_>>();
    let decision = VerifyAcceptance::new(fed, &samples, vocab)?
        .scan(|_| Ok::<_, std::convert::Infallible>(None))?;
    runner.publish_verified_batch(output, &decision)?;
    runner.wait_for_queued_submissions().await?;
    if !runner.try_release_draft_lane(0)? {
        return Err("verify storage remains retained after completion".into());
    }
    Ok(measured)
}
