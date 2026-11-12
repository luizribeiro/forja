use super::{DecodeOptions, EngineRunner, Error, Instant, Profile, Sample, sample};
use crate::args::ProfileSelection;
use forja_host::{BatchPhase, EngineBeginDecode, PackedMask};

pub(super) async fn decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    token: u32,
    options: &Profile,
    decode: DecodeOptions,
    profiling: bool,
) -> Result<Sample, Box<dyn Error>> {
    if options.selection == ProfileSelection::Combined {
        return super::measure_decode_step(runner, token, options.context, decode).await;
    }
    let vocab = runner.describe().await?.vocab;
    let words = vec![u32::MAX; usize::try_from(vocab.div_ceil(32))?];
    let position = u32::try_from(options.context)?;
    let input = EngineBeginDecode {
        phase: BatchPhase::Decode,
        tokens: Some(vec![token]),
        start_pos: position,
    };
    runner.set_profiling(profiling && options.selection == ProfileSelection::Forward);
    let before = runner.metrics();
    let started = Instant::now();
    let logits = runner
        .begin_decode(input)
        .await?
        .map_err(|e| format!("profile forward failed: {e:?}"))?;
    runner.read(&logits).await?;
    let forward = sample(before, runner.metrics(), started.elapsed())?;
    runner.set_profiling(profiling && options.selection == ProfileSelection::Masked);
    let before = runner.metrics();
    let started = Instant::now();
    let output = runner
        .finish_decode(&logits, position, decode.sampling, PackedMask::new(&words))
        .await?
        .map_err(|e| format!("profile selection failed: {e:?}"))?;
    runner.read(&output.token).await?;
    let masked = sample(before, runner.metrics(), started.elapsed())?;
    runner.set_profiling(false);
    Ok(if options.selection == ProfileSelection::Forward {
        forward
    } else {
        masked
    })
}

pub(super) async fn batch(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    options: &Profile,
    mut shape: forja_host::BatchShape,
    profiling: bool,
) -> Result<Sample, Box<dyn Error>> {
    use forja_host::{
        BatchSelection, EngineBatch, EngineBatchLane, EngineBatchSelect, EngineBatchSelectLane,
        SamplingParams, TokenIds,
    };
    let vocab = runner.describe().await?.vocab;
    let position = u32::try_from(
        options
            .context
            .checked_sub(1)
            .ok_or("empty profile context")?,
    )?;
    shape.selection = BatchSelection::Forward;
    let input = EngineBatch {
        shape,
        lanes: (0..options.batch_size)
            .map(|slot| {
                Some(EngineBatchLane {
                    slot,
                    tokens: None,
                    start_pos: position,
                    sampling: SamplingParams::default(),
                })
            })
            .collect(),
    };
    runner.set_profiling(profiling && options.selection == ProfileSelection::Forward);
    let before = runner.metrics();
    let started = Instant::now();
    let logits = runner
        .batch_forward(input)
        .await?
        .map_err(|e| format!("profile batch forward failed: {e:?}"))?;
    runner.read(&logits).await?;
    let forward = sample(before, runner.metrics(), started.elapsed())?;
    shape.selection = BatchSelection::Masked;
    let input = EngineBatchSelect {
        shape,
        lanes: (0..options.batch_size)
            .map(|_| {
                Some(EngineBatchSelectLane {
                    start_pos: position,
                    sampling: SamplingParams::default(),
                    allow: TokenIds::new((0..vocab).collect()),
                })
            })
            .collect(),
    };
    runner.set_profiling(profiling && options.selection == ProfileSelection::Masked);
    let before = runner.metrics();
    let started = Instant::now();
    let output = runner
        .batch_select(&logits, input)
        .await?
        .map_err(|e| format!("profile batch selection failed: {e:?}"))?;
    runner.read(&output.tokens).await?;
    let masked = sample(before, runner.metrics(), started.elapsed())?;
    runner.set_profiling(false);
    Ok(if options.selection == ProfileSelection::Forward {
        forward
    } else {
        masked
    })
}
