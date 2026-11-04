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
    let shapes = &info
        .serving
        .as_ref()
        .ok_or("batch profile requires serving")?
        .batch_shapes;
    let [prompt, decode] = select_shapes(shapes, options.batch_size)?;
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

fn select_shapes(shapes: &[BatchShape], bucket: u32) -> Result<[BatchShape; 2], &'static str> {
    let shape = |phase, bucket| {
        shapes
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
    let decode = shape(BatchPhase::Decode, bucket).ok_or("undeclared decode bucket")?;
    Ok([prompt, decode])
}

fn prompt_offset(index: usize, tokens: u32) -> Result<u32, Box<dyn Error>> {
    u32::try_from(index)?
        .checked_mul(tokens)
        .ok_or_else(|| "prompt offset overflow".into())
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
            let start_pos = prompt_offset(index, prompt.tokens)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_require_combined_selection_and_an_exact_decode_bucket() {
        let prompt = BatchShape {
            phase: BatchPhase::Prompt,
            bucket: 1,
            tokens: 64,
            selection: BatchSelection::Combined,
        };
        let decode = BatchShape {
            phase: BatchPhase::Decode,
            bucket: 4,
            tokens: 1,
            selection: BatchSelection::Combined,
        };
        let shapes = [
            prompt,
            BatchShape {
                tokens: 512,
                ..prompt
            },
            decode,
        ];
        assert_eq!(select_shapes(&shapes, 4), Ok([shapes[1], decode]));
        assert!(select_shapes(&shapes, 2).is_err());
        assert!(select_shapes(&[decode], 4).is_err());
        assert!(select_shapes(&[prompt], 4).is_err());
        assert!(
            select_shapes(
                &[
                    prompt,
                    BatchShape {
                        selection: BatchSelection::Forward,
                        ..decode
                    }
                ],
                4
            )
            .is_err()
        );
    }

    #[test]
    fn prompt_chunk_offsets_keep_the_partial_tail_and_refuse_overflow() {
        for length in [1_usize, 7, 33, 65, 4097] {
            let tokens = vec![0; length];
            let mut covered = 0;
            for (index, chunk) in tokens.chunks(64).enumerate() {
                assert_eq!(prompt_offset(index, 64).unwrap() as usize, covered);
                covered += chunk.len();
            }
            assert_eq!(covered, length);
        }
        assert!(prompt_offset(usize::MAX, 64).is_err());
        assert!(prompt_offset(2, u32::MAX).is_err());
    }

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_batch_profile_records_one_decode_with_all_lanes() -> Result<(), Box<dyn Error>> {
        let models = std::env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?;
        let model = std::path::Path::new(&models).join("Qwen3.8-27B-4bit");
        let component = test_guests::qwen38();
        let crate::args::Command::Profile(mut options) = crate::args::parse([
            "--isolated".to_owned(),
            "profile".to_owned(),
            "--engine".to_owned(),
            component.display().to_string(),
            "--model".to_owned(),
            model.display().to_string(),
            "--context".to_owned(),
            "34".to_owned(),
            "--batch-size".to_owned(),
            "4".to_owned(),
        ])?
        else {
            return Err("expected profile options".into());
        };
        options.warmups = 0;
        let measurement = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(super::super::measure_token_profile(&options))?;
        assert!(measurement.baseline_gpu > Duration::ZERO);
        assert!(measurement.profiled_gpu > Duration::ZERO);
        assert_eq!(measurement.step.submissions.len(), 1);
        assert!(measurement.step.submissions[0].dispatches > 0);
        Ok(())
    }
}
