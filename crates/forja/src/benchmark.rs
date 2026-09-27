use std::{
    error::Error,
    fs,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};

use forja_host::{EngineMetrics, EngineRunner, EngineStep};
use golden_fixtures::sha256_file;

use crate::{
    args::Bench,
    benchmark_stats::{Stats, stats, synthetic_tokens},
    engine::limits,
};

const WARMUPS: usize = 3;
const DECODE_PREFILL: usize = 8;

#[derive(Clone, Copy)]
pub(crate) struct Summary {
    pub(crate) wall_tps: Stats,
    pub(crate) gpu_tps: Stats,
    pub(crate) wall_seconds: Stats,
    pub(crate) gpu_seconds: Stats,
    pub(crate) submissions: Stats,
}

#[derive(Clone, Copy)]
struct Sample {
    wall_seconds: f64,
    gpu_seconds: f64,
    submissions: u32,
}

pub(crate) async fn run(options: &Bench) -> Result<(), Box<dyn Error>> {
    #[cfg(target_os = "macos")]
    return run_metal(options).await;
    #[cfg(not(target_os = "macos"))]
    Err("engine benchmarks require macOS and Metal".into())
}

#[cfg(target_os = "macos")]
async fn run_metal(options: &Bench) -> Result<(), Box<dyn Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let commit = command_output(
        "git",
        &["-C", &repository.to_string_lossy(), "rev-parse", "HEAD"],
    )?;
    let os = command_output("sw_vers", &["-productVersion"])?;
    let mut results = Vec::new();
    println!(
        "precision\tmetric\twall tok/s (95% CI)\tGPU tok/s (95% CI)\twall ms\tGPU ms\tsubmissions"
    );
    for (precision, component) in [
        ("f32", test_guests::qwen3()),
        ("bf16", test_guests::qwen3_bf16()),
    ] {
        let (pp, tg, device) = bench_precision(options, component).await?;
        print_summary(precision, "pp", pp);
        print_summary(precision, "tg", tg);
        results.push(serde_json::json!({
            "provenance": {
                "git_commit": commit,
                "engine_component_sha256": sha256_file(component)?,
                "device": device,
                "os": format!("macOS {os}"),
                "precision": precision,
            },
            "prompt_processing": summary_json(pp),
            "token_generation": summary_json(tg),
        }));
    }
    if let Some(path) = &options.json {
        let report = serde_json::json!({
            "schema_version": 1,
            "implementation": "forja",
            "model": options.model_dir,
            "settings": {
                "prompt_tokens": options.pp,
                "generated_tokens": options.tg,
                "warmups": WARMUPS,
                "repetitions": options.reps,
            },
            "results": results,
        });
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        fs::write(path, bytes)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn bench_precision(
    options: &Bench,
    component: &Path,
) -> Result<(Summary, Summary, String), Box<dyn Error>> {
    let backend = forja_metal::MetalBackend::new()?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        component,
        backend,
        limits(),
        options.model_dir.join("model.safetensors"),
    )
    .await?;
    let info = runner.describe().await?;
    if options.pp > usize::try_from(info.max_context)?
        || DECODE_PREFILL
            .checked_add(1)
            .and_then(|length| length.checked_add(options.tg))
            .ok_or("decode context length overflowed")?
            > usize::try_from(info.max_context)?
    {
        return Err("benchmark shape exceeds the engine context".into());
    }
    runner
        .load()
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let tokens = synthetic_tokens(options.pp.max(DECODE_PREFILL), info.vocab);
    for _ in 0..WARMUPS {
        measure_prefill(&mut runner, &tokens[..options.pp]).await?;
        measure_decode(&mut runner, &tokens[..DECODE_PREFILL], options.tg).await?;
    }
    let mut pp = Vec::with_capacity(options.reps);
    let mut tg = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        pp.push(measure_prefill(&mut runner, &tokens[..options.pp]).await?);
        tg.push(measure_decode(&mut runner, &tokens[..DECODE_PREFILL], options.tg).await?);
    }
    Ok((
        summarize(&pp, options.pp)?,
        summarize(&tg, options.tg)?,
        device,
    ))
}

#[cfg(target_os = "macos")]
async fn measure_prefill(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
) -> Result<Sample, Box<dyn Error>> {
    let before = runner.metrics();
    let started = Instant::now();
    step(runner, tokens.to_vec(), 0).await?;
    sample(before, runner.metrics(), started.elapsed())
}

#[cfg(target_os = "macos")]
async fn measure_decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
) -> Result<Sample, Box<dyn Error>> {
    step(runner, prompt.to_vec(), 0).await?;
    let mut schedule = Vec::with_capacity(steps.saturating_add(1));
    visit_decode_steps(u32::try_from(prompt.len())?, steps, |position, timed| {
        schedule.push((position, timed));
    })?;
    let mut before = None;
    let mut started = None;
    for (position, timed) in schedule {
        step(runner, vec![prompt[0]], position).await?;
        if !timed {
            before = Some(runner.metrics());
            started = Some(Instant::now());
        }
    }
    sample(
        before.ok_or("decode timing did not start")?,
        runner.metrics(),
        started.ok_or("decode timing did not start")?.elapsed(),
    )
}

fn visit_decode_steps(
    context_start: u32,
    timed_steps: usize,
    mut visit: impl FnMut(u32, bool),
) -> Result<(), Box<dyn Error>> {
    visit(context_start, false);
    for index in 0..timed_steps {
        let position = context_start
            .checked_add(1)
            .and_then(|position| position.checked_add(u32::try_from(index).ok()?))
            .ok_or("decode position overflowed")?;
        visit(position, true);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: Vec<u32>,
    start_pos: u32,
) -> Result<(), Box<dyn Error>> {
    runner
        .step(EngineStep {
            tokens,
            start_pos,
            taps: false,
        })
        .await?
        .map_err(|error| format!("engine step failed: {error:?}"))?;
    Ok(())
}

fn sample(
    before: EngineMetrics,
    after: EngineMetrics,
    wall_time: Duration,
) -> Result<Sample, Box<dyn Error>> {
    let submissions = after
        .submissions
        .checked_sub(before.submissions)
        .ok_or("submission counter moved backwards")?;
    let timed = after
        .timed_submissions
        .checked_sub(before.timed_submissions)
        .ok_or("timed submission counter moved backwards")?;
    if timed != submissions {
        return Err("a Metal submission did not report GPU time".into());
    }
    Ok(Sample {
        wall_seconds: wall_time.as_secs_f64(),
        gpu_seconds: after
            .gpu_time
            .checked_sub(before.gpu_time)
            .ok_or("GPU time counter moved backwards")?
            .as_secs_f64(),
        submissions: u32::try_from(submissions)?,
    })
}

fn summarize(samples: &[Sample], tokens: usize) -> Result<Summary, Box<dyn Error>> {
    if samples.is_empty() || samples.iter().any(|sample| sample.gpu_seconds == 0.0) {
        return Err("benchmark produced no timed samples".into());
    }
    let tokens = f64::from(u32::try_from(tokens)?);
    Ok(Summary {
        wall_tps: stats(samples.iter().map(|sample| tokens / sample.wall_seconds)),
        gpu_tps: stats(samples.iter().map(|sample| tokens / sample.gpu_seconds)),
        wall_seconds: stats(samples.iter().map(|sample| sample.wall_seconds)),
        gpu_seconds: stats(samples.iter().map(|sample| sample.gpu_seconds)),
        submissions: stats(samples.iter().map(|sample| f64::from(sample.submissions))),
    })
}

fn print_summary(precision: &str, metric: &str, summary: Summary) {
    println!(
        "{precision}\t{metric}\t{:.2} ({:.2}–{:.2})\t{:.2} ({:.2}–{:.2})\t{:.3}\t{:.3}\t{:.0}",
        summary.wall_tps.median,
        summary.wall_tps.low,
        summary.wall_tps.high,
        summary.gpu_tps.median,
        summary.gpu_tps.low,
        summary.gpu_tps.high,
        summary.wall_seconds.median * 1_000.0,
        summary.gpu_seconds.median * 1_000.0,
        summary.submissions.median,
    );
}
fn summary_json(summary: Summary) -> serde_json::Value {
    serde_json::json!({
        "tokens_per_second": {
            "wall": stats_json(summary.wall_tps),
            "gpu": stats_json(summary.gpu_tps),
        },
        "wall_time_seconds": stats_json(summary.wall_seconds),
        "gpu_time_seconds": stats_json(summary.gpu_seconds),
        "submissions": stats_json(summary.submissions),
    })
}

fn stats_json(stats: Stats) -> serde_json::Value {
    serde_json::json!({
        "median": stats.median,
        "ci95": [stats.low, stats.high],
    })
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} failed with {}", output.status).into());
    }
    let value = String::from_utf8(output.stdout)?.trim().to_owned();
    if value.is_empty() {
        return Err(format!("{program} returned no output").into());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_timing_excludes_one_step_after_prefill() -> Result<(), Box<dyn Error>> {
        let mut visited = Vec::new();
        visit_decode_steps(8, 7, |position, timed| visited.push((position, timed)))?;
        assert_eq!(visited.first(), Some(&(8, false)));
        assert_eq!(visited.iter().filter(|(_, timed)| *timed).count(), 7);
        assert_eq!(visited.last(), Some(&(15, true)));
        Ok(())
    }
}
