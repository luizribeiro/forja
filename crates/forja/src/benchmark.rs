use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fs,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};

use forja_config::{GraphReplay, Selection};
use forja_core::Op;
use forja_host::{
    EngineDecode, EngineMetrics, EngineOutput, EngineRunner, EngineStep, EngineStepProfile,
    ImportProfile,
};
use golden_fixtures::{decode_f32_le, sha256_file};

use crate::{
    args::Bench,
    benchmark_stats::{Stats, stats, synthetic_tokens},
    engine::{argmax, limits, read_token},
};

#[cfg(target_os = "macos")]
use crate::engine::metal_graph_replay;

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

struct ProfileMeasurement {
    context_start: usize,
    baseline: Vec<Sample>,
    steps: Vec<EngineStepProfile>,
}

struct ProfileReport {
    context_start: usize,
    baseline: Summary,
    categories: Vec<ProfileCategory>,
    wall: Stats,
    wall_perturbation: Stats,
    gpu_perturbation: Stats,
    dispatch_coverage: Stats,
    gpu_by_op: Vec<ProfileCategory>,
    gpu_by_dispatch: Vec<DispatchProfile>,
}

struct DispatchProfile {
    index: usize,
    operation: &'static str,
    time: Stats,
}

pub(crate) async fn run(options: &Bench) -> Result<(), Box<dyn Error>> {
    #[cfg(target_os = "macos")]
    return run_metal(options).await;
    #[cfg(not(target_os = "macos"))]
    Err("engine benchmarks require macOS and Metal".into())
}

#[cfg(target_os = "macos")]
async fn run_metal(options: &Bench) -> Result<(), Box<dyn Error>> {
    let commit = crate::provenance::commit();
    let os = command_output("sw_vers", &["-productVersion"])?;
    let mut results = Vec::new();
    println!(
        "engine\tselection\tmetric\twall tok/s (95% CI)\tGPU tok/s (95% CI)\twall ms\tGPU ms\tsubmissions"
    );
    for component in &options.engines {
        let engine = component.display().to_string();
        for &strategy in &options.selection {
            let (host_argmax, overlap) = selection_mode(strategy);
            let selection = selection_name(strategy);
            let (pp, tg, device, profiles) =
                bench_engine(options, component, host_argmax, overlap).await?;
            print_summary(&engine, selection, "pp", pp);
            print_summary(&engine, selection, "tg", tg);
            let profile_reports = profiles
                .iter()
                .map(profile_report)
                .collect::<Result<Vec<_>, _>>()?;
            for report in &profile_reports {
                print_profile_report(&engine, selection, report);
            }
            let mut result = serde_json::json!({
                "provenance": {
                    "git_commit": commit,
                    "dirty": crate::provenance::dirty(),
                    "engine_component_sha256": sha256_file(component)?,
                    "device": device,
                    "os": format!("macOS {os}"),
                    "engine": component,
                    "graph_replay": graph_replay_name(options.graph_replay),
                    "selection": selection,
                },
                "prompt_processing": summary_json(pp),
                "token_generation": summary_json(tg),
            });
            if !profile_reports.is_empty() {
                result["profile"] =
                    serde_json::Value::Array(profile_reports.iter().map(profile_json).collect());
            }
            results.push(result);
        }
    }
    if let Some(path) = &options.json {
        let report = serde_json::json!({
            "schema_version": 1,
            "implementation": "forja",
            "model": options.model_dir,
            "settings": {
                "prompt_tokens": options.pp,
                "generated_tokens": options.tg,
                "warmups": options.warmups,
                "repetitions": options.reps,
                "graph_replay": graph_replay_name(options.graph_replay),
                "selection": options.selection.iter().copied().map(selection_name).collect::<Vec<_>>(),
            },
            "tg_context_start": options.decode_prefill + 1,
            "results": results,
        });
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        fs::write(path, bytes)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn bench_engine(
    options: &Bench,
    component: &Path,
    host_argmax: bool,
    overlap: bool,
) -> Result<(Summary, Summary, String, Vec<ProfileMeasurement>), Box<dyn Error>> {
    let backend =
        forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(options.graph_replay))?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        component,
        backend,
        limits(&options.limits)?,
        options.model_dir.join("model.safetensors"),
    )
    .await?;
    let info = runner.describe().await?;
    let max_context = usize::try_from(info.max_context)?;
    let profile_context = *options
        .contexts
        .last()
        .ok_or("no profile contexts configured")?;
    let tg_context_start = options
        .decode_prefill
        .checked_add(1)
        .ok_or("decode context length overflowed")?;
    if options.pp > max_context
        || tg_context_start
            .checked_add(options.tg)
            .ok_or("decode context length overflowed")?
            > max_context
        || options.breakdown && profile_context >= max_context
    {
        return Err("benchmark shape exceeds the engine context".into());
    }
    runner
        .load()
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let profile_tokens = if options.breakdown {
        profile_context - 1
    } else {
        0
    };
    let tokens = synthetic_tokens(
        options.pp.max(options.decode_prefill).max(profile_tokens),
        info.vocab,
    );
    for _ in 0..options.warmups {
        measure_prefill(&mut runner, &tokens[..options.pp]).await?;
        measure_decode(
            &mut runner,
            &tokens[..options.decode_prefill],
            options.tg,
            host_argmax,
            overlap,
        )
        .await?;
    }
    let mut pp = Vec::with_capacity(options.reps);
    let mut tg = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        pp.push(measure_prefill(&mut runner, &tokens[..options.pp]).await?);
        tg.push(
            measure_decode(
                &mut runner,
                &tokens[..options.decode_prefill],
                options.tg,
                host_argmax,
                overlap,
            )
            .await?,
        );
    }
    let profiles = if options.breakdown {
        measure_profiles(
            &mut runner,
            &tokens,
            options.reps,
            options.warmups,
            &options.contexts,
            host_argmax,
            overlap,
        )
        .await?
    } else {
        Vec::new()
    };
    Ok((
        summarize(&pp, options.pp)?,
        summarize(&tg, options.tg)?,
        device,
        profiles,
    ))
}

const fn graph_replay_name(strategy: GraphReplay) -> &'static str {
    match strategy {
        GraphReplay::Tier1 => "tier1",
        GraphReplay::Tier2 => "tier2",
    }
}

const fn selection_name(selection: Selection) -> &'static str {
    match selection {
        Selection::HostArgmax => "host-argmax",
        Selection::GpuSequential => "gpu-sequential",
        Selection::GpuPipelined => "gpu-pipelined",
    }
}

const fn selection_mode(selection: Selection) -> (bool, bool) {
    match selection {
        Selection::HostArgmax => (true, false),
        Selection::GpuSequential => (false, false),
        Selection::GpuPipelined => (false, true),
    }
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
    host_argmax: bool,
    overlap: bool,
) -> Result<Sample, Box<dyn Error>> {
    if overlap && !host_argmax {
        return measure_pipelined_decode(runner, prompt, steps).await;
    }
    let mut token = select_from_tokens(runner, prompt.to_vec(), 0, host_argmax).await?;
    let mut schedule = Vec::with_capacity(steps.saturating_add(1));
    visit_decode_steps(u32::try_from(prompt.len())?, steps, |position, timed| {
        schedule.push((position, timed));
    })?;
    let mut before = None;
    let mut started = None;
    for (position, timed) in schedule {
        let submission_before = runner.metrics().submissions;
        token = select_next(runner, token, position, host_argmax).await?;
        if !timed {
            runner
                .wait_for_submissions(
                    submission_before
                        .checked_add(1)
                        .ok_or("submission count overflowed")?,
                )
                .await?;
            before = Some(runner.metrics());
            started = Some(Instant::now());
        }
    }
    let before = before.ok_or("decode timing did not start")?;
    runner
        .wait_for_submissions(
            before
                .submissions
                .checked_add(u64::try_from(steps)?)
                .ok_or("submission count overflowed")?,
        )
        .await?;
    sample(
        before,
        runner.metrics(),
        started.ok_or("decode timing did not start")?.elapsed(),
    )
}

#[cfg(target_os = "macos")]
async fn measure_pipelined_decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
) -> Result<Sample, Box<dyn Error>> {
    let token = select_from_tokens(runner, prompt.to_vec(), 0, false).await?;
    let context_start = u32::try_from(prompt.len())?;
    let warmup_before = runner.metrics().submissions;
    select_next(runner, token, context_start, false).await?;
    runner
        .wait_for_submissions(
            warmup_before
                .checked_add(1)
                .ok_or("submission count overflowed")?,
        )
        .await?;
    let before = runner.metrics();
    let started = Instant::now();
    let mut position = context_start
        .checked_add(1)
        .ok_or("decode position overflowed")?;
    let end = position
        .checked_add(u32::try_from(steps)?)
        .ok_or("decode position overflowed")?;
    let mut outputs = VecDeque::with_capacity(2);
    let mut consumed = 0_u64;
    while position < end || !outputs.is_empty() {
        while position < end && outputs.len() < 2 {
            outputs.push_back(
                runner
                    .enqueue_decode(EngineDecode {
                        tokens: None,
                        start_pos: position,
                    })
                    .await?
                    .map_err(|error| format!("engine decode failed: {error:?}"))?,
            );
            position = position
                .checked_add(1)
                .ok_or("decode position overflowed")?;
        }
        let output = outputs.pop_front().ok_or("decode queue is empty")?;
        read_token(&runner.read_queued_token(output).await?)?;
        consumed = consumed
            .checked_add(1)
            .ok_or("submission count overflowed")?;
        runner
            .wait_for_submissions(
                before
                    .submissions
                    .checked_add(consumed)
                    .ok_or("submission count overflowed")?,
            )
            .await?;
    }
    let expected = before
        .submissions
        .checked_add(u64::try_from(steps)?)
        .ok_or("submission count overflowed")?;
    runner.wait_for_submissions(expected).await?;
    sample(before, runner.metrics(), started.elapsed())
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
async fn settle_submission_metrics(
    runner: &EngineRunner<forja_metal::MetalBackend>,
) -> Result<(), Box<dyn Error>> {
    runner
        .wait_for_submissions(runner.metrics().submissions)
        .await?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn measure_profiles(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    reps: usize,
    warmups: usize,
    contexts: &[usize],
    host_argmax: bool,
    overlap: bool,
) -> Result<Vec<ProfileMeasurement>, Box<dyn Error>> {
    let mut measurements = Vec::new();
    for &context_start in contexts {
        let mut baseline = Vec::with_capacity(reps);
        let mut steps = Vec::with_capacity(reps);
        for repetition in 0..warmups.saturating_add(reps) {
            let unprofiled = if overlap && !host_argmax {
                prepare_profile_context(runner, tokens, context_start, host_argmax).await?;
                measure_pipelined_profile_step(runner, context_start).await?
            } else {
                let token =
                    prepare_profile_context(runner, tokens, context_start, host_argmax).await?;
                measure_decode_step(runner, token, context_start, host_argmax).await?
            };
            let token = prepare_profile_context(runner, tokens, context_start, host_argmax).await?;
            runner.set_profiling(true);
            let profiled = measure_decode_step(runner, token, context_start, host_argmax).await;
            runner.set_profiling(false);
            let _ = profiled?;
            let profile = runner
                .take_profile()
                .ok_or("profiled step produced no timing detail")?;
            if repetition >= warmups {
                baseline.push(unprofiled);
                steps.push(profile);
            }
        }
        measurements.push(ProfileMeasurement {
            context_start,
            baseline,
            steps,
        });
    }
    Ok(measurements)
}

#[cfg(target_os = "macos")]
async fn measure_pipelined_profile_step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    context_start: usize,
) -> Result<Sample, Box<dyn Error>> {
    settle_submission_metrics(runner).await?;
    let before = runner.metrics();
    let started = Instant::now();
    let mut outputs = VecDeque::with_capacity(2);
    for offset in 0..2 {
        outputs.push_back(
            runner
                .enqueue_decode(EngineDecode {
                    tokens: None,
                    start_pos: u32::try_from(
                        context_start
                            .checked_add(offset)
                            .ok_or("decode position overflowed")?,
                    )?,
                })
                .await?
                .map_err(|error| format!("engine decode failed: {error:?}"))?,
        );
    }
    for output in outputs {
        read_token(&runner.read_queued_token(output).await?)?;
    }
    let expected = before
        .submissions
        .checked_add(2)
        .ok_or("submission count overflowed")?;
    runner.wait_for_submissions(expected).await?;
    let sample = sample(before, runner.metrics(), started.elapsed())?;
    Ok(Sample {
        wall_seconds: sample.wall_seconds / 2.0,
        gpu_seconds: sample.gpu_seconds / 2.0,
        submissions: sample.submissions / 2,
    })
}

#[cfg(target_os = "macos")]
async fn prepare_profile_context(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    context_start: usize,
    host_argmax: bool,
) -> Result<u32, Box<dyn Error>> {
    let prefill = context_start
        .checked_sub(1)
        .ok_or("profile context must follow a prefill token")?;
    let token = select_from_tokens(runner, tokens[..prefill].to_vec(), 0, host_argmax).await?;
    select_next(runner, token, u32::try_from(prefill)?, host_argmax).await
}

#[cfg(target_os = "macos")]
async fn measure_decode_step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    token: u32,
    context_start: usize,
    host_argmax: bool,
) -> Result<Sample, Box<dyn Error>> {
    settle_submission_metrics(runner).await?;
    let before = runner.metrics();
    let started = Instant::now();
    select_next(runner, token, u32::try_from(context_start)?, host_argmax).await?;
    runner
        .wait_for_submissions(
            before
                .submissions
                .checked_add(1)
                .ok_or("submission count overflowed")?,
        )
        .await?;
    sample(before, runner.metrics(), started.elapsed())
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
) -> Result<EngineOutput, Box<dyn Error>> {
    Ok(runner
        .step(EngineStep {
            tokens,
            start_pos,
            taps: false,
        })
        .await?
        .map_err(|error| format!("engine step failed: {error:?}"))?)
}

#[cfg(target_os = "macos")]
async fn select_from_tokens(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: Vec<u32>,
    start_pos: u32,
    host_argmax: bool,
) -> Result<u32, Box<dyn Error>> {
    if host_argmax {
        return host_select(runner, tokens, start_pos).await;
    }
    let output = runner
        .decode(EngineDecode {
            tokens: Some(tokens),
            start_pos,
        })
        .await?
        .map_err(|error| format!("engine decode failed: {error:?}"))?;
    read_token(&runner.read(&output.token).await?)
}

#[cfg(target_os = "macos")]
async fn select_next(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    token: u32,
    start_pos: u32,
    host_argmax: bool,
) -> Result<u32, Box<dyn Error>> {
    if host_argmax {
        return host_select(runner, vec![token], start_pos).await;
    }
    let output = runner
        .decode(EngineDecode {
            tokens: None,
            start_pos,
        })
        .await?
        .map_err(|error| format!("engine decode failed: {error:?}"))?;
    read_token(&runner.read(&output.token).await?)
}

#[cfg(target_os = "macos")]
async fn host_select(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: Vec<u32>,
    start_pos: u32,
) -> Result<u32, Box<dyn Error>> {
    let output = step(runner, tokens, start_pos).await?;
    argmax(&decode_f32_le(&runner.read(&output.logits).await?)?)
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

fn print_summary(engine: &str, selection: &str, metric: &str, summary: Summary) {
    println!(
        "{engine}\t{selection}\t{metric}\t{:.2} ({:.2}–{:.2})\t{:.2} ({:.2}–{:.2})\t{:.3}\t{:.3}\t{:.0}",
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

fn profile_report(measurement: &ProfileMeasurement) -> Result<ProfileReport, Box<dyn Error>> {
    let submissions = measurement
        .steps
        .iter()
        .map(|step| {
            step.submission
                .as_ref()
                .ok_or("profiled step has no submission detail")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let categories = profile_categories(&measurement.steps, &submissions);
    let wall = stats(
        measurement
            .steps
            .iter()
            .map(|profile| profile.wall_time.as_secs_f64()),
    );
    let wall_perturbation = stats(
        measurement
            .steps
            .iter()
            .zip(&measurement.baseline)
            .map(|(profile, baseline)| profile.wall_time.as_secs_f64() / baseline.wall_seconds),
    );
    let gpu_perturbation = stats(
        submissions
            .iter()
            .zip(&measurement.baseline)
            .map(|(profile, baseline)| profile.gpu_time.as_secs_f64() / baseline.gpu_seconds),
    );
    let dispatch_coverage = stats(submissions.iter().map(|submission| {
        submission
            .per_dispatch
            .iter()
            .map(|dispatch| dispatch.gpu_time.as_secs_f64())
            .sum::<f64>()
            / submission.gpu_time.as_secs_f64()
    }));
    let gpu_by_op = gpu_by_op(&submissions);
    let gpu_by_dispatch = gpu_by_dispatch(&submissions)?;
    Ok(ProfileReport {
        context_start: measurement.context_start,
        baseline: summarize(&measurement.baseline, 1)?,
        categories,
        wall,
        wall_perturbation,
        gpu_perturbation,
        dispatch_coverage,
        gpu_by_op,
        gpu_by_dispatch,
    })
}

fn print_profile_report(engine: &str, selection: &str, report: &ProfileReport) {
    println!(
        "\n{engine} {selection} tg profile at context {}",
        report.context_start
    );
    println!(
        "unprofiled\twall {:.2} tok/s\tGPU {:.2} tok/s\twall {:.3} ms\tGPU {:.3} ms",
        report.baseline.wall_tps.median,
        report.baseline.gpu_tps.median,
        report.baseline.wall_seconds.median * 1_000.0,
        report.baseline.gpu_seconds.median * 1_000.0,
    );
    println!("category\tcount/token\tms/token\t% wall");
    for category in &report.categories {
        let count = stats(category.values.iter().map(|(count, _)| *count));
        let time = stats(category.values.iter().map(|(_, seconds)| *seconds));
        print_profile_row(category.name, count, time, report.wall.median);
    }
    println!(
        "timestamp perturbation\twall {:.3}x\tGPU {:.3}x\tdispatch coverage {:.2}%",
        report.wall_perturbation.median,
        report.gpu_perturbation.median,
        report.dispatch_coverage.median * 100.0
    );
    for operation in &report.gpu_by_op {
        let count = stats(operation.values.iter().map(|(count, _)| *count));
        let time = stats(operation.values.iter().map(|(_, seconds)| *seconds));
        print_profile_row(
            &format!("gpu.{}", operation.name),
            count,
            time,
            report.wall.median,
        );
    }
}

fn profile_json(report: &ProfileReport) -> serde_json::Value {
    let categories = report
        .categories
        .iter()
        .map(|category| {
            let count = stats(category.values.iter().map(|(count, _)| *count));
            let time = stats(category.values.iter().map(|(_, seconds)| *seconds));
            (
                category.name.to_owned(),
                serde_json::json!({
                    "count_per_token": stats_json(count),
                    "time_seconds": stats_json(time),
                    "percent_of_wall": time.median / report.wall.median * 100.0,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let gpu_by_op = report
        .gpu_by_op
        .iter()
        .map(|operation| {
            let count = stats(operation.values.iter().map(|(count, _)| *count));
            let time = stats(operation.values.iter().map(|(_, seconds)| *seconds));
            (
                operation.name.to_owned(),
                serde_json::json!({
                    "count_per_token": stats_json(count),
                    "gpu_time_seconds": stats_json(time),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let gpu_by_dispatch = report
        .gpu_by_dispatch
        .iter()
        .map(|dispatch| {
            serde_json::json!({
                "index": dispatch.index,
                "op": dispatch.operation,
                "gpu_time_seconds": stats_json(dispatch.time),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "context_start": report.context_start,
        "unprofiled_token_generation": summary_json(report.baseline),
        "categories": categories,
        "timestamp_perturbation": {
            "wall_ratio": stats_json(report.wall_perturbation),
            "gpu_ratio": stats_json(report.gpu_perturbation),
            "dispatch_coverage_ratio": stats_json(report.dispatch_coverage),
        },
        "gpu_by_op": gpu_by_op,
        "gpu_by_dispatch": gpu_by_dispatch,
    })
}

fn print_profile_row(name: &str, count: Stats, time: Stats, wall_seconds: f64) {
    println!(
        "{name}\t{:.1}\t{:.3}\t{:.2}",
        count.median,
        time.median * 1_000.0,
        time.median / wall_seconds * 100.0
    );
}

fn profile_categories(
    steps: &[EngineStepProfile],
    submissions: &[&forja_core::SubmissionProfile],
) -> Vec<ProfileCategory> {
    vec![
        category("wall", steps, |step| (1.0, step.wall_time)),
        category("guest", steps, |step| (1.0, step.guest_time)),
        import_category("import.alloc", steps, |step| step.imports.alloc),
        import_category("import.view.slice", steps, |step| step.imports.view_slice),
        import_category("import.view.reshape", steps, |step| {
            step.imports.view_reshape
        }),
        import_category("import.view.permute", steps, |step| {
            step.imports.view_permute
        }),
        import_category("import.view.broadcast", steps, |step| {
            step.imports.view_broadcast
        }),
        import_category("import.write", steps, |step| step.imports.write),
        import_category("import.dispatch", steps, |step| step.imports.dispatch),
        import_category("import.submit", steps, |step| step.imports.submit),
        import_category("import.read", steps, |step| step.imports.read),
        import_category("import.command_list", steps, |step| {
            step.imports.command_list
        }),
        import_category("import.resource_drop", steps, |step| {
            step.imports.resource_drop
        }),
        import_category("buffer.output", steps, |step| step.allocations),
        import_category("buffer.release", steps, |step| step.releases),
        import_category("output.read", steps, |step| step.output_read),
        submission_category("submit.validation", submissions, |submission| {
            (1.0, submission.validation)
        }),
        submission_category("program.recording", submissions, |submission| {
            (
                count_as_f64(submission.program_recording.count),
                submission.program_recording.time,
            )
        }),
        submission_category("program.encoding", submissions, |submission| {
            (
                count_as_f64(submission.program_encoding.count),
                submission.program_encoding.time,
            )
        }),
        submission_category("program.compile-fallbacks", submissions, |submission| {
            (
                count_as_f64(submission.program_compile_fallbacks),
                Duration::ZERO,
            )
        }),
        submission_category("buffer.metadata", submissions, |submission| {
            (
                count_as_f64(submission.metadata_buffers.count),
                submission.metadata_buffers.time,
            )
        }),
        submission_category("submit.residency", submissions, |submission| {
            (1.0, submission.residency)
        }),
        submission_category("submit.encoding", submissions, |submission| {
            (count_as_f64(submission.dispatches), submission.encoding)
        }),
        submission_category("submit.commit", submissions, |submission| {
            (1.0, submission.commit)
        }),
        submission_category("submit.wait", submissions, |submission| {
            (1.0, submission.wait)
        }),
        submission_category("gpu", submissions, |submission| {
            (count_as_f64(submission.dispatches), submission.gpu_time)
        }),
        submission_category("barriers", submissions, |submission| {
            (count_as_f64(submission.barriers), Duration::ZERO)
        }),
    ]
}

struct ProfileCategory {
    name: &'static str,
    values: Vec<(f64, f64)>,
}

fn category(
    name: &'static str,
    steps: &[EngineStepProfile],
    select: impl Fn(&EngineStepProfile) -> (f64, Duration),
) -> ProfileCategory {
    ProfileCategory {
        name,
        values: steps
            .iter()
            .map(|step| {
                let (count, time) = select(step);
                (count, time.as_secs_f64())
            })
            .collect(),
    }
}

fn import_category(
    name: &'static str,
    steps: &[EngineStepProfile],
    select: impl Fn(&EngineStepProfile) -> ImportProfile,
) -> ProfileCategory {
    category(name, steps, |step| {
        let profile = select(step);
        (count_as_f64(profile.count), profile.time)
    })
}

fn submission_category(
    name: &'static str,
    submissions: &[&forja_core::SubmissionProfile],
    select: impl Fn(&forja_core::SubmissionProfile) -> (f64, Duration),
) -> ProfileCategory {
    ProfileCategory {
        name,
        values: submissions
            .iter()
            .map(|submission| {
                let (count, time) = select(submission);
                (count, time.as_secs_f64())
            })
            .collect(),
    }
}

fn gpu_by_op(submissions: &[&forja_core::SubmissionProfile]) -> Vec<ProfileCategory> {
    let mut samples = BTreeMap::<&'static str, Vec<(f64, f64)>>::new();
    for submission in submissions {
        let mut step = BTreeMap::<&'static str, (u64, Duration)>::new();
        for dispatch in &submission.per_dispatch {
            let value = step.entry(op_name(dispatch.op)).or_default();
            value.0 = value.0.saturating_add(1);
            value.1 = value.1.saturating_add(dispatch.gpu_time);
        }
        for (name, (count, time)) in step {
            samples
                .entry(name)
                .or_default()
                .push((count_as_f64(count), time.as_secs_f64()));
        }
    }
    samples
        .into_iter()
        .map(|(name, values)| ProfileCategory { name, values })
        .collect()
}

fn gpu_by_dispatch(
    submissions: &[&forja_core::SubmissionProfile],
) -> Result<Vec<DispatchProfile>, Box<dyn Error>> {
    let first = submissions.first().ok_or("profile has no submissions")?;
    let mut records = Vec::with_capacity(first.per_dispatch.len());
    for (index, dispatch) in first.per_dispatch.iter().enumerate() {
        let times = submissions
            .iter()
            .map(|submission| {
                let candidate = submission
                    .per_dispatch
                    .get(index)
                    .ok_or("profile dispatch count changed between repetitions")?;
                if op_name(candidate.op) != op_name(dispatch.op) {
                    return Err("profile dispatch order changed between repetitions");
                }
                Ok(candidate.gpu_time.as_secs_f64())
            })
            .collect::<Result<Vec<_>, _>>()?;
        records.push(DispatchProfile {
            index,
            operation: op_name(dispatch.op),
            time: stats(times.into_iter()),
        });
    }
    Ok(records)
}

const fn op_name(operation: Op) -> &'static str {
    match operation {
        Op::Program(_) => "program",
        Op::Copy => "copy",
        Op::Add => "add",
        Op::SiluMul => "silu_mul",
        Op::RmsNorm { .. } => "rms_norm",
        Op::Softmax => "softmax",
        Op::Argmax => "argmax",
        Op::Rope { .. } => "rope",
        Op::Embed => "embed",
        Op::Matmul => "matmul",
        Op::Sdpa { .. } => "sdpa",
    }
}

fn count_as_f64(count: u64) -> f64 {
    u32::try_from(count).map_or(f64::INFINITY, f64::from)
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
