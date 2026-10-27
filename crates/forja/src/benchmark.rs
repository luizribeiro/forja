use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    ffi::OsString,
    fs,
    future::Future,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use forja_config::{KeyPath, Origin, Selection};
use forja_core::{Op, ParamSpace};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineDecode, EngineInfo,
    EngineMetrics, EngineOutput, EngineRunner, EngineStep, EngineStepProfile, ImportProfile,
    PagedPrefix, SamplingParams,
};
use golden_fixtures::{decode_f32_le, sha256_file};

use crate::{
    args::{Bench, BenchPoint, Profile, ProfileMode},
    benchmark_record::{self, Input, PerfKey, Recorded},
    benchmark_stats::{Stats, stats, synthetic_tokens},
    engine::{argmax, engine_load_config, read_token, validate_engine, weights_path},
    machine_load,
};

#[derive(Clone, Copy)]
pub(crate) struct Summary {
    pub(crate) wall_tps: Stats,
    pub(crate) gpu_tps: Stats,
    pub(crate) wall_seconds: Stats,
    pub(crate) gpu_seconds: Stats,
    pub(crate) submissions: Stats,
}

#[derive(Clone, Copy, serde::Serialize)]
struct Sample {
    wall_seconds: f64,
    gpu_seconds: f64,
    submissions: u32,
    measurement_window: [f64; 2],
}

pub(crate) struct TokenProfile {
    pub(crate) device: String,
    pub(crate) baseline_wall: Duration,
    pub(crate) baseline_gpu: Duration,
    pub(crate) profiled_wall: Duration,
    pub(crate) profiled_gpu: Duration,
    pub(crate) cache_setup_wall: Duration,
    pub(crate) step: EngineStepProfile,
}

#[derive(Clone, Copy)]
struct DecodeOptions {
    host_argmax: bool,
    overlap: bool,
    sampling: SamplingParams,
}

struct ProfileMeasurement {
    context_start: usize,
    baseline: Vec<Sample>,
    steps: Vec<EngineStepProfile>,
}

struct ContextMeasurement {
    context_start: usize,
    summary: Summary,
    samples: Vec<Sample>,
}

#[derive(Clone)]
struct BatchDecodeMeasurement {
    bucket: usize,
    contexts: Vec<u32>,
    capture_graphs: u64,
    capture_seconds: f64,
    capture_validation_seconds: f64,
    summary: Summary,
    wall_itl: Stats,
    gpu_itl: Stats,
    samples: Vec<Sample>,
}

const REPLAY_VALIDATION_ITERATIONS: u32 = 100_000;

fn measure_replay_parameter_validation(
    repetitions: usize,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let mut results = Vec::new();
    for parameters in [0_usize, 4, 8] {
        let space = ParamSpace::new(vec![0..=u32::MAX; parameters])?;
        let values = vec![2_048; parameters];
        let samples = (0..repetitions.max(5))
            .map(|_| {
                let started = Instant::now();
                for _ in 0..REPLAY_VALIDATION_ITERATIONS {
                    std::hint::black_box(space.values(values.clone())).ok();
                }
                started.elapsed().as_secs_f64() * 1_000_000_000.0
                    / f64::from(REPLAY_VALIDATION_ITERATIONS)
            })
            .collect::<Vec<_>>();
        results.push(serde_json::json!({
            "parameters": parameters,
            "nanoseconds": stats_json(stats(samples.into_iter())),
        }));
    }
    Ok(serde_json::Value::Array(results))
}
const BATCH_CONTEXTS: [u32; 2] = [512, 2_048];
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

struct BenchMeasurement {
    pp: Summary,
    tg: Summary,
    contexts: Vec<ContextMeasurement>,
    device: String,
    profiles: Vec<ProfileMeasurement>,
    output_digest: String,
    peak_footprint: u64,
    raw_samples: serde_json::Value,
}

struct Acceptance {
    profile_path: PathBuf,
    profile_name: String,
    base_profile_hash: String,
    tuning_choice: bool,
}

struct DispatchProfile {
    index: usize,
    operation: &'static str,
    time: Stats,
}

pub(crate) async fn run(options: &Bench) -> Result<(), Box<dyn Error>> {
    for engine in &options.engines {
        validate_engine(
            engine,
            &options.model_dir,
            crate::args::Backend::Metal,
            &options.config.paths.scratch,
        )?;
    }
    if let Some(path) = &options.accept_from
        && !options.mode.remeasure
    {
        return adopt_report(options, path);
    }
    #[cfg(target_os = "macos")]
    return run_metal(options).await;
    #[cfg(not(target_os = "macos"))]
    Err("engine benchmarks require macOS and Metal".into())
}

#[cfg(target_os = "macos")]
pub(crate) async fn measure_token_profile(
    options: &Profile,
) -> Result<TokenProfile, Box<dyn Error>> {
    let (backend, live_bytes) =
        crate::engine::metal_backend(options.graph_replay, options.pool_capacity, &options.limits)?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        &options.engine,
        backend,
        crate::engine::limits_for_live_bytes(&options.limits, live_bytes)?,
        weights_path(&options.model_dir)?,
    )
    .await?;
    let info = runner.describe().await?;
    let max_context = usize::try_from(info.max_context)?;
    let exceeds_context = match options.mode {
        ProfileMode::Decode => options.context >= max_context,
        ProfileMode::Prefill => options.context > max_context,
    };
    if exceeds_context {
        return Err("profile shape exceeds the engine context".into());
    }
    runner
        .load_with_selections(
            None,
            &engine_load_config(&options.engine, &options.engine_config)?,
        )
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let token_count = match options.mode {
        ProfileMode::Decode => options.context - 1,
        ProfileMode::Prefill => options.context,
    };
    let tokens = synthetic_tokens(token_count, info.vocab);
    let decode = DecodeOptions {
        host_argmax: false,
        overlap: false,
        sampling: SamplingParams {
            temperature: options.sampling.temperature,
            top_k: options.sampling.top_k,
            top_p: options.sampling.top_p,
            seed: options.sampling.seed,
        },
    };
    if let Some(step_tokens) = options.step_tokens {
        return measure_serving_prefill_profile(
            &mut runner,
            &info,
            &tokens,
            options,
            step_tokens,
            device,
        )
        .await;
    }
    for _ in 0..options.warmups {
        measure_profile_phase(&mut runner, &tokens, options, decode).await?;
    }
    let baseline = measure_profile_phase(&mut runner, &tokens, options, decode).await?;
    let profiled = measure_profiled_phase(&mut runner, &tokens, options, decode).await?;
    Ok(TokenProfile {
        device,
        baseline_wall: Duration::from_secs_f64(baseline.wall_seconds),
        baseline_gpu: Duration::from_secs_f64(baseline.gpu_seconds),
        profiled_wall: Duration::from_secs_f64(profiled.wall_seconds),
        profiled_gpu: Duration::from_secs_f64(profiled.gpu_seconds),
        cache_setup_wall: Duration::ZERO,
        step: runner
            .take_profile()
            .ok_or("profiled phase produced no timing detail")?,
    })
}

#[cfg(target_os = "macos")]
async fn measure_serving_prefill_profile(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    info: &EngineInfo,
    tokens: &[u32],
    options: &Profile,
    step_tokens: usize,
    device: String,
) -> Result<TokenProfile, Box<dyn Error>> {
    let serving = info
        .serving
        .as_ref()
        .ok_or("serving prefill requires a paged engine")?;
    let shape = serving
        .batch_shapes
        .iter()
        .find(|shape| {
            shape.phase == BatchPhase::Prompt
                && shape.bucket == 1
                && shape.selection == BatchSelection::Combined
                && usize::try_from(shape.tokens) == Ok(step_tokens)
        })
        .copied()
        .ok_or("serving prefill step has no exact batch shape")?;
    let reserved = u64::try_from(options.context)?;
    let prefix =
        prepare_serving_profile_prefix(runner, &tokens[..options.prefix_context], shape, reserved)
            .await?;
    for _ in 0..options.warmups {
        measure_serving_profile_phase(runner, &prefix, tokens, options, shape, reserved, false)
            .await?;
    }
    let (baseline, cache_setup, _) =
        measure_serving_profile_phase(runner, &prefix, tokens, options, shape, reserved, false)
            .await?;
    let (profiled, _, step_profiles) =
        measure_serving_profile_phase(runner, &prefix, tokens, options, shape, reserved, true)
            .await?;
    Ok(TokenProfile {
        device,
        baseline_wall: Duration::from_secs_f64(baseline.wall_seconds),
        baseline_gpu: Duration::from_secs_f64(baseline.gpu_seconds),
        profiled_wall: Duration::from_secs_f64(profiled.wall_seconds),
        profiled_gpu: Duration::from_secs_f64(profiled.gpu_seconds),
        cache_setup_wall: cache_setup,
        step: merge_step_profiles(step_profiles)?,
    })
}

#[cfg(target_os = "macos")]
async fn prepare_serving_profile_prefix(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    shape: BatchShape,
    reserved: u64,
) -> Result<PagedPrefix, Box<dyn Error>> {
    runner.prepare_paged_lane(0, None, 0)?;
    runner.prepare_paged_lane(0, None, reserved)?;
    let step_tokens = usize::try_from(shape.tokens)?;
    for (index, chunk) in tokens.chunks(step_tokens).enumerate() {
        run_serving_profile_step(
            runner,
            shape,
            chunk,
            index
                .checked_mul(step_tokens)
                .ok_or("prefix offset overflowed")?,
        )
        .await?;
    }
    let prefix = runner
        .paged_prefix()?
        .ok_or("serving prefill did not produce a paged prefix")?;
    runner.prepare_paged_lane(0, None, 0)?;
    Ok(prefix)
}

#[cfg(target_os = "macos")]
async fn measure_serving_profile_phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prefix: &PagedPrefix,
    tokens: &[u32],
    options: &Profile,
    shape: BatchShape,
    reserved: u64,
    profiling: bool,
) -> Result<(Sample, Duration, Vec<EngineStepProfile>), Box<dyn Error>> {
    runner.prepare_paged_lane(0, None, 0)?;
    let setup_started = Instant::now();
    let fork = prefix.fork_reserved(u64::try_from(options.prefix_context)?, reserved)?;
    runner.prepare_paged_lane(0, Some(fork), reserved)?;
    let cache_setup = setup_started.elapsed();
    let before = runner.metrics();
    let started = Instant::now();
    let mut profiles = Vec::new();
    runner.set_profiling(profiling);
    let step_tokens = usize::try_from(shape.tokens)?;
    for (index, chunk) in tokens[options.prefix_context..]
        .chunks(step_tokens)
        .enumerate()
    {
        let start = options
            .prefix_context
            .checked_add(
                index
                    .checked_mul(step_tokens)
                    .ok_or("suffix offset overflowed")?,
            )
            .ok_or("suffix position overflowed")?;
        run_serving_profile_step(runner, shape, chunk, start).await?;
        if profiling {
            profiles.push(
                runner
                    .take_profile()
                    .ok_or("serving prefill step produced no timing detail")?,
            );
        }
    }
    runner.set_profiling(false);
    runner.wait_for_queued_submissions().await?;
    let sample = sample(before, runner.metrics(), started.elapsed())?;
    Ok((sample, cache_setup, profiles))
}

#[cfg(target_os = "macos")]
async fn run_serving_profile_step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    shape: BatchShape,
    tokens: &[u32],
    start: usize,
) -> Result<(), Box<dyn Error>> {
    let output = runner
        .batch_step(EngineBatch {
            shape,
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(tokens.to_vec()),
                start_pos: u32::try_from(start)?,
                sampling: SamplingParams::default(),
            })],
        })
        .await?
        .map_err(|error| format!("serving prefill failed: {error:?}"))?;
    runner.read(&output.tokens).await?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn merge_step_profiles(
    mut profiles: Vec<EngineStepProfile>,
) -> Result<EngineStepProfile, &'static str> {
    let Some(last) = profiles.last().cloned() else {
        return Err("serving prefill produced no profiles");
    };
    let mut merged = EngineStepProfile {
        submission: last.submission,
        memory: last.memory,
        ..EngineStepProfile::default()
    };
    for profile in profiles.drain(..) {
        merged.wall_time = merged.wall_time.saturating_add(profile.wall_time);
        merged.guest_time = merged.guest_time.saturating_add(profile.guest_time);
        merged.submissions.extend(profile.submissions);
    }
    Ok(merged)
}

#[cfg(target_os = "macos")]
async fn measure_profiled_phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    options: &Profile,
    decode: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    let result = match options.mode {
        ProfileMode::Prefill => {
            prepare_profile_prefix(runner, tokens, options.prefix_context).await?;
            runner.set_profiling(true);
            measure_prefill_from(
                runner,
                &tokens[options.prefix_context..],
                options.prefix_context,
            )
            .await
        }
        ProfileMode::Decode => {
            let token = prepare_profile_context(runner, tokens, options.context, decode).await?;
            runner.set_profiling(true);
            measure_decode_step(runner, token, options.context, decode).await
        }
    };
    runner.set_profiling(false);
    result
}

#[cfg(target_os = "macos")]
async fn measure_profile_phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    options: &Profile,
    decode: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    match options.mode {
        ProfileMode::Prefill => {
            prepare_profile_prefix(runner, tokens, options.prefix_context).await?;
            measure_prefill_from(
                runner,
                &tokens[options.prefix_context..],
                options.prefix_context,
            )
            .await
        }
        ProfileMode::Decode => {
            let token = prepare_profile_context(runner, tokens, options.context, decode).await?;
            measure_decode_step(runner, token, options.context, decode).await
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_lines)]
async fn run_metal(options: &Bench) -> Result<(), Box<dyn Error>> {
    let acceptance = prepare_acceptance(options)?;
    let started = Instant::now();
    let load_before = machine_load::capture("BEFORE BENCHMARK")?;
    let commit = crate::provenance::commit();
    let os = command_output("sw_vers", &["-productVersion"])?;
    let date = command_output("date", &["-u", "+%Y-%m-%dT%H:%MZ"])?;
    let gpu_cores = gpu_core_count()?;
    let binary_sha256 = sha256_file(std::env::current_exe()?)?;
    let profiles = options
        .engines
        .iter()
        .map(|component| crate::resolution::read_embedded_profile(component))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = benchmark_inputs(options, &profiles)?;
    if let Some(record) = &options.rerun {
        validate_rerun_inputs(
            record,
            &inputs,
            options
                .allow_diff
                .iter()
                .any(|key| key.as_str() == "engine.sha256"),
        )?;
        if record.provenance.commit != commit {
            println!(
                "rerun binary commit differs: recorded {}, current {commit}",
                record.provenance.commit
            );
        }
    }
    let perf_keys = options
        .points
        .iter()
        .map(|point| {
            inputs
                .iter()
                .zip(&profiles)
                .map(|(input, profile)| benchmark_record::perf_key(point, input, profile))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(record) = &options.rerun {
        validate_comparability(options, record, &perf_keys)?;
    }
    let perf_hashes = perf_keys
        .iter()
        .map(|keys| {
            keys.iter()
                .map(benchmark_record::comparison_hash)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (results, device) = measure_points(options, &perf_keys, &perf_hashes).await?;
    let load_after = machine_load::capture("AFTER BENCHMARK")?;
    check_strategy_outputs(&options.strategy_axes, &results)?;
    if options
        .axes
        .get(&KeyPath::new("engine.tunings"))
        .is_some_and(|values| values.len() > 1)
    {
        print_tuning_contributions(&results)?;
    }
    if let Some(record) = &options.rerun {
        print_result_diff(record, &results)?;
    }
    if options.json.is_some() || acceptance.is_some() {
        let snapshot = benchmark_record::snapshot(options, &device, &os)?;
        let report = serde_json::json!({
            "schema_version": benchmark_record::SCHEMA_VERSION,
            "implementation": "forja",
            "measurement_mode": benchmark_record::measurement_mode(options),
            "provenance": {
                "commit": commit,
                "dirty": crate::provenance::dirty(),
                "binary_sha256": binary_sha256,
                "device": device,
                "gpu_cores": gpu_cores,
                "macos": os,
                "date": date,
                "duration_s": started.elapsed().as_secs(),
                "machine_load": {
                    "before": load_before,
                    "after": load_after,
                },
            },
            "inputs": inputs,
            "config": snapshot.config,
            "origins": snapshot.origins,
            "auto_notes": snapshot.auto_notes,
            "config_hash": benchmark_record::config_hash(&snapshot, &inputs)?,
            "perf_hash": benchmark_record::combined_perf_hash(&perf_hashes.iter().flatten().cloned().collect::<Vec<_>>())?,
            "axes": options.axes,
            "allow_diff": options.allow_diff.iter().map(KeyPath::as_str).collect::<Vec<_>>(),
            "results": results,
        });
        if let Some(path) = &options.json {
            if options.breakdown {
                let breakdown_path = breakdown_record_path(path, &options.config.paths.scratch)?;
                if let Some(parent) = breakdown_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                write_json(&breakdown_path, &report)?;
                println!("breakdown record: {}", breakdown_path.display());
            }
            write_json(path, &slim_record(&report))?;
        }
        if let Some(acceptance) = acceptance {
            accept_report(options, &acceptance, &report)?;
        }
    }
    Ok(())
}

fn benchmark_inputs(
    options: &Bench,
    profiles: &[forja_config::Profile],
) -> Result<Vec<Input>, std::io::Error> {
    options
        .engines
        .iter()
        .zip(profiles)
        .map(|(component, profile)| {
            Ok(Input {
                engine_sha256: sha256_file(component)?,
                engine_variant: String::new(),
                engine_build_profile: String::new(),
                profile_name: component
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                profile_family: profile.family().to_owned(),
                numerics: Some(*profile.numerics()),
                profile_hash: Some(profile.sha256().map_err(std::io::Error::other)?),
                weights_sha256: profile.model().weights_sha256.clone(),
                model_revision: Some(profile.model().revision.clone()),
            })
        })
        .collect()
}

fn adopt_report(options: &Bench, path: &Path) -> Result<(), Box<dyn Error>> {
    let acceptance = prepare_acceptance(options)?.ok_or("--accept-from requires --accept")?;
    let bytes = fs::read(path)?;
    let mut report: serde_json::Value = serde_json::from_slice(&bytes)?;
    let record: Recorded = serde_json::from_value(report.clone())?;
    let profiles = options
        .engines
        .iter()
        .map(|component| crate::resolution::read_embedded_profile(component))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = benchmark_inputs(options, &profiles)?;
    let device = report["provenance"]["device"]
        .as_str()
        .ok_or("record has no device")?;
    let os = report["provenance"]["macos"]
        .as_str()
        .ok_or("record has no OS")?;
    let snapshot = benchmark_record::snapshot(options, device, os)?;
    let mut expected = Vec::new();
    let mut hashes = Vec::new();
    for point in &options.points {
        let hash = benchmark_record::comparison_hash(&benchmark_record::perf_key(
            point,
            &inputs[0],
            &profiles[0],
        )?)?;
        for selection in &point.selection {
            expected.push((hash.clone(), serde_json::to_value(selection)?));
        }
        hashes.push(hash);
    }
    benchmark_record::validate_adoption(
        &report,
        &sha256_file(std::env::current_exe()?)?,
        &inputs,
        &benchmark_record::config_hash(&snapshot, &inputs)?,
        &benchmark_record::combined_perf_hash(&hashes)?,
    )?;
    let keys = benchmark_record::recorded_perf_keys(&record)?;
    for (result, key) in record.results.iter().zip(keys) {
        let hash = benchmark_record::comparison_hash(&key)?;
        let index = expected
            .iter()
            .position(|(expected_hash, selection)| {
                expected_hash == &hash && selection == &result["selection"]
            })
            .ok_or("--accept-from has an unexpected or duplicate result")?;
        if result["perf_hash"] != hash || result["output_digest"].as_str().is_none() {
            return Err("--accept-from has an invalid result hash or output digest".into());
        }
        expected.remove(index);
    }
    if !expected.is_empty() || record.results.is_empty() {
        return Err("--accept-from has incomplete result coverage".into());
    }
    report["adoption"] = serde_json::json!({
        "source_record_sha256": format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes)),
        "remeasured": false,
    });
    accept_report(options, &acceptance, &report)?;
    if let Some(output) = &options.json {
        write_json_atomic(output, &slim_record(&report))?;
    }
    Ok(())
}

fn prepare_acceptance(options: &Bench) -> Result<Option<Acceptance>, String> {
    if !options.accept {
        return Ok(None);
    }
    let [component] = options.engines.as_slice() else {
        return Err("--accept requires exactly one engine".to_owned());
    };
    if crate::provenance::dirty() || live_worktree_dirty()? {
        return Err("--accept requires clean provenance at measurement start".to_owned());
    }
    if crate::provenance::commit().is_empty() || crate::provenance::commit() == "unknown" {
        return Err("--accept requires a known source commit".to_owned());
    }
    if options.origins.iter().any(|(key, origin)| {
        matches!(origin, Origin::UserFile(_)) && !key.as_str().starts_with("paths.")
    }) {
        return Err("--accept refuses user-file values outside paths".to_owned());
    }
    let mut tuning_choice = false;
    for (key, values) in &options.axes {
        if is_tuning_axis(key) {
            if values.len() != 1 {
                return Err("--accept requires one-value Tuning-class axes".to_owned());
            }
            tuning_choice = true;
        } else if key.as_str().starts_with("backend.metal.") || key.as_str() == "engine.replay" {
            return Err("--accept refuses Strategy-class axes".to_owned());
        }
    }
    let profile = crate::resolution::read_embedded_profile(component)?;
    let base_profile_hash = profile
        .sha256()
        .map_err(|error| format!("cannot hash engine profile: {error}"))?;
    let profile_path = crate::resolution::source_profile(&base_profile_hash)?;
    let profile_name = profile_path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "source profile name is not valid UTF-8".to_owned())?
        .to_owned();
    Ok(Some(Acceptance {
        profile_path,
        profile_name,
        base_profile_hash,
        tuning_choice,
    }))
}

fn live_worktree_dirty() -> Result<bool, String> {
    let output = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .map_err(|error| format!("cannot inspect worktree provenance: {error}"))?;
    if !output.status.success() {
        return Err("cannot inspect worktree provenance".to_owned());
    }
    Ok(!output.stdout.is_empty())
}

fn is_tuning_axis(key: &KeyPath) -> bool {
    key.as_str() == "engine.tunings"
        || key.as_str().starts_with("engine.variant-picks.")
        || key.as_str().starts_with("engine.variant-rules.")
}

fn accept_report(
    options: &Bench,
    acceptance: &Acceptance,
    report: &serde_json::Value,
) -> Result<(), Box<dyn Error>> {
    let family = acceptance
        .profile_path
        .parent()
        .and_then(Path::parent)
        .ok_or("source profile has no engine family directory")?;
    let directory = family.join("bench").join(&acceptance.profile_name);
    fs::create_dir_all(&directory)?;
    let mut record = slim_record(report);
    if acceptance.tuning_choice {
        let source = fs::read_to_string(&acceptance.profile_path)?;
        let profile: forja_config::Profile = toml::from_str(&source)?;
        let point = options
            .points
            .first()
            .ok_or("benchmark has no acceptance point")?;
        let accepted = accepted_profile(&source, &profile, &point.config.engine)?;
        let accepted_source = accepted.canonical_toml()?;
        let accepted_hash = accepted.sha256()?;
        record["acceptance"] = serde_json::json!({
            "base_profile_hash": acceptance.base_profile_hash,
            "accepted_profile_hash": accepted_hash,
            "measured_component_hash": report["inputs"][0]["engine_sha256"],
        });
        write_atomic(&acceptance.profile_path, accepted_source.as_bytes())?;
        write_json_atomic(&directory.join("tunings.json"), &record)?;
        println!("accepted profile: {}", acceptance.profile_path.display());
    } else {
        write_json_atomic(&directory.join("accepted.json"), &record)?;
    }
    println!("accepted record: {}", directory.display());
    Ok(())
}

fn accepted_profile(
    source: &str,
    profile: &forja_config::Profile,
    engine: &forja_config::Engine,
) -> Result<forja_config::Profile, Box<dyn Error>> {
    let selection = engine.resolve(profile)?;
    let mut value: toml::Value = toml::from_str(source)?;
    let table = value
        .as_table_mut()
        .ok_or("source profile is not a table")?;
    table.insert(
        "default-tunings".to_owned(),
        toml::Value::Array(
            selection
                .tunings
                .iter()
                .cloned()
                .map(toml::Value::String)
                .collect(),
        ),
    );
    table.insert(
        "variants".to_owned(),
        toml::Value::try_from(selection.variants)?,
    );
    Ok(toml::from_str(&toml::to_string(&value)?)?)
}

fn write_json_atomic(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let name = path
        .file_name()
        .ok_or("atomic write target has no file name")?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn validate_rerun_inputs(
    record: &Recorded,
    inputs: &[Input],
    allow_engine_diff: bool,
) -> Result<(), String> {
    if record.inputs.len() != inputs.len() {
        return Err(format!(
            "rerun input count mismatch: record has {}, command has {}",
            record.inputs.len(),
            inputs.len()
        ));
    }
    if allow_engine_diff {
        for (recorded, current) in record.inputs.iter().zip(inputs) {
            validate_engine_identity(recorded, current)?;
            if recorded.weights_sha256 != current.weights_sha256 {
                return Err(format!(
                    "rerun weights sha256 mismatch: recorded {}, current {}",
                    recorded.weights_sha256, current.weights_sha256
                ));
            }
        }
        return Ok(());
    }
    let mut unmatched = inputs.iter().collect::<Vec<_>>();
    for recorded in &record.inputs {
        let Some(index) = unmatched
            .iter()
            .position(|current| current.engine_sha256 == recorded.engine_sha256)
        else {
            return Err(format!(
                "rerun engine sha256 mismatch: recorded {} is not present",
                recorded.engine_sha256
            ));
        };
        let current = unmatched.remove(index);
        validate_engine_identity(recorded, current)?;
        if recorded.weights_sha256 != current.weights_sha256 {
            return Err(format!(
                "rerun weights sha256 mismatch for engine {}: recorded {}, current {}",
                recorded.engine_sha256, recorded.weights_sha256, current.weights_sha256
            ));
        }
    }
    Ok(())
}

fn validate_engine_identity(recorded: &Input, current: &Input) -> Result<(), String> {
    for (name, recorded, current) in [
        ("variant", &recorded.engine_variant, &current.engine_variant),
        (
            "build profile",
            &recorded.engine_build_profile,
            &current.engine_build_profile,
        ),
    ] {
        if !recorded.is_empty() && !current.is_empty() && recorded != current {
            return Err(format!(
                "rerun engine {name} mismatch: recorded {recorded}, current {current}"
            ));
        }
    }
    for (name, recorded, current) in [
        (
            "profile name",
            &recorded.profile_name,
            &current.profile_name,
        ),
        (
            "profile family",
            &recorded.profile_family,
            &current.profile_family,
        ),
    ] {
        if !recorded.is_empty() && recorded != current {
            return Err(format!(
                "rerun engine {name} mismatch: recorded {recorded}, current {current}"
            ));
        }
    }
    if recorded.profile_hash.is_some() && recorded.profile_hash != current.profile_hash {
        return Err("rerun engine profile hash mismatch".to_owned());
    }
    Ok(())
}

fn validate_comparability(
    options: &Bench,
    record: &Recorded,
    perf_keys: &[Vec<PerfKey>],
) -> Result<(), String> {
    let recorded = benchmark_record::recorded_perf_keys(record)?;
    let mut current = options
        .points
        .iter()
        .zip(perf_keys)
        .flat_map(|(point, keys)| {
            keys.iter()
                .flat_map(|key| std::iter::repeat_n(key.clone(), point.selection.len()))
        })
        .collect::<Vec<_>>();
    if record.schema_version < 3 {
        for key in &mut current {
            key.remove("engine.tunings");
            key.remove("engine.picks_hash");
        }
    }
    if record.schema_version < 5 {
        for key in &mut current {
            key.remove("bench.batch-decode.buckets");
            key.remove("bench.batch-decode.contexts");
        }
    }
    if recorded.len() != current.len() {
        return Err(format!(
            "benchmark records are not comparable:\n  results.count: {} -> {}",
            recorded.len(),
            current.len()
        ));
    }
    let ignored = record
        .axes
        .keys()
        .chain(options.axes.keys())
        .chain(&options.allow_diff)
        .map(KeyPath::as_str)
        .collect::<Vec<_>>();
    for allowed in &options.allow_diff {
        if !recorded
            .iter()
            .chain(&current)
            .any(|key| key.contains_key(allowed.as_str()))
        {
            return Err(format!(
                "--allow-diff names unknown performance key {allowed}"
            ));
        }
    }
    if ignored.contains(&"engine.sha256") {
        return compare_perf_keys(&recorded, &current, &ignored);
    }
    let recorded = perf_keys_by_engine(recorded)?;
    let current = perf_keys_by_engine(current)?;
    if recorded.keys().ne(current.keys()) {
        return Err("benchmark records are not comparable:\n  engine.sha256 differs".to_owned());
    }
    let mut differences = BTreeMap::new();
    for (engine, old_keys) in &recorded {
        let new_keys = &current[engine];
        if old_keys.len() != new_keys.len() {
            differences.insert(
                "results.count".to_owned(),
                format!("{} -> {}", old_keys.len(), new_keys.len()),
            );
            continue;
        }
        collect_key_differences(old_keys, new_keys, &ignored, &mut differences)?;
    }
    if differences.is_empty() {
        return Ok(());
    }
    let report = differences
        .into_iter()
        .map(|(key, values)| format!("  {key}: {values}"))
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!("benchmark records are not comparable:\n{report}"))
}

fn compare_perf_keys(
    recorded: &[PerfKey],
    current: &[PerfKey],
    ignored: &[&str],
) -> Result<(), String> {
    let mut differences = BTreeMap::new();
    collect_key_differences(recorded, current, ignored, &mut differences)?;
    if differences.is_empty() {
        return Ok(());
    }
    let report = differences
        .into_iter()
        .map(|(key, values)| format!("  {key}: {values}"))
        .collect::<Vec<_>>()
        .join("\n");
    Err(format!("benchmark records are not comparable:\n{report}"))
}

fn collect_key_differences(
    recorded: &[PerfKey],
    current: &[PerfKey],
    ignored: &[&str],
    differences: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for (old, new) in recorded.iter().zip(current) {
        let old = stripped_key(old, ignored);
        let new = stripped_key(new, ignored);
        if benchmark_record::comparison_hash(&old)? == benchmark_record::comparison_hash(&new)? {
            continue;
        }
        for key in old.keys().chain(new.keys()) {
            if old.get(key) != new.get(key) {
                differences.insert(
                    key.clone(),
                    format!(
                        "{} -> {}",
                        old.get(key)
                            .map_or("<missing>".to_owned(), ToString::to_string),
                        new.get(key)
                            .map_or("<missing>".to_owned(), ToString::to_string)
                    ),
                );
            }
        }
    }
    Ok(())
}

fn perf_keys_by_engine(keys: Vec<PerfKey>) -> Result<BTreeMap<String, Vec<PerfKey>>, String> {
    let mut grouped = BTreeMap::<String, Vec<PerfKey>>::new();
    for key in keys {
        let engine = key
            .get("engine.sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "performance key has no engine.sha256".to_owned())?
            .to_owned();
        grouped.entry(engine).or_default().push(key);
    }
    Ok(grouped)
}

fn stripped_key(key: &PerfKey, ignored: &[&str]) -> PerfKey {
    key.iter()
        .filter(|(name, _)| !ignored.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn print_result_diff(record: &Recorded, current: &[serde_json::Value]) -> Result<(), String> {
    println!("rerun result diff (recorded -> current)");
    for (index, (old, new)) in record.results.iter().zip(current).enumerate() {
        for metric in ["pp", "tg"] {
            let old_stats = wall_stats(old, metric)?;
            let new_stats = wall_stats(new, metric)?;
            let percent = if old_stats.0 == 0.0 {
                f64::NAN
            } else {
                (new_stats.0 / old_stats.0 - 1.0) * 100.0
            };
            println!(
                "result {index} {metric}: {:.2} ({:.2}–{:.2}) -> {:.2} ({:.2}–{:.2}) {percent:+.2}%",
                old_stats.0, old_stats.1, old_stats.2, new_stats.0, new_stats.1, new_stats.2,
            );
        }
    }
    Ok(())
}

fn print_tuning_contributions(results: &[serde_json::Value]) -> Result<(), String> {
    let mut baselines = BTreeMap::<(u64, String), &serde_json::Value>::new();
    println!("tuning\tinput\tselection\tmetric\tcontribution (95% CI)");
    for result in results {
        let input = result["input"]
            .as_u64()
            .ok_or_else(|| "benchmark result input is invalid".to_owned())?;
        let selection = result["selection"]
            .as_str()
            .ok_or_else(|| "benchmark result selection is invalid".to_owned())?;
        let key = (input, selection.to_owned());
        let Some(baseline) = baselines.get(&key) else {
            baselines.insert(key, result);
            continue;
        };
        let (name, base_is_tuned) = tuning_difference(baseline, result)?;
        for metric in ["pp", "tg"] {
            let base = wall_stats(baseline, metric)?;
            let candidate = wall_stats(result, metric)?;
            let (with, without) = if base_is_tuned {
                (base, candidate)
            } else {
                (candidate, base)
            };
            let median = ratio_percent(with.0, without.0);
            let low = ratio_percent(with.1, without.2);
            let high = ratio_percent(with.2, without.1);
            println!(
                "{name}\t{input}\t{selection}\t{metric}\t{median:+.2}% ({low:+.2}%–{high:+.2}%)"
            );
        }
    }
    Ok(())
}

fn tuning_difference(
    baseline: &serde_json::Value,
    candidate: &serde_json::Value,
) -> Result<(String, bool), String> {
    let baseline = tuning_names(baseline)?;
    let candidate = tuning_names(candidate)?;
    let removed = baseline
        .iter()
        .filter(|name| !candidate.contains(name))
        .cloned()
        .collect::<Vec<_>>();
    let added = candidate
        .iter()
        .filter(|name| !baseline.contains(name))
        .cloned()
        .collect::<Vec<_>>();
    match (removed.as_slice(), added.as_slice()) {
        ([name], []) => Ok((name.clone(), true)),
        ([], [name]) => Ok((name.clone(), false)),
        _ => Err("tuning comparison points must differ by exactly one tuning".to_owned()),
    }
}

fn tuning_names(result: &serde_json::Value) -> Result<Vec<String>, String> {
    result["point"]["engine.tunings"]
        .as_array()
        .ok_or_else(|| "tuning comparison point is not an array".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "tuning comparison name is not a string".to_owned())
        })
        .collect()
}

fn ratio_percent(numerator: f64, denominator: f64) -> f64 {
    (numerator / denominator - 1.0) * 100.0
}

fn wall_stats(result: &serde_json::Value, metric: &str) -> Result<(f64, f64, f64), String> {
    let stats = &result[metric]["tokens_per_second"]["wall"];
    let median = stats["median"]
        .as_f64()
        .ok_or_else(|| format!("record result {metric} has no wall median"))?;
    let interval = stats["ci95"]
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| format!("record result {metric} has no wall confidence interval"))?;
    let low = interval[0]
        .as_f64()
        .ok_or_else(|| format!("record result {metric} has invalid confidence bounds"))?;
    let high = interval[1]
        .as_f64()
        .ok_or_else(|| format!("record result {metric} has invalid confidence bounds"))?;
    Ok((median, low, high))
}

fn check_strategy_outputs(
    strategy_axes: &[forja_config::KeyPath],
    results: &[serde_json::Value],
) -> Result<(), String> {
    if strategy_axes.is_empty() {
        return Ok(());
    }
    let mut outputs = BTreeMap::<String, (&serde_json::Value, &str)>::new();
    for result in results {
        let point = result["point"]
            .as_object()
            .ok_or_else(|| "benchmark result point is not an object".to_owned())?;
        let workload = point
            .iter()
            .filter(|(key, _)| {
                !strategy_axes
                    .iter()
                    .any(|axis| axis.as_str() == key.as_str())
            })
            .collect::<BTreeMap<_, _>>();
        let group = serde_json::to_string(&(&result["input"], &result["selection"], workload))
            .map_err(|error| format!("cannot group benchmark outputs: {error}"))?;
        let digest = result["output_digest"]
            .as_str()
            .ok_or_else(|| "benchmark result has no output digest".to_owned())?;
        if let Some((first, expected)) = outputs.get(&group)
            && *expected != digest
        {
            return Err(format!(
                "strategy axis output mismatch for input {} and selection {}:\n  {} -> {expected}\n  {} -> {digest}",
                result["input"], result["selection"], first["point"], result["point"],
            ));
        }
        outputs.entry(group).or_insert((result, digest));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn measure_points(
    options: &Bench,
    perf_keys: &[Vec<PerfKey>],
    perf_hashes: &[Vec<String>],
) -> Result<(Vec<serde_json::Value>, String), Box<dyn Error>> {
    let mut results = Vec::new();
    let mut record_device = None;
    let replay_parameter_validation = measure_replay_parameter_validation(options.reps)?;
    println!(
        "engine\tselection\tmetric\twall tok/s (95% CI)\tGPU tok/s (95% CI)\twall ms\tGPU ms\tsubmissions"
    );
    for (point_index, point) in options.points.iter().enumerate() {
        for (input, component) in options.engines.iter().enumerate() {
            let engine = component.display().to_string();
            let batches = if batch_measurements_enabled(
                options.rerun.as_ref().map(|record| record.schema_version),
            ) {
                measure_record_batches(options, point, component).await?
            } else {
                Vec::new()
            };
            for batch in &batches {
                print_summary(
                    &engine,
                    "batch-combined",
                    &format!("batch-{}", batch.bucket),
                    batch.summary,
                );
            }
            for &strategy in &point.selection {
                let (host_argmax, overlap) = selection_mode(strategy);
                let selection = selection_name(strategy);
                let BenchMeasurement {
                    pp,
                    tg,
                    contexts,
                    device,
                    profiles,
                    output_digest,
                    peak_footprint,
                    raw_samples,
                } = bench_engine(options, point, component, host_argmax, overlap).await?;
                if record_device.as_ref().is_some_and(|known| known != &device) {
                    return Err("benchmark engines opened different Metal devices".into());
                }
                record_device = Some(device.clone());
                print_summary(&engine, selection, "pp", pp);
                print_summary(&engine, selection, "tg", tg);
                for context in &contexts {
                    print_summary(
                        &engine,
                        selection,
                        &format!("tg@{}", context.context_start),
                        context.summary,
                    );
                }
                let profile_reports = report_profiles(&profiles, &engine, selection)?;
                let mut result = serde_json::json!({
                    "input": input,
                    "point": point.values,
                    "perf_hash": perf_hashes[point_index][input],
                    "output_digest": output_digest,
                    "peak_physical_footprint_bytes": peak_footprint,
                    "selection": selection,
                    "pp": summary_json(pp),
                    "tg": summary_json(tg),
                    "context_token_generation": contexts
                        .iter()
                        .map(|context| serde_json::json!({
                            "context_start": context.context_start,
                            "tg": summary_json(context.summary),
                        }))
                        .collect::<Vec<_>>(),
                    "batch_decode": batches.iter().map(batch_json).collect::<Vec<_>>(),
                    "replay_parameter_validation": replay_parameter_validation.clone(),
                    "sampling_fallbacks": profile_reports
                        .iter()
                        .filter_map(sampling_fallbacks_json)
                        .collect::<Vec<_>>(),
                    "engine.tunings": perf_keys[point_index][input]["engine.tunings"],
                    "engine.picks_hash": perf_keys[point_index][input]["engine.picks_hash"],
                });
                if !profile_reports.is_empty() {
                    result["breakdown"] = serde_json::Value::Array(
                        profile_reports.iter().map(profile_json).collect(),
                    );
                }
                if options.mode.raw_samples {
                    attach_raw_samples(&mut result, raw_samples, &batches);
                }
                results.push(result);
            }
        }
    }
    Ok((
        results,
        record_device.ok_or("benchmark produced no results")?,
    ))
}

fn report_profiles(
    profiles: &[ProfileMeasurement],
    engine: &str,
    selection: &str,
) -> Result<Vec<ProfileReport>, Box<dyn Error>> {
    let reports = profiles
        .iter()
        .map(profile_report)
        .collect::<Result<Vec<_>, _>>()?;
    for report in &reports {
        print_profile_report(engine, selection, report);
    }
    Ok(reports)
}

fn attach_raw_samples(
    result: &mut serde_json::Value,
    raw: serde_json::Value,
    batches: &[BatchDecodeMeasurement],
) {
    result["raw_samples"] = raw;
    result["raw_samples"]["batch_decode"] = serde_json::json!(
        batches
            .iter()
            .map(|batch| serde_json::json!({"bucket": batch.bucket, "samples": batch.samples}))
            .collect::<Vec<_>>()
    );
}

fn sampling_fallbacks_json(report: &ProfileReport) -> Option<serde_json::Value> {
    report
        .categories
        .iter()
        .find(|category| category.name == "sample.fallbacks")
        .map(|category| {
            serde_json::json!({
                "context_start": report.context_start,
                "count_per_token": stats_json(stats(
                    category.values.iter().map(|(count, _)| *count),
                )),
            })
        })
}

fn slim_record(report: &serde_json::Value) -> serde_json::Value {
    let mut slim = report.clone();
    if let Some(results) = slim["results"].as_array_mut() {
        for result in results {
            if let Some(result) = result.as_object_mut() {
                result.remove("breakdown");
            }
        }
    }
    slim
}

fn breakdown_record_path(record: &Path, scratch: &Path) -> Result<PathBuf, String> {
    let stem = record.file_stem().ok_or_else(|| {
        format!(
            "benchmark record path has no file name: {}",
            record.display()
        )
    })?;
    let mut name = OsString::from(stem);
    name.push(".breakdown.json");
    Ok(scratch.join(name))
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn sampling_params(point: &BenchPoint) -> SamplingParams {
    let sampling = point.config.bench.sampling;
    SamplingParams {
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        seed: sampling.seed,
    }
}

#[cfg(target_os = "macos")]
fn decode_options(point: &BenchPoint, host_argmax: bool, overlap: bool) -> DecodeOptions {
    DecodeOptions {
        host_argmax,
        overlap,
        sampling: sampling_params(point),
    }
}

#[cfg(target_os = "macos")]
async fn after_runner_drop<R, F, Fut>(runner: R, measure: F) -> Fut::Output
where
    F: FnOnce() -> Fut,
    Fut: Future,
{
    drop(runner);
    measure().await
}

#[cfg(target_os = "macos")]
async fn bench_engine(
    options: &Bench,
    point: &BenchPoint,
    component: &Path,
    host_argmax: bool,
    overlap: bool,
) -> Result<BenchMeasurement, Box<dyn Error>> {
    let decode = decode_options(point, host_argmax, overlap);
    let (mut runner, info, device) = load_benchmark_runner(options, point, component).await?;
    let max_context = usize::try_from(info.max_context)?;
    let profile_context = *point
        .contexts
        .last()
        .ok_or("no profile contexts configured")?;
    let tg_context_start = options
        .decode_prefill
        .checked_add(1)
        .ok_or("decode context length overflowed")?;
    if point.pp > max_context
        || tg_context_start
            .checked_add(point.tg)
            .ok_or("decode context length overflowed")?
            > max_context
        || profile_context
            .checked_add(point.tg)
            .ok_or("decode context length overflowed")?
            > max_context
    {
        return Err("benchmark shape exceeds the engine context".into());
    }
    let reserved_tokens = benchmark_reservation(point, options, tg_context_start, profile_context)?;
    prepare_benchmark_runner(&mut runner, reserved_tokens)?;
    let profile_tokens = profile_context - 1;
    let tokens = synthetic_tokens(
        point.pp.max(options.decode_prefill).max(profile_tokens),
        info.vocab,
    );
    for _ in 0..options.warmups {
        measure_prefill(&mut runner, &tokens[..point.pp]).await?;
        measure_decode(
            &mut runner,
            &tokens[..options.decode_prefill],
            point.tg,
            decode,
        )
        .await?;
    }
    let mut pp = Vec::with_capacity(options.reps);
    let mut tg = Vec::with_capacity(options.reps);
    for _ in 0..options.reps {
        pp.push(measure_prefill(&mut runner, &tokens[..point.pp]).await?);
        tg.push(
            measure_decode(
                &mut runner,
                &tokens[..options.decode_prefill],
                point.tg,
                decode,
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
            &point.contexts,
            decode,
        )
        .await?
    } else {
        Vec::new()
    };
    let (output_digest, peak_footprint) = benchmark_output(
        &mut runner,
        &tokens[..options.decode_prefill],
        point.tg,
        decode.sampling,
    )
    .await?;
    let contexts = after_runner_drop(runner, || {
        measure_record_contexts(&tokens, point, options, component, decode, &tg)
    })
    .await?;
    Ok(BenchMeasurement {
        pp: summarize(&pp, point.pp)?,
        tg: summarize(&tg, point.tg)?,
        raw_samples: serde_json::json!({"clock": "unix-end-with-monotonic-duration",
            "pp": pp, "tg": tg, "contexts": contexts.iter().map(|context|
                serde_json::json!({"context_start": context.context_start, "samples": context.samples})).collect::<Vec<_>>()}),
        contexts,
        device,
        profiles,
        output_digest,
        peak_footprint,
    })
}

fn benchmark_reservation(
    point: &BenchPoint,
    options: &Bench,
    tg_context_start: usize,
    profile_context: usize,
) -> Result<usize, Box<dyn Error>> {
    let decode = tg_context_start
        .checked_add(point.tg)
        .ok_or("decode context length overflowed")?;
    let mut reserved_tokens = point.pp.max(decode);
    if options.breakdown {
        let profile = profile_context
            .checked_add(2)
            .ok_or("profile context length overflowed")?;
        reserved_tokens = reserved_tokens.max(profile);
    }
    Ok(reserved_tokens)
}

#[cfg(target_os = "macos")]
async fn load_benchmark_runner(
    options: &Bench,
    point: &BenchPoint,
    component: &Path,
) -> Result<(EngineRunner<forja_metal::MetalBackend>, EngineInfo, String), Box<dyn Error>> {
    let (backend, live_bytes) =
        crate::engine::metal_backend(point.graph_replay, options.pool_capacity, &options.limits)?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        component,
        backend,
        crate::engine::limits_for_live_bytes(&options.limits, live_bytes)?,
        weights_path(&options.model_dir)?,
    )
    .await?;
    let info = runner.describe().await?;
    runner
        .load_with_selections(None, &engine_load_config(component, &point.config.engine)?)
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    Ok((runner, info, device))
}

fn context_measurements_enabled(rerun_schema: Option<u32>) -> bool {
    rerun_schema.is_none_or(|schema| schema >= 4)
}

fn batch_measurements_enabled(rerun_schema: Option<u32>) -> bool {
    rerun_schema.is_none_or(|schema| schema >= 5)
}

#[cfg(target_os = "macos")]
async fn measure_record_batches(
    options: &Bench,
    point: &BenchPoint,
    component: &Path,
) -> Result<Vec<BatchDecodeMeasurement>, Box<dyn Error>> {
    let (mut runner, info, _) = load_benchmark_runner(options, point, component).await?;
    let Some(serving) = info.serving.as_ref() else {
        return Ok(Vec::new());
    };
    let buckets = [1_u32, 2, 4, 8]
        .into_iter()
        .filter(|&bucket| {
            serving.batch_shapes.contains(&BatchShape {
                phase: BatchPhase::Decode,
                bucket,
                tokens: 1,
                selection: BatchSelection::Combined,
            })
        })
        .collect::<Vec<_>>();
    let Some(&largest) = buckets.last() else {
        return Ok(Vec::new());
    };
    let contexts = batch_contexts(usize::try_from(largest)?);
    let token_count = BATCH_CONTEXTS
        .len()
        .checked_mul(2_048)
        .ok_or("batch benchmark token count overflowed")?;
    let tokens = synthetic_tokens(token_count, info.vocab);
    let mut prefixes = Vec::with_capacity(BATCH_CONTEXTS.len());
    for (index, &context) in BATCH_CONTEXTS.iter().enumerate() {
        let start = index
            .checked_mul(2_048)
            .ok_or("batch benchmark token offset overflowed")?;
        let prompt_len = usize::try_from(
            context
                .checked_sub(1)
                .ok_or("batch benchmark context is empty")?,
        )?;
        let end = start
            .checked_add(prompt_len)
            .ok_or("batch benchmark token range overflowed")?;
        let reserved_tokens = u64::from(context)
            .checked_add(u64::try_from(point.tg)?)
            .ok_or("batch benchmark reservation overflowed")?;
        prefixes.push(
            prepare_batch_prefix(
                &mut runner,
                serving,
                tokens[start..end].to_vec(),
                reserved_tokens,
            )
            .await?,
        );
    }
    let feedback = synthetic_tokens(contexts.len(), info.vocab);
    let mut measurements = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        let capture_before = runner.metrics();
        let mut samples = Vec::with_capacity(options.reps);
        for repetition in 0..options.warmups.saturating_add(options.reps) {
            let sample = measure_batch_repetition(
                &mut runner,
                &prefixes,
                &contexts,
                &feedback,
                bucket,
                point.tg,
                sampling_params(point),
            )
            .await?;
            if repetition >= options.warmups {
                samples.push(sample);
            }
        }
        let aggregate_tokens = usize::try_from(bucket)?
            .checked_mul(point.tg)
            .ok_or("batch benchmark aggregate token count overflowed")?;
        let summary = summarize(&samples, aggregate_tokens)?;
        let steps = f64::from(u32::try_from(point.tg)?);
        let capture_after = runner.metrics();
        measurements.push(BatchDecodeMeasurement {
            bucket: usize::try_from(bucket)?,
            contexts: contexts[..usize::try_from(bucket)?].to_vec(),
            capture_graphs: capture_after
                .prepared_graphs
                .saturating_sub(capture_before.prepared_graphs),
            capture_seconds: capture_after
                .graph_preparation_time
                .saturating_sub(capture_before.graph_preparation_time)
                .as_secs_f64(),
            capture_validation_seconds: capture_after
                .graph_validation_time
                .saturating_sub(capture_before.graph_validation_time)
                .as_secs_f64(),
            summary,
            wall_itl: stats(samples.iter().map(|sample| sample.wall_seconds / steps)),
            gpu_itl: stats(samples.iter().map(|sample| sample.gpu_seconds / steps)),
            samples,
        });
    }
    Ok(measurements)
}

#[cfg(target_os = "macos")]
async fn prepare_batch_prefix(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    serving: &forja_host::ServingInfo,
    tokens: Vec<u32>,
    reserved_tokens: u64,
) -> Result<PagedPrefix, Box<dyn Error>> {
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
    runner.prepare_paged_lane(0, None, 0)?;
    runner.prepare_paged_lane(0, None, reserved_tokens)?;
    for (chunk, start) in tokens
        .chunks(usize::try_from(shape.tokens)?)
        .zip((0..).step_by(usize::try_from(shape.tokens)?))
    {
        let output = runner
            .batch_step(EngineBatch {
                shape,
                lanes: vec![Some(EngineBatchLane {
                    slot: 0,
                    tokens: Some(chunk.to_vec()),
                    start_pos: u32::try_from(start)?,
                    sampling: SamplingParams::default(),
                })],
            })
            .await?
            .map_err(|error| format!("batch benchmark prefix failed: {error:?}"))?;
        runner.read(&output.tokens).await?;
    }
    runner
        .paged_prefix()?
        .ok_or_else(|| "batch benchmark did not produce a paged prefix".into())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
async fn measure_batch_repetition(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prefixes: &[PagedPrefix],
    contexts: &[u32],
    feedback: &[u32],
    bucket: u32,
    steps: usize,
    sampling: SamplingParams,
) -> Result<Sample, Box<dyn Error>> {
    let active = usize::try_from(bucket)?;
    for lane in 0..active {
        let reserved = u64::from(contexts[lane])
            .checked_add(u64::try_from(steps)?)
            .ok_or("batch benchmark reservation overflowed")?;
        let prefix_tokens = contexts[lane]
            .checked_sub(1)
            .ok_or("batch benchmark context is empty")?;
        let prefix =
            prefixes[lane % prefixes.len()].fork_reserved(u64::from(prefix_tokens), reserved)?;
        runner.prepare_paged_lane(u32::try_from(lane)?, None, 0)?;
        runner.prepare_paged_lane(u32::try_from(lane)?, Some(prefix), reserved)?;
    }
    let mut lane_tokens = run_batch_decode_step(
        runner,
        contexts,
        &feedback[..active],
        bucket,
        0,
        true,
        sampling,
    )
    .await?;
    settle_submission_metrics(runner).await?;
    let before = runner.metrics();
    let started = Instant::now();
    for step in 0..u32::try_from(steps)? {
        lane_tokens = run_batch_decode_step(
            runner,
            contexts,
            &lane_tokens,
            bucket,
            step,
            false,
            sampling,
        )
        .await?;
    }
    settle_submission_metrics(runner).await?;
    sample(before, runner.metrics(), started.elapsed())
}

#[cfg(target_os = "macos")]
async fn run_batch_decode_step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    contexts: &[u32],
    tokens: &[u32],
    bucket: u32,
    offset: u32,
    seed: bool,
    sampling: SamplingParams,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let active = usize::try_from(bucket)?;
    let lanes = (0..active)
        .map(|lane| -> Result<_, Box<dyn Error>> {
            let start = contexts[lane]
                .checked_add(offset)
                .and_then(|position| position.checked_sub(u32::from(seed)))
                .ok_or("batch benchmark position overflowed")?;
            Ok(Some(EngineBatchLane {
                slot: u32::try_from(lane)?,
                tokens: Some(vec![tokens[lane]]),
                start_pos: start,
                sampling,
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                bucket,
                tokens: 1,
                selection: BatchSelection::Combined,
            },
            lanes,
        })
        .await?
        .map_err(|error| format!("batch benchmark decode failed: {error:?}"))?;
    decode_batch_tokens(&runner.read(&output.tokens).await?)
}

#[cfg(target_os = "macos")]
fn decode_batch_tokens(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    let (tokens, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("batch token output is not u32-aligned".into());
    }
    Ok(tokens.iter().copied().map(u32::from_le_bytes).collect())
}

fn batch_contexts(bucket: usize) -> Vec<u32> {
    BATCH_CONTEXTS.into_iter().cycle().take(bucket).collect()
}

fn batch_json(batch: &BatchDecodeMeasurement) -> serde_json::Value {
    serde_json::json!({
        "bucket": batch.bucket,
        "contexts": batch.contexts,
        "capture": {
            "graphs": batch.capture_graphs,
            "wall_seconds": batch.capture_seconds,
            "validation_seconds": batch.capture_validation_seconds,
        },
        "aggregate": summary_json(batch.summary),
        "lanes": batch.contexts.iter().enumerate().map(|(lane, context_start)| serde_json::json!({
            "lane": lane,
            "context_start": context_start,
            "itl_seconds": {
                "wall": stats_json(batch.wall_itl),
                "gpu": stats_json(batch.gpu_itl),
            },
        })).collect::<Vec<_>>(),
    })
}

#[cfg(target_os = "macos")]
async fn measure_record_contexts(
    tokens: &[u32],
    point: &BenchPoint,
    options: &Bench,
    component: &Path,
    decode: DecodeOptions,
    primary: &[Sample],
) -> Result<Vec<ContextMeasurement>, Box<dyn Error>> {
    let rerun_schema = options.rerun.as_ref().map(|record| record.schema_version);
    if context_measurements_enabled(rerun_schema) {
        measure_contexts(tokens, point, options, component, decode, primary).await
    } else {
        Ok(Vec::new())
    }
}

#[cfg(target_os = "macos")]
async fn measure_contexts(
    tokens: &[u32],
    point: &BenchPoint,
    options: &Bench,
    component: &Path,
    decode: DecodeOptions,
    primary: &[Sample],
) -> Result<Vec<ContextMeasurement>, Box<dyn Error>> {
    let primary_start = options.decode_prefill + 1;
    let mut measurements = Vec::with_capacity(point.contexts.len());
    for &context_start in &point.contexts {
        let mut samples = Vec::with_capacity(options.reps);
        if context_start == primary_start {
            samples.extend_from_slice(primary);
        } else {
            let (mut context_runner, _, _) =
                load_benchmark_runner(options, point, component).await?;
            prepare_benchmark_runner(
                &mut context_runner,
                context_start
                    .checked_add(point.tg)
                    .ok_or("decode context length overflowed")?,
            )?;
            let prompt = &tokens[..context_start - 1];
            for _ in 0..options.warmups {
                measure_decode(&mut context_runner, prompt, point.tg, decode).await?;
            }
            for _ in 0..options.reps {
                samples.push(measure_decode(&mut context_runner, prompt, point.tg, decode).await?);
            }
        }
        measurements.push(ContextMeasurement {
            context_start,
            summary: summarize(&samples, point.tg)?,
            samples,
        });
    }
    Ok(measurements)
}

#[cfg(target_os = "macos")]
async fn benchmark_output(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
    sampling: SamplingParams,
) -> Result<(String, u64), Box<dyn Error>> {
    let digest = probe_output(runner, prompt, steps, sampling).await?;
    let peak = runner
        .memory_usage()
        .peak_physical_footprint
        .ok_or("peak physical footprint is unavailable")?;
    Ok((digest, peak))
}

#[cfg(target_os = "macos")]
async fn probe_output(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
    sampling: SamplingParams,
) -> Result<String, Box<dyn Error>> {
    let mut tokens = Vec::with_capacity(steps);
    let mut logits = Vec::new();
    for index in 0..steps {
        let output = runner
            .decode(EngineDecode {
                tokens: (index == 0).then(|| prompt.to_vec()),
                start_pos: if index == 0 {
                    0
                } else {
                    u32::try_from(
                        prompt
                            .len()
                            .checked_add(index - 1)
                            .ok_or("decode position overflowed")?,
                    )?
                },
                sampling,
            })
            .await?
            .map_err(|error| format!("engine decode failed: {error:?}"))?;
        tokens.push(read_token(&runner.read(&output.token).await?)?);
        if index + 1 == steps {
            logits = runner.read(&output.logits).await?;
        }
    }
    Ok(benchmark_record::output_digest(&tokens, &logits))
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
    measure_prefill_from(runner, tokens, 0).await
}

#[cfg(target_os = "macos")]
async fn prepare_profile_prefix(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    prefix_context: usize,
) -> Result<(), Box<dyn Error>> {
    if prefix_context > 0 {
        step(runner, tokens[..prefix_context].to_vec(), 0).await?;
        runner.wait_for_queued_submissions().await?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
async fn measure_prefill_from(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    start: usize,
) -> Result<Sample, Box<dyn Error>> {
    let before = runner.metrics();
    let started = Instant::now();
    step(runner, tokens.to_vec(), u32::try_from(start)?).await?;
    runner.wait_for_queued_submissions().await?;
    sample(before, runner.metrics(), started.elapsed())
}

#[cfg(target_os = "macos")]
async fn measure_decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
    options: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    if options.overlap && !options.host_argmax {
        return measure_pipelined_decode(runner, prompt, steps, options.sampling).await;
    }
    let mut token = prepare_decode(runner, prompt, options).await?;
    let mut schedule = Vec::with_capacity(steps.saturating_add(1));
    visit_decode_steps(u32::try_from(prompt.len())?, steps, |position, timed| {
        schedule.push((position, timed));
    })?;
    let mut before = None;
    let mut started = None;
    for (position, timed) in schedule {
        let submission_before = runner.metrics().submissions;
        token = select_next(runner, token, position, options).await?;
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
    sampling: SamplingParams,
) -> Result<Sample, Box<dyn Error>> {
    let options = DecodeOptions {
        host_argmax: false,
        overlap: true,
        sampling,
    };
    let token = prepare_decode(runner, prompt, options).await?;
    let context_start = u32::try_from(prompt.len())?;
    let warmup_before = runner.metrics().submissions;
    select_next(runner, token, context_start, options).await?;
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
                        sampling,
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
async fn settle_submission_metrics(
    runner: &EngineRunner<forja_metal::MetalBackend>,
) -> Result<(), Box<dyn Error>> {
    runner.wait_for_queued_submissions().await?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn prepare_decode(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    options: DecodeOptions,
) -> Result<u32, Box<dyn Error>> {
    let token = select_from_tokens(runner, prompt.to_vec(), 0, options).await?;
    settle_submission_metrics(runner).await?;
    Ok(token)
}

#[cfg(target_os = "macos")]
async fn measure_profiles(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    reps: usize,
    warmups: usize,
    contexts: &[usize],
    options: DecodeOptions,
) -> Result<Vec<ProfileMeasurement>, Box<dyn Error>> {
    let mut measurements = Vec::new();
    for &context_start in contexts {
        let mut baseline = Vec::with_capacity(reps);
        let mut steps = Vec::with_capacity(reps);
        for repetition in 0..warmups.saturating_add(reps) {
            let unprofiled = if options.overlap && !options.host_argmax {
                prepare_profile_context(runner, tokens, context_start, options).await?;
                measure_pipelined_profile_step(runner, context_start, options.sampling).await?
            } else {
                let token = prepare_profile_context(runner, tokens, context_start, options).await?;
                measure_decode_step(runner, token, context_start, options).await?
            };
            let token = prepare_profile_context(runner, tokens, context_start, options).await?;
            runner.set_profiling(true);
            let profiled = measure_decode_step(runner, token, context_start, options).await;
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
    sampling: SamplingParams,
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
                    sampling,
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
        ..sample
    })
}

#[cfg(target_os = "macos")]
async fn prepare_profile_context(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    context_start: usize,
    options: DecodeOptions,
) -> Result<u32, Box<dyn Error>> {
    let prefill = context_start
        .checked_sub(1)
        .ok_or("profile context must follow a prefill token")?;
    let token = select_from_tokens(runner, tokens[..prefill].to_vec(), 0, options).await?;
    select_next(runner, token, u32::try_from(prefill)?, options).await
}

#[cfg(target_os = "macos")]
fn prepare_benchmark_runner(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    reserved_tokens: usize,
) -> Result<(), Box<dyn Error>> {
    runner
        .prepare_paged_request(None, u64::try_from(reserved_tokens)?)
        .map_err(|error| format!("paged benchmark admission failed: {error:?}"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
async fn measure_decode_step(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    token: u32,
    context_start: usize,
    options: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    settle_submission_metrics(runner).await?;
    let before = runner.metrics();
    let started = Instant::now();
    select_next(runner, token, u32::try_from(context_start)?, options).await?;
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
    options: DecodeOptions,
) -> Result<u32, Box<dyn Error>> {
    if options.host_argmax {
        return host_select(runner, tokens, start_pos).await;
    }
    let output = runner
        .decode(EngineDecode {
            tokens: Some(tokens),
            start_pos,
            sampling: options.sampling,
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
    options: DecodeOptions,
) -> Result<u32, Box<dyn Error>> {
    if options.host_argmax {
        return host_select(runner, vec![token], start_pos).await;
    }
    let output = runner
        .decode(EngineDecode {
            tokens: None,
            start_pos,
            sampling: options.sampling,
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
    let end = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs_f64();
    Ok(Sample {
        measurement_window: [end - wall_time.as_secs_f64(), end],
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
    let mut categories = vec![
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
        submission_category("sample.fallbacks", submissions, |submission| {
            (count_as_f64(submission.sample_fallbacks), Duration::ZERO)
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
    ];
    categories.extend(epilogue_profile_categories(submissions));
    categories
}

fn epilogue_profile_categories(
    submissions: &[&forja_core::SubmissionProfile],
) -> [ProfileCategory; 4] {
    [
        submission_category("epilogue.library-compilations", submissions, |submission| {
            (
                count_as_f64(submission.epilogue_library_compilations),
                Duration::ZERO,
            )
        }),
        submission_category(
            "epilogue.pipeline-specializations",
            submissions,
            |submission| {
                (
                    count_as_f64(submission.epilogue_pipeline_specializations),
                    Duration::ZERO,
                )
            },
        ),
        submission_category("epilogue.cache-hits", submissions, |submission| {
            (count_as_f64(submission.epilogue_cache_hits), Duration::ZERO)
        }),
        submission_category("epilogue.compile-fallbacks", submissions, |submission| {
            (
                count_as_f64(submission.epilogue_compile_fallbacks),
                Duration::ZERO,
            )
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
        Op::PagedCopy { .. } => "paged-copy",
        Op::RmsNorm { .. } => "rms_norm",
        Op::Argmax { .. } => "argmax",
        Op::TopK { .. } => "top-k",
        Op::Sample { .. } => "sample",
        Op::SampleMasked { .. } => "sample-masked",
        Op::Rope { .. } => "rope",
        Op::Embed => "embed",
        Op::QuantEmbed { .. } => "quant-embed",
        Op::Matmul => "matmul",
        Op::GatherMatmul => "gather-matmul",
        Op::QuantMatmul { .. } => "quant-matmul",
        Op::GatherQuantMatmul { .. } => "gather-quant-matmul",
        Op::Sdpa { .. } => "sdpa",
        Op::PagedSdpa { .. } => "paged-sdpa",
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

fn gpu_core_count() -> Result<u64, Box<dyn Error>> {
    let output = command_output("system_profiler", &["SPDisplaysDataType", "-json"])?;
    let report: serde_json::Value = serde_json::from_str(&output)?;
    report["SPDisplaysDataType"]
        .as_array()
        .and_then(|devices| devices.first())
        .and_then(|device| device["sppci_cores"].as_str())
        .ok_or_else(|| "system_profiler did not report GPU cores".into())
        .and_then(|cores| cores.parse().map_err(Into::into))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, env, fs, rc::Rc};

    use forja_testing::temporary_directory;

    use super::*;

    #[test]
    fn measures_empty_and_declared_bucket_parameter_counts() {
        let measured = measure_replay_parameter_validation(1).unwrap();
        let counts = measured
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["parameters"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(counts, [0, 4, 8]);
    }

    fn test_profile() -> forja_config::Profile {
        crate::resolution::read_embedded_profile(test_guests::qwen3()).unwrap()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn merges_serving_prefill_step_profiles() {
        let profiles = [2_u64, 3].map(|milliseconds| EngineStepProfile {
            wall_time: Duration::from_millis(milliseconds),
            guest_time: Duration::from_millis(milliseconds - 1),
            ..EngineStepProfile::default()
        });
        let merged = merge_step_profiles(profiles.into()).unwrap();
        assert_eq!(merged.wall_time, Duration::from_millis(5));
        assert_eq!(merged.guest_time, Duration::from_millis(3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn context_measurement_starts_after_primary_runner_drops() {
        struct DropFlag(Rc<Cell<bool>>);

        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let dropped = Rc::new(Cell::new(false));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let observed = runtime
            .block_on(after_runner_drop(DropFlag(Rc::clone(&dropped)), || async {
                dropped.get()
            }));
        assert!(observed);
    }

    #[test]
    fn decode_timing_excludes_one_step_after_prefill() -> Result<(), Box<dyn Error>> {
        let mut visited = Vec::new();
        visit_decode_steps(8, 7, |position, timed| visited.push((position, timed)))?;
        assert_eq!(visited.first(), Some(&(8, false)));
        assert_eq!(visited.iter().filter(|(_, timed)| *timed).count(), 7);
        assert_eq!(visited.last(), Some(&(15, true)));
        Ok(())
    }

    #[test]
    fn acceptance_materializes_effective_tunings_and_picks() {
        let profile = test_profile();
        let source = profile.canonical_toml().unwrap();
        let mut engine = forja_config::Engine::default();
        engine.tunings.base = forja_config::TuningBase::None;
        engine.tunings.add = vec!["final-norm".to_owned()];
        let expected = engine.resolve(&profile).unwrap();
        let accepted = accepted_profile(&source, &profile, &engine).unwrap();
        assert_eq!(accepted.default_tunings(), ["final-norm"]);
        assert_eq!(accepted.variants(), &expected.variants);
    }

    #[test]
    fn raw_samples_retain_rep_counts_and_measurement_windows() {
        let measured = Sample {
            wall_seconds: 2.,
            gpu_seconds: 1.,
            submissions: 1,
            measurement_window: [5., 7.],
        };
        let records = serde_json::to_value(vec![measured; 30]).unwrap();
        assert_eq!(records.as_array().unwrap().len(), 30);
        let window = &records[0]["measurement_window"];
        assert_eq!(
            window[1].as_f64().unwrap() - window[0].as_f64().unwrap(),
            2.0
        );
        assert_eq!(records[0]["wall_seconds"], 2.0);
    }

    #[test]
    fn atomic_writer_replaces_complete_files() {
        let root = temporary_directory("atomic-accept").unwrap();
        let path = root.join("record.json");
        fs::write(&path, b"old").unwrap();
        write_atomic(&path, b"new\n").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn strategy_axes_require_equal_outputs() {
        let axes = [forja_config::KeyPath::new("backend.metal.graph_replay")];
        let results = [
            serde_json::json!({
                "input": 0,
                "selection": "gpu-pipelined",
                "point": {"backend.metal.graph_replay": "tier1", "bench.tg": 7},
                "output_digest": "sha256:first",
            }),
            serde_json::json!({
                "input": 0,
                "selection": "gpu-pipelined",
                "point": {"backend.metal.graph_replay": "tier2", "bench.tg": 7},
                "output_digest": "sha256:second",
            }),
        ];
        let error = check_strategy_outputs(&axes, &results).unwrap_err();
        assert!(error.contains("strategy axis output mismatch"));
        assert!(error.contains("tier1") && error.contains("sha256:first"));
        assert!(error.contains("tier2") && error.contains("sha256:second"));
    }

    #[test]
    fn strategy_checks_separate_workloads_and_engines() {
        let axes = [forja_config::KeyPath::new("backend.metal.graph_replay")];
        let results = [
            serde_json::json!({
                "input": 0,
                "selection": "gpu-pipelined",
                "point": {"backend.metal.graph_replay": "tier1", "bench.tg": 7},
                "output_digest": "sha256:first",
            }),
            serde_json::json!({
                "input": 0,
                "selection": "gpu-pipelined",
                "point": {"backend.metal.graph_replay": "tier2", "bench.tg": 33},
                "output_digest": "sha256:second",
            }),
            serde_json::json!({
                "input": 1,
                "selection": "gpu-pipelined",
                "point": {"backend.metal.graph_replay": "tier2", "bench.tg": 7},
                "output_digest": "sha256:third",
            }),
        ];
        check_strategy_outputs(&axes, &results).unwrap();
    }

    #[test]
    fn rerun_refuses_input_hash_mismatches() {
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![Input {
                engine_sha256: "old-engine".to_owned(),
                engine_variant: "variant".to_owned(),
                engine_build_profile: "release".to_owned(),
                profile_name: String::new(),
                profile_family: String::new(),
                numerics: None,
                profile_hash: None,
                weights_sha256: "weights".to_owned(),
                model_revision: None,
            }],
            config: String::new(),
            axes: BTreeMap::new(),
            results: Vec::new(),
        };
        let current = [Input {
            engine_sha256: "new-engine".to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        }];
        assert!(
            validate_rerun_inputs(&record, &current, false)
                .unwrap_err()
                .contains("engine sha256 mismatch")
        );
    }

    #[test]
    fn rerun_accepts_reordered_inputs() {
        let input = |engine: &str| Input {
            engine_sha256: engine.to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![input("first"), input("second")],
            config: String::new(),
            axes: BTreeMap::new(),
            results: Vec::new(),
        };
        validate_rerun_inputs(&record, &[input("second"), input("first")], false).unwrap();
    }

    #[test]
    fn allow_diff_accepts_an_engine_revision_with_the_same_weights() {
        let input = |engine: &str, weights: &str| Input {
            engine_sha256: engine.to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: weights.to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![input("old-engine", "weights")],
            config: String::new(),
            axes: BTreeMap::new(),
            results: Vec::new(),
        };
        validate_rerun_inputs(&record, &[input("new-engine", "weights")], true).unwrap();
        let mut wrong_variant = input("new-engine", "weights");
        wrong_variant.engine_variant = "other".to_owned();
        assert!(
            validate_rerun_inputs(&record, &[wrong_variant], true)
                .unwrap_err()
                .contains("engine variant mismatch")
        );
        let error = validate_rerun_inputs(&record, &[input("new-engine", "other-weights")], true)
            .unwrap_err();
        assert!(error.contains("weights sha256 mismatch"));
    }

    #[test]
    fn comparability_accepts_reordered_engines() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            [
                "bench", "--engine", "/first", "--engine", "/second", "--model", "/model",
            ]
            .map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let input = |engine: &str| Input {
            engine_sha256: engine.to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let first = input("first");
        let second = input("second");
        let record = Recorded {
            schema_version: 2,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![first.clone(), second.clone()],
            config: benchmark_record::snapshot(&recorded_options, "device", "26.6")
                .unwrap()
                .config,
            axes: BTreeMap::new(),
            results: vec![
                serde_json::json!({"input": 0, "point": {}}),
                serde_json::json!({"input": 0, "point": {}}),
                serde_json::json!({"input": 1, "point": {}}),
                serde_json::json!({"input": 1, "point": {}}),
            ],
        };
        let crate::args::Command::Bench(current) = crate::args::parse(
            [
                "bench", "--engine", "/second", "--engine", "/first", "--model", "/model",
            ]
            .map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let keys = current
            .points
            .iter()
            .map(|point| {
                [&second, &first]
                    .into_iter()
                    .map(|input| benchmark_record::perf_key(point, input, &test_profile()).unwrap())
                    .collect()
            })
            .collect::<Vec<_>>();
        validate_comparability(&current, &record, &keys).unwrap();
    }

    #[test]
    fn allow_diff_overrides_a_structural_performance_difference() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            ["bench", "--engine", "/engine", "--model", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let input = Input {
            engine_sha256: "engine".to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: 2,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![input.clone()],
            config: benchmark_record::snapshot(&recorded_options, "device", "26.6")
                .unwrap()
                .config,
            axes: BTreeMap::new(),
            results: vec![
                serde_json::json!({"input": 0, "point": {}}),
                serde_json::json!({"input": 0, "point": {}}),
            ],
        };
        let crate::args::Command::Bench(mut current) = crate::args::parse(
            [
                "bench", "--engine", "/engine", "--model", "/model", "--reps", "7",
            ]
            .map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let keys = current
            .points
            .iter()
            .map(|point| vec![benchmark_record::perf_key(point, &input, &test_profile()).unwrap()])
            .collect::<Vec<_>>();
        let error = validate_comparability(&current, &record, &keys).unwrap_err();
        assert!(error.contains("bench.reps"));
        current.allow_diff.push(KeyPath::new("bench.reps"));
        validate_comparability(&current, &record, &keys).unwrap();
    }

    #[test]
    fn allow_diff_compares_performance_across_engine_revisions() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            ["bench", "--engine", "/engine", "--model", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let input = |engine: &str| Input {
            engine_sha256: engine.to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: 2,
            provenance: benchmark_record::RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![input("old-engine")],
            config: benchmark_record::snapshot(&recorded_options, "device", "26.6")
                .unwrap()
                .config,
            axes: BTreeMap::new(),
            results: vec![
                serde_json::json!({"input": 0, "point": {}}),
                serde_json::json!({"input": 0, "point": {}}),
            ],
        };
        let crate::args::Command::Bench(mut current) = crate::args::parse(
            ["bench", "--engine", "/engine", "--model", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let keys = current
            .points
            .iter()
            .map(|point| {
                vec![
                    benchmark_record::perf_key(point, &input("new-engine"), &test_profile())
                        .unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        assert!(
            validate_comparability(&current, &record, &keys)
                .unwrap_err()
                .contains("engine.sha256 differs")
        );
        current.allow_diff.push(KeyPath::new("engine.sha256"));
        validate_comparability(&current, &record, &keys).unwrap();
    }

    #[test]
    fn reads_result_confidence_intervals() {
        let result = serde_json::json!({
            "pp": {"tokens_per_second": {"wall": {"median": 10.0, "ci95": [9.0, 11.0]}}}
        });
        assert_eq!(wall_stats(&result, "pp").unwrap(), (10.0, 9.0, 11.0));
    }

    #[test]
    fn identifies_leave_one_out_and_from_default_tuning_changes() {
        let defaults = serde_json::json!({
            "point": {"engine.tunings": ["one", "two"]}
        });
        let without = serde_json::json!({
            "point": {"engine.tunings": ["one"]}
        });
        assert_eq!(
            tuning_difference(&defaults, &without).unwrap(),
            ("two".to_owned(), true)
        );

        let empty = serde_json::json!({"point": {"engine.tunings": []}});
        assert_eq!(
            tuning_difference(&empty, &without).unwrap(),
            ("one".to_owned(), false)
        );
        assert!((ratio_percent(110.0, 100.0) - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn slim_records_keep_results_and_sampling_fallbacks() {
        let report = serde_json::json!({
            "schema_version": 2,
            "provenance": {"commit": "abc"},
            "results": [{
                "input": 0,
                "output_digest": "sha256:output",
                "peak_physical_footprint_bytes": 4096,
                "pp": {"tokens_per_second": {"wall": {"median": 10.0}}},
                "sampling_fallbacks": [{
                    "context_start": 33,
                    "count_per_token": {"median": 1.0, "ci95": [1.0, 1.0]},
                }],
                "breakdown": [{"gpu_by_dispatch": [1, 2, 3]}],
            }],
        });
        let slim = slim_record(&report);
        assert_eq!(slim["schema_version"], 2);
        assert_eq!(slim["results"][0]["output_digest"], "sha256:output");
        assert_eq!(slim["results"][0]["peak_physical_footprint_bytes"], 4096);
        assert_eq!(
            slim["results"][0]["sampling_fallbacks"][0]["count_per_token"]["median"],
            1.0
        );
        assert!(slim["results"][0].get("breakdown").is_none());
        assert!(report["results"][0].get("breakdown").is_some());
    }

    #[test]
    fn puts_detailed_records_in_the_scratch_directory() {
        assert_eq!(
            breakdown_record_path(
                Path::new("bench/rejection-defaults.json"),
                Path::new("target/forja-bench"),
            )
            .unwrap(),
            Path::new("target/forja-bench/rejection-defaults.breakdown.json")
        );
    }

    #[test]
    fn legacy_reruns_keep_their_original_context_workload() {
        assert!(context_measurements_enabled(None));
        assert!(context_measurements_enabled(Some(
            benchmark_record::SCHEMA_VERSION
        )));
        assert!(context_measurements_enabled(Some(4)));
        assert!(!context_measurements_enabled(Some(3)));
    }

    #[test]
    fn batched_decode_alternates_short_and_long_contexts() {
        assert_eq!(batch_contexts(4), [512, 2_048, 512, 2_048]);
        assert!(batch_measurements_enabled(None));
        assert!(batch_measurements_enabled(Some(5)));
        assert!(!batch_measurements_enabled(Some(4)));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_vary_and_rerun_record() -> Result<(), Box<dyn Error>> {
        let models =
            std::path::PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let root = temporary_directory("bench-rerun")?;
        let record = root.join("record.json");
        let record_path = record.display().to_string();
        let engine = test_guests::qwen3().display().to_string();
        let model = models.join("Qwen3-0.6B").display().to_string();
        let crate::args::Command::Bench(first) = crate::args::parse(
            [
                "bench",
                "--engine",
                &engine,
                "--model",
                &model,
                "--pp",
                "9",
                "--tg",
                "2",
                "--reps",
                "1",
                "--set",
                "bench.warmups=0",
                "--set",
                "bench.selection=[\"gpu-pipelined\"]",
                "--vary",
                "backend.metal.graph_replay=tier1,tier2",
                "--json",
                &record_path,
            ]
            .map(str::to_owned),
        )?
        else {
            return Err("expected bench command".into());
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(run(&first))?;
        let crate::args::Command::Bench(second) = crate::args::parse(
            [
                "bench",
                "--engine",
                &engine,
                "--model",
                &model,
                "--rerun",
                &record_path,
            ]
            .map(str::to_owned),
        )?
        else {
            return Err("expected bench command".into());
        };
        runtime.block_on(run(&second))?;
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
