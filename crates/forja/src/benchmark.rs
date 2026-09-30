use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use forja_config::{KeyPath, Selection};
use forja_core::Op;
use forja_host::{
    EngineDecode, EngineMetrics, EngineOutput, EngineRunner, EngineStep, EngineStepProfile,
    ImportProfile, SamplingParams,
};
use golden_fixtures::{decode_f32_le, sha256_file};

use crate::{
    args::{Bench, BenchPoint, Profile, ProfileMode},
    benchmark_record::{self, Input, PerfKey, Recorded},
    benchmark_stats::{Stats, stats, synthetic_tokens},
    engine::{argmax, limits, read_token, weights_path},
    machine_load,
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

pub(crate) struct TokenProfile {
    pub(crate) device: String,
    pub(crate) baseline_wall: Duration,
    pub(crate) baseline_gpu: Duration,
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
pub(crate) async fn measure_token_profile(
    options: &Profile,
) -> Result<TokenProfile, Box<dyn Error>> {
    let backend =
        forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(options.graph_replay))?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        &options.engine,
        backend,
        limits(&options.limits)?,
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
        .load()
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
    for _ in 0..options.warmups {
        measure_profile_phase(&mut runner, &tokens, options, decode).await?;
    }
    let baseline = measure_profile_phase(&mut runner, &tokens, options, decode).await?;
    runner.set_profiling(true);
    let measured = measure_profile_phase(&mut runner, &tokens, options, decode).await;
    runner.set_profiling(false);
    let _ = measured?;
    Ok(TokenProfile {
        device,
        baseline_wall: Duration::from_secs_f64(baseline.wall_seconds),
        baseline_gpu: Duration::from_secs_f64(baseline.gpu_seconds),
        step: runner
            .take_profile()
            .ok_or("profiled step produced no timing detail")?,
    })
}

#[cfg(target_os = "macos")]
async fn measure_profile_phase(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    tokens: &[u32],
    options: &Profile,
    decode: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    match options.mode {
        ProfileMode::Prefill => measure_prefill(runner, tokens).await,
        ProfileMode::Decode => {
            let token = prepare_profile_context(runner, tokens, options.context, decode).await?;
            measure_decode_step(runner, token, options.context, decode).await
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_lines)]
async fn run_metal(options: &Bench) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let load_before = machine_load::capture("BEFORE BENCHMARK")?;
    let commit = crate::provenance::commit();
    let os = command_output("sw_vers", &["-productVersion"])?;
    let date = command_output("date", &["-u", "+%Y-%m-%dT%H:%MZ"])?;
    let gpu_cores = gpu_core_count()?;
    let binary_sha256 = sha256_file(std::env::current_exe()?)?;
    let weights_sha256 = sha256_file(weights_path(&options.model_dir)?)?;
    let inputs = options
        .engines
        .iter()
        .map(|component| {
            Ok(Input {
                engine_sha256: sha256_file(component)?,
                engine_variant: component
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                engine_build_profile: engine_build_profile(component).to_owned(),
                profile_hash: None,
                weights_sha256: weights_sha256.clone(),
                model_revision: None,
            })
        })
        .collect::<Result<Vec<_>, std::io::Error>>()?;
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
                .map(|input| benchmark_record::perf_key(point, input))
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
    let (results, device) = measure_points(options, &perf_hashes).await?;
    let load_after = machine_load::capture("AFTER BENCHMARK")?;
    check_strategy_outputs(&options.strategy_axes, &results)?;
    if let Some(record) = &options.rerun {
        print_result_diff(record, &results)?;
    }
    if let Some(path) = &options.json {
        let snapshot = benchmark_record::snapshot(options, &device, &os)?;
        let report = serde_json::json!({
            "schema_version": benchmark_record::SCHEMA_VERSION,
            "implementation": "forja",
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
        if !recorded.is_empty() && recorded != current {
            return Err(format!(
                "rerun engine {name} mismatch: recorded {recorded}, current {current}"
            ));
        }
    }
    Ok(())
}

fn engine_build_profile(component: &Path) -> &'static str {
    let generated_release = component.ancestors().any(|path| {
        path.file_name() == Some(std::ffi::OsStr::new("out"))
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name.to_string_lossy().starts_with("test-guests-"))
    });
    if generated_release
        || component
            .components()
            .any(|part| part.as_os_str() == "release")
    {
        "release"
    } else if component
        .components()
        .any(|part| part.as_os_str() == "debug")
    {
        "debug"
    } else {
        "unknown"
    }
}

fn validate_comparability(
    options: &Bench,
    record: &Recorded,
    perf_keys: &[Vec<PerfKey>],
) -> Result<(), String> {
    let recorded = benchmark_record::recorded_perf_keys(record)?;
    let current = options
        .points
        .iter()
        .zip(perf_keys)
        .flat_map(|(point, keys)| {
            keys.iter()
                .flat_map(|key| std::iter::repeat_n(key.clone(), point.selection.len()))
        })
        .collect::<Vec<_>>();
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
    perf_hashes: &[Vec<String>],
) -> Result<(Vec<serde_json::Value>, String), Box<dyn Error>> {
    let mut results = Vec::new();
    let mut record_device = None;
    println!(
        "engine\tselection\tmetric\twall tok/s (95% CI)\tGPU tok/s (95% CI)\twall ms\tGPU ms\tsubmissions"
    );
    for (point_index, point) in options.points.iter().enumerate() {
        for (input, component) in options.engines.iter().enumerate() {
            let engine = component.display().to_string();
            for &strategy in &point.selection {
                let (host_argmax, overlap) = selection_mode(strategy);
                let selection = selection_name(strategy);
                let (pp, tg, device, profiles, output_digest) =
                    bench_engine(options, point, component, host_argmax, overlap).await?;
                if record_device.as_ref().is_some_and(|known| known != &device) {
                    return Err("benchmark engines opened different Metal devices".into());
                }
                record_device = Some(device.clone());
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
                    "input": input,
                    "point": point.values,
                    "perf_hash": perf_hashes[point_index][input],
                    "output_digest": output_digest,
                    "selection": selection,
                    "pp": summary_json(pp),
                    "tg": summary_json(tg),
                    "sampling_fallbacks": profile_reports
                        .iter()
                        .filter_map(sampling_fallbacks_json)
                        .collect::<Vec<_>>(),
                });
                if !profile_reports.is_empty() {
                    result["breakdown"] = serde_json::Value::Array(
                        profile_reports.iter().map(profile_json).collect(),
                    );
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
async fn bench_engine(
    options: &Bench,
    point: &BenchPoint,
    component: &Path,
    host_argmax: bool,
    overlap: bool,
) -> Result<(Summary, Summary, String, Vec<ProfileMeasurement>, String), Box<dyn Error>> {
    let decode = DecodeOptions {
        host_argmax,
        overlap,
        sampling: sampling_params(point),
    };
    let backend =
        forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(point.graph_replay))?;
    let device = backend.device_name();
    let mut runner = EngineRunner::new(
        component,
        backend,
        limits(&options.limits)?,
        weights_path(&options.model_dir)?,
    )
    .await?;
    let info = runner.describe().await?;
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
    let output_digest = probe_output(
        &mut runner,
        &tokens[..options.decode_prefill],
        point.tg,
        decode.sampling,
    )
    .await?;
    Ok((
        summarize(&pp, point.pp)?,
        summarize(&tg, point.tg)?,
        device,
        profiles,
        output_digest,
    ))
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
    options: DecodeOptions,
) -> Result<Sample, Box<dyn Error>> {
    if options.overlap && !options.host_argmax {
        return measure_pipelined_decode(runner, prompt, steps, options.sampling).await;
    }
    let mut token = select_from_tokens(runner, prompt.to_vec(), 0, options).await?;
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
    let token = select_from_tokens(runner, prompt.to_vec(), 0, options).await?;
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
        Op::TopK { .. } => "top-k",
        Op::Sample { .. } => "sample",
        Op::Rope { .. } => "rope",
        Op::Embed => "embed",
        Op::QuantEmbed { .. } => "quant-embed",
        Op::Matmul => "matmul",
        Op::GatherMatmul => "gather-matmul",
        Op::QuantMatmul { .. } => "quant-matmul",
        Op::GatherQuantMatmul { .. } => "gather-quant-matmul",
        Op::GatherQuantSiluMul { .. } => "gather-quant-silu-mul",
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
    use std::{env, fs};

    use forja_testing::temporary_directory;

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
    fn identifies_generated_guest_builds_as_release() {
        assert_eq!(
            engine_build_profile(Path::new(
                "/target/debug/build/test-guests-hash/out/qwen3-bf16.wasm"
            )),
            "release"
        );
        assert_eq!(
            engine_build_profile(Path::new("/target/debug/custom.wasm")),
            "debug"
        );
    }

    #[test]
    fn comparability_accepts_reordered_engines() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            [
                "bench",
                "--engine",
                "/first",
                "--engine",
                "/second",
                "--model-dir",
                "/model",
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
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let first = input("first");
        let second = input("second");
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
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
                "bench",
                "--engine",
                "/second",
                "--engine",
                "/first",
                "--model-dir",
                "/model",
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
                    .map(|input| benchmark_record::perf_key(point, input).unwrap())
                    .collect()
            })
            .collect::<Vec<_>>();
        validate_comparability(&current, &record, &keys).unwrap();
    }

    #[test]
    fn allow_diff_overrides_a_structural_performance_difference() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            ["bench", "--engine", "/engine", "--model-dir", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let input = Input {
            engine_sha256: "engine".to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
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
                "bench",
                "--engine",
                "/engine",
                "--model-dir",
                "/model",
                "--reps",
                "7",
            ]
            .map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let keys = current
            .points
            .iter()
            .map(|point| vec![benchmark_record::perf_key(point, &input).unwrap()])
            .collect::<Vec<_>>();
        let error = validate_comparability(&current, &record, &keys).unwrap_err();
        assert!(error.contains("bench.reps"));
        current.allow_diff.push(KeyPath::new("bench.reps"));
        validate_comparability(&current, &record, &keys).unwrap();
    }

    #[test]
    fn allow_diff_compares_performance_across_engine_revisions() {
        let crate::args::Command::Bench(recorded_options) = crate::args::parse(
            ["bench", "--engine", "/engine", "--model-dir", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let input = |engine: &str| Input {
            engine_sha256: engine.to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let record = Recorded {
            schema_version: benchmark_record::SCHEMA_VERSION,
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
            ["bench", "--engine", "/engine", "--model-dir", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected bench command");
        };
        let keys = current
            .points
            .iter()
            .map(|point| vec![benchmark_record::perf_key(point, &input("new-engine")).unwrap()])
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
    fn slim_records_keep_results_and_sampling_fallbacks() {
        let report = serde_json::json!({
            "schema_version": 2,
            "provenance": {"commit": "abc"},
            "results": [{
                "input": 0,
                "output_digest": "sha256:output",
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
                "--model-dir",
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
                "--model-dir",
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
