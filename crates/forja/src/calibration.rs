#[cfg(target_os = "macos")]
use crate::calibration_identity::cache_key;
#[cfg(target_os = "macos")]
pub(crate) use crate::calibration_identity::identity;
use serde_json::{Value, json};
use std::{error::Error, fs, path::PathBuf};

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Directory containing calibration records keyed by device identity.
    #[arg(long)]
    pub(crate) cache: Option<PathBuf>,
    /// Re-measure even when a matching calibration exists.
    #[arg(long)]
    pub(crate) force: bool,
    /// Measured observations per probe (at least six for a 95% median interval).
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u32).range(6..=10000))]
    pub(crate) reps: u32,
    /// Untimed observations per probe.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(0..=10000))]
    pub(crate) warmups: u32,
    /// Also save the complete calibration to this path.
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
}

pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    let record = load_or_measure(options)?;
    if let Some(path) = &options.json {
        fs::write(path, serde_json::to_vec_pretty(&record)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}

pub(crate) fn load_or_measure(options: &Options) -> Result<Value, Box<dyn Error>> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = options;
        Err("calibration requires Metal on macOS".into())
    }
    #[cfg(target_os = "macos")]
    {
        let identity = identity()?;
        let root = options
            .cache
            .clone()
            .or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(PathBuf::from)
                    .map(|path| path.join("forja/calibration"))
            })
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|path| path.join(".cache/forja/calibration"))
            })
            .ok_or("calibration cache path unavailable")?;
        fs::create_dir_all(&root)?;
        let path = root.join(format!("{}.json", cache_key(&identity)));
        if !options.force && path.exists() {
            let record: Value = serde_json::from_slice(&fs::read(&path)?)?;
            validate(&record, &identity)?;
            return Ok(record);
        }
        let record = measure(options, &identity)?;
        validate(&record, &identity)?;
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
        fs::rename(temporary, path)?;
        Ok(record)
    }
}

const PROBES: &[&str] = &[
    "streaming_read",
    "streaming_copy",
    "calibration_mma_f32",
    "calibration_mma_f16",
    "calibration_mma_bf16",
    "calibration_threadgroup",
    "dependent_dispatch",
];

#[cfg(target_os = "macos")]
fn measure(options: &Options, identity: &Value) -> Result<Value, Box<dyn Error>> {
    let load_before = crate::machine_load::capture("BEFORE CALIBRATION")?;
    let backend = forja_metal::MetalBackend::new()?;
    let mut samples = backend.calibrate_memory(options.warmups, options.reps)?;
    samples.extend(backend.calibrate_compute(options.warmups, options.reps)?);
    samples.extend(backend.calibrate_dispatch(options.warmups, options.reps)?);
    let mut probes = serde_json::Map::new();
    for &name in PROBES {
        let selected = samples
            .iter()
            .filter(|sample| sample.name == name)
            .collect::<Vec<_>>();
        #[allow(clippy::cast_precision_loss)]
        let rates = crate::benchmark_stats::stats(
            selected
                .iter()
                .map(|sample| sample.units as f64 / sample.gpu.as_secs_f64()),
        );
        let arithmetic = selected
            .first()
            .and_then(|sample| sample.arithmetic)
            .map(|dtype| match dtype {
                forja_core::DType::F16 => "f16",
                forja_core::DType::BF16 => "bf16",
                _ => "f32",
            });
        probes.insert(name.into(), json!({"arithmetic": arithmetic, "units_per_second": rates.median,
            "ci95": [rates.low, rates.high], "samples": selected.iter().map(|sample| json!({
                "units": sample.units, "wall_seconds": sample.wall.as_secs_f64(), "gpu_seconds": sample.gpu.as_secs_f64()})).collect::<Vec<_>>()}));
    }
    Ok(
        json!({"schema_version": 2, "kind": "device-calibration", "model_benchmark": false,
        "identity": identity, "warmups": options.warmups, "repetitions": options.reps,
        "timestamp_method": "unprofiled-command-buffer", "interval_method": "binomial-order-statistic-median",
        "bf16_arithmetic": if samples.iter().any(|sample| sample.arithmetic == Some(forja_core::DType::BF16)) { "native-bf16" } else { "compiler refused native bf16 MMA; operands promoted to f32" },
        "threadgroup_method": "volatile float4 reads, 128 threads, one initialization barrier",
        "mma_method": "8x8x8 SIMD matrices; four independent accumulators per SIMD group",
        "probes": probes, "provenance": {"commit": crate::provenance::commit(),
            "dirty": crate::provenance::dirty(), "build_environment": crate::provenance::BUILD_ENVIRONMENT,
            "binary_sha256": golden_fixtures::sha256_file(std::env::current_exe()?)?,
            "unix_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),
            "machine_load": {"before": load_before, "after": crate::machine_load::capture("AFTER CALIBRATION")?}}}),
    )
}

pub(crate) fn validate(record: &Value, identity: &Value) -> Result<(), Box<dyn Error>> {
    if record["schema_version"] != 2 || record["identity"] != *identity {
        return Err("calibration identity or schema mismatch; recalibrate".into());
    }
    for &name in PROBES {
        let probe = &record["probes"][name];
        for value in [
            &probe["units_per_second"],
            &probe["ci95"][0],
            &probe["ci95"][1],
        ] {
            if !value
                .as_f64()
                .is_some_and(|value| value.is_finite() && value > 0.0)
            {
                return Err(format!("invalid calibration ceiling: {name}").into());
            }
        }
        validate_samples(
            probe,
            record["repetitions"]
                .as_u64()
                .ok_or("missing calibration repetitions")?,
        )?;
    }
    Ok(())
}

fn validate_samples(probe: &Value, repetitions: u64) -> Result<(), Box<dyn Error>> {
    let samples = probe["samples"]
        .as_array()
        .ok_or("missing calibration samples")?;
    if repetitions < 6 || u64::try_from(samples.len())? != repetitions {
        return Err("incomplete calibration samples".into());
    }
    let mut rates = Vec::new();
    for sample in samples {
        let positive = |name| {
            sample[name]
                .as_f64()
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or("invalid calibration sample")
        };
        let units = positive("units")?;
        positive("wall_seconds")?;
        rates.push(units / positive("gpu_seconds")?);
    }
    let expected = crate::benchmark_stats::stats(rates.into_iter());
    for (actual, expected) in [
        (&probe["units_per_second"], expected.median),
        (&probe["ci95"][0], expected.low),
        (&probe["ci95"][1], expected.high),
    ] {
        if actual
            .as_f64()
            .is_none_or(|actual| (actual - expected).abs() > expected * 1e-12)
        {
            return Err("calibration summary disagrees with raw samples".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn warmups_are_bounded_and_allow_zero() {
        use clap::Parser;
        #[derive(Parser)]
        struct Command {
            #[command(flatten)]
            options: Options,
        }
        for count in ["0", "10000"] {
            assert!(Command::try_parse_from(["calibrate", "--warmups", count]).is_ok());
        }
        assert!(Command::try_parse_from(["calibrate", "--warmups", "10001"]).is_err());
    }

    #[test]
    fn cached_ceilings_require_matching_raw_samples() {
        let samples = (1..=9)
            .map(|time| json!({"units": 1, "wall_seconds": time, "gpu_seconds": time}))
            .collect::<Vec<_>>();
        let summary = crate::benchmark_stats::stats((1..=9).map(|time| 1.0 / f64::from(time)));
        let probe = json!({"units_per_second": summary.median, "ci95": [summary.low, summary.high], "samples": samples});
        assert!(validate_samples(&probe, 9).is_ok());
        assert!(validate_samples(&probe, 8).is_err());
        let mut invalid = probe.clone();
        invalid["ci95"][0] = json!(99);
        assert!(validate_samples(&invalid, 9).is_err());
        invalid = probe.clone();
        invalid["samples"][0]["gpu_seconds"] = json!(0);
        assert!(validate_samples(&invalid, 9).is_err());
    }
}
