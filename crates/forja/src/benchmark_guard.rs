use forja_core::{BackendError, roofline::RateLimit};
use forja_metal::{ExecutionWork, MetalBackend};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    path::Path,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

const MARGIN: f64 = 0.10;
const CACHE_ALLOWANCE: u64 = 512 * 1024 * 1024;
static GUARD: OnceLock<Result<Guard, String>> = OnceLock::new();

struct Guard {
    memory: RateLimit,
    compute: RateLimit,
    record: Value,
    failure: OnceLock<BackendError>,
    missing_timestamps: AtomicU64,
}

impl Guard {
    fn load() -> Result<Self, Box<dyn Error>> {
        let identity = crate::calibration::identity()?;
        let path = crate::calibration::cache_path(None, &identity)?;
        read_cache(&path, &identity)
    }

    fn from_calibration(calibration: &Value) -> Result<Self, Box<dyn Error>> {
        let memory_peak = peak(calibration, &["streaming_read", "streaming_copy"])?;
        let arithmetic = calibration["arithmetic_probes"]
            .as_object()
            .ok_or("missing arithmetic calibration")?
            .values()
            .map(|value| value.as_str().ok_or("invalid arithmetic probe"))
            .collect::<Result<Vec<_>, _>>()?;
        let compute_peak = peak(calibration, &arithmetic)?;
        Ok(Self {
            memory: RateLimit::new("streaming input bytes", memory_peak, MARGIN)
                .ok_or("invalid bandwidth peak")?,
            compute: RateLimit::new("matrix FLOPs", compute_peak, MARGIN)
                .ok_or("invalid arithmetic peak")?,
            failure: OnceLock::new(),
            missing_timestamps: AtomicU64::new(0),
            record: json!({"margin_fraction": MARGIN, "memory_bytes_per_second": memory_peak,
                "matrix_flops_per_second": compute_peak, "cache_allowance_bytes": CACHE_ALLOWANCE,
                "basis": "per completed command buffer: known unconditional dense matrix FLOPs; largest contiguous matrix/copy/argmax input above 512 MiB, minus that cache allowance; unknown or overflowing contributions omitted",
                "peak_method": "fastest raw calibration observation; fastest supported arithmetic",
                "calibration_key": crate::calibration_identity::cache_key(&calibration["identity"]),
                "calibration_sha256": format!("sha256:{:x}", Sha256::digest(serde_json::to_vec(calibration)?)),
                "identity": calibration["identity"], "provenance": calibration["provenance"]}),
        })
    }

    fn check(
        &self,
        work: ExecutionWork,
        elapsed: impl Into<Option<Duration>>,
    ) -> Result<(), BackendError> {
        let Some(elapsed) = elapsed.into() else {
            let _ = self
                .missing_timestamps
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1));
            return Ok(());
        };
        self.memory
            .check(
                work.largest_input_bytes.saturating_sub(CACHE_ALLOWANCE),
                elapsed,
            )
            .and_then(|()| self.compute.check(work.matrix_flops, elapsed))
            .inspect_err(|error| {
                let _ = self.failure.set(*error);
            })
    }

    fn summary(&self) -> Result<Value, BackendError> {
        if let Some(error) = self.failure.get() {
            return Err(*error);
        }
        let mut record = self.record.clone();
        record["buffers_without_gpu_timestamps"] =
            json!(self.missing_timestamps.load(Ordering::Relaxed));
        record["counter_scope"] = json!("process-wide, including warmups");
        Ok(record)
    }
}

fn read_cache(path: &Path, identity: &Value) -> Result<Guard, Box<dyn Error>> {
    let load = || -> Result<Guard, Box<dyn Error>> {
        let calibration = serde_json::from_slice(&fs::read(path)?)?;
        crate::calibration::validate(&calibration, identity)?;
        Guard::from_calibration(&calibration)
    };
    load().map_err(|error| format!("cannot use calibration cache {}: {error}; run forja calibrate to create it, or forja calibrate --force to refresh it before starting benchmarks or workers", path.display()).into())
}

fn peak(record: &Value, names: &[&str]) -> Result<f64, Box<dyn Error>> {
    let mut peak = 0.0_f64;
    for &name in names {
        let samples = record["probes"][name]["samples"]
            .as_array()
            .filter(|samples| !samples.is_empty())
            .ok_or("missing peak samples")?;
        for sample in samples {
            let positive = |key| {
                sample[key]
                    .as_f64()
                    .filter(|n| n.is_finite() && *n > 0.0)
                    .ok_or("invalid peak observation")
            };
            let rate = positive("units")? / positive("gpu_seconds")?;
            if !rate.is_finite() {
                return Err("invalid calibrated peak".into());
            }
            peak = peak.max(rate);
        }
    }
    if peak <= 0.0 {
        return Err("invalid calibrated peak".into());
    }
    Ok(peak)
}

fn guard() -> Result<&'static Guard, Box<dyn Error>> {
    GUARD
        .get_or_init(|| Guard::load().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| error.clone().into())
}

pub(crate) fn require_calibration() -> Result<(), Box<dyn Error>> {
    guard()?;
    Ok(())
}

pub(crate) fn configure(backend: MetalBackend) -> Result<MetalBackend, Box<dyn Error>> {
    let guard = guard()?;
    Ok(backend.with_execution_validator(Arc::new(move |work, elapsed| guard.check(work, elapsed))))
}

pub(crate) fn record() -> Result<Value, Box<dyn Error>> {
    Ok(guard()?.summary()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> Value {
        json!({"identity":{"machine":"fixture"}, "provenance":{"synthetic_fixture":true},
            "arithmetic_probes":{"f32_native":"compute"}, "probes":{
                "streaming_read":{"samples":[{"units":CACHE_ALLOWANCE,"gpu_seconds":2.0}]},
                "streaming_copy":{"samples":[{"units":CACHE_ALLOWANCE,"gpu_seconds":1.0}]},
                "compute":{"samples":[{"units":100,"gpu_seconds":2.0},{"units":100,"gpu_seconds":1.0}]}}})
    }

    #[test]
    fn calibration_constructor_retains_peaks_and_provenance() {
        let input = calibration();
        let summary = Guard::from_calibration(&input).unwrap().summary().unwrap();
        assert_eq!(summary["memory_bytes_per_second"], json!(536_870_912.0));
        assert_eq!(summary["matrix_flops_per_second"], json!(100.0));
        assert_eq!(summary["identity"], input["identity"]);
        assert_eq!(summary["provenance"], input["provenance"]);
        assert_eq!(summary["cache_allowance_bytes"], CACHE_ALLOWANCE);
    }

    #[test]
    fn calibration_constructor_refuses_unknown_or_invalid_peaks() {
        for (pointer, value) in [
            ("/arithmetic_probes", json!({})),
            ("/probes/compute/samples/0/gpu_seconds", json!(0)),
            ("/probes/streaming_copy/samples", json!([])),
            ("/probes/compute/samples/0/units", json!(f64::MAX)),
        ] {
            let mut input = calibration();
            let overflow = value == json!(f64::MAX);
            *input.pointer_mut(pointer).unwrap() = value;
            if overflow {
                input["probes"]["compute"]["samples"][0]["gpu_seconds"] = json!(0.1);
            }
            assert!(Guard::from_calibration(&input).is_err(), "{pointer}");
        }
    }

    #[test]
    fn arithmetic_refusal_is_sticky_before_publication() {
        let guard = Guard::from_calibration(&calibration()).unwrap();
        guard
            .check(
                ExecutionWork {
                    matrix_flops: 110,
                    ..ExecutionWork::default()
                },
                Duration::from_secs(1),
            )
            .unwrap();
        let error = guard
            .check(
                ExecutionWork {
                    matrix_flops: 111,
                    ..ExecutionWork::default()
                },
                Duration::from_secs(1),
            )
            .unwrap_err();
        assert!(error.to_string().contains("forja calibrate --force"));
        assert!(guard.summary().is_err());
    }

    #[test]
    fn bandwidth_only_checks_inputs_above_the_cache_allowance() {
        let guard = Guard::from_calibration(&calibration()).unwrap();
        guard
            .check(
                ExecutionWork {
                    largest_input_bytes: CACHE_ALLOWANCE,
                    ..ExecutionWork::default()
                },
                Duration::ZERO,
            )
            .unwrap();
        assert!(
            guard
                .check(
                    ExecutionWork {
                        largest_input_bytes: CACHE_ALLOWANCE + 2,
                        ..ExecutionWork::default()
                    },
                    Duration::from_nanos(1)
                )
                .is_err()
        );
    }

    #[test]
    fn missing_timestamps_skip_validation_and_are_recorded() {
        let guard = Guard::from_calibration(&calibration()).unwrap();
        guard
            .check(
                ExecutionWork {
                    matrix_flops: u64::MAX,
                    largest_input_bytes: u64::MAX,
                },
                None,
            )
            .unwrap();
        assert_eq!(
            guard.summary().unwrap()["buffers_without_gpu_timestamps"],
            1
        );
    }

    #[test]
    fn cache_failures_name_the_path_and_calibration_remedy() {
        let root = std::env::temp_dir().join(format!("forja-guard-cache-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("missing.json");
        for contents in [
            None,
            Some("not JSON"),
            Some(r#"{"schema_version":10,"identity":{"other":true}}"#),
        ] {
            if let Some(contents) = contents {
                fs::write(&path, contents).unwrap();
            }
            let error = read_cache(&path, &json!({"machine":"fixture"}))
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains(path.to_str().unwrap()));
            assert!(error.contains("run forja calibrate"));
            assert!(error.contains("forja calibrate --force"));
        }
        fs::remove_dir_all(root).unwrap();
    }
}
