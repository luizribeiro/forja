use crate::{
    analysis_input::{Profile, positive},
    analysis_metrics::{Attribution, ceilings},
};
#[cfg(test)]
use forja_core::KernelCost;
use serde_json::{Value, json};
use std::error::Error;

#[path = "analysis/model.rs"]
mod model;
#[path = "analysis/ranking.rs"]
pub(super) mod ranking;

pub(crate) fn report(
    profile: &Profile,
    calibration: &Value,
    target: f64,
) -> Result<Value, Box<dyn Error>> {
    let profiled_gpu = positive(
        profile
            .submissions
            .iter()
            .map(|submission| submission.gpu_time_seconds)
            .sum(),
    )?;
    let scale = positive(profile.unprofiled.gpu_time_seconds / profiled_gpu)?;
    let mut dispatches = Vec::new();
    let mut kernels = Vec::new();
    for submission in &profile.submissions {
        for dispatch in &submission.timeline {
            let ceiling = ceilings(calibration, &dispatch.compute_dtype)?;
            let attribution = Attribution {
                scale,
                wall: profile.unprofiled.wall_time_seconds,
                target,
                ceilings: ceiling,
            };
            let mut outer = attribution.row(
                dispatch,
                dispatch.cost,
                &dispatch.kernel,
                dispatch.gpu_time_seconds,
                dispatch.gap_before_seconds,
                dispatch.physical_dispatches,
            )?;
            outer["submission"] = json!(submission.index);
            outer["dispatch"] = json!(dispatch.index);
            outer["phase"] = submission.phase.clone();
            outer["padded_flops"] = json!(dispatch.padded_flops);
            dispatches.push(outer.clone());
            if dispatch.internal.is_empty() {
                kernels.push(outer);
            } else {
                for (index, kernel) in dispatch.internal.iter().enumerate() {
                    let mut inner = attribution.row(
                        dispatch,
                        kernel.cost,
                        &kernel.kernel,
                        kernel.gpu_time_seconds,
                        if index == 0 {
                            dispatch.gap_before_seconds
                        } else {
                            0.0
                        },
                        1,
                    )?;
                    inner["submission"] = json!(submission.index);
                    inner["dispatch"] = json!(dispatch.index);
                    inner["phase"] = submission.phase.clone();
                    kernels.push(inner);
                }
            }
        }
    }
    sort(&mut dispatches);
    sort(&mut kernels);
    Ok(
        json!({"schema_version": 1, "kind": "roofline-analysis", "model_benchmark": false,
        "mode": profile.mode, "tokens": profile.tokens, "device_identity": profile.device_identity,
        "inputs": profile.inputs, "profile_provenance": profile.provenance, "calibration": calibration,
        "unprofiled": {"wall_seconds": profile.unprofiled.wall_time_seconds, "gpu_seconds": profile.unprofiled.gpu_time_seconds},
        "profiled_gpu_seconds": profiled_gpu, "timestamp_perturbation_ratio": profiled_gpu / profile.unprofiled.gpu_time_seconds,
        "attribution_scale": scale, "target_utilization": target,
        "notes": ["Profile-derived diagnostic, not a model benchmark.",
            "Attribution scales precise intervals plus preceding gaps to unprofiled GPU time; per-kernel perturbation is unquantified.",
            "Traffic is logical, not a hardware DRAM counter; expert reuse is an interval unless proven by the command graph.",
            "Scalar/SFU and integer throughput are uncalibrated; MMA percentages use matrix-product FLOPs only.",
            "Alternatives preserve the selected reduction class and batch-invariance guarantee; screen before adoption."],
        "whole_model": model::summaries(profile, calibration, scale, target)?,
        "by_kernel": ranking::aggregate(&kernels, "kernel"), "ranked_call_sites": ranking::aggregate(&dispatches, "site_group"),
        "dispatches": dispatches, "kernels": kernels }),
    )
}

pub(super) fn sort(rows: &mut [Value]) {
    rows.sort_by(|left, right| {
        right["recoverable_seconds"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&left["recoverable_seconds"].as_f64().unwrap_or(0.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Profile, Value) {
        let cost = |read, flops| {
            KernelCost {
                read_bytes: read,
                flops,
                ..KernelCost::default()
            }
            .json()
        };
        let dispatch = |index, name, read, flops, internal| {
            json!({"index": index, "call_site": format!("block/layer-0/dispatch-{index}"),
            "scope": {"site": "block", "layer_type": "test", "layer": 0}, "shape": "test", "op": "test", "kernel": name,
            "variant": "test.current", "alternatives": ["test.alternative"], "gpu_time_seconds": 4.0, "gap_before_seconds": 0.0,
            "physical_dispatches": if index == 0 {2} else {1}, "compute_dtype": "f32", "cost": cost(read, flops), "padded_flops": [flops, flops], "internal": internal })
        };
        let internal = json!([{"kernel": "read-a", "gpu_time_seconds": 1.5, "cost": cost(40, 0)}, {"kernel": "read-b", "gpu_time_seconds": 2.5, "cost": cost(60, 0)}]);
        let timeline = json!([
            dispatch(0, "composite", 100, 0, internal),
            dispatch(1, "compute", 0, 500, json!([]))
        ]);
        let submissions = [0, 1].map(|index| json!({"index": index, "phase": {"context": (index+1)*33}, "gpu_time_seconds": 10.0, "timeline": timeline}));
        let profile = serde_json::from_value(json!({"schema_version": 2, "device_identity": {}, "provenance": {}, "inputs": {},
            "mode": "prefill", "tokens": 33, "unprofiled": {"wall_time_seconds": 12.0, "gpu_time_seconds": 10.0},
            "submissions": submissions })).unwrap();
        let mut probes = json!({});
        for (name, rate) in [
            ("streaming_read", 100.0),
            ("streaming_copy", 200.0),
            ("calibration_mma_f32", 1000.0),
            ("dependent_dispatch", 100.0),
        ] {
            probes[name] = json!({"units_per_second": rate});
        }
        (profile, json!({"probes": probes}))
    }

    #[test]
    fn joins_all_submissions_without_double_counting_composites() {
        let (profile, calibration) = fixture();
        profile.validate().unwrap();
        let report = report(&profile, &calibration, 0.8).unwrap();
        assert_eq!(report["dispatches"].as_array().unwrap().len(), 4);
        assert_eq!(report["kernels"].as_array().unwrap().len(), 6);
        assert_eq!(report["dispatches"][0]["kernel"], "compute");
        let total = |key: &str| {
            report[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["attributed_gpu_seconds"].as_f64().unwrap())
                .sum::<f64>()
        };
        assert!((total("dispatches") - total("kernels")).abs() < f64::EPSILON);
        assert!(
            report["kernels"]
                .as_array()
                .unwrap()
                .iter()
                .all(|row| row["kernel"] != "composite")
        );
        assert_eq!(report["model_benchmark"], false);
        let mut invalid = profile;
        invalid.schema_version = 1;
        assert!(invalid.validate().is_err());
    }
}
