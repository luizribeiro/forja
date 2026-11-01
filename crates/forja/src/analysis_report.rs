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
    let mut unmeasured = Vec::new();
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
                    let gap = if index == 0 {
                        dispatch.gap_before_seconds
                    } else {
                        0.0
                    };
                    let Some(cost) = kernel.cost else {
                        let attributed = positive((kernel.gpu_time_seconds + gap) * scale)?;
                        unmeasured.push(json!({"submission": submission.index, "dispatch": dispatch.index, "phase": submission.phase,
                            "call_site": dispatch.call_site, "kernel": kernel.kernel, "variant": dispatch.variant,
                            "cost": null, "ranked": false, "reason": "unknown or non-integral internal cost",
                            "profiled_gpu_seconds": kernel.gpu_time_seconds, "attributed_gpu_seconds": attributed,
                            "share_of_step_percent": attributed / profile.unprofiled.wall_time_seconds * 100.0}));
                        continue;
                    };
                    let mut inner = attribution.row(
                        dispatch,
                        cost,
                        &kernel.kernel,
                        kernel.gpu_time_seconds,
                        gap,
                        1,
                    )?;
                    inner["register_spill_risk"] = kernel.register_spill_risk.clone();
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
        "mode": profile.mode, "tokens": profile.tokens, "measured_tokens": profile.measured_count().ok_or("invalid measured token range")?, "cached_prefix_tokens": profile.cached_prefix_tokens, "device_identity": profile.device_identity,
        "inputs": profile.inputs, "profile_provenance": profile.provenance, "calibration": calibration,
        "unprofiled": {"wall_seconds": profile.unprofiled.wall_time_seconds, "gpu_seconds": profile.unprofiled.gpu_time_seconds, "cache_setup_wall_seconds": profile.unprofiled.cache_setup},
        "profiled_gpu_seconds": profiled_gpu, "timestamp_perturbation_ratio": profiled_gpu / profile.unprofiled.gpu_time_seconds,
        "attribution_scale": scale, "target_utilization": target,
        "notes": ["Profile-derived diagnostic, not a model benchmark.",
            "Attribution scales precise intervals plus preceding gaps to unprofiled GPU time; per-kernel perturbation is unquantified.",
            "Host snapshot capture/materialization transfers are outside GPU kernel work; prefix preparation is separate from the measured step.",
            "Sampling scalar/SFU counts describe the logical stochastic path; greedy/conditional branches may execute less work.",
            "Traffic is logical, not a hardware DRAM counter; expert reuse is an interval unless proven by the command graph.",
            "Scalar/SFU and integer throughput are uncalibrated; Floating-point work uses an optimistic MMA-equivalent ceiling, including scalar FMA kernels.",
            "Persistent f32 carry panels risk register spills; register residency and occupancy require device profiling.",
            "Calibration bound envelopes combine per-probe CI endpoints, not a joint 95% interval; resource flags use raw GPU intervals.",
            "Alternatives preserve the selected reduction class and batch-invariance guarantee; screen before adoption."],
        "whole_model": model::summaries(profile, calibration, scale, target)?,
        "by_kernel": ranking::aggregate(&kernels, "kernel")?, "ranked_call_sites": ranking::aggregate(&dispatches, "site_group")?,
        "unmeasured_kernels": unmeasured, "dispatches": dispatches, "kernels": kernels }),
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
    #[test]
    fn excess_within_calibration_uncertainty_is_visible_without_a_floor_violation() {
        let (profile, mut calibration) = fixture();
        calibration["probes"]["calibration_mma_f32"]["ci95"] = json!([800.0, 1200.0]);
        let dispatch = &profile.submissions[0].timeline[1];
        let cost = KernelCost {
            flops: 4400,
            ..KernelCost::default()
        };
        let row = Attribution {
            scale: 10.0,
            wall: 120.0,
            target: 0.8,
            ceilings: ceilings(&calibration, "f32").unwrap(),
        }
        .row(dispatch, cost, "test", 4.0, 0.0, 1)
        .unwrap();
        assert!((row["percent_of_compute_ceiling"].as_f64().unwrap() - 110.0).abs() < 1e-12);
        assert_eq!(row["over_100_percent"], true);
        assert_eq!(row["below_median_floor"], true);
        assert_eq!(row["below_calibrated_floor"], false);
    }

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
        let internal = json!([{"kernel": "read-a", "gpu_time_seconds": 1.5, "cost": cost(40, 0)}, {"kernel": "read-b", "gpu_time_seconds": 2.5, "cost": cost(60, 0), "register_spill_risk": {"private_f32_accumulator_values_per_thread": 64}}]);
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
            probes[name] = json!({"units_per_second": rate, "ci95": [rate, rate], "arithmetic_mode": if name == "calibration_mma_f32" { Some("f32_native") } else { None }});
        }
        (
            profile,
            json!({"probes": probes, "arithmetic_probes": {"f32_native": "calibration_mma_f32"}}),
        )
    }

    #[test]
    fn unknown_subkernel_cost_is_listed_without_ranking_it_as_free() {
        let (mut profile, calibration) = fixture();
        profile.submissions[0].timeline[0].internal[0].cost = None;
        profile.validate().unwrap();
        let report = report(&profile, &calibration, 0.8).unwrap();
        assert_eq!(report["unmeasured_kernels"].as_array().unwrap().len(), 1);
        assert_eq!(report["unmeasured_kernels"][0]["kernel"], "read-a");
        assert_eq!(report["unmeasured_kernels"][0]["ranked"], false);
        assert_eq!(report["kernels"].as_array().unwrap().len(), 5);
        let total = |key: &str| {
            report[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["attributed_gpu_seconds"].as_f64().unwrap())
                .sum::<f64>()
        };
        assert!(
            (total("dispatches") - total("kernels") - total("unmeasured_kernels")).abs()
                < f64::EPSILON
        );
        assert!(ranking::human(&report).contains("Unmeasured: block/layer-0/dispatch-0 / read-a"));
    }

    #[test]
    fn suffix_tokens_and_host_copy_preparation_remain_explicit() {
        let (mut profile, calibration) = fixture();
        profile.cached_prefix_tokens = 7;
        profile.measured_tokens = Some(26);
        profile.unprofiled.cache_setup = 0.25;
        profile.validate().unwrap();
        let row = report(&profile, &calibration, 0.8).unwrap();
        assert_eq!(row["measured_tokens"], 26);
        assert_eq!(row["cached_prefix_tokens"], 7);
        assert_eq!(row["unprofiled"]["cache_setup_wall_seconds"], 0.25);
        profile.measured_tokens = Some(33);
        assert!(profile.validate().is_err());
    }

    #[test]
    fn native_bf16_never_uses_a_promoted_f32_probe() {
        let (_, mut calibration) = fixture();
        calibration["probes"]["calibration_mma_bf16"] = json!({"units_per_second": 2000.0, "ci95": [2000.0, 2000.0], "arithmetic_mode": "bf16_promoted_f32"});
        calibration["arithmetic_probes"]["bf16_promoted_f32"] = json!("calibration_mma_bf16");
        assert!(ceilings(&calibration, "bf16").is_err());
        calibration["arithmetic_probes"]["bf16_native"] = json!("calibration_mma_bf16");
        assert!(ceilings(&calibration, "bf16").is_err());
        calibration["probes"]["calibration_mma_bf16"]["arithmetic_mode"] = json!("bf16_native");
        assert!(
            (ceilings(&calibration, "bf16").unwrap().compute_rate() - 2000.0).abs() < f64::EPSILON
        );
        assert!(
            (ceilings(&calibration, "f32").unwrap().compute_rate() - 1000.0).abs() < f64::EPSILON
        );
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
        assert_eq!(
            report["whole_model"]["by_layer_type"][0]["share_of_gpu_percent"],
            80.0
        );
        assert_eq!(report["whole_model"]["unassigned_gpu_seconds"], 2.0);
        let read_bytes: u64 = report["by_kernel"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["cost"]["read_bytes"].as_u64().unwrap())
            .sum();
        assert_eq!(read_bytes, 200);
        assert!(ranking::human(&report).contains("64 f32 accumulator values/thread"));

        let mut invalid = profile;
        invalid.schema_version = 1;
        assert!(invalid.validate().is_err());
    }
}
