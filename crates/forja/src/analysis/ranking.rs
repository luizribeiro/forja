use crate::analysis_input::CostValue;
use forja_core::KernelCost;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;

const SUMS: &[&str] = &[
    "profiled_gpu_seconds",
    "preceding_gap_seconds",
    "attributed_gpu_seconds",
    "share_of_step_percent",
    "recoverable_seconds",
    "recoverable_at_target_seconds",
];

pub(super) fn aggregate(rows: &[Value], field: &str) -> Result<Vec<Value>, Box<dyn Error>> {
    let mut groups = BTreeMap::<String, Vec<&Value>>::new();
    for row in rows {
        let key = format!("{} | {} | {}", row[field], row["variant"], row["kernel"]);
        groups.entry(key).or_default().push(row);
    }
    let mut result = groups.into_values().map(|rows| {
        let mut cost = KernelCost::default();
        let mut dispatches = 0_u64;
        for row in &rows {
            if let Some(value) = row.get("cost") {
                cost = cost.checked_add(serde_json::from_value::<CostValue>(value.clone())?.0).ok_or("group cost overflow")?;
            }
            dispatches = dispatches.checked_add(row["physical_dispatches"].as_u64().unwrap_or(1)).ok_or("dispatch count overflow")?;
        }
        let first = rows[0];
        let mut group = json!({"name": first[field], "kernel": first["kernel"], "variant": first["variant"], "bound": first["bound"], "instances": rows.len()});
        let mut risks = Vec::new();
        for row in &rows {
            if let Some(risk) = row.get("register_spill_risk").filter(|value| !value.is_null())
                && !risks.contains(risk) { risks.push(risk.clone()); }
        }
        group["register_spill_risks"] = json!(risks);
        group["cost"] = cost.json();
        group["physical_dispatches"] = json!(dispatches);
        for &key in SUMS {
            group[key] = json!(rows.iter().filter_map(|row| row[key].as_f64()).sum::<f64>());
        }
        for key in ["calibration_bound_seconds_envelope",
            "ceiling_seconds_range", "target_seconds_range", "recoverable_seconds_range"] {
            group[key] = json!([0, 1].map(|index| rows.iter().filter_map(|row| row[key][index].as_f64()).sum::<f64>()));
        }
        let names = |key: &str| rows.iter().filter_map(|row| row[key].as_str()).collect::<BTreeSet<_>>();
        group["below_calibrated_floor"] = json!(rows.iter().any(|row| row["below_calibrated_floor"].as_bool().unwrap_or(false)));
        group["below_median_floor"] = json!(rows.iter().any(|row| row["below_median_floor"].as_bool().unwrap_or(false)));
        group["over_100_percent"] = json!(rows.iter().any(|row| row["over_100_percent"].as_bool().unwrap_or(false)));
        group["flagged_instances"] = json!(rows.iter().filter(|row| row["below_calibrated_floor"].as_bool().unwrap_or(false) || row["over_100_percent"].as_bool().unwrap_or(false) || row["below_median_floor"].as_bool().unwrap_or(false)).count());
        group["call_sites"] = json!(names("call_site"));
        group["bounds"] = json!(names("bound"));
        group["arithmetic_modes"] = json!(names("arithmetic_mode"));
        let alternatives = first["alternatives"].as_array().into_iter().flatten().filter(|alternative| rows.iter().all(|row| row["alternatives"].as_array().is_some_and(|values| values.contains(alternative)))).collect::<Vec<_>>();
        group["alternatives"] = json!(alternatives);
        let raw = group["profiled_gpu_seconds"].as_f64().filter(|raw| *raw > 0.0).unwrap_or(1.0);
        for key in ["percent_of_compute_ceiling", "percent_of_streaming_read_ceiling_range", "percent_of_streaming_copy_ceiling_range", "percent_of_memory_ceiling_range"] {
            if key.ends_with("range") {
                group[key] = json!([0, 1].map(|index| rows.iter().map(|row| row[key][index].as_f64().unwrap_or(0.0) * (row["profiled_gpu_seconds"].as_f64().unwrap_or(0.0) / raw)).sum::<f64>()));
            } else {
                group[key] = json!(rows.iter().map(|row| row[key].as_f64().unwrap_or(0.0) * (row["profiled_gpu_seconds"].as_f64().unwrap_or(0.0) / raw)).sum::<f64>());
            }
        }
        group["over_100_percent"] = json!(group["over_100_percent"].as_bool().unwrap_or(false) || crate::analysis_diagnostics::over_100([group["share_of_step_percent"].as_f64().unwrap_or(0.0), group["percent_of_compute_ceiling"].as_f64().unwrap_or(0.0), group["percent_of_memory_ceiling_range"][1].as_f64().unwrap_or(0.0)]));
        Ok(group)
    }).collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    super::sort(&mut result);
    Ok(result)
}

pub(crate) fn human(report: &Value) -> String {
    use std::fmt::Write;
    let mut out = format!(
        "{}: context {}, {} measured tokens — profile diagnostic (not a model benchmark)\n",
        report["mode"].as_str().unwrap_or("unknown"),
        report["tokens"],
        report["measured_tokens"]
    );
    let number = |value: &Value| value.as_f64().unwrap_or(0.0);
    admission_lines(&mut out, report);
    let overall = &report["whole_model"]["overall"];
    let _ = writeln!(
        out,
        "GPU {:.3} ms; wall {:.3} ms; timestamp perturbation {:.3}x",
        number(&report["unprofiled"]["gpu_seconds"]) * 1000.0,
        number(&report["unprofiled"]["wall_seconds"]) * 1000.0,
        number(&report["timestamp_perturbation_ratio"])
    );
    let _ = writeln!(
        out,
        "Model bound {:.3}–{:.3} ms; GPU efficiency {:.1}–{:.1}%; dispatch floor {:.3} ms{}",
        number(&overall["lower_bound_seconds_range"][0]) * 1000.0,
        number(&overall["lower_bound_seconds_range"][1]) * 1000.0,
        number(&overall["gpu_efficiency_percent_range"][0]),
        number(&overall["gpu_efficiency_percent_range"][1]),
        number(&overall["dispatch_count_floor_seconds"]) * 1000.0,
        crate::analysis_diagnostics::marker(overall)
    );
    for row in report["whole_model"]["by_layer_type"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let _ = writeln!(
            out,
            "  {}: {:.1}% GPU / {:.1}% wall; bound {:.3}–{:.3} ms; GPU efficiency {:.1}–{:.1}%{}",
            row["layer_type"].as_str().unwrap_or("?"),
            number(&row["share_of_gpu_percent"]),
            number(&row["share_of_step_percent"]),
            number(&row["lower_bound_seconds_range"][0]) * 1000.0,
            number(&row["lower_bound_seconds_range"][1]) * 1000.0,
            number(&row["gpu_efficiency_percent_range"][0]),
            number(&row["gpu_efficiency_percent_range"][1]),
            crate::analysis_diagnostics::marker(row)
        );
    }
    let _ = writeln!(
        out,
        "\nRanked call sites (summed across layers/chunks); recovery at ceiling / {:.0}% target:",
        number(&report["target_utilization"]) * 100.0
    );
    for row in report["ranked_call_sites"]
        .as_array()
        .into_iter()
        .flatten()
        .take(15)
    {
        let _ = writeln!(
            out,
            "  {}{} | {} | {} | {:.1}% wall | recover {:.3}/{:.3} ms | BW {:.1}–{:.1}% | MMA-equivalent {:.1}% {}",
            row["name"].as_str().unwrap_or("?"),
            crate::analysis_diagnostics::marker(row),
            row["variant"]
                .as_str()
                .or_else(|| row["kernel"].as_str())
                .unwrap_or("?"),
            row["bound"].as_str().unwrap_or("?"),
            number(&row["share_of_step_percent"]),
            number(&row["recoverable_seconds"]) * 1000.0,
            number(&row["recoverable_at_target_seconds"]) * 1000.0,
            number(&row["percent_of_memory_ceiling_range"][0]),
            number(&row["percent_of_memory_ceiling_range"][1]),
            number(&row["percent_of_compute_ceiling"]),
            row["arithmetic_modes"]
        );
        let _ = writeln!(out, "    screen: {}", row["alternatives"]);
    }
    out.push_str(&register_risks(report));
    for row in report["unmeasured_kernels"]
        .as_array()
        .into_iter()
        .flatten()
        .take(10)
    {
        let _ = writeln!(
            out,
            "Unmeasured: {} / {} (cost unavailable; omitted from kernel ranking)",
            row["call_site"].as_str().unwrap_or("?"),
            row["kernel"].as_str().unwrap_or("?")
        );
    }
    out.push_str("\nFull JSON contains every dispatch, kernel, phase and layer type. Traffic is logical; expert reuse is a range. Scalar/SFU throughput is uncalibrated. Screen alternatives before adopting them.\n");
    out
}

fn register_risks(report: &Value) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for row in report["by_kernel"].as_array().into_iter().flatten() {
        for risk in row["register_spill_risks"].as_array().into_iter().flatten() {
            let _ = writeln!(
                out,
                "Register risk: {} / {}: {} f32 accumulator values/thread (source estimate; spills unmeasured)",
                row["kernel"].as_str().unwrap_or("?"),
                row["variant"].as_str().unwrap_or("?"),
                risk["private_f32_accumulator_values_per_thread"]
            );
        }
    }
    out
}

fn admission_lines(out: &mut String, report: &Value) {
    use std::fmt::Write;
    for buffer in report["buffer_admission"].as_array().into_iter().flatten() {
        let _ = writeln!(
            out,
            "Admission submission {} buffer {}: measured/predicted {:.3}x; backstop {}x{}",
            buffer["submission"],
            buffer["buffer"],
            buffer["measured_predicted_ratio"].as_f64().unwrap_or(0.0),
            buffer["backstop_ratio"],
            if buffer["exceeds_backstop"] == true {
                " [exceeded]"
            } else {
                ""
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregate_markers_include_summed_shares_and_memory_percentages() {
        let row = json!({"site_group":"a", "variant":"current", "kernel":"k", "profiled_gpu_seconds":1.0, "share_of_step_percent":70.0, "percent_of_streaming_read_ceiling_range":[50.0,50.0], "percent_of_streaming_copy_ceiling_range":[70.0,70.0], "percent_of_memory_ceiling_range":[70.0,70.0]});
        let result = aggregate(&[row.clone(), row], "site_group").unwrap();
        assert_eq!(result[0]["share_of_step_percent"], 140.0);
        assert_eq!(
            result[0]["percent_of_memory_ceiling_range"],
            json!([70.0, 70.0])
        );
        assert_eq!(result[0]["over_100_percent"], true);
    }

    #[test]
    fn human_marks_resource_percentages_and_model_bound_excess() {
        let report = json!({"mode":"decode", "tokens":1, "whole_model":{"overall":{"bound_exceeds_observed":true}, "by_layer_type":[{"layer_type":"test", "over_100_percent":true}]}, "ranked_call_sites":[{"name":"site", "over_100_percent":true}]});
        let text = human(&report);
        assert!(
            text.lines().any(|line| line.starts_with("Model bound")
                && line.contains("[! outside calibration CI]"))
        );
        assert!(text.lines().any(|line| line.contains("test:")
            && line.contains("bound")
            && line.contains("[! >100%]")));
        assert!(text.contains("site [! >100%]"));
    }

    #[test]
    fn candidates_must_satisfy_every_instance() {
        let rows = [
            json!({"site_group":"a", "variant":"current", "kernel":"k", "alternatives":["one", "two"], "recoverable_seconds": 1.0}),
            json!({"site_group":"a", "variant":"current", "kernel":"k", "alternatives":["two"], "recoverable_seconds": 2.0}),
        ];
        let result = aggregate(&rows, "site_group").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["recoverable_seconds"], 3.0);
        assert_eq!(result[0]["alternatives"], json!(["two"]));
    }
}
