use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const SUMS: &[&str] = &[
    "profiled_gpu_seconds",
    "preceding_gap_seconds",
    "attributed_gpu_seconds",
    "share_of_step_percent",
    "recoverable_seconds",
    "recoverable_at_target_seconds",
];

pub(super) fn aggregate(rows: &[Value], field: &str) -> Vec<Value> {
    let mut groups = BTreeMap::<String, Vec<&Value>>::new();
    for row in rows {
        let key = format!("{} | {} | {}", row[field], row["variant"], row["kernel"]);
        groups.entry(key).or_default().push(row);
    }
    let mut result = groups.into_values().map(|rows| {
        let first = rows[0];
        let mut group = json!({"name": first[field], "kernel": first["kernel"], "variant": first["variant"], "bound": first["bound"], "instances": rows.len()});
        for &key in SUMS {
            group[key] = json!(rows.iter().filter_map(|row| row[key].as_f64()).sum::<f64>());
        }
        for key in ["ceiling_seconds_range", "target_seconds_range", "recoverable_seconds_range"] {
            group[key] = json!([0, 1].map(|index| rows.iter().filter_map(|row| row[key][index].as_f64()).sum::<f64>()));
        }
        let names = |key: &str| rows.iter().filter_map(|row| row[key].as_str()).collect::<BTreeSet<_>>();
        group["call_sites"] = json!(names("call_site"));
        group["bounds"] = json!(names("bound"));
        let alternatives = first["alternatives"].as_array().into_iter().flatten().filter(|alternative| rows.iter().all(|row| row["alternatives"].as_array().is_some_and(|values| values.contains(alternative)))).collect::<Vec<_>>();
        group["alternatives"] = json!(alternatives);
        let raw = group["profiled_gpu_seconds"].as_f64().unwrap_or(1.0);
        for key in ["percent_of_compute_ceiling", "percent_of_streaming_read_ceiling_range"] {
            if key.ends_with("range") {
                group[key] = json!([0, 1].map(|index| rows.iter().map(|row| row[key][index].as_f64().unwrap_or(0.0) * row["profiled_gpu_seconds"].as_f64().unwrap_or(0.0)).sum::<f64>() / raw));
            } else {
                group[key] = json!(rows.iter().map(|row| row[key].as_f64().unwrap_or(0.0) * row["profiled_gpu_seconds"].as_f64().unwrap_or(0.0)).sum::<f64>() / raw);
            }
        }
        group
    }).collect::<Vec<_>>();
    super::sort(&mut result);
    result
}

pub(crate) fn human(report: &Value) -> String {
    use std::fmt::Write;
    let mut out = format!(
        "{}: {} tokens — profile diagnostic (not a model benchmark)\n",
        report["mode"].as_str().unwrap_or("unknown"),
        report["tokens"]
    );
    let number = |value: &Value| value.as_f64().unwrap_or(0.0);
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
        "Model bound {:.3}–{:.3} ms; GPU efficiency {:.1}–{:.1}%; dispatch floor {:.3} ms",
        number(&overall["lower_bound_seconds_range"][0]) * 1000.0,
        number(&overall["lower_bound_seconds_range"][1]) * 1000.0,
        number(&overall["gpu_efficiency_percent_range"][0]),
        number(&overall["gpu_efficiency_percent_range"][1]),
        number(&overall["dispatch_count_floor_seconds"]) * 1000.0
    );
    for row in report["whole_model"]["by_layer_type"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let _ = writeln!(
            out,
            "  {}: bound {:.3}–{:.3} ms; GPU efficiency {:.1}–{:.1}%",
            row["layer_type"].as_str().unwrap_or("?"),
            number(&row["lower_bound_seconds_range"][0]) * 1000.0,
            number(&row["lower_bound_seconds_range"][1]) * 1000.0,
            number(&row["gpu_efficiency_percent_range"][0]),
            number(&row["gpu_efficiency_percent_range"][1])
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
            "  {} | {} | {} | {:.1}% wall | recover {:.3}/{:.3} ms | BW {:.1}–{:.1}% | MMA {:.1}%",
            row["name"].as_str().unwrap_or("?"),
            row["variant"]
                .as_str()
                .or_else(|| row["kernel"].as_str())
                .unwrap_or("?"),
            row["bound"].as_str().unwrap_or("?"),
            number(&row["share_of_step_percent"]),
            number(&row["recoverable_seconds"]) * 1000.0,
            number(&row["recoverable_at_target_seconds"]) * 1000.0,
            number(&row["percent_of_streaming_read_ceiling_range"][0]),
            number(&row["percent_of_streaming_read_ceiling_range"][1]),
            number(&row["percent_of_compute_ceiling"])
        );
        let _ = writeln!(out, "    screen: {}", row["alternatives"]);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidates_must_satisfy_every_instance() {
        let rows = [
            json!({"site_group":"a", "variant":"current", "kernel":"k", "alternatives":["one", "two"], "recoverable_seconds": 1.0}),
            json!({"site_group":"a", "variant":"current", "kernel":"k", "alternatives":["two"], "recoverable_seconds": 2.0}),
        ];
        let result = aggregate(&rows, "site_group");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["recoverable_seconds"], 3.0);
        assert_eq!(result[0]["alternatives"], json!(["two"]));
    }
}
