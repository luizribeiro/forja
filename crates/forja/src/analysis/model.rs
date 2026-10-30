use crate::analysis_input::{Profile, positive};
use forja_core::{
    KernelCost,
    roofline::{self, Ceilings},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
};

#[derive(Default)]
struct Totals {
    read: [u64; 2],
    weights: [u64; 2],
    write: u64,
    kv: u64,
    flops: u64,
    scalar_ops: u64,
    sfu_ops: u64,
    state_read_bytes: u64,
    state_write_bytes: u64,
    scratch_read_bytes: u64,
    scratch_write_bytes: u64,
    compute: [f64; 3],
    dispatches: u64,
    attributed: f64,
    layers: BTreeSet<u32>,
}

impl Totals {
    #[allow(clippy::cast_precision_loss)]
    fn add(
        &mut self,
        cost: KernelCost,
        count: u64,
        seconds: f64,
        ceiling: Ceilings,
    ) -> Result<(), Box<dyn Error>> {
        let weights = if cost.maximum_weight_bytes == 0 {
            [cost.weight_bytes; 2]
        } else {
            [cost.minimum_weight_bytes, cost.maximum_weight_bytes]
        };
        for (index, weight) in weights.into_iter().enumerate() {
            self.read[index] = self.read[index]
                .checked_add(
                    roofline::traffic(cost, index == 0)
                        .ok_or("traffic overflow")?
                        .checked_sub(cost.write_bytes)
                        .ok_or("invalid traffic")?,
                )
                .ok_or("read total overflow")?;
            self.weights[index] = self.weights[index]
                .checked_add(weight)
                .ok_or("weight total overflow")?;
        }
        self.write = self
            .write
            .checked_add(cost.write_bytes)
            .ok_or("write total overflow")?;
        self.kv = self
            .kv
            .checked_add(cost.kv_bytes)
            .ok_or("KV total overflow")?;
        self.flops = self
            .flops
            .checked_add(cost.flops)
            .ok_or("FLOP total overflow")?;
        self.scalar_ops = self
            .scalar_ops
            .checked_add(cost.scalar_ops)
            .ok_or("scalar total overflow")?;
        self.sfu_ops = self
            .sfu_ops
            .checked_add(cost.sfu_ops)
            .ok_or("SFU total overflow")?;
        self.state_read_bytes = self
            .state_read_bytes
            .checked_add(cost.state_read_bytes)
            .ok_or("state read total overflow")?;
        self.state_write_bytes = self
            .state_write_bytes
            .checked_add(cost.state_write_bytes)
            .ok_or("state write total overflow")?;
        self.scratch_read_bytes = self
            .scratch_read_bytes
            .checked_add(cost.scratch_read_bytes)
            .ok_or("scratch read total overflow")?;
        self.scratch_write_bytes = self
            .scratch_write_bytes
            .checked_add(cost.scratch_write_bytes)
            .ok_or("scratch write total overflow")?;
        self.dispatches = self
            .dispatches
            .checked_add(count)
            .ok_or("dispatch total overflow")?;
        for (seconds, ceiling) in
            self.compute
                .iter_mut()
                .zip([ceiling.optimistic(), ceiling, ceiling.pessimistic()])
        {
            *seconds += cost.flops as f64 / ceiling.compute_rate();
        }
        self.attributed += seconds;
        positive(self.attributed)?;
        if self.compute.iter().any(|seconds| !seconds.is_finite()) {
            return Err("compute total overflow".into());
        }
        Ok(())
    }

    #[allow(clippy::cast_precision_loss)]
    fn memory(&self, mode: &str, ceiling: Ceilings) -> Result<[f64; 2], Box<dyn Error>> {
        let mut memory = [0.0_f64; 2];
        for (index, seconds) in memory.iter_mut().enumerate() {
            *seconds = if mode == "decode" {
                self.weights[index]
                    .checked_add(self.kv)
                    .ok_or("decode stream overflow")? as f64
                    / ceiling.read_bandwidth()
            } else {
                (self.read[index] as f64 / ceiling.read_bandwidth()).max(
                    self.read[index]
                        .checked_add(self.write)
                        .ok_or("prefill traffic overflow")? as f64
                        / ceiling.copy_bandwidth(),
                )
            };
        }
        Ok(memory)
    }

    #[allow(clippy::cast_precision_loss)]
    fn json(
        &self,
        mode: &str,
        gpu: f64,
        wall: Option<f64>,
        target: f64,
        ceiling: Ceilings,
    ) -> Result<Value, Box<dyn Error>> {
        positive(gpu)?;
        let memory = self.memory(mode, ceiling)?;
        let fastest_memory = self.memory(mode, ceiling.optimistic())?;
        let slowest_memory = self.memory(mode, ceiling.pessimistic())?;
        let interval = [
            fastest_memory[0]
                .max(self.compute[0])
                .max(self.dispatches as f64 * ceiling.optimistic().dispatch_floor()),
            slowest_memory[1]
                .max(self.compute[2])
                .max(self.dispatches as f64 * ceiling.pessimistic().dispatch_floor()),
        ];
        let floor = self.dispatches as f64 * ceiling.dispatch_floor();
        let bounds = memory.map(|memory| memory.max(self.compute[1]).max(floor));
        let targets = memory.map(|memory| (memory.max(self.compute[1]) / target).max(floor));
        if interval.iter().any(|value| !value.is_finite()) {
            return Err("model confidence envelope overflow".into());
        }
        if bounds
            .iter()
            .chain(&targets)
            .any(|value| !value.is_finite())
        {
            return Err("model bound overflow".into());
        }
        let gpu_efficiency = bounds.map(|bound| bound / gpu * 100.0);
        let wall_efficiency = wall.map(|wall| bounds.map(|bound| bound / wall * 100.0));
        if gpu_efficiency
            .iter()
            .chain(wall_efficiency.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err("model efficiency overflow".into());
        }
        let over_100 = crate::analysis_diagnostics::over_100(
            gpu_efficiency
                .into_iter()
                .chain(wall_efficiency.unwrap_or([0.0; 2])),
        );
        Ok(
            json!({"bound_kind": if mode == "decode" { "matrix-weights-plus-KV-once" } else { "aggregate-prefill-roofline" },
            "weight_bytes_range": self.weights, "kv_bytes": self.kv, "logical_read_bytes_range": self.read, "write_bytes": self.write,
            "state_read_bytes": self.state_read_bytes, "state_write_bytes": self.state_write_bytes,
            "scratch_read_bytes": self.scratch_read_bytes, "scratch_write_bytes": self.scratch_write_bytes,
            "scalar_ops": self.scalar_ops, "sfu_ops": self.sfu_ops, "scalar_sfu_ceiling": "uncalibrated",
            "matrix_product_flops": self.flops, "physical_dispatches": self.dispatches, "layers": self.layers, "layer_count": self.layers.len(),
            "streaming_bound_seconds_range": memory, "compute_bound_seconds": self.compute[1],
            "dispatch_count_floor_seconds": floor, "lower_bound_seconds_range": bounds, "target_seconds_range": targets,
            "observed_or_attributed_gpu_seconds": gpu, "attributed_interval_seconds": self.attributed,
            "gpu_efficiency_percent_range": gpu_efficiency,
            "wall_efficiency_percent_range": wall_efficiency,
            "recoverable_seconds_range": [(gpu-bounds[1]).max(0.0), (gpu-bounds[0]).max(0.0)],
            "calibration_bound_seconds_envelope": interval, "over_100_percent": over_100,
            "timing_is_attributed_estimate": wall.is_none(), "median_bound_exceeds_observed": bounds[0] > gpu,
            "bound_exceeds_observed": interval[0] > gpu }),
        )
    }
}

pub(super) fn summaries(
    profile: &Profile,
    calibration: &Value,
    scale: f64,
    target: f64,
) -> Result<Value, Box<dyn Error>> {
    let shared = super::ceilings(calibration, "f32")?;
    let mut all = Totals::default();
    let mut phases = Vec::new();
    let mut all_layers = BTreeMap::<&str, Totals>::new();
    for submission in &profile.submissions {
        let mut phase = Totals::default();
        let mut layers = BTreeMap::<&str, Totals>::new();
        for dispatch in &submission.timeline {
            let ceiling = super::ceilings(calibration, &dispatch.compute_dtype)?;
            let kind = dispatch
                .scope
                .as_ref()
                .map_or("unscoped", |scope| scope.layer_type.as_str());
            for total in [
                &mut all,
                &mut phase,
                layers.entry(kind).or_default(),
                all_layers.entry(kind).or_default(),
            ] {
                total.add(
                    dispatch.cost,
                    dispatch.physical_dispatches,
                    (dispatch.gpu_time_seconds + dispatch.gap_before_seconds) * scale,
                    ceiling,
                )?;
                if let Some(layer) = dispatch.scope.as_ref().and_then(|scope| scope.layer) {
                    total.layers.insert(layer);
                }
            }
        }
        let mut summary = phase.json(
            &profile.mode,
            submission.gpu_time_seconds * scale,
            None,
            target,
            shared,
        )?;
        let by_layer_type = layers
            .into_iter()
            .map(|(kind, total)| {
                let mut row = total.json(&profile.mode, total.attributed, None, target, shared)?;
                row["layer_type"] = json!(kind);
                Ok(row)
            })
            .collect::<Result<Vec<Value>, Box<dyn Error>>>()?;
        summary["submission"] = json!(submission.index);
        summary["phase"] = submission.phase.clone();
        summary["timing_is_attributed_estimate"] = json!(true);
        summary["by_layer_type"] = json!(by_layer_type);
        phases.push(summary);
    }
    let by_layer_type = all_layers
        .into_iter()
        .map(|(kind, total)| {
            let mut row = total.json(&profile.mode, total.attributed, None, target, shared)?;
            row["layer_type"] = json!(kind);
            Ok(row)
        })
        .collect::<Result<Vec<Value>, Box<dyn Error>>>()?;
    Ok(
        json!({"by_layer_type": by_layer_type, "overall": all.json(&profile.mode, profile.unprofiled.gpu_time_seconds, Some(profile.unprofiled.wall_time_seconds), target, shared)?, "phases": phases}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_efficiency_flags_use_optimistic_calibration_endpoints() {
        use forja_core::roofline::Estimate;
        let ceiling = Ceilings::with_intervals(
            Estimate::new(100.0, [90.0, 110.0]).unwrap(),
            Estimate::new(200.0, [180.0, 220.0]).unwrap(),
            Estimate::new(1000.0, [900.0, 1100.0]).unwrap(),
            Estimate::new(0.01, [0.009, 0.011]).unwrap(),
        );
        let mut total = Totals::default();
        total
            .add(
                KernelCost {
                    flops: 500,
                    scalar_ops: 33,
                    sfu_ops: 7,
                    state_read_bytes: 132,
                    state_write_bytes: 132,
                    scratch_read_bytes: 4096,
                    scratch_write_bytes: 2048,
                    ..KernelCost::default()
                },
                1,
                0.48,
                ceiling,
            )
            .unwrap();
        let within = total.json("prefill", 0.48, None, 0.8, ceiling).unwrap();
        assert_eq!(within["scalar_ops"], 33);
        assert_eq!(within["sfu_ops"], 7);
        assert_eq!(within["state_read_bytes"], 132);
        assert_eq!(within["state_write_bytes"], 132);
        assert_eq!(within["scratch_read_bytes"], 4096);
        assert_eq!(within["scratch_write_bytes"], 2048);
        assert_eq!(within["scalar_sfu_ceiling"], "uncalibrated");
        assert_eq!(within["over_100_percent"], true);
        assert_eq!(within["bound_exceeds_observed"], false);
        let outside = total.json("prefill", 0.4, None, 0.8, ceiling).unwrap();
        assert_eq!(outside["bound_exceeds_observed"], true);
    }

    #[test]
    fn decode_streams_weights_and_kv_once() {
        let ceiling = Ceilings::new(100.0, 200.0, 1000.0, 0.01).unwrap();
        let cost = KernelCost {
            read_bytes: 120,
            write_bytes: 20,
            weight_bytes: 80,
            kv_bytes: 20,
            flops: 500,
            ..KernelCost::default()
        };
        let mut total = Totals::default();
        total.add(cost, 7, 2.0, ceiling).unwrap();
        let decode = total.json("decode", 2.0, Some(4.0), 0.8, ceiling).unwrap();
        let prefill = total.json("prefill", 2.0, None, 0.8, ceiling).unwrap();
        assert!(
            (decode["lower_bound_seconds_range"][0].as_f64().unwrap() - 1.0).abs() < f64::EPSILON
        );
        assert!(
            (prefill["lower_bound_seconds_range"][0].as_f64().unwrap() - 1.2).abs() < f64::EPSILON
        );
        assert!(
            total
                .add(
                    KernelCost {
                        read_bytes: u64::MAX,
                        ..KernelCost::default()
                    },
                    1,
                    1.0,
                    ceiling
                )
                .is_err()
        );
    }
}
