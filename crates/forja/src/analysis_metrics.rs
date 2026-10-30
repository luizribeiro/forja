use crate::analysis_input::{Dispatch, positive};
use forja_core::{
    KernelCost,
    roofline::{self, Ceilings, Estimate},
};
use serde_json::{Value, json};
use std::error::Error;

pub(super) fn ceilings(calibration: &Value, dtype: &str) -> Result<Ceilings, Box<dyn Error>> {
    let mode = format!("{dtype}_native");
    let compute_probe = calibration["arithmetic_probes"][&mode].as_str().ok_or_else(|| format!("no calibrated {mode} ceiling; promoted arithmetic cannot stand in for native arithmetic"))?;
    if calibration["probes"][compute_probe]["arithmetic_mode"] != mode {
        return Err("calibration arithmetic key disagrees with the executed probe".into());
    }
    let rate = |name: &str| -> Result<Estimate, Box<dyn Error>> {
        let probe = &calibration["probes"][name];
        let number = |value: &Value| value.as_f64().ok_or("missing calibration estimate");
        Estimate::new(
            number(&probe["units_per_second"])?,
            [number(&probe["ci95"][0])?, number(&probe["ci95"][1])?],
        )
        .ok_or_else(|| "invalid calibration confidence interval".into())
    };
    let dispatch = rate("dependent_dispatch")?;
    let latency = Estimate::new(
        1.0 / dispatch.median(),
        [1.0 / dispatch.upper(), 1.0 / dispatch.lower()],
    )
    .ok_or("unrepresentable dispatch confidence interval")?;
    Ok(Ceilings::with_intervals(
        rate("streaming_read")?,
        rate("streaming_copy")?,
        rate(compute_probe)?,
        latency,
    ))
}

fn site_group(dispatch: &Dispatch) -> String {
    dispatch.scope.as_ref().map_or_else(
        || dispatch.call_site.clone(),
        |scope| {
            format!(
                "{}/{}/{}",
                scope.layer_type,
                scope.site,
                dispatch.call_site.rsplit('/').next().unwrap_or("dispatch")
            )
        },
    )
}

#[derive(Clone, Copy)]
pub(super) struct Attribution {
    pub(super) scale: f64,
    pub(super) wall: f64,
    pub(super) target: f64,
    pub(super) ceilings: Ceilings,
}

impl Attribution {
    pub(super) fn row(
        self,
        dispatch: &Dispatch,
        cost: KernelCost,
        kernel: &str,
        raw: f64,
        gap: f64,
        count: u64,
    ) -> Result<Value, Box<dyn Error>> {
        let Self {
            scale,
            wall,
            target,
            ceilings,
        } = self;
        #[allow(clippy::cast_precision_loss)]
        let minimum_bytes = roofline::traffic(cost, true).ok_or("cost overflow")? as f64;
        #[allow(clippy::cast_precision_loss)]
        let maximum_bytes = roofline::traffic(cost, false).ok_or("cost overflow")? as f64;
        #[allow(clippy::cast_precision_loss)]
        let compute = cost.flops as f64 / raw;
        if [minimum_bytes / raw, maximum_bytes / raw, compute]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("unrepresentable achieved rate".into());
        }
        let limits = roofline::limits(cost, count, ceilings).ok_or("cost overflow")?;
        let mut high_cost = cost;
        if cost.maximum_weight_bytes != 0 {
            high_cost.minimum_weight_bytes = cost.maximum_weight_bytes;
        }
        let upper = roofline::limits(high_cost, count, ceilings).ok_or("cost overflow")?;
        positive(limits.seconds())?;
        positive(upper.seconds())?;
        let optimistic =
            roofline::limits(cost, count, ceilings.optimistic()).ok_or("cost overflow")?;
        let pessimistic =
            roofline::limits(high_cost, count, ceilings.pessimistic()).ok_or("cost overflow")?;
        #[allow(clippy::cast_precision_loss)]
        let read_percent = [
            roofline::traffic(cost, true)
                .and_then(|bytes| bytes.checked_sub(cost.write_bytes))
                .ok_or("cost overflow")? as f64
                / raw
                / ceilings.read_bandwidth()
                * 100.0,
            roofline::traffic(cost, false)
                .and_then(|bytes| bytes.checked_sub(cost.write_bytes))
                .ok_or("cost overflow")? as f64
                / raw
                / ceilings.read_bandwidth()
                * 100.0,
        ];
        let copy_percent = [
            minimum_bytes / raw / ceilings.copy_bandwidth() * 100.0,
            maximum_bytes / raw / ceilings.copy_bandwidth() * 100.0,
        ];
        let memory_percent = [
            read_percent[0].max(copy_percent[0]),
            read_percent[1].max(copy_percent[1]),
        ];
        let target_low = positive(
            limits
                .target_seconds(target)
                .ok_or("invalid utilization target")?,
        )?;
        let target_high = positive(
            upper
                .target_seconds(target)
                .ok_or("invalid utilization target")?,
        )?;
        let current = positive((raw + gap) * scale)?;
        let share = current / wall * 100.0;
        let compute_percent = compute / ceilings.compute_rate() * 100.0;
        if memory_percent
            .iter()
            .chain([&share, &compute_percent])
            .any(|value| !value.is_finite())
        {
            return Err("resource percentage overflow".into());
        }
        let recoverable = |bound: f64| (current - bound).max(0.0);
        Ok(
            json!({"call_site": dispatch.call_site, "site_group": site_group(dispatch),
        "layer_type": dispatch.scope.as_ref().map_or("unscoped", |scope| scope.layer_type.as_str()),
        "layer": dispatch.scope.as_ref().and_then(|scope| scope.layer), "op": dispatch.op, "kernel": kernel,
        "variant": dispatch.variant, "alternatives": dispatch.alternatives, "shape": dispatch.shape,
        "scalar_sfu_ceiling": "uncalibrated", "compute_dtype": dispatch.compute_dtype, "arithmetic_mode": format!("{}_native", dispatch.compute_dtype), "cost": cost.json(), "physical_dispatches": count,
        "profiled_gpu_seconds": raw, "preceding_gap_seconds": gap, "attributed_gpu_seconds": current,
        "share_of_step_percent": share,
        "raw_gigabytes_per_second_range": [minimum_bytes / raw / 1e9, maximum_bytes / raw / 1e9],
        "percent_of_streaming_read_ceiling_range": read_percent, "percent_of_streaming_copy_ceiling_range": copy_percent,
        "percent_of_memory_ceiling_range": memory_percent, "resource_rate_time_basis": "raw-profiled-gpu-interval",
        "raw_teraflops_per_second": compute / 1e12, "percent_of_compute_ceiling": compute_percent,
        "bound": limits.bound(), "bound_at_maximum_footprint": upper.bound(),
        "ceiling_seconds_range": [limits.seconds(), upper.seconds()], "target_seconds_range": [target_low, target_high],
        "recoverable_seconds": recoverable(limits.seconds()), "recoverable_seconds_range": [recoverable(upper.seconds()), recoverable(limits.seconds())],
        "recoverable_at_target_seconds": recoverable(target_low),
        "calibration_bound_seconds_envelope": [optimistic.seconds(), pessimistic.seconds()],
        "over_100_percent": crate::analysis_diagnostics::over_100([share, memory_percent[1], compute_percent]),
        "below_median_floor": raw < limits.seconds(), "below_calibrated_floor": raw < optimistic.seconds() }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resource_percentages_and_floor_flags_share_the_raw_interval() {
        let dispatch = Dispatch {
            index: 0,
            call_site: "test/dispatch-0".into(),
            scope: None,
            shape: "test".into(),
            op: "matmul".into(),
            kernel: "test".into(),
            variant: None,
            alternatives: vec![],
            gpu_time_seconds: 4.0,
            gap_before_seconds: 1.0,
            physical_dispatches: 1,
            compute_dtype: "f32".into(),
            cost: KernelCost {
                flops: 6000,
                ..KernelCost::default()
            },
            padded_flops: [6000; 2],
            internal: vec![],
        };
        let attribution = Attribution {
            scale: 10.0,
            wall: 120.0,
            target: 0.8,
            ceilings: Ceilings::new(100.0, 200.0, 1000.0, 0.01).unwrap(),
        };
        let row = attribution
            .row(&dispatch, dispatch.cost, "test", 4.0, 1.0, 1)
            .unwrap();
        assert_eq!(row["percent_of_compute_ceiling"], 150.0);
        assert_eq!(row["below_calibrated_floor"], true);
    }
}
