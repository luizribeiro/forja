use forja_core::KernelCost;
use serde::Deserialize;
use serde_json::Value;
use std::error::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(remote = "KernelCost")]
pub(crate) struct Cost {
    read_bytes: u64,
    write_bytes: u64,
    flops: u64,
    scalar_ops: u64,
    #[serde(default)]
    sfu_ops: u64,
    #[serde(default)]
    state_read_bytes: u64,
    #[serde(default)]
    state_write_bytes: u64,
    weight_bytes: u64,
    minimum_weight_bytes: u64,
    maximum_weight_bytes: u64,
    kv_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Scope {
    pub(crate) site: String,
    pub(crate) layer_type: String,
    pub(crate) layer: Option<u32>,
}

#[derive(Deserialize)]
struct CostValue(#[serde(with = "Cost")] KernelCost);

fn optional_cost<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<KernelCost>, D::Error> {
    Ok(Option::<CostValue>::deserialize(deserializer)?.map(|value| value.0))
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Interval {
    pub(crate) kernel: String,
    pub(crate) gpu_time_seconds: f64,
    #[serde(deserialize_with = "optional_cost")]
    pub(crate) cost: Option<KernelCost>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Dispatch {
    pub(crate) index: u64,
    pub(crate) call_site: String,
    pub(crate) scope: Option<Scope>,
    pub(crate) shape: String,
    pub(crate) op: String,
    pub(crate) kernel: String,
    pub(crate) variant: Option<String>,
    pub(crate) alternatives: Vec<String>,
    pub(crate) gpu_time_seconds: f64,
    pub(crate) gap_before_seconds: f64,
    pub(crate) physical_dispatches: u64,
    pub(crate) compute_dtype: String,
    #[serde(with = "Cost")]
    pub(crate) cost: KernelCost,
    pub(crate) padded_flops: [u64; 2],
    pub(crate) internal: Vec<Interval>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Submission {
    pub(crate) index: u64,
    pub(crate) phase: Value,
    pub(crate) gpu_time_seconds: f64,
    pub(crate) timeline: Vec<Dispatch>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Timing {
    pub(crate) wall_time_seconds: f64,
    pub(crate) gpu_time_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Profile {
    pub(crate) schema_version: u32,
    pub(crate) device_identity: Value,
    pub(crate) provenance: Value,
    pub(crate) inputs: Value,
    pub(crate) mode: String,
    pub(crate) tokens: u64,
    pub(crate) unprofiled: Timing,
    pub(crate) submissions: Vec<Submission>,
}

pub(crate) fn positive(value: f64) -> Result<f64, Box<dyn Error>> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("analysis requires finite positive timings/ceilings".into())
    }
}

impl Profile {
    pub(crate) fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.schema_version != 2
            || self.tokens == 0
            || !matches!(self.mode.as_str(), "prefill" | "decode")
            || self.submissions.is_empty()
        {
            return Err("unsupported or empty profile; rerun forja profile --json".into());
        }
        positive(self.unprofiled.wall_time_seconds)?;
        positive(self.unprofiled.gpu_time_seconds)?;
        for (index, submission) in self.submissions.iter().enumerate() {
            if submission.index != u64::try_from(index)? || submission.timeline.is_empty() {
                return Err("incomplete submission timeline".into());
            }
            positive(submission.gpu_time_seconds)?;
            for (index, dispatch) in submission.timeline.iter().enumerate() {
                if dispatch.index != u64::try_from(index)?
                    || dispatch.physical_dispatches == 0
                    || dispatch.call_site.is_empty()
                {
                    return Err("invalid dispatch identity/count".into());
                }
                positive(dispatch.gpu_time_seconds)?;
                if !dispatch.gap_before_seconds.is_finite() || dispatch.gap_before_seconds < 0.0 {
                    return Err("invalid dispatch gap".into());
                }
                if !matches!(dispatch.compute_dtype.as_str(), "f16" | "bf16" | "f32") {
                    return Err("unknown arithmetic ceiling".into());
                }
                if let Some(scope) = &dispatch.scope {
                    forja_core::ProfileScope::new(
                        scope.site.clone(),
                        scope.layer_type.clone(),
                        scope.layer,
                    )?;
                }
                validate_cost(dispatch.cost)?;
                if dispatch.padded_flops[0] < dispatch.cost.flops
                    || dispatch.padded_flops[1] < dispatch.padded_flops[0]
                {
                    return Err("invalid padded work bounds".into());
                }
                for kernel in &dispatch.internal {
                    positive(kernel.gpu_time_seconds)?;
                    if let Some(cost) = kernel.cost {
                        validate_cost(cost)?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn validate_cost(cost: KernelCost) -> Result<(), Box<dyn Error>> {
    if (cost.read_bytes == 0 && cost.write_bytes == 0 && cost.flops == 0 && cost.scalar_ops == 0)
        || cost.minimum_weight_bytes > cost.maximum_weight_bytes
        || cost.maximum_weight_bytes > cost.weight_bytes
        || cost.read_bytes < cost.weight_bytes
        || cost.read_bytes.checked_add(cost.write_bytes).is_none()
        || cost.kv_bytes > cost.read_bytes
    {
        return Err("invalid or overflowing declared kernel cost".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn costs_refuse_overflow_and_inconsistent_weight_reuse() {
        assert!(validate_cost(KernelCost::default()).is_err());
        assert!(
            validate_cost(KernelCost {
                read_bytes: 1,
                ..KernelCost::default()
            })
            .is_ok()
        );
        assert!(
            validate_cost(KernelCost {
                read_bytes: u64::MAX,
                write_bytes: 1,
                ..KernelCost::default()
            })
            .is_err()
        );
        assert!(
            validate_cost(KernelCost {
                read_bytes: 7,
                weight_bytes: 8,
                ..KernelCost::default()
            })
            .is_err()
        );
        assert!(positive(f64::NAN).is_err());
        assert!(positive(0.0).is_err());
    }
}
