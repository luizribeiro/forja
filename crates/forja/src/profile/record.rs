use crate::args::{Profile, ProfileMode};
use forja_core::{DispatchProfile, Op, SubmissionProfile};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error};

pub(super) fn timeline(dispatches: &[DispatchProfile]) -> Result<Vec<Value>, &'static str> {
    let mut ordinals = BTreeMap::<String, u64>::new();
    dispatches.iter().enumerate().map(|(index, dispatch)| {
        let scope = dispatch.scope.as_ref().map_or_else(|| "unscoped".to_owned(), |scope| {
            scope.layer().map_or_else(|| scope.site().to_owned(), |layer| format!("{}/layer-{layer}", scope.site()))
        });
        let ordinal = ordinals.entry(scope.clone()).or_default();
        let call_site = format!("{scope}/dispatch-{ordinal}");
        *ordinal = ordinal.checked_add(1).ok_or("call-site count overflow")?;
        let bytes = dispatch.cost.read_bytes.checked_add(dispatch.cost.write_bytes).ok_or("cost overflow")?;
        Ok(json!({"index": index, "call_site": call_site, "scope": dispatch.scope.as_ref().map(forja_core::ProfileScope::json),
            "op": super::op_name(dispatch.op), "kernel": dispatch.kernel,
            "variant": dispatch.variant, "alternatives": dispatch.alternatives,
            "shape": super::shape(dispatch), "cost": dispatch.cost.json(), "padded_flops": dispatch.padded_flops,
            "physical_dispatches": dispatch.physical_dispatches,
            "compute_dtype": super::dtype_name(dispatch.compute_dtype),
            "start_seconds": dispatch.gpu_start.as_secs_f64(), "end_seconds": dispatch.gpu_end.as_secs_f64(),
            "gap_before_seconds": dispatch.gap_before.as_secs_f64(), "gpu_time_seconds": dispatch.gpu_time.as_secs_f64(),
            "barrier_before": dispatch.barrier, "bytes": bytes,
            "gigabytes_per_second": super::throughput(bytes, dispatch.gpu_time),
            "internal": dispatch.internal.iter().map(|kernel| json!({"kernel": kernel.kernel,
                "cost": kernel.cost.map(forja_core::KernelCost::json), "start_seconds": kernel.gpu_start.as_secs_f64(),
                "end_seconds": kernel.gpu_end.as_secs_f64(), "gpu_time_seconds": kernel.gpu_time.as_secs_f64()})).collect::<Vec<_>>() }))
    }).collect()
}

pub(super) fn submissions(
    mode: ProfileMode,
    submissions: &[SubmissionProfile],
) -> Result<Vec<Value>, &'static str> {
    submissions.iter().enumerate().map(|(index, submission)| {
        let (mut context, mut queries) = (0, 0);
        for dispatch in &submission.per_dispatch {
            match dispatch.op {
                Op::PagedSdpa { key_length, .. } => { context = context.max(key_length); }
                Op::Sdpa { .. } => { context = context.max(dispatch.inputs[1].shape[1]); }
                _ => continue,
            }
            queries = queries.max(dispatch.inputs[0].shape[1]);
        }
        Ok(json!({"index": index, "phase": {"mode": super::mode_name(mode), "context": context, "query_tokens": queries},
            "gpu_time_seconds": submission.gpu_time.as_secs_f64(), "timeline": timeline(&submission.per_dispatch)? }))
    }).collect()
}

pub(super) fn inputs(options: &Profile) -> Result<Value, Box<dyn Error>> {
    let profile = crate::resolution::read_embedded_profile(&options.engine)?;
    Ok(
        json!({"component_sha256": golden_fixtures::sha256_file(&options.engine)?,
        "profile_sha256": profile.sha256()?, "model": profile.model(), "numerics": profile.numerics(),
        "engine_config": options.engine_config, "graph_replay": options.graph_replay,
        "warmups": options.warmups, "sampling": options.sampling,
        "model_path": options.model_dir, "weights_hash_source": "embedded-profile" }),
    )
}
