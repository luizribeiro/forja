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
            "shape": super::shape(dispatch), "cost": dispatch.cost.json(), "scheduled_cost": dispatch.scheduled_cost.json(), "critical_path_cost": dispatch.critical_path_cost.json(), "padded_flops": dispatch.padded_flops,
            "physical_dispatches": dispatch.physical_dispatches, "physical_threadgroups": dispatch.physical_threadgroups,
            "predicted_gpu_seconds": dispatch.predicted_gpu.as_secs_f64(),
            "measured_predicted_ratio": (!dispatch.predicted_gpu.is_zero()).then(|| dispatch.gpu_time.as_secs_f64()/dispatch.predicted_gpu.as_secs_f64()),
            "register_spill_risk": register_pressure(dispatch, dispatch.kernel),
            "compute_dtype": super::dtype_name(dispatch.compute_dtype),
            "start_seconds": dispatch.gpu_start.as_secs_f64(), "end_seconds": dispatch.gpu_end.as_secs_f64(),
            "gap_before_seconds": dispatch.gap_before.as_secs_f64(), "gpu_time_seconds": dispatch.gpu_time.as_secs_f64(),
            "barrier_before": dispatch.barrier, "bytes": bytes,
            "gigabytes_per_second": super::throughput(bytes, dispatch.gpu_time),
            "internal": dispatch.internal.iter().map(|kernel| json!({"kernel": kernel.kernel,
                "cost": kernel.cost.map(forja_core::KernelCost::json), "register_spill_risk": register_pressure(dispatch, kernel.kernel), "start_seconds": kernel.gpu_start.as_secs_f64(),
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
            "gpu_time_seconds": submission.gpu_time.as_secs_f64(),
            "buffer_admission": submission.buffer_admission.iter().map(|buffer| json!({
                "predicted_gpu_seconds": buffer.predicted_gpu.as_secs_f64(),
                "reserve_seconds": buffer.reserve.as_secs_f64(),
                "gpu_time_seconds": buffer.gpu_time.as_secs_f64(),
                "backstop_ratio": buffer.backstop_ratio,
            })).collect::<Vec<_>>(),
            "admission_calibration": submission.work_calibration.as_ref().map(|c| c.json()),
            "predicted_gpu_seconds": submission.predicted_gpu.as_secs_f64(), "command_buffer_reserve_seconds":submission.command_buffer_reserve.as_secs_f64(), "command_buffers": submission.command_buffers,
            "host_phases": {
                "validation_seconds": submission.validation.as_secs_f64(),
                "encoding_seconds": submission.encoding.as_secs_f64(),
                "metadata_seconds": submission.metadata_buffers.time.as_secs_f64(),
                "metadata_writes": submission.metadata_buffers.count,
                "residency_seconds": submission.residency.as_secs_f64(),
                "commit_seconds": submission.commit.as_secs_f64(),
                "explicit_wait_seconds": submission.wait.as_secs_f64(),
            }, "timeline": timeline(&submission.per_dispatch)? }))
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

fn register_pressure(dispatch: &DispatchProfile, kernel: &str) -> Value {
    #[cfg(target_os = "macos")]
    let values = forja_metal::private_accumulator_values_per_thread(dispatch, kernel);
    #[cfg(not(target_os = "macos"))]
    let values = {
        let _ = (dispatch, kernel);
        None::<u64>
    };
    values.map_or(Value::Null, |values| json!({
        "private_f32_accumulator_values_per_thread": values,
        "basis": "source-level persistent carry/result arrays; excludes temporaries and compiler allocation",
        "spill_risk": "private arrays may spill; confirm with compiler/device profiling",
        "spill_observed": null, "register_ceiling": "uncalibrated"
    }))
}

pub(super) fn host_step(step: &forja_host::EngineStepProfile) -> Value {
    let imports = &step.imports;
    let calls = [
        ("alloc", imports.alloc),
        ("slice", imports.view_slice),
        ("reshape", imports.view_reshape),
        ("permute", imports.view_permute),
        ("broadcast", imports.view_broadcast),
        ("write", imports.write),
        ("dispatch", imports.dispatch),
        ("submit", imports.submit),
        ("read", imports.read),
        ("command_list", imports.command_list),
        ("resource_drop", imports.resource_drop),
        ("backend_allocations", step.allocations),
        ("backend_releases", step.releases),
        ("output_read", step.output_read),
    ]
    .map(|(name, profile)| {
        (
            name,
            json!({"calls": profile.count, "seconds": profile.time.as_secs_f64()}),
        )
    })
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    json!({"wall_seconds": step.wall_time.as_secs_f64(), "guest_seconds": step.guest_time.as_secs_f64(), "calls": calls,
        "scopes": "import and backend intervals are nested; GPU and host waits overlap"})
}
