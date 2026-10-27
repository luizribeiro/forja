use std::{error::Error, fs, path::PathBuf, time::Instant};

use forja_core::Backend;
mod shape;
use shape::{Shape, TensorShape};

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Catalog variant to execute, without automatic algorithm selection.
    pub(crate) variant: String,
    /// JSON operation, tensor shapes, data and optional permutations.
    #[arg(long)]
    pub(crate) shape: PathBuf,
    /// Number of measured single-dispatch submissions.
    #[arg(long, default_value = "5")]
    pub(crate) reps: std::num::NonZeroU32,
    /// Number of untimed submissions.
    #[arg(long, default_value_t = 1)]
    pub(crate) warmups: u32,
    /// Write a kernel diagnostic record.
    #[arg(long)]
    pub(crate) json: PathBuf,
}

pub(crate) fn run(
    options: &Options,
    config: &forja_config::DevConfig,
) -> Result<(), Box<dyn Error>> {
    #[cfg(target_os = "macos")]
    return run_metal(options, config);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (options, config);
        Err("kernel diagnostics require Metal on macOS".into())
    }
}

#[cfg(target_os = "macos")]
fn run_metal(options: &Options, config: &forja_config::DevConfig) -> Result<(), Box<dyn Error>> {
    use forja_core::{BackendDispatchData, CommandList, Submission};
    let shape: Shape = serde_json::from_slice(&fs::read(&options.shape)?)?;
    let (backend, live_bytes) = crate::engine::metal_backend(
        config.backend.metal.resolve().graph_replay,
        config.backend.metal.pool_capacity,
        &config.limits,
    )?;
    let bytes = shape
        .inputs
        .iter()
        .chain(&shape.outputs)
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.layout()?.1)
                .ok_or_else(|| Box::<dyn Error>::from("allocation size overflow"))
        })?;
    if bytes > live_bytes {
        return Err("kernel diagnostic exceeds live-byte quota".into());
    }
    let allocate = |specs: &[TensorShape]| {
        specs
            .iter()
            .map(|spec| spec.allocate(&backend))
            .collect::<Result<Vec<_>, _>>()
    };
    let inputs = allocate(&shape.inputs)?;
    let outputs = allocate(&shape.outputs)?;
    let inputs = inputs.iter().collect::<Vec<_>>();
    let outputs = outputs.iter().collect::<Vec<_>>();
    let variant =
        backend.validate_variant_for_op(shape.operation, &inputs, &outputs, &options.variant)?;
    let mut samples = Vec::new();
    let count = options
        .warmups
        .checked_add(options.reps.get())
        .ok_or("repetition count overflow")?;
    for index in 0..count {
        let mut commands = CommandList::new();
        commands.dispatch_many_with_backend_data(
            shape.operation,
            &inputs,
            &outputs,
            BackendDispatchData::new(variant),
        )?;
        let started = Instant::now();
        let submission = backend.submit(commands)?;
        submission.wait()?;
        if index >= options.warmups {
            samples.push(serde_json::json!({"wall_seconds": started.elapsed().as_secs_f64(),
                "gpu_seconds": submission.gpu_time().ok_or("GPU timing unavailable")?.as_secs_f64()}));
        }
    }
    let report = serde_json::json!({"kind": "kernel-diagnostic", "model_benchmark": false,
        "variant": variant.name(), "shape": serde_json::from_slice::<serde_json::Value>(&fs::read(&options.shape)?)?,
        "warmups": options.warmups, "samples": samples, "provenance": {
            "commit": crate::provenance::commit(), "dirty": crate::provenance::dirty(),
            "binary_sha256": golden_fixtures::sha256_file(std::env::current_exe()?)?,
            "shape_sha256": golden_fixtures::sha256_file(&options.shape)?,
            "device": backend.device_name()}});
    fs::write(&options.json, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "kernel diagnostic: {} ({} reps)",
        variant.name(),
        options.reps
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_diagnostics_execute_explicit_variants() {
        let root = std::env::temp_dir().join(format!("forja-kernel-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for (variant, shape) in [
            (
                "matmul.gemv",
                serde_json::json!({
                "operation": {"kind": "matmul"},
                "inputs": [{"dtype":"f32","shape":[1,7]}, {"dtype":"f32","shape":[7,33]}],
                "outputs": [{"dtype":"f32","shape":[1,33]}]}),
            ),
            (
                "paged-sdpa-append.vector-gqa-two-pass-32",
                serde_json::json!({
                "operation": {"kind":"paged-sdpa-append","scale":0.1,"q_start":6,"page_tokens":32},
                "inputs": [{"dtype":"f32","shape":[16,1,128]},
                    {"dtype":"f32","shape":[1,8,128]}, {"dtype":"f32","shape":[1,8,128]},
                    {"dtype":"u32","shape":[3,2],"u32_values":[2,1,0,1,1,1]},
                    {"dtype":"u32","shape":[3],"u32_values":[1,1,1]}],
                "outputs": [{"dtype":"f32","shape":[16,1,128]},
                    {"dtype":"f16","shape":[3,32,8,128]}, {"dtype":"f16","shape":[3,32,8,128]}]}),
            ),
            (
                "paged-sdpa.vector",
                serde_json::json!({
                "operation": {"kind":"paged-sdpa","scale":0.1,"causal":true,"q_start":6,
                              "key_length":7,"page_tokens":32},
                "inputs": [{"dtype":"f32","shape":[16,1,128]},
                    {"dtype":"f16","shape":[3,32,8,128]}, {"dtype":"f16","shape":[3,32,8,128]},
                    {"dtype":"u32","shape":[3,2],"u32_values":[2,1,0,1,1,1]},
                    {"dtype":"u32","shape":[3],"u32_values":[1,1,1]}],
                "outputs": [{"dtype":"f32","shape":[16,1,128]}]}),
            ),
        ] {
            let options = Options {
                variant: variant.into(),
                shape: root.join("shape.json"),
                json: root.join("record.json"),
                reps: std::num::NonZeroU32::new(1).unwrap(),
                warmups: 0,
            };
            fs::write(&options.shape, serde_json::to_vec(&shape).unwrap()).unwrap();
            run(&options, &forja_config::DevConfig::default()).unwrap();
            let record: serde_json::Value =
                serde_json::from_slice(&fs::read(&options.json).unwrap()).unwrap();
            assert_eq!(record["kind"], "kernel-diagnostic");
            assert_eq!(record["variant"], variant);
            assert_eq!(record["model_benchmark"], false);
            assert_eq!(record["samples"].as_array().unwrap().len(), 1);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
