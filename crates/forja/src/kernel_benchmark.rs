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
    /// Number of measured submissions.
    #[arg(long, default_value = "5")]
    pub(crate) reps: std::num::NonZeroU32,
    /// Independent operand sets, visited cyclically within each submission.
    #[arg(long, default_value = "1")]
    pub(crate) copies: std::num::NonZeroU32,
    /// Dispatches per measured submission, amortizing command-buffer latency.
    #[arg(long, default_value = "1")]
    pub(crate) dispatches: std::num::NonZeroU32,
    /// Reuse the first output to serialize dispatches through write dependencies.
    #[arg(long)]
    pub(crate) serial: bool,
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
    use forja_core::Submission;
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
    let bytes = bytes
        .checked_mul(u64::from(options.copies.get()))
        .ok_or("operand-set size overflow")?;
    if bytes > live_bytes {
        return Err("kernel diagnostic exceeds live-byte quota".into());
    }
    let commands = diagnostic_commands(&backend, &shape, options)?;
    let cost = commands
        .last_dispatch()
        .and_then(forja_core::KernelCost::of);
    let mut samples = Vec::new();
    let count = options
        .warmups
        .checked_add(options.reps.get())
        .ok_or("repetition count overflow")?;
    for index in 0..count {
        let started = Instant::now();
        let submission = backend.submit(commands.clone())?;
        submission.wait()?;
        if index >= options.warmups {
            samples.push(serde_json::json!({"wall_seconds": started.elapsed().as_secs_f64(),
                "gpu_seconds": submission.gpu_time().ok_or("GPU timing unavailable")?.as_secs_f64()}));
        }
    }
    let report = serde_json::json!({"kind": "kernel-diagnostic", "model_benchmark": false,
        "variant": options.variant, "serial": options.serial, "copies": options.copies.get(), "resident_operand_bytes": bytes,
        "dispatches_per_sample": options.dispatches.get(), "cost_per_dispatch": cost.map(forja_core::KernelCost::json), "shape": serde_json::from_slice::<serde_json::Value>(&fs::read(&options.shape)?)?,
        "warmups": options.warmups, "samples": samples, "provenance": {
            "commit": crate::provenance::commit(), "dirty": crate::provenance::dirty(),
            "binary_sha256": golden_fixtures::sha256_file(std::env::current_exe()?)?,
            "shape_sha256": golden_fixtures::sha256_file(&options.shape)?,
            "device": backend.device_name()}});
    fs::write(&options.json, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "kernel diagnostic: {} ({} reps)",
        options.variant, options.reps
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn diagnostic_commands(
    backend: &forja_metal::MetalBackend,
    shape: &Shape,
    options: &Options,
) -> Result<forja_core::CommandList, Box<dyn Error>> {
    use forja_core::{BackendDispatchData, CommandList};
    let allocate = |specs: &[TensorShape]| {
        specs
            .iter()
            .map(|spec| spec.allocate(backend))
            .collect::<Result<Vec<_>, _>>()
    };
    if options.dispatches.get() < options.copies.get() {
        return Err("dispatches must visit every operand set".into());
    }
    let operands = (0..options.copies.get())
        .map(|_| Ok((allocate(&shape.inputs)?, allocate(&shape.outputs)?)))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let epilogue = shape.prepare_epilogue()?;
    let mut commands = CommandList::new();
    for index in 0..options.dispatches.get() {
        let (inputs, outputs) = &operands[usize::try_from(index % options.copies.get())?];
        let inputs = inputs.iter().collect::<Vec<_>>();
        let outputs = outputs
            .iter()
            .enumerate()
            .map(|(index, output)| {
                if options.serial && index == 0 {
                    &operands[0].1[0]
                } else {
                    output
                }
            })
            .collect::<Vec<_>>();
        let variant = backend.validate_variant_for_op(
            shape.operation,
            &inputs,
            &outputs,
            &options.variant,
        )?;
        if let Some(epilogue) = &epilogue {
            commands.dispatch_epilogue(shape.operation, &inputs, outputs[0], epilogue, &[])?;
            if !commands.set_last_backend_data(BackendDispatchData::new(variant)) {
                return Err("missing diagnostic dispatch".into());
            }
        } else {
            commands.dispatch_many_with_backend_data(
                shape.operation,
                &inputs,
                &outputs,
                BackendDispatchData::new(variant),
            )?;
        }
    }
    Ok(commands)
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
                "gather-quant-matmul.vector8",
                serde_json::json!({
                    "operation": {"kind":"gather-quant-matmul","bits":4,"group_size":64},
                    "epilogue":"silu-mul",
                    "inputs":[{"dtype":"f32","shape":[1,768]},
                        {"dtype":"u32","shape":[2,33,96]}, {"dtype":"f16","shape":[2,33,12]}, {"dtype":"f16","shape":[2,33,12]},
                        {"dtype":"u32","shape":[2,33,96]}, {"dtype":"f16","shape":[2,33,12]}, {"dtype":"f16","shape":[2,33,12]},
                        {"dtype":"u32","shape":[1,2],"u32_values":[0,1]}],
                    "outputs":[{"dtype":"f32","shape":[1,2,33]}]
                }),
            ),
            (
                "gather-quant-matmul.vector8",
                serde_json::json!({
                    "operation": {"kind":"gather-quant-matmul","bits":4,"group_size":64},
                    "epilogue":"identity",
                    "inputs":[{"dtype":"f32","shape":[1,2,768]},
                        {"dtype":"u32","shape":[2,33,96]}, {"dtype":"f16","shape":[2,33,12]}, {"dtype":"f16","shape":[2,33,12]},
                        {"dtype":"u32","shape":[1,2],"u32_values":[0,1]}, {"dtype":"f32","shape":[1,2]}],
                    "outputs":[{"dtype":"f32","shape":[1,33]}]
                }),
            ),
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
            assert_diagnostic_record(&root, variant, &shape);
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_diagnostics_authenticate_delta_block_stages() {
        let root = std::env::temp_dir().join(format!("forja-block-kernel-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let scratch = forja_core::DeltaBlockScratch::new(7, 3, 7, 33)
            .unwrap()
            .elements();
        let shape = serde_json::json!({
            "operation": {"kind":"state", "operation":"delta-rule-block"},
            "inputs": [{"dtype":"f32","shape":[7,1,7],"seed":1},
                {"dtype":"f32","shape":[7,1,7],"seed":2}, {"dtype":"f32","shape":[7,3,33],"seed":3},
                {"dtype":"f32","shape":[7,3]}, {"dtype":"f32","shape":[7,3]}, {"dtype":"f32","shape":[3,7,33]},
                {"dtype":"u32","shape":[2,2],"u32_values":[0,1,1,1]},
                {"dtype":"u32","shape":[3],"u32_values":[1,1,1]}, {"dtype":"u32","shape":[1],"u32_values":[1]}],
            "outputs": [{"dtype":"f32","shape":[7,3,33]}, {"dtype":"f32","shape":[3,7,33]},
                {"dtype":"f32","shape":[scratch]}]
        });
        for variant in [
            "delta-rule-block.wy64-mma-v16",
            "delta-rule-block.wy64-mma-v32",
        ] {
            assert_diagnostic_record(&root, variant, &shape);
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[cfg(target_os = "macos")]
    fn assert_diagnostic_record(root: &std::path::Path, variant: &str, shape: &serde_json::Value) {
        let mut options = Options {
            variant: variant.into(),
            shape: root.join("shape.json"),
            json: root.join("record.json"),
            reps: std::num::NonZeroU32::new(1).unwrap(),
            warmups: 0,
            serial: true,
            copies: std::num::NonZeroU32::new(3).unwrap(),
            dispatches: std::num::NonZeroU32::new(7).unwrap(),
        };
        fs::write(&options.shape, serde_json::to_vec(shape).unwrap()).unwrap();
        let backend = forja_metal::MetalBackend::new().unwrap();
        let spec = serde_json::from_value(shape.clone()).unwrap();
        let commands = diagnostic_commands(&backend, &spec, &options).unwrap();
        assert_eq!(
            forja_core::required_barriers(&commands),
            [false, true, true, true, true, true, true]
        );
        run(&options, &forja_config::DevConfig::default()).unwrap();
        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(&options.json).unwrap()).unwrap();
        assert_eq!(record["kind"], "kernel-diagnostic");
        assert_eq!(record["variant"], variant);
        assert_eq!(record["model_benchmark"], false);
        assert_eq!(record["serial"], true);
        assert_eq!(record["copies"], 3);
        assert_eq!(record["dispatches_per_sample"], 7);
        options.dispatches = std::num::NonZeroU32::new(2).unwrap();
        assert!(run(&options, &forja_config::DevConfig::default()).is_err());
        assert_eq!(record["samples"].as_array().unwrap().len(), 1);
    }
}
