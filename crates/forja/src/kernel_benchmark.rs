use std::{error::Error, fs, path::PathBuf, time::Instant};

use forja_core::Backend;
mod shape;
use shape::{Shape, TensorShape};

#[cfg(any(target_os = "macos", test))]
const STATE_OUTPUT_INDEX: usize = 1;

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
    /// Chain this many single-token state updates through distinct outputs.
    #[arg(long, default_value = "1")]
    pub(crate) state_steps: std::num::NonZeroU32,
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
    let dispatches = effective_dispatches(options)?;
    let sum_bytes = |specs: &[TensorShape]| {
        specs.iter().try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.layout()?.1)
                .ok_or_else(|| Box::<dyn Error>::from("allocation size overflow"))
        })
    };
    let inputs = sum_bytes(&shape.inputs)?;
    let bytes = sum_bytes(&shape.outputs)?
        .checked_mul(u64::from(options.state_steps.get()))
        .and_then(|outputs| outputs.checked_add(inputs))
        .ok_or("operand size overflow")?;
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
    let resources = commands
        .last_dispatch()
        .map(|dispatch| backend.kernel_resources(dispatch))
        .transpose()?
        .flatten();
    let report = serde_json::json!({"kind": "kernel-diagnostic", "model_benchmark": false,
        "variant": options.variant, "serial": options.serial, "copies": options.copies.get(), "resident_operand_bytes": bytes,
        "state_steps": options.state_steps.get(), "dispatches_per_sample": dispatches.get(), "cost_per_dispatch": cost.map(forja_core::KernelCost::json), "shape": serde_json::from_slice::<serde_json::Value>(&fs::read(&options.shape)?)?,
        "pipeline_resources": resources.map(|resources| serde_json::json!({
            "kernel": resources.kernel, "execution_width": resources.execution_width,
            "maximum_threadgroup_threads": resources.maximum_threads,
            "static_threadgroup_bytes": resources.static_threadgroup_bytes,
            "observed_registers_per_thread": null, "observed_occupancy": null,
            "basis": "compiled pipeline limits; not measured occupancy or register allocation"
        })),
        "rates": cost.map(|cost| diagnostic_rates(cost, dispatches, &samples)).transpose()?,
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

#[cfg(any(target_os = "macos", test))]
#[allow(clippy::cast_precision_loss)]
fn diagnostic_rates(
    cost: forja_core::KernelCost,
    dispatches: std::num::NonZeroU32,
    samples: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let bytes = cost
        .read_bytes
        .checked_add(cost.write_bytes)
        .ok_or("cost overflow")?;
    samples.iter().map(|sample| {
        let seconds = sample["gpu_seconds"].as_f64().filter(|seconds| seconds.is_finite() && *seconds > 0.0).ok_or("invalid GPU duration")? / f64::from(dispatches.get());
        Ok(serde_json::json!({"gpu_seconds_per_dispatch": seconds,
            "logical_gigabytes_per_second": bytes as f64 / seconds / 1e9,
            "logical_teraflops_per_second": cost.flops as f64 / seconds / 1e12,
            "basis": "logical operation work and operands, not measured DRAM transactions or instruction throughput"}))
    }).collect()
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
    validate_operand_visits(options.dispatches, options.copies)?;
    if options.state_steps.get() > 1
        && !matches!(
            shape.operation,
            forja_core::Op::State(
                forja_core::StateOp::DeltaRule(_) | forja_core::StateOp::CausalDepthwiseConv(_)
            )
        )
    {
        return Err("state steps require single-token state operations".into());
    }
    if options.state_steps.get() > 1
        && matches!(
            shape.operation,
            forja_core::Op::State(forja_core::StateOp::CausalDepthwiseConv(_))
        )
        && shape
            .inputs
            .first()
            .ok_or("missing convolution tokens")?
            .dimensions()
            .first()
            != Some(&1)
    {
        return Err("state steps require one convolution token".into());
    }
    let chained_input = (options.state_steps.get() > 1)
        .then(|| chain_state_input(shape.operation, &shape.inputs))
        .transpose()?;
    let operands = (0..options.copies.get())
        .map(|_| {
            Ok((
                allocate(&shape.inputs)?,
                (0..options.state_steps.get())
                    .map(|_| allocate(&shape.outputs))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let epilogue = shape.prepare_epilogue()?;
    let mut commands = CommandList::new();
    for index in 0..options.dispatches.get() {
        let (inputs, outputs) = &operands[usize::try_from(index % options.copies.get())?];
        for step in 0..usize::try_from(options.state_steps.get())? {
            let inputs = inputs
                .iter()
                .enumerate()
                .map(|(index, tensor)| {
                    if step > 0 && Some(index) == chained_input {
                        &outputs[step - 1][STATE_OUTPUT_INDEX]
                    } else {
                        tensor
                    }
                })
                .collect::<Vec<_>>();
            let outputs = outputs[step]
                .iter()
                .enumerate()
                .map(|(index, output)| {
                    if options.serial && index == 0 {
                        &operands[0].1[0][0]
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
    }
    Ok(commands)
}

#[cfg(any(target_os = "macos", test))]
fn chain_state_input<T>(operation: forja_core::Op, inputs: &[T]) -> Result<usize, Box<dyn Error>> {
    let forja_core::Op::State(state) = operation else {
        return Err("state steps require a state operation".into());
    };
    let suffix = state.input_suffix(inputs)?;
    inputs
        .len()
        .checked_sub(suffix.len())
        .ok_or_else(|| "missing state input suffix".into())
}

#[cfg(any(target_os = "macos", test))]
fn validate_operand_visits(
    dispatches: std::num::NonZeroU32,
    copies: std::num::NonZeroU32,
) -> Result<(), Box<dyn Error>> {
    if dispatches < copies {
        return Err("dispatches must visit every operand set".into());
    }
    Ok(())
}

fn effective_dispatches(options: &Options) -> Result<std::num::NonZeroU32, Box<dyn Error>> {
    options
        .dispatches
        .checked_mul(options.state_steps)
        .ok_or_else(|| "dispatch count overflow".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chained_state_inputs_require_the_complete_suffix() {
        for (kind, arity, source) in [("delta-rule", 9, 5), ("causal-depthwise-conv", 6, 2)] {
            let shape: Shape = serde_json::from_value(serde_json::json!({
                "operation":{"kind":"state","operation":kind},"inputs":[],"outputs":[]
            }))
            .unwrap();
            for len in 0..arity {
                assert!(chain_state_input(shape.operation, &vec![0; len]).is_err());
            }
            let mut inputs = (0..arity).collect::<Vec<_>>();
            let index = chain_state_input(shape.operation, &inputs).unwrap();
            assert_eq!(index, source);
            let previous_outputs = [7, 99];
            inputs[index] = previous_outputs[STATE_OUTPUT_INDEX];
            assert_eq!(inputs[index], 99);
            assert_eq!(
                &inputs[index + 1..],
                &(source + 1..arity).collect::<Vec<_>>()
            );
        }
        assert!(chain_state_input(forja_core::Op::Matmul, &[0; 9]).is_err());
    }

    #[test]
    fn diagnostic_dispatches_visit_every_operand_set() {
        let count = |n| std::num::NonZeroU32::new(n).unwrap();
        assert!(validate_operand_visits(count(1), count(2)).is_err());
        assert!(validate_operand_visits(count(2), count(2)).is_ok());
        assert!(validate_operand_visits(count(7), count(2)).is_ok());
    }

    #[test]
    fn state_dispatch_counts_refuse_overflow() {
        let mut options = state_chain_options();
        assert_eq!(effective_dispatches(&options).unwrap().get(), 3);
        options.dispatches = std::num::NonZeroU32::MAX;
        assert!(effective_dispatches(&options).is_err());
    }

    #[test]
    fn diagnostic_rates_check_costs_and_normalize_dispatches() {
        let count = std::num::NonZeroU32::new(4).unwrap();
        let cost = forja_core::KernelCost {
            read_bytes: 8,
            write_bytes: 8,
            flops: 100,
            ..Default::default()
        };
        let rates =
            diagnostic_rates(cost, count, &[serde_json::json!({"gpu_seconds": 1.0})]).unwrap();
        assert_eq!(rates[0]["gpu_seconds_per_dispatch"], 0.25);
        assert_eq!(rates[0]["logical_gigabytes_per_second"], 64.0 / 1e9);
        for sample in [
            serde_json::json!({}),
            serde_json::json!({"gpu_seconds": 0.0}),
            serde_json::json!({"gpu_seconds": -1.0}),
        ] {
            assert!(diagnostic_rates(cost, count, &[sample]).is_err());
        }
        let overflow = forja_core::KernelCost {
            read_bytes: u64::MAX,
            write_bytes: 1,
            ..Default::default()
        };
        assert!(diagnostic_rates(overflow, count, &[]).is_err());
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_kernel_diagnostics_execute_explicit_variants() {
        let root = std::env::temp_dir().join(format!("forja-kernel-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for (variant, shape) in [
            (
                "quant-matmul.vector8-shared",
                serde_json::json!({"operation":{"kind":"quant-matmul","bits":4,"group_size":64},
                    "inputs":[{"dtype":"bf16","shape":[4,64]}, {"dtype":"u32","shape":[33,8]},
                    {"dtype":"f16","shape":[33,1]}, {"dtype":"f16","shape":[33,1]}],
                    "outputs":[{"dtype":"bf16","shape":[4,33]}]}),
            ),
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
    fn metal_kernel_diagnostics_chain_state_outputs() {
        use forja_core::Submission;
        let shape: Shape = serde_json::from_value(serde_json::json!({
            "operation":{"kind":"state","operation":"delta-rule"},
            "inputs":[{"dtype":"f32","shape":[1,1],"seed":1},
                {"dtype":"f32","shape":[1,1],"seed":2},
                {"dtype":"f32","shape":[1,1],"seed":3},
                {"dtype":"f32","shape":[1],"seed":4},
                {"dtype":"f32","shape":[1],"seed":5},
                {"dtype":"f32","shape":[1,1,1]},
                {"dtype":"u32","shape":[2,2],"u32_values":[0,1,1,1]},
                {"dtype":"u32","shape":[3],"u32_values":[1,1,1]},
                {"dtype":"u32","shape":[1],"u32_values":[1]}],
            "outputs":[{"dtype":"f32","shape":[1,1]},
                {"dtype":"f32","shape":[1,1,1]}]}))
        .unwrap();
        let options = state_chain_options();
        let backend = forja_metal::MetalBackend::new().unwrap();
        let commands = diagnostic_commands(&backend, &shape, &options).unwrap();
        assert_eq!(
            forja_core::required_barriers(&commands),
            [false, true, true]
        );
        backend.submit(commands.clone()).unwrap().wait().unwrap();
        let read = |tensor| f32::from_le_bytes(backend.read(tensor).unwrap().try_into().unwrap());
        let inputs = commands.dispatches()[0].inputs();
        let (key, value, decay, beta) = (
            read(&inputs[1]),
            read(&inputs[2]),
            read(&inputs[3]),
            read(&inputs[4]),
        );
        let mut expected = 0.0;
        for dispatch in commands.dispatches() {
            expected *= decay;
            expected += (value - expected * key) * beta * key;
            let actual = read(&dispatch.outputs()[STATE_OUTPUT_INDEX]);
            assert!(
                golden_fixtures::normwise_relative_error(&[expected], &[actual]).unwrap() <= 1e-5
            );
        }
    }

    fn state_chain_options() -> Options {
        Options {
            variant: "delta-rule.simd".into(),
            shape: PathBuf::new(),
            json: PathBuf::new(),
            reps: std::num::NonZeroU32::new(1).unwrap(),
            warmups: 0,
            serial: false,
            copies: std::num::NonZeroU32::new(1).unwrap(),
            dispatches: std::num::NonZeroU32::new(1).unwrap(),
            state_steps: std::num::NonZeroU32::new(3).unwrap(),
        }
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
            state_steps: std::num::NonZeroU32::new(1).unwrap(),
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
        assert_eq!(record["rates"].as_array().unwrap().len(), 1);
        if variant == "quant-matmul.vector8-shared" {
            assert_eq!(record["pipeline_resources"]["execution_width"], 32);
            assert!(
                record["pipeline_resources"]["maximum_threadgroup_threads"]
                    .as_u64()
                    .unwrap()
                    >= 128
            );
            assert!(record["pipeline_resources"]["observed_occupancy"].is_null());
        }
    }
}
