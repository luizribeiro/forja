use std::{error::Error, fs, path::PathBuf, time::Instant};

use forja_core::Backend;
#[cfg(any(target_os = "macos", test))]
mod row_chunks;
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
    /// Independent operand sets, visited cyclically unless advancing positions.
    #[arg(long, default_value = "1")]
    pub(crate) copies: std::num::NonZeroU32,
    /// Dispatches per measured submission, amortizing command-buffer latency.
    #[arg(long, default_value = "1")]
    pub(crate) dispatches: std::num::NonZeroU32,
    /// Chain this many single-token state updates through distinct outputs.
    #[arg(long, default_value = "1")]
    pub(crate) state_steps: std::num::NonZeroU32,
    /// Advance paged attention by one position on each visit to an operand set.
    #[arg(long)]
    pub(crate) advance_position: bool,
    /// Split dense matrix rows into bounded dispatches sharing the weights.
    #[arg(long)]
    pub(crate) row_chunk: Option<std::num::NonZeroU32>,
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
    crate::benchmark_guard::require_calibration()?;
    let shape: Shape = serde_json::from_slice(&fs::read(&options.shape)?)?;
    let (backend, live_bytes) = crate::engine::metal_backend(
        config.backend.metal.resolve().graph_replay,
        config.backend.metal.pool_capacity,
        &config.limits,
    )?;
    let backend = crate::benchmark_guard::configure(backend)?;
    let dispatches = effective_dispatches(options)?;
    let sum_bytes = |specs: &[TensorShape]| {
        specs.iter().try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.layout()?.1)
                .ok_or_else(|| Box::<dyn Error>::from("allocation size overflow"))
        })
    };
    let inputs = sum_bytes(&shape.inputs)?
        .checked_add(
            shape
                .append
                .as_ref()
                .map_or(Ok(0), |rows| sum_bytes(rows))?,
        )
        .ok_or("append operand size overflow")?;
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
    let actual_dispatches = u32::try_from(commands.dispatches().len())?;
    let primary_dispatches = if shape.append.is_some() {
        dispatches.get()
    } else {
        actual_dispatches
    };
    let rate_dispatches =
        std::num::NonZeroU32::new(actual_dispatches).ok_or("diagnostic requires a dispatch")?;
    let cost = if options.advance_position {
        None
    } else {
        uniform_dispatch_cost(&commands)
    };
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
    let profiled = backend.submit_profiled(commands.clone())?;
    profiled.wait()?;
    let work = profiled
        .profile()
        .ok_or("kernel work profile unavailable")?;
    let resources = commands
        .last_dispatch()
        .map(|dispatch| backend.kernel_resources(dispatch))
        .transpose()?
        .flatten();
    let report = serde_json::json!({"kind": "kernel-diagnostic", "model_benchmark": false,
        "roofline_guard": crate::benchmark_guard::record()?,
        "work_profile": work_profile(&work),
        "variant": options.variant, "serial": options.serial, "advance_position": options.advance_position, "operand_order": if options.advance_position {"consecutive"} else {"cyclic"}, "copies": options.copies.get(), "resident_operand_bytes": bytes,
        "state_steps": options.state_steps.get(), "row_chunk": options.row_chunk.map(std::num::NonZeroU32::get), "dispatches_per_sample": actual_dispatches, "primary_dispatches_per_sample": primary_dispatches, "cost_per_dispatch": cost.map(forja_core::KernelCost::json), "shape": serde_json::from_slice::<serde_json::Value>(&fs::read(&options.shape)?)?,
        "pipeline_resources": resources.map(|resources| serde_json::json!({
            "kernel": resources.kernel, "execution_width": resources.execution_width,
            "maximum_threadgroup_threads": resources.maximum_threads,
            "static_threadgroup_bytes": resources.static_threadgroup_bytes,
            "observed_registers_per_thread": null, "observed_occupancy": null,
            "basis": "compiled pipeline limits; not measured occupancy or register allocation"
        })),
        "rates": cost.map(|cost| diagnostic_rates(cost, rate_dispatches, &samples)).transpose()?,
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
fn work_profile(profile: &forja_core::SubmissionProfile) -> serde_json::Value {
    serde_json::json!({
        "basis": "separate precise diagnostic submission; excluded from benchmark samples",
        "calibration": profile.work_calibration.as_ref().map(|rates| rates.json()),
        "predicted_gpu_seconds": profile.predicted_gpu.as_secs_f64(),
        "dispatches": profile.per_dispatch.iter().map(|d| serde_json::json!({
            "kernel": d.kernel, "variant": d.variant, "physical_dispatches": d.physical_dispatches,
            "scheduled_cost": d.scheduled_cost.json(), "critical_path_cost": d.critical_path_cost.json(),
            "predicted_gpu_seconds": d.predicted_gpu.as_secs_f64(),
            "gpu_time_seconds": d.gpu_time.as_secs_f64(),
        })).collect::<Vec<_>>(),
        "buffers": profile.buffer_admission.iter().map(|b| serde_json::json!({
            "predicted_gpu_seconds": b.predicted_gpu.as_secs_f64(),
            "reserve_seconds": b.reserve.as_secs_f64(), "gpu_time_seconds": b.gpu_time.as_secs_f64(),
            "backstop_ratio": b.backstop_ratio,
        })).collect::<Vec<_>>()
    })
}

#[cfg(any(target_os = "macos", test))]
fn uniform_dispatch_cost(commands: &forja_core::CommandList) -> Option<forja_core::KernelCost> {
    let cost = commands
        .dispatches()
        .first()
        .and_then(forja_core::KernelCost::of)?;
    commands
        .dispatches()
        .iter()
        .all(|dispatch| forja_core::KernelCost::of(dispatch) == Some(cost))
        .then_some(cost)
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
    use forja_core::CommandList;
    let allocate = |specs: &[TensorShape]| {
        specs
            .iter()
            .map(|spec| spec.allocate(backend))
            .collect::<Result<Vec<_>, _>>()
    };
    validate_diagnostic_options(shape, options)?;
    let chained_input = (options.state_steps.get() > 1)
        .then(|| chain_state_input(shape.operation, &shape.inputs))
        .transpose()?;
    let group = if options.advance_position {
        consecutive_rows(options.dispatches, options.copies)?
    } else {
        options.copies
    };
    let operands = (0..options.copies.get())
        .map(|_| {
            Ok((
                allocate(&shape.inputs)?,
                (0..options.state_steps.get())
                    .map(|_| allocate(&shape.outputs))
                    .collect::<Result<Vec<_>, _>>()?,
                shape
                    .append
                    .as_ref()
                    .map(|rows| allocate(rows))
                    .transpose()?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let epilogue = shape.prepare_epilogue()?;
    let mut commands = CommandList::new();
    for index in 0..options.dispatches.get() {
        let (operand, position) = diagnostic_visit(index, group, options.advance_position);
        let (inputs, outputs, append) = &operands[usize::try_from(operand)?];
        let operation = if options.advance_position {
            advance_attention(shape.operation, position)?
        } else {
            shape.operation
        };
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
            if let Some(rows) = append {
                record_append(backend, &mut commands, operation, &inputs, rows)?;
            }
            record_diagnostic_dispatch(
                backend,
                &mut commands,
                operation,
                &inputs,
                &outputs,
                epilogue.as_ref(),
                options,
            )?;
        }
    }
    Ok(commands)
}

#[cfg(target_os = "macos")]
fn record_diagnostic_dispatch(
    backend: &forja_metal::MetalBackend,
    commands: &mut forja_core::CommandList,
    operation: forja_core::Op,
    inputs: &[&forja_core::Tensor],
    outputs: &[&forja_core::Tensor],
    epilogue: Option<&std::sync::Arc<forja_core::program::PreparedEpilogue>>,
    options: &Options,
) -> Result<(), Box<dyn Error>> {
    use forja_core::BackendDispatchData;
    let mut record = |inputs: &[&forja_core::Tensor],
                      outputs: &[&forja_core::Tensor]|
     -> Result<(), Box<dyn Error>> {
        let variant =
            backend.validate_variant_for_op(operation, inputs, outputs, &options.variant)?;
        if let Some(epilogue) = epilogue {
            commands.dispatch_epilogue(operation, inputs, outputs[0], epilogue, &[])?;
            if !commands.set_last_backend_data(BackendDispatchData::new(variant)) {
                return Err("missing diagnostic dispatch".into());
            }
        } else {
            commands.dispatch_many_with_backend_data(
                operation,
                inputs,
                outputs,
                BackendDispatchData::new(variant),
            )?;
        }
        Ok(())
    };
    if let Some(chunk) = options.row_chunk {
        for (inputs, outputs) in row_chunks::views(backend, operation, inputs, outputs, chunk)? {
            record(
                &inputs.iter().collect::<Vec<_>>(),
                &outputs.iter().collect::<Vec<_>>(),
            )?;
        }
    } else {
        record(inputs, outputs)?;
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn validate_diagnostic_options(shape: &Shape, options: &Options) -> Result<(), Box<dyn Error>> {
    validate_operand_visits(options.dispatches, options.copies)?;
    if options.row_chunk.is_some() && options.serial {
        return Err("--row-chunk cannot be combined with --serial".into());
    }
    if options.row_chunk.is_some()
        && (shape.epilogue.is_some()
            || shape.append.is_some()
            || options.state_steps.get() != 1
            || options.advance_position)
    {
        return Err("row chunks require an ordinary matrix diagnostic".into());
    }
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
    if shape.append.is_some()
        && (options.state_steps.get() > 1
            || shape.epilogue.is_some()
            || !matches!(shape.operation, forja_core::Op::PagedSdpa { causal: true, q_start, key_length, .. } if q_start.checked_add(1) == Some(key_length)))
    {
        return Err("append requires causal single-row paged attention through the new key".into());
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn record_append(
    backend: &impl Backend,
    commands: &mut forja_core::CommandList,
    operation: forja_core::Op,
    inputs: &[&forja_core::Tensor],
    rows: &[forja_core::Tensor],
) -> Result<(), Box<dyn Error>> {
    use forja_core::{Op, PagedCopyDirection, ViewOp};
    let Op::PagedSdpa {
        q_start,
        page_tokens,
        ..
    } = operation
    else {
        return Err("append requires paged attention".into());
    };
    let [query, key, value, table, generations] = inputs else {
        return Err("append requires two page planes and authentication metadata".into());
    };
    if query.layout().shape().get(1) != Some(&1)
        || rows
            .iter()
            .any(|row| row.layout().shape().first() != Some(&1))
    {
        return Err("append requires exactly one query and one new key/value row".into());
    }
    for (row, plane) in rows.iter().zip([key, value]) {
        let [pages, tokens, heads, width]: [u32; 4] = plane.layout().shape().try_into()?;
        let elements = heads.checked_mul(width).ok_or("page row size overflow")?;
        let plane = backend.view(plane, ViewOp::Reshape(vec![pages, tokens, elements]))?;
        commands.dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: q_start,
                page_tokens,
            },
            &[row, table, generations],
            &plane,
        )?;
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn consecutive_rows(
    dispatches: std::num::NonZeroU32,
    copies: std::num::NonZeroU32,
) -> Result<std::num::NonZeroU32, Box<dyn Error>> {
    if !dispatches.get().is_multiple_of(copies.get()) {
        return Err("successive positions require an equal row count per operand set".into());
    }
    std::num::NonZeroU32::new(dispatches.get() / copies.get())
        .ok_or_else(|| "successive positions require at least one row per operand set".into())
}

#[cfg(any(target_os = "macos", test))]
fn diagnostic_visit(index: u32, group: std::num::NonZeroU32, consecutive: bool) -> (u32, u32) {
    if consecutive {
        (index / group.get(), index % group.get())
    } else {
        (index % group.get(), 0)
    }
}

#[cfg(any(target_os = "macos", test))]
fn advance_attention(
    operation: forja_core::Op,
    offset: u32,
) -> Result<forja_core::Op, Box<dyn Error>> {
    use forja_core::Op;
    let add = |value: u32| {
        value
            .checked_add(offset)
            .ok_or("attention position overflow")
    };
    Ok(match operation {
        Op::PagedSdpaAppend {
            scale,
            q_start,
            page_tokens,
        } => Op::PagedSdpaAppend {
            scale,
            q_start: add(q_start)?,
            page_tokens,
        },
        Op::PagedSdpa {
            scale,
            causal,
            q_start,
            key_length,
            page_tokens,
        } => Op::PagedSdpa {
            scale,
            causal,
            q_start: add(q_start)?,
            key_length: add(key_length)?,
            page_tokens,
        },
        _ => return Err("position advancement requires paged attention".into()),
    })
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
    fn work_diagnostics_keep_raw_forecasts_and_buffer_intervals() {
        let profile = forja_core::SubmissionProfile {
            predicted_gpu: std::time::Duration::from_millis(20),
            buffer_admission: vec![forja_core::CommandBufferProfile {
                predicted_gpu: std::time::Duration::from_millis(20),
                reserve: std::time::Duration::from_millis(3),
                gpu_time: std::time::Duration::from_millis(7),
                backstop_ratio: 4,
            }],
            ..forja_core::SubmissionProfile::default()
        };
        let record = work_profile(&profile);
        assert_eq!(record["predicted_gpu_seconds"], 0.02);
        assert_eq!(record["buffers"][0]["gpu_time_seconds"], 0.007);
        assert_eq!(record["buffers"][0]["reserve_seconds"], 0.003);
    }

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
    fn row_chunks_refuse_incompatible_diagnostic_options() {
        type Edit = fn(&mut Shape, &mut Options);
        let ordinary = || {
            let shape = serde_json::from_value(serde_json::json!({
                "operation":{"kind":"matmul"},
                "inputs":[{"dtype":"f32","shape":[9,7]}, {"dtype":"f32","shape":[7,33]}],
                "outputs":[{"dtype":"f32","shape":[9,33]}]
            }))
            .unwrap();
            let options = Options {
                variant: "matmul.gemv".into(),
                state_steps: std::num::NonZeroU32::new(1).unwrap(),
                row_chunk: std::num::NonZeroU32::new(8),
                ..state_chain_options()
            };
            (shape, options)
        };
        let (shape, mut options) = ordinary();
        assert!(validate_diagnostic_options(&shape, &options).is_ok());
        options.row_chunk = None;
        options.serial = true;
        assert!(validate_diagnostic_options(&shape, &options).is_ok());
        let cases: [(&str, Edit); 5] = [
            ("epilogue", |s, _| {
                s.epilogue = Some(shape::DiagnosticEpilogue::Identity);
            }),
            ("append", |s, _| {
                s.append = Some(
                    serde_json::from_value(serde_json::json!([
                        {"dtype":"f32","shape":[1,7]}, {"dtype":"f32","shape":[1,7]}
                    ]))
                    .unwrap(),
                );
            }),
            ("state steps", |_, o| {
                o.state_steps = std::num::NonZeroU32::new(2).unwrap();
            }),
            ("positions", |_, o| o.advance_position = true),
            ("serial", |_, o| o.serial = true),
        ];
        for (name, edit) in cases {
            let (mut shape, mut options) = ordinary();
            edit(&mut shape, &mut options);
            let error = validate_diagnostic_options(&shape, &options).unwrap_err();
            let expected = if name == "serial" {
                "--row-chunk cannot be combined with --serial"
            } else {
                "row chunks require an ordinary matrix diagnostic"
            };
            assert_eq!(error.to_string(), expected, "{name}");
        }
    }

    #[test]
    fn state_dispatch_counts_refuse_overflow() {
        let mut options = state_chain_options();
        assert_eq!(effective_dispatches(&options).unwrap().get(), 3);
        options.dispatches = std::num::NonZeroU32::MAX;
        assert!(effective_dispatches(&options).is_err());
    }

    #[test]
    fn attention_rows_stay_consecutive_on_each_operand_set() {
        let count = |n| std::num::NonZeroU32::new(n).unwrap();
        let rows = consecutive_rows(count(6), count(2)).unwrap();
        assert_eq!(
            (0..6)
                .map(|i| diagnostic_visit(i, rows, true))
                .collect::<Vec<_>>(),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        );
        assert_eq!(
            (0..4)
                .map(|i| diagnostic_visit(i, count(2), false))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0), (0, 0), (1, 0)]
        );
        assert!(consecutive_rows(count(7), count(2)).is_err());
        assert!(consecutive_rows(count(1), count(2)).is_err());
    }

    #[test]
    fn attention_rows_advance_both_position_and_key_length() {
        use forja_core::Op;
        let attention = Op::PagedSdpa {
            scale: 0.5,
            causal: true,
            q_start: 4095,
            key_length: 4096,
            page_tokens: 32,
        };
        assert_eq!(
            advance_attention(attention, 2).unwrap(),
            Op::PagedSdpa {
                scale: 0.5,
                causal: true,
                q_start: 4097,
                key_length: 4098,
                page_tokens: 32
            }
        );
        let append = Op::PagedSdpaAppend {
            scale: 0.5,
            q_start: 4095,
            page_tokens: 32,
        };
        assert_eq!(
            advance_attention(append, 2).unwrap(),
            Op::PagedSdpaAppend {
                scale: 0.5,
                q_start: 4097,
                page_tokens: 32
            }
        );
        assert!(advance_attention(attention, u32::MAX).is_err());
        assert!(advance_attention(Op::Matmul, 0).is_err());
    }

    #[test]
    fn diagnostic_rates_require_uniform_dispatch_work() {
        use forja_core::{CommandList, DType, Op};
        let backend = forja_cpu::CpuBackend::new();
        let x = backend.alloc(DType::F32, &[1, 7]).unwrap();
        let w = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let y = backend.alloc(DType::F32, &[1, 33]).unwrap();
        let mut commands = CommandList::new();
        assert!(uniform_dispatch_cost(&commands).is_none());
        for _ in 0..2 {
            commands.dispatch(Op::Matmul, &[&x, &w], &y).unwrap();
        }
        assert!(uniform_dispatch_cost(&commands).is_some());
        let w = backend.alloc(DType::F32, &[7, 7]).unwrap();
        let y = backend.alloc(DType::F32, &[1, 7]).unwrap();
        commands.dispatch(Op::Matmul, &[&x, &w], &y).unwrap();
        assert!(uniform_dispatch_cost(&commands).is_none());
    }

    #[test]
    fn chunked_dispatch_costs_distinguish_unequal_tails() {
        use forja_core::{CommandList, DType, Op};
        let backend = forja_cpu::CpuBackend::new();
        for rows in [9, 16] {
            let input = backend.alloc(DType::F32, &[rows, 7]).unwrap();
            let weight = backend.alloc(DType::F32, &[7, 33]).unwrap();
            let output = backend.alloc(DType::F32, &[rows, 33]).unwrap();
            let chunks = row_chunks::views(
                &backend,
                Op::Matmul,
                &[&input, &weight],
                &[&output],
                std::num::NonZeroU32::new(8).unwrap(),
            )
            .unwrap();
            let mut commands = CommandList::new();
            for (inputs, outputs) in &chunks {
                commands
                    .dispatch(Op::Matmul, &inputs.iter().collect::<Vec<_>>(), &outputs[0])
                    .unwrap();
            }
            assert_eq!(commands.dispatches().len(), 2);
            assert_eq!(uniform_dispatch_cost(&commands).is_some(), rows == 16);
        }
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
        let appended = serde_json::json!({
            "operation":{"kind":"paged-sdpa","scale":0.1,"causal":true,"q_start":6,"key_length":7,"page_tokens":32},
            "inputs":[{"dtype":"f32","shape":[16,1,128]},
                {"dtype":"f16","shape":[1,32,8,128]}, {"dtype":"f16","shape":[1,32,8,128]},
                {"dtype":"u32","shape":[1,2],"u32_values":[0,1]}, {"dtype":"u32","shape":[1],"u32_values":[1]}],
            "outputs":[{"dtype":"f32","shape":[16,1,128]}],
            "append":[{"dtype":"f32","shape":[1,8,128],"seed":1}, {"dtype":"f32","shape":[1,8,128],"seed":2}]});
        assert_diagnostic_record(&root, "paged-sdpa.vector", &appended);
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
            advance_position: false,
            row_chunk: None,
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
    fn assert_appended_planes(
        backend: &forja_metal::MetalBackend,
        commands: &forja_core::CommandList,
        operation: forja_core::Op,
    ) {
        use forja_core::{CommandList, Op, Submission};
        let Op::PagedSdpa { q_start, .. } = operation else {
            panic!("expected paged attention");
        };
        let cpu = forja_cpu::CpuBackend::new();
        let attention = &commands.dispatches()[2];
        for (copy, plane) in commands.dispatches()[..2]
            .iter()
            .zip(&attention.inputs()[1..3])
        {
            let Op::PagedCopy { start, .. } = copy.op() else {
                panic!("expected page append");
            };
            assert_eq!(start, q_start);
            let row = &copy.inputs()[0];
            let source = cpu
                .alloc(row.layout().dtype(), row.layout().shape())
                .unwrap();
            cpu.write(&source, &backend.read(row).unwrap()).unwrap();
            let cast = cpu
                .alloc(plane.layout().dtype(), row.layout().shape())
                .unwrap();
            let mut cast_commands = CommandList::new();
            cast_commands.dispatch(Op::Copy, &[&source], &cast).unwrap();
            cpu.submit(cast_commands).unwrap().wait().unwrap();
            let row_bytes = cpu.read(&cast).unwrap();
            let actual = backend.read(plane).unwrap();
            let offset = usize::try_from(q_start).unwrap() * row_bytes.len();
            let end = offset + row_bytes.len();
            forja_testing::assert_outputs_agree(
                plane.layout().dtype(),
                &row_bytes,
                &actual[offset..end],
            )
            .unwrap();
            assert!(
                actual[..offset]
                    .iter()
                    .chain(&actual[end..])
                    .all(|&byte| byte == 0)
            );
        }
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
            advance_position: false,
            row_chunk: None,
            state_steps: std::num::NonZeroU32::new(1).unwrap(),
            copies: std::num::NonZeroU32::new(3).unwrap(),
            dispatches: std::num::NonZeroU32::new(7).unwrap(),
        };
        fs::write(&options.shape, serde_json::to_vec(shape).unwrap()).unwrap();
        let backend = forja_metal::MetalBackend::new().unwrap();
        let spec: Shape = serde_json::from_value(shape.clone()).unwrap();
        let commands = diagnostic_commands(&backend, &spec, &options).unwrap();
        if spec.append.is_some() {
            use forja_core::Submission;
            assert_eq!(commands.dispatches().len(), 21);
            backend.submit(commands.clone()).unwrap().wait().unwrap();
            assert_appended_planes(&backend, &commands, spec.operation);
        } else {
            assert_eq!(
                forja_core::required_barriers(&commands),
                [false, true, true, true, true, true, true]
            );
        }
        run(&options, &forja_config::DevConfig::default()).unwrap();
        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(&options.json).unwrap()).unwrap();
        assert_eq!(record["kind"], "kernel-diagnostic");
        assert_eq!(record["variant"], variant);
        assert_eq!(record["model_benchmark"], false);
        assert_eq!(record["serial"], true);
        assert_eq!(record["copies"], 3);
        assert_eq!(record["primary_dispatches_per_sample"], 7);
        assert_eq!(
            record["dispatches_per_sample"],
            if spec.append.is_some() { 21 } else { 7 }
        );
        options.dispatches = std::num::NonZeroU32::new(2).unwrap();
        assert!(run(&options, &forja_config::DevConfig::default()).is_err());
        assert_eq!(record["samples"].as_array().unwrap().len(), 1);
        if spec.append.is_some() {
            assert!(record["rates"].is_null());
        } else {
            assert_eq!(record["rates"].as_array().unwrap().len(), 1);
        }
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
        if matches!(spec.operation, forja_core::Op::PagedSdpa { .. }) {
            options.advance_position = true;
            options.dispatches = options.copies;
            run(&options, &forja_config::DevConfig::default()).unwrap();
            let advanced: serde_json::Value =
                serde_json::from_slice(&fs::read(&options.json).unwrap()).unwrap();
            assert_eq!(advanced["operand_order"], "consecutive");
            assert!(advanced["cost_per_dispatch"].is_null());
            assert!(advanced["rates"].is_null());
        }
    }
}
