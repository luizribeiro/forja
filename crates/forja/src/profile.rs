use std::{collections::BTreeMap, error::Error, fs, time::Duration};

use forja_core::{Backend, CommandList, DType, DispatchProfile, Op, ProfileTensor, Submission};

use crate::{
    args::{Profile, ProfileMode},
    benchmark::{TokenProfile, measure_token_profile},
    machine_load,
};

const COPY_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Default)]
struct Total {
    time: Duration,
    idle: Duration,
    dispatches: u64,
    barriers: u64,
}

pub(crate) async fn run(options: &Profile) -> Result<(), Box<dyn Error>> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = options;
        return Err("engine profiling requires macOS and Metal".into());
    }
    #[cfg(target_os = "macos")]
    {
        let load_before = machine_load::capture("BEFORE PROFILE")?;
        let peak = measure_copy_peak()?;
        let measurement = measure_token_profile(options).await?;
        let load_after = machine_load::capture("AFTER PROFILE")?;
        let report = report(options, &measurement, peak, &load_before, &load_after)?;
        if options.json {
            fs::create_dir_all(&options.scratch)?;
            let path = options.scratch.join("profile.json");
            let mut bytes = serde_json::to_vec_pretty(&report)?;
            bytes.push(b'\n');
            fs::write(&path, bytes)?;
            println!("profile JSON: {}", path.display());
        } else {
            print_markdown(options, &measurement, peak)?;
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::cast_precision_loss)]
fn measure_copy_peak() -> Result<f64, Box<dyn Error>> {
    let backend = forja_metal::MetalBackend::new()?;
    let elements = u32::try_from(COPY_BYTES / DType::U32.byte_size())?;
    let input = backend.alloc(DType::U32, &[elements])?;
    let output = backend.alloc(DType::U32, &[elements])?;
    let commands = || {
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output)?;
        Ok::<_, forja_core::OpError>(commands)
    };
    let warmup = backend.submit(commands()?)?;
    warmup.wait()?;
    let submission = backend.submit(commands()?)?;
    submission.wait()?;
    let elapsed = submission
        .gpu_time()
        .ok_or("copy GPU timing is unavailable")?
        .as_secs_f64();
    if elapsed == 0.0 {
        return Err("copy profile reported zero device time".into());
    }
    Ok(COPY_BYTES as f64 * 2.0 / elapsed / 1e9)
}

fn dispatch_bytes(dispatch: &DispatchProfile) -> Result<u64, &'static str> {
    if dispatch.op == Op::Embed {
        let [embeddings, ids] = dispatch.inputs.as_slice() else {
            return Err("embed profile has invalid inputs");
        };
        let width = embeddings
            .shape
            .last()
            .copied()
            .ok_or("embed profile has no width")?;
        let rows = element_count(ids)?;
        return rows
            .checked_mul(u64::from(width))
            .and_then(|count| count.checked_mul(embeddings.dtype.byte_size()))
            .and_then(|bytes| bytes.checked_add(tensor_bytes(ids).ok()?))
            .and_then(|bytes| {
                dispatch.outputs.iter().try_fold(bytes, |sum, tensor| {
                    sum.checked_add(tensor_bytes(tensor).ok()?)
                })
            })
            .ok_or("dispatch byte count overflowed");
    }
    dispatch
        .inputs
        .iter()
        .chain(&dispatch.outputs)
        .try_fold(0_u64, |sum, tensor| {
            sum.checked_add(tensor_bytes(tensor)?)
                .ok_or("dispatch byte count overflowed")
        })
}

fn tensor_bytes(tensor: &ProfileTensor) -> Result<u64, &'static str> {
    element_count(tensor)?
        .checked_mul(tensor.dtype.byte_size())
        .ok_or("tensor byte count overflowed")
}

fn element_count(tensor: &ProfileTensor) -> Result<u64, &'static str> {
    tensor.shape.iter().try_fold(1_u64, |count, &extent| {
        count
            .checked_mul(u64::from(extent))
            .ok_or("tensor element count overflowed")
    })
}

fn aggregate<'a>(
    dispatches: &'a [DispatchProfile],
    key: impl Fn(&'a DispatchProfile) -> &'a str,
) -> BTreeMap<&'a str, Total> {
    let mut totals = BTreeMap::new();
    for dispatch in dispatches {
        let total = totals.entry(key(dispatch)).or_insert(Total::default());
        total.time = total.time.saturating_add(dispatch.gpu_time);
        total.idle = total.idle.saturating_add(dispatch.gap_before);
        total.dispatches = total.dispatches.saturating_add(1);
        total.barriers = total.barriers.saturating_add(u64::from(dispatch.barrier));
    }
    totals
}

fn op_name(operation: Op) -> &'static str {
    match operation {
        Op::Program(_) => "program",
        Op::Copy => "copy",
        Op::Add => "add",
        Op::SiluMul => "silu_mul",
        Op::RmsNorm { .. } => "rms_norm",
        Op::Softmax => "softmax",
        Op::Argmax => "argmax",
        Op::TopK { .. } => "top-k",
        Op::Sample { .. } => "sample",
        Op::Rope { .. } => "rope",
        Op::Embed => "embed",
        Op::QuantEmbed { .. } => "quant-embed",
        Op::Matmul => "matmul",
        Op::GatherMatmul => "gather-matmul",
        Op::QuantMatmul { .. } => "quant-matmul",
        Op::GatherQuantMatmul { .. } => "gather-quant-matmul",
        Op::GatherQuantMatmulCombine { .. } => "gather-quant-matmul-combine",
        Op::GatherQuantSiluMul { .. } => "gather-quant-silu-mul",
        Op::Sdpa { .. } => "sdpa",
    }
}

fn shape(dispatch: &DispatchProfile) -> String {
    let tensors = |values: &[ProfileTensor]| {
        values
            .iter()
            .map(|tensor| {
                let extents = tensor
                    .shape
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                format!("{}[{extents}]", dtype_name(tensor.dtype))
            })
            .collect::<Vec<_>>()
            .join(" + ")
    };
    format!(
        "{} -> {}",
        tensors(&dispatch.inputs),
        tensors(&dispatch.outputs)
    )
}

const fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
    }
}

#[allow(clippy::cast_precision_loss)]
fn throughput(bytes: u64, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        0.0
    } else {
        bytes as f64 / seconds / 1e9
    }
}

fn print_markdown(
    options: &Profile,
    measurement: &TokenProfile,
    peak: f64,
) -> Result<(), Box<dyn Error>> {
    let submission = measurement
        .step
        .submission
        .as_ref()
        .ok_or("profiled step has no submission detail")?;
    let dispatch_time = submission
        .per_dispatch
        .iter()
        .fold(Duration::ZERO, |sum, dispatch| sum + dispatch.gpu_time);
    let idle = submission
        .per_dispatch
        .iter()
        .fold(Duration::ZERO, |sum, dispatch| sum + dispatch.gap_before);
    println!("# forja profile\n");
    println!("- Device: {}", measurement.device);
    println!("- Mode: {}", mode_name(options.mode));
    println!("- Tokens: {}", options.context);
    println!("- Measured copy peak: {peak:.1} GB/s");
    println!(
        "- Unprofiled step: {:.3} ms wall, {:.3} ms GPU",
        milliseconds(measurement.baseline_wall),
        milliseconds(measurement.baseline_gpu)
    );
    println!(
        "- Profiled step: {:.3} ms wall, {:.3} ms GPU ({:.2}x timestamp perturbation)",
        milliseconds(measurement.step.wall_time),
        milliseconds(submission.gpu_time),
        submission.gpu_time.as_secs_f64() / measurement.baseline_gpu.as_secs_f64()
    );
    println!(
        "- Accounted: {:.3} ms dispatch, {:.3} ms idle/barrier, {} dispatches, {} barriers\n",
        milliseconds(dispatch_time),
        milliseconds(idle),
        submission.dispatches,
        submission.barriers
    );
    println!("## Dispatch timeline\n");
    println!(
        "| # | op | kernel | shape | start ms | end ms | gap us | bytes | GB/s | peak | barrier |"
    );
    println!("|---:|---|---|---|---:|---:|---:|---:|---:|---:|:---:|");
    for (index, dispatch) in submission.per_dispatch.iter().enumerate() {
        let bytes = dispatch_bytes(dispatch)?;
        let rate = throughput(bytes, dispatch.gpu_time);
        println!(
            "| {index} | {} | {} | {} | {:.3} | {:.3} | {:.1} | {bytes} | {rate:.1} | {:.1}% | {} |",
            op_name(dispatch.op),
            dispatch.kernel,
            shape(dispatch),
            milliseconds(dispatch.gpu_start),
            milliseconds(dispatch.gpu_end),
            dispatch.gap_before.as_secs_f64() * 1e6,
            rate / peak * 100.0,
            if dispatch.barrier { "yes" } else { "no" },
        );
    }
    print_totals(
        "Operation totals",
        aggregate(&submission.per_dispatch, |item| op_name(item.op)),
        submission.gpu_time,
    );
    print_totals(
        "Kernel totals",
        aggregate(&submission.per_dispatch, |item| item.kernel),
        submission.gpu_time,
    );
    Ok(())
}

fn print_totals(title: &str, totals: BTreeMap<&str, Total>, gpu_time: Duration) {
    println!("\n## {title}\n");
    println!("| name | time ms | share | dispatches | barriers | idle gaps ms |");
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, total) in totals {
        println!(
            "| {name} | {:.3} | {:.1}% | {} | {} | {:.3} |",
            milliseconds(total.time),
            total.time.as_secs_f64() / gpu_time.as_secs_f64() * 100.0,
            total.dispatches,
            total.barriers,
            milliseconds(total.idle),
        );
    }
}

fn report(
    options: &Profile,
    measurement: &TokenProfile,
    peak: f64,
    load_before: &machine_load::Snapshot,
    load_after: &machine_load::Snapshot,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let submission = measurement
        .step
        .submission
        .as_ref()
        .ok_or("profiled step has no submission detail")?;
    let timeline = submission
        .per_dispatch
        .iter()
        .enumerate()
        .map(|(index, dispatch)| {
            let bytes = dispatch_bytes(dispatch)?;
            let rate = throughput(bytes, dispatch.gpu_time);
            Ok(serde_json::json!({
                "index": index,
                "op": op_name(dispatch.op),
                "kernel": dispatch.kernel,
                "shape": shape(dispatch),
                "start_seconds": dispatch.gpu_start.as_secs_f64(),
                "end_seconds": dispatch.gpu_end.as_secs_f64(),
                "gap_before_seconds": dispatch.gap_before.as_secs_f64(),
                "gpu_time_seconds": dispatch.gpu_time.as_secs_f64(),
                "bytes": bytes,
                "gigabytes_per_second": rate,
                "percent_of_peak": rate / peak * 100.0,
                "barrier_before": dispatch.barrier,
            }))
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    Ok(serde_json::json!({
        "provenance": {
            "commit": crate::provenance::commit(),
            "dirty": crate::provenance::dirty(),
            "machine_load": {
                "before": load_before,
                "after": load_after,
            },
        },
        "device": measurement.device,
        "mode": mode_name(options.mode),
        "tokens": options.context,
        "measured_copy_peak_gigabytes_per_second": peak,
        "unprofiled": {
            "wall_time_seconds": measurement.baseline_wall.as_secs_f64(),
            "gpu_time_seconds": measurement.baseline_gpu.as_secs_f64(),
        },
        "profiled": {
            "wall_time_seconds": measurement.step.wall_time.as_secs_f64(),
            "gpu_time_seconds": submission.gpu_time.as_secs_f64(),
            "dispatches": submission.dispatches,
            "barriers": submission.barriers,
        },
        "timeline": timeline,
        "by_op": totals_json(aggregate(&submission.per_dispatch, |item| op_name(item.op)), submission.gpu_time),
        "by_kernel": totals_json(aggregate(&submission.per_dispatch, |item| item.kernel), submission.gpu_time),
    }))
}

fn totals_json(totals: BTreeMap<&str, Total>, gpu_time: Duration) -> serde_json::Value {
    serde_json::Value::Array(
        totals
            .into_iter()
            .map(|(name, total)| {
                serde_json::json!({
                    "name": name,
                    "gpu_time_seconds": total.time.as_secs_f64(),
                    "percent_of_gpu_time": total.time.as_secs_f64() / gpu_time.as_secs_f64() * 100.0,
                    "dispatches": total.dispatches,
                    "barriers": total.barriers,
                    "idle_gap_seconds": total.idle.as_secs_f64(),
                })
            })
            .collect(),
    )
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

const fn mode_name(mode: ProfileMode) -> &'static str {
    match mode {
        ProfileMode::Decode => "decode",
        ProfileMode::Prefill => "prefill",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(dtype: DType, shape: &[u32]) -> ProfileTensor {
        ProfileTensor {
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn dispatch(
        op: Op,
        inputs: Vec<ProfileTensor>,
        outputs: Vec<ProfileTensor>,
    ) -> DispatchProfile {
        DispatchProfile {
            op,
            kernel: "test",
            inputs,
            outputs,
            barrier: true,
            gpu_start: Duration::ZERO,
            gpu_end: Duration::from_micros(7),
            gap_before: Duration::from_micros(2),
            gpu_time: Duration::from_micros(7),
        }
    }

    #[test]
    fn counts_logical_matmul_bytes() {
        let profile = dispatch(
            Op::Matmul,
            vec![
                tensor(DType::BF16, &[1, 1024]),
                tensor(DType::BF16, &[1024, 3072]),
            ],
            vec![tensor(DType::BF16, &[1, 3072])],
        );
        assert_eq!(
            dispatch_bytes(&profile),
            Ok((1024 + 1024 * 3072 + 3072) * 2)
        );
    }

    #[test]
    fn embedding_counts_selected_rows_not_the_table() {
        let profile = dispatch(
            Op::Embed,
            vec![
                tensor(DType::BF16, &[151_936, 1024]),
                tensor(DType::U32, &[7]),
            ],
            vec![tensor(DType::BF16, &[7, 1024])],
        );
        assert_eq!(dispatch_bytes(&profile), Ok(7 * 1024 * 2 * 2 + 7 * 4));
    }

    #[test]
    fn aggregates_time_barriers_and_idle_gaps() {
        let profiles = vec![
            dispatch(
                Op::Copy,
                vec![tensor(DType::F32, &[7])],
                vec![tensor(DType::F32, &[7])],
            ),
            dispatch(
                Op::Copy,
                vec![tensor(DType::F32, &[33])],
                vec![tensor(DType::F32, &[33])],
            ),
        ];
        let totals = aggregate(&profiles, |item| op_name(item.op));
        let copy = totals["copy"];
        assert_eq!(copy.dispatches, 2);
        assert_eq!(copy.barriers, 2);
        assert_eq!(copy.time, Duration::from_micros(14));
        assert_eq!(copy.idle, Duration::from_micros(4));
    }
}
