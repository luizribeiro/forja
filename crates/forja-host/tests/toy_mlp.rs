//! End-to-end checks for the SDK toy engine.

use std::path::PathBuf;

use forja_core::Backend;
use forja_host::{EngineRunner, EngineStep, Limits};
use forja_sdk::{Engine, NativeDevice, StepInput, Tensor, Weights, set_native_device};
use forja_testing::{DeterministicValues, F32_TOLERANCE, normwise_relative_error};
use golden_fixtures::decode_f32_le;
use toy_mlp::{HIDDEN, INTERMEDIATE, LAYERS, MAX_CONTEXT, ToyEngine, VOCAB};

const LIMITS: Limits = Limits::new(16 * 1024 * 1024, 8, 1024 * 1024, 512, 1024 * 1024)
    .with_command_limits(64, 64 * 1024 * 1024);
const TOKENS: [u32; 3] = [1, 7, 32];

#[tokio::test(flavor = "multi_thread")]
async fn cpu_toy_mlp_matches_native_sdk() -> wasmtime::Result<()> {
    compare(forja_cpu::CpuBackend::new(), NativeDevice::Cpu, "cpu").await
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_toy_mlp_matches_native_sdk() -> wasmtime::Result<()> {
    compare(
        forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?,
        NativeDevice::Metal,
        "metal",
    )
    .await
}

async fn compare<B>(backend: B, device: NativeDevice, label: &str) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    let path = weight_file(label)?;
    let expected = native_outputs(&path, device)?;
    let mut runner = EngineRunner::new(test_guests::toy_mlp(), backend, LIMITS, &path).await?;
    let info = runner.describe().await?;
    assert_eq!(info.vocab, VOCAB);
    assert_eq!(info.max_context, MAX_CONTEXT);
    assert_eq!(info.tap_layers, [0, 1]);
    runner.load().await??;
    let output = runner
        .step(EngineStep {
            tokens: TOKENS.to_vec(),
            start_pos: 0,
            taps: true,
        })
        .await??;

    let logits = decode(&runner.read(&output.logits).await?)?;
    assert_agrees(&expected.0, &logits);
    assert_eq!(output.taps.len(), expected.1.len());
    for (tensor, expected) in output.taps.iter().zip(&expected.1) {
        assert_agrees(expected, &decode(&runner.read(tensor).await?)?);
    }
    Ok(())
}

fn native_outputs(
    path: &std::path::Path,
    device: NativeDevice,
) -> wasmtime::Result<(Vec<f32>, Vec<Vec<f32>>)> {
    set_native_device(device);
    let weights = Weights::open(path).map_err(wasmtime::Error::msg)?;
    let mut engine = ToyEngine::load(&weights).map_err(wasmtime::Error::msg)?;
    let output = engine
        .step(StepInput {
            tokens: Tensor::from_slice(&TOKENS, &[3]).map_err(wasmtime::Error::msg)?,
            start_pos: 0,
            taps: true,
        })
        .map_err(wasmtime::Error::msg)?;
    let logits = output.logits.to_vec().map_err(wasmtime::Error::msg)?;
    let taps = output
        .taps
        .iter()
        .map(|tensor| tensor.to_vec().map_err(wasmtime::Error::msg))
        .collect::<wasmtime::Result<Vec<_>>>()?;
    Ok((logits, taps))
}

fn assert_agrees(expected: &[f32], actual: &[f32]) {
    let error = normwise_relative_error(expected, actual);
    assert!(
        error <= F32_TOLERANCE,
        "toy engine relative error {error} exceeded {F32_TOLERANCE}: expected {expected:?}, actual {actual:?}"
    );
}

fn decode(bytes: &[u8]) -> wasmtime::Result<Vec<f32>> {
    decode_f32_le(bytes).map_err(wasmtime::Error::msg)
}

fn weight_file(label: &str) -> wasmtime::Result<PathBuf> {
    let mut values = DeterministicValues::new(0x243f_6a88_85a3_08d3);
    let mut tensors = vec![tensor("embed.weight", &[VOCAB, HIDDEN], &mut values, 0.1)?];
    for layer in 0..LAYERS {
        tensors.push(norm_tensor(
            &format!("blocks.{layer}.norm.weight"),
            &mut values,
        ));
        tensors.push(tensor(
            &format!("blocks.{layer}.up.weight"),
            &[INTERMEDIATE * 2, HIDDEN],
            &mut values,
            0.1,
        )?);
        tensors.push(tensor(
            &format!("blocks.{layer}.down.weight"),
            &[HIDDEN, INTERMEDIATE],
            &mut values,
            0.1,
        )?);
    }
    tensors.push(norm_tensor("norm.weight", &mut values));
    tensors.push(tensor(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &mut values,
        0.1,
    )?);

    let bytes = safetensors(&tensors)?;
    let path = std::env::temp_dir().join(format!("forja-toy-mlp-{}-{label}", std::process::id()));
    std::fs::write(&path, bytes).map_err(wasmtime::Error::msg)?;
    Ok(path)
}

struct WeightTensor {
    name: String,
    shape: Vec<u32>,
    values: Vec<f32>,
}

fn tensor(
    name: &str,
    shape: &[u32],
    values: &mut DeterministicValues,
    scale: f32,
) -> wasmtime::Result<WeightTensor> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| wasmtime::Error::msg("toy weight shape overflowed"))?;
    Ok(WeightTensor {
        name: name.to_owned(),
        shape: shape.to_vec(),
        values: (0..count).map(|_| values.next_f32() * scale).collect(),
    })
}

fn norm_tensor(name: &str, values: &mut DeterministicValues) -> WeightTensor {
    WeightTensor {
        name: name.to_owned(),
        shape: vec![HIDDEN],
        values: (0..HIDDEN)
            .map(|_| values.next_f32().mul_add(0.1, 1.0))
            .collect(),
    }
}

fn safetensors(tensors: &[WeightTensor]) -> wasmtime::Result<Vec<u8>> {
    let mut offset = 0_u64;
    let entries = tensors
        .iter()
        .map(|tensor| {
            let start = offset;
            let bytes = u64::try_from(tensor.values.len())
                .ok()
                .and_then(|count| count.checked_mul(4))
                .ok_or_else(|| wasmtime::Error::msg("toy weight byte size overflowed"))?;
            offset = offset
                .checked_add(bytes)
                .ok_or_else(|| wasmtime::Error::msg("toy weight offset overflowed"))?;
            Ok(format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{start},{offset}]}}",
                tensor.name, tensor.shape
            ))
        })
        .collect::<wasmtime::Result<Vec<_>>>()?
        .join(",");
    let mut header = format!("{{{entries}}}").into_bytes();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(
        tensors
            .iter()
            .flat_map(|tensor| tensor.values.iter().flat_map(|value| value.to_le_bytes())),
    );
    Ok(bytes)
}
