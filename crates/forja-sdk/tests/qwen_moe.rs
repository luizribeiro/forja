#![cfg(all(feature = "native-metal", target_os = "macos"))]

//! Real-weight validation of the SDK sparse `MoE` composition.

use std::{env, error::Error, path::Path, process::Command, time::Instant};

use forja_sdk::{NativeDevice, Tensor, Weights, bf16, eval, nn, set_native_device};
use forja_testing::{QUANTIZED_TOLERANCE, normwise_relative_error};

const EXPERTS: u32 = 128;
const HIDDEN: u32 = 2048;
const INTERMEDIATE: u32 = 768;
const TOP_K: u32 = 8;
const WARMUPS: usize = 5;
const REPETITIONS: usize = 30;
const BLOCKS_PER_SAMPLE: u32 = 20;

struct QuantizedProjection {
    packed: Tensor<u32>,
    scales: Tensor<bf16>,
    biases: Tensor<bf16>,
}

impl QuantizedProjection {
    fn load(weights: &Weights<'_>, output: u32, input: u32) -> forja_sdk::Result<Self> {
        Ok(Self {
            packed: weights.tensor("weight", &[EXPERTS, output, input / 8])?,
            scales: weights.tensor("scales", &[EXPERTS, output, input / 64])?,
            biases: weights.tensor("biases", &[EXPERTS, output, input / 64])?,
        })
    }

    fn apply(
        &self,
        input: &Tensor<bf16>,
        indices: &Tensor<u32>,
    ) -> forja_sdk::Result<Tensor<bf16>> {
        input.gather_quant_matmul(&self.packed, &self.scales, &self.biases, indices, 4, 64)
    }
}

struct QwenMoe {
    router_packed: Tensor<u32>,
    router_scales: Tensor<bf16>,
    router_biases: Tensor<bf16>,
    gate: QuantizedProjection,
    up: QuantizedProjection,
    down: QuantizedProjection,
}

impl QwenMoe {
    fn load(shard: &Path) -> forja_sdk::Result<Self> {
        let weights = Weights::open(shard)?;
        let router = weights.scoped("model.layers.0.mlp.gate");
        let experts = weights.scoped("model.layers.0.mlp.switch_mlp");
        Ok(Self {
            router_packed: router.tensor("weight", &[EXPERTS, HIDDEN / 4])?,
            router_scales: router.tensor("scales", &[EXPERTS, HIDDEN / 64])?,
            router_biases: router.tensor("biases", &[EXPERTS, HIDDEN / 64])?,
            gate: QuantizedProjection::load(&experts.scoped("gate_proj"), INTERMEDIATE, HIDDEN)?,
            up: QuantizedProjection::load(&experts.scoped("up_proj"), INTERMEDIATE, HIDDEN)?,
            down: QuantizedProjection::load(&experts.scoped("down_proj"), HIDDEN, INTERMEDIATE)?,
        })
    }

    fn forward(&self, input: &Tensor<bf16>) -> forja_sdk::Result<(Tensor<bf16>, Tensor<u32>)> {
        let logits = input.quant_matmul(
            &self.router_packed,
            &self.router_scales,
            &self.router_biases,
            8,
            64,
        )?;
        let (route_weights, indices) = nn::moe_router(&logits, TOP_K, true)?;
        let hidden = input
            .gather_quant_silu_mul(
                &self.gate.packed,
                &self.gate.scales,
                &self.gate.biases,
                &self.up.packed,
                &self.up.scales,
                &self.up.biases,
                &indices,
                4,
                64,
            )?
            .reshape(&[TOP_K, INTERMEDIATE])?;
        let down_indices = indices.reshape(&[TOP_K, 1])?;
        let output = self
            .down
            .apply(&hidden, &down_indices)?
            .reshape(&[1, TOP_K, HIDDEN])?;
        Ok((nn::moe_combine(&output, &route_weights)?, indices))
    }
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn qwen_layer_zero_moe_matches_mlx() -> Result<(), Box<dyn Error>> {
    set_native_device(NativeDevice::Metal);
    let model = model_path()?;
    let moe = QwenMoe::load(&model.join("model-00001-of-00004.safetensors"))?;
    let input = fixed_input()?;
    let (output, indices) = moe.forward(&input)?;
    let mut actual_indices = indices.to_vec()?;
    let actual = output
        .to_vec()?
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    let reference = mlx(&model, "reference")?;
    let (mut expected_indices, expected) = decode_reference(&reference)?;
    actual_indices.sort_unstable();
    expected_indices.sort_unstable();
    assert_eq!(actual_indices, expected_indices);
    let error = normwise_relative_error(&expected, &actual);
    if error > QUANTIZED_TOLERANCE {
        return Err(format!(
            "quantized normwise relative error {error} exceeds {QUANTIZED_TOLERANCE}"
        )
        .into());
    }
    Ok(())
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn qwen_layer_zero_moe_synchronized_timing() -> Result<(), Box<dyn Error>> {
    set_native_device(NativeDevice::Metal);
    let model = model_path()?;
    let moe = QwenMoe::load(&model.join("model-00001-of-00004.safetensors"))?;
    let input = fixed_input()?;
    let forja_us = benchmark_forja(&moe, &input)?;
    let mlx_us = String::from_utf8(mlx(&model, "benchmark")?)?
        .trim()
        .parse::<f64>()?;
    eprintln!("Qwen layer-zero MoE synchronized: Forja {forja_us:.3} us, MLX {mlx_us:.3} us");
    Ok(())
}

fn model_path() -> Result<std::path::PathBuf, Box<dyn Error>> {
    Ok(Path::new(&env::var("FORJA_MODELS")?).join("Qwen3-Coder-30B-A3B-Instruct-4bit"))
}

fn fixed_input() -> Result<Tensor<bf16>, Box<dyn Error>> {
    let values = (0_u16..u16::try_from(HIDDEN)?)
        .map(|index| bf16::from_f32(f32::from(index % 31) / 16.0 - 1.0))
        .collect::<Vec<_>>();
    Ok(Tensor::from_slice(&values, &[1, HIDDEN])?)
}

fn benchmark_forja(moe: &QwenMoe, input: &Tensor<bf16>) -> Result<f64, Box<dyn Error>> {
    for _ in 0..WARMUPS {
        run_forja(moe, input)?;
    }
    let mut samples = (0..REPETITIONS)
        .map(|_| run_forja(moe, input))
        .collect::<Result<Vec<_>, _>>()?;
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn run_forja(moe: &QwenMoe, input: &Tensor<bf16>) -> Result<f64, Box<dyn Error>> {
    let outputs = (0..BLOCKS_PER_SAMPLE)
        .map(|_| moe.forward(input).map(|pair| pair.0))
        .collect::<forja_sdk::Result<Vec<_>>>()?;
    let started = Instant::now();
    eval()?;
    let elapsed = started.elapsed().as_secs_f64() / f64::from(BLOCKS_PER_SAMPLE);
    drop(outputs);
    Ok(elapsed * 1_000_000.0)
}

fn mlx(model: &Path, operation: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("forja-sdk is not inside the workspace")?;
    let support = workspace.join("support/bench-mlx");
    let output = Command::new(support.join(".venv/bin/python"))
        .arg(support.join("moe_block.py"))
        .arg(operation)
        .arg(model)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "MLX MoE process failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output.stdout)
}

fn decode_reference(bytes: &[u8]) -> Result<(Vec<u32>, Vec<f32>), Box<dyn Error>> {
    let split = usize::try_from(TOP_K)? * size_of::<u32>();
    let (index_bytes, value_bytes) = bytes.split_at_checked(split).ok_or("truncated reference")?;
    let indices = index_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&bytes| u32::from_le_bytes(bytes))
        .collect();
    let (chunks, remainder) = value_bytes.as_chunks::<4>();
    if !remainder.is_empty() || chunks.len() != usize::try_from(HIDDEN)? {
        return Err("invalid reference output length".into());
    }
    let values = chunks
        .iter()
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect();
    Ok((indices, values))
}
