//! Model-file validation that runs in the pre-push hook.

use std::{
    collections::HashMap,
    env,
    error::Error,
    fs,
    mem::size_of,
    path::{Path, PathBuf},
    process::Command,
};

use forja_core::{Backend, DType};
use forja_host::{Safetensors, WeightSource};
use serde_json::Value;

#[cfg(target_os = "macos")]
use forja_core::{CommandList, Op, Submission, Tensor};
#[cfg(target_os = "macos")]
use forja_testing::{QUANTIZED_TOLERANCE, assert_outputs_agree, normwise_relative_error};

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn qwen_weight_names_and_shapes_match_config() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let model = root.join("Qwen3-0.6B");
    let config = serde_json::from_slice::<Value>(&fs::read(model.join("config.json"))?)?;
    let layers = config_u32(&config, "num_hidden_layers")?;
    let hidden = config_u32(&config, "hidden_size")?;
    let heads = config_u32(&config, "num_attention_heads")?;
    let kv_heads = config_u32(&config, "num_key_value_heads")?;
    let head_dim = config_u32(&config, "head_dim")?;
    let intermediate = config_u32(&config, "intermediate_size")?;
    let vocab = config_u32(&config, "vocab_size")?;
    assert_eq!(
        (layers, hidden, heads, kv_heads, head_dim),
        (28, 1024, 16, 8, 128)
    );

    let mut expected = HashMap::new();
    expected.insert("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]);
    expected.insert("lm_head.weight".to_owned(), vec![vocab, hidden]);
    expected.insert("model.norm.weight".to_owned(), vec![hidden]);
    for layer in 0..layers {
        let prefix = format!("model.layers.{layer}");
        expected.insert(format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        expected.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        expected.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![heads * head_dim, hidden],
        );
        expected.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_heads * head_dim, hidden],
        );
        expected.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_heads * head_dim, hidden],
        );
        expected.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden, heads * head_dim],
        );
        expected.insert(format!("{prefix}.self_attn.q_norm.weight"), vec![head_dim]);
        expected.insert(format!("{prefix}.self_attn.k_norm.weight"), vec![head_dim]);
        expected.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        expected.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        expected.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
    }

    let source = Safetensors::open(model.join("model.safetensors"))?;
    let region = source.mapped_region()?;
    let buffer_len = u64::try_from(region.len())?;
    let backend = forja_cpu::CpuBackend::new();
    let buffer = backend.import_readonly(region)?;
    let mut opened = Vec::new();
    let actual = source
        .tensors()
        .iter()
        .map(|metadata| {
            if metadata.dtype() != DType::BF16 {
                return Err(format!("{} is not bf16", metadata.name()));
            }
            let layout = forja_core::Layout::contiguous(
                metadata.dtype(),
                metadata.byte_offset() / metadata.dtype().byte_size(),
                metadata.shape().to_vec(),
                buffer_len,
            )
            .map_err(|error| error.to_string())?;
            opened.push(
                backend
                    .tensor(buffer, layout)
                    .map_err(|error| error.to_string())?,
            );
            Ok((metadata.name().to_owned(), metadata.shape().to_vec()))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;
    assert_eq!(actual, expected);
    backend.release(opened.first().ok_or("model has no tensors")?)?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn metal_quantized_matmul_matches_cpu_for_real_qwen_weights() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let model = root.join("Qwen3-Coder-30B-A3B-Instruct-4bit");
    let index =
        serde_json::from_slice::<Value>(&fs::read(model.join("model.safetensors.index.json"))?)?;
    let names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.biases",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.scales",
        "model.layers.0.mlp.gate.biases",
    ];
    let shard_name = names
        .iter()
        .map(|name| weight_shard(&index, name))
        .collect::<Result<Vec<_>, _>>()?;
    if shard_name.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("quantized test tensors are not in one shard".into());
    }
    let source = Safetensors::open(model.join(shard_name[0]))?;
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new()?;
    let cpu_weights = import_named(&cpu, &source, &names)?;
    let metal_weights = import_named(&metal, &source, &names)?;
    let activation = (0_u16..2048)
        .map(|index| f32::from(index % 31) / 16.0 - 1.0)
        .collect::<Vec<_>>();
    let bytes = activation
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let cpu_input = cpu.alloc(DType::F32, &[1, 2048])?;
    cpu.write(&cpu_input, &bytes)?;
    let metal_input = metal.alloc(DType::F32, &[1, 2048])?;
    metal.write(&metal_input, &bytes)?;
    let mlx = mlx_quantized_outputs(&model)?;
    let mut mlx_offset = 0_usize;

    for (prefix, bits, output) in [
        ("model.layers.0.self_attn.q_proj", 4, 4096),
        ("model.layers.0.mlp.gate", 8, 128),
    ] {
        let expected = run_quantized(
            &cpu,
            &cpu_input,
            quantized_parts(&cpu_weights, prefix)?,
            bits,
            output,
        )?;
        let mlx_byte_len = usize::try_from(output)?
            .checked_mul(size_of::<f32>())
            .ok_or("MLX output byte length overflowed")?;
        let mlx_end = mlx_offset
            .checked_add(mlx_byte_len)
            .ok_or("MLX output byte range overflowed")?;
        let mlx_output = mlx
            .get(mlx_offset..mlx_end)
            .ok_or("MLX output was truncated")?;
        assert_quantized_outputs_agree(&expected, mlx_output)?;
        mlx_offset = mlx_end;
        let actual = run_quantized(
            &metal,
            &metal_input,
            quantized_parts(&metal_weights, prefix)?,
            bits,
            output,
        )?;
        assert_outputs_agree(DType::F32, &expected, &actual)?;
    }
    if mlx_offset != mlx.len() {
        return Err("MLX output contained trailing bytes".into());
    }
    cpu.release(&cpu_input)?;
    metal.release(&metal_input)?;
    for tensor in cpu_weights.values() {
        cpu.release(tensor)?;
    }
    for tensor in metal_weights.values() {
        metal.release(tensor)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn mlx_quantized_outputs(model: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("forja-host is not inside the workspace")?;
    let support = workspace.join("support/bench-mlx");
    let output = Command::new(support.join(".venv/bin/python"))
        .arg(support.join("real_quantized_matmul.py"))
        .arg(model)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "MLX ground-truth process failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output.stdout)
}

#[cfg(target_os = "macos")]
fn assert_quantized_outputs_agree(expected: &[u8], actual: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected = decode_f32(expected)?;
    let actual = decode_f32(actual)?;
    let error = normwise_relative_error(&expected, &actual);
    if error > QUANTIZED_TOLERANCE {
        return Err(format!(
            "quantized normwise relative error {error} exceeds {QUANTIZED_TOLERANCE}"
        )
        .into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    let values = chunks
        .iter()
        .map(|&chunk| f32::from_le_bytes(chunk))
        .collect::<Vec<_>>();
    if remainder.is_empty() {
        Ok(values)
    } else {
        Err("float output byte length is not divisible by four".into())
    }
}

#[cfg(target_os = "macos")]
fn weight_shard<'a>(index: &'a Value, name: &str) -> Result<&'a str, Box<dyn Error>> {
    index
        .get("weight_map")
        .and_then(|map| map.get(name))
        .and_then(Value::as_str)
        .ok_or_else(|| format!("weight index is missing {name}").into())
}

#[cfg(target_os = "macos")]
fn import_named<B: Backend>(
    backend: &B,
    source: &Safetensors,
    names: &[&str],
) -> Result<HashMap<String, Tensor>, Box<dyn Error>> {
    let region = source.mapped_region()?;
    let tensors = names
        .iter()
        .map(|&name| {
            let metadata = source
                .tensors()
                .iter()
                .find(|tensor| tensor.name() == name)
                .ok_or_else(|| format!("weight shard is missing {name}"))?;
            let start = usize::try_from(metadata.byte_offset())?;
            let end = start
                .checked_add(usize::try_from(metadata.byte_len())?)
                .ok_or("real weight byte range overflowed host address space")?;
            let bytes = region
                .bytes()
                .get(start..end)
                .ok_or("real weight byte range exceeds its shard")?;
            let tensor = backend.alloc(metadata.dtype(), metadata.shape())?;
            backend.write(&tensor, bytes)?;
            Ok((name.to_owned(), tensor))
        })
        .collect::<Result<HashMap<_, _>, Box<dyn Error>>>()?;
    Ok(tensors)
}

#[cfg(target_os = "macos")]
fn quantized_parts<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
) -> Result<[&'a Tensor; 3], Box<dyn Error>> {
    let get = |suffix| {
        weights
            .get(&format!("{prefix}.{suffix}"))
            .ok_or_else(|| format!("imported weights are missing {prefix}.{suffix}"))
    };
    Ok([get("weight")?, get("scales")?, get("biases")?])
}

#[cfg(target_os = "macos")]
fn run_quantized<B: Backend>(
    backend: &B,
    input: &Tensor,
    [packed, scales, biases]: [&Tensor; 3],
    bits: u8,
    output_width: u32,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = backend.alloc(DType::F32, &[1, output_width])?;
    let mut commands = CommandList::new();
    commands.dispatch(
        Op::QuantMatmul {
            bits,
            group_size: 64,
        },
        &[input, packed, scales, biases],
        &output,
    )?;
    backend.submit(commands)?.wait()?;
    let values = backend.read(&output)?;
    backend.release(&output)?;
    Ok(values)
}

fn config_u32(config: &Value, name: &str) -> Result<u32, Box<dyn Error>> {
    let value = config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("config field {name} is missing"))?;
    Ok(u32::try_from(value)?)
}
