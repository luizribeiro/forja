//! Model-file validation that runs in the pre-push hook.

use std::{collections::HashMap, env, error::Error, fs, path::PathBuf};

use forja_core::{Backend, DType};
use forja_host::{Safetensors, WeightSource};
use serde_json::Value;

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

fn config_u32(config: &Value, name: &str) -> Result<u32, Box<dyn Error>> {
    let value = config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("config field {name} is missing"))?;
    Ok(u32::try_from(value)?)
}
