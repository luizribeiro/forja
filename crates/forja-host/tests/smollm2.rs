//! Random-weight Llama exercises the same projection and cache geometry as the full model.

use forja_sdk::{Engine, EngineLoadConfig, Tensor, Weights};
use forja_testing::DeterministicValues;
use smollm2::{Config, SmolLM2};
use std::{error::Error, fs, path::PathBuf};

#[test]
fn metal_tiny_reference_accepts_chunked_prompts() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let path = tiny_weights()?;
    let weights = Weights::open(&path)?;
    let load = EngineLoadConfig {
        max_context: Some(64),
        ..EngineLoadConfig::default()
    };
    let mut whole = SmolLM2::with_config(&weights, &load, Config::tiny())?;
    let expected = step(&mut whole, &[1, 7, 32], 0)?;
    let mut chunks = SmolLM2::with_config(&weights, &load, Config::tiny())?;
    step(&mut chunks, &[1], 0)?;
    let actual = step(&mut chunks, &[7, 32], 1)?;
    forja_testing::assert_f32_values_agree(&expected, &actual)?;
    assert!(step(&mut chunks, &[1], 64).is_err());
    assert!(step(&mut chunks, &[], 0).is_err());
    fs::remove_file(path)?;
    Ok(())
}

fn step(engine: &mut SmolLM2, tokens: &[u32], start_pos: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let count = u32::try_from(tokens.len())?;
    Ok(engine
        .step(forja_sdk::StepInput {
            phase: forja_sdk::BatchPhase::Prompt,
            tokens: Tensor::from_slice(tokens, &[count])?,
            start_pos,
            taps: false,
        })?
        .logits
        .to_vec()?)
}

fn tiny_weights() -> Result<PathBuf, Box<dyn Error>> {
    let mut random = DeterministicValues::new(0x243f_6a88_85a3_08d3);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut tensor = |name: String, shape: &[u32], norm: bool| {
        let start = data.len();
        for _ in 0..shape.iter().product::<u32>() {
            let value = random
                .next_f32()
                .mul_add(0.05, if norm { 1.0 } else { 0.0 });
            let bf16 = forja_sdk::bf16::from_f32(value);
            data.extend(bf16.to_bits().to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({"dtype":"BF16", "shape":shape,
            "data_offsets":[start,data.len()]}),
        );
    };
    tensor("model.embed_tokens.weight".to_owned(), &[33, 24], false);
    tensor("model.norm.weight".to_owned(), &[24], true);
    for layer in 0..2 {
        for name in ["input_layernorm", "post_attention_layernorm"] {
            tensor(format!("model.layers.{layer}.{name}.weight"), &[24], true);
        }
        for (name, shape) in [
            ("self_attn.q_proj", [24, 24]),
            ("self_attn.k_proj", [8, 24]),
            ("self_attn.v_proj", [8, 24]),
            ("self_attn.o_proj", [24, 24]),
            ("mlp.gate_proj", [40, 24]),
            ("mlp.up_proj", [40, 24]),
            ("mlp.down_proj", [24, 40]),
        ] {
            tensor(format!("model.layers.{layer}.{name}.weight"), &shape, false);
        }
    }
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    let path = std::env::temp_dir().join(format!("forja-smollm2-{}", std::process::id()));
    fs::write(&path, bytes)?;
    Ok(path)
}

fn optimized_load() -> EngineLoadConfig {
    EngineLoadConfig {
        max_context: Some(64),
        tunings: vec!["optimized".to_owned()],
        fixed_variant_picks: [
            ("dense.decode", "matmul.gemv-transposed"),
            ("dense.prefill", "matmul.steel-32x64x16-1x2"),
            ("attention.decode", "sdpa.decomposed"),
            ("attention.prefill", "sdpa.decomposed"),
            ("paged-attention.decode", "paged-sdpa.vector"),
            ("paged-attention.prefill", "paged-sdpa.steel"),
        ]
        .map(|(site, name)| forja_sdk::FixedVariantPick {
            site: site.to_owned(),
            name: name.to_owned(),
        })
        .to_vec(),
        ..EngineLoadConfig::default()
    }
}

#[test]
fn optimized_registry_requires_complete_known_selections() {
    use smollm2::variants::Variants;
    let load = optimized_load();
    let picks = Variants::new(&load, 64).unwrap();
    assert_eq!(picks.linear(1).unwrap().name(), "matmul.gemv-transposed");
    assert_eq!(picks.linear(7).unwrap().name(), "matmul.steel-32x64x16-1x2");
    assert!(picks.attention(33, 0, None, false).is_ok());
    let mut missing = load.clone();
    missing.fixed_variant_picks.pop();
    assert!(Variants::new(&missing, 64).is_err());
    let mut unknown = load;
    unknown.tunings.push("unknown".to_owned());
    assert!(Variants::new(&unknown, 64).is_err());
}

#[test]
fn metal_tiny_optimized_matches_reference() -> Result<(), Box<dyn Error>> {
    compare_paths(&optimized_load())
}

fn compare_paths(load: &EngineLoadConfig) -> Result<(), Box<dyn Error>> {
    compare_paths_on(load, Config::tiny(), forja_sdk::NativeDevice::Metal)
}

fn compare_paths_on(
    load: &EngineLoadConfig,
    config: Config,
    device: forja_sdk::NativeDevice,
) -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(device);
    let path = tiny_weights()?;
    let weights = Weights::open(&path)?;
    let plain = EngineLoadConfig {
        max_context: Some(64),
        ..EngineLoadConfig::default()
    };
    for count in [1, 7, 33] {
        let mut reference = SmolLM2::with_config(&weights, &plain, config)?;
        let mut optimized = SmolLM2::with_config(&weights, load, config)?;
        let tokens = (0..count).map(|index| index % 33).collect::<Vec<_>>();
        let mut outputs = Vec::new();
        for engine in [&mut reference, &mut optimized] {
            outputs.push(engine.step(forja_sdk::StepInput {
                phase: forja_sdk::BatchPhase::Prompt,
                tokens: Tensor::from_slice(&tokens, &[count])?,
                start_pos: 0,
                taps: true,
            })?);
        }
        assert_eq!(outputs[0].taps.len(), 2);
        assert_eq!(outputs[1].taps.len(), 2);
        for (expected, actual) in std::iter::once((&outputs[0].logits, &outputs[1].logits))
            .chain(outputs[0].taps.iter().zip(&outputs[1].taps))
        {
            forja_testing::assert_f32_values_agree(&expected.to_vec()?, &actual.to_vec()?)?;
        }
        forja_testing::assert_f32_values_agree(
            &step(&mut reference, &[7], count)?,
            &step(&mut optimized, &[7], count)?,
        )?;
    }
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn metal_tiny_fused_epilogues_match_reference() -> Result<(), Box<dyn Error>> {
    let mut load = optimized_load();
    load.tunings.push("fused-epilogues".to_owned());
    compare_paths(&load)
}

#[test]
fn serving_metadata_preserves_fixed_prompt_boundaries() {
    let info = SmolLM2::describe().serving.unwrap();
    assert_eq!(info.page_tokens, 64);
    assert_eq!(info.planes.len(), 60);
    assert_eq!(info.planes[0].elements_per_token, 192);
    assert_eq!(info.prefill_chunk_multiple, None);
}

#[test]
fn cpu_tiny_paths_match_with_and_without_fusion() -> Result<(), Box<dyn Error>> {
    let mut load = optimized_load();
    compare_paths_on(&load, Config::tiny_cpu(), forja_sdk::NativeDevice::Cpu)?;
    load.tunings.push("fused-epilogues".to_owned());
    compare_paths_on(&load, Config::tiny_cpu(), forja_sdk::NativeDevice::Cpu)
}
