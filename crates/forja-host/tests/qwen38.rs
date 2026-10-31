//! Independent projection and hybrid architecture coverage.
#[path = "qwen38/numerics.rs"]
mod numerics;
#[path = "qwen38/weights.rs"]
mod weights;
use forja_sdk::{Tensor, Weights, bf16};
use std::error::Error;
#[test]
fn affine_projection_validates_storage_and_matches_dequantized_rows() -> Result<(), Box<dyn Error>>
{
    let mut source = weights::Weights::default();
    source.tensor("weight", "U32", &[2, 8], &[0x11; 64]);
    source.bf16("scales", &[2, 1], &[0.125, 0.25]);
    source.bf16("biases", &[2, 1], &[-0.25, 0.0]);
    let path = source.save("affine")?;
    let weights = Weights::open(&path)?;
    let projection = qwen38::quant::Affine::load(&weights, 64, 2)?;
    let x = Tensor::from_slice(&[bf16::from_f32(1.0); 64], &[1, 64])?;
    let projected = projection.forward(&x, None)?.to_dtype::<f32>()?.to_vec()?;
    assert_eq!(projected, [-8.0, 16.0]);
    let ids = Tensor::from_slice(&[1, 0], &[2])?;
    let embedded = projection.embed(&ids)?.to_dtype::<f32>()?.to_vec()?;
    assert_eq!(embedded[..64], [0.25; 64]);
    assert_eq!(embedded[64..], [-0.125; 64]);
    assert!(qwen38::quant::Affine::load(&weights, 7, 2).is_err());
    assert!(qwen38::quant::Affine::load(&weights, 128, 2).is_err());
    std::fs::remove_file(path)?;
    Ok(())
}

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/qwen38")
        .join(name)
}
fn agree(expected: &[f32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    let error = golden_fixtures::normwise_relative_error(expected, actual)?;
    assert!(error <= 1.0e-2, "relative error {error}");
    Ok(())
}
#[test]
fn recurrent_block_matches_transformers_token_by_token() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let reference = Weights::open(fixture("blocks.safetensors"))?;
    let inputs = reference
        .tensor::<f32>("block_input_0", &[3, 64])?
        .to_dtype::<bf16>()?;
    let outputs = reference
        .tensor::<f32>("block_output_0", &[3, 64])?
        .to_vec()?;
    let mut block = qwen38::DeltaNet::load(
        &weights.scoped("language_model.model.layers.0.linear_attn"),
        qwen38::Config::tiny(),
    )?;
    for _ in 0..2 {
        for row in 0..3 {
            let actual = block
                .deltanet_layer(&inputs.narrow(0, row, 1)?, None)?
                .to_dtype::<f32>()?
                .to_vec()?;
            let start = usize::try_from(row)? * 64;
            agree(&outputs[start..start + 64], &actual)?;
        }
        block.reset()?;
    }
    Ok(())
}

#[test]
fn recurrent_prefill_matches_transformers_block_outputs() -> Result<(), Box<dyn Error>> {
    check_recurrent_prefill(None)
}

#[test]
fn metal_recurrent_prefill_matches_transformers_block_outputs() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    check_recurrent_prefill(Some(&forja_sdk::target::metal::Variant::new(
        "delta-rule-block.wy64-fixed",
    )?))
}

fn check_recurrent_prefill(
    pick: Option<&forja_sdk::target::metal::Variant>,
) -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let reference = Weights::open(fixture("blocks.safetensors"))?;
    let input = reference
        .tensor::<f32>("block_input_0", &[3, 64])?
        .to_dtype::<bf16>()?;
    let expected = reference
        .tensor::<f32>("block_output_0", &[3, 64])?
        .to_vec()?;
    let mut block = qwen38::DeltaNet::load(
        &weights.scoped("language_model.model.layers.0.linear_attn"),
        qwen38::Config::tiny(),
    )?;
    agree(
        &expected,
        &block
            .prefill(&input, None, pick)?
            .to_dtype::<f32>()?
            .to_vec()?,
    )?;
    block.reset()?;
    agree(
        &expected,
        &block
            .prefill(&input, None, pick)?
            .to_dtype::<f32>()?
            .to_vec()?,
    )
}

#[test]
fn attention_block_matches_transformers_with_per_head_gates() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let reference = Weights::open(fixture("blocks.safetensors"))?;
    let inputs = reference
        .tensor::<f32>("block_input_3", &[3, 64])?
        .to_dtype::<bf16>()?;
    let outputs = reference
        .tensor::<f32>("block_output_3", &[3, 64])?
        .to_vec()?;
    let mut block = qwen38::Attention::load(
        &weights.scoped("language_model.model.layers.3.self_attn"),
        qwen38::Config::tiny(),
        64,
    )?;
    for row in 0..3 {
        let actual = block
            .attention_layer(&inputs.narrow(0, row, 1)?, row, None, None)?
            .to_dtype::<f32>()?
            .to_vec()?;
        let start = usize::try_from(row)? * 64;
        agree(&outputs[start..start + 64], &actual)?;
    }
    Ok(())
}

#[test]
fn attention_prefill_preserves_per_row_reference_outputs() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let reference = Weights::open(fixture("blocks.safetensors"))?;
    let input = reference
        .tensor::<f32>("block_input_3", &[3, 64])?
        .to_dtype::<bf16>()?;
    let expected = reference
        .tensor::<f32>("block_output_3", &[3, 64])?
        .to_vec()?;
    let mut block = qwen38::Attention::load(
        &weights.scoped("language_model.model.layers.3.self_attn"),
        qwen38::Config::tiny(),
        64,
    )?;
    agree(
        &expected,
        &block
            .prefill(&input, 0, None, None)?
            .to_dtype::<f32>()?
            .to_vec()?,
    )
}

#[test]
fn feed_forward_preserves_the_bf16_silu_boundary() -> Result<(), Box<dyn Error>> {
    let mut source = weights::Weights::default();
    for (name, input, output) in [
        ("gate_proj", 64_u32, 128_u32),
        ("up_proj", 64, 128),
        ("down_proj", 128, 64),
    ] {
        let bytes = usize::try_from(input * output / 2)?;
        let groups = usize::try_from(input * output / 64)?;
        source.tensor(
            &format!("{name}.weight"),
            "U32",
            &[output, input / 8],
            &vec![0x11; bytes],
        );
        let scale = if input == 64 { 0.015_625 } else { 0.007_812_5 };
        source.bf16(
            &format!("{name}.scales"),
            &[output, input / 64],
            &vec![scale; groups],
        );
        source.bf16(
            &format!("{name}.biases"),
            &[output, input / 64],
            &vec![0.0; groups],
        );
    }
    let path = source.save("ffn")?;
    let weights = Weights::open(&path)?;
    let block = qwen38::FeedForward::load(&weights, qwen38::Config::tiny())?;
    let input = Tensor::from_slice(&[bf16::from_f32(1.0); 64], &[1, 64])?;
    assert_eq!(
        block.ffn(&input, None)?.to_dtype::<f32>()?.to_vec()?,
        [0.730_468_75; 64]
    );
    std::fs::remove_file(path)?;
    Ok(())
}

#[test]
fn decoder_residuals_match_independent_transformers_layers() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let fixtures = golden_fixtures::FixtureDirectory::open(fixture(""))?;
    let reference = fixtures.prompt("tiny").ok_or("missing tiny fixture")?;
    for index in [0, 3] {
        let input = reference
            .hidden_state(index)
            .ok_or("missing decoder input")?;
        let input = Tensor::from_slice(input.values(), &[3, 64])?.to_dtype::<bf16>()?;
        let expected = reference
            .hidden_state(index + 1)
            .ok_or("missing decoder output")?
            .values();
        let mut layer = qwen38::DecoderLayer::load(
            &weights.scoped(format!("language_model.model.layers.{index}")),
            qwen38::Config::tiny(),
            u32::try_from(index)?,
            64,
        )?;
        for row in 0..3 {
            let mut output = layer.forward(&input.narrow(0, row, 1)?, row, None, None)?;
            if index == 3 {
                output = output.rms_norm(
                    &weights.tensor("language_model.model.norm.weight", &[64])?,
                    1.0e-6,
                )?;
            }
            let start = usize::try_from(row)? * 64;
            agree(
                &expected[start..start + 64],
                &output.to_dtype::<f32>()?.to_vec()?,
            )?;
        }
    }
    Ok(())
}

#[path = "qwen38/verification.rs"]
mod verification;

#[test]
fn serving_catalog_preserves_eager_bucket_and_tail_budgets() {
    use forja_sdk::{BatchPhase, BatchSelection};
    let shapes = qwen38::Qwen38::serving_shapes();
    assert_eq!(shapes.len(), 15);
    for bucket in [1, 2, 4, 8] {
        let decode = shapes
            .iter()
            .filter(|shape| shape.phase == BatchPhase::Decode && shape.bucket == bucket)
            .collect::<Vec<_>>();
        assert_eq!(decode.len(), 3);
        for selection in [
            BatchSelection::Combined,
            BatchSelection::Forward,
            BatchSelection::Masked,
        ] {
            assert_eq!(
                decode
                    .iter()
                    .filter(|shape| shape.selection == selection && shape.tokens == 1)
                    .count(),
                1
            );
        }
    }
    let prompts = shapes
        .iter()
        .filter(|shape| shape.phase == BatchPhase::Prompt)
        .collect::<Vec<_>>();
    assert_eq!(
        prompts.iter().map(|shape| shape.tokens).collect::<Vec<_>>(),
        [64, 512, 2048]
    );
    assert!(
        prompts
            .iter()
            .all(|shape| shape.bucket == 1 && shape.selection == BatchSelection::Combined)
    );
    let retained = shapes
        .iter()
        .filter(|shape| shape.phase == BatchPhase::Prompt || shape.bucket <= 4)
        .collect::<Vec<_>>();
    assert_eq!(retained.len() + 1, 13);
    assert!(retained.iter().all(|shape| shape.bucket <= 4));
}

#[test]
fn eager_engine_feedback_matches_explicit_tokens() -> Result<(), Box<dyn Error>> {
    use forja_sdk::{DecodeInput, Engine, SamplingParams};
    let weights = Weights::open(fixture("model.safetensors"))?;
    let load = verification::tiny_load();
    let mut retained = qwen38::Qwen38::with_config(&weights, &load, qwen38::Config::tiny())?;
    let mut explicit = qwen38::Qwen38::with_config(&weights, &load, qwen38::Config::tiny())?;
    let sampling = SamplingParams::default();
    let input = || Tensor::from_slice(&[1, 7, 32], &[3]);
    let first = retained.decode(DecodeInput {
        phase: forja_sdk::BatchPhase::Prompt,
        tokens: Some(input()?),
        start_pos: 0,
        sampling,
    })?;
    let other = explicit.decode(DecodeInput {
        phase: forja_sdk::BatchPhase::Prompt,
        tokens: Some(input()?),
        start_pos: 0,
        sampling,
    })?;
    assert_eq!(first.token.to_vec()?, other.token.to_vec()?);
    let mut token = first.token.to_vec()?;
    for position in 3..10 {
        let actual = retained.decode(DecodeInput {
            phase: forja_sdk::BatchPhase::Decode,
            tokens: None,
            start_pos: position,
            sampling,
        })?;
        let expected = explicit.decode(DecodeInput {
            phase: forja_sdk::BatchPhase::Decode,
            tokens: Some(Tensor::from_slice(&token, &[1])?),
            start_pos: position,
            sampling,
        })?;
        token = actual.token.to_vec()?;
        assert_eq!(token, expected.token.to_vec()?);
        assert_eq!(actual.logits.to_vec()?, expected.logits.to_vec()?);
    }
    assert!(
        retained
            .decode(DecodeInput {
                phase: forja_sdk::BatchPhase::Decode,
                tokens: None,
                start_pos: 0,
                sampling
            })
            .is_err()
    );
    assert!(qwen38::Qwen38::describe().serving.is_some());
    Ok(())
}

#[cfg(target_os = "macos")]
#[path = "qwen38/metal.rs"]
mod metal;

#[path = "qwen38/execution.rs"]
mod execution;

#[test]
fn component_metadata_declares_the_text_only_batched_contract() -> Result<(), Box<dyn Error>> {
    use forja_sdk::Engine;
    let profile = forja_config::read_profile(&std::fs::read(test_guests::qwen38())?)?;
    assert_eq!(profile.family(), "qwen38");
    assert_eq!(profile.workload().max_context.get(), 32768);
    assert!(
        profile
            .serving()
            .ok_or("missing serving settings")?
            .paged_cache
    );
    let info = qwen38::Qwen38::describe();
    assert_eq!(info.vocab, 248_320);
    assert_eq!(info.tap_layers.len(), 64);
    let serving = info.serving.ok_or("missing serving geometry")?;
    assert_eq!(
        (
            serving.page_tokens,
            serving.active_lanes,
            serving.planes.len()
        ),
        (64, 8, 32)
    );
    assert!(
        serving
            .planes
            .iter()
            .all(|plane| plane.elements_per_token == 1024 && plane.alignment == 256)
    );
    assert_eq!(serving.batch_shapes.len(), 15);
    assert_eq!(profile.workload().prefill_chunk.get(), 2048);
    assert_eq!(
        profile
            .serving()
            .ok_or("missing serving settings")?
            .prompt_token_budget
            .get(),
        2048
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[path = "qwen38/real.rs"]
mod real;

#[path = "qwen38/lifecycle.rs"]
mod lifecycle;

#[test]
fn invalid_spans_refuse_cleanly_and_execution_failure_requires_reload() -> Result<(), Box<dyn Error>>
{
    let weights = Weights::open(fixture("model.safetensors"))?;
    let mut model =
        qwen38::Model::load(&weights, &verification::tiny_load(), qwen38::Config::tiny())?;
    let token = Tensor::from_slice(&[1], &[1])?;
    model.forward(&token, 0, false)?;
    assert!(model.forward(&token, 2, false).is_err());
    assert!(model.forward(&token, 64, false).is_err());
    assert!(
        model
            .forward(&Tensor::<u32>::from_slice(&[], &[0])?, 1, false)
            .is_err()
    );
    model.decode_token(&token, 1, false)?;
    assert!(
        model
            .decode_token(&Tensor::from_slice(&[33], &[1])?, 2, false)
            .is_err()
    );
    assert!(model.forward(&token, 0, false).is_err());
    Ok(())
}

#[test]
fn metal_prefill_layer_preserves_attention_variant_refusals() -> Result<(), Box<dyn Error>> {
    use forja_sdk::target::metal::Variant;
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let weights = Weights::open(fixture("model.safetensors"))?;
    let input = Tensor::from_slice(&[bf16::from_f32(0.125); 3 * 64], &[3, 64])?;
    let mut layer = qwen38::DecoderLayer::load(
        &weights.scoped("language_model.model.layers.3"),
        qwen38::Config::tiny(),
        3,
        64,
    )?;
    let valid = Variant::new("sdpa.decomposed")?;
    layer.prefill(&input, 0, None, Some(&valid), None)?;
    let wrong = Variant::new("matmul.portable")?;
    assert!(layer.prefill(&input, 3, None, Some(&wrong), None).is_err());
    Ok(())
}

#[test]
fn prefill_blocks_refuse_invalid_token_and_hidden_widths() -> Result<(), Box<dyn Error>> {
    let weights = Weights::open(fixture("model.safetensors"))?;
    let config = qwen38::Config::tiny();
    let mut recurrent = qwen38::DeltaNet::load(
        &weights.scoped("language_model.model.layers.0.linear_attn"),
        config,
    )?;
    let mut attention = qwen38::Attention::load(
        &weights.scoped("language_model.model.layers.3.self_attn"),
        config,
        128,
    )?;
    for shape in [&[65, 64][..], &[1, 7], &[1, 1, 64]] {
        let input = Tensor::<bf16>::zeros(shape)?;
        assert!(recurrent.prefill(&input, None, None).is_err());
        assert!(attention.prefill(&input, 0, None, None).is_err());
    }
    Ok(())
}

#[test]
fn private_serving_planes_cover_each_recurrent_layer() {
    for (config, layers, channels, heads, width) in [
        (qwen38::Config::model(), 48, 10_240, 48, 128),
        (qwen38::Config::tiny(), 3, 96, 4, 16),
    ] {
        let planes = config.state_planes().unwrap();
        assert_eq!(planes.len(), layers * 2);
        for pair in planes.as_chunks::<2>().0 {
            assert_eq!(pair[0].dtype, forja_sdk::DType::BF16);
            assert_eq!(pair[0].shape, [3, channels]);
            assert_eq!(pair[1].dtype, forja_sdk::DType::F32);
            assert_eq!(pair[1].shape, [heads, width, width]);
        }
    }
    assert!(
        qwen38::Config {
            key_heads: 0,
            ..qwen38::Config::tiny()
        }
        .state_planes()
        .is_err()
    );
}

#[path = "qwen38/component.rs"]
mod component;

#[path = "qwen38/serving.rs"]
mod serving;
