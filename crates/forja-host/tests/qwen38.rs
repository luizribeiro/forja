//! Independent projection and hybrid architecture coverage.
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
