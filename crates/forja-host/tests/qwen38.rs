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
