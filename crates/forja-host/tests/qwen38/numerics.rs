use super::{Error, Tensor};
#[path = "../../../../engines/qwen38/src/numerics/decay.rs"]
mod guest;

// Frozen from torch 2.8.0: float32 F.softplus with beta=1 and threshold=20.
const INPUT: [u32; 15] = [
    0xc2c8_0000,
    0xc1f0_0000,
    0xc1a0_0000,
    0xc180_0000,
    0xc120_0000,
    0xc0a0_0000,
    0xc090_0000,
    0xbf80_0000,
    0x0000_0000,
    0x3f80_0000,
    0x4120_0000,
    0x41a0_0000,
    0x41a0_0001,
    0x41a8_0000,
    0x42c8_0000,
];
const SOFTPLUS: [u32; 15] = [
    0x0000_001c,
    0x29d2_b706,
    0x310d_a433,
    0x33f1_aadd,
    0x383e_6ab3,
    0x3bdc_0c6d,
    0x3c35_019a,
    0x3ea0_63d6,
    0x3f31_7218,
    0x3fa8_18f5,
    0x4120_0030,
    0x41a0_0000,
    0x41a0_0001,
    0x41a8_0000,
    0x42c8_0000,
];
const DECAY: [u32; 15] = [
    0x3f80_0000,
    0x3f7f_fd06,
    0x3ebc_5ab2,
    0x1816_482c,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
];

fn inputs() -> forja_sdk::Result<Tensor<f32>> {
    Tensor::from_slice(&INPUT.map(f32::from_bits), &[15])
}
fn agree(bits: &[u32], actual: &[f32]) -> Result<(), Box<dyn Error>> {
    let expected = bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
    let error = golden_fixtures::normwise_relative_error(&expected, actual)?;
    assert!(error <= 1.0e-5, "float32 reference error {error}");
    Ok(())
}
#[test]
fn softplus_matches_transformers_float32_and_threshold() -> Result<(), Box<dyn Error>> {
    agree(&SOFTPLUS, &INPUT.map(f32::from_bits).map(guest::softplus))
}
#[test]
fn negative_softplus_retains_transformers_decay() -> Result<(), Box<dyn Error>> {
    let bias = Tensor::<f32>::zeros(&[15])?;
    let log_a = Tensor::from_slice(&[20.0_f32; 15], &[15])?;
    let actual = guest::decay(&inputs()?, &bias, &log_a)?.to_vec()?;
    for (expected, actual) in DECAY.iter().zip(actual) {
        agree(std::slice::from_ref(expected), &[actual])?;
    }
    Ok(())
}
