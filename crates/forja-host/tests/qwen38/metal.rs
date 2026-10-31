use super::{Error, Weights, fixture, verification};

pub(super) fn tiny_load() -> forja_sdk::EngineLoadConfig {
    let mut load = verification::tiny_load();
    load.fixed_variant_picks = [
        ("quant", "quant-matmul.portable"),
        ("attention", "sdpa.decomposed"),
        ("sample", "sample.parallel"),
        ("delta-prefill", "delta-rule-block.wy64-fixed"),
    ]
    .map(|(site, name)| forja_sdk::FixedVariantPick {
        site: site.to_owned(),
        name: name.to_owned(),
    })
    .to_vec();
    load
}
#[tokio::test]
async fn metal_tiny_model_matches_independent_transformers() -> Result<(), Box<dyn Error>> {
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let weights = Weights::open(fixture("model.safetensors"))?;
    let fixtures = golden_fixtures::FixtureDirectory::open(fixture(""))?;
    let mut config = qwen38::Config::tiny();
    config.cpu = false;
    let mut model = qwen38::Model::load(&weights, &tiny_load(), config)?;
    verification::verify(
        &mut model,
        fixtures.prompt("tiny").ok_or("missing fixture")?,
    )
    .await
}

#[test]
fn metal_tiled_projection_padding_preserves_independent_row_bits() -> Result<(), Box<dyn Error>> {
    use forja_sdk::{Tensor, bf16, target::metal::Variant};
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    let weights = Weights::open(fixture("model.safetensors"))?;
    let projection = qwen38::quant::Affine::load(
        &weights.scoped("language_model.model.layers.0.linear_attn.in_proj_qkv"),
        64,
        96,
    )?;
    let pick = Variant::new("quant-matmul.tiled")?;
    for rows in [1, 2, 4, 7, 8, 16, 33] {
        let values = (0..rows * 64)
            .map(|i| bf16::from_f64(f64::from(i % 33) / 32.0 - 0.5))
            .collect::<Vec<_>>();
        let input = Tensor::from_slice(&values, &[rows, 64])?;
        let actual = projection.forward(&input, Some(&pick))?.to_vec()?;
        let mut singles = Vec::new();
        for row in 0..rows {
            singles.extend(
                projection
                    .forward(&input.narrow(0, row, 1)?, Some(&pick))?
                    .to_vec()?,
            );
        }
        assert_eq!(actual, singles);
    }
    Ok(())
}
