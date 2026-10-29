use super::{Error, Weights, fixture, verification};

pub(super) fn tiny_load() -> forja_sdk::EngineLoadConfig {
    let mut load = verification::tiny_load();
    load.fixed_variant_picks = [
        ("quant", "quant-matmul.portable"),
        ("attention", "sdpa.decomposed"),
        ("sample", "sample.parallel"),
    ]
    .map(|(site, name)| forja_sdk::FixedVariantPick {
        site: site.to_owned(),
        name: name.to_owned(),
    })
    .to_vec();
    load
}
#[test]
fn metal_tiny_model_matches_independent_transformers() -> Result<(), Box<dyn Error>> {
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
}
