use super::{Error, verification};
use std::{env, path::PathBuf, time::Duration};

async fn verify_component(name: &str) -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?)
        .join("Qwen3.8-27B-4bit");
    let fixtures = golden_fixtures::FixtureDirectory::open(root.join("forja-fixtures"))?;
    fixtures.require_complete_model_outputs()?;
    let weights = root.join("model.safetensors");
    assert_eq!(
        golden_fixtures::sha256_file(&weights)?,
        fixtures.model_sha256()
    );
    let component = test_guests::qwen38();
    let profile = forja_config::read_profile(&std::fs::read(component)?)?;
    assert_eq!(
        profile.model().weights_sha256,
        format!("sha256:{}", fixtures.model_sha256())
    );
    eprintln!(
        "component={} component-sha256={} weights-sha256={} profile-sha256={}",
        component.display(),
        golden_fixtures::sha256_file(component)?,
        fixtures.model_sha256(),
        profile.sha256()?
    );
    let limits = forja_host::Limits::new(
        48 * 1024 * 1024 * 1024,
        8,
        1 << 34,
        20_000,
        128 * 1024 * 1024,
    )
    .with_gpu_limits(Duration::from_secs(10), Duration::from_secs(3600));
    let mut engine = forja_host::EngineRunner::new(
        component,
        forja_metal::MetalBackend::new()?,
        limits,
        weights,
    )
    .await?;
    let mut config = crate::common::load_config(component)?;
    config.replay = false;
    engine.load_with_selections(None, &config).await??;
    verification::verify(
        &mut engine,
        fixtures.prompt(name).ok_or("missing prompt fixture")?,
    )
    .await
}
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_hello_matches_teacher_forced_transformers() -> Result<(), Box<dyn Error>> {
    verify_component("hello").await
}
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_arithmetic_matches_teacher_forced_transformers() -> Result<(), Box<dyn Error>> {
    verify_component("arithmetic").await
}
#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_completion_matches_teacher_forced_transformers() -> Result<(), Box<dyn Error>> {
    verify_component("completion").await
}
