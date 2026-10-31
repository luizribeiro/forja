use super::{Error, verification};
use std::{env, path::PathBuf, time::Duration};

type Runner = forja_host::EngineRunner<forja_metal::MetalBackend>;
#[path = "real/serving.rs"]
mod serving;

async fn load() -> Result<(Runner, golden_fixtures::FixtureDirectory), Box<dyn Error>> {
    load_with(false, None).await
}
async fn load_with(
    replay: bool,
    layers: Option<u32>,
) -> Result<(Runner, golden_fixtures::FixtureDirectory), Box<dyn Error>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?)
        .join("Qwen3.8-27B-4bit");
    let fixtures = golden_fixtures::FixtureDirectory::open(root.join("forja-fixtures"))?;
    fixtures.require_complete_model_outputs()?;
    let weights = root.join("model.safetensors");
    if layers.is_none() && !replay {
        assert_eq!(
            golden_fixtures::sha256_file(&weights)?,
            fixtures.model_sha256()
        );
    }
    let component = env::var_os("FORJA_QWEN38_COMPONENT")
        .map_or_else(|| test_guests::qwen38().to_path_buf(), PathBuf::from);
    let profile = forja_config::read_profile(&std::fs::read(&component)?)?;
    assert_eq!(
        profile.model().weights_sha256,
        format!("sha256:{}", fixtures.model_sha256())
    );
    eprintln!(
        "component={} component-sha256={} weights-sha256={} profile-sha256={}",
        component.display(),
        golden_fixtures::sha256_file(&component)?,
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
    .with_command_limits(4096, u64::MAX)
    .with_gpu_limits(Duration::from_secs(10), Duration::from_secs(3600));
    let mut engine = forja_host::EngineRunner::new(
        &component,
        forja_metal::MetalBackend::new()?,
        limits,
        weights,
    )
    .await?;
    let mut config = crate::common::load_config(&component)?;
    config.replay = replay;
    if layers.is_some() {
        config.max_context = Some(64);
    }
    engine
        .load_with_selections(layers, &config)
        .await
        .map_err(|error| format!("outer load replay={replay} layers={layers:?}: {error:?}"))?
        .map_err(|error| format!("load replay={replay} layers={layers:?}: {error:?}"))?;
    Ok((engine, fixtures))
}
async fn verify_component(name: &str) -> Result<(), Box<dyn Error>> {
    let (mut engine, fixtures) = load().await?;
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

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_feedback_and_reset_match_explicit_tokens() -> Result<(), Box<dyn Error>> {
    let (mut engine, fixtures) = load().await?;
    let prompt = fixtures.prompt("hello").ok_or("missing prompt fixture")?;
    let tokens = prompt
        .prompt_ids()
        .iter()
        .map(|&id| u32::try_from(id))
        .collect::<Result<Vec<_>, _>>()?;
    let sampling = forja_host::SamplingParams::default();
    let first = engine
        .decode(forja_host::EngineDecode {
            phase: forja_host::BatchPhase::Prompt,
            tokens: Some(tokens.clone()),
            start_pos: 0,
            sampling,
        })
        .await??;
    let selected = u32::from_le_bytes(
        engine
            .read(&first.token)
            .await?
            .try_into()
            .map_err(|_| "invalid selected token")?,
    );
    let logits = golden_fixtures::decode_f32_le(&engine.read(&first.logits).await?)?;
    assert_eq!(
        Some(usize::try_from(selected)?),
        verification::top1(&logits)
    );
    let next = engine
        .decode(forja_host::EngineDecode {
            phase: forja_host::BatchPhase::Decode,
            tokens: None,
            start_pos: u32::try_from(tokens.len())?,
            sampling,
        })
        .await??;
    let retained = golden_fixtures::decode_f32_le(&engine.read(&next.logits).await?)?;
    let position = u32::try_from(tokens.len())?;
    engine
        .step(forja_host::EngineStep {
            phase: forja_host::BatchPhase::Prompt,
            tokens,
            start_pos: 0,
            taps: false,
        })
        .await??;
    let explicit = engine
        .decode(forja_host::EngineDecode {
            phase: forja_host::BatchPhase::Decode,
            tokens: Some(vec![selected]),
            start_pos: position,
            sampling,
        })
        .await??;
    assert_eq!(
        retained,
        golden_fixtures::decode_f32_le(&engine.read(&explicit.logits).await?)?
    );
    Ok(())
}
