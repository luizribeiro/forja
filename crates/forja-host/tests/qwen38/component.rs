use super::{Error, fixture, verification};

#[tokio::test]
async fn tiny_component_matches_teacher_forced_transformers() -> Result<(), Box<dyn Error>> {
    let mut runner = forja_host::EngineRunner::new(
        test_guests::qwen38_tiny(),
        forja_cpu::CpuBackend::new(),
        forja_host::Limits::new(1 << 30, 8, 1 << 30, 20_000, 1 << 27),
        fixture("model.safetensors"),
    )
    .await?;
    runner
        .load_with_selections(
            None,
            &forja_host::EngineLoadConfig {
                max_context: Some(64),
                replay: false,
                kv_cache: forja_host::KvCacheFormat::F16,
                ..forja_host::EngineLoadConfig::default()
            },
        )
        .await??;
    let fixtures = golden_fixtures::FixtureDirectory::open(fixture(""))?;
    verification::verify(
        &mut runner,
        fixtures.prompt("tiny").ok_or("missing fixture")?,
    )
    .await
}
