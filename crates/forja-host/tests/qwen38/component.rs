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

#[tokio::test]
async fn slotted_recurrent_component_matches_eager_state_bits() -> Result<(), Box<dyn Error>> {
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
                paged_cache: true,
                kv_cache: forja_host::KvCacheFormat::F16,
                tunings: vec!["recurrent-fixture".into()],
                ..forja_host::EngineLoadConfig::default()
            },
        )
        .await??;
    assert!(!runner.supports_prefix_reuse());
    let weights = forja_sdk::Weights::open(fixture("model.safetensors"))?;
    let mut native = qwen38::DeltaNet::load(
        &weights.scoped("language_model.model.layers.0.linear_attn"),
        qwen38::Config::tiny(),
    )?;
    for request in 0..2 {
        native.reset()?;
        if request == 0 {
            runner.prepare_paged_request(None, 0)?;
            runner.prepare_paged_request(None, 64)?;
        }
        for (position, token) in [1_u16, 2, 7].into_iter().enumerate() {
            let input = forja_sdk::Tensor::from_slice(
                &[forja_sdk::bf16::from_f32(f32::from(token)); 64],
                &[1, 64],
            )?;
            let expected = native
                .deltanet_layer(&input, None)?
                .narrow(1, 0, 33)?
                .to_dtype::<f32>()?
                .to_vec()?;
            let output = runner
                .step(forja_host::EngineStep {
                    phase: if position == 0 {
                        forja_host::BatchPhase::Prompt
                    } else {
                        forja_host::BatchPhase::Decode
                    },
                    tokens: vec![u32::from(token)],
                    start_pos: u32::try_from(position)?,
                    taps: false,
                })
                .await??;
            assert_eq!(
                runner.read(&output.logits).await?,
                expected
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}
