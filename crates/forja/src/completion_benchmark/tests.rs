use super::*;

fn fixture() -> Result<(std::path::PathBuf, Options, DevConfig), Box<dyn Error>> {
    let root = forja_testing::temporary_directory("completion-record")?;
    let weights = root.join("model.safetensors");
    fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../forja-host/tests/fixtures/qwen38/model.safetensors"),
        &weights,
    )?;
    let profile: Profile = serde_json::from_value(json!({
        "schema-version":1,"family":"fixture",
        "model":{"id":"fixture","revision":"0".repeat(40),"weights-sha256":forja_config::single_weights_sha256(&weights, fs::metadata(&weights)?.len())?},
        "target":{"component":"wasm32-wasip2","backend":"cpu","device-family":"cpu"},
        "numerics":{"activations":"f32","weights":"f32"},
        "workload":{"kind":"causal-lm","batch":1,"max-context":128,"prefill-chunk":16},
        "serving":{"prompt-token-budget":16,"paged-cache":true},
        "prompt-lookup":{"min-match":8,"max-suffix":32,"max-drafts":2,"k-rule":[{"lo":0,"hi":127,"k":2}]}
    }))?;
    let component = root.join("engine.wasm");
    fs::write(
        &component,
        forja_config::embed_profile(&fs::read(test_guests::verify_smoke())?, &profile)?,
    )?;
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(
            [
                ("a".to_owned(), 0),
                ("b".to_owned(), 1),
                ("c".to_owned(), 2),
                ("d".to_owned(), 3),
            ],
            Vec::new(),
        )
        .build()
        .map_err(|e| e.to_string())?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .save(root.join("tokenizer.json"), false)
        .map_err(|e| e.to_string())?;
    fs::write(root.join("prompt.txt"), "abcd".repeat(8))?;
    fs::write(root.join("padding.txt"), "abcd")?;
    let suite = root.join("suite.json");
    fs::write(
        &suite,
        serde_json::to_vec(
            &json!({"schema_version":1,"source":"deterministic CPU fixture","target":"fixture",
        "padding":"padding.txt","context_semantics":"natural prompt","policy":{"min_match":8,"max_suffix":32,"max_drafts":2},
        "rows":[{"name":"copy","group":"regular","prompt":"prompt.txt","context_tokens":0,"output_tokens":7,
            "concurrency":1,"temperature":0.0,"seed":42,"minimum_speedup":1.15}]}),
        )?,
    )?;
    let options = Options {
        engine: component,
        model: root.clone(),
        chat_format: test_guests::continuous_chat_format().to_owned(),
        suite,
        group: "regular".to_owned(),
        row: Vec::new(),
        mode: Mode::Plain,
        cached: false,
        reps: 1,
        warmups: 0,
        json: root.join("plain.json"),
    };
    let mut config = DevConfig::default();
    config.backend.kind = BackendKind::Cpu;
    config.paths.scratch = root.join("scratch");
    config.limits.dispatches_per_list = forja_config::Unbounded::Limited(8192);
    Ok((root, options, config))
}

#[tokio::test]
async fn completion_loading_uses_the_validated_profile() -> Result<(), Box<dyn Error>> {
    let (root, options, config) = fixture()?;
    let (profile, _, inputs) = prepare(&options, &config)?;
    let session = engine::load(
        forja_cpu::CpuBackend::new(),
        crate::engine::limits(&config.limits)?,
        &options,
        &config,
        &profile,
        &inputs,
    )
    .await?;
    session.close().await?;
    fs::remove_dir_all(root)?;
    Ok(())
}

#[tokio::test]
async fn initialization_admits_the_requested_lane_group() -> Result<(), Box<dyn Error>> {
    let (root, options, config) = fixture()?;
    let (profile, _, inputs) = prepare(&options, &config)?;
    let session = engine::load(
        forja_cpu::CpuBackend::new(),
        crate::engine::limits(&config.limits)?,
        &options,
        &config,
        &profile,
        &inputs,
    )
    .await?;
    let request = serde_json::from_value(
        json!({"model":profile.model().id,"prompt":"abcd".repeat(8),"max_tokens":12,"temperature":0.0,"seed":42}),
    )?;
    let warmup = warmup::run(&session, request, 1, 4, 32).await?;
    assert_eq!(warmup.groups, 1);
    assert_eq!(warmup.concurrency, 4);
    assert_eq!(warmup.output_tokens_per_lane, 12);
    assert_eq!(warmup.full_prompt_requests, 4);
    session.close().await?;
    fs::remove_dir_all(root)?;
    Ok(())
}
