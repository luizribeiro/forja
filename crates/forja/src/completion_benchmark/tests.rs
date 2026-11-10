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
        "rows":[{"name":"copy","group":"regular","prompt":"prompt.txt","context_tokens":0,"output_tokens":7,"min_output_tokens":7,
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
        plain_record: None,
        stream_comparison: crate::completion_stream::Comparison::Candidate,
        cached: false,
        measurement: options::Measurement {
            raw_samples: true,
            ..options::Measurement::default()
        },
        reps: 1,
        warmups: 1,
        json: root.join("plain.json"),
    };
    let mut config = DevConfig::default();
    config.backend.kind = BackendKind::Cpu;
    config.paths.scratch = root.join("scratch");
    config.limits.dispatches_per_list = forja_config::Unbounded::Limited(8192);
    Ok((root, options, config))
}

#[tokio::test]
async fn completion_records_prove_draft_streams_on_a_cpu_fixture() -> Result<(), Box<dyn Error>> {
    let (root, mut options, config) = fixture()?;
    run(&options, &config).await?;
    let document: Value = serde_json::from_slice(&fs::read(&options.json)?)?;
    assert_eq!(document["results"][0]["rounds"], 0);
    assert_eq!(document["results"][0]["identity"]["min_output_tokens"], 7);
    let sample = &document["results"][0]["samples"][0];
    assert!(
        (sample["decode_tokens_per_second"].as_f64().unwrap()
            * sample["decode_wall_seconds"].as_f64().unwrap()
            - 7.0)
            .abs()
            < 1e-9
    );
    assert!(document["provenance"]["date"].is_string());
    assert!(document["provenance"]["os_version"].is_string());
    assert!(document["provenance"]["machine_load"].is_object());
    assert_eq!(document["results"][0]["k"], 0);
    assert_eq!(document["results"][0]["drafter"]["identity"], "none");
    options.mode = Mode::Draft;
    options.plain_record = Some(options.json.clone());
    options.json = root.join("draft.json");
    let outcome = run(&options, &config).await;
    if crate::provenance::dirty() {
        assert!(outcome.unwrap_err().to_string().contains("dirty"));
    } else {
        outcome?;
    }
    let draft: Value = serde_json::from_slice(&fs::read(&options.json)?)?;
    assert_eq!(draft["results"][0]["k"], 2);
    let warmup = &draft["results"][0]["warmup"];
    assert_eq!(warmup["concurrency"], 1);
    assert_eq!(warmup["output_tokens_per_lane"], 12);
    assert!(warmup["observed_rounds"].as_u64().unwrap() > 0);
    assert_eq!(
        warmup["output_tokens_per_lane"],
        document["results"][0]["warmup"]["output_tokens_per_lane"]
    );

    assert!(draft["results"][0]["rounds"].as_u64().unwrap() > 0);
    assert!(
        draft["results"][0]["mean_accepted_per_round"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert_eq!(
        draft["results"][0]["token_stream_hashes"],
        document["results"][0]["token_stream_hashes"]
    );
    assert_eq!(draft["config_hash"], document["config_hash"]);
    assert_ne!(draft["perf_hash"], document["perf_hash"]);
    assert_eq!(
        draft["stream_check"]["status"],
        if crate::provenance::dirty() {
            "failed"
        } else {
            "pass"
        }
    );
    assert_eq!(
        draft["results"][0]["drafter"]["identity"],
        "host-prompt-lookup"
    );
    if let Some(destination) = std::env::var_os("FORJA_BENCH_FIXTURE_RECORDS") {
        fs::create_dir_all(&destination)?;
        for name in [
            "plain.json",
            "draft.json",
            "engine.wasm",
            "tokenizer.json",
            "model.safetensors",
            "suite.json",
            "prompt.txt",
            "padding.txt",
        ] {
            fs::copy(
                root.join(name),
                std::path::Path::new(&destination).join(name),
            )?;
        }
    }
    assert!(
        run(&options, &config)
            .await
            .unwrap_err()
            .to_string()
            .contains("overwrite")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn a_stream_mismatch_keeps_a_failed_record_and_returns_an_error() -> Result<(), Box<dyn Error>> {
    let (root, mut options, _) = fixture()?;
    let (plain, _) = crate::completion_stream::tests::pair();
    fs::write(&options.json, serde_json::to_vec(&plain)?)?;
    options.mode = Mode::Draft;
    options.plain_record = Some(options.json.clone());
    options.json = root.join("failed.json");
    let mut draft = plain;
    draft["mode"] = json!("draft");
    draft["results"][0]["token_stream_hashes"][0] = json!(format!("sha256:{}", "d".repeat(64)));
    assert!(
        write_record(&options, draft)
            .unwrap_err()
            .to_string()
            .contains("token-stream hash")
    );
    let failed: Value = serde_json::from_slice(&fs::read(&options.json)?)?;
    assert_eq!(failed["stream_check"]["status"], "failed");
    assert!(
        failed["stream_check"]["plain_record_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[tokio::test]
async fn warmup_and_measurement_complete_the_same_lane_group() -> Result<(), Box<dyn Error>> {
    let (root, options, config) = fixture()?;
    let mut suite: Value = serde_json::from_slice(&fs::read(&options.suite)?)?;
    suite["rows"][0]["concurrency"] = 4.into();
    suite["rows"][0]["minimum_speedup"] = 0.98.into();
    fs::write(&options.suite, serde_json::to_vec(&suite)?)?;
    run(&options, &config).await?;
    let record: Value = serde_json::from_slice(&fs::read(&options.json)?)?;
    let row = &record["results"][0];
    assert_eq!(row["warmup"]["concurrency"], 4);
    assert_eq!(row["warmup"]["full_prompt_requests"], 4);
    assert_eq!(row["samples"][0]["output_tokens"], json!([7, 7, 7, 7]));
    assert_eq!(row["finish_reasons"]["length"], 4);
    fs::remove_dir_all(root)?;
    Ok(())
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

#[test]
fn row_metadata_requires_a_declared_draft_cap() -> Result<(), Box<dyn Error>> {
    let (root, options, config) = fixture()?;
    let (_, suite, inputs) = prepare(&options, &config)?;
    let prompt = fs::read_to_string(root.join("prompt.txt"))?;
    let context = sample::RowContext {
        options: &options,
        inputs: &inputs,
        row: &suite.rows[0],
        prompt: &prompt,
        prompt_tokens: 32,
    };
    let warmup = || warmup::Warmup {
        groups: 1,
        concurrency: 1,
        prompt_tokens_per_lane: 32,
        output_tokens_per_lane: 12,
        observed_rounds: 0,
        full_prompt_requests: 1,
    };
    let samples = || {
        vec![sample::Sample {
            repetition: 0,
            wall_seconds: 1.0,
            gpu_seconds: 0.0,
            measurement_window: [0.0, 1.0],
            decode_wall_seconds: 1.0,
            decode_tokens_per_second: 7.0,
            verify_gpu_seconds: 0.0,
            plain_gpu_seconds: 0.0,
            prompt_tokens: vec![32],
            cached_tokens: vec![0],
            output_tokens: vec![7],
            finish_reasons: vec!["length".to_owned()],
        }]
    };
    let row = sample::RowRecord::new(
        context,
        forja_serve::DraftMetrics::default(),
        samples(),
        Vec::new(),
        warmup(),
    )?;
    assert_eq!(row.output_counts[&7], 1);
    assert_eq!(row.finish_reasons["length"], 1);
    let mut missing = inputs.clone();
    missing["lookup_policy"]
        .as_object_mut()
        .unwrap()
        .remove("max-drafts");
    let invalid = sample::RowContext {
        inputs: &missing,
        ..context
    };
    assert!(
        sample::RowRecord::new(
            invalid,
            forja_serve::DraftMetrics::default(),
            samples(),
            Vec::new(),
            warmup()
        )
        .is_err()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn workload_minima_must_be_positive_and_within_the_output_budget() -> Result<(), Box<dyn Error>> {
    let (root, options, config) = fixture()?;
    let mut suite: Value = serde_json::from_slice(&fs::read(&options.suite)?)?;
    for minimum in [json!(0), json!(8), Value::Null] {
        suite["rows"][0]["min_output_tokens"] = minimum;
        fs::write(&options.suite, serde_json::to_vec(&suite)?)?;
        assert!(prepare(&options, &config).is_err());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
