mod engine;
mod observed;
mod options;
mod record;
mod sample;
#[cfg(test)]
mod tests;
mod timing;
mod warmup;
mod worker;
mod workload;

use forja_config::{BackendKind, DevConfig, Profile};
use forja_serve::{ChatService, CompletionRequest, CompletionSession};
use golden_fixtures::sha256_file;
use serde_json::{Value, json};
use std::{error::Error, fs, io::Write as _, time::Instant};
use tokenizers::Tokenizer;

pub(crate) use options::{Mode, Options};

pub(crate) async fn run(options: &Options, config: &DevConfig) -> Result<(), Box<dyn Error>> {
    if options.json.exists() {
        return Err("refusing to overwrite a benchmark record".into());
    }
    let started = Instant::now();
    let (profile, suite, identities) = prepare(options, config)?;
    match config.backend.kind {
        BackendKind::Cpu => {
            execute(
                forja_cpu::CpuBackend::new(),
                crate::engine::limits(&config.limits)?,
                "CPU",
                options,
                config,
                &profile,
                &suite,
                &identities,
                started,
            )
            .await
        }
        BackendKind::Metal => {
            #[cfg(target_os = "macos")]
            {
                let metal = config.backend.metal.resolve();
                let (backend, live) = crate::engine::metal_backend(
                    metal.graph_replay,
                    config.backend.metal.pool_capacity,
                    &config.limits,
                )?;
                let device = backend.device_name();
                execute(
                    backend,
                    crate::engine::limits_for_live_bytes(&config.limits, live)?,
                    &device,
                    options,
                    config,
                    &profile,
                    &suite,
                    &identities,
                    started,
                )
                .await
            }
            #[cfg(not(target_os = "macos"))]
            return Err("Metal completion benchmarks require macOS".into());
        }
    }
}

fn prepare(
    options: &Options,
    config: &DevConfig,
) -> Result<(Profile, workload::Suite, Value), Box<dyn Error>> {
    let backend = match config.backend.kind {
        BackendKind::Cpu => crate::args::Backend::Cpu,
        BackendKind::Metal => crate::args::Backend::Metal,
    };
    let (profile, _) = crate::engine::validate_engine(
        &options.engine,
        &options.model,
        backend,
        &config.paths.scratch,
    )?;
    let suite = workload::Suite::load(&options.suite, &options.row, &options.group)?;
    if suite.target != profile.model().id {
        return Err("workload target differs from engine model".into());
    }
    let policy = serde_json::to_value(profile.prompt_lookup())?;
    for (key, expected) in suite
        .policy
        .as_object()
        .ok_or("suite policy must be an object")?
    {
        if policy.get(key.replace('_', "-")) != Some(expected) {
            return Err("workload and profile lookup policies differ".into());
        }
    }
    let identities = json!({"engine_sha256":sha256_file(&options.engine)?, "profile_hash":profile.sha256()?, "profile_family":profile.family(), "model":profile.model(), "numerics":profile.numerics(),
        "weights_sha256":profile.model().weights_sha256, "model_revision":profile.model().revision,
        "tokenizer_sha256":sha256_file(options.model.join("tokenizer.json"))?,
        "chat_format_sha256":sha256_file(&options.chat_format)?, "lookup_policy":policy});
    Ok((profile, suite, identities))
}

#[allow(clippy::too_many_arguments)]
async fn execute<B: forja_core::Backend + Send + Sync + 'static>(
    backend: B,
    limits: forja_host::Limits,
    device: &str,
    options: &Options,
    config: &DevConfig,
    profile: &Profile,
    suite: &workload::Suite,
    inputs: &Value,
    started: Instant,
) -> Result<(), Box<dyn Error>> {
    let session = engine::load(backend, limits, options, config, profile, inputs).await?;
    let result = async {
        if options.measurement.worker {
            worker::run(
                options, config, profile, suite, inputs, &session, device, started,
            )
            .await
        } else {
            write_record(
                options,
                measure(options, config, profile, suite, inputs, &session, device).await?,
            )
        }
    }
    .await;
    let cleanup = session.close().await;
    result?;
    cleanup?;
    Ok(())
}

fn write_record(options: &Options, mut document: Value) -> Result<(), Box<dyn Error>> {
    record::decorate(&mut document)?;
    let stream_check = if options.mode == Mode::Draft {
        let path = options
            .plain_record
            .as_ref()
            .ok_or("draft mode requires a plain record")?;
        let plain = serde_json::from_slice(&fs::read(path)?)?;
        let check = crate::completion_stream::assert_matches(&plain, &document);
        document["stream_check"] = json!({"status":if check.is_ok() {"pass"} else {"failed"},
            "plain_record_hash":record::hash(&plain)?, "error":check.as_ref().err()});
        check
    } else {
        Ok(())
    };
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.json)?;
    serde_json::to_writer_pretty(&mut output, &document)?;
    output.write_all(b"\n")?;
    stream_check?;
    eprintln!("completion record written; mode={:?}", options.mode);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn measure<S: ChatService>(
    options: &Options,
    config: &DevConfig,
    profile: &Profile,
    suite: &workload::Suite,
    inputs: &Value,
    session: &CompletionSession<S>,
    device: &str,
) -> Result<Value, Box<dyn Error>> {
    let tokenizer = Tokenizer::from_file(options.model.join("tokenizer.json"))
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let root = options
        .suite
        .parent()
        .ok_or("suite has no parent directory")?;
    let padding = fs::read_to_string(root.join(&suite.padding))?;
    let mut results = Vec::new();
    for row in &suite.rows {
        let source = fs::read_to_string(root.join(&row.prompt))?;
        let prompt = workload::render(
            &tokenizer,
            &source,
            &padding,
            row.context_tokens,
            (&suite.padding_prefix, &suite.padding_suffix),
        )?;
        let prompt_tokens = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|error| error.to_string())?
            .len();
        if prompt_tokens
            .checked_add(usize::try_from(row.output_tokens)?)
            .ok_or("context overflow")?
            > usize::try_from(profile.workload().max_context.get())?
        {
            return Err(format!("{} exceeds the profile context", row.name).into());
        }
        results.push(measure_row(session, options, inputs, row, &prompt, prompt_tokens).await?);
    }
    let mut measured_config = config.clone();
    measured_config.bench.reps =
        std::num::NonZeroU32::new(options.reps).ok_or("repetitions must be positive")?;
    measured_config.bench.warmups = options.warmups;
    measured_config.backend.metal.graph_replay =
        forja_config::Choice::Fixed(config.backend.metal.resolve().graph_replay);
    let mut effective = toml::Value::try_from(&measured_config)?;
    effective
        .as_table_mut()
        .ok_or("invalid benchmark config")?
        .remove("paths");
    effective["verify"]
        .as_table_mut()
        .ok_or("invalid verify config")?
        .remove("fixtures");
    let policy = &inputs["lookup_policy"];
    let mut host = serde_json::to_value(engine::host_config(options, config))?;
    host.as_object_mut()
        .ok_or("invalid host config")?
        .remove("model");
    let toml_config = toml::to_string(&effective)?;
    Ok(
        json!({"schema_version":1, "kind":"completion-benchmark", "implementation":"forja", "mode":options.mode,
        "inputs":[inputs], "config":toml_config,
        "stream_config":{"inputs":inputs, "config":effective, "host_config":host, "cached":options.cached, "policy":policy,
            "suite_sha256":sha256_file(&options.suite)?, "padding_sha256":sha256_file(root.join(&suite.padding))?,
            "reps":options.reps,"warmups":options.warmups},
        "provenance":{"commit":crate::provenance::commit(),"dirty":crate::provenance::dirty(),
            "binary_sha256":sha256_file(std::env::current_exe()?)?, "build_environment":crate::provenance::BUILD_ENVIRONMENT,
            "machine":crate::benchmark::command_output("hostname", &[])?, "os":std::env::consts::OS,
            "os_version":crate::benchmark::command_output("sw_vers", &["-productVersion"])?,
            "os_build":crate::benchmark::command_output("sw_vers", &["-buildVersion"])?, "device":device},
        "source":suite.source, "context_semantics":suite.context_semantics,
        "measurement_mode":{"sampling":"fixed", "early_stop":false, "explicit_no_early_stop":options.measurement.no_early_stop,
            "explicit_raw_samples":options.measurement.raw_samples, "execution":"scheduled-request-groups", "decoding":options.mode,
            "reps":options.reps,"warmups":options.warmups,"raw_samples":true,"cache_state":if options.cached {"prefix-reuse-enabled"} else {"fresh-request"}},
        "timing_semantics":"Wall samples cover a whole request group. Decode wall is the maximum lane decode span. Plain GPU includes prompt work and excludes Verify. Whole request groups are the independent trials.",
        "results":results}),
    )
}

async fn measure_row<S: ChatService>(
    session: &CompletionSession<S>,
    options: &Options,
    inputs: &Value,
    row: &workload::Row,
    prompt: &str,
    prompt_tokens: usize,
) -> Result<Value, Box<dyn Error>> {
    let policy = &inputs["lookup_policy"];
    let model = inputs["model"]["id"]
        .as_str()
        .ok_or("missing model identity")?;
    let warmup_tokens = warmup::output_tokens(policy, row.output_tokens)?;
    let request = |output| {
        serde_json::from_value::<CompletionRequest>(json!({"model":model,
        "prompt":prompt,"max_tokens":output,"temperature":row.temperature,"seed":row.seed,"top_p":1.0,"top_k":0}))
    };
    let warmup = warmup::run(
        session,
        request(warmup_tokens)?,
        options.warmups,
        row.concurrency,
        prompt_tokens,
    )
    .await?;
    let mut observations = forja_serve::DraftMetrics::default();
    let (mut samples, mut hashes) = (Vec::new(), Vec::new());
    for repetition in 0..options.reps {
        if options.cached {
            session.complete_texts(vec![request(1)?]).await?;
        }
        let requests = (0..row.concurrency)
            .map(|_| request(row.output_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let gpu_before = session.checkpoint().await?.gpu_time();
        let started = Instant::now();
        let window_start = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let completions = session.complete_texts(requests).await?;
        let wall = started.elapsed().as_secs_f64();
        let gpu = session
            .metrics()
            .gpu_time()
            .checked_sub(gpu_before)
            .ok_or("GPU counter moved backwards")?;
        let mut verify = 0_u64;
        for completion in &completions {
            sample::validate_usage(
                completion.usage,
                completion.finish_reason,
                prompt_tokens,
                row.output_tokens,
            )?;
            hashes.push(crate::benchmark_record::output_digest(
                completion.token_ids(),
                &[],
            ));
            observations.merge(&completion.drafting);
            verify = verify
                .checked_add(completion.drafting.verify_gpu_ns)
                .ok_or("verify GPU counter overflow")?;
        }
        let spans = completions
            .iter()
            .map(|completion| completion.drafting.max_decode_wall_ns)
            .collect::<Vec<_>>();
        let timing = timing::Timing::new(&spans, gpu, verify)?;
        samples.push(sample::Sample {
            repetition,
            wall_seconds: wall,
            gpu_seconds: gpu.as_secs_f64(),
            measurement_window: [window_start, window_start + wall],
            decode_wall_seconds: timing.decode,
            verify_gpu_seconds: timing.verify,
            plain_gpu_seconds: timing.plain,
            prompt_tokens: completions.iter().map(|c| c.usage.prompt_tokens).collect(),
            cached_tokens: completions.iter().map(|c| c.usage.cached_tokens).collect(),
            finish_reasons: completions
                .iter()
                .map(|c| sample::finish_name(c.finish_reason).to_owned())
                .collect(),
            output_tokens: completions
                .iter()
                .map(|c| c.usage.completion_tokens)
                .collect(),
        });
    }
    let result = sample::RowRecord::new(
        sample::RowContext {
            options,
            inputs,
            row,
            prompt,
            prompt_tokens,
        },
        observations,
        samples,
        hashes,
        warmup,
    )?;
    Ok(serde_json::to_value(result)?)
}
