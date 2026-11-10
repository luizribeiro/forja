mod engine;
mod options;
mod record;
mod sample;
#[cfg(test)]
mod tests;
mod timing;
mod warmup;
mod workload;

use forja_config::{BackendKind, DevConfig, Profile};
use forja_serve::{ChatService, CompletionRequest, CompletionSession};
use golden_fixtures::sha256_file;
pub(crate) use options::{Mode, Options};
use serde_json::{Value, json};
use std::{error::Error, fs, time::Instant};
use tokenizers::Tokenizer;

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
