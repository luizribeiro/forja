use std::{error::Error, fs};

use forja_config::{BackendKind, Engine, Profile};
use forja_core::Backend;
use forja_host::EngineRunner;
use golden_fixtures::sha256_file;
use tokenizers::Tokenizer;

use crate::{
    args::{Backend as BackendArg, Serve},
    engine::{engine_load_config, limits, validate_engine, weights_path},
};

pub(crate) async fn run(options: &Serve) -> Result<(), Box<dyn Error>> {
    let config = &options.config;
    let backend_arg = match config.backend.kind {
        BackendKind::Metal => BackendArg::Metal,
        BackendKind::Cpu => BackendArg::Cpu,
    };
    let (embedded, _) = validate_engine(
        &config.model.engine,
        &config.model.model,
        backend_arg,
        std::path::Path::new("target/forja-serve"),
    )?;
    let source: Profile = toml::from_str(&fs::read_to_string(&config.model.profile)?)?;
    if source != embedded {
        return Err("configured profile does not match the engine profile".into());
    }
    let load_config = engine_load_config(&config.model.engine, &Engine::default())?;
    match config.backend.kind {
        BackendKind::Cpu => {
            serve_backend(
                forja_cpu::CpuBackend::new(),
                limits(&config.limits)?,
                options,
                &embedded,
                &load_config,
            )
            .await
        }
        BackendKind::Metal => {
            #[cfg(target_os = "macos")]
            {
                let replay = config.backend.metal.resolve().graph_replay;
                let (backend, live_bytes) = crate::engine::metal_backend(
                    replay,
                    config.backend.metal.pool_capacity,
                    &config.limits,
                )?;
                let host_limits = crate::engine::limits_for_live_bytes(&config.limits, live_bytes)?;
                serve_backend(backend, host_limits, options, &embedded, &load_config).await
            }
            #[cfg(not(target_os = "macos"))]
            Err("the Metal backend requires macOS".into())
        }
    }
}

async fn serve_backend<B>(
    backend: B,
    host_limits: forja_host::Limits,
    options: &Serve,
    profile: &Profile,
    load_config: &forja_host::EngineLoadConfig,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let config = options.config.clone();
    let tokenizer = Tokenizer::from_file(config.model.model.join("tokenizer.json"))
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let chat_format = forja_serve::ChatFormat::load(
        &config.model.chat_format,
        forja_serve::ChatFormatLimits::default(),
    )?;
    let identity = forja_serve::PrefixIdentity::new([
        &sha256_file(&config.model.engine)?,
        &profile.sha256()?,
        &profile.model().weights_sha256,
        &sha256_file(config.model.model.join("tokenizer.json"))?,
        &sha256_file(&config.model.chat_format)?,
    ])?;
    let mut runner = EngineRunner::new(
        &config.model.engine,
        backend,
        host_limits,
        weights_path(&config.model.model)?,
    )
    .await?;
    let info = runner.describe().await?;
    runner
        .load_with_selections(None, load_config)
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let engine = forja_serve::ServingEngine::new(
        profile.model().id.clone(),
        runner,
        tokenizer,
        chat_format,
        config.clone(),
        info,
        profile.workload().max_context.get(),
        identity,
    )?;
    let listener = tokio::net::TcpListener::bind(config.server.listen).await?;
    let (router, drain) = forja_serve::serving_router(engine);
    let (metrics_shutdown, mut metrics_shutdown_rx) = tokio::sync::watch::channel(false);
    let metrics_task = if let Some(address) = config.metrics.listen {
        let listener = tokio::net::TcpListener::bind(address).await?;
        let router = forja_serve::operational_router(&drain, &config.metrics.path)?;
        Some(tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = metrics_shutdown_rx.changed().await;
                })
                .await
        }))
    } else {
        None
    };
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            if let Err(error) = shutdown_signal().await {
                tracing::error!(event = "shutdown.listen_failed", %error);
            }
            drain.drain().await;
        })
        .await;
    let _ = metrics_shutdown.send(true);
    if let Some(task) = metrics_task {
        task.await??;
    }
    result?;
    Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<(), Box<dyn Error>> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let interrupt_sender = sender.clone();
    tokio::spawn(async move {
        if interrupt.recv().await.is_some() {
            let _ = interrupt_sender.send(()).await;
        }
    });
    tokio::spawn(async move {
        if terminate.recv().await.is_some() {
            let _ = sender.send(()).await;
        }
    });
    receiver
        .recv()
        .await
        .ok_or_else(|| "shutdown signal listeners closed".into())
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<(), Box<dyn Error>> {
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use forja_serve::{FinishReason, GenerationEvent};

    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen3_coder_agent_uses_tool_then_answers() -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let mut engine = agent_engine(false, true).await?;
            let first: forja_serve::ChatCompletionRequest = serde_json::from_value(
                serde_json::json!({
                    "model": engine.model(),
                    "messages": [{
                        "role": "user",
                        "content": "Call read_file for src/main.rs with content source and mode text, then explain the result."
                    }],
                    "tools": [{"type": "function", "function": {
                        "name": "read_file",
                        "description": "Read one source file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "enum": ["src/main.rs"]},
                                "content": {"type": "string", "enum": ["source"]},
                                "mode": {"type": "string", "enum": ["text"]}
                            },
                            "required": ["path", "content", "mode"],
                            "additionalProperties": false
                        }
                    }}],
                    "tool_choice": {"type": "function", "function": {"name": "read_file"}},
                    "parallel_tool_calls": false,
                    "max_completion_tokens": 128,
                    "temperature": 0.0,
                    "stream": true
                }),
            )?;
            let (sender, mut receiver) = tokio::sync::mpsc::channel(1_024);
            engine.stream_chat(&first, &sender).await?;
            drop(sender);
            let mut name = None;
            let mut arguments = String::new();
            let mut finished = None;
            let mut done = false;
            while let Some(event) = receiver.recv().await {
                match event {
                    GenerationEvent::ToolCallStart(value) => name = Some(value),
                    GenerationEvent::ToolArguments(value) => arguments.push_str(&value),
                    GenerationEvent::Finished(reason, _) => finished = Some(reason),
                    GenerationEvent::Done => done = true,
                    GenerationEvent::Error(error) => return Err(error.into()),
                    GenerationEvent::Role
                    | GenerationEvent::Content(_)
                    | GenerationEvent::Usage(_) => {}
                }
            }
            if name.as_deref() != Some("read_file")
                || finished != Some(FinishReason::ToolCalls)
                || !done
            {
                return Err(format!(
                    "stream did not complete one read_file call: name={name:?}, arguments={arguments:?}, finished={finished:?}, done={done}"
                )
                .into());
            }
            let decoded: serde_json::Value = serde_json::from_str(&arguments)?;
            if decoded["path"] != "src/main.rs"
                || decoded["content"] != "source"
                || decoded["mode"] != "text"
                || arguments != r#"{"path":"src/main.rs","content":"source","mode":"text"}"#
            {
                return Err("read_file arguments do not preserve schema order".into());
            }
            let second: forja_serve::ChatCompletionRequest = serde_json::from_value(
                serde_json::json!({
                    "model": engine.model(),
                    "messages": [
                        {"role": "user", "content": "Call read_file for src/main.rs with content source and mode text, then explain the result."},
                        {"role": "assistant", "content": null, "tool_calls": [{
                            "id": "call_0", "type": "function", "function": {
                                "name": "read_file", "arguments": arguments
                            }
                        }]},
                        {"role": "tool", "tool_call_id": "call_0", "content": "fn main() { println!(\"forged\"); }"}
                    ],
                    "max_completion_tokens": 64,
                    "temperature": 0.0
                }),
            )?;
            let answer = engine.complete_chat(&second).await?;
            if answer.text.trim().is_empty()
                || !answer
                    .text
                    .chars()
                    .any(|character| character.is_ascii_alphanumeric())
            {
                return Err("agent final answer is empty".into());
            }
            Ok(())
        })
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen3_coder_reuses_repeated_agent_prefix() -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
            let model = root.join("Qwen3-Coder-30B-A3B-Instruct-4bit");
            let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
                .map_err(|error| format!("cannot load tokenizer: {error}"))?;
            let prefix = format!("{} token", " x".repeat(2_048));
            let prompt = format!("{prefix}{}", " token".repeat(32));
            let token_count = |text: String| {
                tokenizer
                    .encode(text, false)
                    .map(|encoding| encoding.len())
                    .map_err(|error| error.to_string())
            };
            if token_count(prefix.clone())? != 2_049 || token_count(prompt.clone())? != 2_081 {
                return Err("repeated-agent fixture does not have 2,049 + 32 tokens".into());
            }
            let mut engine = agent_engine(true, true).await?;
            let mut cached = Vec::new();
            for (temperature, seed) in [(0.0, 0), (0.7, 17)] {
                cached.push(
                    compare_prefix_reuse(&mut engine, &prefix, &prompt, temperature, seed).await?,
                );
            }
            drop(engine);
            let mut uncached = agent_engine(true, false).await?;
            for ((temperature, seed), cached) in [(0.0, 0), (0.7, 17)].into_iter().zip(cached) {
                let request: forja_serve::CompletionRequest =
                    serde_json::from_value(serde_json::json!({
                        "model": uncached.model(), "prompt": prompt, "max_tokens": 8,
                        "temperature": temperature, "top_k": 20, "top_p": 0.8, "seed": seed
                    }))?;
                let fresh = uncached.complete_text(&request).await?;
                if fresh.usage.cached_tokens != 0
                    || fresh.token_ids() != cached.token_ids()
                    || fresh.finish_reason != cached.finish_reason
                    || fresh.usage.completion_tokens != cached.usage.completion_tokens
                {
                    return Err("cached and uncached continuations differ".into());
                }
            }
            Ok(())
        })
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_paged_prefix_reuse_after_lru_eviction() -> Result<(), Box<dyn Error>> {
        // A one-request queue leaves the index sized by the engine's eight active lanes.
        const PREFIX_ENTRIES: u32 = 8;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let mut engine = agent_engine_with_queue(true, true, 1).await?;
            let model = engine.model().to_owned();
            let system = " x".repeat(2_040);
            let chat = |user: String| {
                serde_json::from_value::<forja_serve::ChatCompletionRequest>(serde_json::json!({
                    "model": &model, "max_completion_tokens": 1, "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": &system},
                        {"role": "user", "content": user}
                    ]
                }))
            };
            let warm = || chat(" warm".to_owned());
            let extension = || chat(" y".repeat(28));

            engine.complete_chat(&warm()?).await?;
            engine.complete_chat(&extension()?).await?;
            for index in 0..=PREFIX_ENTRIES {
                let unrelated = serde_json::from_value(serde_json::json!({
                    "model": &model, "max_completion_tokens": 1, "temperature": 0.0,
                    "messages": [{"role": "user", "content": format!(" eviction {index}")}]
                }))?;
                engine.complete_chat(&unrelated).await?;
            }
            let rewarmed = engine.complete_chat(&warm()?).await?;
            let reused = engine.complete_chat(&extension()?).await?;

            if rewarmed.usage.cached_tokens >= 2_048 || reused.usage.cached_tokens != 2_048 {
                return Err(format!(
                    "cached tokens after eviction were {}/{} instead of below 2048/2048",
                    rewarmed.usage.cached_tokens, reused.usage.cached_tokens
                )
                .into());
            }
            Ok(())
        })
    }

    #[cfg(target_os = "macos")]
    async fn compare_prefix_reuse(
        engine: &mut forja_serve::ServingEngine<forja_metal::MetalBackend>,
        prefix: &str,
        prompt: &str,
        temperature: f32,
        seed: u64,
    ) -> Result<forja_serve::Completion, Box<dyn Error>> {
        let model = engine.model().to_owned();
        let request = |prompt: &str, max_tokens| {
            serde_json::from_value::<forja_serve::CompletionRequest>(serde_json::json!({
                "model": &model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_k": 20, "top_p": 0.8, "seed": seed
            }))
        };
        engine.complete_text(&request(prefix, 1)?).await?;
        let cached = engine.complete_text(&request(prompt, 8)?).await?;
        engine.complete_text(&request(" unrelated", 1)?).await?;
        let interleaved = engine.complete_text(&request(prompt, 8)?).await?;
        if !(2_049..=2_080).contains(&cached.usage.cached_tokens)
            || interleaved.usage.cached_tokens != 2_080
        {
            return Err(format!(
                "prefix reuse reported cached tokens {}/{} outside 2049..=2080/2080",
                cached.usage.cached_tokens, interleaved.usage.cached_tokens
            )
            .into());
        }
        if cached.token_ids() != interleaved.token_ids()
            || cached.finish_reason != interleaved.finish_reason
            || cached.usage.completion_tokens != interleaved.usage.completion_tokens
        {
            return Err("interleaved cached continuations differ".into());
        }
        Ok(cached)
    }

    #[cfg(target_os = "macos")]
    async fn agent_engine(
        paged_cache: bool,
        prefix_reuse: bool,
    ) -> Result<forja_serve::ServingEngine<forja_metal::MetalBackend>, Box<dyn Error>> {
        let queue_length = forja_config::HostConfig::default().server.queue_length;
        agent_engine_with_queue(paged_cache, prefix_reuse, queue_length).await
    }

    #[cfg(target_os = "macos")]
    async fn agent_engine_with_queue(
        paged_cache: bool,
        prefix_reuse: bool,
        queue_length: u32,
    ) -> Result<forja_serve::ServingEngine<forja_metal::MetalBackend>, Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let model = root.join("Qwen3-Coder-30B-A3B-Instruct-4bit");
        let component = test_guests::qwen3_coder();
        let profile = crate::resolution::read_embedded_profile(component)?;
        let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
            .map_err(|error| format!("cannot load tokenizer: {error}"))?;
        let chat_format = forja_serve::ChatFormat::load(
            test_guests::qwen3_coder_chat_format(),
            forja_serve::ChatFormatLimits::default(),
        )?;
        let limits = forja_config::Limits {
            live_bytes: forja_config::Choice::Fixed(forja_config::ByteSize::new(
                64 * 1024 * 1024 * 1024,
            )),
            ..forja_config::Limits::default()
        };
        let mut runner = EngineRunner::new(
            component,
            forja_metal::MetalBackend::new()?,
            crate::engine::limits(&limits)?,
            crate::engine::weights_path(&model)?,
        )
        .await?;
        let info = runner.describe().await?;
        let mut load = engine_load_config(component, &Engine::default())?;
        load.paged_cache = paged_cache;
        runner.load_with_selections(None, &load).await??;
        let mut config = forja_config::HostConfig::default();
        config.server.queue_length = queue_length;
        config.request.max_output_tokens = 256;
        config.serving.prefix_reuse = prefix_reuse;
        let identity = forja_serve::PrefixIdentity::new([
            &sha256_file(component)?,
            &profile.sha256()?,
            &profile.model().weights_sha256,
            &sha256_file(model.join("tokenizer.json"))?,
            &sha256_file(test_guests::qwen3_coder_chat_format())?,
        ])?;
        Ok(forja_serve::ServingEngine::new(
            profile.model().id.clone(),
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            profile.workload().max_context.get(),
            identity,
        )?)
    }
}
