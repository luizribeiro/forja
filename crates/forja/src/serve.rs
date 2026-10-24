use std::{error::Error, fs, future::Future, path::Path, pin::Pin};

use forja_config::{BackendKind, Choice, Engine, Profile};
use forja_core::Backend;
use forja_host::EngineRunner;
use golden_fixtures::sha256_file;
use tokenizers::Tokenizer;

use crate::{
    args::{Backend as BackendArg, Serve},
    engine::{engine_load_config, limits, validate_engine, weights_path},
    supervisor::Supervisor,
};

pub(crate) struct Loaded<S> {
    pub(crate) service: S,
    pub(crate) retained_bytes: u64,
}

pub(crate) type LoadFuture<'a, S> = Pin<Box<dyn Future<Output = Result<Loaded<S>, String>> + 'a>>;

pub(crate) trait RuntimeFactory {
    type Service: forja_serve::ChatService;

    fn plan(&self, config: &forja_config::HostConfig) -> Result<forja_serve::LoadPlan, String>;

    fn load<'a>(
        &'a mut self,
        config: &'a forja_config::HostConfig,
    ) -> LoadFuture<'a, Self::Service>;
}

pub(crate) async fn run(options: &Serve) -> Result<(), Box<dyn Error>> {
    match options.config.backend.kind {
        BackendKind::Cpu => run_factory(CpuFactory, options).await,
        BackendKind::Metal => {
            #[cfg(target_os = "macos")]
            {
                run_factory(MetalFactory, options).await
            }
            #[cfg(not(target_os = "macos"))]
            Err("the Metal backend requires macOS".into())
        }
    }
}

struct CpuFactory;

impl RuntimeFactory for CpuFactory {
    type Service = forja_serve::ServingEngine<forja_cpu::CpuBackend>;

    fn plan(&self, config: &forja_config::HostConfig) -> Result<forja_serve::LoadPlan, String> {
        validate_config(config, BackendArg::Cpu)?;
        let retained_bytes = match config.limits.live_bytes {
            Choice::Auto => 48 * 1024 * 1024 * 1024,
            Choice::Fixed(bytes) => bytes.get(),
        };
        Ok(forja_serve::LoadPlan { retained_bytes })
    }

    fn load<'a>(
        &'a mut self,
        config: &'a forja_config::HostConfig,
    ) -> LoadFuture<'a, Self::Service> {
        Box::pin(async move {
            let host_limits = page_pool_limit(
                limits(&config.limits).map_err(|error| error.to_string())?,
                config,
            );
            load_engine(forja_cpu::CpuBackend::new(), host_limits, config).await
        })
    }
}

#[cfg(target_os = "macos")]
struct MetalFactory;

#[cfg(target_os = "macos")]
impl RuntimeFactory for MetalFactory {
    type Service = forja_serve::ServingEngine<forja_metal::MetalBackend>;

    fn plan(&self, config: &forja_config::HostConfig) -> Result<forja_serve::LoadPlan, String> {
        validate_config(config, BackendArg::Metal)?;
        let working_set = forja_metal::MetalBackend::recommended_max_working_set_size()
            .map_err(|error| error.to_string())?;
        let physical =
            forja_metal::MetalBackend::physical_memory_size().map_err(|error| error.to_string())?;
        let retained_bytes = config
            .limits
            .resolve_live_bytes(working_set, physical)
            .map_err(|error| error.to_string())?
            .get();
        Ok(forja_serve::LoadPlan { retained_bytes })
    }

    fn load<'a>(
        &'a mut self,
        config: &'a forja_config::HostConfig,
    ) -> LoadFuture<'a, Self::Service> {
        Box::pin(async move {
            let replay = config.backend.metal.resolve().graph_replay;
            let (backend, live_bytes) = crate::engine::metal_backend(
                replay,
                config.backend.metal.pool_capacity,
                &config.limits,
            )
            .map_err(|error| error.to_string())?;
            let host_limits = page_pool_limit(
                crate::engine::limits_for_live_bytes(&config.limits, live_bytes)
                    .map_err(|error| error.to_string())?,
                config,
            );
            load_engine(backend, host_limits, config).await
        })
    }
}

fn validate_config(
    config: &forja_config::HostConfig,
    backend: BackendArg,
) -> Result<Profile, String> {
    let (embedded, _) = validate_engine(
        &config.model.engine,
        &config.model.model,
        backend,
        Path::new("target/forja-serve"),
    )
    .map_err(|error| error.to_string())?;
    let source = fs::read_to_string(&config.model.profile)
        .map_err(|error| format!("cannot read configured profile: {error}"))?;
    let source: Profile = toml::from_str(&source).map_err(|error| error.to_string())?;
    if source != embedded {
        return Err("configured profile does not match the engine profile".to_owned());
    }
    Ok(embedded)
}

fn page_pool_limit(
    limits: forja_host::Limits,
    config: &forja_config::HostConfig,
) -> forja_host::Limits {
    match config.serving.page_pool_bytes {
        Choice::Auto => limits,
        Choice::Fixed(bytes) => limits.with_page_pool_limit(bytes.get()),
    }
}

async fn load_engine<B>(
    backend: B,
    host_limits: forja_host::Limits,
    config: &forja_config::HostConfig,
) -> Result<Loaded<forja_serve::ServingEngine<B>>, String>
where
    B: Backend + Send + Sync + 'static,
{
    let profile = validate_config(
        config,
        match config.backend.kind {
            BackendKind::Metal => BackendArg::Metal,
            BackendKind::Cpu => BackendArg::Cpu,
        },
    )?;
    let load_config = engine_load_config(&config.model.engine, &Engine::default())?;
    let tokenizer = Tokenizer::from_file(config.model.model.join("tokenizer.json"))
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let chat_format = forja_serve::ChatFormat::load(
        &config.model.chat_format,
        forja_serve::ChatFormatLimits::default(),
    )
    .map_err(|error| error.to_string())?;
    let identity = forja_serve::PrefixIdentity::new([
        &sha256_file(&config.model.engine).map_err(|error| error.to_string())?,
        &profile.sha256().map_err(|error| error.to_string())?,
        &profile.model().weights_sha256,
        &sha256_file(config.model.model.join("tokenizer.json"))
            .map_err(|error| error.to_string())?,
        &sha256_file(&config.model.chat_format).map_err(|error| error.to_string())?,
    ])
    .map_err(str::to_owned)?;
    let mut runner = EngineRunner::new(
        &config.model.engine,
        backend,
        host_limits,
        weights_path(&config.model.model)?,
    )
    .await
    .map_err(|error| error.to_string())?;
    let info = runner.describe().await.map_err(|error| error.to_string())?;
    runner
        .load_with_selections(None, &load_config)
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let retained_bytes = runner
        .memory_usage()
        .accounted_bytes()
        .ok_or_else(|| "retained runtime charge overflowed".to_owned())?;
    let engine = forja_serve::ServingEngine::new(
        profile.model().id.clone(),
        runner,
        tokenizer,
        chat_format,
        config.clone(),
        info,
        profile.workload().max_context.get(),
        profile
            .serving()
            .map_or(1, |serving| serving.prompt_token_budget.get()),
        identity,
    )
    .map_err(|error| error.to_string())?;
    Ok(Loaded {
        service: engine,
        retained_bytes,
    })
}

async fn run_factory<F>(mut factory: F, options: &Serve) -> Result<(), Box<dyn Error>>
where
    F: RuntimeFactory + 'static,
{
    let config = options.config.clone();
    let plan = factory.plan(&config)?;
    let loaded = factory.load(&config).await?;
    let retained_bytes = loaded.retained_bytes;
    let listener = tokio::net::TcpListener::bind(config.server.listen).await?;
    let (router, drain) = forja_serve::serving_router(loaded.service);
    let mut supervisor = Supervisor::new(
        options.sources.clone(),
        config.clone(),
        retained_bytes,
        plan.retained_bytes,
        factory,
        drain.clone(),
    )?;
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
    let (shutdown, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.changed().await;
            })
            .await
    });
    if let Err(error) = crate::signals::shutdown().await {
        tracing::error!(event = "shutdown.listen_failed", %error);
    }
    let shutdown_result = supervisor.shutdown().await;
    let _ = shutdown.send(true);
    let result = server.await?;
    shutdown_result?;
    let _ = metrics_shutdown.send(true);
    if let Some(task) = metrics_task {
        task.await??;
    }
    result?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        io::{Read, Write},
        net::{SocketAddr, TcpStream},
        path::Path,
        path::PathBuf,
        thread,
        time::Duration,
    };

    use forja_serve::{FinishReason, GenerationEvent};

    use super::*;

    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    async fn metal_concurrent_mixed_requests_match_serial() -> Result<(), Box<dyn Error>> {
        let engine = agent_engine(true, true).await?;
        let model = engine.model().to_owned();
        let mut groups = Vec::new();
        for (temperature, seed) in [(0.0, 0), (0.7, 17)] {
            for specs in [
                vec![(" x".to_owned(), 8, 0), (" y".repeat(7), 2, 30)],
                vec![
                    (" a".to_owned(), 8, 0),
                    (" b".repeat(7), 2, 20),
                    (" c".repeat(33), 3, 35),
                    (" d".repeat(7), 2, 50),
                ],
            ] {
                groups.push(
                    specs
                        .into_iter()
                        .map(|(prompt, max_tokens, delay_ms)| {
                            completion_request(&model, &prompt, max_tokens, temperature, seed)
                                .map(|request| (request, Duration::from_millis(delay_ms)))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let router = forja_serve::serving_router(engine).0;
        let server = tokio::spawn(async move { axum::serve(listener, router).await });
        for (temperature, seed) in [(0.0, 0), (0.7, 17)] {
            let request =
                completion_request(&model, &" repeated".repeat(33), 16, temperature, seed)?;
            let reference = {
                let request = request.clone();
                tokio::task::spawn_blocking(move || http_completion(address, &request))
                    .await?
                    .map_err(|error| format!("repeated reference {temperature}/{seed}: {error}"))?
            };
            let clients = (0..4)
                .map(|_| {
                    let request = request.clone();
                    tokio::task::spawn_blocking(move || http_completion(address, &request))
                })
                .collect::<Vec<_>>();
            for (index, client) in clients.into_iter().enumerate() {
                assert_eq!(
                    client.await?.map_err(|error| format!(
                        "repeated client {temperature}/{seed}/{index}: {error}"
                    ))?,
                    reference
                );
            }
        }
        let mut expected = Vec::with_capacity(groups.len());
        for (group_index, group) in groups.iter().enumerate() {
            let mut outputs = Vec::with_capacity(group.len());
            for (request_index, (request, _)) in group.iter().enumerate() {
                let request = request.clone();
                outputs.push(
                    tokio::task::spawn_blocking(move || http_completion(address, &request))
                        .await?
                        .map_err(|error| {
                            format!("serial group {group_index}/{request_index}: {error}")
                        })?,
                );
            }
            expected.push(outputs);
        }
        let mut actual = Vec::with_capacity(groups.len());
        for (group_index, group) in groups.into_iter().enumerate() {
            let clients = group
                .into_iter()
                .map(|(request, delay)| {
                    tokio::task::spawn_blocking(move || {
                        thread::sleep(delay);
                        http_completion(address, &request)
                    })
                })
                .collect::<Vec<_>>();
            let mut outputs = Vec::with_capacity(clients.len());
            for (request_index, client) in clients.into_iter().enumerate() {
                outputs.push(client.await?.map_err(|error| {
                    format!("concurrent group {group_index}/{request_index}: {error}")
                })?);
            }
            actual.push(outputs);
        }
        server.abort();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn completion_request(
        model: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: u64,
    ) -> Result<forja_serve::CompletionRequest, serde_json::Error> {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 20,
            "top_p": 0.9,
            "seed": seed
        }))
    }

    #[cfg(target_os = "macos")]
    fn http_completion(
        address: SocketAddr,
        request: &forja_serve::CompletionRequest,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": request.model,
            "prompt": request.prompt,
            "max_tokens": request.params.max_tokens,
            "temperature": request.params.temperature,
            "top_k": request.params.top_k,
            "top_p": request.params.top_p,
            "seed": request.params.seed
        }))?;
        let mut stream = TcpStream::connect(address)?;
        write!(
            stream,
            "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )?;
        stream.write_all(&body)?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response)?;
        let split = response
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .ok_or("HTTP response has no body")?;
        let value: serde_json::Value = serde_json::from_slice(&response[split + 4..])?;
        value["choices"][0]["text"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("completion response has no text: {value}").into())
    }

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
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen3_paged_prefix_reuse_after_lru_eviction() -> Result<(), Box<dyn Error>> {
        paged_completion_prefix_reuse(test_guests::qwen3(), "Qwen3-0.6B")
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_olmoe_paged_prefix_reuse_after_lru_eviction() -> Result<(), Box<dyn Error>> {
        paged_completion_prefix_reuse(test_guests::olmoe(), "OLMoE-1B-7B-0924")
    }

    #[cfg(target_os = "macos")]
    fn paged_completion_prefix_reuse(
        component: &Path,
        model_directory: &str,
    ) -> Result<(), Box<dyn Error>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        runtime.block_on(async {
            let prefix = format!("{} token", " x".repeat(128));
            let extension = format!("{prefix}{}", " token".repeat(32));
            let root = PathBuf::from(
                env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?,
            );
            let tokenizer = Tokenizer::from_file(root.join(model_directory).join("tokenizer.json"))
                .map_err(|error| format!("cannot load tokenizer: {error}"))?;
            let cached_tokens = u32::try_from(
                tokenizer
                    .encode(prefix.as_str(), false)
                    .map_err(|error| error.to_string())?
                    .len(),
            )?;
            let mut engine = completion_engine(component, model_directory).await?;
            let model = engine.model().to_owned();
            let request = |prompt: &str| {
                serde_json::from_value::<forja_serve::CompletionRequest>(serde_json::json!({
                    "model": &model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0
                }))
            };

            engine.complete_text(&request(&prefix)?).await?;
            engine.complete_text(&request(&extension)?).await?;
            for index in 0..=1 {
                engine
                    .complete_text(&request(&format!(" eviction {index}"))?)
                    .await?;
            }
            let rewarmed = engine.complete_text(&request(&prefix)?).await?;
            let reused = engine.complete_text(&request(&extension)?).await?;
            if rewarmed.usage.cached_tokens >= cached_tokens
                || reused.usage.cached_tokens != cached_tokens
            {
                return Err(format!(
                    "cached tokens after eviction were {}/{} instead of below {cached_tokens}/{cached_tokens}",
                    rewarmed.usage.cached_tokens, reused.usage.cached_tokens
                )
                .into());
            }
            Ok(())
        })
    }

    #[cfg(target_os = "macos")]
    async fn completion_engine(
        component: &Path,
        model_directory: &str,
    ) -> Result<forja_serve::ServingEngine<forja_metal::MetalBackend>, Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let model = root.join(model_directory);
        let profile = crate::resolution::read_embedded_profile(component)?;
        let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
            .map_err(|error| format!("cannot load tokenizer: {error}"))?;
        let chat_component = test_guests::chat_format_smoke();
        let chat_format = forja_serve::ChatFormat::load(
            chat_component,
            forja_serve::ChatFormatLimits::default(),
        )?;
        let mut runner = EngineRunner::new(
            component,
            forja_metal::MetalBackend::new()?,
            crate::engine::limits(&forja_config::Limits::default())?,
            crate::engine::weights_path(&model)?,
        )
        .await?;
        let info = runner.describe().await?;
        let mut load = engine_load_config(component, &Engine::default())?;
        load.paged_cache = true;
        runner.load_with_selections(None, &load).await??;
        let mut config = forja_config::HostConfig::default();
        config.server.queue_length = 1;
        config.serving.prefix_reuse = true;
        let identity = forja_serve::PrefixIdentity::new([
            &sha256_file(component)?,
            &profile.sha256()?,
            &profile.model().weights_sha256,
            &sha256_file(model.join("tokenizer.json"))?,
            &sha256_file(chat_component)?,
        ])?;
        Ok(forja_serve::ServingEngine::new(
            profile.model().id.clone(),
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            profile.workload().max_context.get(),
            profile
                .serving()
                .map_or(1, |serving| serving.prompt_token_budget.get()),
            identity,
        )?)
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
            profile
                .serving()
                .map_or(1, |serving| serving.prompt_token_budget.get()),
            identity,
        )?)
    }
}
