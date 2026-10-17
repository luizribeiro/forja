use std::{error::Error, fs};

use forja_config::{BackendKind, Engine, Profile};
use forja_core::Backend;
use forja_host::EngineRunner;
use tokenizers::Tokenizer;

use crate::{
    args::{Backend as BackendArg, Serve},
    engine::{engine_load_config, limits, validate_engine, weights_path},
};

#[cfg(target_os = "macos")]
use crate::engine::metal_graph_replay;

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
                let backend =
                    forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(replay))?;
                serve_backend(backend, options, &embedded, &load_config).await
            }
            #[cfg(not(target_os = "macos"))]
            Err("the Metal backend requires macOS".into())
        }
    }
}

async fn serve_backend<B>(
    backend: B,
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
    let mut runner = EngineRunner::new(
        &config.model.engine,
        backend,
        limits(&config.limits)?,
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
    )?;
    let listener = tokio::net::TcpListener::bind(config.server.listen).await?;
    let (router, drain) = forja_serve::serving_router(engine);
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            if let Err(error) = shutdown_signal().await {
                eprintln!("cannot listen for shutdown: {error}");
            }
            drain.drain().await;
        })
        .await?;
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
            let mut engine = agent_engine().await?;
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
    async fn agent_engine()
    -> Result<forja_serve::ServingEngine<forja_metal::MetalBackend>, Box<dyn Error>> {
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
            live_bytes: forja_config::ByteSize::new(64 * 1024 * 1024 * 1024),
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
        let load = engine_load_config(component, &Engine::default())?;
        runner.load_with_selections(None, &load).await??;
        let mut config = forja_config::HostConfig::default();
        config.request.max_output_tokens = 256;
        Ok(forja_serve::ServingEngine::new(
            profile.model().id.clone(),
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            profile.workload().max_context.get(),
        )?)
    }
}
