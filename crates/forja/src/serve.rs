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
