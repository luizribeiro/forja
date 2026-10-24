use std::sync::Arc;

use forja_config::{HostConfig, HostConfigDiff};
use forja_serve::{LoadPlan, ProcessLedger};

use crate::{
    args::{HostConfigSources, load_host_config},
    serve::{Loaded, RuntimeFactory},
};

#[derive(Debug)]
pub(crate) struct Candidate {
    pub(crate) generation: u64,
    pub(crate) config: Arc<HostConfig>,
    pub(crate) diff: HostConfigDiff,
    pub(crate) plan: LoadPlan,
}

impl Candidate {
    pub(crate) fn from_sources(
        generation: u64,
        active: &HostConfig,
        sources: &HostConfigSources,
        plan: impl FnOnce(&HostConfig) -> Result<LoadPlan, String>,
    ) -> Result<Self, String> {
        let config = load_host_config(sources).map_err(|error| error.to_string())?;
        let diff = HostConfigDiff::between(active, &config)?;
        if !diff.restart.is_empty() {
            return Err(format!(
                "reload changes restart-only fields: {}",
                diff.restart.join(", ")
            ));
        }
        let plan = plan(&config)?;
        Ok(Self {
            generation,
            config: Arc::new(config),
            diff,
            plan,
        })
    }
}

pub(crate) struct Supervisor<F: RuntimeFactory> {
    sources: HostConfigSources,
    active: Arc<HostConfig>,
    generation: u64,
    runtime_id: u64,
    next_runtime_id: u64,
    retained_bytes: Option<u64>,
    ledger: ProcessLedger,
    factory: F,
    logging: Box<dyn crate::logging::Reload>,
    runtime: forja_serve::DrainHandle<F::Service>,
}

impl<F: RuntimeFactory> Supervisor<F> {
    pub(crate) fn new(
        sources: HostConfigSources,
        config: HostConfig,
        retained_bytes: u64,
        quota: u64,
        factory: F,
        logging: Box<dyn crate::logging::Reload>,
        runtime: forja_serve::DrainHandle<F::Service>,
    ) -> Result<Self, String> {
        let mut ledger = ProcessLedger::new(quota);
        ledger.reserve(retained_bytes)?;
        Ok(Self {
            sources,
            active: Arc::new(config),
            generation: 0,
            runtime_id: 0,
            next_runtime_id: 1,
            retained_bytes: Some(retained_bytes),
            ledger,
            factory,
            logging,
            runtime,
        })
    }

    pub(crate) async fn reload(&mut self) -> Result<(), String> {
        let result = self.reload_candidate().await;
        if let Err(error) = &result {
            self.runtime.note_reload_failure();
            tracing::error!(
                event = "config.reload_failed",
                config_generation = self.generation,
                %error
            );
        } else {
            tracing::info!(
                event = "config.reloaded",
                config_generation = self.generation
            );
        }
        result
    }

    pub(crate) fn header_read_timeout(&self) -> std::time::Duration {
        self.active.server.header_read_timeout.get()
    }

    async fn reload_candidate(&mut self) -> Result<(), String> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| "configuration generation overflowed".to_owned())?;
        let candidate =
            Candidate::from_sources(generation, &self.active, &self.sources, |config| {
                self.factory.plan(config)
            })?;
        if candidate.diff.is_empty() {
            return Ok(());
        }
        if candidate.diff.model.is_empty() {
            self.apply_live_policy(&candidate).await?;
            self.active = candidate.config;
            self.generation = generation;
            return Ok(());
        }
        self.replace(candidate).await
    }

    async fn apply_live_policy(&mut self, candidate: &Candidate) -> Result<(), String> {
        let previous = self
            .runtime
            .acknowledge_policy(candidate.generation, (*candidate.config).clone())
            .await
            .map_err(|error| error.to_string())?;
        if let Err(error) = self.logging.reconfigure(&candidate.config.logging) {
            self.runtime
                .acknowledge_policy(self.generation, (*self.active).clone())
                .await
                .map_err(|rollback| format!("{error}; scheduler rollback failed: {rollback}"))?;
            return Err(error);
        }
        if let Err(error) =
            self.runtime
                .publish_policy(previous, candidate.generation, (*candidate.config).clone())
        {
            let logging = self.logging.reconfigure(&self.active.logging);
            let scheduler = self
                .runtime
                .acknowledge_policy(self.generation, (*self.active).clone())
                .await;
            return Err(format!(
                "policy publication failed: {error}; logging rollback: {logging:?}; scheduler rollback: {scheduler:?}"
            ));
        }
        Ok(())
    }

    pub(crate) async fn shutdown(&mut self) -> Result<(), String> {
        self.runtime.close_admission();
        tracing::info!(
            event = "model.drain_started",
            config_generation = self.generation,
            shutdown = true
        );
        self.runtime
            .try_drain_generation(
                self.runtime_id,
                self.active.server.model_drain_deadline.get(),
            )
            .await
            .map_err(|error| error.to_string())?;
        tracing::info!(
            event = "model.drain_finished",
            config_generation = self.generation,
            shutdown = true
        );
        self.runtime.unload().await?;
        if let Some(retained_bytes) = self.retained_bytes.take() {
            self.ledger.release(retained_bytes)?;
        }
        Ok(())
    }

    async fn replace(&mut self, candidate: Candidate) -> Result<(), String> {
        let preloaded = match candidate.plan.retained_bytes {
            Some(bytes) if self.ledger.fits(bytes) => self.preload(&candidate).await,
            _ => None,
        };
        self.runtime.set_ready(false);
        tracing::info!(
            event = "model.drain_started",
            config_generation = self.generation,
            replacement_generation = candidate.generation
        );
        self.runtime
            .try_drain_generation(
                self.runtime_id,
                self.active.server.model_drain_deadline.get(),
            )
            .await
            .map_err(|error| error.to_string())?;
        tracing::info!(
            event = "model.drain_finished",
            config_generation = self.generation,
            replacement_generation = candidate.generation
        );
        self.runtime.unload().await?;
        if let Some(retained_bytes) = self.retained_bytes.take() {
            self.ledger.release(retained_bytes)?;
        }
        let loaded = match preloaded {
            Some(loaded) => loaded,
            None => match self.factory.load(&candidate.config).await {
                Ok(loaded) => {
                    self.ledger.reserve(loaded.retained_bytes)?;
                    loaded
                }
                Err(load_error) => return self.rollback(load_error).await,
            },
        };
        let runtime_id = self.allocate_runtime_id()?;
        self.runtime
            .publish_runtime(runtime_id, candidate.generation, loaded.service)?;
        self.runtime.set_ready(true);
        self.runtime_id = runtime_id;
        self.retained_bytes = Some(loaded.retained_bytes);
        self.active = candidate.config;
        self.generation = candidate.generation;
        tracing::info!(
            event = "model.published",
            config_generation = self.generation
        );
        Ok(())
    }

    async fn preload(&mut self, candidate: &Candidate) -> Option<Loaded<F::Service>> {
        let loaded = self.factory.load(&candidate.config).await.ok()?;
        if candidate.plan.retained_bytes != Some(loaded.retained_bytes) {
            return None;
        }
        self.ledger.reserve(loaded.retained_bytes).ok()?;
        Some(loaded)
    }

    async fn rollback(&mut self, load_error: String) -> Result<(), String> {
        match self.factory.load(&self.active).await {
            Ok(loaded) => {
                self.ledger.reserve(loaded.retained_bytes)?;
                let runtime_id = self.allocate_runtime_id()?;
                self.runtime
                    .publish_runtime(runtime_id, self.generation, loaded.service)?;
                self.runtime.set_ready(true);
                self.runtime_id = runtime_id;
                self.retained_bytes = Some(loaded.retained_bytes);
                Err(format!(
                    "candidate load failed and the active generation was restored: {load_error}"
                ))
            }
            Err(rollback_error) => Err(format!(
                "candidate load failed: {load_error}; rollback failed: {rollback_error}"
            )),
        }
    }

    fn allocate_runtime_id(&mut self) -> Result<u64, String> {
        let runtime_id = self.next_runtime_id;
        self.next_runtime_id = runtime_id
            .checked_add(1)
            .ok_or_else(|| "runtime identity overflowed".to_owned())?;
        Ok(runtime_id)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        fs,
        future::Future,
        pin::Pin,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use forja_serve::{
        ChatCompletionRequest, ChatService, Completion, CompletionRequest, GenerationControl,
        GenerationError, GenerationEvent,
    };
    use forja_testing::temporary_directory;

    use super::*;

    struct FakeService;

    impl ChatService for FakeService {
        fn model(&self) -> &'static str {
            "fake"
        }
        fn max_body_bytes(&self) -> usize {
            1
        }
        fn queue_length(&self) -> u32 {
            1
        }
        fn request_deadline(&self) -> Duration {
            Duration::from_secs(1)
        }
        fn write_stall_timeout(&self) -> Duration {
            Duration::from_secs(1)
        }
        fn complete(
            &mut self,
            _: ChatCompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(std::future::pending())
        }
        fn complete_text(
            &mut self,
            _: CompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(std::future::pending())
        }
        fn stream_chat(
            &mut self,
            _: ChatCompletionRequest,
            _: tokio::sync::mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(std::future::pending())
        }
        fn stream_text(
            &mut self,
            _: CompletionRequest,
            _: tokio::sync::mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(std::future::pending())
        }
    }

    struct FakeFactory(VecDeque<Result<u64, String>>);

    struct ObservedFactory {
        planned_bytes: Option<u64>,
        runtime: forja_serve::DrainHandle<FakeService>,
        readiness: Arc<Mutex<Vec<bool>>>,
    }

    struct IdentityFactory {
        next_id: u64,
        quiesced: Arc<Mutex<Vec<u64>>>,
    }

    struct IdentityService {
        id: u64,
        quiesced: Arc<Mutex<Vec<u64>>>,
    }

    struct FakeLogging(bool);

    impl crate::logging::Reload for FakeLogging {
        fn reconfigure(&self, _: &forja_config::Logging) -> Result<(), String> {
            if self.0 {
                Err("logger rejected policy".to_owned())
            } else {
                Ok(())
            }
        }
    }

    impl RuntimeFactory for FakeFactory {
        type Service = FakeService;

        fn quota(&self, _: &HostConfig) -> Result<u64, String> {
            Ok(8)
        }

        fn plan(&mut self, _: &HostConfig) -> Result<LoadPlan, String> {
            Ok(LoadPlan {
                retained_bytes: Some(8),
            })
        }

        fn load<'a>(&'a mut self, _: &'a HostConfig) -> crate::serve::LoadFuture<'a, FakeService> {
            let outcome = self.0.pop_front().unwrap();
            Box::pin(async move {
                outcome.map(|retained_bytes| Loaded {
                    service: FakeService,
                    retained_bytes,
                })
            })
        }
    }

    impl RuntimeFactory for ObservedFactory {
        type Service = FakeService;

        fn quota(&self, _: &HostConfig) -> Result<u64, String> {
            Ok(8)
        }

        fn plan(&mut self, _: &HostConfig) -> Result<LoadPlan, String> {
            Ok(LoadPlan {
                retained_bytes: self.planned_bytes,
            })
        }

        fn load<'a>(&'a mut self, _: &'a HostConfig) -> crate::serve::LoadFuture<'a, FakeService> {
            self.readiness.lock().unwrap().push(self.runtime.is_ready());
            Box::pin(async {
                Ok(Loaded {
                    service: FakeService,
                    retained_bytes: 4,
                })
            })
        }
    }

    impl ChatService for IdentityService {
        fn model(&self) -> &'static str {
            "fake"
        }

        fn max_body_bytes(&self) -> usize {
            1
        }

        fn queue_length(&self) -> u32 {
            1
        }

        fn request_deadline(&self) -> Duration {
            Duration::from_secs(1)
        }

        fn write_stall_timeout(&self) -> Duration {
            Duration::from_secs(1)
        }

        fn complete(
            &mut self,
            _: ChatCompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(std::future::pending())
        }

        fn complete_text(
            &mut self,
            _: CompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(std::future::pending())
        }

        fn stream_chat(
            &mut self,
            _: ChatCompletionRequest,
            _: tokio::sync::mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(std::future::pending())
        }

        fn stream_text(
            &mut self,
            _: CompletionRequest,
            _: tokio::sync::mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(std::future::pending())
        }

        fn quiesce(
            &mut self,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            self.quiesced.lock().unwrap().push(self.id);
            Box::pin(async { Ok(()) })
        }
    }

    impl RuntimeFactory for IdentityFactory {
        type Service = IdentityService;

        fn quota(&self, _: &HostConfig) -> Result<u64, String> {
            Ok(8)
        }

        fn plan(&mut self, _: &HostConfig) -> Result<LoadPlan, String> {
            Ok(LoadPlan {
                retained_bytes: Some(8),
            })
        }

        fn load<'a>(
            &'a mut self,
            _: &'a HostConfig,
        ) -> crate::serve::LoadFuture<'a, IdentityService> {
            let id = self.next_id;
            self.next_id += 1;
            let quiesced = Arc::clone(&self.quiesced);
            Box::pin(async move {
                Ok(Loaded {
                    service: IdentityService { id, quiesced },
                    retained_bytes: 4,
                })
            })
        }
    }

    #[test]
    fn candidate_rereads_sources_and_refuses_restart_changes() {
        let root = temporary_directory("supervisor-candidate").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[server]\nqueue_length=7\n").unwrap();
        let sources = HostConfigSources {
            files: vec![path.clone()],
            sets: Vec::new(),
        };
        let active = HostConfig::default();
        let plan = |_: &HostConfig| {
            Ok(LoadPlan {
                retained_bytes: Some(7),
            })
        };
        let candidate = Candidate::from_sources(1, &active, &sources, plan).unwrap();
        assert_eq!(candidate.diff.live, ["server.queue_length"]);
        assert_eq!(candidate.plan.retained_bytes, Some(7));
        fs::write(path, "[server]\nlisten='127.0.0.1:9000'\n").unwrap();
        assert!(
            Candidate::from_sources(2, &active, &sources, plan)
                .unwrap_err()
                .contains("server.listen")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn failed_candidate_load_restores_the_active_generation() {
        let root = temporary_directory("supervisor-rollback").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[model]\nengine='candidate'\n").unwrap();
        let sources = HostConfigSources {
            files: vec![path],
            sets: Vec::new(),
        };
        let (_, runtime) = forja_serve::serving_router(FakeService);
        let readiness = runtime.clone();
        let factory = FakeFactory(VecDeque::from([Err("candidate failed".to_owned()), Ok(4)]));
        let mut supervisor = Supervisor::new(
            sources,
            HostConfig::default(),
            4,
            8,
            factory,
            Box::new(FakeLogging(false)),
            runtime,
        )
        .unwrap();
        let error = supervisor.reload().await.unwrap_err();
        assert!(error.contains("active generation was restored"));
        assert_eq!(supervisor.generation, 0);
        assert!(readiness.is_ready());
        supervisor.shutdown().await.unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn double_load_failure_still_allows_shutdown_drain() {
        let root = temporary_directory("supervisor-double-failure").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[model]\nengine='candidate'\n").unwrap();
        let sources = HostConfigSources {
            files: vec![path],
            sets: Vec::new(),
        };
        let (_, runtime) = forja_serve::serving_router(FakeService);
        let readiness = runtime.clone();
        let factory = FakeFactory(VecDeque::from([
            Err("candidate failed".to_owned()),
            Err("rollback failed".to_owned()),
        ]));
        let mut supervisor = Supervisor::new(
            sources,
            HostConfig::default(),
            4,
            8,
            factory,
            Box::new(FakeLogging(false)),
            runtime,
        )
        .unwrap();
        let error = supervisor.reload().await.unwrap_err();
        assert!(error.contains("candidate load failed: candidate failed"));
        assert!(error.contains("rollback failed: rollback failed"));
        assert!(!readiness.is_ready());
        supervisor.shutdown().await.unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exact_preload_charge_preserves_readiness_and_fallback_drains_first() {
        let root = temporary_directory("supervisor-load-paths").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[model]\nengine='candidate'\n").unwrap();
        let readiness = Arc::new(Mutex::new(Vec::new()));
        for planned_bytes in [Some(4), None, Some(8)] {
            let sources = HostConfigSources {
                files: vec![path.clone()],
                sets: Vec::new(),
            };
            let (_, runtime) = forja_serve::serving_router(FakeService);
            let factory = ObservedFactory {
                planned_bytes,
                runtime: runtime.clone(),
                readiness: Arc::clone(&readiness),
            };
            let mut supervisor = Supervisor::new(
                sources,
                HostConfig::default(),
                4,
                8,
                factory,
                Box::new(FakeLogging(false)),
                runtime,
            )
            .unwrap();
            supervisor.reload().await.unwrap();
            supervisor.shutdown().await.unwrap();
        }
        assert_eq!(*readiness.lock().unwrap(), [true, false, false]);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn policy_generation_does_not_change_the_runtime_drain_identity() {
        let root = temporary_directory("supervisor-runtime-identity").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[server]\nqueue_length=7\n").unwrap();
        let sources = HostConfigSources {
            files: vec![path.clone()],
            sets: Vec::new(),
        };
        let quiesced = Arc::new(Mutex::new(Vec::new()));
        let initial = IdentityService {
            id: 0,
            quiesced: Arc::clone(&quiesced),
        };
        let (_, runtime) = forja_serve::serving_router(initial);
        let factory = IdentityFactory {
            next_id: 1,
            quiesced: Arc::clone(&quiesced),
        };
        let mut supervisor = Supervisor::new(
            sources,
            HostConfig::default(),
            4,
            8,
            factory,
            Box::new(FakeLogging(false)),
            runtime,
        )
        .unwrap();

        supervisor.reload().await.unwrap();
        assert_eq!(supervisor.generation, 1);
        assert_eq!(supervisor.runtime_id, 0);

        fs::write(&path, "[model]\nengine='candidate'\n").unwrap();
        supervisor.reload().await.unwrap();
        assert_eq!(supervisor.generation, 2);
        assert_eq!(supervisor.runtime_id, 1);
        assert_eq!(*quiesced.lock().unwrap(), [0]);

        supervisor.shutdown().await.unwrap();
        assert_eq!(*quiesced.lock().unwrap(), [0, 1]);
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn logging_failure_rolls_back_scheduler_policy() {
        let root = temporary_directory("supervisor-logging-rollback").unwrap();
        let path = root.join("host.toml");
        fs::write(&path, "[logging]\nlevel='debug'\n").unwrap();
        let sources = HostConfigSources {
            files: vec![path],
            sets: Vec::new(),
        };
        let (_, runtime) = forja_serve::serving_router(FakeService);
        let readiness = runtime.clone();
        let mut supervisor = Supervisor::new(
            sources,
            HostConfig::default(),
            4,
            8,
            FakeFactory(VecDeque::new()),
            Box::new(FakeLogging(true)),
            runtime,
        )
        .unwrap();
        let error = supervisor.reload().await.unwrap_err();
        assert!(error.contains("logger rejected policy"));
        assert_eq!(supervisor.generation, 0);
        assert!(readiness.is_ready());
        supervisor.shutdown().await.unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
