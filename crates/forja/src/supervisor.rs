use std::sync::Arc;

use forja_config::{HostConfig, HostConfigDiff};
use forja_serve::{LoadPlan, ProcessLedger};

use crate::{
    args::{HostConfigSources, load_host_config},
    serve::RuntimeFactory,
};

#[derive(Debug)]
#[allow(dead_code, reason = "constructed during configuration reload")]
pub(crate) struct Candidate {
    pub(crate) generation: u64,
    pub(crate) config: Arc<HostConfig>,
    pub(crate) diff: HostConfigDiff,
    pub(crate) plan: LoadPlan,
}

impl Candidate {
    #[allow(dead_code, reason = "called during configuration reload")]
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
    retained_bytes: u64,
    drain_deadline: std::time::Duration,
    ledger: ProcessLedger,
    runtime: forja_serve::DrainHandle<F::Service>,
}

impl<F: RuntimeFactory> Supervisor<F> {
    pub(crate) fn new(
        retained_bytes: u64,
        quota: u64,
        drain_deadline: std::time::Duration,
        runtime: forja_serve::DrainHandle<F::Service>,
    ) -> Result<Self, String> {
        let mut ledger = ProcessLedger::new(quota);
        ledger.reserve(retained_bytes)?;
        Ok(Self {
            retained_bytes,
            drain_deadline,
            ledger,
            runtime,
        })
    }

    pub(crate) async fn shutdown(&mut self) -> Result<(), String> {
        self.runtime.close_admission();
        self.runtime.drain_generation(0, self.drain_deadline).await;
        self.runtime.unload()?;
        self.ledger.release(self.retained_bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use forja_testing::temporary_directory;

    use super::*;

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
        let plan = |_: &HostConfig| Ok(LoadPlan { retained_bytes: 7 });
        let candidate = Candidate::from_sources(1, &active, &sources, plan).unwrap();
        assert_eq!(candidate.diff.live, ["server.queue_length"]);
        assert_eq!(candidate.plan.retained_bytes, 7);
        fs::write(path, "[server]\nlisten='127.0.0.1:9000'\n").unwrap();
        assert!(
            Candidate::from_sources(2, &active, &sources, plan)
                .unwrap_err()
                .contains("server.listen")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
