use std::sync::Arc;

use forja_config::{HostConfig, HostConfigDiff};
use forja_serve::LoadPlan;

use crate::args::{HostConfigSources, load_host_config};

#[derive(Debug)]
#[allow(dead_code, reason = "constructed by the runtime supervisor")]
pub(crate) struct Candidate {
    pub(crate) generation: u64,
    pub(crate) config: Arc<HostConfig>,
    pub(crate) diff: HostConfigDiff,
    pub(crate) plan: LoadPlan,
}

impl Candidate {
    #[allow(dead_code, reason = "called by the runtime supervisor")]
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
