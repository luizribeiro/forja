/// How a serving configuration leaf may be applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReloadClass {
    /// Apply to future work without restarting the process.
    Live,
    /// Replace the loaded model as one generation.
    Model,
    /// Restart the service before applying the value.
    Restart,
}

/// Changed serving leaves grouped by their application boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HostConfigDiff {
    /// Leaves applied to future work in the current runtime.
    pub live: Vec<&'static str>,
    /// Leaves requiring replacement of the loaded model.
    pub model: Vec<&'static str>,
    /// Leaves requiring process restart.
    pub restart: Vec<&'static str>,
}

impl HostConfigDiff {
    /// Compares two complete host configurations using the registered reload metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if either complete configuration cannot be inspected.
    pub fn between(
        previous: &crate::HostConfig,
        candidate: &crate::HostConfig,
    ) -> Result<Self, String> {
        let previous = toml::Value::try_from(previous)
            .map_err(|error| format!("cannot inspect active host configuration: {error}"))?;
        let candidate = toml::Value::try_from(candidate)
            .map_err(|error| format!("cannot inspect candidate host configuration: {error}"))?;
        let mut diff = Self::default();
        for &(path, class) in HOST_LEAVES {
            if leaf(&previous, path) == leaf(&candidate, path) {
                continue;
            }
            match class {
                ReloadClass::Live => diff.live.push(path),
                ReloadClass::Model => diff.model.push(path),
                ReloadClass::Restart => diff.restart.push(path),
            }
        }
        Ok(diff)
    }

    /// Returns whether every host configuration leaf is unchanged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live.is_empty() && self.model.is_empty() && self.restart.is_empty()
    }
}

fn leaf<'a>(root: &'a toml::Value, path: &str) -> Option<&'a toml::Value> {
    path.split('.')
        .try_fold(root, |value, part| value.get(part))
}

impl ReloadClass {
    /// Returns the stable JSON Schema spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Model => "model",
            Self::Restart => "restart",
        }
    }
}

/// Every host configuration leaf and its reload class.
pub const HOST_LEAVES: &[(&str, ReloadClass)] = &[
    ("server.listen", ReloadClass::Restart),
    ("server.queue_length", ReloadClass::Live),
    ("server.request_deadline", ReloadClass::Live),
    ("server.header_read_timeout", ReloadClass::Live),
    ("server.body_read_timeout", ReloadClass::Live),
    ("server.sse_write_stall_timeout", ReloadClass::Live),
    ("server.model_drain_deadline", ReloadClass::Live),
    ("model.engine", ReloadClass::Model),
    ("model.profile", ReloadClass::Model),
    ("model.model", ReloadClass::Model),
    ("model.chat_format", ReloadClass::Model),
    ("serving.prefix-reuse", ReloadClass::Live),
    ("serving.max_context", ReloadClass::Model),
    ("serving.max_active_sequences", ReloadClass::Live),
    ("serving.max_batch_size", ReloadClass::Live),
    ("serving.max_prefill_chunks", ReloadClass::Live),
    ("serving.prompt_token_budget", ReloadClass::Live),
    ("serving.idle_prompt_token_budget", ReloadClass::Live),
    ("serving.prefix_cache_entries", ReloadClass::Live),
    ("serving.prefix_cache_tokens", ReloadClass::Live),
    ("serving.prefix_cache_pages", ReloadClass::Live),
    ("serving.page_pool_bytes", ReloadClass::Restart),
    ("backend.kind", ReloadClass::Restart),
    ("backend.metal.graph_replay", ReloadClass::Restart),
    ("backend.metal.pool_capacity", ReloadClass::Restart),
    ("limits.live_bytes", ReloadClass::Restart),
    ("limits.tensor_rank", ReloadClass::Restart),
    ("limits.tensor_elements", ReloadClass::Restart),
    ("limits.live_tensor_handles", ReloadClass::Restart),
    ("limits.live_kernels", ReloadClass::Restart),
    ("limits.live_graphs", ReloadClass::Restart),
    ("limits.read_bytes", ReloadClass::Restart),
    ("limits.guest_memory_bytes", ReloadClass::Restart),
    ("limits.table_elements", ReloadClass::Restart),
    ("limits.instances", ReloadClass::Restart),
    ("limits.dispatches_per_list", ReloadClass::Restart),
    ("limits.work_per_dispatch", ReloadClass::Restart),
    ("limits.guest_call_timeout", ReloadClass::Live),
    ("limits.submission_timeout", ReloadClass::Live),
    ("limits.gpu_time_budget", ReloadClass::Live),
    ("request.max_body_bytes", ReloadClass::Live),
    ("request.max_messages", ReloadClass::Live),
    ("request.max_text_bytes", ReloadClass::Live),
    ("request.max_prompt_tokens", ReloadClass::Live),
    ("request.max_output_tokens", ReloadClass::Live),
    ("request.default_output_tokens", ReloadClass::Live),
    ("request.max_message_bytes", ReloadClass::Live),
    ("request.max_rendered_bytes", ReloadClass::Live),
    ("request.max_stops", ReloadClass::Live),
    ("request.max_stop_bytes", ReloadClass::Live),
    ("request.max_tools", ReloadClass::Live),
    ("request.max_tool_bytes", ReloadClass::Live),
    ("request.max_schema_depth", ReloadClass::Live),
    ("request.max_parser_events", ReloadClass::Live),
    ("request.max_grammar_bytes", ReloadClass::Live),
    ("request.max_grammar_states", ReloadClass::Live),
    ("request.max_grammar_compile_time", ReloadClass::Live),
    ("pools.grammar_workers", ReloadClass::Restart),
    ("pools.grammar_queue_length", ReloadClass::Restart),
    ("logging.level", ReloadClass::Live),
    ("logging.format", ReloadClass::Live),
    ("logging.include-spans", ReloadClass::Live),
    ("metrics.listen", ReloadClass::Restart),
    ("metrics.path", ReloadClass::Restart),
];

/// Returns the reload class for one host configuration leaf.
#[must_use]
pub fn host_reload_class(path: &str) -> Option<ReloadClass> {
    HOST_LEAVES
        .iter()
        .find_map(|&(leaf, class)| (leaf == path).then_some(class))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroU32;

    use super::*;
    use crate::Choice;

    #[test]
    fn leaf_registry_is_unique_and_classifies_sensitive_boundaries() {
        let unique = HOST_LEAVES
            .iter()
            .map(|(path, _)| *path)
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), HOST_LEAVES.len());
        assert_eq!(host_reload_class("model.profile"), Some(ReloadClass::Model));
        assert_eq!(
            host_reload_class("serving.max_context"),
            Some(ReloadClass::Model)
        );
        assert_eq!(
            host_reload_class("serving.page_pool_bytes"),
            Some(ReloadClass::Restart)
        );
        assert_eq!(
            host_reload_class("limits.submission_timeout"),
            Some(ReloadClass::Live)
        );
        assert_eq!(host_reload_class("request.unknown"), None);
    }

    #[test]
    fn typed_diff_groups_each_changed_leaf() {
        let previous = crate::HostConfig::default();
        let mut candidate = previous.clone();
        candidate.server.queue_length = 7;
        candidate.model.engine = "next.wasm".into();
        candidate.serving.max_context = Choice::Fixed(NonZeroU32::new(2_048).unwrap());
        candidate.metrics.path = "/new-metrics".to_owned();
        let diff = HostConfigDiff::between(&previous, &candidate).unwrap();
        assert_eq!(diff.live, ["server.queue_length"]);
        assert_eq!(diff.model, ["model.engine", "serving.max_context"]);
        assert_eq!(diff.restart, ["metrics.path"]);
        assert!(!diff.is_empty());
    }
}
