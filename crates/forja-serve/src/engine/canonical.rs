use super::{GenerationControl, GenerationError, ServingEngine, prefix_limits};
use forja_config::Profile;
use forja_core::{Backend, CanonicalPrefixSchedule, HybridPrefixNamespace};
use forja_host::{CanonicalPrompt, HybridPrefix, HybridPrefixStore, SnapshotAllowance};
use std::sync::Arc;
type Admission<B> = (Arc<CanonicalPrompt>, Option<Arc<HybridPrefix<B>>>);

pub(super) struct CanonicalCache<B: Backend> {
    pub(super) schedule: CanonicalPrefixSchedule,
    namespace: HybridPrefixNamespace,
    allowance: SnapshotAllowance,
    declared: u64,
    pub(super) store: Option<HybridPrefixStore<B>>,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
    /// Enables paired prefix caching using the profile's canonical numerical contract.
    ///
    /// # Errors
    /// Refuses invalid spacing, missing protected storage or cache ceilings above host quotas.
    pub fn with_profile_prefixes(mut self, profile: &Profile) -> Result<Self, GenerationError> {
        let Some(declared) = profile
            .serving()
            .and_then(|serving| serving.state_snapshot_bytes)
        else {
            return Ok(self);
        };
        let interval = profile
            .serving()
            .and_then(|serving| serving.state_snapshot_interval)
            .ok_or_else(|| {
                GenerationError::new("snapshot allowance requires a canonical interval")
            })?;
        let block = self.info.prompt_block_tokens.ok_or_else(|| {
            GenerationError::new("snapshot allowance requires canonical prompt blocks")
        })?;
        let schedule = CanonicalPrefixSchedule::new(block, interval).map_err(cache_error)?;
        let effective = self
            .config
            .resolve_state_snapshot_bytes(profile)
            .map_err(GenerationError::new)?;
        let allowance =
            SnapshotAllowance::new(declared.get(), Some(effective)).map_err(cache_error)?;
        let classes = profile
            .variants()
            .fixed
            .values()
            .map(String::as_str)
            .chain(
                profile
                    .variants()
                    .rules
                    .values()
                    .flat_map(|rule| rule.arms.iter().map(|arm| arm.name.as_str())),
            )
            .collect::<Vec<_>>();
        let namespace = self
            .runner
            .canonical_prefix_namespace(self.prefix_namespace.0, &classes)
            .map_err(cache_error)?;
        let store = if allowance.bytes() == 0 {
            None
        } else {
            Some(
                self.runner
                    .create_canonical_prefix_store(namespace, allowance)
                    .map_err(cache_error)?,
            )
        };
        self.canonical_cache = Some(CanonicalCache {
            schedule,
            namespace,
            allowance,
            declared: declared.get(),
            store,
        });
        self.configure_canonical_cache(&self.config.clone())?;
        Ok(self)
    }

    pub(super) fn configure_canonical_cache(
        &mut self,
        config: &forja_config::HostConfig,
    ) -> Result<(), GenerationError> {
        let Some(cache) = &mut self.canonical_cache else {
            return Ok(());
        };
        let effective = match config.serving.state_snapshot_bytes {
            forja_config::Choice::Auto => cache
                .allowance
                .with_operator(cache.declared)
                .map_err(cache_error)?,
            forja_config::Choice::Fixed(bytes) => cache
                .allowance
                .with_operator(bytes.get())
                .map_err(cache_error)?,
        };
        let serving = self
            .info
            .serving
            .as_ref()
            .ok_or_else(|| GenerationError::new("canonical cache requires serving geometry"))?;
        let limits = prefix_limits(serving, config, self.max_context)?;
        if let Some(store) = &mut cache.store {
            store
                .reconfigure_policy(
                    limits.entries,
                    limits.tokens as u64,
                    limits.referenced_pages as u64,
                    effective,
                    config.serving.prefix_reuse,
                )
                .map_err(cache_error)?;
        } else if effective.bytes() > 0 {
            let mut store = self
                .runner
                .create_canonical_prefix_store(cache.namespace, effective)
                .map_err(cache_error)?;
            store
                .reconfigure_policy(
                    limits.entries,
                    limits.tokens as u64,
                    limits.referenced_pages as u64,
                    effective,
                    config.serving.prefix_reuse,
                )
                .map_err(cache_error)?;
            cache.store = Some(store);
        }
        cache.allowance = effective;
        Ok(())
    }

    pub(super) fn canonical_admission(
        &self,
        tokens: &[u32],
        reuse: bool,
    ) -> Result<Option<Admission<B>>, GenerationError> {
        let Some(cache) = &self.canonical_cache else {
            return Ok(None);
        };
        if tokens.len() < 2 {
            return Ok(None);
        }
        let hit = cache
            .store
            .as_ref()
            .filter(|_| reuse && self.config.serving.prefix_reuse)
            .map(|store| store.lookup(cache.namespace, tokens))
            .transpose()
            .map_err(cache_error)?
            .flatten();
        let prompt = self
            .runner
            .canonical_prompt(cache.namespace, tokens.to_vec())
            .map_err(cache_error)?;
        Ok(Some((prompt, hit)))
    }

    pub(super) fn evict_canonical_prefix(&self) -> Result<bool, GenerationError> {
        self.canonical_cache
            .as_ref()
            .and_then(|cache| cache.store.as_ref())
            .map(HybridPrefixStore::evict_lru)
            .transpose()
            .map_err(cache_error)
            .map(|evicted| evicted.unwrap_or(false))
    }

    pub(super) fn canonical_piece(
        &self,
        position: u32,
        prompt_tokens: u32,
        proposed: u32,
    ) -> Result<u32, GenerationError> {
        self.canonical_cache.as_ref().map_or(Ok(proposed), |cache| {
            cache
                .schedule
                .limit(u64::from(position), u64::from(prompt_tokens), proposed)
                .map_err(cache_error)
        })
    }
    pub(super) fn publish_canonical_boundary(
        &mut self,
        lane: u32,
        position: u32,
        prompt_tokens: u32,
        reuse: bool,
        control: &GenerationControl,
    ) -> Result<(), GenerationError> {
        let Some(cache) = &self.canonical_cache else {
            return Ok(());
        };
        let Some(store) = &cache.store else {
            return Ok(());
        };
        if cache.allowance.bytes() == 0
            || !reuse
            || !self.config.serving.prefix_reuse
            || control.is_cancelled()
            || !cache
                .schedule
                .contains(u64::from(position), u64::from(prompt_tokens))
        {
            return Ok(());
        }
        let result = self
            .runner
            .capture_canonical_prefix(lane, cache.allowance)
            .and_then(|entry| store.publish(entry, &control.cancellation.cancelled));
        match result {
            Ok(_) => Ok(()),
            Err(forja_host::bindings::l9o::gpu::compute::Error::Quota(_)) => {
                self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
                Ok(())
            }
            Err(error) => Err(cache_error(error)),
        }
    }
}

fn cache_error(error: impl std::fmt::Display) -> GenerationError {
    GenerationError::unavailable(error.to_string())
}
