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
        let effective = self
            .config
            .resolve_state_snapshot_bytes(profile)
            .map_err(GenerationError::new)?;
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
        self.canonical_cache = Some(CanonicalCache {
            schedule,
            namespace,
            allowance,
            declared: declared.get(),
            store: None,
        });
        self.configure_canonical_cache(&self.config.clone())?;
        Ok(self)
    }

    pub(super) fn configure_canonical_cache(
        &mut self,
        config: &forja_config::HostConfig,
    ) -> Result<(), GenerationError> {
        let Some(cache) = &mut self.canonical_cache else {
            if matches!(config.serving.state_snapshot_bytes, forja_config::Choice::Fixed(bytes) if bytes.get() > 0)
            {
                return Err(GenerationError::new(
                    "snapshot retention requires a profile declaration",
                ));
            }
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
        let mut created = if cache.store.is_none() && effective.bytes() > 0 {
            Some(
                self.runner
                    .create_canonical_prefix_store(cache.namespace, effective)
                    .map_err(cache_error)?,
            )
        } else {
            None
        };
        if let Some(store) = cache.store.as_mut().or(created.as_mut()) {
            store
                .reconfigure_policy(
                    limits.entries,
                    u64::try_from(limits.tokens).map_err(cache_error)?,
                    u64::try_from(limits.referenced_pages).map_err(cache_error)?,
                    effective,
                    config.serving.prefix_reuse,
                )
                .map_err(cache_error)?;
        }
        if created.is_some() {
            cache.store = created;
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
        self.canonical_cache
            .as_ref()
            .filter(|cache| cache.allowance.bytes() > 0)
            .map_or(Ok(proposed), |cache| {
                cache
                    .schedule
                    .limit(u64::from(position), u64::from(prompt_tokens), proposed)
                    .map_err(cache_error)
            })
    }
    pub(super) async fn prefill_pieces(
        &mut self,
        tokens: &[u32],
        mut position: u32,
        sampling: forja_host::SamplingParams,
        control: &GenerationControl,
    ) -> Result<forja_host::EngineDecodeOutput, GenerationError> {
        let length = u32::try_from(tokens.len())
            .map_err(|_| GenerationError::new("prompt length overflow"))?;
        loop {
            if control.is_cancelled() {
                return Err(GenerationError::cancelled());
            }
            let proposed = length
                .checked_sub(position)
                .ok_or_else(|| GenerationError::new("prompt position exceeds ledger"))?;
            let proposed = if self.canonical_cache.is_some() {
                proposed.min(self.scheduler.profile_prefill_chunk)
            } else {
                proposed
            };
            let count = self.canonical_piece(position, length, proposed)?;
            let end = position
                .checked_add(count)
                .ok_or_else(|| GenerationError::new("prompt extent overflow"))?;
            let start = usize::try_from(position)
                .map_err(|_| GenerationError::new("prompt position exceeds usize"))?;
            let output =
                self.runner
                    .enqueue_decode(forja_host::EngineDecode {
                        phase: forja_host::BatchPhase::Prompt,
                        tokens: Some(
                            tokens[start..usize::try_from(end).map_err(|_| {
                                GenerationError::new("prompt extent exceeds usize")
                            })?]
                                .to_vec(),
                        ),
                        start_pos: position,
                        sampling,
                    })
                    .await
                    .map_err(cache_error)?
                    .map_err(cache_error)?;
            if end == length {
                return Ok(output);
            }
            self.runner
                .read_queued_token(output)
                .await
                .map_err(cache_error)?;
            self.publish_canonical_boundary(0, end, length, true, control)?;
            position = end;
        }
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
