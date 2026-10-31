use crate::{Backend, CanonicalPrompt, Host, HybridPrefix, compute, invalid_handle};
use std::sync::Arc;

impl<B: Backend + Send + Sync + 'static> Host<B> {
    pub(crate) fn validate_canonical_input(
        &self,
        pool: u32,
        lane: u32,
        phase: crate::BatchPhase,
        start: u32,
        tokens: Option<&[u32]>,
    ) -> Result<(), compute::Error> {
        if let Some(slot) = self.paged_state_slot(pool)? {
            self.state_slots
                .get(&slot)
                .ok_or_else(|| invalid_handle("protected state is unavailable"))?
                .validate_prompt_input(
                    usize::try_from(lane).map_err(invalid_handle)?,
                    phase,
                    start,
                    tokens,
                )?;
        }
        Ok(())
    }

    pub(crate) fn prepare_canonical_lane(
        &mut self,
        pool: u32,
        lane: u32,
        prompt: Arc<CanonicalPrompt>,
        hit: Option<&Arc<HybridPrefix<B>>>,
        reserved: u64,
    ) -> Result<u64, compute::Error> {
        let slot = self
            .paged_state_slot(pool)?
            .ok_or_else(|| invalid_handle("canonical prefix requires protected state"))?;
        let index = usize::try_from(lane).map_err(invalid_handle)?;
        let state = self
            .state_slots
            .get(&slot)
            .ok_or_else(|| invalid_handle("protected state is unavailable"))?;
        let staged = state.stage_prefix(index, prompt, hit, reserved)?;
        let attention = staged.attention().cloned();
        let count = attention
            .as_ref()
            .map_or(Ok(0), crate::PagedPrefix::logical_len)
            .map_err(invalid_handle)?;
        self.bind_paged_lane_staged(pool, lane, attention, reserved, Some(staged))?;
        Ok(count)
    }
}

impl<B: Backend + Send + Sync + 'static> crate::EngineRunner<B> {
    /// Admits paired carry/attention at an exact canonical prompt hit or a cold origin.
    ///
    /// # Errors
    /// Refuses occupied lanes, foreign state, mismatched ledgers or insufficient quota.
    pub fn prepare_canonical_lane(
        &mut self,
        lane: u32,
        prompt: Arc<CanonicalPrompt>,
        hit: Option<&Arc<HybridPrefix<B>>>,
        reserved: u64,
    ) -> Result<u64, compute::Error> {
        let pool = self
            .page_pool
            .ok_or_else(|| invalid_handle("canonical prefix requires paged storage"))?;
        let count = self
            .store
            .data_mut()
            .prepare_canonical_lane(pool, lane, prompt, hit, reserved)?;
        if lane == 0 {
            self.paged_origin = None;
            self.paged_checkpoint = None;
        }
        Ok(count)
    }
}

impl<B: Backend + Send + Sync + 'static> crate::EngineRunner<B> {
    /// Extends the serving artifact identity with actual cache geometry and reduction picks.
    ///
    /// # Errors
    /// Refuses engines without protected paged state or canonical prompt blocks.
    pub fn canonical_prefix_namespace(
        &self,
        base: [u8; 32],
        classes: &[&str],
    ) -> Result<forja_core::HybridPrefixNamespace, compute::Error> {
        let pool = self
            .page_pool
            .ok_or_else(|| invalid_handle("paged storage is unavailable"))?;
        let host = self.store.data();
        let slot = host
            .paged_state_slot(pool)?
            .ok_or_else(|| invalid_handle("protected state is unavailable"))?;
        let state = host
            .state_slots
            .get(&slot)
            .ok_or_else(|| invalid_handle("protected state is unavailable"))?;
        let pages = host
            .table
            .get(&wasmtime::component::Resource::<crate::PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?;
        let block = self
            .info
            .as_ref()
            .and_then(|info| info.prompt_block_tokens)
            .ok_or_else(|| invalid_handle("canonical prompt blocks are unavailable"))?;
        forja_core::HybridPrefixNamespace::new(
            base,
            state.prefix_layout(),
            pages.pool.layout(),
            block,
            classes,
        )
        .map_err(invalid_handle)
    }

    /// Allocates an immutable, charged prompt ledger for canonical evaluation.
    ///
    /// # Errors
    /// Refuses invalid prompt lengths or exhausted host memory.
    pub fn canonical_prompt(
        &self,
        namespace: forja_core::HybridPrefixNamespace,
        tokens: Vec<u32>,
    ) -> Result<Arc<CanonicalPrompt>, compute::Error> {
        self.store.data().canonical_prompt(namespace, tokens)
    }

    /// Allocates a charged completed context, including an aligned final boundary.
    ///
    /// # Errors
    /// Refuses empty contexts or exhausted host memory.
    pub fn canonical_context(
        &self,
        namespace: forja_core::HybridPrefixNamespace,
        tokens: Vec<u32>,
    ) -> Result<Arc<CanonicalPrompt>, compute::Error> {
        self.store.data().canonical_context(namespace, tokens)
    }

    /// Constructs a cache within the existing attention-byte and host quotas.
    ///
    /// # Errors
    /// Refuses unavailable storage or an index that does not fit the byte ceiling.
    pub fn create_canonical_prefix_store(
        &self,
        namespace: forja_core::HybridPrefixNamespace,
        allowance: crate::SnapshotAllowance,
    ) -> Result<crate::HybridPrefixStore<B>, compute::Error> {
        let pool = self
            .page_pool
            .ok_or_else(|| invalid_handle("paged storage is unavailable"))?;
        let host = self.store.data();
        let bytes = host
            .limits
            .page_pool_bytes
            .unwrap_or(host.limits.live_bytes)
            .min(host.limits.live_bytes);
        host.create_hybrid_prefix_store(
            &wasmtime::component::Resource::new_borrow(pool),
            namespace,
            allowance,
            bytes,
            host.limits.table_elements,
        )
    }

    /// Captures paired attention and all protected planes at a completed prompt boundary.
    ///
    /// # Errors
    /// Refuses noncanonical positions, unfinished submissions or exhausted snapshot quotas.
    pub fn capture_canonical_prefix(
        &self,
        lane: u32,
        allowance: crate::SnapshotAllowance,
    ) -> Result<Arc<HybridPrefix<B>>, compute::Error> {
        let pool = self
            .page_pool
            .ok_or_else(|| invalid_handle("paged storage is unavailable"))?;
        let host = self.store.data();
        let sequence = host
            .lane_state(pool, lane)?
            .ok_or_else(|| invalid_handle("protected lane is unavailable"))?;
        sequence
            .wait_drained(host.limits.submission_timeout)
            .map_err(invalid_handle)?;
        let image = sequence.capture_snapshot(allowance)?;
        let attention = host.paged_lane_prefix(pool, lane)?;
        host.hybrid_prefix(
            &wasmtime::component::Resource::new_borrow(pool),
            &attention,
            image.clone(),
            image.tokens(),
        )
    }

    /// Fingerprints the raw bits of every completed protected plane, including prompt tails.
    ///
    /// # Errors
    /// Refuses unfinished work, unavailable lanes or diagnostic scratch exhaustion.
    pub fn state_lane_fingerprint(
        &self,
        lane: u32,
    ) -> Result<(crate::LanePosition, [u8; 32]), compute::Error> {
        let pool = self
            .page_pool
            .ok_or_else(|| invalid_handle("paged storage is unavailable"))?;
        let host = self.store.data();
        let sequence = host
            .lane_state(pool, lane)?
            .ok_or_else(|| invalid_handle("protected lane is unavailable"))?;
        sequence
            .wait_drained(host.limits.submission_timeout)
            .map_err(invalid_handle)?;
        sequence.fingerprint()
    }
}
