use crate::{Backend, CanonicalPrompt, Host, HybridPrefix, compute, invalid_handle};
use std::sync::Arc;

impl<B: Backend + Send + Sync + 'static> Host<B> {
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
