use super::{
    Arc, Backend, HybridPrefixNamespace, HybridPrefixStore, compute, invalid_handle, quota,
};

impl<B: Backend> HybridPrefixStore<B> {
    /// Exact artifact, geometry and numerical identity accepted by this cache.
    #[must_use]
    pub const fn namespace(&self) -> HybridPrefixNamespace {
        self.namespace
    }

    /// Evicts the oldest unpinned entry to relieve request-admission pressure.
    ///
    /// # Errors
    /// Refuses poisoned cache metadata.
    pub fn evict_lru(&self) -> Result<bool, compute::Error> {
        let mut entries = self.entries.lock().map_err(invalid_handle)?;
        let Some(index) = entries
            .iter()
            .position(|entry| Arc::strong_count(entry) == 1)
        else {
            return Ok(false);
        };
        entries.remove(index);
        Ok(true)
    }

    /// Applies bounded reference ceilings, preserving pinned entries and atomic refusal.
    ///
    /// # Errors
    /// Refuses growth beyond the allocated index or ceilings that pinned entries exceed.
    pub fn reconfigure(
        &mut self,
        maximum: usize,
        tokens: u64,
        pages: u64,
        enabled: bool,
    ) -> Result<(), compute::Error> {
        self.reconfigure_policy(maximum, tokens, pages, self.allowance, enabled)
    }

    /// Atomically resolves snapshot and reference ceilings, evicting only unpinned entries.
    ///
    /// # Errors
    /// Refuses ceilings above the guest declaration or below retained pinned allocations.
    pub fn reconfigure_policy(
        &mut self,
        maximum: usize,
        tokens: u64,
        pages: u64,
        allowance: crate::SnapshotAllowance,
        enabled: bool,
    ) -> Result<(), compute::Error> {
        let allowance = self.allowance.with_operator(allowance.bytes())?;
        let mut entries = self.entries.lock().map_err(invalid_handle)?;
        if maximum > entries.capacity() {
            return Err(quota("hybrid index capacity exceeded"));
        }
        let mut removed = Vec::new();
        removed
            .try_reserve_exact(entries.len())
            .map_err(|_| quota("hybrid reconfiguration allocation failed"))?;
        removed.resize(entries.len(), false);
        loop {
            let usage = self.measure(
                entries
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !removed[*index])
                    .map(|(_, entry)| entry),
            )?;
            let fits = usage.entries <= maximum
                && usage.tokens <= tokens
                && usage.pages <= pages
                && usage.snapshot_bytes <= allowance.bytes();
            let victim = entries
                .iter()
                .enumerate()
                .position(|(index, entry)| !removed[index] && Arc::strong_count(entry) == 1);
            if fits && (enabled || victim.is_none()) {
                break;
            }
            let Some(victim) = victim else {
                return Err(quota("pinned hybrid references exceed the new ceilings"));
            };
            removed[victim] = true;
        }
        let mut index = 0;
        entries.retain(|_| {
            let keep = !removed[index];
            index += 1;
            keep
        });
        drop(entries);
        self.maximum = maximum;
        self.tokens = tokens;
        self.pages = pages;
        self.enabled = enabled;
        self.allowance = allowance;
        Ok(())
    }
}
