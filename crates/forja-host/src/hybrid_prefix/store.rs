use super::{HybridPrefix, SnapshotAllowance};
use crate::{Host, PagePoolEntry, compute, invalid_handle, quota, state::StateCharge};
use forja_core::{Backend, HybridPrefixNamespace, PagePool};
use std::{
    collections::HashSet,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use wasmtime::component::Resource;

/// Shared physical allocations retained by the hybrid cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HybridCacheUsage {
    /// Unique attention backing, state images, ledgers and per-entry references.
    pub bytes: u64,
    /// Unique immutable state images and ledgers, excluding attention pages.
    pub snapshot_bytes: u64,
    /// Number of complete cached boundaries.
    pub entries: usize,
}

/// One bounded owner for complete hybrid boundaries in one artifact/page namespace.
pub struct HybridPrefixStore<B: Backend> {
    namespace: HybridPrefixNamespace,
    pool: PagePool,
    backend: Arc<B>,
    live: Arc<AtomicU64>,
    limit: u64,
    allowance: SnapshotAllowance,
    bytes: u64,
    maximum: usize,
    metadata_bytes: u64,
    entries: Mutex<Vec<Arc<HybridPrefix<B>>>>,
    _charge: StateCharge,
}

impl<B: Backend + 'static> Host<B> {
    /// Creates a cache subordinate to the existing total byte and table quotas.
    ///
    /// # Errors
    /// Refuses foreign layouts, invalid ceilings or quota exhaustion.
    pub fn create_hybrid_prefix_store(
        &self,
        pool: &Resource<PagePoolEntry>,
        namespace: HybridPrefixNamespace,
        allowance: SnapshotAllowance,
        bytes: u64,
        maximum: usize,
    ) -> Result<HybridPrefixStore<B>, compute::Error> {
        let pool = &self.table.get(pool).map_err(invalid_handle)?.pool;
        if !namespace.matches_attention(pool.layout())
            || bytes > self.limits.live_bytes
            || maximum == 0
            || maximum > self.limits.table_elements
        {
            return Err(quota("invalid hybrid cache limits or layout"));
        }
        let metadata = u64::try_from(maximum)
            .ok()
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| quota("cache metadata overflow"))?;
        if bytes < metadata {
            return Err(quota("cache ceiling cannot hold its index"));
        }
        let charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            metadata,
        )?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(maximum)
            .map_err(|_| quota("cache index allocation failed"))?;
        Ok(HybridPrefixStore {
            namespace,
            pool: pool.clone(),
            backend: Arc::clone(&self.backend),
            live: Arc::clone(&self.live_bytes),
            limit: self.limits.live_bytes,
            allowance,
            bytes,
            maximum,
            metadata_bytes: metadata,
            entries: Mutex::new(entries),
            _charge: charge,
        })
    }
}

impl<B: Backend> HybridPrefixStore<B> {
    fn measure<'a>(
        &self,
        entries: impl Clone + Iterator<Item = &'a Arc<HybridPrefix<B>>>,
    ) -> Result<HybridCacheUsage, compute::Error>
    where
        B: 'a,
    {
        let count = entries
            .clone()
            .try_fold(0_u64, |n, entry| {
                u64::try_from(entry.pages.len())
                    .ok()
                    .and_then(|pages| pages.checked_add(2))
                    .and_then(|resources| n.checked_add(resources))
            })
            .ok_or_else(|| quota("cache accounting extent overflow"))?;
        let temporary = count
            .checked_mul(64)
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| quota("cache accounting bytes overflow"))?;
        let _temporary = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live),
            self.limit,
            temporary,
        )?;
        let count = usize::try_from(count).map_err(|_| quota("cache accounting overflow"))?;
        let mut images = HashSet::new();
        let mut pages = HashSet::new();
        images
            .try_reserve(count)
            .map_err(|_| quota("cache accounting allocation failed"))?;
        pages
            .try_reserve(count)
            .map_err(|_| quota("cache accounting allocation failed"))?;
        let mut usage = HybridCacheUsage {
            bytes: self.metadata_bytes,
            snapshot_bytes: 0,
            entries: 0,
        };
        for entry in entries {
            usage.entries = usage
                .entries
                .checked_add(1)
                .ok_or_else(|| quota("cache entry count overflow"))?;
            usage.bytes = add_bytes(usage.bytes, entry.metadata_bytes)?;
            usage.snapshot_bytes = add_bytes(usage.snapshot_bytes, entry.metadata_bytes)?;
            for (identity, bytes) in entry.image.retained_allocations() {
                if images.insert(identity) {
                    usage.snapshot_bytes = add_bytes(usage.snapshot_bytes, bytes)?;
                    usage.bytes = add_bytes(usage.bytes, bytes)?;
                }
            }
            for page in &entry.pages {
                if pages.insert(*page) {
                    usage.bytes = add_bytes(usage.bytes, self.pool.layout().total_page_bytes())?;
                }
            }
        }
        Ok(usage)
    }

    /// Reports deduplicated retained allocations; references are charged per entry.
    ///
    /// # Errors
    /// Refuses corrupt metadata or unavailable temporary accounting quota.
    pub fn usage(&self) -> Result<HybridCacheUsage, compute::Error> {
        self.measure(self.entries.lock().map_err(invalid_handle)?.iter())
    }

    /// Pins the longest complete matching boundary and makes it most recently used.
    /// A hit always leaves at least the final prompt token for replay.
    ///
    /// # Errors
    /// Refuses poisoned cache metadata.
    pub fn lookup(
        &self,
        namespace: HybridPrefixNamespace,
        prompt: &[u32],
    ) -> Result<Option<Arc<HybridPrefix<B>>>, compute::Error> {
        if namespace != self.namespace {
            return Ok(None);
        }
        let mut entries = self.entries.lock().map_err(invalid_handle)?;
        let index = entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.tokens().len() < prompt.len() && prompt.starts_with(entry.tokens())
            })
            .max_by_key(|(_, entry)| entry.tokens().len())
            .map(|(index, _)| index);
        let Some(index) = index else {
            return Ok(None);
        };
        let hit = entries.remove(index);
        entries.push(Arc::clone(&hit));
        Ok(Some(hit))
    }

    fn eviction_plan(
        &self,
        entries: &[Arc<HybridPrefix<B>>],
        entry: &Arc<HybridPrefix<B>>,
    ) -> Result<Vec<bool>, compute::Error> {
        let mut evicted = Vec::new();
        evicted
            .try_reserve_exact(entries.len())
            .map_err(|_| quota("cache eviction plan allocation failed"))?;
        evicted.resize(entries.len(), false);
        loop {
            let retained = entries
                .iter()
                .enumerate()
                .filter(|(index, _)| !evicted[*index])
                .map(|(_, entry)| entry);
            let usage = self.measure(retained.chain(std::iter::once(entry)))?;
            if usage.entries <= self.maximum
                && usage.bytes <= self.bytes
                && usage.snapshot_bytes <= self.allowance.bytes()
            {
                return Ok(evicted);
            }
            let victim = entries
                .iter()
                .enumerate()
                .position(|(index, old)| !evicted[index] && Arc::strong_count(old) == 1)
                .ok_or_else(|| quota("hybrid cache ceiling exceeded; no unpinned victim"))?;
            evicted[victim] = true;
        }
    }

    /// Publishes a complete boundary, evicting the least recently used unpinned entries.
    /// Cancellation is sampled at the publication point; later cancellation does not revoke it.
    ///
    /// # Errors
    /// Refuses foreign identities or quota without changing retained entries.
    pub fn publish(
        &self,
        entry: Arc<HybridPrefix<B>>,
        cancelled: &AtomicBool,
    ) -> Result<bool, compute::Error> {
        let sequence = entry.attention.sequence.lock().map_err(invalid_handle)?;
        if entry.namespace() != self.namespace
            || !self.pool.owns_sequence(&sequence)
            || !entry.image.owned_by(&self.backend, &self.live)
        {
            return Err(invalid_handle(
                "hybrid publication belongs to another namespace or host",
            ));
        }
        drop(sequence);
        let mut entries = self.entries.lock().map_err(invalid_handle)?;
        if entries.iter().any(|old| old.tokens() == entry.tokens()) {
            return Ok(false);
        }
        let evicted = self.eviction_plan(&entries, &entry)?;
        if cancelled.load(Ordering::Acquire) {
            return Ok(false);
        }
        let mut index = 0;
        entries.retain(|_| {
            let keep = !evicted[index];
            index += 1;
            keep
        });
        entries.push(entry);
        Ok(true)
    }
}

fn add_bytes(left: u64, right: u64) -> Result<u64, compute::Error> {
    left.checked_add(right)
        .ok_or_else(|| quota("cache byte count overflow"))
}
