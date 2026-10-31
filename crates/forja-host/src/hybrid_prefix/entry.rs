use crate::{
    Host, PagePoolEntry, PagedPrefix, StateSnapshot, compute, invalid_handle, state::StateCharge,
};
use forja_core::{Backend, HybridPrefixNamespace, PagePool};
use std::sync::Arc;
use wasmtime::component::Resource;

/// One indivisible attention-and-carry image at an exact prompt boundary.
pub struct HybridPrefix<B: Backend> {
    pub(super) image: Arc<StateSnapshot<B>>,
    pub(super) attention: PagedPrefix,
    pub(super) pool: PagePool,
    pub(super) pages: Vec<(u32, u32)>,
    _charge: StateCharge,
    retained: u64,
    pub(super) metadata_bytes: u64,
}

impl<B: Backend> HybridPrefix<B> {
    /// Exact ledger shared by every attention and state plane.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        self.image.tokens()
    }

    /// Artifact and numerical identity of this complete boundary.
    #[must_use]
    pub fn namespace(&self) -> HybridPrefixNamespace {
        self.image.namespace()
    }

    /// Bytes retained by this entry before deduplicating allocations shared with other entries.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }

    /// Number of shared physical attention pages in this image.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Charged descriptors and references, excluding shared tensor/page allocations.
    #[must_use]
    pub const fn reference_bytes(&self) -> u64 {
        self.metadata_bytes
    }

    /// Stored attention length; this is always the complete carry boundary.
    ///
    /// # Errors
    /// Refuses poisoned attention metadata.
    pub fn attention_tokens(&self) -> Result<u64, compute::Error> {
        self.attention.logical_len().map_err(invalid_handle)
    }

    /// Physical attention-plane geometry retained by this entry.
    #[must_use]
    pub fn attention_layout(&self) -> &forja_core::PagedLayout {
        self.pool.layout()
    }
}

impl<B: Backend + 'static> Host<B> {
    /// Prepares an exact hybrid boundary without admitting it to a cache.
    ///
    /// # Errors
    /// Refuses foreign pools/hosts, mismatched ledgers, layouts, lengths or quota exhaustion.
    pub fn hybrid_prefix(
        &self,
        pool: &Resource<PagePoolEntry>,
        attention: &PagedPrefix,
        image: Arc<StateSnapshot<B>>,
        attention_tokens: &[u32],
    ) -> Result<Arc<HybridPrefix<B>>, compute::Error> {
        let pool = &self.table.get(pool).map_err(invalid_handle)?.pool;
        let mut source = attention.sequence.lock().map_err(invalid_handle)?;
        if !image.owned_by(&self.backend, &self.live_bytes)
            || !image.namespace().matches_attention(pool.layout())
            || !pool.owns_sequence(&source)
            || image.tokens() != attention_tokens
            || source.logical_len() < image.position().tokens
        {
            return Err(invalid_handle(
                "hybrid boundary identity or token ledger mismatch",
            ));
        }
        let count = image
            .position()
            .tokens
            .div_ceil(u64::from(pool.layout().page_tokens()));
        let metadata_bytes = count
            .checked_mul(64)
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| crate::quota("hybrid reference bytes overflow"))?;
        let charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            metadata_bytes,
        )?;
        let child = source
            .fork_reserved(image.position().tokens, image.position().tokens)
            .map_err(invalid_handle)?;
        let pages = child
            .table()
            .map_err(invalid_handle)?
            .entries()
            .iter()
            .map(|entry| (entry.page(), entry.generation()))
            .collect();
        let retained = image
            .retained_allocations()
            .iter()
            .try_fold(metadata_bytes, |sum, (_, bytes)| sum.checked_add(*bytes))
            .and_then(|n| {
                pool.layout()
                    .total_page_bytes()
                    .checked_mul(count)
                    .and_then(|pages| n.checked_add(pages))
            })
            .ok_or_else(|| crate::quota("hybrid byte count overflow"))?;
        let entry = Arc::new(HybridPrefix {
            image,
            attention: PagedPrefix::new(child),
            pool: pool.clone(),
            pages,
            _charge: charge,
            metadata_bytes,
            retained,
        });

        Ok(entry)
    }
}
