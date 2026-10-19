use std::{
    error::Error,
    fmt,
    sync::{Arc, Mutex},
};

use crate::{PageTable, PageTableEntry, PagedLayout};

/// A reason a paged resource operation was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagePoolError {
    /// A slab contains no pages.
    EmptySlab,
    /// A byte, token, page, or serial calculation overflowed.
    ArithmeticOverflow,
    /// The operation would exceed the pool's fixed byte quota.
    QuotaExceeded,
    /// A write exceeds or does not begin at the reserved logical boundary.
    UnreservedWrite,
    /// A write targets a shared or immutable tail without copy-on-write.
    SharedTail,
    /// A resource belongs to another pool.
    ForeignResource,
    /// A checkpoint is stale or does not belong to this sequence.
    InvalidCheckpoint,
    /// Internal state was poisoned or inconsistent.
    CorruptState,
}

impl fmt::Display for PagePoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "paged pool error: {self:?}")
    }
}

impl Error for PagePoolError {}

/// Bytes retained by one paged pool and its opaque resources.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PagedMemoryUsage {
    /// Physical plane backing plus allocation-generation metadata.
    pub slab_bytes: u64,
    /// Sequence page-table storage.
    pub table_bytes: u64,
    /// Rebindable slot-bank storage.
    pub slot_bytes: u64,
    /// Retained graph arenas attributed to this pool.
    pub graph_bytes: u64,
    /// Retained high-water submission scratch.
    pub scratch_bytes: u64,
}

impl PagedMemoryUsage {
    /// Returns the checked total retained bytes.
    #[must_use]
    pub fn total(self) -> Option<u64> {
        self.slab_bytes
            .checked_add(self.table_bytes)?
            .checked_add(self.slot_bytes)?
            .checked_add(self.graph_bytes)?
            .checked_add(self.scratch_bytes)
    }
}

/// Host-owned allocation metadata for one opaque page pool.
#[derive(Clone)]
pub struct PagePool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    layout: PagedLayout,
    pages_per_slab: u32,
    byte_limit: u64,
    state: Mutex<PoolState>,
}

struct PoolState {
    pages: Vec<PageState>,
    free: Vec<u32>,
    memory: PagedMemoryUsage,
}

#[derive(Clone, Copy)]
struct PageState {
    generation: u32,
    references: u32,
    lifecycle: PageLifecycle,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PageLifecycle {
    Free,
    Allocated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageLease {
    index: u32,
    generation: u32,
}

impl PagePool {
    /// Creates a bounded pool with one checked initial slab.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty slab, arithmetic overflow, or insufficient quota.
    pub fn new(
        layout: PagedLayout,
        pages_per_slab: u32,
        byte_limit: u64,
    ) -> Result<Self, PagePoolError> {
        if pages_per_slab == 0 {
            return Err(PagePoolError::EmptySlab);
        }
        let slab_bytes = slab_bytes(&layout, pages_per_slab)?;
        if slab_bytes > byte_limit {
            return Err(PagePoolError::QuotaExceeded);
        }
        let mut state = PoolState {
            pages: Vec::new(),
            free: Vec::new(),
            memory: PagedMemoryUsage {
                slab_bytes,
                ..PagedMemoryUsage::default()
            },
        };
        add_slab(&mut state, pages_per_slab)?;
        Ok(Self {
            inner: Arc::new(PoolInner {
                layout,
                pages_per_slab,
                byte_limit,
                state: Mutex::new(state),
            }),
        })
    }

    /// Returns the pool's validated plane geometry.
    #[must_use]
    pub fn layout(&self) -> &PagedLayout {
        &self.inner.layout
    }

    /// Atomically reserves physical pages and a hidden table for one sequence.
    ///
    /// # Errors
    ///
    /// Returns quota or overflow without changing allocation, reference, or byte state.
    pub fn reserve(&self, tokens: u64) -> Result<PagedSequence, PagePoolError> {
        let page_count = pages_for(tokens, self.inner.layout.page_tokens())?;
        let table_bytes = page_count
            .checked_mul(8)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let mut state = self.lock()?;
        preflight_pages(self, &state, page_count, table_bytes)?;
        grow_for(self, &mut state, page_count)?;
        let pages = allocate_pages(&mut state, page_count)?;
        state.memory.table_bytes = state
            .memory
            .table_bytes
            .checked_add(table_bytes)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        drop(state);
        Ok(PagedSequence {
            pool: self.clone(),
            pages,
            immutable: vec![
                false;
                usize::try_from(page_count)
                    .map_err(|_| PagePoolError::ArithmeticOverflow)?
            ],
            logical_len: 0,
            reserved_tokens: tokens,
        })
    }

    /// Returns current charged memory and the immutable quota.
    ///
    /// # Errors
    ///
    /// Returns an error if pool state is poisoned.
    pub fn memory_usage(&self) -> Result<(PagedMemoryUsage, u64), PagePoolError> {
        Ok((self.lock()?.memory, self.inner.byte_limit))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, PoolState>, PagePoolError> {
        self.inner
            .state
            .lock()
            .map_err(|_| PagePoolError::CorruptState)
    }

    fn refcount(&self, lease: PageLease) -> Result<u32, PagePoolError> {
        let mut state = self.lock()?;
        Ok(live_page_mut(&mut state, lease)?.references)
    }
}

/// An opaque sequence reservation and its hidden page table.
pub struct PagedSequence {
    pool: PagePool,
    pages: Vec<PageLease>,
    immutable: Vec<bool>,
    logical_len: u64,
    reserved_tokens: u64,
}

impl PagedSequence {
    /// Returns the committed logical token length.
    #[must_use]
    pub const fn logical_len(&self) -> u64 {
        self.logical_len
    }

    /// Returns the atomically reserved token capacity.
    #[must_use]
    pub const fn reserved_tokens(&self) -> u64 {
        self.reserved_tokens
    }

    /// Validates that a write is an append into exclusive reserved pages.
    ///
    /// # Errors
    ///
    /// Returns unreserved-write or shared-tail errors without changing sequence state.
    pub fn validate_write(&self, start: u64, tokens: u64) -> Result<(), PagePoolError> {
        let end = start
            .checked_add(tokens)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        if start != self.logical_len || end > self.reserved_tokens {
            return Err(PagePoolError::UnreservedWrite);
        }
        if tokens == 0 {
            return Ok(());
        }
        let page_tokens = u64::from(self.pool.layout().page_tokens());
        let first =
            usize::try_from(start / page_tokens).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let last = usize::try_from((end - 1) / page_tokens)
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        for page in first..=last {
            if self.immutable.get(page).copied().unwrap_or(true)
                || self
                    .pool
                    .refcount(*self.pages.get(page).ok_or(PagePoolError::UnreservedWrite)?)?
                    != 1
            {
                return Err(PagePoolError::SharedTail);
            }
        }
        Ok(())
    }

    /// Commits a previously validated append.
    ///
    /// # Errors
    ///
    /// Returns the same refusal as [`Self::validate_write`] if state changed.
    pub fn commit_write(&mut self, tokens: u64) -> Result<(), PagePoolError> {
        self.validate_write(self.logical_len, tokens)?;
        self.logical_len = self
            .logical_len
            .checked_add(tokens)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        Ok(())
    }

    /// Builds an authenticated table snapshot for trusted backend upload.
    ///
    /// # Errors
    ///
    /// Returns an error for stale internal allocator state.
    pub fn table(&self) -> Result<PageTable, PagePoolError> {
        let state = self.pool.lock()?;
        let generations = state.pages.iter().map(|page| page.generation).collect();
        let entries = self
            .pages
            .iter()
            .map(|page| PageTableEntry::new(page.index, page.generation))
            .collect();
        PageTable::new(entries, generations).map_err(|_| PagePoolError::CorruptState)
    }
}

fn slab_bytes(layout: &PagedLayout, pages_per_slab: u32) -> Result<u64, PagePoolError> {
    layout
        .total_page_bytes()
        .checked_add(4)
        .and_then(|bytes| bytes.checked_mul(u64::from(pages_per_slab)))
        .ok_or(PagePoolError::ArithmeticOverflow)
}

/// `page_tokens` is nonzero because it comes from a validated [`PagedLayout`].
fn pages_for(tokens: u64, page_tokens: u32) -> Result<u64, PagePoolError> {
    if tokens == 0 {
        return Ok(0);
    }
    tokens
        .checked_add(u64::from(page_tokens) - 1)
        .map(|tokens| tokens / u64::from(page_tokens))
        .ok_or(PagePoolError::ArithmeticOverflow)
}

fn add_slab(state: &mut PoolState, pages_per_slab: u32) -> Result<(), PagePoolError> {
    let start = u32::try_from(state.pages.len()).map_err(|_| PagePoolError::ArithmeticOverflow)?;
    let end = start
        .checked_add(pages_per_slab)
        .ok_or(PagePoolError::ArithmeticOverflow)?;
    state.pages.extend((start..end).map(|_| PageState {
        generation: 0,
        references: 0,
        lifecycle: PageLifecycle::Free,
    }));
    state.free.extend((start..end).rev());
    Ok(())
}

fn preflight_pages(
    pool: &PagePool,
    state: &PoolState,
    pages: u64,
    extra_bytes: u64,
) -> Result<(), PagePoolError> {
    let free = u64::try_from(state.free.len()).map_err(|_| PagePoolError::ArithmeticOverflow)?;
    let missing = pages.saturating_sub(free);
    let slabs = missing.div_ceil(u64::from(pool.inner.pages_per_slab));
    let growth = slab_bytes(&pool.inner.layout, pool.inner.pages_per_slab)?
        .checked_mul(slabs)
        .ok_or(PagePoolError::ArithmeticOverflow)?;
    let projected = state
        .memory
        .total()
        .and_then(|bytes| bytes.checked_add(growth))
        .and_then(|bytes| bytes.checked_add(extra_bytes))
        .ok_or(PagePoolError::ArithmeticOverflow)?;
    if projected > pool.inner.byte_limit {
        return Err(PagePoolError::QuotaExceeded);
    }
    Ok(())
}

fn grow_for(pool: &PagePool, state: &mut PoolState, pages: u64) -> Result<(), PagePoolError> {
    while u64::try_from(state.free.len()).map_err(|_| PagePoolError::ArithmeticOverflow)? < pages {
        add_slab(state, pool.inner.pages_per_slab)?;
        state.memory.slab_bytes = state
            .memory
            .slab_bytes
            .checked_add(slab_bytes(&pool.inner.layout, pool.inner.pages_per_slab)?)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
    }
    Ok(())
}

fn allocate_pages(state: &mut PoolState, count: u64) -> Result<Vec<PageLease>, PagePoolError> {
    let mut pages =
        Vec::with_capacity(usize::try_from(count).map_err(|_| PagePoolError::ArithmeticOverflow)?);
    for _ in 0..count {
        let index = state.free.pop().ok_or(PagePoolError::CorruptState)?;
        let page = state
            .pages
            .get_mut(usize::try_from(index).map_err(|_| PagePoolError::ArithmeticOverflow)?)
            .ok_or(PagePoolError::CorruptState)?;
        if page.lifecycle != PageLifecycle::Free || page.references != 0 {
            return Err(PagePoolError::CorruptState);
        }
        page.generation = page
            .generation
            .checked_add(1)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        page.references = 1;
        page.lifecycle = PageLifecycle::Allocated;
        pages.push(PageLease {
            index,
            generation: page.generation,
        });
    }
    Ok(pages)
}

fn live_page_mut(state: &mut PoolState, lease: PageLease) -> Result<&mut PageState, PagePoolError> {
    let page = state
        .pages
        .get_mut(usize::try_from(lease.index).map_err(|_| PagePoolError::ArithmeticOverflow)?)
        .ok_or(PagePoolError::CorruptState)?;
    if page.lifecycle != PageLifecycle::Allocated || page.generation != lease.generation {
        return Err(PagePoolError::CorruptState);
    }
    Ok(page)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, PagedPlane};

    fn pool(limit: u64) -> PagePool {
        PagePool::new(
            PagedLayout::new(
                4,
                vec![PagedPlane {
                    dtype: DType::F32,
                    elements_per_token: 2,
                    alignment: 4,
                    label: "plane".to_owned(),
                }],
            )
            .unwrap(),
            2,
            limit,
        )
        .unwrap()
    }

    #[test]
    fn quota_refusal_is_atomic() {
        let pool = pool(72);
        let before = pool.memory_usage().unwrap();
        assert!(matches!(
            pool.reserve(12),
            Err(PagePoolError::QuotaExceeded)
        ));
        assert_eq!(pool.memory_usage().unwrap(), before);
    }
}
