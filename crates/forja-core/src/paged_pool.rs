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
    completed_serial: u64,
    memory: PagedMemoryUsage,
}

#[derive(Clone, Copy)]
struct PageState {
    generation: u32,
    references: u32,
    last_use: u64,
    lifecycle: PageLifecycle,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PageLifecycle {
    Free,
    Allocated,
    Retiring,
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
            completed_serial: 0,
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

    /// Makes pages retired by completed submissions available for reuse.
    ///
    /// # Errors
    ///
    /// Returns an error if pool state is poisoned or inconsistent.
    pub fn observe_completion(&self, serial: u64) -> Result<(), PagePoolError> {
        let mut state = self.lock()?;
        state.completed_serial = state.completed_serial.max(serial);
        let completed = state.completed_serial;
        let mut reclaimed = Vec::new();
        for (index, page) in state.pages.iter_mut().enumerate() {
            if page.lifecycle == PageLifecycle::Retiring && page.last_use <= completed {
                page.lifecycle = PageLifecycle::Free;
                reclaimed
                    .push(u32::try_from(index).map_err(|_| PagePoolError::ArithmeticOverflow)?);
            }
        }
        reclaimed.sort_unstable_by(|left, right| right.cmp(left));
        state.free.extend(reclaimed);
        Ok(())
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
            table_bytes,
            identity: Arc::new(()),
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

    fn same_pool(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn retain(&self, pages: &[PageLease]) -> Result<(), PagePoolError> {
        let mut state = self.lock()?;
        for lease in pages {
            let page = live_page_mut(&mut state, *lease)?;
            page.references = page
                .references
                .checked_add(1)
                .ok_or(PagePoolError::ArithmeticOverflow)?;
        }
        Ok(())
    }

    fn release(&self, pages: &[PageLease]) -> Result<(), PagePoolError> {
        let mut state = self.lock()?;
        for lease in pages {
            let completed = state.completed_serial;
            let page = live_page_mut(&mut state, *lease)?;
            page.references = page
                .references
                .checked_sub(1)
                .ok_or(PagePoolError::CorruptState)?;
            if page.references == 0 {
                if page.last_use <= completed {
                    page.lifecycle = PageLifecycle::Free;
                    state.free.push(lease.index);
                } else {
                    page.lifecycle = PageLifecycle::Retiring;
                }
            }
        }
        Ok(())
    }

    fn allocate_one(&self) -> Result<PageLease, PagePoolError> {
        let mut state = self.lock()?;
        preflight_pages(self, &state, 1, 0)?;
        grow_for(self, &mut state, 1)?;
        allocate_pages(&mut state, 1)?
            .pop()
            .ok_or(PagePoolError::CorruptState)
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
    table_bytes: u64,
    identity: Arc<()>,
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

    /// Marks every table page as read by a submitted command serial.
    ///
    /// # Errors
    ///
    /// Returns an error for stale internal allocator state.
    pub fn mark_submitted(&self, serial: u64) -> Result<(), PagePoolError> {
        let mut state = self.pool.lock()?;
        for lease in &self.pages {
            let page = live_page_mut(&mut state, *lease)?;
            page.last_use = page.last_use.max(serial);
        }
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

    /// Shares the committed prefix with a sibling sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the prefix exceeds the committed length or refcounts overflow.
    pub fn fork(&mut self, prefix_tokens: u64) -> Result<Self, PagePoolError> {
        if prefix_tokens > self.logical_len {
            return Err(PagePoolError::UnreservedWrite);
        }
        let page_count = pages_for(prefix_tokens, self.pool.layout().page_tokens())?;
        let count = usize::try_from(page_count).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let pages = self
            .pages
            .get(..count)
            .ok_or(PagePoolError::CorruptState)?
            .to_vec();
        let table_bytes = page_count
            .checked_mul(8)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let mut state = self.pool.lock()?;
        let projected = state
            .memory
            .total()
            .and_then(|bytes| bytes.checked_add(table_bytes))
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        if projected > self.pool.inner.byte_limit {
            return Err(PagePoolError::QuotaExceeded);
        }
        for lease in &pages {
            let page = live_page_mut(&mut state, *lease)?;
            page.references = page
                .references
                .checked_add(1)
                .ok_or(PagePoolError::ArithmeticOverflow)?;
        }
        state.memory.table_bytes = state
            .memory
            .table_bytes
            .checked_add(table_bytes)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        drop(state);
        self.immutable
            .get_mut(..count)
            .ok_or(PagePoolError::CorruptState)?
            .fill(true);
        Ok(Self {
            pool: self.pool.clone(),
            pages,
            immutable: vec![true; count],
            logical_len: prefix_tokens,
            reserved_tokens: prefix_tokens,
            table_bytes,
            identity: Arc::new(()),
        })
    }

    /// Replaces a shared tail with a fresh private page before backend copying.
    ///
    /// # Errors
    ///
    /// Returns quota or state errors atomically before changing the table.
    pub fn cow_tail(&mut self) -> Result<Option<CowPageCopy>, PagePoolError> {
        if self.logical_len == 0 {
            return Ok(None);
        }
        let page_tokens = u64::from(self.pool.layout().page_tokens());
        let tail = usize::try_from((self.logical_len - 1) / page_tokens)
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let old = *self.pages.get(tail).ok_or(PagePoolError::CorruptState)?;
        if !self.immutable.get(tail).copied().unwrap_or(true) && self.pool.refcount(old)? == 1 {
            return Ok(None);
        }
        let new = self.pool.allocate_one()?;
        self.pages[tail] = new;
        self.immutable[tail] = false;
        if let Err(error) = self.pool.release(&[old]) {
            self.pages[tail] = old;
            self.immutable[tail] = true;
            self.pool.release(&[new])?;
            return Err(error);
        }
        Ok(Some(CowPageCopy {
            source: old.index,
            destination: new.index,
            token_count: u32::try_from(self.logical_len % page_tokens)
                .map_err(|_| PagePoolError::ArithmeticOverflow)?,
        }))
    }

    /// Captures page, length, tail, and reservation state for rollback.
    ///
    /// # Errors
    ///
    /// Returns an error if retaining checkpoint references fails.
    pub fn checkpoint(&self) -> Result<SequenceCheckpoint, PagePoolError> {
        let retained_page = if self.logical_len == 0 {
            None
        } else {
            let tail = usize::try_from(
                (self.logical_len - 1) / u64::from(self.pool.layout().page_tokens()),
            )
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
            let page = *self.pages.get(tail).ok_or(PagePoolError::CorruptState)?;
            if self.immutable.get(tail).copied().unwrap_or(true) || self.pool.refcount(page)? > 1 {
                self.pool.retain(&[page])?;
                Some(page)
            } else {
                None
            }
        };
        Ok(SequenceCheckpoint {
            pool: self.pool.clone(),
            sequence: Arc::clone(&self.identity),
            pages: self.pages.clone(),
            immutable: self.immutable.clone(),
            logical_len: self.logical_len,
            reserved_tokens: self.reserved_tokens,
            retained_page,
            active: true,
        })
    }

    /// Restores a consumed checkpoint without affecting sibling sequences.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign or stale checkpoint.
    pub fn rollback(&mut self, mut checkpoint: SequenceCheckpoint) -> Result<(), PagePoolError> {
        if !self.pool.same_pool(&checkpoint.pool)
            || !Arc::ptr_eq(&self.identity, &checkpoint.sequence)
            || !checkpoint.active
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let replaced = self
            .pages
            .iter()
            .zip(&checkpoint.pages)
            .filter_map(|(current, saved)| (current != saved).then_some(*current))
            .collect::<Vec<_>>();
        if self.pages.len() != checkpoint.pages.len() {
            return Err(PagePoolError::CorruptState);
        }
        self.pool.release(&replaced)?;
        // If COW replaced the saved tail, its retained reference becomes the restored
        // sequence's reference; otherwise the checkpoint's extra retain must be released.
        if checkpoint
            .retained_page
            .is_some_and(|page| self.pages.contains(&page))
        {
            self.pool.release(&[checkpoint
                .retained_page
                .take()
                .ok_or(PagePoolError::CorruptState)?])?;
        }
        self.pages = std::mem::take(&mut checkpoint.pages);
        self.immutable = std::mem::take(&mut checkpoint.immutable);
        self.logical_len = checkpoint.logical_len;
        self.reserved_tokens = checkpoint.reserved_tokens;
        checkpoint.retained_page = None;
        checkpoint.active = false;
        Ok(())
    }
}

impl Drop for PagedSequence {
    fn drop(&mut self) {
        let _ = self.pool.release(&self.pages);
        if let Ok(mut state) = self.pool.lock() {
            state.memory.table_bytes = state.memory.table_bytes.saturating_sub(self.table_bytes);
        }
    }
}

/// A retained sequence boundary consumed by rollback.
pub struct SequenceCheckpoint {
    pool: PagePool,
    sequence: Arc<()>,
    pages: Vec<PageLease>,
    immutable: Vec<bool>,
    logical_len: u64,
    reserved_tokens: u64,
    retained_page: Option<PageLease>,
    active: bool,
}

impl Drop for SequenceCheckpoint {
    fn drop(&mut self) {
        if let Some(page) = self.retained_page {
            let _ = self.pool.release(&[page]);
        }
    }
}

/// Physical page copy required to complete a tail copy-on-write transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CowPageCopy {
    source: u32,
    destination: u32,
    token_count: u32,
}

impl CowPageCopy {
    /// Returns the source physical page.
    #[must_use]
    pub const fn source(self) -> u32 {
        self.source
    }

    /// Returns the new private physical page.
    #[must_use]
    pub const fn destination(self) -> u32 {
        self.destination
    }

    /// Returns valid prefix tokens that must be copied.
    #[must_use]
    pub const fn token_count(self) -> u32 {
        self.token_count
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
        last_use: 0,
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

    #[test]
    fn dropped_pages_are_reused() {
        let pool = pool(1_024);
        let sequence = pool.reserve(4).unwrap();
        let old = sequence.table().unwrap().resolve(0).unwrap();
        drop(sequence);

        let replacement = pool.reserve(4).unwrap();

        assert_eq!(replacement.table().unwrap().resolve(0).unwrap(), old);
    }

    #[test]
    fn in_flight_pages_are_not_reused() {
        let pool = pool(1_024);
        let sequence = pool.reserve(4).unwrap();
        let old = sequence.table().unwrap().resolve(0).unwrap();
        sequence.mark_submitted(7).unwrap();
        drop(sequence);
        let replacement = pool.reserve(4).unwrap();
        assert_ne!(replacement.table().unwrap().resolve(0).unwrap(), old);
        drop(replacement);
        pool.observe_completion(7).unwrap();
        let reused = pool.reserve(4).unwrap();
        assert_eq!(reused.table().unwrap().resolve(0).unwrap(), old);
    }

    #[test]
    fn fork_shares_prefix_until_tail_copy_on_write() {
        let pool = pool(1_024);
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();
        let sibling = original.fork(7).unwrap();
        assert_eq!(original.table().unwrap(), sibling.table().unwrap());
        assert_eq!(
            original.validate_write(7, 1),
            Err(PagePoolError::SharedTail)
        );

        let copy = original.cow_tail().unwrap().unwrap();

        assert_ne!(copy.source(), copy.destination());
        assert_eq!(copy.token_count(), 3);
        original.commit_write(1).unwrap();
        assert_ne!(original.table().unwrap(), sibling.table().unwrap());
    }

    #[test]
    fn copy_on_write_and_rollback_restore_the_shared_tail() {
        let pool = pool(1_024);
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();
        let sibling = original.fork(7).unwrap();
        let checkpoint = original.checkpoint().unwrap();
        original.cow_tail().unwrap().unwrap();
        original.commit_write(1).unwrap();

        original.rollback(checkpoint).unwrap();

        assert_eq!(original.logical_len(), 7);
        assert_eq!(original.table().unwrap(), sibling.table().unwrap());
    }

    #[test]
    fn rollback_crosses_a_cow_tail_and_private_page_boundary() {
        let pool = pool(2_048);
        let mut original = pool.reserve(12).unwrap();
        original.commit_write(7).unwrap();
        let sibling = original.fork(7).unwrap();
        let checkpoint = original.checkpoint().unwrap();
        let shared = sibling.table().unwrap();

        original.cow_tail().unwrap().unwrap();
        original.commit_write(5).unwrap();
        assert_eq!(original.logical_len(), 12);
        original.rollback(checkpoint).unwrap();

        assert_eq!(original.logical_len(), 7);
        let restored = original.table().unwrap();
        assert_eq!(restored.resolve(0), shared.resolve(0));
        assert_eq!(restored.resolve(1), shared.resolve(1));
    }
    #[test]
    fn rollback_restores_refcounts_after_interleaved_cow() {
        let layout = PagedLayout::new(
            4,
            vec![PagedPlane {
                dtype: DType::F32,
                elements_per_token: 2,
                alignment: 4,
                label: "plane".to_owned(),
            }],
        )
        .unwrap();
        let pool = PagePool::new(layout, 8, 4_096).unwrap();
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();
        let before_counts = original
            .pages
            .iter()
            .map(|&page| pool.refcount(page).unwrap())
            .collect::<Vec<_>>();
        let before_memory = pool.memory_usage().unwrap();

        let mut sibling = original.fork(7).unwrap();
        let checkpoint = original.checkpoint().unwrap();
        original.cow_tail().unwrap().unwrap();
        sibling.cow_tail().unwrap().unwrap();
        original.rollback(checkpoint).unwrap();
        drop(sibling);

        let after_counts = original
            .pages
            .iter()
            .map(|&page| pool.refcount(page).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(after_counts, before_counts);
        assert_eq!(pool.memory_usage().unwrap(), before_memory);
    }
}
