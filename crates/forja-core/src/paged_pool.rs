use std::{
    error::Error,
    fmt,
    ops::Range,
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
    /// A slot bank is still referenced by an in-flight submission.
    BankInFlight,
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
    fixed_capacity: bool,
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
    /// Creates a pool whose physical capacity is exactly one slab.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty slab, arithmetic overflow, or insufficient quota.
    pub fn new_fixed(
        layout: PagedLayout,
        pages: u32,
        byte_limit: u64,
    ) -> Result<Self, PagePoolError> {
        Self::new_with_capacity(layout, pages, byte_limit, true)
    }

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
        Self::new_with_capacity(layout, pages_per_slab, byte_limit, false)
    }

    fn new_with_capacity(
        layout: PagedLayout,
        pages_per_slab: u32,
        byte_limit: u64,
        fixed_capacity: bool,
    ) -> Result<Self, PagePoolError> {
        if pages_per_slab == 0 {
            return Err(PagePoolError::EmptySlab);
        }
        let slab_bytes = paged_slab_bytes(&layout, pages_per_slab)?;
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
                fixed_capacity,
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

    /// Atomically charges two reusable graph arenas at the largest retained arena size.
    ///
    /// # Errors
    ///
    /// Returns quota or overflow while preserving the prior charge.
    pub fn set_graph_arena_bytes(&self, maximum_arena_bytes: u64) -> Result<(), PagePoolError> {
        let bytes = maximum_arena_bytes
            .checked_mul(2)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        self.replace_aux_charge(AuxCharge::Graph, bytes)
    }

    /// Atomically charges primary high-water scratch plus the trimmed overlap slot.
    ///
    /// # Errors
    ///
    /// Returns quota or overflow while preserving the prior charge.
    pub fn set_scratch_bytes(
        &self,
        primary_bytes: u64,
        overlap_bytes: u64,
    ) -> Result<(), PagePoolError> {
        let bytes = primary_bytes
            .checked_add(overlap_bytes)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        self.replace_aux_charge(AuxCharge::Scratch, bytes)
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
            live_page_mut(&mut state, *lease)?
                .references
                .checked_add(1)
                .ok_or(PagePoolError::ArithmeticOverflow)?;
        }
        for lease in pages {
            let page = live_page_mut(&mut state, *lease)?;
            page.references = page
                .references
                .checked_add(1)
                .ok_or(PagePoolError::CorruptState)?;
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

    fn charge_slots(&self, bytes: u64) -> Result<(), PagePoolError> {
        let mut state = self.lock()?;
        let projected = state
            .memory
            .total()
            .and_then(|total| total.checked_add(bytes))
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        if projected > self.inner.byte_limit {
            return Err(PagePoolError::QuotaExceeded);
        }
        state.memory.slot_bytes = state
            .memory
            .slot_bytes
            .checked_add(bytes)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        Ok(())
    }

    fn uncharge_slots(&self, bytes: u64) {
        if let Ok(mut state) = self.lock() {
            state.memory.slot_bytes = state.memory.slot_bytes.saturating_sub(bytes);
        }
    }

    fn replace_aux_charge(&self, kind: AuxCharge, bytes: u64) -> Result<(), PagePoolError> {
        let mut state = self.lock()?;
        let old = match kind {
            AuxCharge::Graph => state.memory.graph_bytes,
            AuxCharge::Scratch => state.memory.scratch_bytes,
        };
        let projected = state
            .memory
            .total()
            .and_then(|total| total.checked_sub(old))
            .and_then(|total| total.checked_add(bytes))
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        if projected > self.inner.byte_limit {
            return Err(PagePoolError::QuotaExceeded);
        }
        match kind {
            AuxCharge::Graph => state.memory.graph_bytes = bytes,
            AuxCharge::Scratch => state.memory.scratch_bytes = bytes,
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum AuxCharge {
    Graph,
    Scratch,
}

/// One immutable lane snapshot read from a rebindable slot bank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedSlot {
    table: PageTable,
    logical_len: u64,
    append_start: u64,
    active: bool,
}

impl PagedSlot {
    /// Returns the fixed-size table, including poisoned padding entries.
    #[must_use]
    pub const fn table(&self) -> &PageTable {
        &self.table
    }

    /// Returns the lane's logical key length.
    #[must_use]
    pub const fn logical_len(&self) -> u64 {
        self.logical_len
    }

    /// Returns the tokens appended since the lane's last committed boundary.
    #[must_use]
    pub const fn append_window(&self) -> Range<u64> {
        self.append_start..self.logical_len
    }

    /// Reports whether this lane has a bound sequence capability.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

/// Exactly two globally rebindable banks of fixed-size opaque lane slots.
pub struct PagedSlotBanks {
    pool: PagePool,
    table_pages: usize,
    banks: [Mutex<Vec<PagedSlot>>; 2],
    submitted: Mutex<[Option<u64>; 2]>,
    charged_bytes: u64,
}

impl PagedSlotBanks {
    /// Allocates and charges two banks initialized with poisoned inactive lanes.
    ///
    /// # Errors
    ///
    /// Returns quota or overflow before changing pool accounting.
    pub fn new(pool: PagePool, lanes: usize, table_pages: usize) -> Result<Self, PagePoolError> {
        if lanes == 0 || table_pages == 0 {
            return Err(PagePoolError::UnreservedWrite);
        }
        let lane_count = u64::try_from(lanes).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let table_page_count =
            u64::try_from(table_pages).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let lane_metadata_bytes = lane_count
            .checked_mul(16)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let charged_bytes = lane_count
            .checked_mul(table_page_count)
            .and_then(|entries| entries.checked_mul(8))
            .and_then(|one_bank| one_bank.checked_add(lane_metadata_bytes))
            .and_then(|one_bank| one_bank.checked_mul(2))
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        pool.charge_slots(charged_bytes)?;
        let lane = PagedSlot {
            table: PageTable::poisoned(table_pages),
            logical_len: 0,
            append_start: 0,
            active: false,
        };
        let first = vec![lane.clone(); lanes];
        let second = vec![lane; lanes];
        Ok(Self {
            pool,
            table_pages,
            banks: [Mutex::new(first), Mutex::new(second)],
            submitted: Mutex::new([None, None]),
            charged_bytes,
        })
    }

    /// Reports whether a sequence belongs to this slot bank's pool.
    #[must_use]
    pub fn owns_sequence(&self, sequence: &PagedSequence) -> bool {
        self.pool.same_pool(&sequence.pool)
    }

    /// Rebinds an active sequence capability into one lane.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign sequence, invalid bank/lane, oversized table, or bank whose
    /// prior submission has not completed.
    pub fn bind(
        &self,
        bank: usize,
        lane: usize,
        sequence: &PagedSequence,
    ) -> Result<(), PagePoolError> {
        if !self.owns_sequence(sequence) {
            return Err(PagePoolError::ForeignResource);
        }
        self.ensure_rebindable(bank)?;
        let table = sequence
            .table()?
            .pad_with_poison(self.table_pages)
            .map_err(|_| PagePoolError::UnreservedWrite)?;
        let mut lanes = self
            .banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        *lanes.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = PagedSlot {
            table,
            logical_len: sequence.logical_len,
            append_start: sequence.logical_len,
            active: true,
        };
        Ok(())
    }

    /// Atomically rebinds `sequence` into `lane` in both banks.
    ///
    /// # Errors
    ///
    /// Returns the prior slots for rollback, or the same refusal as [`Self::bind`] before either
    /// bank is mutated.
    pub fn bind_all(
        &self,
        lane: usize,
        sequence: &PagedSequence,
    ) -> Result<([PagedSlot; 2], PagedSlot), PagePoolError> {
        if !self.owns_sequence(sequence) {
            return Err(PagePoolError::ForeignResource);
        }
        let table = sequence
            .table()?
            .pad_with_poison(self.table_pages)
            .map_err(|_| PagePoolError::UnreservedWrite)?;
        for bank in 0..self.banks.len() {
            self.validate_rebindable(bank, lane)?;
        }
        let mut first = self.banks[0]
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let mut second = self.banks[1]
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let old = [
            first
                .get(lane)
                .cloned()
                .ok_or(PagePoolError::ForeignResource)?,
            second
                .get(lane)
                .cloned()
                .ok_or(PagePoolError::ForeignResource)?,
        ];
        let replacement = PagedSlot {
            table,
            logical_len: sequence.logical_len,
            append_start: sequence.logical_len,
            active: true,
        };
        *first.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = replacement.clone();
        *second.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = replacement.clone();
        Ok((old, replacement))
    }

    /// Restores slots returned by [`Self::bind_all`] before the rebound lane can be submitted.
    ///
    /// # Errors
    ///
    /// Returns an error if either bank or lane is unavailable.
    pub fn restore_all(&self, lane: usize, slots: [PagedSlot; 2]) -> Result<(), PagePoolError> {
        let [first_slot, second_slot] = slots;
        let mut first = self.banks[0]
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let mut second = self.banks[1]
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        *first.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = first_slot;
        *second.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = second_slot;
        Ok(())
    }

    /// Replaces one lane with a poisoned inactive slot.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or still-in-flight bank/lane.
    pub fn poison(&self, bank: usize, lane: usize) -> Result<(), PagePoolError> {
        self.ensure_rebindable(bank)?;
        let mut lanes = self
            .banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        *lanes.get_mut(lane).ok_or(PagePoolError::ForeignResource)? = PagedSlot {
            table: PageTable::poisoned(self.table_pages),
            logical_len: 0,
            append_start: 0,
            active: false,
        };
        Ok(())
    }

    /// Returns a lane snapshot for trusted backend upload.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid bank/lane or poisoned lock.
    pub fn slot(&self, bank: usize, lane: usize) -> Result<PagedSlot, PagePoolError> {
        self.banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .get(lane)
            .cloned()
            .ok_or(PagePoolError::ForeignResource)
    }

    /// Opens the append window `start..logical_len` without changing the bound page table.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid bank, lane, inactive slot, or a window that ends before
    /// it starts.
    pub fn open_append(
        &self,
        bank: usize,
        lane: usize,
        start: u64,
        logical_len: u64,
    ) -> Result<(), PagePoolError> {
        if start > logical_len {
            return Err(PagePoolError::UnreservedWrite);
        }
        let mut lanes = self
            .banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let slot = lanes.get_mut(lane).ok_or(PagePoolError::ForeignResource)?;
        if !slot.active {
            return Err(PagePoolError::ForeignResource);
        }
        slot.logical_len = logical_len;
        slot.append_start = start;
        Ok(())
    }

    /// Returns an active lane's append window without copying its page table.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid bank, lane, or inactive slot.
    pub fn append_window(&self, bank: usize, lane: usize) -> Result<Range<u64>, PagePoolError> {
        let lanes = self
            .banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let slot = lanes.get(lane).ok_or(PagePoolError::ForeignResource)?;
        if !slot.active {
            return Err(PagePoolError::ForeignResource);
        }
        Ok(slot.append_window())
    }

    /// Marks one bank unavailable until its submission completion is observed.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid bank or poisoned lock.
    pub fn mark_submitted(&self, bank: usize, serial: u64) -> Result<(), PagePoolError> {
        let mut submitted = self
            .submitted
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        *submitted
            .get_mut(bank)
            .ok_or(PagePoolError::ForeignResource)? = Some(serial);
        Ok(())
    }

    fn ensure_rebindable(&self, bank: usize) -> Result<(), PagePoolError> {
        let completed = self.pool.lock()?.completed_serial;
        let mut submitted = self
            .submitted
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        let pending = submitted
            .get_mut(bank)
            .ok_or(PagePoolError::ForeignResource)?;
        if pending.is_some_and(|serial| serial > completed) {
            return Err(PagePoolError::BankInFlight);
        }
        *pending = None;
        Ok(())
    }

    fn validate_rebindable(&self, bank: usize, lane: usize) -> Result<(), PagePoolError> {
        self.banks
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .get(lane)
            .ok_or(PagePoolError::ForeignResource)?;
        let completed = self.pool.lock()?.completed_serial;
        let submitted = self
            .submitted
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?;
        if submitted
            .get(bank)
            .ok_or(PagePoolError::ForeignResource)?
            .is_some_and(|serial| serial > completed)
        {
            return Err(PagePoolError::BankInFlight);
        }
        Ok(())
    }
}

impl Drop for PagedSlotBanks {
    fn drop(&mut self) {
        self.pool.uncharge_slots(self.charged_bytes);
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
        if start != self.logical_len {
            return Err(PagePoolError::UnreservedWrite);
        }
        self.validate_exclusive(start, end)
    }

    /// Validates a pending write that must stay inside a lane's open append window.
    ///
    /// # Errors
    ///
    /// Returns unreserved-write or shared-tail errors without changing sequence state.
    pub fn validate_append(
        &self,
        window: &Range<u64>,
        start: u64,
        tokens: u64,
    ) -> Result<(), PagePoolError> {
        let end = start
            .checked_add(tokens)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        if start < window.start || end > window.end {
            return Err(PagePoolError::UnreservedWrite);
        }
        self.validate_exclusive(start, end)
    }

    fn validate_exclusive(&self, start: u64, end: u64) -> Result<(), PagePoolError> {
        if end > self.reserved_tokens {
            return Err(PagePoolError::UnreservedWrite);
        }
        if end == start {
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

    /// Moves an exclusively owned sequence to an earlier logical boundary.
    ///
    /// # Errors
    ///
    /// Returns an invalid-checkpoint or shared-tail error without changing sequence state.
    pub fn rewind_private(&mut self, logical_len: u64) -> Result<(), PagePoolError> {
        if logical_len >= self.logical_len {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let page_tokens = u64::from(self.pool.layout().page_tokens());
        let first = usize::try_from(logical_len.div_ceil(page_tokens))
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let last = usize::try_from(self.logical_len.div_ceil(page_tokens))
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        for (&page, &immutable) in self.pages[first..last]
            .iter()
            .zip(&self.immutable[first..last])
        {
            if immutable || self.pool.refcount(page)? != 1 {
                return Err(PagePoolError::SharedTail);
            }
        }
        self.logical_len = logical_len;
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

    /// Returns the allocator reference count for each page in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error for stale internal allocator state.
    pub fn page_reference_counts(&self) -> Result<Vec<u32>, PagePoolError> {
        let mut state = self.pool.lock()?;
        self.pages
            .iter()
            .map(|&page| Ok(live_page_mut(&mut state, page)?.references))
            .collect()
    }

    /// Shares the committed prefix with a sibling sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the prefix exceeds the committed length or refcounts overflow.
    pub fn fork(&mut self, prefix_tokens: u64) -> Result<Self, PagePoolError> {
        self.fork_reserved(prefix_tokens, prefix_tokens)
    }

    /// Shares a committed prefix while atomically reserving private suffix capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when lengths are invalid, quota is insufficient, or refcounts overflow.
    pub fn fork_reserved(
        &mut self,
        prefix_tokens: u64,
        reserved_tokens: u64,
    ) -> Result<Self, PagePoolError> {
        if prefix_tokens > self.logical_len {
            return Err(PagePoolError::UnreservedWrite);
        }
        if reserved_tokens < prefix_tokens {
            return Err(PagePoolError::UnreservedWrite);
        }
        let shared_pages = pages_for(prefix_tokens, self.pool.layout().page_tokens())?;
        let reserved_pages = pages_for(reserved_tokens, self.pool.layout().page_tokens())?;
        let private_pages = reserved_pages
            .checked_sub(shared_pages)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let shared_count =
            usize::try_from(shared_pages).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let mut pages = self
            .pages
            .get(..shared_count)
            .ok_or(PagePoolError::CorruptState)?
            .to_vec();
        let table_bytes = reserved_pages
            .checked_mul(8)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        {
            let state = self.pool.lock()?;
            preflight_pages(&self.pool, &state, private_pages, table_bytes)?;
        }
        self.pool.retain(&pages)?;
        let allocated = (|| {
            let mut state = self.pool.lock()?;
            preflight_pages(&self.pool, &state, private_pages, table_bytes)?;
            let table_bytes = state
                .memory
                .table_bytes
                .checked_add(table_bytes)
                .ok_or(PagePoolError::ArithmeticOverflow)?;
            grow_for(&self.pool, &mut state, private_pages)?;
            let private = allocate_pages(&mut state, private_pages)?;
            state.memory.table_bytes = table_bytes;
            Ok(private)
        })();
        let private = match allocated {
            Ok(private) => private,
            Err(error) => {
                self.pool.release(&pages)?;
                return Err(error);
            }
        };
        pages.extend(private);
        self.immutable
            .get_mut(..shared_count)
            .ok_or(PagePoolError::CorruptState)?
            .fill(true);
        let reserved_count =
            usize::try_from(reserved_pages).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let mut immutable = vec![false; reserved_count];
        immutable[..shared_count].fill(true);
        Ok(Self {
            pool: self.pool.clone(),
            pages,
            immutable,
            logical_len: prefix_tokens,
            reserved_tokens,
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
        if self.logical_len.is_multiple_of(page_tokens) {
            return Ok(None);
        }
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
        let tail = if self.logical_len == 0 {
            None
        } else {
            let tail = usize::try_from(
                (self.logical_len - 1) / u64::from(self.pool.layout().page_tokens()),
            )
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
            let page = *self.pages.get(tail).ok_or(PagePoolError::CorruptState)?;
            let immutable = self.immutable.get(tail).copied().unwrap_or(true);
            let retained = immutable || self.pool.refcount(page)? > 1;
            if retained {
                self.pool.retain(&[page])?;
            }
            Some(CheckpointTail {
                index: tail,
                page,
                immutable,
                retained,
            })
        };
        Ok(SequenceCheckpoint {
            pool: self.pool.clone(),
            sequence: Arc::clone(&self.identity),
            logical_len: self.logical_len,
            tail,
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
        if let Some(tail) = checkpoint.tail.as_mut() {
            let current = self
                .pages
                .get_mut(tail.index)
                .ok_or(PagePoolError::CorruptState)?;
            let immutable = self
                .immutable
                .get_mut(tail.index)
                .ok_or(PagePoolError::CorruptState)?;
            if *current == tail.page {
                if tail.retained {
                    self.pool.release(&[tail.page])?;
                }
            } else {
                self.pool.release(&[*current])?;
                *current = tail.page;
                tail.retained = false;
            }
            *immutable = tail.immutable;
        }
        self.logical_len = checkpoint.logical_len;
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
    logical_len: u64,
    tail: Option<CheckpointTail>,
    active: bool,
}

struct CheckpointTail {
    index: usize,
    page: PageLease,
    immutable: bool,
    retained: bool,
}

impl Drop for SequenceCheckpoint {
    fn drop(&mut self) {
        if let Some(tail) = &self.tail
            && tail.retained
        {
            let _ = self.pool.release(&[tail.page]);
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

/// Returns the charged bytes for a slab with the requested number of pages.
///
/// # Errors
///
/// Returns an error when the byte calculation overflows.
pub fn paged_slab_bytes(layout: &PagedLayout, pages_per_slab: u32) -> Result<u64, PagePoolError> {
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
    if missing > 0 && pool.inner.fixed_capacity {
        return Err(PagePoolError::QuotaExceeded);
    }
    let slabs = missing.div_ceil(u64::from(pool.inner.pages_per_slab));
    let growth = paged_slab_bytes(&pool.inner.layout, pool.inner.pages_per_slab)?
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
            .checked_add(paged_slab_bytes(
                &pool.inner.layout,
                pool.inner.pages_per_slab,
            )?)
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
    fn fixed_pool_refuses_to_grow_past_its_backing() {
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
        let pool = PagePool::new_fixed(layout, 2, 1_024).unwrap();
        let _resident = pool.reserve(8).unwrap();
        let before = pool.memory_usage().unwrap();

        assert!(matches!(pool.reserve(4), Err(PagePoolError::QuotaExceeded)));
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
    fn reserved_fork_shares_prefix_and_owns_suffix_pages() {
        let pool = pool(2_048);
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();

        let mut sibling = original.fork_reserved(7, 12).unwrap();

        assert_eq!(sibling.logical_len(), 7);
        assert_eq!(sibling.reserved_tokens(), 12);
        assert_eq!(sibling.validate_write(7, 1), Err(PagePoolError::SharedTail));
        sibling.cow_tail().unwrap().unwrap();
        sibling.commit_write(5).unwrap();
        assert_eq!(sibling.logical_len(), 12);
        assert_eq!(original.logical_len(), 7);
    }

    #[test]
    fn refused_reserved_fork_preserves_memory_and_references() {
        let pool = pool(100);
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();
        let before_memory = pool.memory_usage().unwrap();
        let before_references = original.page_reference_counts().unwrap();

        assert!(matches!(
            original.fork_reserved(7, 12),
            Err(PagePoolError::QuotaExceeded)
        ));

        assert_eq!(pool.memory_usage().unwrap(), before_memory);
        assert_eq!(original.page_reference_counts().unwrap(), before_references);
    }

    #[test]
    fn page_aligned_fork_appends_without_copying_the_full_tail() {
        let pool = pool(1_024);
        let mut original = pool.reserve(12).unwrap();
        original.commit_write(8).unwrap();
        let mut sibling = original.fork_reserved(8, 12).unwrap();

        assert!(sibling.cow_tail().unwrap().is_none());
        sibling.commit_write(1).unwrap();
        assert_eq!(sibling.logical_len(), 9);
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

    #[test]
    fn private_rewind_preserves_reserved_capacity() {
        let pool = pool(1_024);
        let mut sequence = pool.reserve(12).unwrap();
        sequence.commit_write(9).unwrap();

        sequence.rewind_private(3).unwrap();
        sequence.commit_write(9).unwrap();

        assert_eq!(sequence.logical_len(), 12);
        assert_eq!(sequence.reserved_tokens(), 12);
    }

    #[test]
    fn private_rewind_allows_an_untouched_shared_prefix() {
        let pool = pool(2_048);
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(4).unwrap();
        let mut sibling = original.fork_reserved(4, 12).unwrap();
        sibling.commit_write(5).unwrap();

        sibling.rewind_private(7).unwrap();

        assert_eq!(sibling.logical_len(), 7);
        assert_eq!(original.logical_len(), 4);
    }

    #[test]
    fn inactive_slots_stay_poisoned_across_bank_rebinding() {
        let pool = pool(4_096);
        let mut sequence = pool.reserve(8).unwrap();
        sequence.commit_write(4).unwrap();
        let banks = PagedSlotBanks::new(pool.clone(), 2, 2).unwrap();
        banks.bind(0, 0, &sequence).unwrap();
        assert!(banks.slot(0, 0).unwrap().is_active());
        banks.open_append(0, 0, 3, 4).unwrap();
        assert_eq!(banks.slot(0, 0).unwrap().logical_len(), 4);
        let inactive = banks.slot(0, 1).unwrap();
        assert!(!inactive.is_active());
        assert!(inactive.table().resolve(0).is_err());
        banks.mark_submitted(0, 9).unwrap();
        assert_eq!(banks.poison(0, 0), Err(PagePoolError::BankInFlight));
        banks.bind(1, 1, &sequence).unwrap();
        pool.observe_completion(9).unwrap();
        banks.poison(0, 0).unwrap();
        assert!(banks.slot(0, 0).unwrap().table().resolve(0).is_err());
        assert!(banks.slot(1, 1).unwrap().is_active());
    }

    #[test]
    fn slot_append_window_opens_and_closes_on_rebind() {
        let pool = pool(4_096);
        let mut sequence = pool.reserve(8).unwrap();
        sequence.commit_write(4).unwrap();
        let banks = PagedSlotBanks::new(pool.clone(), 2, 2).unwrap();
        banks.bind(0, 0, &sequence).unwrap();
        assert_eq!(banks.append_window(0, 0), Ok(4..4));

        sequence.commit_write(3).unwrap();
        banks.open_append(0, 0, 4, 7).unwrap();
        assert_eq!(banks.append_window(0, 0), Ok(4..7));
        assert_eq!(banks.slot(0, 0).unwrap().append_window(), 4..7);
        assert_eq!(
            banks.open_append(0, 0, 8, 7),
            Err(PagePoolError::UnreservedWrite)
        );
        assert_eq!(
            banks.append_window(0, 1),
            Err(PagePoolError::ForeignResource)
        );

        banks.bind(0, 0, &sequence).unwrap();
        assert_eq!(banks.append_window(0, 0), Ok(7..7));
    }

    #[test]
    fn append_validation_admits_only_the_private_open_window() {
        let pool = pool(4_096);
        let mut sequence = pool.reserve(8).unwrap();
        sequence.commit_write(7).unwrap();
        let window = 5..7;

        assert_eq!(sequence.validate_append(&window, 5, 2), Ok(()));
        assert_eq!(sequence.validate_append(&window, 6, 1), Ok(()));
        assert_eq!(
            sequence.validate_append(&window, 4, 1),
            Err(PagePoolError::UnreservedWrite)
        );
        assert_eq!(
            sequence.validate_append(&window, 6, 2),
            Err(PagePoolError::UnreservedWrite)
        );
        assert_eq!(
            sequence.validate_append(&(5..9), 7, 2),
            Err(PagePoolError::UnreservedWrite)
        );
        assert_eq!(
            sequence.validate_append(&window, u64::MAX, 2),
            Err(PagePoolError::ArithmeticOverflow)
        );

        let _published = sequence.fork(7).unwrap();
        assert_eq!(
            sequence.validate_append(&window, 5, 2),
            Err(PagePoolError::SharedTail)
        );
    }

    #[test]
    fn slots_refuse_sequences_from_another_pool() {
        let primary = pool(4_096);
        let foreign = pool(4_096).reserve(4).unwrap();
        let banks = PagedSlotBanks::new(primary, 1, 1).unwrap();

        assert_eq!(
            banks.bind(0, 0, &foreign),
            Err(PagePoolError::ForeignResource)
        );
    }

    #[test]
    fn graph_and_scratch_charges_never_raise_the_quota() {
        let pool = pool(512);
        pool.set_graph_arena_bytes(32).unwrap();
        pool.set_scratch_bytes(40, 8).unwrap();
        let accepted = pool.memory_usage().unwrap().0;
        assert_eq!(accepted.graph_bytes, 64);
        assert_eq!(accepted.scratch_bytes, 48);
        assert_eq!(
            pool.set_graph_arena_bytes(512),
            Err(PagePoolError::QuotaExceeded)
        );
        assert_eq!(pool.memory_usage().unwrap().0, accepted);
        assert_eq!(pool.memory_usage().unwrap().1, 512);
    }
}
