use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::{PagePool, PagePoolError, PageTable, PagedMemoryUsage, PagedSequence, StateLayout};

const VERSIONS: u64 = 3;

/// Allocator for opaque sequence state, including two speculative descendants.
#[derive(Clone)]
pub struct StatePool {
    layout: Arc<StateLayout>,
    pages: PagePool,
    completed: Arc<AtomicU64>,
}

impl StatePool {
    /// Reserves bounded backing for `sequences` concurrent sequences.
    ///
    /// # Errors
    /// Returns a quota, overflow or empty-capacity refusal without allocation changes.
    pub fn new(
        layout: StateLayout,
        sequences: u32,
        byte_limit: u64,
    ) -> Result<Self, PagePoolError> {
        let records = sequences
            .checked_mul(3)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let pages = PagePool::new_fixed(layout.storage_layout().clone(), records, byte_limit)?;
        Ok(Self {
            layout: Arc::new(layout),
            pages,
            completed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Validated geometry of the pool's private backing.
    #[must_use]
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Allocates all versions and their hidden allocation table atomically.
    ///
    /// # Errors
    /// Returns a quota error without a partial sequence when capacity is exhausted.
    pub fn reserve(&self) -> Result<StateSequence, PagePoolError> {
        Ok(StateSequence {
            pool: self.clone(),
            storage: self.pages.reserve(VERSIONS)?,
            identity: Arc::new(()),
            committed: 0,
            latest_serial: 0,
            latest_submission: 0,
            pending: VecDeque::new(),
            drain_until: 0,
            quarantine_until: 0,
        })
    }

    /// Observes a drained submission; failed work still retires its storage safely.
    ///
    /// # Errors
    /// Returns an error if allocator metadata is inconsistent.
    pub fn observe_completion(&self, serial: u64) -> Result<(), PagePoolError> {
        self.pages.observe_completion(serial)?;
        self.completed.fetch_max(serial, Ordering::AcqRel);
        Ok(())
    }

    /// Checks that a lane capability belongs to this pool before recording work.
    ///
    /// # Errors
    /// Returns a foreign-resource refusal without touching any backing memory.
    pub fn validate_slot(&self, slot: &StateSlot) -> Result<(), PagePoolError> {
        if !slot.active
            || slot
                .owner
                .as_ref()
                .is_some_and(|owner| Arc::ptr_eq(owner, &self.completed))
        {
            Ok(())
        } else {
            Err(PagePoolError::ForeignResource)
        }
    }

    /// Bytes charged for physical records and hidden tables, including retired records.
    ///
    /// # Errors
    /// Returns an error for poisoned allocator metadata.
    pub fn memory_usage(&self) -> Result<(PagedMemoryUsage, u64), PagePoolError> {
        self.pages.memory_usage()
    }
}

/// Unforgeable sequence ownership and its ordered state-version journal.
pub struct StateSequence {
    pool: StatePool,
    storage: PagedSequence,
    identity: Arc<()>,
    committed: u32,
    latest_serial: u64,
    latest_submission: u64,
    pending: VecDeque<StateTransaction>,
    drain_until: u64,
    quarantine_until: u64,
}

/// A capability for one out-of-place update, tied to its original sequence.
#[derive(Clone)]
pub struct StateTransaction {
    identity: Arc<()>,
    serial: u64,
    input: u32,
    output: u32,
    submitted: Arc<AtomicU64>,
    bound: Arc<AtomicBool>,
}

impl StateTransaction {
    /// Monotonic transaction serial within its owning sequence.
    #[must_use]
    pub const fn serial(&self) -> u64 {
        self.serial
    }

    /// Logical input version; physical allocation identities remain hidden.
    #[must_use]
    pub const fn input_version(&self) -> u32 {
        self.input
    }
}

impl StateSequence {
    /// Reserves an update from the newest declared predecessor.
    ///
    /// # Errors
    /// Refuses unordered serials, a third descendant, or reuse before failure drain.
    pub fn begin(&mut self, serial: u64) -> Result<StateTransaction, PagePoolError> {
        if serial <= self.latest_serial {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        if self.pending.len() == 2 || self.pool.completed.load(Ordering::Acquire) < self.drain_until
        {
            return Err(PagePoolError::BankInFlight);
        }
        let input = self.pending.back().map_or(self.committed, |tx| tx.output);
        let output = (0..3)
            .find(|&version| version != self.committed && version != input)
            .ok_or(PagePoolError::BankInFlight)?;
        let transaction = StateTransaction {
            identity: Arc::clone(&self.identity),
            serial,
            input,
            output,
            submitted: Arc::new(AtomicU64::new(0)),
            bound: Arc::new(AtomicBool::new(false)),
        };
        self.latest_serial = serial;
        self.pending.push_back(transaction.clone());
        Ok(transaction)
    }

    /// Seals a version using its transaction serial as the queue position.
    /// Use [`Self::mark_submitted_at`] when sequences have independent serials.
    ///
    /// # Errors
    /// Refuses stale, duplicate or out-of-order submissions without publishing state.
    pub fn mark_submitted(&mut self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.mark_submitted_at(transaction, transaction.serial)
    }

    /// Seals an update at a host-assigned queue position, independent of sequence serials.
    ///
    /// # Errors
    /// Refuses zero, stale, duplicate or out-of-order queue positions.
    pub fn mark_submitted_at(
        &mut self,
        transaction: &StateTransaction,
        queue_serial: u64,
    ) -> Result<(), PagePoolError> {
        if self.pool.completed.load(Ordering::Acquire) < self.quarantine_until {
            return Err(PagePoolError::BankInFlight);
        }
        let index = self.pending_index(transaction)?;
        if queue_serial <= self.latest_submission
            || transaction.submitted.load(Ordering::Acquire) != 0
            || self
                .pending
                .iter()
                .take(index)
                .any(|tx| tx.submitted.load(Ordering::Acquire) == 0)
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.storage.mark_submitted(queue_serial)?;
        self.latest_submission = queue_serial;
        transaction.submitted.store(queue_serial, Ordering::Release);
        Ok(())
    }

    /// Cancels an unsubmitted reservation and every dependent reservation.
    ///
    /// # Errors
    /// Refuses stale capabilities or cancellation after submission has been sealed.
    pub fn cancel_unsubmitted(
        &mut self,
        transaction: &StateTransaction,
    ) -> Result<(), PagePoolError> {
        let index = self.pending_index(transaction)?;
        if transaction.submitted.load(Ordering::Acquire) != 0 {
            return Err(PagePoolError::BankInFlight);
        }
        self.pending.truncate(index);
        Ok(())
    }

    /// Restores the committed journal without claiming uncertain GPU work has completed.
    /// New logical reservations are allowed; execution and backing reuse remain quarantined.
    ///
    /// # Errors
    /// Refuses foreign or already retired transaction capabilities.
    pub fn abandon(&mut self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.pending_index(transaction)?;
        self.quarantine_until = self.quarantine_until.max(self.drain_until).max(
            self.pending
                .iter()
                .map(|tx| tx.submitted.load(Ordering::Acquire))
                .max()
                .unwrap_or(0),
        );
        self.pending.clear();
        self.drain_until = 0;
        Ok(())
    }

    fn pending_index(&self, transaction: &StateTransaction) -> Result<usize, PagePoolError> {
        if !Arc::ptr_eq(&self.identity, &transaction.identity) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.pending
            .iter()
            .position(|tx| tx.serial == transaction.serial)
            .ok_or(PagePoolError::InvalidCheckpoint)
    }

    /// Publishes successful work or taints its speculative descendants after failure.
    ///
    /// # Errors
    /// Refuses foreign/out-of-order capabilities and completion before the GPU drained.
    pub fn finish(
        &mut self,
        transaction: &StateTransaction,
        success: bool,
    ) -> Result<(), PagePoolError> {
        if self.pending_index(transaction)? != 0 {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let submitted = transaction.submitted.load(Ordering::Acquire);
        if success && submitted == 0 {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        if submitted != 0 && self.pool.completed.load(Ordering::Acquire) < submitted {
            return Err(PagePoolError::BankInFlight);
        }
        if success {
            self.committed = transaction.output;
            self.pending.pop_front();
        } else {
            self.drain_until = self
                .pending
                .iter()
                .map(|tx| tx.submitted.load(Ordering::Acquire))
                .max()
                .unwrap_or(0);
            self.pending.clear();
        }
        Ok(())
    }
}

impl StateSequence {
    /// Returns reserved version identities for trusted backend initialization.
    ///
    /// # Errors
    /// Returns an error for stale or corrupted allocation metadata.
    #[doc(hidden)]
    pub fn allocation_table(&self) -> Result<PageTable, PagePoolError> {
        self.storage.table()
    }

    /// Builds the slot-bound input/output allocation table for a live transaction.
    ///
    /// # Errors
    /// Refuses foreign, retired or already bound transaction capabilities.
    pub fn bind(&self, transaction: &StateTransaction) -> Result<StateSlot, PagePoolError> {
        self.pending_index(transaction)?;
        let table = self.storage.table()?;
        let entries = [transaction.input, transaction.output]
            .iter()
            .map(|&index| table.entries()[index as usize])
            .collect();
        let slot = StateSlot {
            table: PageTable::new(entries, table.generations().to_vec())
                .map_err(|_| PagePoolError::CorruptState)?,
            active: true,
            owner: Some(Arc::clone(&self.pool.completed)),
        };
        transaction
            .bound
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PagePoolError::InvalidCheckpoint)?;
        Ok(slot)
    }
}

/// Host-only immutable lane binding; allocation IDs never become guest tensors.
#[derive(Clone, Debug)]
pub struct StateSlot {
    table: PageTable,
    active: bool,
    owner: Option<Arc<AtomicU64>>,
}

impl StateSlot {
    /// A padding lane with impossible generations and no authorized state access.
    #[must_use]
    pub fn poisoned() -> Self {
        Self {
            table: PageTable::poisoned(2),
            active: false,
            owner: None,
        }
    }

    /// Generation-authenticated input/output mappings for trusted backend upload.
    #[must_use]
    pub const fn table(&self) -> &PageTable {
        &self.table
    }

    /// Whether the slot may read or update state.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, StatePlane};

    #[test]
    fn versions_publish_in_order_and_failed_descendants_must_drain() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 2, 1024).unwrap();
        let mut a = pool.reserve().unwrap();
        let mut b = pool.reserve().unwrap();
        assert!(pool.reserve().is_err());
        let first = a.begin(1).unwrap();
        assert!(b.finish(&first, true).is_err());
        let second = a.begin(2).unwrap();
        assert!(a.mark_submitted(&second).is_err());
        a.mark_submitted(&first).unwrap();
        assert!(a.mark_submitted(&first).is_err());
        a.mark_submitted(&second).unwrap();
        assert!(a.begin(3).is_err());
        assert!(a.finish(&first, true).is_err());
        pool.observe_completion(1).unwrap();
        a.finish(&first, false).unwrap();
        assert!(a.finish(&second, true).is_err());
        assert!(a.begin(3).is_err());
        pool.observe_completion(2).unwrap();
        let retry = a.begin(3).unwrap();
        let slot = a.bind(&retry).unwrap();
        pool.validate_slot(&slot).unwrap();
        assert_ne!(
            slot.table.resolve(0).unwrap(),
            slot.table.resolve(1).unwrap()
        );
        assert!(b.bind(&retry).is_err());
        let other = StatePool::new(pool.layout().clone(), 1, 1024).unwrap();
        assert!(other.validate_slot(&slot).is_err());
        pool.validate_slot(&StateSlot::poisoned()).unwrap();
        a.mark_submitted(&retry).unwrap();
        pool.observe_completion(3).unwrap();
        a.finish(&retry, true).unwrap();
        assert!(a.finish(&retry, true).is_err());
        assert!(a.bind(&retry).is_err());
    }
    #[test]
    fn transaction_clones_cannot_bind_a_second_recording() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 1, 1024).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let tx = sequence.begin(1).unwrap();
        sequence.bind(&tx).unwrap();
        assert!(matches!(
            sequence.bind(&tx.clone()),
            Err(PagePoolError::InvalidCheckpoint)
        ));
    }

    #[test]
    fn abandoning_restores_the_checkpoint_but_quarantines_uncertain_backing() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 1, 1024).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let first = sequence.begin(1).unwrap();
        sequence.mark_submitted_at(&first, 11).unwrap();
        pool.observe_completion(11).unwrap();
        sequence.finish(&first, true).unwrap();
        let second = sequence.begin(2).unwrap();
        sequence.mark_submitted_at(&second, 12).unwrap();
        sequence.abandon(&second).unwrap();
        let retry = sequence.begin(3).unwrap();
        assert_eq!(retry.input_version(), 1);
        assert_eq!(
            sequence.mark_submitted_at(&retry, 13),
            Err(PagePoolError::BankInFlight)
        );
        drop(sequence);
        assert!(pool.reserve().is_err());
        pool.observe_completion(12).unwrap();
        assert!(pool.reserve().is_ok());
    }

    #[test]
    fn cancelling_an_unsubmitted_descendant_keeps_its_predecessor() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 1, 1024).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let first = sequence.begin(1).unwrap();
        sequence.mark_submitted(&first).unwrap();
        let second = sequence.begin(2).unwrap();
        sequence.cancel_unsubmitted(&second).unwrap();
        assert_eq!(
            sequence.cancel_unsubmitted(&first),
            Err(PagePoolError::BankInFlight)
        );
        let retry = sequence.begin(3).unwrap();
        assert_eq!(retry.input_version(), 1);
        sequence.cancel_unsubmitted(&retry).unwrap();
        pool.observe_completion(1).unwrap();
        sequence.finish(&first, true).unwrap();
    }

    #[test]
    fn independent_sequence_serials_use_distinct_queue_positions() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 2, 1024).unwrap();
        let mut a = pool.reserve().unwrap();
        let mut b = pool.reserve().unwrap();
        let first = a.begin(1).unwrap();
        let second = b.begin(1).unwrap();
        a.mark_submitted_at(&first, 11).unwrap();
        b.mark_submitted_at(&second, 12).unwrap();
        pool.observe_completion(11).unwrap();
        a.finish(&first, true).unwrap();
        assert_eq!(b.finish(&second, true), Err(PagePoolError::BankInFlight));
        pool.observe_completion(12).unwrap();
        b.finish(&second, true).unwrap();
    }

    #[test]
    fn retiring_state_is_never_reallocated_before_its_reader_drains() {
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![33]).unwrap()]).unwrap();
        assert!(StatePool::new(layout.clone(), 1, 1).is_err());
        let pool = StatePool::new(layout, 1, 4096).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let transaction = sequence.begin(7).unwrap();
        sequence.mark_submitted(&transaction).unwrap();
        let slot = sequence.bind(&transaction).unwrap();
        let access = crate::StateAccess::from_slot(&slot).unwrap();
        assert_ne!(access.source(), access.destination());
        assert!(crate::StateAccess::from_slot(&StateSlot::poisoned()).is_err());
        drop(sequence);
        assert!(pool.reserve().is_err());
        pool.observe_completion(6).unwrap();
        assert!(pool.reserve().is_err());
        pool.observe_completion(7).unwrap();
        let mut replacement = pool.reserve().unwrap();
        let next = replacement.begin(8).unwrap();
        let current = replacement.bind(&next).unwrap();
        assert!(
            PageTable::new(
                slot.table.entries().to_vec(),
                current.table.generations().to_vec()
            )
            .is_err()
        );
        assert!(replacement.bind(&transaction).is_err());
    }
    #[test]
    fn an_unsubmitted_cancellation_needs_no_synthetic_gpu_completion() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 1, 1024).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let tx = sequence.begin(10).unwrap();
        assert!(sequence.finish(&tx, true).is_err());
        sequence.finish(&tx, false).unwrap();
        assert!(sequence.mark_submitted(&tx).is_err());
        let next = sequence.begin(11).unwrap();
        sequence.finish(&next, false).unwrap();
        drop(sequence);
        assert!(pool.reserve().is_ok());
    }
}
