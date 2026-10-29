use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{PagePool, PagePoolError, PagedMemoryUsage, PagedSequence, StateLayout};

const VERSIONS: u64 = 3;

/// Allocator for opaque sequence state, including two speculative descendants.
#[derive(Clone)]
pub struct StatePool {
    layout: StateLayout,
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
            layout,
            pages,
            completed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Validated geometry of the pool's private backing.
    #[must_use]
    pub const fn layout(&self) -> &StateLayout {
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
            pending: VecDeque::new(),
            drain_until: 0,
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
    pending: VecDeque<StateTransaction>,
    drain_until: u64,
}

/// A capability for one out-of-place update, tied to its original sequence.
#[derive(Clone)]
pub struct StateTransaction {
    identity: Arc<()>,
    serial: u64,
    input: u32,
    output: u32,
}

impl StateTransaction {
    /// Submission serial whose completion governs publication and reuse.
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
    /// Starts an update from the newest submitted predecessor.
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
        self.storage.mark_submitted(serial)?;
        let transaction = StateTransaction {
            identity: Arc::clone(&self.identity),
            serial,
            input,
            output,
        };
        self.latest_serial = serial;
        self.pending.push_back(transaction.clone());
        Ok(transaction)
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
        if !Arc::ptr_eq(&self.identity, &transaction.identity)
            || self
                .pending
                .front()
                .is_none_or(|tx| tx.serial != transaction.serial)
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        if self.pool.completed.load(Ordering::Acquire) < transaction.serial {
            return Err(PagePoolError::BankInFlight);
        }
        if success {
            self.committed = transaction.output;
            self.pending.pop_front();
        } else {
            self.drain_until = self.latest_serial;
            self.pending.clear();
        }
        Ok(())
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
        assert!(a.begin(3).is_err());
        assert!(a.finish(&first, true).is_err());
        pool.observe_completion(1).unwrap();
        a.finish(&first, false).unwrap();
        assert!(a.finish(&second, true).is_err());
        assert!(a.begin(3).is_err());
        pool.observe_completion(2).unwrap();
        let retry = a.begin(3).unwrap();
        assert_ne!(retry.input, retry.output);
        pool.observe_completion(3).unwrap();
        a.finish(&retry, true).unwrap();
        assert!(a.finish(&retry, true).is_err());
    }
}
