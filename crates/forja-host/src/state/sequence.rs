use super::StateStorage;
use forja_core::{Backend, PagePoolError, StateSequence, StateTransaction};
use std::sync::{Arc, Mutex, MutexGuard};

/// Shared sequence ownership retained by every recording and completion guard.
pub struct StateSequenceHandle<B: Backend> {
    inner: Arc<Mutex<StateSequence>>,
    pub(super) storage: Arc<StateStorage<B>>,
}

impl<B: Backend> Clone for StateSequenceHandle<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            storage: Arc::clone(&self.storage),
        }
    }
}

impl<B: Backend> StateSequenceHandle<B> {
    pub(super) fn new(storage: Arc<StateStorage<B>>, sequence: StateSequence) -> Self {
        Self {
            inner: Arc::new(Mutex::new(sequence)),
            storage,
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, StateSequence> {
        match self.inner.lock() {
            Ok(sequence) => sequence,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Reserves a transaction using a serial monotonically increasing within this sequence.
    ///
    /// # Errors
    /// Refuses stale serials, excess descendants or versions awaiting a failed submission's drain.
    pub fn begin(&self, serial: u64) -> Result<StateTransaction, PagePoolError> {
        self.lock().begin(serial)
    }

    /// Cancels an unsubmitted reservation and its descendants.
    ///
    /// # Errors
    /// Refuses stale capabilities and reservations already sealed for submission.
    pub fn cancel(&self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.lock().cancel_unsubmitted(transaction)
    }
}
