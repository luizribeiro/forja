use super::{LanePosition, StateStorage, journal::StateJournal};
use forja_core::{Backend, PagePoolError, StateSequence, StateTransaction};
use std::sync::{Arc, Mutex, MutexGuard};

/// Shared sequence ownership retained by every recording and completion guard.
pub struct StateSequenceHandle<B: Backend> {
    inner: Arc<Mutex<StateJournal>>,
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
    pub(super) fn order_key(&self) -> usize {
        Arc::as_ptr(&self.inner).addr()
    }

    pub(super) fn new(storage: Arc<StateStorage<B>>, sequence: StateSequence) -> Self {
        Self {
            inner: Arc::new(Mutex::new(StateJournal::new(sequence))),
            storage,
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, StateJournal> {
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
        self.lock().begin(serial, None)
    }

    /// Reserves a version and all serving cursors in one lane transaction.
    ///
    /// # Errors
    /// Refuses invalid capacity or an unavailable transaction version.
    pub fn begin_at(
        &self,
        serial: u64,
        position: LanePosition,
    ) -> Result<StateTransaction, PagePoolError> {
        self.lock().begin(serial, Some(position))
    }

    /// Returns the cursors published with the last successful state version.
    #[must_use]
    pub fn position(&self) -> LanePosition {
        self.lock().position
    }

    /// Selects the retained predecessor and all its serving cursors after drain.
    ///
    /// # Errors
    /// Refuses a missing boundary or outstanding device work.
    pub fn rewind(&self, tokens: u64) -> Result<(), PagePoolError> {
        self.lock().rewind(tokens)
    }

    /// Cancels an unsubmitted reservation and its descendants.
    ///
    /// # Errors
    /// Refuses stale capabilities and reservations already sealed for submission.
    pub fn cancel(&self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.lock().cancel_unsubmitted(transaction)
    }
}
