use super::{LanePosition, StateStorage, journal::StateJournal};
use forja_core::{
    Backend, BackendError, PagePoolError, StateCheckpoint, StateSequence, StateTransaction,
};
use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::{Duration, Instant},
};

/// A retained state version paired with its complete serving cursor boundary.
#[derive(Clone)]
pub struct StateLaneCheckpoint {
    pub(super) version: StateCheckpoint,
    pub(super) position: LanePosition,
    pub(super) feedback: Option<super::feedback::FeedbackBoundary>,
}

impl StateLaneCheckpoint {
    /// Returns the cursors associated with this exact state version.
    #[must_use]
    pub const fn position(&self) -> LanePosition {
        self.position
    }
}

#[derive(Default)]
struct RecordingDrain {
    count: Mutex<u32>,
    ready: Condvar,
}

pub(super) struct RecordingGuard(Arc<RecordingDrain>);

impl Drop for RecordingGuard {
    fn drop(&mut self) {
        let mut count = match self.0.count.lock() {
            Ok(count) => count,
            Err(poisoned) => poisoned.into_inner(),
        };
        *count = count.checked_sub(1).unwrap_or(u32::MAX);
        self.0.ready.notify_all();
    }
}

/// Shared sequence ownership retained by every recording and completion guard.
pub struct StateSequenceHandle<B: Backend> {
    inner: Arc<Mutex<StateJournal>>,
    drain: Arc<RecordingDrain>,
    pub(super) storage: Arc<StateStorage<B>>,
}

impl<B: Backend> Clone for StateSequenceHandle<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            drain: Arc::clone(&self.drain),
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
            drain: Arc::default(),
            storage,
        }
    }

    pub(super) fn recording_guard(&self) -> Result<RecordingGuard, BackendError> {
        let mut count = self
            .drain
            .count
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        *count = count.checked_add(1).ok_or(BackendError::AllocationFailed)?;
        Ok(RecordingGuard(Arc::clone(&self.drain)))
    }

    /// Waits until all recordings release their request leases after completion or cancellation.
    ///
    /// # Errors
    /// Returns timeout for outstanding owners or failure for an unrecorded pending transaction.
    pub fn wait_drained(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        let mut count = self
            .drain
            .count
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        while *count != 0 {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(BackendError::Timeout);
            }
            count = self
                .drain
                .ready
                .wait_timeout(count, remaining)
                .map_err(|_| BackendError::ExecutionFailed)?
                .0;
        }
        drop(count);
        let mut journal = self.lock();
        let checkpoint = journal
            .checkpoint()
            .map_err(|_| BackendError::ExecutionFailed)?;
        StateSequence::rewind(&mut journal, &checkpoint).map_err(|_| BackendError::ExecutionFailed)
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

    /// Pins the latest declared boundary, which must succeed before it can be restored.
    ///
    /// # Errors
    /// Refuses exhausted checkpoint reference counts.
    pub fn checkpoint_latest(&self) -> Result<StateLaneCheckpoint, PagePoolError> {
        self.lock().checkpoint_latest()
    }

    pub(crate) fn restore_with(
        &self,
        checkpoint: &StateLaneCheckpoint,
        action: impl FnOnce() -> Result<(), PagePoolError>,
    ) -> Result<(), PagePoolError> {
        self.lock().restore_with(checkpoint, action)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, Limits};
    use forja_core::{DType, StateLayout, StatePlane};

    #[test]
    fn lane_drain_waits_for_recording_owners_before_reusing_storage() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
                1,
            )
            .unwrap();
        let sequence = storage.reserve().unwrap();
        let first = storage
            .record(&sequence, sequence.begin(1).unwrap())
            .unwrap();
        let second = storage
            .record(&sequence, sequence.begin(2).unwrap())
            .unwrap();
        std::thread::scope(|scope| {
            let (sender, receiver) = std::sync::mpsc::channel();
            let sequence = &sequence;
            let waiter = scope.spawn(move || {
                sender
                    .send(sequence.wait_drained(Duration::from_secs(10)))
                    .unwrap();
            });
            assert!(matches!(
                receiver.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            drop(first);
            assert!(matches!(
                receiver.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ));
            drop(second);
            receiver
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            waiter.join().unwrap();
        });
        drop(sequence);
        storage.reserve().unwrap();
    }
}
