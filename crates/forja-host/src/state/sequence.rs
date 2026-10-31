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
    pub(super) canonical: super::canonical::CanonicalPosition,
    pub(super) position: LanePosition,
    pub(super) feedback: Option<super::feedback::FeedbackBoundary>,
    pub(super) phase: super::journal::SelectionPhase,
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
    pub(super) snapshot: Arc<Mutex<Option<Arc<super::StateSnapshot<B>>>>>,
    drain: Arc<RecordingDrain>,
    pub(super) storage: Arc<StateStorage<B>>,
}

impl<B: Backend> Clone for StateSequenceHandle<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            snapshot: Arc::clone(&self.snapshot),
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
            snapshot: Arc::default(),
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

    /// Reserves state and the requested selection boundary in one construction operation.
    ///
    /// # Errors
    /// Refuses invalid capacity or an unavailable version.
    pub fn begin_action_at(
        &self,
        serial: u64,
        position: LanePosition,
        action: super::StateAction,
    ) -> Result<StateTransaction, PagePoolError> {
        self.lock()
            .begin_action(serial, Some(position), Some(action))
    }

    /// Whether this sequence still represents its initial, unmodified state.
    ///
    /// # Errors
    /// Refuses outstanding state work.
    pub fn has_pristine_state(&self) -> Result<bool, PagePoolError> {
        self.lock().has_pristine_state()
    }

    /// Reserves canonically evaluated prompt blocks before the prompt's final token.
    ///
    /// # Errors
    /// Refuses foreign ledgers, changed prefixes, generated state or unavailable versions.
    pub fn begin_prefill(
        &self,
        serial: u64,
        prompt: Arc<super::CanonicalPrompt>,
        end: u64,
        reserved_tokens: u64,
        action: super::StateAction,
    ) -> Result<StateTransaction, PagePoolError> {
        if !Arc::ptr_eq(&prompt.charge.live, &self.storage.charge.live)
            || !prompt.namespace.matches_state(self.storage.pool.layout())
        {
            return Err(PagePoolError::ForeignResource);
        }
        self.lock()
            .begin_prefill(serial, prompt, end, reserved_tokens, action)
    }

    /// Returns the completed canonical boundary, refusing work still in flight.
    ///
    /// # Errors
    /// Refuses pending transactions or selection guards and undrained device work.
    pub fn canonical_token_count(&self) -> Result<Option<u64>, PagePoolError> {
        self.lock()
            .canonical_boundary()
            .map(|boundary| boundary.map(|(_, end)| end))
    }

    /// Returns the newest declared cursor boundary, including a pending forward operation.
    ///
    /// # Errors
    /// Refuses exhausted checkpoint reference counts.
    pub fn latest_position(&self) -> Result<LanePosition, PagePoolError> {
        Ok(self.lock().checkpoint_latest()?.position)
    }

    /// Returns the cursors published with the last successful state version.
    #[must_use]
    pub fn position(&self) -> LanePosition {
        self.lock().position
    }

    /// Pins the latest declared boundary, which must succeed before it can be restored.
    ///
    /// # Errors
    /// Refuses exhausted checkpoint counts or a forward boundary awaiting selection.
    pub fn checkpoint_latest(&self) -> Result<StateLaneCheckpoint, PagePoolError> {
        let checkpoint = self.lock().checkpoint_latest()?;
        if checkpoint.phase == super::journal::SelectionPhase::Awaiting {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        Ok(checkpoint)
    }

    pub(crate) fn restore_with(
        &self,
        checkpoint: &StateLaneCheckpoint,
        action: impl FnOnce() -> Result<(), PagePoolError>,
    ) -> Result<(), crate::recovery::RecoveryError<PagePoolError>> {
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
    #[test]
    fn forward_selection_phase_preserves_consumer_cursors_until_selection() {
        use crate::StateAction;
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
        let first = sequence
            .begin_action_at(
                1,
                LanePosition {
                    tokens: 64,
                    reserved_tokens: 129,
                    ..LanePosition::default()
                },
                StateAction::Combined,
            )
            .unwrap();
        let selected = sequence.checkpoint_latest().unwrap();
        assert_eq!(
            (
                selected.position.sampler,
                selected.position.grammar,
                selected.position.parser
            ),
            (63, 64, 64)
        );
        sequence.lock().mark_submitted(&first).unwrap();
        storage.pool.observe_completion(1).unwrap();
        sequence.lock().finish(&first, true).unwrap();
        let forward = sequence
            .begin_action_at(
                2,
                LanePosition {
                    tokens: 65,
                    reserved_tokens: 129,
                    ..LanePosition::default()
                },
                StateAction::Forward,
            )
            .unwrap();
        assert!(sequence.checkpoint_latest().is_err());
        assert_eq!(sequence.latest_position().unwrap().sampler, 63);
        sequence.lock().mark_submitted(&forward).unwrap();
        storage.pool.observe_completion(2).unwrap();
        sequence.lock().finish(&forward, true).unwrap();
        assert!(sequence.checkpoint_latest().is_err());
        assert_eq!(
            (sequence.position().tokens, sequence.position().parser),
            (65, 64)
        );
        sequence.rewind(64).unwrap();
        assert_eq!(
            sequence.checkpoint_latest().unwrap().position,
            selected.position
        );
        let step = sequence
            .begin_action_at(
                3,
                LanePosition {
                    tokens: 65,
                    reserved_tokens: 129,
                    ..LanePosition::default()
                },
                StateAction::Step,
            )
            .unwrap();
        assert_eq!(sequence.latest_position().unwrap().grammar, 64);
        sequence.cancel(&step).unwrap();
    }
}
