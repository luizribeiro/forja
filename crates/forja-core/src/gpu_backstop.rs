//! Quarantine of GPU work signatures after prediction failures.

use crate::BackendError;
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    time::Duration,
};

struct Fault {
    events: u32,
    uncertain: u32,
    confirmed: bool,
    error: BackendError,
}
struct State<K> {
    generation: u64,
    faults: HashMap<K, Fault>,
}

/// Load-local prediction evidence, isolated by a host-defined composition identity.
pub struct GpuWorkTracker<K> {
    state: Mutex<State<K>>,
}
impl<K> Default for GpuWorkTracker<K> {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                generation: 0,
                faults: HashMap::new(),
            }),
        }
    }
}
impl<K: Clone + Eq + Hash> GpuWorkTracker<K> {
    /// Clears quarantine only at an explicit host load boundary.
    /// # Errors
    /// Returns failure for poisoned state or an exhausted generation counter.
    pub fn begin_load(&self) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        state.generation = state
            .generation
            .checked_add(1)
            .ok_or(BackendError::QuotaExceeded)?;
        state.faults.clear();
        Ok(())
    }
    /// Checks for three failures of one signature or exhaustion of bounded evidence storage.
    /// # Errors
    /// Returns a quota error for suspension, or a state synchronization failure.
    pub fn check_backend(&self) -> Result<(), BackendError> {
        let state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if Self::suspended(&state) {
            Err(BackendError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    fn suspended(state: &State<K>) -> bool {
        state.faults.len() >= 256 || state.faults.values().any(|fault| fault.events >= 3)
    }
    fn check_scope(state: &State<K>, scope: &K) -> Result<(), BackendError> {
        if Self::suspended(state) {
            return Err(BackendError::QuotaExceeded);
        }
        if let Some(fault) = state.faults.get(scope)
            && (fault.confirmed || fault.uncertain >= 1)
        {
            return Err(fault.error);
        }
        Ok(())
    }
    /// Creates an observation ticket without resetting previous evidence.
    /// # Errors
    /// Refuses invalid bounds, quarantined work or a suspended load.
    pub fn track(
        self: &Arc<Self>,
        scope: K,
        prediction: Duration,
        ratio: u32,
    ) -> Result<GpuWorkTicket<K>, BackendError> {
        let limit = prediction
            .checked_mul(ratio)
            .filter(|_| ratio > 0 && !prediction.is_zero())
            .ok_or(BackendError::InvalidInput)?;
        let state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        Self::check_scope(&state, &scope)?;
        Ok(GpuWorkTicket {
            tracker: Arc::clone(self),
            generation: state.generation,
            scope,
            prediction,
            ratio,
            limit,
            observation: AtomicU8::new(0),
            failed: AtomicBool::new(false),
        })
    }
    fn record(
        &self,
        generation: u64,
        scope: K,
        error: BackendError,
        confirmed: bool,
        first: bool,
    ) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if state.generation != generation {
            return Ok(());
        }
        if state.faults.len() >= 256 && !state.faults.contains_key(&scope) {
            return Err(BackendError::QuotaExceeded);
        }
        let fault = state.faults.entry(scope).or_insert(Fault {
            events: 0,
            uncertain: 0,
            confirmed: false,
            error,
        });
        if first {
            fault.events = fault
                .events
                .checked_add(1)
                .ok_or(BackendError::QuotaExceeded)?;
        }
        if confirmed {
            fault.confirmed = true;
            if !first {
                fault.uncertain = fault
                    .uncertain
                    .checked_sub(1)
                    .ok_or(BackendError::ExecutionFailed)?;
            }
        } else if first {
            fault.uncertain = fault
                .uncertain
                .checked_add(1)
                .ok_or(BackendError::QuotaExceeded)?;
        }
        fault.error = error;
        Ok(())
    }
    fn retract(&self, generation: u64, scope: &K) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if state.generation != generation {
            return Ok(());
        }
        if let Some(fault) = state.faults.get_mut(scope)
            && fault.uncertain > 0
        {
            fault.uncertain -= 1;
            let remove = fault.uncertain == 0 && !fault.confirmed;
            fault.events = fault
                .events
                .checked_sub(1)
                .ok_or(BackendError::ExecutionFailed)?;
            if remove {
                state.faults.remove(scope);
            }
        }
        Ok(())
    }
}

/// One command buffer's prediction and at most one independent failure event.
pub struct GpuWorkTicket<K> {
    tracker: Arc<GpuWorkTracker<K>>,
    generation: u64,
    scope: K,
    prediction: Duration,
    ratio: u32,
    limit: Duration,
    observation: AtomicU8,
    failed: AtomicBool,
}
impl<K: Clone + Eq + Hash> GpuWorkTicket<K> {
    /// Returns the admitted forecast.
    #[must_use]
    pub const fn prediction(&self) -> Duration {
        self.prediction
    }
    /// Returns the measured-to-predicted limit.
    #[must_use]
    pub const fn ratio(&self) -> u32 {
        self.ratio
    }
    /// Returns whether completed work lacked a trustworthy prediction match.
    #[must_use]
    pub fn prediction_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
    /// Checks quarantine again immediately before committing prepared work.
    /// # Errors
    /// Refuses quarantined work or a suspended load.
    pub fn check_admission(&self) -> Result<(), BackendError> {
        let state = self
            .tracker
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if state.generation != self.generation {
            return Err(BackendError::InvalidInput);
        }
        GpuWorkTracker::check_scope(&state, &self.scope)
    }
    fn error(&self, measured: Option<Duration>) -> BackendError {
        BackendError::GpuPredictionExceeded {
            predicted: self.prediction,
            measured,
            ratio: self.ratio,
        }
    }
    /// Records a timeout once; successful later completion retracts this uncertain evidence.
    /// # Errors
    /// Returns a synchronization or evidence-counter failure.
    pub fn timed_out(&self) -> Result<(), BackendError> {
        if self
            .observation
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.tracker.record(
                self.generation,
                self.scope.clone(),
                self.error(None),
                false,
                true,
            )?;
        }
        Ok(())
    }
    /// Observes completed work. Valid outputs remain usable; quarantine affects future work.
    /// # Errors
    /// Returns a synchronization or evidence-counter failure.
    pub fn complete(&self, measured: Option<Duration>) -> Result<(), BackendError> {
        let previous = self.observation.swap(2, Ordering::AcqRel);
        if previous == 2 {
            return Ok(());
        }
        let measured = measured.filter(|time| !time.is_zero());
        match measured {
            Some(time) if time <= self.limit => {
                if previous == 1 {
                    self.tracker.retract(self.generation, &self.scope)?;
                }
            }
            measured => {
                self.failed.store(true, Ordering::Release);
                self.tracker.record(
                    self.generation,
                    self.scope.clone(),
                    self.error(measured),
                    measured.is_some(),
                    previous == 0,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ticket(tracker: &Arc<GpuWorkTracker<u32>>, key: u32) -> GpuWorkTicket<u32> {
        tracker.track(key, Duration::from_millis(7), 4).unwrap()
    }
    #[test]
    fn one_overrun_quarantines_only_its_composition() {
        let tracker = Arc::new(GpuWorkTracker::default());
        let slow = ticket(&tracker, 1);
        slow.complete(Some(Duration::from_millis(29))).unwrap();
        slow.complete(None).unwrap();
        assert!(slow.prediction_failed());
        assert!(tracker.track(1, Duration::from_millis(7), 4).is_err());
        ticket(&tracker, 2)
            .complete(Some(Duration::from_millis(28)))
            .unwrap();
        tracker.check_backend().unwrap();
        tracker.begin_load().unwrap();
        ticket(&tracker, 1).check_admission().unwrap();
    }
    #[test]
    fn unrelated_failures_do_not_form_a_repeated_composition_pattern() {
        let tracker = Arc::new(GpuWorkTracker::default());
        for key in 1..=3 {
            ticket(&tracker, key)
                .complete(Some(Duration::from_millis(29)))
                .unwrap();
        }
        tracker.check_backend().unwrap();
        ticket(&tracker, 4).check_admission().unwrap();
        assert!(tracker.track(4, Duration::ZERO, 4).is_err());
        assert!(tracker.track(4, Duration::MAX, 4).is_err());
    }

    #[test]
    fn uncertain_timeouts_can_be_retracted_without_double_counting() {
        let tracker = Arc::new(GpuWorkTracker::default());
        let tickets = [
            ticket(&tracker, 1),
            ticket(&tracker, 1),
            ticket(&tracker, 1),
        ];
        for pending in &tickets {
            pending.timed_out().unwrap();
            pending.timed_out().unwrap();
        }
        assert!(tracker.check_backend().is_err());
        for pending in tickets {
            pending.complete(Some(Duration::from_millis(7))).unwrap();
        }
        tracker.check_backend().unwrap();
        ticket(&tracker, 1).check_admission().unwrap();
    }
    #[test]
    fn missing_timestamps_and_broader_suspension_require_repetition() {
        let tracker = Arc::new(GpuWorkTracker::default());
        let first = ticket(&tracker, 1);
        let second = ticket(&tracker, 1);
        let old = ticket(&tracker, 1);
        first.complete(None).unwrap();
        assert!(tracker.track(1, Duration::from_millis(7), 4).is_err());
        second.complete(None).unwrap();
        assert!(tracker.track(1, Duration::from_millis(7), 4).is_err());
        tracker.check_backend().unwrap();
        ticket(&tracker, 2).complete(None).unwrap();
        tracker.check_backend().unwrap();
        old.complete(None).unwrap();
        assert!(tracker.check_backend().is_err());
        tracker.begin_load().unwrap();
        old.timed_out().unwrap();
        ticket(&tracker, 1).check_admission().unwrap();
    }
}
