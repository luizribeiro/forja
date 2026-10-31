use super::{StateCharge, StateSelection, StateSelectionBoundary};
use crate::{compute, invalid_handle};
use forja_core::Backend;
use std::sync::{Arc, Mutex};

/// Scope ownership that cancels an unsubmitted selection when its last owner drops.
#[must_use = "Keep the scope alive until the guard is submitted or cancelled."]
pub struct StateSelectionLease<B: Backend> {
    pub(in crate::state) inner: Arc<Mutex<Option<StateSelection<B>>>>,
    _charge: Arc<StateCharge>,
}

impl<B: Backend> StateSelectionLease<B> {
    pub(super) fn new(selection: StateSelection<B>) -> Self {
        Self {
            _charge: Arc::clone(&selection.charge),
            inner: Arc::new(Mutex::new(Some(selection))),
        }
    }

    /// Returns whether the scope still owns an unsubmitted selection guard.
    ///
    /// # Errors
    /// Refuses poisoned ownership metadata.
    pub fn is_pending(&self) -> Result<bool, compute::Error> {
        self.inner
            .lock()
            .map(|guard| guard.is_some())
            .map_err(|_| invalid_handle("selection ownership is poisoned"))
    }

    /// Retains the input checkpoint paired with the exact output scalar reserved by this scope.
    ///
    /// # Errors
    /// Refuses transferred guards, poisoned ownership or exhausted reference quota.
    pub fn boundary(&self) -> Result<(StateSelectionBoundary<B>, u32), compute::Error> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| invalid_handle("selection ownership is poisoned"))?;
        let selection = guard
            .as_ref()
            .ok_or_else(|| invalid_handle("selection guard was transferred"))?;
        Ok((selection.boundary()?, selection.destination()))
    }

    /// Transfers the guard once; all later transfers return `None`.
    ///
    /// # Errors
    /// Refuses poisoned ownership metadata.
    pub fn take(&self) -> Result<Option<StateSelection<B>>, compute::Error> {
        self.inner
            .lock()
            .map(|mut guard| guard.take())
            .map_err(|_| invalid_handle("selection ownership is poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LanePosition, StateAction};
    use std::sync::atomic::Ordering;

    #[test]
    fn shared_selection_ownership_cancels_until_transferred() {
        let sequence = super::super::tests::fixture(1).pop().unwrap();
        let transaction = sequence
            .begin_action_at(
                1,
                LanePosition {
                    tokens: 7,
                    reserved_tokens: 33,
                    ..LanePosition::default()
                },
                StateAction::Forward,
            )
            .unwrap();
        sequence.lock().mark_submitted(&transaction).unwrap();
        sequence.storage.pool.observe_completion(1).unwrap();
        sequence.lock().finish(&transaction, true).unwrap();
        let before = sequence.storage.charge.live.load(Ordering::Acquire);
        let lease = StateSelection::new(sequence.clone(), 6, 0).unwrap().share();
        let (boundary, scalar) = lease.boundary().unwrap();
        assert!(lease.is_pending().unwrap());
        assert!(sequence.begin(2).is_err());
        lease.take().unwrap().unwrap().finish(true).unwrap();
        assert!(!lease.is_pending().unwrap());
        assert!(lease.take().unwrap().is_none());
        assert!(lease.boundary().is_err());
        boundary.verify_selected(scalar).unwrap();
        drop(boundary);
        assert!(sequence.storage.charge.live.load(Ordering::Acquire) > before);
        drop(lease);
        assert_eq!(sequence.storage.charge.live.load(Ordering::Acquire), before);
        let lease = StateSelection::new(sequence.clone(), 6, 1).unwrap().share();
        let (boundary, scalar) = lease.boundary().unwrap();
        drop(lease);
        StateSelection::new(sequence.clone(), 6, 1)
            .unwrap()
            .finish(false)
            .unwrap();
        assert!(boundary.verify_selected(scalar).is_err());
        boundary.restore().unwrap();
        let transaction = sequence.begin(2).unwrap();
        sequence.cancel(&transaction).unwrap();
    }
}
