use super::StateSelectionBatch;
use crate::{compute, guest_error};
use forja_core::{Backend, BackendError, Submission};
use std::time::Duration;

/// Drains selection work before releasing its exclusive publication guards.
#[must_use = "Finish publishes selection; dropping drains and preserves the previous boundary."]
pub struct StateSelectionSubmission<B: Backend> {
    batch: Option<StateSelectionBatch<B>>,
    submission: B::Submission,
    timeout: Duration,
}

impl<B: Backend> StateSelectionSubmission<B> {
    pub(super) const fn new(
        batch: StateSelectionBatch<B>,
        submission: B::Submission,
        timeout: Duration,
    ) -> Self {
        Self {
            batch: Some(batch),
            submission,
            timeout,
        }
    }

    /// Returns the backend work observed by the host's ordinary completion worker.
    #[must_use]
    pub const fn submission(&self) -> &B::Submission {
        &self.submission
    }

    /// Drains device work and publishes all selected lanes after success.
    ///
    /// # Errors
    /// Returns device or publication failure, preserving every previous boundary on device failure.
    pub fn finish(self) -> Result<(), compute::Error> {
        let result = self.submission.wait_timeout(self.timeout);
        self.finish_after(result)
    }

    /// Drains device work and leaves all feedback and consumer cursors unchanged.
    ///
    /// # Errors
    /// Returns any device failure after cancellation.
    pub fn cancel(mut self) -> Result<(), compute::Error> {
        let result = self.submission.wait_timeout(self.timeout);
        self.complete(result, false)
    }

    /// Reconciles a result already observed by the host completion worker.
    ///
    /// # Errors
    /// Returns device or publication failure without publishing failed work.
    pub fn finish_after(mut self, result: Result<(), BackendError>) -> Result<(), compute::Error> {
        self.complete(result, true)
    }

    fn complete(
        &mut self,
        result: Result<(), BackendError>,
        publish: bool,
    ) -> Result<(), compute::Error> {
        if let Some(batch) = self.batch.take() {
            if result == Err(BackendError::Timeout) {
                batch.selections[0]
                    .sequence
                    .storage
                    .poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
            }
            batch.finish(publish && result.is_ok())?;
        }
        result.map_err(guest_error)
    }
}

impl<B: Backend> Drop for StateSelectionSubmission<B> {
    fn drop(&mut self) {
        if self.batch.is_some() {
            let result = self.submission.wait_timeout(self.timeout);
            let _ = self.complete(result, false);
        }
    }
}
