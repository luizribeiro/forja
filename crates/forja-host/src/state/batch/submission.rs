use super::StateBatch;
use crate::compute;
use forja_core::{Backend, BackendError, Submission};

/// Completion owner for every lane in a queued protected-state batch.
#[must_use = "Finish publishes the batch; dropping cancels it after drain."]
pub struct StateBatchSubmission<B: Backend> {
    pub(super) batch: StateBatch<B>,
    pub(super) submission: B::Submission,
    pub(super) position: u64,
    pub(super) reconciled: bool,
}

impl<B: Backend> StateBatchSubmission<B> {
    /// Drains once and publishes all lane versions and cursors on success.
    ///
    /// # Errors
    /// Reports execution failure or taint from a failed predecessor.
    pub fn finish(mut self) -> Result<(), compute::Error> {
        self.complete(true)
    }

    /// Drains once, cancels all lanes and taints their queued descendants.
    ///
    /// # Errors
    /// Reports execution or reconciliation failure after cancellation.
    pub fn cancel(mut self) -> Result<(), compute::Error> {
        self.complete(false)
    }

    pub(crate) fn submission(&self) -> &B::Submission {
        &self.submission
    }

    pub(crate) fn finish_after(
        mut self,
        execution: Result<(), BackendError>,
    ) -> Result<(), compute::Error> {
        self.reconcile_after(execution, true)
    }

    fn complete(&mut self, publish: bool) -> Result<(), compute::Error> {
        let execution = self
            .submission
            .wait_timeout(self.batch.recordings[0].limits.submission_timeout);
        self.reconcile_after(execution, publish)
    }

    fn reconcile_after(
        &mut self,
        mut execution: Result<(), BackendError>,
        publish: bool,
    ) -> Result<(), compute::Error> {
        self.reconciled = true;
        if execution != Err(BackendError::Timeout) {
            for recording in &self.batch.recordings {
                if let Some(predecessor) = &recording.predecessor {
                    match predecessor.wait(recording.limits.submission_timeout) {
                        Ok(true) => (),
                        Ok(false) => execution = Err(BackendError::ExecutionFailed),
                        Err(error) => {
                            execution = Err(error);
                            break;
                        }
                    }
                }
            }
        }
        self.batch.reconcile(self.position, execution, publish)
    }
}

impl<B: Backend> Drop for StateBatchSubmission<B> {
    fn drop(&mut self) {
        if !self.reconciled {
            let _ = self.complete(false);
        }
    }
}
