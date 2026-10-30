use super::{StateRecording, pool_error};
use crate::{compute, guest_error, invalid_handle};
use forja_core::{Backend, BackendError, Submission};
use std::sync::atomic::Ordering;

/// Retains sequence ownership and reconciles a transaction even when dropped early.
#[must_use = "Finishing publishes state; dropping cancels after one wait."]
pub struct StateSubmission<B: Backend> {
    recording: StateRecording<B>,
    submission: B::Submission,
    completion: Option<Result<(), BackendError>>,
    reconciled: bool,
    queue_position: u64,
}

impl<B: Backend> StateRecording<B> {
    /// Queues all plane updates in version order, retaining their private backing.
    ///
    /// # Errors
    /// Refuses incomplete updates, stale capabilities or exhausted queue positions.
    /// A backend submission failure invalidates the storage until reload.
    pub fn submit(mut self) -> Result<StateSubmission<B>, compute::Error> {
        self.storage.check_usable()?;
        if self.updated.iter().any(|&updated| !updated) {
            return Err(invalid_handle(
                "every state plane must be updated before submission",
            ));
        }
        let storage = std::sync::Arc::clone(&self.storage);
        let mut last = storage
            .queue_position
            .lock()
            .map_err(|_| invalid_handle("state submission order is poisoned"))?;
        let queue_position = last
            .checked_add(1)
            .ok_or_else(|| invalid_handle("state queue timeline is exhausted"))?;
        self.sequence
            .lock()
            .mark_submitted_at(&self.transaction, queue_position)
            .map_err(pool_error)?;
        *last = queue_position;
        let commands = std::mem::take(&mut self.commands);
        let submission = storage
            .backend
            .submit(commands)
            .inspect_err(|_| {
                storage.poisoned.store(true, Ordering::Release);
            })
            .map_err(guest_error)?;
        self.handed_off = true;
        Ok(StateSubmission {
            recording: self,
            submission,
            completion: None,
            reconciled: false,
            queue_position,
        })
    }
}

impl<B: Backend> StateSubmission<B> {
    /// Drains execution and publishes every plane together on success.
    ///
    /// # Errors
    /// Returns a reconciliation or execution failure without publishing failed work.
    pub fn finish(self) -> Result<(), compute::Error> {
        self.finish_with_time().map(|_| ())
    }

    /// Drains and publishes state, retaining the backend's completed GPU duration.
    ///
    /// # Errors
    /// Returns reconciliation, timeout or execution failures without publishing failed work.
    pub fn finish_with_time(mut self) -> Result<Option<std::time::Duration>, compute::Error> {
        self.complete(true)?;
        Ok(self.submission.gpu_time())
    }

    /// Waits once and restores the retained checkpoint without publication.
    ///
    /// # Errors
    /// Returns a reconciliation or execution failure after restoring the logical checkpoint.
    pub fn cancel(mut self) -> Result<(), compute::Error> {
        self.complete(false)
    }

    fn complete(&mut self, publish: bool) -> Result<(), compute::Error> {
        let execution = self.drain();
        self.reconciled = true;
        let mut sequence = self.recording.sequence.lock();
        if matches!(execution, Err(BackendError::Timeout)) {
            sequence
                .abandon(&self.recording.transaction)
                .map_err(pool_error)?;
        } else if let Err(error) =
            sequence.finish(&self.recording.transaction, publish && execution.is_ok())
        {
            sequence
                .abandon(&self.recording.transaction)
                .map_err(pool_error)?;
            return Err(pool_error(error));
        }
        execution.map_err(guest_error)
    }

    fn drain(&mut self) -> Result<(), BackendError> {
        if let Some(result) = self.completion {
            return result;
        }
        let result = self.submission.wait();
        self.completion = Some(result);
        if matches!(result, Err(BackendError::Timeout)) {
            self.recording
                .storage
                .poisoned
                .store(true, Ordering::Release);
        } else if self
            .recording
            .storage
            .pool
            .observe_completion(self.queue_position)
            .is_err()
        {
            self.completion = Some(Err(BackendError::ExecutionFailed));
            return Err(BackendError::ExecutionFailed);
        }
        result
    }
}

impl<B: Backend> Drop for StateSubmission<B> {
    fn drop(&mut self) {
        if !self.reconciled {
            let _ = self.complete(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Host, Limits};
    use forja_core::{DType, StateLayout, StatePlane};

    #[test]
    fn all_planes_publish_together_and_cancellation_preserves_predecessor() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let layout = StateLayout::new(vec![
            StatePlane::new(DType::F32, vec![1, 1, 1]).unwrap(),
            StatePlane::new(DType::F32, vec![1, 1]).unwrap(),
        ])
        .unwrap();
        let storage = host.create_state_storage(layout, 1).unwrap();
        let sequence = storage.reserve().unwrap();
        let tensor = |shape: &[u32], values: &[f32]| {
            let t = host.backend.alloc(DType::F32, shape).unwrap();
            host.backend
                .write(
                    &t,
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            t
        };
        let q = tensor(&[1, 1], &[1.]);
        let v = tensor(&[1, 1], &[2.]);
        let decay = tensor(&[1], &[1.]);
        let beta = tensor(&[1], &[0.5]);
        let weights = tensor(&[2, 1], &[1., 1.]);
        let y = tensor(&[1, 1], &[0.]);
        let conv_y = tensor(&[1, 1], &[0.]);
        for (serial, expected) in [(1, 1_f32), (2, 1.5), (3, 1.5)] {
            let tx = sequence.begin(serial * 2 - 1).unwrap();
            let mut incomplete = storage.record(&sequence, tx.clone()).unwrap();
            incomplete
                .delta_rule(0, [&q, &q, &v, &decay, &beta], &y)
                .unwrap();
            assert!(incomplete.submit().is_err());
            let tx = sequence.begin(serial * 2).unwrap();
            let mut recording = storage.record(&sequence, tx).unwrap();
            recording
                .delta_rule(0, [&q, &q, &v, &decay, &beta], &y)
                .unwrap();
            recording
                .causal_convolution(1, &v, &weights, &conv_y)
                .unwrap();
            let submission = recording.submit().unwrap();
            if serial == 2 {
                submission.cancel().unwrap();
            } else {
                submission.finish().unwrap();
            }
            assert_eq!(host.backend.read(&y).unwrap(), expected.to_le_bytes());
            let expected_conv = if serial == 1 { 2_f32 } else { 4_f32 };
            assert_eq!(
                host.backend.read(&conv_y).unwrap(),
                expected_conv.to_le_bytes()
            );
        }
    }
}
