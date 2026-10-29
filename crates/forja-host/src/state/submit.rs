use super::{StateRecording, pool_error};
use crate::{compute, guest_error, invalid_handle};
use forja_core::{Backend, BackendError, StateSequence, Submission};
use std::sync::atomic::Ordering;

/// Retains private state and authentication storage until submitted work drains.
pub struct StateSubmission<B: Backend> {
    recording: StateRecording<B>,
    submission: B::Submission,
    drained: bool,
}

impl<B: Backend> StateRecording<B> {
    /// Queues all plane updates in version order, retaining their private backing.
    ///
    /// # Errors
    /// Refuses incomplete updates, stale capabilities or nonmonotonic submission serials.
    /// A backend submission failure invalidates the storage until reload.
    pub fn submit(
        mut self,
        sequence: &mut StateSequence,
    ) -> Result<StateSubmission<B>, compute::Error> {
        self.storage.check_usable()?;
        let slot = sequence.bind(&self.transaction).map_err(pool_error)?;
        self.storage.pool.validate_slot(&slot).map_err(pool_error)?;
        if self.updated.iter().any(|&updated| !updated) {
            return Err(invalid_handle(
                "every state plane must be updated before submission",
            ));
        }
        let storage = std::sync::Arc::clone(&self.storage);
        let mut last = storage
            .last_submission
            .lock()
            .map_err(|_| invalid_handle("state submission order is poisoned"))?;
        if self.serial() <= *last {
            return Err(invalid_handle(
                "state submission serial must increase across sequences",
            ));
        }
        sequence
            .mark_submitted(&self.transaction)
            .map_err(pool_error)?;
        *last = self.serial();
        let commands = std::mem::take(&mut self.commands);
        let submission = storage
            .backend
            .submit(commands)
            .inspect_err(|_| {
                storage.poisoned.store(true, Ordering::Release);
            })
            .map_err(guest_error)?;
        Ok(StateSubmission {
            recording: self,
            submission,
            drained: false,
        })
    }
}

impl<B: Backend> StateSubmission<B> {
    /// Drains execution and publishes every plane together on success.
    ///
    /// # Errors
    /// Returns execution failure without publishing; descendants remain tainted until drain.
    pub fn finish(self, sequence: &mut StateSequence) -> Result<(), compute::Error> {
        self.complete(sequence, true)
    }

    /// Drains execution and restores the retained predecessor without publication.
    ///
    /// # Errors
    /// Returns an execution or transaction-order error after draining the command.
    pub fn cancel(self, sequence: &mut StateSequence) -> Result<(), compute::Error> {
        self.complete(sequence, false)
    }

    fn complete(
        mut self,
        sequence: &mut StateSequence,
        publish: bool,
    ) -> Result<(), compute::Error> {
        let result = self.drain();
        if matches!(result, Err(BackendError::Timeout)) {
            result.map_err(guest_error)?;
        } else {
            let publication = sequence
                .finish(&self.recording.transaction, publish && result.is_ok())
                .map_err(pool_error);
            result.map_err(guest_error)?;
            publication?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<(), BackendError> {
        if self.drained {
            return Ok(());
        }
        let result = self.submission.wait();
        if matches!(result, Err(BackendError::Timeout)) {
            self.recording
                .storage
                .poisoned
                .store(true, Ordering::Release);
        } else {
            self.recording
                .storage
                .pool
                .observe_completion(self.recording.serial())
                .map_err(|_| BackendError::ExecutionFailed)?;
            self.drained = true;
        }
        result
    }
}

impl<B: Backend> Drop for StateSubmission<B> {
    fn drop(&mut self) {
        let _ = self.drain();
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
        let mut sequence = storage.reserve().unwrap();
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
            let tx = sequence.begin(serial).unwrap();
            let mut incomplete = storage.record(&sequence, tx.clone()).unwrap();
            incomplete
                .delta_rule(0, [&q, &q, &v, &decay, &beta], &y)
                .unwrap();
            assert!(incomplete.submit(&mut sequence).is_err());
            let mut recording = storage.record(&sequence, tx).unwrap();
            recording
                .delta_rule(0, [&q, &q, &v, &decay, &beta], &y)
                .unwrap();
            recording
                .causal_convolution(1, &v, &weights, &conv_y)
                .unwrap();
            let submission = recording.submit(&mut sequence).unwrap();
            if serial == 2 {
                submission.cancel(&mut sequence).unwrap();
            } else {
                submission.finish(&mut sequence).unwrap();
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
