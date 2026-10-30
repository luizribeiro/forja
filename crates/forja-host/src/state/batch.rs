mod submission;
pub use submission::StateBatchSubmission;
#[cfg(test)]
mod queue_tests;

use super::{StateRecording, pool_error};
use crate::{compute, guest_error, invalid_handle};
use forja_core::{Backend, BackendError, CommandList};
use std::sync::{Arc, atomic::Ordering};

/// Plane-complete updates for distinct lanes sharing one protected state pool.
pub struct StateBatch<B: Backend> {
    recordings: Vec<StateRecording<B>>,
}

impl<B: Backend> StateBatch<B> {
    /// Validates the whole batch before any lane can be sealed for submission.
    ///
    /// # Errors
    /// Refuses empty, incomplete, duplicated or foreign-pool lane updates.
    pub fn new(mut recordings: Vec<StateRecording<B>>) -> Result<Self, compute::Error> {
        if recordings.is_empty() {
            return Err(invalid_handle("a state batch requires active lanes"));
        }
        recordings.sort_by_key(|recording| recording.sequence.order_key());
        let storage = &recordings[0].storage;
        if recordings.iter().any(|recording| {
            !Arc::ptr_eq(storage, &recording.storage) || recording.updated.contains(&false)
        }) || recordings
            .windows(2)
            .any(|pair| pair[0].sequence.order_key() == pair[1].sequence.order_key())
        {
            return Err(invalid_handle(
                "state batch lanes are incomplete, duplicated or foreign",
            ));
        }
        Ok(Self { recordings })
    }

    /// Submits once and publishes every lane only after the complete batch succeeds.
    ///
    /// # Errors
    /// Returns admission, authentication or execution failure without partial publication.
    pub fn submit_and_finish(self) -> Result<(), compute::Error> {
        self.submit()?.finish()
    }

    /// Queues every lane as one submission while retaining all version owners.
    ///
    /// # Errors
    /// Returns admission or submission failure; uncertain execution requires reload.
    pub fn submit(mut self) -> Result<StateBatchSubmission<B>, compute::Error> {
        let mut commands = CommandList::new();
        for recording in &mut self.recordings {
            for dispatch in std::mem::take(&mut recording.commands).into_dispatches() {
                commands.record_validated(dispatch);
            }
        }
        let storage = Arc::clone(&self.recordings[0].storage);
        self.submit_using(|owners| storage.backend.submit_retained(commands, owners))
    }

    pub(crate) fn submit_using(
        mut self,
        submit: impl FnOnce(Vec<forja_core::CompletionResource>) -> Result<B::Submission, BackendError>,
    ) -> Result<StateBatchSubmission<B>, compute::Error> {
        let storage = Arc::clone(&self.recordings[0].storage);
        storage.check_usable()?;
        let retained = self
            .recordings
            .iter()
            .flat_map(|recording| recording.retained_resources())
            .collect();
        let mut queue = storage
            .queue_position
            .lock()
            .map_err(|_| invalid_handle("state queue is poisoned"))?;
        storage.check_usable()?;
        let position = queue
            .checked_add(1)
            .ok_or_else(|| invalid_handle("state queue timeline exhausted"))?;
        let mut lanes = self
            .recordings
            .iter()
            .map(|recording| recording.sequence.lock())
            .collect::<Vec<_>>();
        for (lane, recording) in lanes.iter().zip(&self.recordings) {
            lane.validate_submission_at(&recording.transaction, position)
                .map_err(pool_error)?;
        }
        for (lane, recording) in lanes.iter_mut().zip(&self.recordings) {
            lane.mark_submitted_at(&recording.transaction, position)
                .inspect_err(|_| storage.poisoned.store(true, Ordering::Release))
                .map_err(pool_error)?;
        }
        drop(lanes);
        for recording in &mut self.recordings {
            recording.handed_off = true;
        }
        *queue = position;
        let submission = submit(retained)
            .inspect_err(|_| storage.poisoned.store(true, Ordering::Release))
            .map_err(guest_error)?;
        drop(queue);
        Ok(StateBatchSubmission {
            batch: self,
            submission,
            position,
            reconciled: false,
        })
    }

    fn reconcile(
        &mut self,
        position: u64,
        execution: Result<(), BackendError>,
        publish: bool,
    ) -> Result<(), compute::Error> {
        let storage = Arc::clone(&self.recordings[0].storage);
        if execution == Err(BackendError::Timeout) {
            storage.poisoned.store(true, Ordering::Release);
            for recording in &self.recordings {
                let _ = recording.sequence.lock().abandon(&recording.transaction);
            }
            return execution.map_err(guest_error);
        }
        storage
            .pool
            .observe_completion(position)
            .map_err(pool_error)?;
        let mut lanes = self
            .recordings
            .iter()
            .map(|recording| recording.sequence.lock())
            .collect::<Vec<_>>();
        let validation = lanes
            .iter()
            .zip(&self.recordings)
            .try_for_each(|(lane, recording)| {
                lane.validate_finish(&recording.transaction, publish && execution.is_ok())
            });
        if let Err(error) = validation {
            for (lane, recording) in lanes.iter_mut().zip(&self.recordings) {
                let _ = lane.abandon(&recording.transaction);
            }
            return Err(pool_error(error));
        }
        for (lane, recording) in lanes.iter_mut().zip(&self.recordings) {
            lane.finish(&recording.transaction, publish && execution.is_ok())
                .map_err(pool_error)?;
        }
        execution.map_err(guest_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, LanePosition, Limits};
    use forja_core::{DType, StateLayout, StatePlane};

    const LIMITS: Limits = Limits::new(131_072, 8, 4096, 128, 4096);

    fn batch_publication<B: Backend + 'static>(host: &Host<B>, corrupt: bool, failed: bool) {
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                2,
            )
            .unwrap();
        let lanes = [storage.reserve().unwrap(), storage.reserve().unwrap()];
        let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        host.backend
            .write(&input, &2_f32.to_le_bytes().repeat(7))
            .unwrap();
        host.backend
            .write(&weights, &1_f32.to_le_bytes().repeat(28))
            .unwrap();
        let position = LanePosition {
            tokens: 64,
            sampler: 63,
            grammar: 7,
            parser: 33,
            reserved_tokens: 128,
        };
        let mut recordings = Vec::new();
        let mut outputs = Vec::new();
        for lane in &lanes {
            let mut recording = storage
                .record(lane, lane.begin_at(1, position).unwrap())
                .unwrap();
            let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
            recording
                .causal_convolution(0, &input, &weights, &output)
                .unwrap();
            outputs.push(output);
            recordings.push(recording);
        }
        let batch = StateBatch::new(recordings).unwrap();
        if corrupt {
            host.backend
                .write(&batch.recordings.last().unwrap().metadata[1], &[0; 24])
                .unwrap();
        }
        assert_eq!(batch.submit_and_finish().is_err(), failed);
        for lane in &lanes {
            assert_eq!(
                lane.position(),
                if failed {
                    LanePosition::default()
                } else {
                    position
                }
            );
            if failed {
                let retry = lane.begin(2).unwrap();
                assert_eq!(retry.input_version(), 0);
                lane.cancel(&retry).unwrap();
            }
        }
        if !failed {
            for output in outputs {
                assert_eq!(
                    host.backend.read(&output).unwrap(),
                    2_f32.to_le_bytes().repeat(7)
                );
            }
        }
    }

    #[test]
    fn complete_state_batches_publish_all_lanes_or_none() {
        for corrupt in [false, true] {
            batch_publication(
                &Host::new(forja_cpu::CpuBackend::new(), LIMITS),
                corrupt,
                corrupt,
            );
        }
    }

    #[test]
    fn a_failed_command_buffer_preserves_every_lane_checkpoint() {
        let (control, host) = super::super::lifecycle_tests::controlled_host();
        control.failure.store(2, Ordering::Release);
        batch_publication(&host, false, true);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_complete_state_batches_publish_all_lanes_or_none() {
        for corrupt in [false, true] {
            batch_publication(
                &Host::new(forja_metal::MetalBackend::new().unwrap(), LIMITS),
                corrupt,
                corrupt,
            );
        }
    }
}
