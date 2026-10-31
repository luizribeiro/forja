use super::{StateCharge, StateLaneCheckpoint, StateSequenceHandle, sequence::RecordingGuard};
use crate::{compute, guest_error, invalid_handle};
use forja_core::{Backend, CompletionResource, FeedbackPin};
use std::sync::Arc;

/// An exclusive selection at a physically drained, complete carry boundary.
pub struct StateSelection<B: Backend> {
    sequence: StateSequenceHandle<B>,
    checkpoint: StateLaneCheckpoint,
    scratch: u32,
    selected: FeedbackPin,
    identity: Arc<()>,
    charge: Arc<StateCharge>,
    _drain: RecordingGuard,
}

impl<B: Backend> StateSelection<B> {
    /// Reserves scratch without changing published feedback or consumer cursors.
    ///
    /// # Errors
    /// Refuses an unfinished forward, external feedback, invalid position or exhausted quota.
    pub fn new(
        sequence: StateSequenceHandle<B>,
        start: u64,
        scratch: u32,
    ) -> Result<Self, compute::Error> {
        let storage = &sequence.storage;
        let charge = Arc::new(StateCharge::reserve(
            storage.backend.as_ref(),
            Arc::clone(&storage.charge.live),
            storage.limit,
            1024,
        )?);
        sequence
            .wait_drained(storage.limits.submission_timeout)
            .map_err(guest_error)?;
        let identity = Arc::new(());
        let checkpoint = sequence
            .lock()
            .reserve_selection(start, &identity)
            .map_err(invalid_handle)?;
        let feedback = checkpoint
            .feedback
            .as_ref()
            .ok_or_else(|| invalid_handle("selection requires feedback storage"))?;
        let selected = feedback
            .binding
            .versions
            .lock()
            .map_err(|_| invalid_handle("feedback associations are poisoned"))?
            .claim_scratch(scratch)
            .map_err(invalid_handle)?;
        let drain = sequence.recording_guard().map_err(guest_error)?;
        Ok(Self {
            sequence,
            checkpoint,
            scratch,
            selected,
            identity,
            charge,
            _drain: drain,
        })
    }

    /// Returns the private output scalar to trusted recording code.
    #[must_use]
    pub fn destination(&self) -> u32 {
        self.selected.index()
    }

    /// Retains only scalar ownership and accounting in backend completion handlers.
    #[must_use]
    pub fn retained_resource(&self) -> CompletionResource {
        CompletionResource::new(Arc::new((self.selected.clone(), Arc::clone(&self.charge))))
    }

    /// Publishes a successful selection, or leaves the previous boundary unchanged on failure.
    ///
    /// # Errors
    /// Refuses a stale boundary or scalar association without changing consumer cursors.
    pub fn finish(self, succeeded: bool) -> Result<(), compute::Error> {
        if !succeeded {
            return Ok(());
        }
        let mut journal = self.sequence.lock();
        journal
            .validate_selection(&self.checkpoint, &self.identity)
            .map_err(invalid_handle)?;
        let feedback = self
            .checkpoint
            .feedback
            .as_ref()
            .ok_or_else(|| invalid_handle("selection requires feedback storage"))?;
        feedback
            .binding
            .versions
            .lock()
            .map_err(|_| invalid_handle("feedback associations are poisoned"))?
            .publish_selection(&feedback.checkpoint, self.scratch, &self.selected)
            .map_err(invalid_handle)?;
        journal.publish_selection();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, LanePosition, Limits, StateAction, state::feedback::FeedbackBinding};
    use forja_core::{DType, StateLayout, StatePlane};

    #[test]
    fn selection_guards_preserve_failed_boundaries_and_block_descendants() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                1,
            )
            .unwrap();
        let sequence = storage.reserve().unwrap();
        let binding = FeedbackBinding::new(&host, 1).unwrap();
        sequence.lock().attach_feedback(binding.clone()).unwrap();
        assert!(StateSelection::new(sequence.clone(), 0, 0).is_err());
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
        assert!(StateSelection::new(sequence.clone(), 6, 0).is_err());
        sequence.lock().mark_submitted(&transaction).unwrap();
        storage.pool.observe_completion(1).unwrap();
        sequence.lock().finish(&transaction, true).unwrap();
        assert!(StateSelection::new(sequence.clone(), 5, 0).is_err());
        let before = sequence.position();
        let selection = StateSelection::new(sequence.clone(), 6, 0).unwrap();
        assert!(sequence.begin(2).is_err());
        assert!(sequence.rewind(0).is_err());
        let resource = selection.retained_resource();
        selection.finish(false).unwrap();
        assert_eq!(sequence.position(), before);
        assert!(sequence.checkpoint_latest().is_err());
        assert!(StateSelection::new(sequence.clone(), 6, 0).is_err());
        drop(resource);
        let selection = StateSelection::new(sequence.clone(), 6, 0).unwrap();
        let selected = selection.destination();
        selection.finish(true).unwrap();
        let output = sequence.checkpoint_latest().unwrap();
        let index = |checkpoint: &StateLaneCheckpoint| {
            checkpoint
                .feedback
                .as_ref()
                .unwrap()
                .checkpoint
                .scalar()
                .index()
        };
        assert_eq!(index(&output), selected);
        assert_eq!(
            (
                output.position.sampler,
                output.position.grammar,
                output.position.parser
            ),
            (6, 7, 7)
        );
        let selection = StateSelection::new(sequence.clone(), 6, 1).unwrap();
        selection.finish(false).unwrap();
        assert_eq!(sequence.position(), output.position);
        assert_eq!(index(&sequence.checkpoint_latest().unwrap()), selected);
        sequence.begin(2).unwrap();
    }
}
