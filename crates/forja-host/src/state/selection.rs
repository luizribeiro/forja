mod submission;
pub use submission::StateSelectionSubmission;

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
        storage.check_usable()?;
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

    /// Retains the input boundary until the engine output has passed host validation.
    ///
    /// # Errors
    /// Refuses exhausted reference accounting before cloning any checkpoint or request lease.
    pub fn boundary(&self) -> Result<StateSelectionBoundary<B>, compute::Error> {
        let storage = &self.sequence.storage;
        let charge = StateCharge::reserve(
            storage.backend.as_ref(),
            Arc::clone(&storage.charge.live),
            storage.limit,
            1024,
        )?;
        Ok(StateSelectionBoundary {
            sequence: self.sequence.clone(),
            checkpoint: self.checkpoint.clone(),
            _charge: charge,
        })
    }

    pub(crate) fn reservation(&self) -> (std::sync::Weak<()>, u32) {
        (Arc::downgrade(&self.identity), self.destination())
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
        StateSelectionBatch::new(vec![self])?.finish(succeeded)
    }
}

/// A charged input boundary for rolling back a refused engine selection result.
pub struct StateSelectionBoundary<B: Backend> {
    sequence: StateSequenceHandle<B>,
    checkpoint: StateLaneCheckpoint,
    _charge: StateCharge,
}

impl<B: Backend> StateSelectionBoundary<B> {
    /// Verifies that the exact selected scalar and its consumer cursors were published.
    ///
    /// # Errors
    /// Refuses incomplete, failed, stale or advanced selection boundaries after draining owners.
    pub fn verify_selected(&self, scalar: u32) -> Result<(), compute::Error> {
        self.sequence
            .wait_drained(self.sequence.storage.limits.submission_timeout)
            .map_err(guest_error)?;
        let checkpoint = self
            .sequence
            .lock()
            .checkpoint_latest()
            .map_err(invalid_handle)?;
        let input = self.checkpoint.position;
        let expected = super::LanePosition {
            sampler: input
                .tokens
                .checked_sub(1)
                .ok_or_else(|| invalid_handle("selection input is empty"))?,
            grammar: input.tokens,
            parser: input.tokens,
            ..input
        };
        if checkpoint.phase != super::journal::SelectionPhase::Ready
            || checkpoint.position != expected
            || checkpoint.version.version() != self.checkpoint.version.version()
            || checkpoint
                .feedback
                .as_ref()
                .is_none_or(|feedback| feedback.checkpoint.scalar().index() != scalar)
        {
            return Err(invalid_handle(
                "selection did not publish its reserved scalar and cursors",
            ));
        }
        Ok(())
    }

    /// Drains selection work and restores feedback/cursors while preserving retained carry.
    ///
    /// # Errors
    /// Refuses timeouts or a boundary whose carry version has advanced.
    pub fn restore(self) -> Result<(), compute::Error> {
        self.sequence
            .wait_drained(self.sequence.storage.limits.submission_timeout)
            .map_err(guest_error)?;
        self.sequence
            .lock()
            .restore_selection(&self.checkpoint)
            .map_err(invalid_handle)
    }
}

/// Distinct selection guards sharing one feedback permutation and publication boundary.
pub struct StateSelectionBatch<B: Backend> {
    selections: Vec<StateSelection<B>>,
}

impl<B: Backend> StateSelectionBatch<B> {
    /// Validates and orders every lane before publication can acquire its locks.
    ///
    /// # Errors
    /// Refuses empty, duplicate or foreign lane selections.
    pub fn new(mut selections: Vec<StateSelection<B>>) -> Result<Self, compute::Error> {
        selections.sort_by_key(|selection| selection.sequence.order_key());
        let first = selections
            .first()
            .ok_or_else(|| invalid_handle("selection batch is empty"))?;
        let feedback = first
            .checkpoint
            .feedback
            .as_ref()
            .ok_or_else(|| invalid_handle("selection requires feedback storage"))?;
        if selections
            .windows(2)
            .any(|pair| pair[0].sequence.order_key() == pair[1].sequence.order_key())
            || selections.iter().any(|selection| {
                !Arc::ptr_eq(&selection.sequence.storage, &first.sequence.storage)
                    || selection.checkpoint.feedback.as_ref().is_none_or(|other| {
                        !Arc::ptr_eq(&other.binding.versions, &feedback.binding.versions)
                    })
            })
        {
            return Err(invalid_handle(
                "selection batch has duplicate or foreign lanes",
            ));
        }
        Ok(Self { selections })
    }

    /// Retains scalar owners and charges without retaining a sequence lock or drain guard.
    pub fn retained_resources(&self) -> impl Iterator<Item = CompletionResource> + '_ {
        self.selections
            .iter()
            .map(StateSelection::retained_resource)
    }

    /// Queues ordinary selection commands while retaining every private scalar owner.
    ///
    /// # Errors
    /// Returns backend failure without publishing any feedback or consumer cursor.
    pub fn submit(
        self,
        commands: forja_core::CommandList,
    ) -> Result<StateSelectionSubmission<B>, compute::Error> {
        let backend = Arc::clone(&self.selections[0].sequence.storage.backend);
        self.submit_using(|owners| backend.submit_retained(commands, owners))
    }

    pub(crate) fn submit_using(
        self,
        submit: impl FnOnce(Vec<CompletionResource>) -> Result<B::Submission, forja_core::BackendError>,
    ) -> Result<StateSelectionSubmission<B>, compute::Error> {
        let storage = &self.selections[0].sequence.storage;
        storage.check_usable()?;
        let timeout = storage.limits.submission_timeout;
        let submission = submit(self.retained_resources().collect())
            .inspect_err(|_| {
                storage
                    .poisoned
                    .store(true, std::sync::atomic::Ordering::Release);
            })
            .map_err(guest_error)?;
        Ok(StateSelectionSubmission::new(self, submission, timeout))
    }

    /// Publishes all feedback and consumer cursors together after successful device completion.
    ///
    /// # Errors
    /// Refuses any stale boundary before changing any participating lane.
    pub fn finish(self, succeeded: bool) -> Result<(), compute::Error> {
        if !succeeded {
            return Ok(());
        }
        let mut journals = self
            .selections
            .iter()
            .map(|selection| selection.sequence.lock())
            .collect::<Vec<_>>();
        for (journal, selection) in journals.iter().zip(&self.selections) {
            journal
                .validate_selection(&selection.checkpoint, &selection.identity)
                .map_err(invalid_handle)?;
        }
        let feedback = self
            .selections
            .iter()
            .map(|selection| {
                selection
                    .checkpoint
                    .feedback
                    .as_ref()
                    .map(|feedback| (&feedback.checkpoint, selection.scratch, &selection.selected))
                    .ok_or_else(|| invalid_handle("selection requires feedback storage"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let binding = &self.selections[0]
            .checkpoint
            .feedback
            .as_ref()
            .ok_or_else(|| invalid_handle("selection requires feedback storage"))?
            .binding;
        let mut versions = binding
            .versions
            .lock()
            .map_err(|_| invalid_handle("feedback associations are poisoned"))?;
        versions
            .publish_selections(&feedback)
            .map_err(invalid_handle)?;
        for journal in &mut journals {
            journal.publish_selection();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, LanePosition, Limits, StateAction, state::feedback::FeedbackBinding};
    use forja_core::{DType, StateLayout, StatePlane};

    fn fixture(lanes: u32) -> Vec<StateSequenceHandle<forja_cpu::CpuBackend>> {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                lanes,
            )
            .unwrap();
        let binding = FeedbackBinding::new(&host, lanes).unwrap();
        (0..lanes)
            .map(|_| {
                let sequence = storage.reserve().unwrap();
                sequence.lock().attach_feedback(binding.clone()).unwrap();
                sequence
            })
            .collect()
    }

    #[test]
    fn selection_rollback_preserves_retained_carry_and_refuses_newer_versions() {
        for advanced in [false, true] {
            let sequence = fixture(1).pop().unwrap();
            for (serial, action) in [(1, StateAction::Combined), (2, StateAction::Forward)] {
                let transaction = sequence
                    .begin_action_at(
                        serial,
                        LanePosition {
                            tokens: serial,
                            reserved_tokens: 7,
                            ..LanePosition::default()
                        },
                        action,
                    )
                    .unwrap();
                sequence.lock().mark_submitted(&transaction).unwrap();
                sequence.storage.pool.observe_completion(serial).unwrap();
                sequence.lock().finish(&transaction, true).unwrap();
            }
            let before = sequence.position();
            let selection = StateSelection::new(sequence.clone(), 1, 0).unwrap();
            let boundary = selection.boundary().unwrap();
            let scalar = selection.destination();
            selection.finish(true).unwrap();
            boundary.verify_selected(scalar).unwrap();
            assert!(boundary.verify_selected(u32::MAX).is_err());
            assert_eq!(sequence.position().grammar, 2);
            if advanced {
                let transaction = sequence
                    .begin_action_at(
                        3,
                        LanePosition {
                            tokens: 3,
                            reserved_tokens: 7,
                            ..LanePosition::default()
                        },
                        StateAction::Step,
                    )
                    .unwrap();
                sequence.lock().mark_submitted(&transaction).unwrap();
                sequence.storage.pool.observe_completion(3).unwrap();
                sequence.lock().finish(&transaction, true).unwrap();
                assert!(boundary.restore().is_err());
                assert_eq!(sequence.position().tokens, 3);
            } else {
                boundary.restore().unwrap();
                assert_eq!(sequence.position(), before);
                assert!(sequence.checkpoint_latest().is_err());
                sequence.rewind(1).unwrap();
                assert_eq!(sequence.position().tokens, 1);
                assert_eq!(sequence.position().grammar, 1);
            }
        }
    }

    #[test]
    fn selection_submissions_drain_cancellation_before_allowing_retry() {
        let sequence = fixture(1).pop().unwrap();
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
        for mode in 0..3 {
            let selection = StateSelection::new(sequence.clone(), 6, 0).unwrap();
            let batch = StateSelectionBatch::new(vec![selection]).unwrap();
            let submission = batch.submit(forja_core::CommandList::new()).unwrap();
            assert!(sequence.begin(2).is_err());
            match mode {
                0 => drop(submission),
                1 => submission.cancel().unwrap(),
                _ => submission.finish().unwrap(),
            }
            sequence
                .wait_drained(std::time::Duration::from_secs(1))
                .unwrap();
            assert_eq!(sequence.checkpoint_latest().is_ok(), mode == 2);
            assert_eq!(sequence.position().grammar, if mode == 2 { 7 } else { 0 });
        }
    }

    #[test]
    fn batch_selection_rejects_stale_peers_without_publishing_other_lanes() {
        let sequences = fixture(2);
        for sequence in &sequences {
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
        }
        for succeeded in [false, true] {
            let selections = sequences
                .iter()
                .enumerate()
                .map(|(lane, sequence)| {
                    StateSelection::new(sequence.clone(), 6, u32::try_from(lane).unwrap()).unwrap()
                })
                .collect();
            let mut batch = StateSelectionBatch::new(selections).unwrap();
            assert_eq!(batch.retained_resources().count(), 2);
            if succeeded {
                batch.finish(true).unwrap();
                for sequence in &sequences {
                    let checkpoint = sequence.checkpoint_latest().unwrap();
                    assert_eq!(
                        (checkpoint.position.grammar, checkpoint.position.parser),
                        (7, 7)
                    );
                }
            } else {
                batch.selections[1].identity = Arc::new(());
                assert!(batch.finish(true).is_err());
                for sequence in &sequences {
                    assert!(sequence.checkpoint_latest().is_err());
                    assert_eq!(sequence.position().grammar, 0);
                }
            }
        }
    }

    #[test]
    fn selection_guards_preserve_failed_boundaries_and_block_descendants() {
        let sequence = fixture(1).pop().unwrap();
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
        sequence.storage.pool.observe_completion(1).unwrap();
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
