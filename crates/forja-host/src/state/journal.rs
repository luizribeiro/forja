use super::{outcome::StateOutcome, paired::PagedInput};
use forja_core::{PagePoolError, ReleasedStateCheckpoint, StateSequence, StateTransaction};
use std::sync::Arc;
use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
};

/// Publication requested by a stateful engine entry point.
#[derive(Clone, Copy, Debug)]
pub enum StateAction {
    /// Publishes carry with externally supplied token feedback.
    Step,
    /// Advances carry and selects the next token.
    Combined,
    /// Advances carry and awaits a separate token selection.
    Forward,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum SelectionPhase {
    #[default]
    External,
    Ready,
    Awaiting,
}

impl StateAction {
    const fn phase(self) -> SelectionPhase {
        match self {
            Self::Step => SelectionPhase::External,
            Self::Combined => SelectionPhase::Ready,
            Self::Forward => SelectionPhase::Awaiting,
        }
    }
}

/// Host-owned cursors published together with a complete recurrent/conv version.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LanePosition {
    /// Committed logical KV length.
    pub tokens: u64,
    /// Absolute sampling cursor.
    pub sampler: u64,
    /// Grammar consumer cursor.
    pub grammar: u64,
    /// Streaming parser consumer cursor.
    pub parser: u64,
    /// Capacity reserved for this request.
    pub reserved_tokens: u64,
}

struct RetainedBoundary {
    version: Option<ReleasedStateCheckpoint>,
    position: LanePosition,
    feedback: Option<super::feedback::FeedbackBoundary>,
    phase: SelectionPhase,
    paged: Option<PagedInput>,
}

struct Pending {
    retired: Option<RetainedBoundary>,
    paged: Option<PagedInput>,
    serial: u64,
    position: LanePosition,
    phase: SelectionPhase,
    predecessor: super::StateLaneCheckpoint,
    outcome: Arc<StateOutcome>,
    dependency: Option<Arc<StateOutcome>>,
}

pub(super) struct StateJournal {
    rollback_paged: Option<PagedInput>,
    retained_paged: Option<PagedInput>,
    sequence: StateSequence,
    feedback: Option<super::feedback::LaneFeedback>,
    pub(super) position: LanePosition,
    phase: SelectionPhase,
    pending: VecDeque<Pending>,
    retained: Option<super::StateLaneCheckpoint>,
}

impl Deref for StateJournal {
    type Target = StateSequence;
    fn deref(&self) -> &Self::Target {
        &self.sequence
    }
}
impl DerefMut for StateJournal {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sequence
    }
}

impl StateJournal {
    pub(super) fn new(sequence: StateSequence) -> Self {
        Self {
            rollback_paged: None,
            retained_paged: None,
            sequence,
            feedback: None,
            position: LanePosition::default(),
            phase: SelectionPhase::External,
            pending: VecDeque::new(),
            retained: None,
        }
    }

    pub(super) fn attach_feedback(
        &mut self,
        binding: super::feedback::FeedbackBinding,
    ) -> Result<(), PagePoolError> {
        if self.feedback.is_some() {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.feedback = Some(super::feedback::LaneFeedback::new(binding, &self.sequence)?);
        Ok(())
    }

    pub(super) fn begin(
        &mut self,
        serial: u64,
        position: Option<LanePosition>,
    ) -> Result<StateTransaction, PagePoolError> {
        self.begin_action(serial, position, None)
    }

    pub(super) fn begin_action(
        &mut self,
        serial: u64,
        position: Option<LanePosition>,
        action: Option<StateAction>,
    ) -> Result<StateTransaction, PagePoolError> {
        self.restore_paged_if_drained()?;
        if self.rollback_paged.is_some() {
            return Err(PagePoolError::BankInFlight);
        }
        let input = self
            .pending
            .back()
            .map_or(self.position, |entry| entry.position);
        let mut position = position.unwrap_or(input);
        if let Some(action) = action {
            position = LanePosition {
                tokens: position.tokens,
                reserved_tokens: position.reserved_tokens,
                ..input
            };
            if matches!(action, StateAction::Combined) {
                position.sampler = position
                    .tokens
                    .checked_sub(1)
                    .ok_or(PagePoolError::InvalidCheckpoint)?;
                position.grammar = position.tokens;
                position.parser = position.tokens;
            }
        }
        if position.tokens > position.reserved_tokens {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let checkpoint = self.checkpoint_latest()?;
        let (mut retained, boundary_before) =
            self.retained.take().map_or((None, None), |checkpoint| {
                (
                    Some(checkpoint.version),
                    Some((checkpoint.position, checkpoint.feedback, checkpoint.phase)),
                )
            });
        let (transaction, released) =
            match self
                .sequence
                .begin_preserving_checkpoint(serial, &mut retained)
            {
                Ok(transaction) => transaction,
                Err(error) => {
                    self.retained = retained.zip(boundary_before).map(
                        |(version, (position, feedback, phase))| super::StateLaneCheckpoint {
                            version,
                            position,
                            feedback,
                            phase,
                        },
                    );
                    return Err(error);
                }
            };
        let retired =
            released
                .zip(boundary_before)
                .map(|(version, (position, feedback, phase))| RetainedBoundary {
                    version: Some(version),
                    position,
                    feedback,
                    phase,
                    paged: self.retained_paged.take(),
                });
        let dependency = self.pending.back().map(|entry| Arc::clone(&entry.outcome));
        self.retained_paged = None;
        self.pending.push_back(Pending {
            retired,
            paged: None,
            serial,
            position,
            phase: action.map_or(SelectionPhase::External, StateAction::phase),
            predecessor: checkpoint,
            outcome: Arc::default(),
            dependency,
        });
        self.retained = None;
        Ok(transaction)
    }

    pub(super) fn cancel_unsubmitted(
        &mut self,
        transaction: &StateTransaction,
    ) -> Result<(), PagePoolError> {
        let index = self
            .pending
            .iter()
            .position(|entry| entry.serial == transaction.serial())
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        let restored = if let Some(retired) = self.pending[index].retired.as_mut() {
            self.sequence
                .cancel_preserving_checkpoint(transaction, &mut retired.version)?
        } else {
            self.sequence.cancel_unsubmitted(transaction)?;
            None
        };
        if let Some((retired, version)) = self.pending[index].retired.take().zip(restored) {
            self.retained = Some(super::StateLaneCheckpoint {
                version,
                position: retired.position,
                feedback: retired.feedback,
                phase: retired.phase,
            });
            self.retained_paged = retired.paged;
        }
        self.fail_pending(index);
        self.restore_paged_if_drained()
    }

    #[cfg(test)]
    pub(super) fn mark_submitted(
        &mut self,
        transaction: &StateTransaction,
    ) -> Result<(), PagePoolError> {
        self.mark_submitted_at(transaction, transaction.serial())
    }

    pub(super) fn mark_submitted_at(
        &mut self,
        transaction: &StateTransaction,
        serial: u64,
    ) -> Result<(), PagePoolError> {
        let index = self
            .pending
            .iter()
            .position(|entry| entry.serial == transaction.serial())
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        self.sequence.mark_submitted_at(transaction, serial)?;
        self.pending[index].retired = None;
        Ok(())
    }

    pub(super) fn abandon(&mut self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        let result = self.sequence.abandon(transaction);
        if result.is_ok() {
            self.fail_pending(0);
        }
        self.restore_paged_if_drained()?;
        result
    }

    pub(super) fn finish(
        &mut self,
        transaction: &StateTransaction,
        success: bool,
    ) -> Result<(), PagePoolError> {
        if self.pending.front().map(|entry| entry.serial) != Some(transaction.serial()) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.sequence.finish(transaction, success)?;
        if success {
            if let Some(entry) = self.pending.pop_front() {
                self.position = entry.position;
                self.phase = entry.phase;
                self.retained = Some(entry.predecessor);
                self.retained_paged = entry.paged;
                entry.outcome.finish(true);
            }
        } else {
            self.fail_pending(0);
        }
        self.restore_paged_if_drained()
    }

    pub(super) fn attach_paged(
        &mut self,
        serial: u64,
        paged: PagedInput,
    ) -> Result<(), PagePoolError> {
        let entry = self
            .pending
            .iter_mut()
            .find(|entry| entry.serial == serial)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        if entry.paged.is_some() || entry.predecessor.position.tokens != paged.tokens() {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        entry.paged = Some(paged);
        Ok(())
    }

    fn fail_pending(&mut self, index: usize) {
        for mut entry in self.pending.drain(index..) {
            if let Some(paged) = entry.paged.take()
                && self
                    .rollback_paged
                    .as_ref()
                    .is_none_or(|old| old.tokens() > paged.tokens())
            {
                self.rollback_paged = Some(paged);
            }
            entry.outcome.finish(false);
        }
    }

    fn restore_paged_if_drained(&mut self) -> Result<(), PagePoolError> {
        let Some(paged) = &self.rollback_paged else {
            return Ok(());
        };
        let checkpoint = self.sequence.checkpoint()?;
        match self.sequence.rewind(&checkpoint) {
            Err(PagePoolError::BankInFlight) => return Ok(()),
            result => result?,
        }
        if paged.tokens() != self.position.tokens {
            return Err(PagePoolError::CorruptState);
        }
        if let Some(paged) = self.rollback_paged.take() {
            paged.restore()?;
        }
        Ok(())
    }

    pub(super) fn dependency(
        &self,
        transaction: &StateTransaction,
    ) -> Result<Option<Arc<StateOutcome>>, PagePoolError> {
        self.pending
            .iter()
            .find(|entry| entry.serial == transaction.serial())
            .map(|entry| entry.dependency.clone())
            .ok_or(PagePoolError::InvalidCheckpoint)
    }

    pub(super) fn feedback_use(
        &self,
        transaction: &StateTransaction,
        destination: u32,
        charge: Arc<super::StateCharge>,
    ) -> Result<Option<super::feedback::FeedbackUse>, PagePoolError> {
        let Some(feedback) = &self.feedback else {
            return Ok(None);
        };
        let pending = self
            .pending
            .iter()
            .find(|entry| entry.serial == transaction.serial())
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        let input = &pending.predecessor;
        let source = if input.phase == SelectionPhase::Ready {
            Some(
                input
                    .feedback
                    .as_ref()
                    .ok_or(PagePoolError::CorruptState)?
                    .checkpoint
                    .scalar()
                    .clone(),
            )
        } else {
            None
        };
        Ok(Some(super::feedback::FeedbackUse {
            combined: pending.phase == SelectionPhase::Ready,
            explicit: std::sync::atomic::AtomicBool::new(true),
            source,
            destination: feedback.destination(destination)?,
            charge,
        }))
    }

    pub(super) fn checkpoint_latest(&self) -> Result<super::StateLaneCheckpoint, PagePoolError> {
        let version = self.sequence.checkpoint_latest()?;
        let feedback = self
            .feedback
            .as_ref()
            .map(|feedback| feedback.checkpoint(version.version()))
            .transpose()?;
        Ok(super::StateLaneCheckpoint {
            version,
            feedback,
            phase: self
                .pending
                .back()
                .map_or(self.phase, |pending| pending.phase),
            position: self
                .pending
                .back()
                .map_or(self.position, |pending| pending.position),
        })
    }

    pub(super) fn restore_with(
        &mut self,
        checkpoint: &super::StateLaneCheckpoint,
        action: impl FnOnce() -> Result<(), PagePoolError>,
    ) -> Result<(), PagePoolError> {
        if let Some(feedback) = &self.feedback {
            feedback.validate(checkpoint.feedback.as_ref())?;
        } else if checkpoint.feedback.is_some() {
            return Err(PagePoolError::ForeignResource);
        }
        let previous = self.sequence.checkpoint()?;
        self.sequence.rewind(&checkpoint.version)?;
        if let Err(error) = action() {
            self.sequence.rewind(&previous)?;
            return Err(error);
        }
        if let (Some(feedback), Some(boundary)) = (&self.feedback, &checkpoint.feedback) {
            feedback.restore(boundary)?;
        }
        self.position = checkpoint.position;
        self.phase = checkpoint.phase;
        self.retained = None;
        self.retained_paged = None;
        Ok(())
    }

    pub(super) fn rewind(&mut self, tokens: u64) -> Result<(), PagePoolError> {
        let checkpoint = self
            .retained
            .as_ref()
            .filter(|checkpoint| checkpoint.position.tokens == tokens)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        self.sequence.rewind(&checkpoint.version)?;
        if let Some(paged) = self.retained_paged.take() {
            paged.restore()?;
        }
        if let (Some(feedback), Some(boundary)) = (&self.feedback, &checkpoint.feedback) {
            feedback.restore(boundary)?;
        }
        self.position = checkpoint.position;
        self.phase = checkpoint.phase;
        self.retained = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{DType, StateLayout, StatePlane, StatePool};

    #[test]
    fn accepted_predecessors_release_the_next_depth_two_version() {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut journal = StateJournal::new(pool.reserve().unwrap());
        let position = |tokens| LanePosition {
            tokens,
            reserved_tokens: 128,
            ..LanePosition::default()
        };
        let caller = journal.checkpoint_latest().unwrap();
        let first = journal.begin(1, Some(position(1))).unwrap();
        journal.mark_submitted(&first).unwrap();
        let second = journal.begin(2, Some(position(2))).unwrap();
        journal.mark_submitted(&second).unwrap();
        pool.observe_completion(1).unwrap();
        journal.finish(&first, true).unwrap();
        assert!(journal.begin(3, Some(position(3))).is_err());
        assert_eq!(journal.retained.as_ref().unwrap().position.tokens, 0);
        drop(caller);
        let third = journal.begin(3, Some(position(3))).unwrap();
        journal.mark_submitted(&third).unwrap();
        pool.observe_completion(3).unwrap();
        journal.finish(&second, true).unwrap();
        journal.finish(&third, true).unwrap();
        journal.rewind(2).unwrap();
        assert_eq!(journal.position, position(2));
    }

    #[test]
    fn positions_publish_atomically_and_rewind_to_the_queued_input() {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut journal = StateJournal::new(pool.reserve().unwrap());
        let position = |tokens| LanePosition {
            tokens,
            sampler: tokens + 1,
            grammar: tokens / 2,
            parser: tokens / 3,
            reserved_tokens: 128,
        };
        assert!(journal.begin(1, Some(position(129))).is_err());
        let initial = journal.checkpoint_latest().unwrap();
        let first = journal.begin(1, Some(position(64))).unwrap();
        let predicted = journal.checkpoint_latest().unwrap();
        journal.mark_submitted(&first).unwrap();
        let second = journal.begin(2, Some(position(65))).unwrap();
        journal.mark_submitted(&second).unwrap();
        assert_eq!(journal.position, LanePosition::default());
        assert_eq!(journal.rewind(64), Err(PagePoolError::InvalidCheckpoint));
        pool.observe_completion(1).unwrap();
        journal.finish(&first, true).unwrap();
        assert_eq!(journal.position, position(64));
        assert!(journal.rewind(0).is_err());
        pool.observe_completion(2).unwrap();
        journal.finish(&second, true).unwrap();
        assert_eq!(journal.position, position(65));
        assert_eq!(
            journal.restore_with(&predicted, || Err(PagePoolError::InvalidCheckpoint)),
            Err(PagePoolError::InvalidCheckpoint)
        );
        assert_eq!(journal.position, position(65));
        journal.rewind(64).unwrap();
        assert_eq!(journal.position, position(64));
        journal.restore_with(&initial, || Ok(())).unwrap();
        assert_eq!(journal.position, LanePosition::default());
        journal.restore_with(&predicted, || Ok(())).unwrap();
        assert_eq!(journal.position, position(64));
        drop(initial);
        drop(predicted);
        let failed = journal.begin(3, Some(position(66))).unwrap();
        journal.mark_submitted(&failed).unwrap();
        let descendant = journal.begin(4, Some(position(67))).unwrap();
        journal.mark_submitted(&descendant).unwrap();
        pool.observe_completion(3).unwrap();
        journal.finish(&failed, false).unwrap();
        assert_eq!(journal.position, position(64));
        assert!(journal.begin(5, Some(position(68))).is_err());
        pool.observe_completion(4).unwrap();
        let retry = journal.begin(5, Some(position(66))).unwrap();
        journal.cancel_unsubmitted(&retry).unwrap();
        assert_eq!(journal.position, position(64));
    }
    #[test]
    fn retained_predecessor_restores_its_exact_feedback_association() {
        use crate::{Host, Limits, state::feedback::FeedbackBinding};
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let binding = FeedbackBinding::new(&host, 1).unwrap();
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut journal = StateJournal::new(pool.reserve().unwrap());
        journal.attach_feedback(binding.clone()).unwrap();
        for tokens in 1..=2 {
            let transaction = journal
                .begin(
                    tokens,
                    Some(crate::LanePosition {
                        tokens,
                        sampler: tokens - 1,
                        reserved_tokens: 7,
                        ..crate::LanePosition::default()
                    }),
                )
                .unwrap();
            journal.mark_submitted(&transaction).unwrap();
            pool.observe_completion(tokens).unwrap();
            journal.finish(&transaction, true).unwrap();
        }
        let predecessor = journal
            .retained
            .as_ref()
            .unwrap()
            .feedback
            .as_ref()
            .unwrap()
            .checkpoint
            .clone();
        let scalar = predecessor.scalar().index();
        let mut map = binding.versions.lock().unwrap();
        let selected = map.claim_scratch(1).unwrap();
        map.publish_selection(&predecessor, 1, &selected).unwrap();
        assert!(map.claim_scratch(1).is_err());
        drop(map);
        journal.rewind(1).unwrap();
        assert_eq!(journal.position.tokens, 1);
        assert_eq!(journal.position.sampler, 0);
        assert_eq!(
            journal
                .checkpoint_latest()
                .unwrap()
                .feedback
                .unwrap()
                .checkpoint
                .scalar()
                .index(),
            scalar
        );
    }
}
