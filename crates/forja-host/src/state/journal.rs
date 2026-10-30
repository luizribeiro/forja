use super::outcome::StateOutcome;
use forja_core::{PagePoolError, StateCheckpoint, StateSequence, StateTransaction};
use std::sync::Arc;
use std::{
    collections::VecDeque,
    ops::{Deref, DerefMut},
};

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

struct Pending {
    serial: u64,
    position: LanePosition,
    predecessor: StateCheckpoint,
    input: LanePosition,
    outcome: Arc<StateOutcome>,
    dependency: Option<Arc<StateOutcome>>,
}

pub(super) struct StateJournal {
    sequence: StateSequence,
    pub(super) position: LanePosition,
    pending: VecDeque<Pending>,
    retained: Option<(StateCheckpoint, LanePosition)>,
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
            sequence,
            position: LanePosition::default(),
            pending: VecDeque::new(),
            retained: None,
        }
    }

    pub(super) fn begin(
        &mut self,
        serial: u64,
        position: Option<LanePosition>,
    ) -> Result<StateTransaction, PagePoolError> {
        let input = self
            .pending
            .back()
            .map_or(self.position, |entry| entry.position);
        let position = position.unwrap_or(input);
        if position.tokens > position.reserved_tokens {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let checkpoint = self.sequence.checkpoint_latest()?;
        let (mut retained, position_before) = self
            .retained
            .take()
            .map_or((None, None), |(checkpoint, position)| {
                (Some(checkpoint), Some(position))
            });
        let transaction = match self
            .sequence
            .begin_releasing_checkpoint(serial, &mut retained)
        {
            Ok(transaction) => transaction,
            Err(error) => {
                self.retained = retained.zip(position_before);
                return Err(error);
            }
        };
        let dependency = self.pending.back().map(|entry| Arc::clone(&entry.outcome));
        self.pending.push_back(Pending {
            serial,
            position,
            predecessor: checkpoint,
            input,
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
        self.sequence.cancel_unsubmitted(transaction)?;
        for entry in self.pending.drain(index..) {
            entry.outcome.finish(false);
        }
        Ok(())
    }

    pub(super) fn abandon(&mut self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.sequence.abandon(transaction)?;
        for entry in self.pending.drain(..) {
            entry.outcome.finish(false);
        }
        Ok(())
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
                self.retained = Some((entry.predecessor, entry.input));
                entry.outcome.finish(true);
            }
        } else {
            for entry in self.pending.drain(..) {
                entry.outcome.finish(false);
            }
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

    pub(super) fn checkpoint_latest(&self) -> Result<super::StateLaneCheckpoint, PagePoolError> {
        Ok(super::StateLaneCheckpoint {
            version: self.sequence.checkpoint_latest()?,
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
        let previous = self.sequence.checkpoint()?;
        self.sequence.rewind(&checkpoint.version)?;
        if let Err(error) = action() {
            self.sequence.rewind(&previous)?;
            return Err(error);
        }
        self.position = checkpoint.position;
        self.retained = None;
        Ok(())
    }

    pub(super) fn rewind(&mut self, tokens: u64) -> Result<(), PagePoolError> {
        let (checkpoint, position) = self
            .retained
            .as_ref()
            .filter(|(_, position)| position.tokens == tokens)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        self.sequence.rewind(checkpoint)?;
        self.position = *position;
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
        assert_eq!(journal.retained.as_ref().unwrap().1.tokens, 0);
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
}
