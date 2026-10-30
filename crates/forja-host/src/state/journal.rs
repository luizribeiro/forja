use forja_core::{PagePoolError, StateCheckpoint, StateSequence, StateTransaction};
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

type Pending = (u64, LanePosition, StateCheckpoint, LanePosition);

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
        let input = self.pending.back().map_or(self.position, |entry| entry.1);
        let position = position.unwrap_or(input);
        if position.tokens > position.reserved_tokens {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let transaction = self.sequence.begin(serial)?;
        let checkpoint = match self.sequence.predecessor(&transaction) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                self.sequence.cancel_unsubmitted(&transaction)?;
                return Err(error);
            }
        };
        self.pending
            .push_back((serial, position, checkpoint, input));
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
            .position(|entry| entry.0 == transaction.serial())
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        self.sequence.cancel_unsubmitted(transaction)?;
        self.pending.truncate(index);
        Ok(())
    }

    pub(super) fn abandon(&mut self, transaction: &StateTransaction) -> Result<(), PagePoolError> {
        self.sequence.abandon(transaction)?;
        self.pending.clear();
        Ok(())
    }

    pub(super) fn finish(
        &mut self,
        transaction: &StateTransaction,
        success: bool,
    ) -> Result<(), PagePoolError> {
        if self.pending.front().map(|entry| entry.0) != Some(transaction.serial()) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.sequence.finish(transaction, success)?;
        if success {
            if let Some((_, position, predecessor, input)) = self.pending.pop_front() {
                self.position = position;
                self.retained = Some((predecessor, input));
            }
        } else {
            self.pending.clear();
        }
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
        let first = journal.begin(1, Some(position(64))).unwrap();
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
        journal.rewind(64).unwrap();
        assert_eq!(journal.position, position(64));
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
