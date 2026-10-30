use super::StateRecording;
use forja_core::{Backend, PagePoolError, PagedSequence, SequenceCheckpoint};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct PagedInput {
    sequence: Arc<Mutex<PagedSequence>>,
    checkpoint: SequenceCheckpoint,
    dirty: [Arc<AtomicBool>; 2],
}

impl PagedInput {
    pub(super) const fn tokens(&self) -> u64 {
        self.checkpoint.logical_len()
    }

    pub(super) fn restore(self) -> Result<(), PagePoolError> {
        for dirty in &self.dirty {
            dirty.store(true, Ordering::Release);
        }
        self.sequence
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .rollback(self.checkpoint)
    }
}

impl<B: Backend> StateRecording<B> {
    /// Couples the input KV checkpoint to this transaction's private state boundary.
    /// Both slot tables become dirty whenever completion or cancellation restores it.
    ///
    /// # Errors
    /// Refuses duplicate checkpoints or mismatched state and KV token boundaries.
    pub fn pair_paged_checkpoint(
        &self,
        sequence: Arc<Mutex<PagedSequence>>,
        checkpoint: SequenceCheckpoint,
        dirty: [Arc<AtomicBool>; 2],
    ) -> Result<(), PagePoolError> {
        if !checkpoint.belongs_to(&*sequence.lock().map_err(|_| PagePoolError::CorruptState)?) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.sequence.lock().attach_paged(
            self.serial(),
            PagedInput {
                sequence,
                checkpoint,
                dirty,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{LanePosition, journal::StateJournal};
    use forja_core::{
        DType, PagePool, PagedLayout, PagedPlane, StateLayout, StatePlane, StatePool,
    };

    fn paired_retirement(fail: bool) {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut journal = StateJournal::new(pool.reserve().unwrap());
        let kv_pool = PagePool::new(
            PagedLayout::new(
                4,
                vec![PagedPlane {
                    dtype: DType::F32,
                    elements_per_token: 7,
                    alignment: 4,
                    label: "kv".into(),
                }],
            )
            .unwrap(),
            4,
            4096,
        )
        .unwrap();
        let kv = Arc::new(Mutex::new(kv_pool.reserve(16).unwrap()));
        let dirty = std::array::from_fn(|_| Arc::new(AtomicBool::new(false)));
        let mut transactions = Vec::new();
        for tokens in 1..=2 {
            let checkpoint = kv.lock().unwrap().checkpoint().unwrap();
            kv.lock().unwrap().cow_tail().unwrap();
            kv.lock().unwrap().commit_write(1).unwrap();
            let transaction = journal
                .begin(
                    tokens,
                    Some(LanePosition {
                        tokens,
                        sampler: tokens - 1,
                        grammar: tokens + 7,
                        parser: tokens + 33,
                        reserved_tokens: 16,
                    }),
                )
                .unwrap();
            journal.mark_submitted(&transaction).unwrap();
            journal
                .attach_paged(
                    tokens,
                    PagedInput {
                        sequence: Arc::clone(&kv),
                        checkpoint,
                        dirty: dirty.clone(),
                    },
                )
                .unwrap();
            transactions.push(transaction);
        }
        pool.observe_completion(1).unwrap();
        journal.finish(&transactions[0], !fail).unwrap();
        assert_eq!(kv.lock().unwrap().logical_len(), 2);
        assert!(dirty.iter().all(|flag| !flag.load(Ordering::Acquire)));
        pool.observe_completion(2).unwrap();
        if fail {
            assert!(journal.abandon(&transactions[1]).is_err());
            assert_eq!(journal.position, LanePosition::default());
            assert_eq!(kv.lock().unwrap().logical_len(), 0);
        } else {
            journal.finish(&transactions[1], true).unwrap();
            journal.rewind(1).unwrap();
            assert_eq!(journal.position.tokens, 1);
            assert_eq!(journal.position.grammar, 8);
            assert_eq!(journal.position.parser, 34);
            assert_eq!(kv.lock().unwrap().logical_len(), 1);
        }
        assert!(dirty.iter().all(|flag| flag.load(Ordering::Acquire)));
    }

    #[test]
    fn failed_state_waits_for_descendant_drain_before_restoring_kv() {
        paired_retirement(true);
    }

    #[test]
    fn retained_state_rewind_restores_kv_and_all_consumer_cursors() {
        paired_retirement(false);
    }
}
