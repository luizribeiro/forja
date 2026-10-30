use super::{Arc, AtomicBool, AtomicU32, Ordering, PagePoolError, StateSequence, StateTransaction};

struct VersionPin {
    counts: Arc<[AtomicU32; 3]>,
    version: u32,
}

impl Drop for VersionPin {
    fn drop(&mut self) {
        self.counts[self.version as usize].fetch_sub(1, Ordering::AcqRel);
    }
}

/// Retained predecessor of one sequence; its version cannot be overwritten while pinned.
#[derive(Clone)]
pub struct StateCheckpoint {
    identity: Arc<()>,
    pin: Arc<VersionPin>,
    published: Option<Arc<AtomicBool>>,
}

impl StateCheckpoint {
    /// Version retained for rollback after physical descendants drain.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.pin.version
    }
}

impl StateSequence {
    /// Pins the last successfully published state without copying its planes.
    ///
    /// # Errors
    /// Refuses exhausted checkpoint reference counts.
    pub fn checkpoint(&self) -> Result<StateCheckpoint, PagePoolError> {
        self.pin(self.committed)
    }

    /// Pins the newest declared output, restoring it only after successful publication.
    ///
    /// # Errors
    /// Refuses exhausted checkpoint reference counts.
    pub fn checkpoint_latest(&self) -> Result<StateCheckpoint, PagePoolError> {
        self.pin(
            self.pending
                .back()
                .map_or(self.committed, |transaction| transaction.output),
        )
    }

    /// Replaces a retained checkpoint only when a new reservation succeeds.
    ///
    /// # Errors
    /// Refuses foreign checkpoints, unavailable versions or invalid transaction serials.
    pub fn begin_releasing_checkpoint(
        &mut self,
        serial: u64,
        checkpoint: &mut Option<StateCheckpoint>,
    ) -> Result<StateTransaction, PagePoolError> {
        if checkpoint
            .as_ref()
            .is_some_and(|checkpoint| !Arc::ptr_eq(&self.identity, &checkpoint.identity))
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let previous = checkpoint.take().map(|checkpoint| {
            let bookmark = (checkpoint.version(), checkpoint.published.clone());
            drop(checkpoint);
            bookmark
        });
        match self.begin(serial) {
            Ok(transaction) => Ok(transaction),
            Err(error) => {
                if let Some((version, published)) = previous {
                    let mut restored = self.pin(version)?;
                    restored.published = published;
                    *checkpoint = Some(restored);
                }
                Err(error)
            }
        }
    }

    /// Retains the input version of a live speculative transaction.
    ///
    /// # Errors
    /// Refuses foreign/retired transactions or exhausted checkpoint reference counts.
    pub fn predecessor(
        &self,
        transaction: &StateTransaction,
    ) -> Result<StateCheckpoint, PagePoolError> {
        self.pending_index(transaction)?;
        self.pin(transaction.input)
    }

    fn pin(&self, version: u32) -> Result<StateCheckpoint, PagePoolError> {
        self.pins[version as usize]
            .try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1)
            })
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        Ok(StateCheckpoint {
            identity: Arc::clone(&self.identity),
            published: self
                .pending
                .iter()
                .find(|transaction| transaction.output == version)
                .map(|transaction| Arc::clone(&transaction.published)),
            pin: Arc::new(VersionPin {
                counts: Arc::clone(&self.pins),
                version,
            }),
        })
    }

    /// Restores a retained predecessor after every speculative reader/writer has drained.
    ///
    /// # Errors
    /// Refuses foreign checkpoints, pending transactions or quarantined device work.
    pub fn rewind(&mut self, checkpoint: &StateCheckpoint) -> Result<(), PagePoolError> {
        if !Arc::ptr_eq(&self.identity, &checkpoint.identity) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        if !self.pending.is_empty()
            || self.pool.completed.load(Ordering::Acquire)
                < self.drain_until.max(self.quarantine_until)
        {
            return Err(PagePoolError::BankInFlight);
        }
        if checkpoint
            .published
            .as_ref()
            .is_some_and(|published| !published.load(Ordering::Acquire))
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        self.committed = checkpoint.version();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, StateLayout, StatePlane, StatePool};

    #[test]
    fn cloned_checkpoint_keeps_its_version_pinned_during_reservation() {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut sequence = pool.reserve().unwrap();
        let mut retained = Some(sequence.checkpoint().unwrap());
        let caller = retained.as_ref().unwrap().clone();
        let first = sequence.begin(1).unwrap();
        sequence.mark_submitted(&first).unwrap();
        pool.observe_completion(1).unwrap();
        sequence.finish(&first, true).unwrap();
        let second = sequence
            .begin_releasing_checkpoint(2, &mut retained)
            .unwrap();
        assert_ne!(second.output, caller.version());
        assert!(retained.is_none());
        sequence.cancel_unsubmitted(&second).unwrap();
        sequence.rewind(&caller).unwrap();
    }

    #[test]
    fn speculative_checkpoints_require_successful_publication() {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        for outcome in [0, 1, 2] {
            let mut sequence = pool.reserve().unwrap();
            let first = sequence.begin(1).unwrap();
            let checkpoint = sequence.checkpoint_latest().unwrap();
            let second = sequence.begin(2).unwrap();
            let predecessor = sequence.predecessor(&second).unwrap();
            assert_eq!(checkpoint.version(), predecessor.version());
            if outcome == 0 {
                sequence.cancel_unsubmitted(&first).unwrap();
            } else {
                sequence.mark_submitted(&first).unwrap();
                sequence.mark_submitted(&second).unwrap();
                pool.observe_completion(2).unwrap();
                sequence.finish(&first, outcome == 2).unwrap();
                if outcome == 2 {
                    sequence.finish(&second, true).unwrap();
                }
            }
            for checkpoint in [&checkpoint, &predecessor] {
                let restored = sequence.rewind(checkpoint);
                if outcome == 2 {
                    restored.unwrap();
                } else {
                    assert_eq!(restored, Err(PagePoolError::InvalidCheckpoint));
                }
            }
        }
    }

    #[test]
    fn queued_descendants_retain_their_own_input_boundary() {
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut sequence = pool.reserve().unwrap();
        let first = sequence.begin(1).unwrap();
        sequence.mark_submitted(&first).unwrap();
        let second = sequence.begin(2).unwrap();
        let predecessor = sequence.predecessor(&second).unwrap();
        sequence.mark_submitted(&second).unwrap();
        assert_eq!(
            sequence.rewind(&predecessor),
            Err(PagePoolError::BankInFlight)
        );
        pool.observe_completion(2).unwrap();
        sequence.finish(&first, true).unwrap();
        sequence.finish(&second, true).unwrap();
        sequence.rewind(&predecessor).unwrap();
        assert_eq!(sequence.begin(3).unwrap().input_version(), first.output);
        assert!(sequence.predecessor(&second).is_err());
    }

    #[test]
    fn retained_predecessors_survive_updates_and_refuse_early_reuse() {
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap();
        let pool = StatePool::new(layout, 2, 65536).unwrap();
        let mut sequence = pool.reserve().unwrap();
        let predecessor = sequence.checkpoint().unwrap();
        let first = sequence.begin(1).unwrap();
        sequence.mark_submitted(&first).unwrap();
        assert_eq!(
            sequence.rewind(&predecessor),
            Err(PagePoolError::BankInFlight)
        );
        pool.observe_completion(1).unwrap();
        sequence.finish(&first, true).unwrap();
        let second = sequence.begin(2).unwrap();
        assert_eq!(second.output, 2);
        assert_eq!(sequence.begin(3).err(), Some(PagePoolError::BankInFlight));
        sequence.cancel_unsubmitted(&second).unwrap();
        sequence.rewind(&predecessor).unwrap();
        assert_eq!(
            sequence.begin(3).unwrap().input_version(),
            predecessor.version()
        );
        let mut other = pool.reserve().unwrap();
        assert_eq!(
            other.rewind(&predecessor),
            Err(PagePoolError::InvalidCheckpoint)
        );
    }
}
