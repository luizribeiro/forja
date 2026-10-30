use super::{Arc, AtomicU32, Ordering, PagePoolError, StateSequence};

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
        self.pins[self.committed as usize]
            .try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1)
            })
            .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        Ok(StateCheckpoint {
            identity: Arc::clone(&self.identity),
            pin: Arc::new(VersionPin {
                counts: Arc::clone(&self.pins),
                version: self.committed,
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
        self.committed = checkpoint.version();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, StateLayout, StatePlane, StatePool};

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
