use super::{CapacityLease, PagePoolError, StatePool, StateSequence};

impl StateSequence {
    /// Number of currently owned versions, including any exclusively reserved spare records.
    #[must_use]
    pub fn version_count(&self) -> u32 {
        if self.spare.is_some() {
            self.pool.maximum_versions()
        } else {
            StatePool::VERSIONS
        }
    }

    /// Atomically claims the pool's spare records for this physically drained sequence.
    ///
    /// # Errors
    /// Refuses outstanding work, another spare owner or insufficient allocation quota.
    pub fn reserve_spares(&mut self) -> Result<(), PagePoolError> {
        self.ensure_idle()?;
        if self.spare.is_some() || self.pool.spare_records == 0 {
            return Ok(());
        }
        let owner = CapacityLease::reserve(&self.pool.spare_owner, 1)?;
        let storage = self
            .pool
            .pages
            .reserve(u64::from(self.pool.spare_records))?;
        self.spare = Some((storage, owner));
        Ok(())
    }

    /// Returns spare records after their writers and retained boundaries have drained.
    ///
    /// # Errors
    /// Refuses outstanding work or a spare still used as committed state or by a checkpoint.
    pub fn release_spares(&mut self) -> Result<(), PagePoolError> {
        self.ensure_idle()?;
        if self.committed >= StatePool::VERSIONS
            || self.pins[StatePool::VERSIONS as usize..]
                .iter()
                .any(|pin| pin.load(super::Ordering::Acquire) != 0)
        {
            return Err(PagePoolError::BankInFlight);
        }
        self.spare = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, StateAccess, StateLayout, StatePlane};

    fn pool() -> StatePool {
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap();
        StatePool::with_spares(layout, 2, 1, 65536).unwrap()
    }

    #[test]
    fn simultaneous_spare_requests_have_one_owner_and_no_shared_records() {
        use std::sync::{Arc, Barrier};
        let pool = pool();
        let sequences = [pool.reserve().unwrap(), pool.reserve().unwrap()];
        let start = Arc::new(Barrier::new(2));
        let results = std::thread::scope(|scope| {
            sequences
                .map(|mut sequence| {
                    let start = Arc::clone(&start);
                    scope.spawn(move || {
                        start.wait();
                        (sequence.reserve_spares(), sequence)
                    })
                })
                .map(|job| job.join().unwrap())
        });
        assert_eq!(
            results.iter().filter(|(result, _)| result.is_ok()).count(),
            1
        );
        assert_eq!(
            results
                .iter()
                .filter(|(result, _)| *result == Err(PagePoolError::QuotaExceeded))
                .count(),
            1
        );
        let records = results
            .iter()
            .flat_map(|(_, sequence)| {
                sequence
                    .allocation_table()
                    .unwrap()
                    .entries()
                    .iter()
                    .map(|entry| entry.page())
                    .collect::<Vec<_>>()
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(records.len(), 7);
        drop(results);
        pool.reserve().unwrap().reserve_spares().unwrap();
    }

    #[test]
    fn exclusive_spares_extend_only_the_owning_sequence_chain() {
        let pool = pool();
        let mut first = pool.reserve().unwrap();
        let mut second = pool.reserve().unwrap();
        first.reserve_spares().unwrap();
        first.reserve_spares().unwrap();
        assert_eq!(first.version_count(), 4);
        assert_eq!(second.version_count(), 3);
        assert_eq!(second.reserve_spares(), Err(PagePoolError::QuotaExceeded));
        assert_eq!(
            second.begin_chain(1, 3).err(),
            Some(PagePoolError::BankInFlight)
        );
        let transaction = first.begin_chain(1, 3).unwrap();
        let access = StateAccess::from_slot(&first.bind(&transaction).unwrap()).unwrap();
        assert_eq!(access.chain_len(), 3);
        assert_eq!(first.reserve_spares(), Err(PagePoolError::BankInFlight));
        first.mark_submitted(&transaction).unwrap();
        drop(first);
        assert_eq!(second.reserve_spares(), Err(PagePoolError::QuotaExceeded));
        pool.observe_completion(1).unwrap();
        second.reserve_spares().unwrap();
        assert_eq!(second.version_count(), 4);
    }

    #[test]
    fn spare_release_preserves_committed_state_and_retained_boundaries() {
        let pool = pool();
        let mut owner = pool.reserve().unwrap();
        let mut waiting = pool.reserve().unwrap();
        owner.reserve_spares().unwrap();
        let chain = owner.begin_chain(1, 3).unwrap();
        let boundary = owner.checkpoint_latest().unwrap();
        assert_eq!(owner.release_spares(), Err(PagePoolError::BankInFlight));
        owner.mark_submitted(&chain).unwrap();
        pool.observe_completion(1).unwrap();
        owner.finish(&chain, true).unwrap();
        assert_eq!(owner.release_spares(), Err(PagePoolError::BankInFlight));
        let plain = owner.begin(2).unwrap();
        assert!(plain.output_version() < StatePool::VERSIONS);
        owner.mark_submitted(&plain).unwrap();
        pool.observe_completion(2).unwrap();
        owner.finish(&plain, true).unwrap();
        assert_eq!(owner.release_spares(), Err(PagePoolError::BankInFlight));
        drop(boundary);
        owner.release_spares().unwrap();
        owner.release_spares().unwrap();
        assert_eq!(owner.version_count(), StatePool::VERSIONS);
        assert_eq!(
            owner.begin_chain(3, 3).err(),
            Some(PagePoolError::BankInFlight)
        );
        waiting.reserve_spares().unwrap();
        let next = owner.begin(3).unwrap();
        owner.cancel_unsubmitted(&next).unwrap();
    }

    #[test]
    fn cancelled_failed_and_abandoned_spares_preserve_pins_and_drain() {
        for outcome in 0..3 {
            let pool = pool();
            let mut owner = pool.reserve().unwrap();
            let mut waiting = pool.reserve().unwrap();
            owner.reserve_spares().unwrap();
            let chain = owner.begin_chain(1, 3).unwrap();
            let boundary = owner.checkpoint_latest().unwrap();
            if outcome == 0 {
                owner.cancel_unsubmitted(&chain).unwrap();
            } else {
                owner.mark_submitted(&chain).unwrap();
                if outcome == 1 {
                    pool.observe_completion(1).unwrap();
                    owner.finish(&chain, false).unwrap();
                } else {
                    owner.abandon(&chain).unwrap();
                    assert_eq!(owner.release_spares(), Err(PagePoolError::BankInFlight));
                    pool.observe_completion(1).unwrap();
                }
            }
            assert_eq!(owner.release_spares(), Err(PagePoolError::BankInFlight));
            assert_eq!(
                owner.rewind(&boundary),
                Err(PagePoolError::InvalidCheckpoint)
            );
            drop(boundary);
            owner.release_spares().unwrap();
            waiting.reserve_spares().unwrap();
            assert_eq!(owner.checkpoint().unwrap().version(), 0);
        }
    }
}
