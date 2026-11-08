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
}
