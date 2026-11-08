use super::{Arc, AtomicU32, Ordering, PagePoolError};

pub(super) struct CapacityLease(Arc<AtomicU32>);

impl CapacityLease {
    pub(super) fn reserve(used: &Arc<AtomicU32>, maximum: u32) -> Result<Self, PagePoolError> {
        used.try_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_add(1).filter(|&count| count <= maximum)
        })
        .map_err(|_| PagePoolError::QuotaExceeded)?;
        Ok(Self(Arc::clone(used)))
    }
}

impl Drop for CapacityLease {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, StateLayout, StatePlane, StatePool};

    #[test]
    fn spare_records_do_not_make_an_empty_sequence_pool_valid() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        for spares in [0, 1, 7] {
            assert_eq!(
                StatePool::with_spares(layout.clone(), 0, spares, 65536).err(),
                Some(PagePoolError::EmptySlab)
            );
        }
    }

    #[test]
    fn spare_backing_does_not_increase_sequence_capacity() {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let pool = StatePool::with_spares(layout, 2, 7, 65536).unwrap();
        assert_eq!(pool.maximum_versions(), 10);
        let first = pool.reserve().unwrap();
        let _second = pool.reserve().unwrap();
        assert_eq!(pool.reserve().err(), Some(PagePoolError::QuotaExceeded));
        drop(first);
        pool.reserve().unwrap();
        assert_eq!(StatePool::record_count_with_spares(2, 7), Ok(13));
        assert_eq!(
            StatePool::record_count_with_spares(u32::MAX / 3, 1),
            Err(PagePoolError::ArithmeticOverflow)
        );
    }
}
