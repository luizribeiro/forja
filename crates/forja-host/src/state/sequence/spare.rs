use super::{Backend, PagePoolError, StateSequence, StateSequenceHandle};
use crate::{compute, state::pool_error};

fn spare_refusal(error: PagePoolError) -> Result<bool, compute::Error> {
    match error {
        PagePoolError::QuotaExceeded | PagePoolError::BankInFlight => Ok(false),
        error => Err(pool_error(error)),
    }
}

impl<B: Backend> StateSequenceHandle<B> {
    /// Releases shared records, returning false while work or retained state still needs them.
    ///
    /// # Errors
    /// Returns storage or metadata failures; a busy reservation can be retried after plain decoding.
    pub fn try_release_spares(&self) -> Result<bool, compute::Error> {
        self.storage.check_usable()?;
        let mut journal = self.lock();
        if let Err(error) = journal
            .ensure_snapshot_idle()
            .and_then(|()| journal.release_spares_with_feedback())
        {
            return spare_refusal(error);
        }
        Ok(true)
    }

    /// Claims and clears shared spare records, returning false to continue plain decoding on refusal.
    ///
    /// # Errors
    /// Returns storage, backend or metadata failures; quota and busy records are ordinary fallback.
    pub fn try_reserve_spares(&self) -> Result<bool, compute::Error> {
        self.storage.check_usable()?;
        let mut journal = self.lock();
        let owned = journal.version_count();
        if let Err(error) = journal
            .ensure_snapshot_idle()
            .and_then(|()| StateSequence::reserve_spares(&mut journal))
        {
            return spare_refusal(error);
        }
        if journal.version_count() == owned {
            return Ok(true);
        }
        let initialized = (|| {
            let table = journal.allocation_table().map_err(pool_error)?;
            let owned = usize::try_from(owned).map_err(crate::invalid_handle)?;
            let entries = table
                .entries()
                .get(owned..)
                .ok_or_else(|| crate::invalid_handle("missing spare state records"))?;
            self.storage.clear_records(entries)?;
            journal.refresh_feedback().map_err(pool_error)
        })();
        if let Err(error) = initialized {
            journal.release_spares_with_feedback().map_err(pool_error)?;
            return match error {
                compute::Error::Quota(_) => Ok(false),
                error => Err(error),
            };
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, Limits, state::feedback::FeedbackBinding};
    use forja_core::{DType, StateLayout, StatePlane};
    use forja_cpu::CpuBackend;

    fn fixture() -> (Host<CpuBackend>, [StateSequenceHandle<CpuBackend>; 2]) {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage_with_spares(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
                2,
                1,
            )
            .unwrap();
        let feedback = FeedbackBinding::with_spares(&host, 2, 1).unwrap();
        let sequences = std::array::from_fn(|_| {
            let sequence = storage.reserve().unwrap();
            sequence.lock().attach_feedback(feedback.clone()).unwrap();
            sequence
        });
        (host, sequences)
    }

    #[test]
    fn host_spare_admission_refreshes_feedback_and_preserves_plain_fallback() {
        let (_host, [first, second]) = fixture();
        let pending = first.begin(1).unwrap();
        assert!(!first.try_reserve_spares().unwrap());
        assert!(!first.try_release_spares().unwrap());
        first.cancel(&pending).unwrap();
        assert!(first.try_reserve_spares().unwrap());
        assert!(first.try_reserve_spares().unwrap());
        assert!(!second.try_reserve_spares().unwrap());
        assert_eq!(second.lock().version_count(), 3);
        let plain = second.begin(1).unwrap();
        second.cancel(&plain).unwrap();
        let mut journal = first.lock();
        let chain = StateSequence::begin_chain(&mut journal, 2, 3).unwrap();
        drop(journal);
        assert!(!first.try_release_spares().unwrap());
        let mut journal = first.lock();
        let checkpoint = journal.checkpoint_latest().unwrap();
        assert_eq!(checkpoint.version.version(), 3);
        assert!(checkpoint.feedback.is_some());
        StateSequence::cancel_unsubmitted(&mut journal, &chain).unwrap();
    }

    #[test]
    fn spare_reassignment_clears_only_the_new_owners_records() {
        let (host, [first, second]) = fixture();
        let boundary = first.checkpoint_latest().unwrap();
        let base = first.lock().allocation_table().unwrap().entries()[0].page();
        let carry = first.storage.version_view(0, base).unwrap();
        let carry_bytes = 17_f32.to_le_bytes().repeat(7 * 33);
        host.backend.write(&carry, &carry_bytes).unwrap();
        assert!(first.try_reserve_spares().unwrap());
        let record = first.lock().allocation_table().unwrap().entries()[3].page();
        let view = first.storage.version_view(0, record).unwrap();
        let written = 9_f32.to_le_bytes().repeat(7 * 33);
        host.backend.write(&view, &written).unwrap();
        assert!(first.try_reserve_spares().unwrap());
        assert_eq!(host.backend.read(&view).unwrap(), written);
        assert!(first.try_release_spares().unwrap());
        assert_eq!(first.lock().version_count(), 3);
        assert_eq!(
            first.checkpoint_latest().unwrap().position(),
            boundary.position()
        );
        assert!(second.try_reserve_spares().unwrap());
        let reused = second.lock().allocation_table().unwrap().entries()[3].page();
        assert_eq!(record, reused);
        assert_eq!(host.backend.read(&view).unwrap(), vec![0; 7 * 33 * 4]);
        assert!(!first.try_reserve_spares().unwrap());
        assert!(second.try_release_spares().unwrap());
        assert!(first.try_reserve_spares().unwrap());
        assert_eq!(host.backend.read(&carry).unwrap(), carry_bytes);
    }

    #[test]
    fn initialization_quota_refusal_restores_spare_ownership_for_plain_fallback() {
        use crate::state::StateCharge;
        use std::sync::{Arc, atomic::Ordering};
        let (host, [first, second]) = fixture();
        let before = host.live_bytes.load(Ordering::Acquire);
        let budget = StateCharge::reserve(
            host.backend.as_ref(),
            Arc::clone(&host.live_bytes),
            host.limits.live_bytes,
            host.limits.live_bytes - before,
        )
        .unwrap();
        assert!(!first.try_reserve_spares().unwrap());
        assert_eq!(first.lock().version_count(), 3);
        drop(budget);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
        assert!(second.try_reserve_spares().unwrap());
        let plain = first.begin(1).unwrap();
        first.cancel(&plain).unwrap();
        assert_eq!(first.position(), super::super::LanePosition::default());
    }
}
