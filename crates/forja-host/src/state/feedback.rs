use super::StateCharge;
use crate::{Host, compute, invalid_handle};
use forja_core::{Backend, FeedbackCheckpoint, FeedbackVersions, PagePoolError, StateSequence};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct FeedbackBinding {
    pub(crate) versions: Arc<Mutex<FeedbackVersions>>,
    _charge: Arc<StateCharge>,
}

impl FeedbackBinding {
    pub(crate) fn new<B: Backend>(host: &Host<B>, lanes: u32) -> Result<Self, compute::Error> {
        let records = lanes
            .checked_mul(3)
            .ok_or_else(|| invalid_handle("feedback record overflow"))?;
        let scratch = lanes
            .checked_mul(2)
            .ok_or_else(|| invalid_handle("feedback scratch overflow"))?;
        let bytes = FeedbackVersions::metadata_bytes(records, scratch)
            .map_err(invalid_handle)?
            .checked_add(
                (size_of::<Mutex<FeedbackVersions>>()
                    + size_of::<StateCharge>()
                    + 4 * size_of::<usize>()) as u64,
            )
            .ok_or_else(|| invalid_handle("feedback metadata overflow"))?;
        let charge = StateCharge::reserve(
            host.backend.as_ref(),
            Arc::clone(&host.live_bytes),
            host.limits.live_bytes,
            bytes,
        )?;
        Ok(Self {
            versions: Arc::new(Mutex::new(
                FeedbackVersions::new(records, scratch).map_err(invalid_handle)?,
            )),
            _charge: Arc::new(charge),
        })
    }
}

pub(super) struct LaneFeedback {
    binding: FeedbackBinding,
    records: [u32; 3],
}

#[derive(Clone)]
pub(super) struct FeedbackBoundary {
    binding: FeedbackBinding,
    pub(super) checkpoint: FeedbackCheckpoint,
}

impl LaneFeedback {
    pub(super) fn new(
        binding: FeedbackBinding,
        sequence: &StateSequence,
    ) -> Result<Self, PagePoolError> {
        let table = sequence.allocation_table()?;
        let [a, b, c] = table.entries() else {
            return Err(PagePoolError::CorruptState);
        };
        Ok(Self {
            binding,
            records: [a.page(), b.page(), c.page()],
        })
    }

    pub(super) fn checkpoint(&self, version: u32) -> Result<FeedbackBoundary, PagePoolError> {
        let record = *self
            .records
            .get(usize::try_from(version).map_err(|_| PagePoolError::ArithmeticOverflow)?)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        Ok(FeedbackBoundary {
            checkpoint: self
                .binding
                .versions
                .lock()
                .map_err(|_| PagePoolError::CorruptState)?
                .checkpoint(record)?,
            binding: self.binding.clone(),
        })
    }

    pub(super) fn validate(
        &self,
        boundary: Option<&FeedbackBoundary>,
    ) -> Result<(), PagePoolError> {
        if boundary
            .is_none_or(|boundary| !Arc::ptr_eq(&self.binding.versions, &boundary.binding.versions))
        {
            return Err(PagePoolError::ForeignResource);
        }
        Ok(())
    }

    pub(super) fn restore(&self, boundary: &FeedbackBoundary) -> Result<(), PagePoolError> {
        self.validate(Some(boundary))?;
        self.binding
            .versions
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .restore(&boundary.checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Limits, state::journal::StateJournal};
    use forja_core::{DType, StateLayout, StatePlane, StatePool};

    #[test]
    fn lane_restore_preserves_feedback_when_paged_restore_refuses() {
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
        let input = journal.checkpoint_latest().unwrap();
        let before = input.feedback.as_ref().unwrap().checkpoint.scalar().index();
        let mut map = binding.versions.lock().unwrap();
        let selected = map.claim_scratch(0).unwrap();
        map.publish_selection(&input.feedback.as_ref().unwrap().checkpoint, 0, &selected)
            .unwrap();
        drop(map);
        assert!(
            journal
                .restore_with(&input, || Err(PagePoolError::InvalidCheckpoint))
                .is_err()
        );
        assert_eq!(
            journal
                .checkpoint_latest()
                .unwrap()
                .feedback
                .unwrap()
                .checkpoint
                .scalar()
                .index(),
            selected.index()
        );
        journal.restore_with(&input, || Ok(())).unwrap();
        assert_eq!(
            journal
                .checkpoint_latest()
                .unwrap()
                .feedback
                .unwrap()
                .checkpoint
                .scalar()
                .index(),
            before
        );
    }
}
