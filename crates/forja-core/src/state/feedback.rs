use crate::PagePoolError;
use std::sync::Arc;

/// Physical scalar retained by a reader, writer or feedback checkpoint.
#[derive(Clone, Debug)]
pub struct FeedbackPin(Arc<u32>);

impl FeedbackPin {
    /// Scalar offset in the host-owned feedback allocation.
    #[must_use]
    pub fn index(&self) -> u32 {
        *self.0
    }
}

/// Immutable feedback association at a complete state boundary.
#[derive(Clone, Debug)]
pub struct FeedbackCheckpoint {
    record: usize,
    scalar: FeedbackPin,
}

impl FeedbackCheckpoint {
    /// Scalar associated with this boundary, retained until the checkpoint is dropped.
    #[must_use]
    pub const fn scalar(&self) -> &FeedbackPin {
        &self.scalar
    }
}

/// A permutation of version scalars and temporary selection scalars.
/// Publication swaps ownership; a failed selection leaves every version unchanged.
pub struct FeedbackVersions {
    records: Vec<FeedbackPin>,
    scratch: Vec<FeedbackPin>,
}

impl FeedbackVersions {
    /// Required metadata charge, including each scalar's shared ownership allocation.
    ///
    /// # Errors
    /// Refuses zero geometry or overflow.
    pub fn metadata_bytes(records: u32, scratch: u32) -> Result<u64, PagePoolError> {
        if records == 0 || scratch == 0 {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        u64::from(records)
            .checked_add(u64::from(scratch))
            .and_then(|count| {
                count.checked_mul((size_of::<Arc<u32>>() + 3 * size_of::<usize>()) as u64)
            })
            .ok_or(PagePoolError::ArithmeticOverflow)
    }

    /// Constructs scalar associations after the caller reserves their metadata and backing bytes.
    ///
    /// # Errors
    /// Refuses zero geometry, scalar-count overflow or unaddressable metadata.
    pub fn new(records: u32, scratch: u32) -> Result<Self, PagePoolError> {
        let bytes = Self::metadata_bytes(records, scratch)?;
        usize::try_from(bytes).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let count = records
            .checked_add(scratch)
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let pin = |index| FeedbackPin(Arc::new(index));
        Ok(Self {
            records: (0..records).map(pin).collect(),
            scratch: (records..count).map(pin).collect(),
        })
    }

    /// Retains the feedback associated with one physical state record.
    ///
    /// # Errors
    /// Refuses an out-of-range record.
    pub fn checkpoint(&self, record: u32) -> Result<FeedbackCheckpoint, PagePoolError> {
        let record = usize::try_from(record).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        let scalar = self
            .records
            .get(record)
            .ok_or(PagePoolError::InvalidCheckpoint)?
            .clone();
        Ok(FeedbackCheckpoint { record, scalar })
    }

    /// Claims an unpinned destination version before declaring its new output boundary.
    ///
    /// # Errors
    /// Refuses invalid records or scalars retained by an earlier boundary or GPU user.
    pub fn claim_record(&self, record: u32) -> Result<FeedbackPin, PagePoolError> {
        Self::claim(&self.records, record)
    }

    /// Claims selection scratch while preserving every published feedback scalar.
    ///
    /// # Errors
    /// Refuses invalid scratch slots or scalars pinned by checkpoints or GPU users.
    pub fn claim_scratch(&self, scratch: u32) -> Result<FeedbackPin, PagePoolError> {
        Self::claim(&self.scratch, scratch)
    }

    fn claim(slots: &[FeedbackPin], index: u32) -> Result<FeedbackPin, PagePoolError> {
        let slot = slots
            .get(usize::try_from(index).map_err(|_| PagePoolError::ArithmeticOverflow)?)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        if Arc::strong_count(&slot.0) != 1 {
            return Err(PagePoolError::BankInFlight);
        }
        Ok(slot.clone())
    }

    /// Preflights selection publication while the caller holds all participating lane locks.
    ///
    /// # Errors
    /// Refuses foreign, stale or replaced version/scratch associations.
    pub fn validate_selection(
        &self,
        expected: &FeedbackCheckpoint,
        scratch: u32,
        selected: &FeedbackPin,
    ) -> Result<(), PagePoolError> {
        let scratch = usize::try_from(scratch).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        if !self
            .records
            .get(expected.record)
            .is_some_and(|pin| Arc::ptr_eq(&pin.0, &expected.scalar.0))
            || !self
                .scratch
                .get(scratch)
                .is_some_and(|pin| Arc::ptr_eq(&pin.0, &selected.0))
        {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        Ok(())
    }

    /// Publishes a successfully completed selection by swapping scalar associations.
    ///
    /// # Errors
    /// Refuses stale or foreign associations without changing any record.
    pub fn publish_selection(
        &mut self,
        expected: &FeedbackCheckpoint,
        scratch: u32,
        selected: &FeedbackPin,
    ) -> Result<(), PagePoolError> {
        self.validate_selection(expected, scratch, selected)?;
        let scratch = usize::try_from(scratch).map_err(|_| PagePoolError::ArithmeticOverflow)?;
        std::mem::swap(
            &mut self.records[expected.record],
            &mut self.scratch[scratch],
        );
        Ok(())
    }

    /// Restores the exact scalar of a retained boundary after all physical descendants drain.
    ///
    /// # Errors
    /// Refuses a checkpoint from another permutation without changing any record.
    pub fn restore(&mut self, checkpoint: &FeedbackCheckpoint) -> Result<(), PagePoolError> {
        if checkpoint.record >= self.records.len() {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        if let Some(index) = self
            .records
            .iter()
            .position(|pin| Arc::ptr_eq(&pin.0, &checkpoint.scalar.0))
        {
            self.records.swap(checkpoint.record, index);
        } else if let Some(index) = self
            .scratch
            .iter()
            .position(|pin| Arc::ptr_eq(&pin.0, &checkpoint.scalar.0))
        {
            std::mem::swap(
                &mut self.records[checkpoint.record],
                &mut self.scratch[index],
            );
        } else {
            return Err(PagePoolError::ForeignResource);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_keeps_checkpoint_feedback_immutable() {
        let mut versions = FeedbackVersions::new(3, 2).unwrap();
        let input = versions.checkpoint(0).unwrap();
        let selected = versions.claim_scratch(0).unwrap();
        let mut values = [11, 0, 0, 0, 0];
        values[selected.index() as usize] = 29;
        versions.publish_selection(&input, 0, &selected).unwrap();
        let output = versions.checkpoint(0).unwrap();
        assert_eq!(values[output.scalar().index() as usize], 29);
        assert_eq!(values[input.scalar().index() as usize], 11);
        assert!(matches!(
            versions.claim_scratch(0),
            Err(PagePoolError::BankInFlight)
        ));
        assert!(versions.publish_selection(&input, 0, &selected).is_err());
        versions.restore(&input).unwrap();
        assert_eq!(
            versions.checkpoint(0).unwrap().scalar().index(),
            input.scalar().index()
        );
        assert_eq!(values[output.scalar().index() as usize], 29);
        assert!(versions.claim_scratch(0).is_err());
        drop(selected);
        drop(output);
        assert!(versions.claim_scratch(0).is_ok());
        let foreign = FeedbackVersions::new(3, 2).unwrap().checkpoint(0).unwrap();
        assert!(versions.restore(&foreign).is_err());
        assert_eq!(
            versions.checkpoint(0).unwrap().scalar().index(),
            input.scalar().index()
        );
    }

    #[test]
    fn feedback_geometry_and_reader_pins_refuse_reuse() {
        assert!(FeedbackVersions::new(0, 2).is_err());
        assert!(FeedbackVersions::new(u32::MAX, 2).is_err());
        let versions = FeedbackVersions::new(3, 2).unwrap();
        let writer = versions.claim_record(2).unwrap();
        assert!(versions.claim_record(2).is_err());
        let boundary = versions.checkpoint(2).unwrap();
        assert_eq!(writer.index(), boundary.scalar().index());
        drop(writer);
        assert!(versions.claim_record(2).is_err());
        drop(boundary);
        assert!(versions.claim_record(2).is_ok());
        assert!(versions.claim_scratch(2).is_err());
    }
}
