use crate::BackendError;
use std::num::NonZeroU32;

/// Guest-declared sparse checkpoints within a canonical prompt trajectory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalPrefixSchedule {
    block: NonZeroU32,
    interval: NonZeroU32,
}
impl CanonicalPrefixSchedule {
    /// Validates an interval made of complete numerical blocks.
    ///
    /// # Errors
    /// Refuses an interval that splits a numerical block.
    pub fn new(block: NonZeroU32, interval: NonZeroU32) -> Result<Self, BackendError> {
        if !interval.get().is_multiple_of(block.get()) {
            return Err(BackendError::InvalidInput);
        }
        Ok(Self { block, interval })
    }
    /// Next sparse boundary, leaving the final prompt token for replay.
    ///
    /// # Errors
    /// Refuses overflowing positions.
    pub fn next(self, position: u64, prompt_tokens: u64) -> Result<Option<u64>, BackendError> {
        let block = u64::from(self.block.get());
        let last = prompt_tokens.saturating_sub(1) / block * block;
        if position >= last {
            return Ok(None);
        }
        let interval = u64::from(self.interval.get());
        let periodic = (position / interval)
            .checked_add(1)
            .and_then(|n| n.checked_mul(interval));
        Ok(Some(periodic.map_or(last, |periodic| periodic.min(last))))
    }
    /// Whether a completed prompt position must be saved.
    #[must_use]
    pub fn contains(self, position: u64, prompt_tokens: u64) -> bool {
        position > 0 && self.next(position - 1, prompt_tokens) == Ok(Some(position))
    }
    /// Clips a proposed piece so its internal save boundary becomes an explicit completion.
    ///
    /// # Errors
    /// Refuses invalid spans or nonfinal pieces that split a numerical block.
    pub fn limit(
        self,
        position: u64,
        prompt_tokens: u64,
        proposed: u32,
    ) -> Result<u32, BackendError> {
        let end = position
            .checked_add(u64::from(proposed))
            .filter(|&end| proposed > 0 && end <= prompt_tokens)
            .ok_or(BackendError::InvalidInput)?;
        let end = self
            .next(position, prompt_tokens)?
            .map_or(end, |boundary| end.min(boundary));
        if !position.is_multiple_of(u64::from(self.block.get()))
            || (end < prompt_tokens && !end.is_multiple_of(u64::from(self.block.get())))
        {
            return Err(BackendError::InvalidInput);
        }
        u32::try_from(end - position).map_err(|_| BackendError::InvalidInput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_boundaries_leave_the_tail_and_clip_internal_boundaries() {
        let schedule = CanonicalPrefixSchedule::new(
            NonZeroU32::new(64).unwrap(),
            NonZeroU32::new(4096).unwrap(),
        )
        .unwrap();
        for (length, expected) in [
            (63, None),
            (64, None),
            (65, Some(64)),
            (4096, Some(4032)),
            (4097, Some(4096)),
            (4161, Some(4096)),
        ] {
            assert_eq!(schedule.next(0, length).unwrap(), expected);
        }
        assert_eq!(schedule.next(4096, 4161).unwrap(), Some(4160));
        assert_eq!(schedule.limit(2048, 4096, 2048).unwrap(), 1984);
        assert_eq!(schedule.limit(4032, 4096, 64).unwrap(), 64);
        assert_eq!(schedule.limit(0, 65, 65).unwrap(), 64);
        assert_eq!(schedule.limit(64, 65, 1).unwrap(), 1);
        assert!(schedule.contains(4096, 4097));
        assert!(!schedule.contains(64, 64));
        assert!(schedule.limit(1, 65, 64).is_err());
        assert!(schedule.limit(0, 129, 7).is_err());
        assert!(schedule.limit(u64::MAX, u64::MAX, 1).is_err());
        assert!(
            CanonicalPrefixSchedule::new(
                NonZeroU32::new(64).unwrap(),
                NonZeroU32::new(4097).unwrap()
            )
            .is_err()
        );
    }
}
