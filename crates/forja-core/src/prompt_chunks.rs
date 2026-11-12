use crate::BackendError;

/// Guest-selected maximum prompt pieces, keyed by the resulting context length.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptChunkPolicy(Vec<(u32, u32)>);

impl PromptChunkPolicy {
    /// Validates complete, ordered context coverage and nonincreasing positive chunks.
    /// # Errors
    /// Refuses gaps in the envelope, oversized chunks or unbounded rule counts.
    pub fn new(arms: Vec<(u32, u32)>, context: u32, maximum: u32) -> Result<Self, BackendError> {
        let mut previous = (0, maximum);
        if arms.is_empty() || arms.len() > 64 || arms.last().map(|a| a.0) != Some(context) {
            return Err(BackendError::InvalidInput);
        }
        for &(through, tokens) in &arms {
            if through <= previous.0 || tokens == 0 || tokens > previous.1 || tokens > through {
                return Err(BackendError::InvalidInput);
            }
            previous = (through, tokens);
        }
        Ok(Self(arms))
    }

    /// Returns the validated context endpoints and token limits.
    #[must_use]
    pub fn arms(&self) -> &[(u32, u32)] {
        &self.0
    }

    /// Clips a proposed piece using its worst resulting context.
    /// # Errors
    /// Refuses empty pieces, arithmetic overflow or contexts outside the declaration.
    pub fn limit(&self, start: u32, proposed: u32) -> Result<u32, BackendError> {
        let end = start
            .checked_add(proposed)
            .filter(|_| proposed > 0)
            .ok_or(BackendError::InvalidInput)?;
        self.0
            .iter()
            .find(|&&(through, _)| end <= through)
            .map(|&(_, tokens)| proposed.min(tokens))
            .ok_or(BackendError::InvalidInput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_ends_bound_crossings_tails_and_overflow() {
        let policy = PromptChunkPolicy::new(vec![(512, 128), (4097, 32)], 4097, 128).unwrap();
        for (start, proposed, expected) in
            [(0, 128, 128), (500, 128, 32), (512, 33, 32), (4096, 1, 1)]
        {
            assert_eq!(policy.limit(start, proposed), Ok(expected));
        }
        assert!(policy.limit(4097, 1).is_err());
        assert!(policy.limit(u32::MAX, 7).is_err());
        assert!(policy.limit(0, 0).is_err());
        for arms in [
            vec![],
            vec![(4096, 32)],
            vec![(512, 0), (4097, 32)],
            vec![(512, 32), (4097, 128)],
            vec![(512, 128), (512, 32), (4097, 32)],
        ] {
            assert!(PromptChunkPolicy::new(arms, 4097, 128).is_err());
        }
    }
}
