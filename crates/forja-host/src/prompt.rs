//! Stable prompt pieces within a scheduler's variable work allowance.

use std::num::NonZeroU32;

/// Fixed per-request boundaries for prompt execution.
#[derive(Clone, Copy, Debug)]
pub struct PromptChunking {
    piece: NonZeroU32,
}

impl PromptChunking {
    /// Pins boundaries to this many tokens from the start of a prompt suffix.
    #[must_use]
    pub const fn new(piece: NonZeroU32) -> Self {
        Self { piece }
    }

    /// Returns work that fits without cutting a piece before the suffix ends.
    #[must_use]
    pub const fn work(self, remaining: u32, allowance: u32) -> u32 {
        if remaining <= allowance {
            remaining
        } else {
            allowance / self.piece.get() * self.piece.get()
        }
    }

    /// Merges full pieces without absorbing the final shorter tail.
    #[must_use]
    pub const fn merged_work(self, remaining: u32, allowance: u32) -> u32 {
        let available = if remaining < allowance {
            remaining
        } else {
            allowance
        };
        if remaining < self.piece.get() {
            available
        } else {
            available / self.piece.get() * self.piece.get()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowances_preserve_pieces_and_allow_a_final_tail() {
        let chunks = PromptChunking::new(NonZeroU32::new(512).unwrap());
        assert_eq!(chunks.work(4097, 2047), 1536);
        assert_eq!(chunks.work(4097, 511), 0);
        assert_eq!(chunks.work(33, 511), 33);
    }
    #[test]
    fn merged_work_preserves_the_final_tail() {
        let chunks = PromptChunking::new(NonZeroU32::new(512).unwrap());
        assert_eq!(chunks.merged_work(2752, 2048), 2048);
        assert_eq!(chunks.merged_work(704, 2048), 512);
        assert_eq!(chunks.merged_work(192, 2048), 192);
        assert_eq!(chunks.merged_work(2048, 1537), 1536);
        assert_eq!(chunks.merged_work(512, 511), 0);
    }
}
