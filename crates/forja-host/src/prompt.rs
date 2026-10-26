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

    /// Splits an aligned work allocation into stable execution pieces.
    pub fn pieces(self, tokens: &[u32]) -> impl Iterator<Item = &[u32]> {
        tokens.chunks(usize::try_from(self.piece.get()).unwrap_or(usize::MAX))
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
        assert_eq!(
            chunks
                .pieces(&vec![0; 2047])
                .map(<[u32]>::len)
                .collect::<Vec<_>>(),
            [512, 512, 512, 511]
        );
    }
}
