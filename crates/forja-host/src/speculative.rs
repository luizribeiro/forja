//! CPU policy for exact speculative decoding.

use std::num::NonZeroU64;

/// Checked limits shared by shape selection and acceptance for one round.
#[derive(Clone, Copy, Debug)]
pub struct VerifyBudget {
    max_tokens_left: NonZeroU64,
    draft_cap: u64,
}

impl VerifyBudget {
    /// Constructs a round budget at the pending token's fed position.
    ///
    /// `max_tokens_left` counts newly sampled output tokens, excluding the pending
    /// input token. `max_context` and `reserved_tokens` are exclusive absolute
    /// ends. The profile's `rule_cap` limits drafts, leaving one target sample.
    /// Returns `None` if no output token or no pending input row fits.
    #[must_use]
    pub fn new(
        start_pos: u64,
        rule_cap: u32,
        max_tokens_left: u64,
        max_context: u64,
        reserved_tokens: u64,
    ) -> Option<Self> {
        let max_tokens_left = NonZeroU64::new(max_tokens_left)?;
        let headroom = max_context
            .min(reserved_tokens)
            .checked_sub(start_pos)?
            .checked_sub(1)?;
        let draft_cap = u64::from(rule_cap)
            .min(max_tokens_left.get().checked_sub(1)?)
            .min(headroom);
        Some(Self {
            max_tokens_left,
            draft_cap,
        })
    }

    /// Remaining newly sampled output tokens, including the final target sample.
    #[must_use]
    pub const fn max_tokens_left(self) -> u64 {
        self.max_tokens_left.get()
    }

    /// Maximum draft count after output, context, reservation and profile clamps.
    #[must_use]
    pub const fn draft_cap(self) -> u64 {
        self.draft_cap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_clamp_each_limit_with_checked_absolute_headroom() {
        for (start, rule, left, context, reserved, expected) in [
            (4097, 2, 33, 8192, 8192, Some(2)),
            (4097, 1, 33, 8192, 8192, Some(1)),
            (4097, 2, 2, 8192, 8192, Some(1)),
            (4097, 2, 33, 4099, 8192, Some(1)),
            (4097, 2, 33, 8192, 4099, Some(1)),
            (4097, 0, 33, 8192, 8192, Some(0)),
            (4097, 2, 1, 8192, 8192, Some(0)),
            (4097, 2, 33, 4098, 8192, Some(0)),
            (4097, 2, 0, 8192, 8192, None),
            (4097, 2, 33, 8192, 4096, None),
            (u64::MAX, 2, 33, u64::MAX, u64::MAX, None),
        ] {
            let budget = VerifyBudget::new(start, rule, left, context, reserved);
            assert_eq!(budget.map(VerifyBudget::draft_cap), expected);
            if let Some(budget) = budget {
                assert_eq!(budget.max_tokens_left(), left);
            }
        }
    }

    #[test]
    fn output_budgets_preserve_values_above_u32() {
        let left = u64::from(u32::MAX) + 33;
        let budget = VerifyBudget::new(4097, 2, left, 8192, 8192).unwrap();
        assert_eq!(budget.max_tokens_left(), left);
        assert_eq!(budget.draft_cap(), 2);
    }
}
