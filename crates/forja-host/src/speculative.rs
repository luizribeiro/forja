//! CPU policy for exact speculative decoding.

use crate::{BatchPhase, BatchSelection, BatchShape, MAX_VERIFY_ROWS};
use std::num::NonZeroU64;

/// Selects the largest declared verify shape fitting real drafts and the budget.
///
/// No fitting shape, including no lookup match, means plain decode; no padding
/// is synthesized. Equal-token shapes use the last entry in declaration order.
#[must_use]
pub fn select_verify_shape(
    shapes: &[BatchShape],
    real_drafts: u64,
    budget: VerifyBudget,
) -> Option<BatchShape> {
    let cap = budget.draft_cap().min(real_drafts);
    shapes
        .iter()
        .copied()
        .filter(|shape| {
            shape.phase == BatchPhase::Verify
                && shape.selection == BatchSelection::Combined
                && shape.bucket == 1
                && (2..=MAX_VERIFY_ROWS).contains(&shape.tokens)
                && u64::from(shape.tokens - 1) <= cap
        })
        .max_by_key(|shape| shape.tokens)
}

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

    fn shape(tokens: u32) -> BatchShape {
        BatchShape {
            phase: BatchPhase::Verify,
            bucket: 1,
            tokens,
            selection: BatchSelection::Combined,
        }
    }

    #[test]
    fn shape_selection_uses_only_declared_shapes_without_padding() {
        for (real, rule, left, expected) in [
            (7, 2, 33, Some(3)),
            (1, 2, 33, Some(2)),
            (0, 2, 33, None),
            (7, 1, 33, Some(2)),
            (7, 0, 33, None),
            (7, 2, 1, None),
        ] {
            let budget = VerifyBudget::new(4097, rule, left, 8192, 8192).unwrap();
            assert_eq!(
                select_verify_shape(&[shape(3), shape(2)], real, budget),
                expected.map(shape)
            );
        }
        let budget = VerifyBudget::new(4097, 2, 33, 8192, 8192).unwrap();
        assert_eq!(select_verify_shape(&[shape(3)], 1, budget), None);
    }

    #[test]
    fn shape_selection_refuses_unsupported_catalog_entries() {
        for (phase, selection, bucket, tokens) in [
            (BatchPhase::Verify, BatchSelection::Combined, 1, 1),
            (BatchPhase::Verify, BatchSelection::Combined, 1, 9),
            (BatchPhase::Verify, BatchSelection::Combined, 2, 3),
            (BatchPhase::Decode, BatchSelection::Combined, 1, 3),
            (BatchPhase::Verify, BatchSelection::Masked, 1, 3),
        ] {
            let budget = VerifyBudget::new(4097, 2, 33, 8192, 8192).unwrap();
            assert_eq!(
                select_verify_shape(
                    &[BatchShape {
                        phase,
                        bucket,
                        tokens,
                        selection
                    }],
                    7,
                    budget
                ),
                None
            );
        }
    }

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
