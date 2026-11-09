//! CPU policy for exact speculative decoding.

use crate::{BatchPhase, BatchSelection, BatchShape, MAX_VERIFY_ROWS, compute, invalid_handle};
use std::num::NonZeroU64;

/// The exact immutable draft prefix selected for staging with the pending token.
#[derive(Debug)]
pub struct FedDrafts<'a> {
    shape: BatchShape,
    tokens: &'a [u32],
    budget: VerifyBudget,
}

impl<'a> FedDrafts<'a> {
    /// Declared verify shape to run with this draft prefix.
    #[must_use]
    pub const fn shape(&self) -> BatchShape {
        self.shape
    }

    /// Tokens to stage immediately after the pending input token.
    #[must_use]
    pub const fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// The same checked budget used to select this prefix.
    #[must_use]
    pub const fn budget(&self) -> VerifyBudget {
        self.budget
    }
}

/// Why a verified output prefix ends before plain decoding can continue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyCut {
    /// A stop token or a stop string completed at this token.
    Stop,
    /// The target selected an end-of-sequence token.
    Eos,
    /// Feed the pending trigger through the constrained path's masked selection.
    Activate,
    /// The request exhausted its output token allowance.
    MaxTokens,
}

/// Verified tokens awaiting their one token-boundary scan.
#[derive(Debug)]
pub struct VerifyAcceptance<'a> {
    decision: VerifyDecision<'a>,
}

/// Scanned target output, including the final sampled, unfed pending token.
#[derive(Debug)]
pub struct VerifyDecision<'a> {
    tokens: &'a [u32],
    accepted: u32,
    cut: Option<VerifyCut>,
}

impl<'a> VerifyAcceptance<'a> {
    /// Accepts exact drafts followed by one target sample, within the output budget.
    ///
    /// Consumes the exact fed draft prefix and its selection budget. The budget
    /// counts newly sampled output tokens, excluding the pending input token.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid row geometry or a target sample outside `vocab`.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "Acceptance consumes the selected draft prefix to prevent reuse."
    )]
    pub fn new(
        fed: FedDrafts<'_>,
        target_samples: &'a [u32],
        vocab: u32,
    ) -> Result<Self, compute::Error> {
        let matched = exact_match_prefix(&fed, target_samples, vocab)?;
        let left = fed.budget().max_tokens_left();
        let accepted = u64::from(matched).min(left - 1);
        let index = usize::try_from(accepted).map_err(invalid_handle)?;
        Ok(Self {
            decision: VerifyDecision {
                tokens: &target_samples[..=index],
                accepted: u32::try_from(accepted).map_err(invalid_handle)?,
                cut: (left <= u64::from(matched) + 1).then_some(VerifyCut::MaxTokens),
            },
        })
    }

    /// Scans output in order, cutting at the first boundary, including its token.
    ///
    /// The callback uses the request's detokenizer and boundary state. It never
    /// sees a rejected draft's successor or a token beyond the output budget.
    /// Stop/EOS/activation at the last permitted token precedes `MaxTokens`.
    /// Activation at draft `d_m` leaves `m-1` accepted drafts, with `d_m` pending.
    /// Activation at the bonus sample leaves all matched drafts accepted.
    ///
    /// # Errors
    ///
    /// Propagates callback errors without returning a decision to publish.
    ///
    /// ```compile_fail
    /// use forja_host::speculative::VerifyAcceptance;
    /// fn scan_twice(acceptance: VerifyAcceptance<'_>) {
    ///     let decision = acceptance.scan(|_| Ok::<_, ()>(None)).unwrap();
    ///     let _ = decision.scan(|_| Ok::<_, ()>(None));
    /// }
    /// ```
    pub fn scan<E>(
        self,
        mut boundary: impl FnMut(u32) -> Result<Option<VerifyCut>, E>,
    ) -> Result<VerifyDecision<'a>, E> {
        let mut decision = self.decision;
        for (index, row) in (0..=decision.accepted).enumerate() {
            if let Some(cut) = boundary(decision.tokens[index])? {
                decision.tokens = &decision.tokens[..=index];
                decision.accepted = row;
                decision.cut = Some(cut);
                break;
            }
        }
        Ok(decision)
    }
}

impl<'a> VerifyDecision<'a> {
    /// Tokens to account for, with stop text suppression left to the detokenizer.
    #[must_use]
    pub const fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// State destination to publish; fed length becomes `n + accepted + 1`.
    ///
    /// The sampler position becomes `n + accepted`, the last fed row's position.
    #[must_use]
    pub const fn accepted(&self) -> u32 {
        self.accepted
    }

    /// Target sample that remains unfed and must be host-staged on continuation.
    #[must_use]
    pub fn pending(&self) -> u32 {
        self.tokens[self.tokens.len() - 1]
    }

    /// Boundary reached while accepting this round, if any.
    #[must_use]
    pub const fn cut(&self) -> Option<VerifyCut> {
        self.cut
    }
}

/// Counts consecutive drafts equal to the target's own selected tokens.
///
/// Target row `i` must have sampled with the lane's seed at fed position `n+i`,
/// exactly as plain decode. Its selection is used directly at every temperature.
///
/// # Errors
///
/// Refuses unsupported row counts, missing/extra target rows, or any target
/// sample outside `vocab`, including discarded rows. Drafts are the immutable
/// prefix selected for feeding, without padding or a replacement draft list.
pub fn exact_match_prefix(
    fed: &FedDrafts<'_>,
    target_samples: &[u32],
    vocab: u32,
) -> Result<u32, compute::Error> {
    let rows = u32::try_from(target_samples.len()).map_err(invalid_handle)?;
    if !(2..=MAX_VERIFY_ROWS).contains(&rows) || fed.tokens.len() != target_samples.len() - 1 {
        return Err(invalid_handle(
            "verify samples must contain one more row than drafts",
        ));
    }
    if target_samples.iter().any(|token| *token >= vocab) {
        return Err(invalid_handle(
            "verify target sample is outside the vocabulary",
        ));
    }
    u32::try_from(
        fed.tokens
            .iter()
            .zip(target_samples)
            .take_while(|(draft, target)| draft == target)
            .count(),
    )
    .map_err(invalid_handle)
}

/// Selects a declared verify shape with the exact draft prefix it will feed.
///
/// No fitting shape, including no lookup match, means plain decode; no padding
/// is synthesized. Equal-token shapes use the last entry in declaration order.
#[must_use]
pub fn select_verify_shape<'a>(
    shapes: &[BatchShape],
    drafts: &'a [u32],
    budget: VerifyBudget,
) -> Option<FedDrafts<'a>> {
    let cap = budget.draft_cap().min(u64::try_from(drafts.len()).ok()?);
    let shape = shapes
        .iter()
        .copied()
        .filter(|shape| {
            shape.phase == BatchPhase::Verify
                && shape.selection == BatchSelection::Combined
                && shape.bucket == 1
                && (2..=MAX_VERIFY_ROWS).contains(&shape.tokens)
                && u64::from(shape.tokens - 1) <= cap
        })
        .max_by_key(|shape| shape.tokens)?;
    let count = usize::try_from(shape.tokens - 1).ok()?;
    Some(FedDrafts {
        shape,
        tokens: &drafts[..count],
        budget,
    })
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

    #[test]
    fn stop_and_eos_cut_at_each_row_before_length_without_scanning_the_tail() {
        use VerifyCut::{Eos, MaxTokens, Stop};
        let target = [7, 33, 4097];
        for (reason, row, left, end, expected) in [
            (Stop, 0, 7, 0, Stop),
            (Stop, 1, 7, 1, Stop),
            (Stop, 2, 7, 2, Stop),
            (Eos, 0, 7, 0, Eos),
            (Eos, 1, 7, 1, Eos),
            (Eos, 2, 7, 2, Eos),
            (Stop, 1, 2, 1, Stop),
            (Eos, 2, 3, 2, Eos),
            (Stop, 2, 2, 1, MaxTokens),
            (Eos, 2, 2, 1, MaxTokens),
        ] {
            let fed = fed_drafts(&target[..2], left);
            let rows = fed.tokens().len() + 1;
            let mut seen = Vec::new();
            let output = VerifyAcceptance::new(fed, &target[..rows], 4098)
                .unwrap()
                .scan(|token| {
                    seen.push(token);
                    Ok::<_, ()>((token == target[row]).then_some(reason))
                })
                .unwrap();
            assert_eq!(output.accepted(), u32::try_from(end).unwrap());
            assert_eq!(output.pending(), target[end]);
            assert_eq!(output.tokens(), &target[..=end]);
            assert_eq!(seen, output.tokens());
            assert_eq!(output.cut(), Some(expected));
        }
    }

    #[test]
    fn scan_never_observes_unaccepted_successors_and_propagates_errors() {
        let output = VerifyAcceptance::new(fed_drafts(&[7, 1], 7), &[7, 33, 4097], 4098)
            .unwrap()
            .scan(|token| {
                assert_ne!(token, 4097);
                Ok::<_, ()>(None)
            })
            .unwrap();
        assert_eq!(output.tokens(), [7, 33]);
        assert_eq!(output.cut(), None);
        let mut seen = Vec::new();
        let error = VerifyAcceptance::new(fed_drafts(&[7, 1], 7), &[7, 33, 4097], 4098)
            .unwrap()
            .scan(|token| {
                seen.push(token);
                Err::<Option<VerifyCut>, _>("decode failed")
            })
            .unwrap_err();
        assert_eq!(error, "decode failed");
        assert_eq!(seen, [7]);
    }

    #[test]
    fn accepted_output_includes_exactly_one_pending_target_sample() {
        let drafts = [7, 33];
        for (target, expected) in [([4097, 33, 1], 0_u32), ([7, 4097, 1], 1), ([7, 33, 1], 2)] {
            for left in 2..=7 {
                let fed = fed_drafts(&drafts, left);
                let rows = fed.tokens().len() + 1;
                let output = VerifyAcceptance::new(fed, &target[..rows], 4098)
                    .unwrap()
                    .scan(|_| Ok::<_, ()>(None))
                    .unwrap();
                let accepted = u64::from(expected).min(left - 1);
                let index = usize::try_from(accepted).unwrap();
                assert_eq!(output.accepted(), u32::try_from(accepted).unwrap());
                assert_eq!(output.tokens(), &target[..=index]);
                assert_eq!(output.pending(), target[index]);
                assert_eq!(
                    output.cut(),
                    (left <= u64::from(expected) + 1).then_some(VerifyCut::MaxTokens)
                );
            }
        }
    }

    #[test]
    fn acceptance_uses_the_same_large_budget_and_supports_the_maximum_extent() {
        let target = vec![7; usize::try_from(MAX_VERIFY_ROWS).unwrap()];
        let left = u64::from(u32::MAX) + 2;
        let fed = fed_drafts(&target[..target.len() - 1], left);
        assert_eq!(fed.budget().max_tokens_left(), left);
        let output = VerifyAcceptance::new(fed, &target, 33)
            .unwrap()
            .scan(|_| Ok::<_, ()>(None))
            .unwrap();
        assert_eq!(output.tokens(), target.as_slice());
        assert_eq!(output.accepted(), MAX_VERIFY_ROWS - 1);
        assert_eq!(output.pending(), 7);
        assert_eq!(output.cut(), None);
    }

    #[test]
    fn acceptance_refuses_invalid_samples_before_pending_or_detokenization() {
        for target in [[7, 4098, 1], [7, 33, 4098], [1, 33, u32::MAX]] {
            assert!(VerifyAcceptance::new(fed_drafts(&[7, 33], 33), &target, 4098).is_err());
        }
    }

    fn fed_drafts(drafts: &[u32], left: u64) -> FedDrafts<'_> {
        let budget = VerifyBudget::new(4097, MAX_VERIFY_ROWS, left, 8192, 8192).unwrap();
        let shapes = [shape(2), shape(3), shape(MAX_VERIFY_ROWS)];
        select_verify_shape(&shapes, drafts, budget).unwrap()
    }

    #[test]
    fn acceptance_ends_at_the_first_mismatch_for_every_match_pattern() {
        for k in [1, 2, 7] {
            let drafts: Vec<u32> = (1..=k).collect();
            for pattern in 0..(1_u32 << k) {
                let mut target: Vec<u32> = drafts
                    .iter()
                    .enumerate()
                    .map(|(i, draft)| {
                        if pattern & (1 << i) == 0 {
                            *draft
                        } else {
                            4097
                        }
                    })
                    .collect();
                target.push(33);
                assert_eq!(
                    exact_match_prefix(&fed_drafts(&drafts, 33), &target, 4098).unwrap(),
                    pattern.trailing_zeros().min(k)
                );
            }
        }
    }

    #[test]
    fn acceptance_refuses_missing_bonus_rows_and_unsupported_extents() {
        let fed = fed_drafts(&[7], 33);
        for target in [vec![], vec![7], vec![7, 33, 1], vec![7; 9]] {
            assert!(exact_match_prefix(&fed, &target, 4098).is_err());
        }
    }

    #[test]
    fn acceptance_uses_the_clamped_fed_prefix() {
        let available = [7, 33, 4097];
        let budget = VerifyBudget::new(4097, 1, 33, 8192, 8192).unwrap();
        let fed = select_verify_shape(&[shape(3), shape(2)], &available, budget).unwrap();
        assert_eq!(fed.shape(), shape(2));
        assert_eq!(fed.tokens(), [7]);
        assert_eq!(exact_match_prefix(&fed, &[7, 1], 4098).unwrap(), 1);
        assert!(exact_match_prefix(&fed, &[7, 33, 1], 4098).is_err());
    }

    #[test]
    fn acceptance_refuses_out_of_vocab_target_samples_in_every_row() {
        let fed = fed_drafts(&[7, 33], 33);
        for target in [
            [4098, 33, 1],
            [7, 4098, 1],
            [7, 33, 4098],
            [1, 33, u32::MAX],
        ] {
            assert!(exact_match_prefix(&fed, &target, 4098).is_err());
        }
        assert!(exact_match_prefix(&fed, &[7, 33, 1], 0).is_err());
    }

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
                select_verify_shape(&[shape(3), shape(2)], &[7; 7][..real], budget)
                    .map(|fed| fed.shape()),
                expected.map(shape)
            );
        }
        let budget = VerifyBudget::new(4097, 2, 33, 8192, 8192).unwrap();
        assert!(select_verify_shape(&[shape(3)], &[7], budget).is_none());
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
            assert!(
                select_verify_shape(
                    &[BatchShape {
                        phase,
                        bucket,
                        tokens,
                        selection
                    }],
                    &[7; 7],
                    budget
                )
                .is_none()
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
