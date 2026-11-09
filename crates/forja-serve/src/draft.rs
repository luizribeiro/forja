use forja_host::SamplingParams;

mod lookup;
pub use lookup::PromptLookup;

/// Host-visible continuation and the guest policy's cap at the current context.
#[derive(Default)]
pub struct DraftProposal {
    /// Proposed tokens after the sampled, unfed pending token, without padding.
    pub tokens: Vec<u32>,
    /// Maximum drafts permitted by the engine policy at this position.
    pub rule_cap: u32,
}

/// Supplies host drafts from accepted history, including its pending token.
pub trait DraftSource: Send {
    /// Proposes a continuation at the pending token's absolute fed position.
    fn propose(&mut self, prefix: &[u32], position: u32, sampling: SamplingParams)
    -> DraftProposal;
}

impl DraftProposal {
    pub(crate) fn truncate_invalid_tokens(&mut self, vocab: u32) {
        let valid = self
            .tokens
            .iter()
            .position(|token| *token >= vocab)
            .unwrap_or(self.tokens.len());
        self.tokens.truncate(valid);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum DraftMode {
    #[default]
    Plain,
    Drafting,
    Exiting,
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_host::{BatchPhase, BatchSelection, BatchShape, speculative::VerifyBudget};

    #[test]
    fn proposals_shrink_at_invalid_tokens_and_declared_shape_gaps() {
        let shapes: Vec<_> = [2, 3]
            .map(|tokens| BatchShape {
                phase: BatchPhase::Verify,
                selection: BatchSelection::Combined,
                bucket: 1,
                tokens,
            })
            .into();
        for (tokens, expected) in [
            (vec![], None),
            (vec![33, 1], None),
            (vec![7, 33, 1], Some(2)),
            (vec![7, 1], Some(3)),
        ] {
            let mut proposal = DraftProposal {
                tokens,
                rule_cap: 2,
            };
            let budget = VerifyBudget::new(7, proposal.rule_cap, 33, 64, 64).unwrap();
            proposal.truncate_invalid_tokens(33);
            let fed =
                forja_host::speculative::select_verify_shape(&shapes, &proposal.tokens, budget);
            assert_eq!(fed.map(|fed| fed.shape().tokens), expected);
        }
    }
}
