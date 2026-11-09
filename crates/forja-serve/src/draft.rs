use forja_host::SamplingParams;

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
