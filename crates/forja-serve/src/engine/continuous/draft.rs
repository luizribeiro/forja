#[cfg(test)]
mod tests;
use super::lane::BatchLane;
use crate::{ServingEngine, draft::DraftMode};
use forja_core::Backend;
use forja_host::{
    BatchPhase,
    speculative::{FedDrafts, VerifyBudget, select_verify_shape},
};
pub(super) struct Candidate {
    fed: FedDrafts<'static>,
}
pub(super) fn wants_draft_drain(available: bool, exclusive: bool, active: &[BatchLane]) -> bool {
    matches!(active, [lane] if available && exclusive && !lane.draft_refused
        && lane.draft_mode != DraftMode::Exiting && !lane.is_constrained()
        && !lane.control.is_cancelled() && lane.draft_probe_position != Some(lane.position))
}
pub(super) fn reset_draft_refusals(active: &mut [BatchLane]) {
    for lane in active {
        lane.draft_refused = false;
        lane.draft_probe_position = None;
    }
}
impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
    pub(super) fn draft_candidate(&mut self, lane: &mut BatchLane) -> Option<Candidate> {
        if !wants_draft_drain(self.drafter.is_some(), true, std::slice::from_ref(lane)) {
            return None;
        }
        let shapes = &self.info.serving.as_ref()?.batch_shapes;
        if !shapes.iter().any(|shape| shape.phase == BatchPhase::Verify) {
            return None;
        }
        lane.draft_probe_position = Some(lane.position);
        let mut proposal =
            self.drafter
                .as_mut()?
                .propose(lane.ledger().tokens(), lane.position, lane.sampling);
        let budget = VerifyBudget::new(
            u64::from(lane.position),
            proposal.rule_cap,
            lane.remaining_tokens(),
            u64::from(self.max_context),
            lane.reserved_tokens,
        )?;
        proposal.truncate_invalid_tokens(self.info.vocab);
        Some(Candidate {
            fed: select_verify_shape(shapes, &proposal.tokens, budget)?.into_owned(),
        })
    }
}
