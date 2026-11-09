#[cfg(test)]
mod tests;

use forja_core::Backend;
use forja_host::{
    BatchPhase, EngineBatch, EngineBatchLane, EngineBatchOutput,
    speculative::{FedDrafts, VerifyAcceptance, VerifyBudget, VerifyCut, select_verify_shape},
};

use super::{lane::BatchLane, read_batch_tokens};
use crate::{FinishReason, GenerationError, ServingEngine, draft::DraftMode};

pub(super) struct Candidate {
    fed: FedDrafts<'static>,
}

pub(super) fn wants_draft_drain(available: bool, exclusive: bool, active: &[BatchLane]) -> bool {
    matches!(active, [lane] if available && exclusive && !lane.draft_refused
        && lane.draft_mode != DraftMode::Exiting && lane.can_draft()
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

    pub(super) async fn advance_drafting(
        &mut self,
        active: &mut Vec<BatchLane>,
        exclusive: bool,
    ) -> bool {
        if active.len() != 1 {
            return false;
        }
        let wanted = wants_draft_drain(self.drafter.is_some(), exclusive, active);
        if !wanted && active[0].draft_mode == DraftMode::Plain {
            return false;
        }
        let candidate = wanted
            .then(|| self.draft_candidate(&mut active[0]))
            .flatten();
        let mut lane = active.remove(0);
        if let Some(candidate) = candidate {
            let reserved = if lane.draft_mode == DraftMode::Drafting {
                Ok(true)
            } else {
                self.runner.try_reserve_draft_lane(lane.slot)
            };
            match reserved {
                Ok(true) => {
                    lane.draft_mode = DraftMode::Drafting;
                    match self.run_draft_round(&mut lane, candidate).await {
                        Ok(Some(reason)) => self.finish_batch_lane(lane, reason).await,
                        Ok(None) => active.push(lane),
                        Err(error) => self.fail_batch_lane(lane, error),
                    }
                    return true;
                }
                Ok(false) => lane.draft_refused = true,
                Err(error) => {
                    self.fail_batch_lane(lane, draft_error(error));
                    return true;
                }
            }
        }
        if lane.draft_mode != DraftMode::Plain {
            lane.draft_mode = DraftMode::Exiting;
            match self.runner.try_release_draft_lane(lane.slot) {
                Ok(true) => {
                    lane.draft_mode = DraftMode::Plain;
                    reset_draft_refusals(std::slice::from_mut(&mut lane));
                }
                Ok(false) => {}
                Err(error) => {
                    self.fail_batch_lane(lane, draft_error(error));
                    return true;
                }
            }
        }
        active.push(lane);
        false
    }

    async fn run_draft_round(
        &mut self,
        lane: &mut BatchLane,
        candidate: Candidate,
    ) -> Result<Option<FinishReason>, GenerationError> {
        let checkpoint = self
            .checkpoint_lanes(&[lane])?
            .pop()
            .ok_or_else(|| GenerationError::new("draft checkpoint is missing"))?;
        let result = self.verify_tokens(lane, candidate).await;
        let (accepted, finish) = match result {
            Ok(output) => output,
            Err(error) => {
                return Err(self
                    .runner
                    .rollback_lane(lane.slot, checkpoint)
                    .err()
                    .map_or(error, draft_error));
            }
        };
        lane.position = lane
            .position
            .checked_add(accepted)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| GenerationError::new("verify position overflowed"))?;
        Ok(finish)
    }

    async fn verify_tokens(
        &mut self,
        lane: &mut BatchLane,
        Candidate { fed }: Candidate,
    ) -> Result<(u32, Option<FinishReason>), GenerationError> {
        let shape = fed.shape();
        let mut tokens = vec![lane.pending_token];
        tokens.extend_from_slice(fed.tokens());
        let output = self
            .runner
            .enqueue_batch_step(EngineBatch {
                shape,
                lanes: vec![Some(EngineBatchLane {
                    slot: lane.slot,
                    tokens: Some(tokens),
                    start_pos: lane.position,
                    sampling: lane.sampling,
                })],
            })
            .await
            .map_err(draft_error)?
            .map_err(draft_error)?;
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.pause_once().await;
        }
        let samples = match self
            .runner
            .read(&output.tokens)
            .await
            .map_err(draft_error)
            .and_then(|bytes| read_batch_tokens(&bytes))
        {
            Ok(samples) => samples,
            Err(error) => {
                return Err(self.abandon_output(output, error).await);
            }
        };
        let decision = match VerifyAcceptance::new(fed, &samples, self.info.vocab) {
            Ok(acceptance) => {
                lane.accept_verified(
                    acceptance,
                    &self.chat_info.stop_token_ids,
                    &mut self.metrics,
                )
                .await
            }
            Err(error) => Err(draft_error(error)),
        };
        let decision = match decision {
            Ok(decision) if !lane.control.is_cancelled() => decision,
            result => {
                return Err(self
                    .abandon_output(
                        output,
                        result.err().unwrap_or_else(GenerationError::cancelled),
                    )
                    .await);
            }
        };
        let accepted = decision.accepted();
        let finish = match decision.cut() {
            Some(VerifyCut::Stop | VerifyCut::Eos) => Some(FinishReason::Stop),
            Some(VerifyCut::MaxTokens) => Some(FinishReason::Length),
            Some(VerifyCut::Activate) => {
                return Err(self
                    .abandon_output(
                        output,
                        GenerationError::new("verify round cannot activate a tool constraint"),
                    )
                    .await);
            }
            None => None,
        };
        self.runner
            .publish_verified_batch(output, &decision)
            .map_err(draft_error)?;
        lane.pending_token = decision.pending();
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier
                .verified
                .lock()
                .unwrap()
                .push((shape.tokens, accepted));
        }
        Ok((accepted, finish))
    }
    async fn abandon_output(
        &mut self,
        output: EngineBatchOutput,
        error: GenerationError,
    ) -> GenerationError {
        let _ = self.runner.read_queued_batch_tokens(output).await;
        error
    }
}

fn draft_error(error: impl std::fmt::Display) -> GenerationError {
    GenerationError::unavailable(format!("verify round failed: {error}"))
}
