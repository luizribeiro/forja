use std::collections::VecDeque;

use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchOutput,
    EngineBatchSelect, EngineBatchSelectLane, EngineLaneCheckpoint,
};
use tokio::sync::mpsc;

use super::lane::BatchLane;
use super::{
    PreparedRequest, RequestEnvelope, batch_admission_error, read_batch_tokens, repeated_error,
};
use crate::engine::{GenerationError, ServingEngine};
use crate::scheduler::SchedulerPolicy;
use crate::{FinishReason, http::ServiceCommand, metrics::observe_phase};

struct PendingBatch {
    output: EngineBatchOutput,
    lanes: Vec<PendingLane>,
}

struct PendingLane {
    slot: u32,
    checkpoint: EngineLaneCheckpoint<()>,
    retirement: Option<Retirement>,
}

enum Retirement {
    Finish(BatchLane, FinishReason),
    Fail(BatchLane, GenerationError),
}

impl Retirement {
    fn slot(&self) -> u32 {
        match self {
            Self::Finish(lane, _) | Self::Fail(lane, _) => lane.slot,
        }
    }
}

impl PendingBatch {
    fn contains_active(&self, slot: u32) -> bool {
        self.lanes
            .iter()
            .any(|lane| lane.slot == slot && lane.retirement.is_none())
    }
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    pub(super) async fn run_batch_lanes(
        &mut self,
        receiver: &mut mpsc::Receiver<ServiceCommand>,
        pending: &mut VecDeque<RequestEnvelope>,
    ) {
        let mut active = Vec::new();
        let mut deferred = None;
        self.admit_pending(receiver, pending, &mut active, &mut deferred)
            .await;
        let mut current = None;
        loop {
            if current.is_none() {
                self.remove_unsubmitted_cancellations(&mut active);
                self.admit_pending(receiver, pending, &mut active, &mut deferred)
                    .await;
                if active.is_empty() {
                    return;
                }
                current = match self
                    .submit_batch(&active.iter().collect::<Vec<_>>(), &[])
                    .await
                {
                    Ok(batch) => batch,
                    Err(error) => {
                        for lane in active.drain(..) {
                            self.fail_batch_lane(lane, repeated_error(&error));
                        }
                        return;
                    }
                };
            }
            let Some(mut batch) = current.take() else {
                continue;
            };
            self.mark_submitted_cancellations(&mut batch, &mut active);
            drain_pending(receiver, pending);
            let admission_waiting = deferred.is_some()
                || pending
                    .front()
                    .is_some_and(|request| !request.requires_serial_path());
            let next = if admission_waiting {
                Ok(None)
            } else {
                self.submit_successor(&active, &batch).await
            };
            let mut successor = match next {
                Ok(successor) => successor,
                Err(error) => {
                    let _ = self.runner.read_queued_batch_tokens(batch.output).await;
                    self.fail_pipeline(batch.lanes, None, &mut active, error)
                        .await;
                    return;
                }
            };
            let tokens = self.runner.read_queued_batch_tokens(batch.output).await;
            let tokens = match tokens {
                Ok(bytes) => match read_batch_tokens(&bytes) {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        self.fail_pipeline(batch.lanes, successor, &mut active, error)
                            .await;
                        return;
                    }
                },
                Err(error) => {
                    self.fail_pipeline(
                        batch.lanes,
                        successor,
                        &mut active,
                        GenerationError::unavailable(format!("decode batch read failed: {error}")),
                    )
                    .await;
                    return;
                }
            };
            self.apply_batch_tokens(batch.lanes, &mut active, &mut successor, &tokens)
                .await;
            if successor.is_none() {
                self.admit_pending(receiver, pending, &mut active, &mut deferred)
                    .await;
            }
            current = successor;
        }
    }

    async fn apply_batch_tokens(
        &mut self,
        submitted: Vec<PendingLane>,
        active: &mut Vec<BatchLane>,
        successor: &mut Option<PendingBatch>,
        tokens: &[u32],
    ) {
        for pending in submitted {
            if let Some(retirement) = pending.retirement {
                self.finish_retirement(retirement, pending.checkpoint).await;
                continue;
            }
            let Some(index) = active.iter().position(|lane| lane.slot == pending.slot) else {
                continue;
            };
            let mut lane = active.remove(index);
            let Some(position) = lane.position.checked_add(1) else {
                self.retire_or_finish(
                    Retirement::Fail(lane, GenerationError::new("decode position overflowed")),
                    successor,
                )
                .await;
                continue;
            };
            lane.position = position;
            let Some(&token) = usize::try_from(lane.slot)
                .ok()
                .and_then(|slot| tokens.get(slot))
            else {
                self.retire_or_finish(
                    Retirement::Fail(lane, GenerationError::new("batch token is missing")),
                    successor,
                )
                .await;
                continue;
            };
            match lane
                .accept(token, &self.chat_info.stop_token_ids, &mut self.metrics)
                .await
            {
                Ok(Some(reason)) => {
                    self.retire_or_finish(Retirement::Finish(lane, reason), successor)
                        .await;
                }
                Ok(None) => active.push(lane),
                Err(error) => {
                    self.retire_or_finish(Retirement::Fail(lane, error), successor)
                        .await;
                }
            }
        }
    }

    async fn submit_successor(
        &mut self,
        active: &[BatchLane],
        current: &PendingBatch,
    ) -> Result<Option<PendingBatch>, GenerationError> {
        let lanes = active
            .iter()
            .filter(|lane| !current.contains_active(lane.slot) || lane.needs_successor())
            .collect::<Vec<_>>();
        let feedback = lanes
            .iter()
            .filter(|lane| current.contains_active(lane.slot))
            .map(|lane| lane.slot)
            .collect::<Vec<_>>();
        self.submit_batch(&lanes, &feedback).await
    }

    async fn submit_batch(
        &mut self,
        lanes: &[&BatchLane],
        feedback: &[u32],
    ) -> Result<Option<PendingBatch>, GenerationError> {
        if lanes.is_empty() {
            return Ok(None);
        }
        let shape = self.decode_shape(lanes)?;
        let checkpoints = self.checkpoint_lanes(lanes)?;
        let inputs = batch_inputs(shape, lanes, feedback)?;
        let output = self
            .runner
            .enqueue_batch_step(EngineBatch {
                shape,
                lanes: inputs,
            })
            .await
            .map_err(|error| GenerationError::unavailable(format!("decode batch failed: {error}")))?
            .map_err(|error| {
                GenerationError::unavailable(format!("decode batch failed: {error:?}"))
            })?;
        Ok(Some(PendingBatch {
            output,
            lanes: lanes
                .iter()
                .zip(checkpoints)
                .map(|(lane, checkpoint)| PendingLane {
                    slot: lane.slot,
                    checkpoint,
                    retirement: None,
                })
                .collect(),
        }))
    }

    fn checkpoint_lanes(
        &self,
        lanes: &[&BatchLane],
    ) -> Result<Vec<EngineLaneCheckpoint<()>>, GenerationError> {
        lanes
            .iter()
            .map(|lane| {
                let prefix = self
                    .runner
                    .paged_lane_prefix(lane.slot)
                    .map_err(|error| {
                        GenerationError::unavailable(format!(
                            "paged lane {} checkpoint failed: {error}",
                            lane.slot
                        ))
                    })?
                    .ok_or_else(|| GenerationError::new("paged lane is unavailable"))?;
                let reserved_tokens = prefix.reserved_tokens().map_err(|error| {
                    GenerationError::unavailable(format!("paged checkpoint failed: {error}"))
                })?;
                let paged = self
                    .runner
                    .checkpoint_paged_lane(lane.slot)
                    .map_err(|error| batch_admission_error(&error))?
                    .ok_or_else(|| GenerationError::new("paged lane is unavailable"))?;
                Ok(EngineLaneCheckpoint::new(
                    lane.position,
                    lane.pending_token,
                    (),
                    reserved_tokens,
                    paged,
                ))
            })
            .collect()
    }

    fn remove_unsubmitted_cancellations(&mut self, active: &mut Vec<BatchLane>) {
        let mut index = 0;
        while index < active.len() {
            if active[index].control.is_cancelled() {
                let lane = active.remove(index);
                self.fail_batch_lane(lane, GenerationError::cancelled());
            } else {
                index += 1;
            }
        }
    }

    fn mark_submitted_cancellations(
        &mut self,
        batch: &mut PendingBatch,
        active: &mut Vec<BatchLane>,
    ) {
        let mut index = 0;
        while index < active.len() {
            if !active[index].control.is_cancelled() {
                index += 1;
                continue;
            }
            let lane = active.remove(index);
            if let Some(pending) = batch
                .lanes
                .iter_mut()
                .find(|pending| pending.slot == lane.slot && pending.retirement.is_none())
            {
                pending.retirement = Some(Retirement::Fail(lane, GenerationError::cancelled()));
            } else {
                self.fail_batch_lane(lane, GenerationError::cancelled());
            }
        }
    }

    async fn retire_or_finish(
        &mut self,
        retirement: Retirement,
        successor: &mut Option<PendingBatch>,
    ) {
        if let Some(pending) = successor.as_mut().and_then(|batch| {
            batch
                .lanes
                .iter_mut()
                .find(|pending| pending.slot == retirement.slot())
        }) {
            pending.retirement = Some(retirement);
            return;
        }
        match retirement {
            Retirement::Finish(lane, reason) => self.finish_batch_lane(lane, reason).await,
            Retirement::Fail(lane, error) => self.fail_batch_lane(lane, error),
        }
    }

    async fn finish_retirement(
        &mut self,
        retirement: Retirement,
        checkpoint: EngineLaneCheckpoint<()>,
    ) {
        let rollback = self
            .runner
            .rollback_paged_lane(retirement.slot(), checkpoint.paged)
            .map_err(|error| batch_admission_error(&error));
        match (retirement, rollback) {
            (Retirement::Finish(lane, reason), Ok(())) => {
                self.finish_batch_lane(lane, reason).await;
            }
            (Retirement::Fail(lane, error), Ok(())) => self.fail_batch_lane(lane, error),
            (Retirement::Finish(lane, _) | Retirement::Fail(lane, _), Err(error)) => {
                self.fail_batch_lane(lane, error);
            }
        }
    }

    async fn fail_pipeline(
        &mut self,
        current: Vec<PendingLane>,
        successor: Option<PendingBatch>,
        active: &mut Vec<BatchLane>,
        error: GenerationError,
    ) {
        let current_slots = current.iter().map(|lane| lane.slot).collect::<Vec<_>>();
        let successor = if let Some(successor) = successor {
            let PendingBatch { output, lanes } = successor;
            let _ = self.runner.read_queued_batch_tokens(output).await;
            lanes
        } else {
            Vec::new()
        };
        let mut lanes = Vec::new();
        for (is_current, pending) in current
            .into_iter()
            .map(|pending| (true, pending))
            .chain(successor.into_iter().map(|pending| (false, pending)))
        {
            if is_current || !current_slots.contains(&pending.slot) {
                let _ = self
                    .runner
                    .rollback_paged_lane(pending.slot, pending.checkpoint.paged);
            }
            if let Some(retirement) = pending.retirement {
                lanes.push(match retirement {
                    Retirement::Finish(lane, _) | Retirement::Fail(lane, _) => lane,
                });
            }
        }
        lanes.append(active);
        for lane in lanes {
            self.fail_batch_lane(lane, repeated_error(&error));
        }
    }

    #[allow(dead_code)]
    async fn advance_constrained_lane_scheduled(
        &mut self,
        active: &mut Vec<BatchLane>,
        policy: &SchedulerPolicy,
    ) {
        let Some(index) = active.iter().position(BatchLane::is_constrained) else {
            return;
        };
        let mut lane = active.remove(index);
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.pause_once().await;
        }
        if lane.control.is_cancelled() {
            self.fail_batch_lane(lane, GenerationError::cancelled());
            return;
        }
        let checkpoint = match self.checkpoint_lanes(&[&lane]) {
            Ok(checkpoints) => {
                let Some(checkpoint) = checkpoints.into_iter().next() else {
                    self.fail_batch_lane(
                        lane,
                        GenerationError::new("constrained checkpoint is missing"),
                    );
                    return;
                };
                checkpoint
            }
            Err(error) => {
                self.fail_batch_lane(lane, error);
                return;
            }
        };
        let result = self.constrained_token(&lane, policy).await;
        let token = match result {
            Ok(token) => token,
            Err(error) => {
                let error = self
                    .runner
                    .rollback_paged_lane(lane.slot, checkpoint.paged)
                    .err()
                    .map_or(error, |failure| batch_admission_error(&failure));
                self.fail_batch_lane(lane, error);
                return;
            }
        };
        let Some(position) = lane.position.checked_add(1) else {
            self.fail_batch_lane(lane, GenerationError::new("decode position overflowed"));
            return;
        };
        lane.position = position;
        match lane
            .accept(token, &self.chat_info.stop_token_ids, &mut self.metrics)
            .await
        {
            Ok(Some(reason)) => self.finish_batch_lane(lane, reason).await,
            Ok(None) => active.push(lane),
            Err(error) => self.fail_batch_lane(lane, error),
        }
    }

    #[allow(dead_code)]
    async fn constrained_token(
        &mut self,
        lane: &BatchLane,
        policy: &SchedulerPolicy,
    ) -> Result<u32, GenerationError> {
        let forward_shape =
            self.decode_selection_shape(policy, lane.slot, BatchSelection::Forward)?;
        let count = usize::try_from(forward_shape.bucket)
            .map_err(|_| GenerationError::new("forward bucket exceeds usize"))?;
        let slot = usize::try_from(lane.slot)
            .map_err(|_| GenerationError::new("constrained lane exceeds usize"))?;
        let mut forward_lanes = vec![None; count];
        *forward_lanes
            .get_mut(slot)
            .ok_or_else(|| GenerationError::new("constrained lane exceeds its forward bucket"))? =
            Some(EngineBatchLane {
                slot: lane.slot,
                tokens: Some(vec![lane.pending_token]),
                start_pos: lane.position,
                sampling: lane.sampling,
            });
        let logits = self
            .runner
            .batch_forward(EngineBatch {
                shape: forward_shape,
                lanes: forward_lanes,
            })
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("constrained forward failed: {error}"))
            })?
            .map_err(|error| {
                GenerationError::unavailable(format!("constrained forward failed: {error:?}"))
            })?;
        let masked_shape =
            self.decode_selection_shape(policy, lane.slot, BatchSelection::Masked)?;
        let mut masked_lanes = vec![None; count];
        *masked_lanes
            .get_mut(slot)
            .ok_or_else(|| GenerationError::new("constrained lane exceeds its masked bucket"))? =
            Some(EngineBatchSelectLane {
                start_pos: lane.position,
                sampling: lane.sampling,
                allow: lane
                    .constraint_allow()
                    .ok_or_else(|| GenerationError::new("constraint mask is missing"))?
                    .to_vec(),
            });
        let output = self
            .runner
            .batch_select(
                &logits,
                EngineBatchSelect {
                    shape: masked_shape,
                    lanes: masked_lanes,
                },
            )
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("constrained selection failed: {error}"))
            })?
            .map_err(|error| {
                GenerationError::unavailable(format!("constrained selection failed: {error:?}"))
            })?;
        let bytes = self
            .runner
            .read_queued_batch_tokens(output)
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("constrained token read failed: {error}"))
            })?;
        read_batch_tokens(&bytes)?
            .get(slot)
            .copied()
            .ok_or_else(|| GenerationError::new("constrained token is missing"))
    }

    #[allow(dead_code)]
    fn decode_selection_shape(
        &self,
        policy: &SchedulerPolicy,
        slot: u32,
        selection: BatchSelection,
    ) -> Result<BatchShape, GenerationError> {
        let bucket = policy
            .decode_bucket([slot])
            .ok_or_else(|| GenerationError::new("no decode bucket can accept constrained lane"))?;
        self.info
            .serving
            .iter()
            .flat_map(|serving| &serving.batch_shapes)
            .find(|shape| {
                shape.phase == BatchPhase::Decode
                    && shape.selection == selection
                    && shape.bucket == bucket
            })
            .copied()
            .ok_or_else(|| GenerationError::new("constrained batch shape is unavailable"))
    }

    async fn admit_pending(
        &mut self,
        receiver: &mut mpsc::Receiver<ServiceCommand>,
        pending: &mut VecDeque<RequestEnvelope>,
        active: &mut Vec<BatchLane>,
        deferred: &mut Option<PreparedRequest>,
    ) {
        drain_pending(receiver, pending);
        while active.len() < usize::try_from(self.scheduler.active_lanes).unwrap_or(usize::MAX) {
            if deferred.is_none()
                && pending
                    .front()
                    .is_none_or(RequestEnvelope::requires_serial_path)
            {
                return;
            }
            let Some(slot) = (0..self.scheduler.active_lanes)
                .find(|slot| active.iter().all(|lane| lane.slot != *slot))
            else {
                return;
            };
            let prepared = if let Some(prepared) = deferred.take() {
                prepared
            } else {
                let Some(envelope) = pending.pop_front() else {
                    return;
                };
                observe_phase(
                    &mut self.metrics.queue_time,
                    &mut self.metrics.queue_time_ns,
                    envelope.queued().elapsed(),
                );
                let Some(prepared) = envelope.prepare(self).await else {
                    continue;
                };
                prepared
            };
            if !self.reservation_fits(active, &prepared) {
                *deferred = Some(prepared);
                return;
            }
            let Some(mut lane) = self.start_batch_lane(prepared, slot).await else {
                continue;
            };
            match lane
                .accept(
                    lane.pending_token,
                    &self.chat_info.stop_token_ids,
                    &mut self.metrics,
                )
                .await
            {
                Ok(Some(reason)) => self.finish_batch_lane(lane, reason).await,
                Ok(None) => active.push(lane),
                Err(error) => self.fail_batch_lane(lane, error),
            }
        }
    }

    fn reservation_fits(&self, active: &[BatchLane], request: &PreparedRequest) -> bool {
        let Some(serving) = self.info.serving.as_ref() else {
            return false;
        };
        let candidate =
            u64::from(request.prompt_tokens).checked_add(u64::from(request.params.max_tokens));
        candidate.is_some_and(|candidate| {
            reservations_fit(
                serving.page_tokens,
                serving.max_context,
                active
                    .iter()
                    .map(|lane| lane.reserved_tokens)
                    .chain(std::iter::once(candidate)),
            )
        })
    }

    fn decode_shape(&self, lanes: &[&BatchLane]) -> Result<BatchShape, GenerationError> {
        let needed = lanes
            .iter()
            .map(|lane| lane.slot.saturating_add(1))
            .max()
            .unwrap_or(1);
        self.info
            .serving
            .iter()
            .flat_map(|serving| &serving.batch_shapes)
            .filter(|shape| {
                shape.phase == BatchPhase::Decode
                    && shape.selection == BatchSelection::Combined
                    && shape.bucket >= needed
            })
            .min_by_key(|shape| shape.bucket)
            .copied()
            .ok_or_else(|| GenerationError::new("no decode batch shape can accept active lanes"))
    }

    async fn finish_batch_lane(&mut self, mut lane: BatchLane, reason: crate::FinishReason) {
        let completion = lane.completion(reason).await;
        let publish = if completion.is_ok() && self.config.serving.prefix_reuse {
            self.runner
                .paged_lane_prefix(lane.slot)
                .map_err(|error| batch_admission_error(&error))
                .and_then(|prefix| {
                    let prefix = prefix
                        .ok_or_else(|| GenerationError::new("paged prefix is unavailable"))?;
                    self.prefix_index
                        .as_mut()
                        .ok_or_else(|| GenerationError::new("prefix index is unavailable"))?
                        .publish(
                            self.prefix_namespace,
                            lane.take_ledger().into_tokens(),
                            &prefix,
                        )
                        .map_err(|error| GenerationError::unavailable(error.to_string()))
                })
        } else {
            Ok(())
        };
        let release = self
            .runner
            .prepare_paged_lane(lane.slot, None, 0)
            .map_err(|error| batch_admission_error(&error));
        let cleanup = publish.and(release.map(|_| ()));
        match (completion, cleanup) {
            (Ok(completion), Ok(())) => lane.reply(completion).await,
            (Err(error), _) | (Ok(_), Err(error)) => lane.fail(error),
        }
    }

    fn fail_batch_lane(&mut self, lane: BatchLane, error: GenerationError) {
        let _ = self.runner.prepare_paged_lane(lane.slot, None, 0);
        lane.fail(error);
    }
}

fn drain_pending(
    receiver: &mut mpsc::Receiver<ServiceCommand>,
    pending: &mut VecDeque<RequestEnvelope>,
) {
    while let Ok(command) = receiver.try_recv() {
        pending.push_back(RequestEnvelope::from_command(command));
    }
}

fn reservations_fit(
    page_tokens: u32,
    max_context: u32,
    reservations: impl IntoIterator<Item = u64>,
) -> bool {
    let page_tokens = u64::from(page_tokens);
    let capacity = u64::from(max_context).div_ceil(page_tokens);
    reservations
        .into_iter()
        .try_fold(0_u64, |total, tokens| {
            total.checked_add(tokens.div_ceil(page_tokens))
        })
        .is_some_and(|pages| pages <= capacity)
}

fn batch_inputs(
    shape: BatchShape,
    lanes: &[&BatchLane],
    feedback: &[u32],
) -> Result<Vec<Option<EngineBatchLane>>, GenerationError> {
    let count = usize::try_from(shape.bucket)
        .map_err(|_| GenerationError::new("decode bucket exceeds usize"))?;
    let mut inputs = vec![None; count];
    for lane in lanes {
        let slot = usize::try_from(lane.slot)
            .map_err(|_| GenerationError::new("decode lane exceeds usize"))?;
        let input = inputs
            .get_mut(slot)
            .ok_or_else(|| GenerationError::new("decode lane exceeds its bucket"))?;
        *input = Some(EngineBatchLane {
            slot: lane.slot,
            tokens: (!feedback.contains(&lane.slot)).then(|| vec![lane.pending_token]),
            start_pos: if feedback.contains(&lane.slot) {
                lane.position
                    .checked_add(1)
                    .ok_or_else(|| GenerationError::new("decode position overflowed"))?
            } else {
                lane.position
            },
            sampling: lane.sampling,
        });
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::reservations_fit;

    #[test]
    fn page_reservations_defer_the_first_request_that_does_not_fit() {
        assert!(reservations_fit(64, 32_768, [4_225; 7]));
        assert!(!reservations_fit(64, 32_768, [4_225; 8]));
    }
}
