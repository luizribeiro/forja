use std::collections::{BTreeMap, VecDeque};

use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchOutput,
    EngineBatchSelect, EngineBatchSelectLane, EngineLaneCheckpoint,
};
use tokio::sync::mpsc;

use super::lane::BatchLane;
use super::prefill::PendingPrefill;
use super::{
    PreparedRequest, RequestEnvelope, batch_admission_error, read_batch_tokens, repeated_error,
};
use crate::engine::{GenerationError, ServingEngine};
use crate::scheduler::{Admission, RequestId, SchedulerPolicy};
use crate::{FinishReason, http::ServiceCommand, metrics::observe_phase};

struct PendingBatch {
    output: EngineBatchOutput,
    lanes: Vec<PendingLane>,
    deferred_retirements: Vec<Retirement>,
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

    fn has_retirement(&self) -> bool {
        !self.deferred_retirements.is_empty()
            || self.lanes.iter().any(|lane| lane.retirement.is_some())
    }
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    #[allow(
        clippy::too_many_lines,
        reason = "the loop keeps successor ownership and admission ordering visible in one place"
    )]
    pub(super) async fn run_batch_lanes(
        &mut self,
        receiver: &mut mpsc::Receiver<ServiceCommand>,
        pending: &mut VecDeque<RequestEnvelope>,
    ) {
        let Ok(policy) = SchedulerPolicy::new(
            self.scheduler.decode_buckets.clone(),
            usize::try_from(self.scheduler.active_lanes).unwrap_or(usize::MAX),
            usize::try_from(self.scheduler.max_prefill_chunks).unwrap_or(usize::MAX),
            self.scheduler.prompt_token_budget,
            self.scheduler.idle_prompt_token_budget,
        ) else {
            return;
        };
        let mut policy = policy.with_prompt_chunk_multiple(self.scheduler.prefill_chunk_multiple);
        let mut active = Vec::new();
        let mut waiting = BTreeMap::new();
        let mut prefilling = BTreeMap::new();
        let mut next_id = 0_u64;
        let mut current = None;
        loop {
            if current.is_none() {
                self.remove_unsubmitted_cancellations(&mut active);
                self.admit_pending(
                    receiver,
                    pending,
                    &active,
                    &mut prefilling,
                    &mut waiting,
                    &mut policy,
                    &mut next_id,
                )
                .await;
                if !active.is_empty()
                    && active.iter().all(BatchLane::is_constrained)
                    && !prefilling.is_empty()
                {
                    self.advance_prefills(&mut prefilling, &mut active, &mut current, &mut policy)
                        .await;
                }
                if current.is_some() {
                    continue;
                }
                if active.is_empty() {
                    self.advance_prefills(&mut prefilling, &mut active, &mut current, &mut policy)
                        .await;
                    if current.is_some() {
                        continue;
                    }
                    if active.is_empty() {
                        if prefilling.is_empty() && waiting.is_empty() {
                            return;
                        }
                        continue;
                    }
                }
                let ordinary = active
                    .iter()
                    .filter(|lane| !lane.is_constrained())
                    .collect::<Vec<_>>();
                if ordinary.is_empty() {
                    self.advance_constrained_lane(&mut active, &policy).await;
                    continue;
                }
                let submitted_slots = ordinary.iter().map(|lane| lane.slot).collect::<Vec<_>>();
                current = match self.submit_batch(&policy, &ordinary, &[]).await {
                    Ok(batch) => batch,
                    Err(error) => {
                        let mut index = 0;
                        while index < active.len() {
                            if submitted_slots.contains(&active[index].slot) {
                                let lane = active.remove(index);
                                self.fail_batch_lane(lane, repeated_error(&error));
                            } else {
                                index += 1;
                            }
                        }
                        continue;
                    }
                };
            }
            let Some(mut batch) = current.take() else {
                continue;
            };
            #[cfg(test)]
            if active.iter().any(BatchLane::is_constrained)
                && let Some(barrier) = &self.batch_test_barrier
            {
                barrier.cancel_lane();
                barrier.pause_once().await;
            }
            Self::mark_submitted_cancellations(&mut batch, &mut active);
            drain_pending(receiver, pending);
            let admission_waiting = active.iter().any(BatchLane::is_constrained)
                || !waiting.is_empty()
                || !prefilling.is_empty()
                || pending
                    .front()
                    .is_some_and(|request| !request.requires_serial_path());
            let next = if admission_waiting {
                Ok(None)
            } else {
                self.submit_successor(&policy, &active, &batch).await
            };
            let mut successor = match next {
                Ok(successor) => successor,
                Err(error) => {
                    let _ = self.read_completed_batch(batch.output).await;
                    self.fail_pipeline(
                        batch.lanes,
                        batch.deferred_retirements,
                        None,
                        &mut active,
                        error,
                    )
                    .await;
                    if active.is_empty() {
                        return;
                    }
                    continue;
                }
            };
            #[cfg(test)]
            if batch.lanes.len() >= 2
                && successor.is_some()
                && let Some(barrier) = &self.batch_test_barrier
            {
                barrier.pause_once().await;
            }
            let tokens = self.read_completed_batch(batch.output).await;
            let tokens = match tokens {
                Ok(bytes) => match read_batch_tokens(&bytes) {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        self.fail_pipeline(
                            batch.lanes,
                            batch.deferred_retirements,
                            successor,
                            &mut active,
                            error,
                        )
                        .await;
                        if active.is_empty() {
                            return;
                        }
                        continue;
                    }
                },
                Err(error) => {
                    self.fail_pipeline(
                        batch.lanes,
                        batch.deferred_retirements,
                        successor,
                        &mut active,
                        GenerationError::unavailable(format!("decode batch read failed: {error}")),
                    )
                    .await;
                    if active.is_empty() {
                        return;
                    }
                    continue;
                }
            };
            #[cfg(test)]
            if let Some(barrier) = &self.batch_test_barrier {
                barrier
                    .decoded_batches
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
            self.apply_batch_tokens(
                batch.lanes,
                batch.deferred_retirements,
                &mut active,
                &mut successor,
                &tokens,
            )
            .await;
            if successor.is_none() {
                #[cfg(test)]
                if successor.is_some()
                    && let Some(barrier) = &self.batch_test_barrier
                {
                    barrier.note_admission_overlap();
                }
                self.advance_constrained_lane(&mut active, &policy).await;
                self.admit_pending(
                    receiver,
                    pending,
                    &active,
                    &mut prefilling,
                    &mut waiting,
                    &mut policy,
                    &mut next_id,
                )
                .await;
                self.advance_prefills(&mut prefilling, &mut active, &mut successor, &mut policy)
                    .await;
            }
            current = successor;
        }
    }

    async fn apply_batch_tokens(
        &mut self,
        submitted: Vec<PendingLane>,
        deferred_retirements: Vec<Retirement>,
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
        for retirement in deferred_retirements {
            self.retire_or_finish(retirement, successor).await;
        }
    }

    async fn submit_successor(
        &mut self,
        policy: &SchedulerPolicy,
        active: &[BatchLane],
        current: &PendingBatch,
    ) -> Result<Option<PendingBatch>, GenerationError> {
        let reaches_retirement = current.has_retirement()
            || active
                .iter()
                .any(|lane| current.contains_active(lane.slot) && !lane.needs_successor());
        if reaches_retirement {
            #[cfg(test)]
            if let Some(barrier) = &self.batch_test_barrier {
                barrier.note_retirement_drain();
            }
            return Ok(None);
        }
        let lanes = active
            .iter()
            .filter(|lane| {
                !lane.is_constrained()
                    && (!current.contains_active(lane.slot) || lane.needs_successor())
            })
            .collect::<Vec<_>>();
        let feedback = lanes
            .iter()
            .filter(|lane| current.contains_active(lane.slot))
            .map(|lane| lane.slot)
            .collect::<Vec<_>>();
        self.submit_batch(policy, &lanes, &feedback).await
    }

    async fn submit_batch(
        &mut self,
        policy: &SchedulerPolicy,
        lanes: &[&BatchLane],
        feedback: &[u32],
    ) -> Result<Option<PendingBatch>, GenerationError> {
        if lanes.is_empty() {
            return Ok(None);
        }
        let shape = self.decode_shape(policy, lanes)?;
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
            deferred_retirements: Vec::new(),
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
                let mut checkpoint = EngineLaneCheckpoint::new(
                    lane.position,
                    lane.pending_token,
                    (),
                    reserved_tokens,
                    paged,
                );
                checkpoint.state_version = self
                    .runner
                    .checkpoint_state_lane(lane.slot)
                    .map_err(|error| batch_admission_error(&error))?;
                Ok(checkpoint)
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

    fn mark_submitted_cancellations(batch: &mut PendingBatch, active: &mut Vec<BatchLane>) {
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
                batch
                    .deferred_retirements
                    .push(Retirement::Fail(lane, GenerationError::cancelled()));
            }
        }
    }

    async fn retire_or_finish(
        &mut self,
        retirement: Retirement,
        successor: &mut Option<PendingBatch>,
    ) {
        if let Some(batch) = successor.as_mut() {
            if let Some(pending) = batch
                .lanes
                .iter_mut()
                .find(|pending| pending.slot == retirement.slot())
            {
                pending.retirement = Some(retirement);
            } else {
                #[cfg(test)]
                if let Some(barrier) = &self.batch_test_barrier {
                    barrier.note_deferred_fallthrough();
                }
                batch.deferred_retirements.push(retirement);
            }
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
            .rollback_lane(retirement.slot(), checkpoint)
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
        current_retirements: Vec<Retirement>,
        successor: Option<PendingBatch>,
        active: &mut Vec<BatchLane>,
        error: GenerationError,
    ) {
        let current_slots = current.iter().map(|lane| lane.slot).collect::<Vec<_>>();
        let (successor, successor_retirements) = if let Some(successor) = successor {
            let PendingBatch {
                output,
                lanes,
                deferred_retirements,
            } = successor;
            let _ = self.read_completed_batch(output).await;
            (lanes, deferred_retirements)
        } else {
            (Vec::new(), Vec::new())
        };
        let affected = current_slots
            .iter()
            .copied()
            .chain(successor.iter().map(|lane| lane.slot))
            .collect::<Vec<_>>();
        let mut lanes = current_retirements
            .into_iter()
            .chain(successor_retirements)
            .map(|retirement| match retirement {
                Retirement::Finish(lane, _) | Retirement::Fail(lane, _) => lane,
            })
            .collect::<Vec<_>>();
        for (is_current, pending) in current
            .into_iter()
            .map(|pending| (true, pending))
            .chain(successor.into_iter().map(|pending| (false, pending)))
        {
            if is_current || !current_slots.contains(&pending.slot) {
                let _ = self.runner.rollback_lane(pending.slot, pending.checkpoint);
            }
            if let Some(retirement) = pending.retirement {
                lanes.push(match retirement {
                    Retirement::Finish(lane, _) | Retirement::Fail(lane, _) => lane,
                });
            }
        }
        let mut index = 0;
        while index < active.len() {
            if affected.contains(&active[index].slot) {
                lanes.push(active.remove(index));
            } else {
                index += 1;
            }
        }
        for lane in lanes {
            self.fail_batch_lane(lane, repeated_error(&error));
        }
    }

    async fn read_completed_batch(
        &mut self,
        output: EngineBatchOutput,
    ) -> Result<Vec<u8>, forja_host::bindings::l9o::gpu::compute::Error> {
        let result = self.runner.read_queued_batch_tokens(output).await;
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.observe_batch_completion();
        }
        result
    }

    async fn advance_constrained_lane(
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
                    .rollback_lane(lane.slot, checkpoint)
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

    #[allow(
        clippy::too_many_arguments,
        reason = "admission reconciles the mailbox, scheduler, and both lane states atomically"
    )]
    async fn admit_pending(
        &mut self,
        receiver: &mut mpsc::Receiver<ServiceCommand>,
        pending: &mut VecDeque<RequestEnvelope>,
        active: &[BatchLane],
        prefilling: &mut BTreeMap<RequestId, PendingPrefill>,
        waiting: &mut BTreeMap<RequestId, PreparedRequest>,
        policy: &mut SchedulerPolicy,
        next_id: &mut u64,
    ) {
        drain_pending(receiver, pending);
        while pending
            .front()
            .is_some_and(|request| !request.requires_serial_path())
        {
            let Some(envelope) = pending.pop_front() else {
                break;
            };
            observe_phase(
                &mut self.metrics.queue_time,
                &mut self.metrics.queue_time_ns,
                envelope.queued().elapsed(),
            );
            let Some(prepared) = envelope.prepare(self).await else {
                continue;
            };
            let Some(reserved_tokens) = prepared.reserved_tokens() else {
                prepared
                    .sink
                    .fail(GenerationError::new("paged reservation overflowed"));
                continue;
            };
            let id = RequestId(*next_id);
            let Some(next) = next_id.checked_add(1) else {
                prepared
                    .sink
                    .fail(GenerationError::new("scheduler request id overflowed"));
                continue;
            };
            *next_id = next;
            policy.enqueue(Admission {
                id,
                reserved_tokens,
            });
            waiting.insert(id, prepared);
        }
        policy.reconcile_active(active.len().saturating_add(prefilling.len()));
        let mut reservations = active
            .iter()
            .map(|lane| lane.reserved_tokens)
            .chain(prefilling.values().map(PendingPrefill::reserved_tokens))
            .collect::<Vec<_>>();
        let admitted = policy.admit(|admission| {
            let fits = self.reservations_fit(
                reservations
                    .iter()
                    .copied()
                    .chain(std::iter::once(admission.reserved_tokens)),
            );
            if fits {
                reservations.push(admission.reserved_tokens);
            }
            fits
        });
        for id in admitted {
            let Some(prepared) = waiting.remove(&id) else {
                policy.complete(id);
                continue;
            };
            let Some(slot) = (0..self.scheduler.active_lanes).find(|slot| {
                active.iter().all(|lane| lane.slot != *slot)
                    && prefilling.values().all(|prefill| prefill.slot() != *slot)
            }) else {
                prepared.sink.fail(GenerationError::new(
                    "scheduler admitted without a free lane",
                ));
                policy.complete(id);
                continue;
            };
            match self.begin_batch_prefill(prepared, slot).await {
                Ok(prefill) => {
                    #[cfg(test)]
                    self.trace_request(id, false);
                    policy.ready_prompt(id, prefill.remaining());
                    prefilling.insert(id, prefill);
                }
                Err((request, error)) => {
                    if error.paged_bank_in_flight()
                        && let Some(reserved_tokens) = request.reserved_tokens()
                    {
                        policy.retry_admission(Admission {
                            id,
                            reserved_tokens,
                        });
                        waiting.insert(id, request);
                    } else {
                        request.sink.fail(error);
                        policy.complete(id);
                    }
                }
            }
        }
    }

    async fn advance_prefills(
        &mut self,
        prefilling: &mut BTreeMap<RequestId, PendingPrefill>,
        active: &mut Vec<BatchLane>,
        queued: &mut Option<PendingBatch>,
        policy: &mut SchedulerPolicy,
    ) {
        let mut completed = Vec::new();
        let decoding = !active.is_empty();
        #[cfg(test)]
        self.trace_prompt_round(decoding);
        for (id, budget) in policy.prompt_batch(decoding) {
            #[cfg(test)]
            self.trace_prompt_work(budget, decoding);
            let Some(mut prefill) = prefilling.remove(&id) else {
                continue;
            };
            match self.advance_batch_prefill(&mut prefill, budget).await {
                Ok(true) => match Self::finish_batch_prefill(prefill) {
                    Ok(lane) => completed.push((id, lane)),
                    Err((prefill, error)) => {
                        policy.complete(id);
                        self.fail_batch_prefill(prefill, error);
                    }
                },
                Ok(false) => {
                    prefilling.insert(id, prefill);
                }
                Err(error) => {
                    policy.complete(id);
                    self.fail_batch_prefill(prefill, error);
                }
            }
        }
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.pause_prefill_once().await;
        }
        let (submitted_slots, submission) = {
            let ordinary = active
                .iter()
                .chain(completed.iter().map(|(_, lane)| lane))
                .filter(|lane| !lane.is_constrained())
                .collect::<Vec<_>>();
            let slots = ordinary.iter().map(|lane| lane.slot).collect::<Vec<_>>();
            let submission = if queued.is_none() {
                self.submit_batch(policy, &ordinary, &[]).await
            } else {
                Ok(None)
            };
            (slots, submission)
        };
        match submission {
            Ok(batch) => {
                if batch.is_some() {
                    *queued = batch;
                }
                #[cfg(test)]
                if queued.is_some()
                    && !completed.is_empty()
                    && let Some(barrier) = &self.batch_test_barrier
                {
                    barrier.note_prefill_overlap(submitted_slots.len());
                }
            }
            Err(error) => {
                let mut index = 0;
                while index < active.len() {
                    if submitted_slots.contains(&active[index].slot) {
                        let lane = active.remove(index);
                        self.fail_batch_lane(lane, repeated_error(&error));
                    } else {
                        index += 1;
                    }
                }
                for (id, lane) in completed {
                    policy.complete(id);
                    self.fail_batch_lane(lane, repeated_error(&error));
                }
                return;
            }
        }
        for (id, mut lane) in completed {
            #[cfg(test)]
            self.trace_request(id, true);
            let _ = id;
            match lane
                .accept(
                    lane.pending_token,
                    &self.chat_info.stop_token_ids,
                    &mut self.metrics,
                )
                .await
            {
                Ok(Some(reason)) => {
                    self.retire_or_finish(Retirement::Finish(lane, reason), queued)
                        .await;
                }
                Ok(None) => active.push(lane),
                Err(error) => {
                    self.retire_or_finish(Retirement::Fail(lane, error), queued)
                        .await;
                }
            }
        }
    }

    fn reservations_fit(&self, reservations: impl IntoIterator<Item = u64>) -> bool {
        let Some(serving) = self.info.serving.as_ref() else {
            return false;
        };
        reservations_fit(serving.page_tokens, serving.max_context, reservations)
    }

    fn decode_shape(
        &self,
        policy: &SchedulerPolicy,
        lanes: &[&BatchLane],
    ) -> Result<BatchShape, GenerationError> {
        let bucket = policy
            .decode_bucket(lanes.iter().map(|lane| lane.slot))
            .ok_or_else(|| GenerationError::new("no decode bucket can accept active lanes"))?;
        self.info
            .serving
            .iter()
            .flat_map(|serving| &serving.batch_shapes)
            .filter(|shape| {
                shape.phase == BatchPhase::Decode
                    && shape.selection == BatchSelection::Combined
                    && shape.bucket == bucket
            })
            .min_by_key(|shape| shape.bucket)
            .copied()
            .ok_or_else(|| GenerationError::new("no decode batch shape can accept active lanes"))
    }

    async fn finish_batch_lane(&mut self, mut lane: BatchLane, reason: crate::FinishReason) {
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.note_finish(lane.is_constrained());
        }
        let completion = lane.completion(reason).await;
        let publish = if completion.is_ok() && lane.prefix_reuse() {
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
        let error = self.release_failed_lane(lane.slot).unwrap_or(error);
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
