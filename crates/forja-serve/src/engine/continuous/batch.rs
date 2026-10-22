use std::collections::VecDeque;

use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchOutput,
    EngineLaneCheckpoint,
};
use tokio::sync::mpsc;

use super::lane::BatchLane;
use super::{
    PreparedRequest, RequestEnvelope, batch_admission_error, read_batch_tokens, repeated_error,
};
use crate::engine::{GenerationError, ServingEngine};
use crate::{http::ServiceCommand, metrics::observe_phase};

struct PendingBatch {
    output: EngineBatchOutput,
    lanes: Vec<PendingLane>,
}

struct PendingLane {
    slot: u32,
    checkpoint: EngineLaneCheckpoint<()>,
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
        while !active.is_empty() {
            active = active
                .into_iter()
                .filter_map(|lane| {
                    if lane.control.is_cancelled() {
                        self.fail_batch_lane(lane, GenerationError::cancelled());
                        None
                    } else {
                        Some(lane)
                    }
                })
                .collect();
            if active.is_empty() {
                self.admit_pending(receiver, pending, &mut active, &mut deferred)
                    .await;
                if active.is_empty() {
                    return;
                }
            }
            let batch = match self.submit_batch(&active, &[]).await {
                Ok(batch) => batch,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            let tokens = match self.runner.read_queued_batch_tokens(batch.output).await {
                Ok(bytes) => read_batch_tokens(&bytes),
                Err(error) => Err(GenerationError::unavailable(format!(
                    "decode batch read failed: {error}"
                ))),
            };
            let tokens = match tokens {
                Ok(tokens) => tokens,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            active = self.apply_batch_tokens(active, batch.lanes, &tokens).await;
            self.admit_pending(receiver, pending, &mut active, &mut deferred)
                .await;
        }
    }

    async fn apply_batch_tokens(
        &mut self,
        active: Vec<BatchLane>,
        submitted: Vec<PendingLane>,
        tokens: &[u32],
    ) -> Vec<BatchLane> {
        let mut next = Vec::new();
        for (mut lane, submitted) in active.into_iter().zip(submitted) {
            if lane.control.is_cancelled() {
                let rollback = self
                    .runner
                    .rollback_paged_lane(submitted.slot, submitted.checkpoint.paged)
                    .map_err(|error| batch_admission_error(&error));
                self.fail_batch_lane(
                    lane,
                    rollback.err().unwrap_or_else(GenerationError::cancelled),
                );
                continue;
            }
            let Some(position) = lane.position.checked_add(1) else {
                self.fail_batch_lane(lane, GenerationError::new("decode position overflowed"));
                continue;
            };
            lane.position = position;
            let Some(&token) = usize::try_from(lane.slot)
                .ok()
                .and_then(|slot| tokens.get(slot))
            else {
                self.fail_batch_lane(lane, GenerationError::new("batch token is missing"));
                continue;
            };
            match lane
                .accept(token, &self.chat_info.stop_token_ids, &mut self.metrics)
                .await
            {
                Ok(Some(reason)) => self.finish_batch_lane(lane, reason).await,
                Ok(None) => next.push(lane),
                Err(error) => self.fail_batch_lane(lane, error),
            }
        }
        next
    }

    async fn submit_batch(
        &mut self,
        lanes: &[BatchLane],
        feedback: &[u32],
    ) -> Result<PendingBatch, GenerationError> {
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
        Ok(PendingBatch {
            output,
            lanes: lanes
                .iter()
                .zip(checkpoints)
                .map(|(lane, checkpoint)| PendingLane {
                    slot: lane.slot,
                    checkpoint,
                })
                .collect(),
        })
    }

    fn checkpoint_lanes(
        &self,
        lanes: &[BatchLane],
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

    async fn admit_pending(
        &mut self,
        receiver: &mut mpsc::Receiver<ServiceCommand>,
        pending: &mut VecDeque<RequestEnvelope>,
        active: &mut Vec<BatchLane>,
        deferred: &mut Option<PreparedRequest>,
    ) {
        while let Ok(command) = receiver.try_recv() {
            pending.push_back(RequestEnvelope::from_command(command));
        }
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

    fn decode_shape(&self, lanes: &[BatchLane]) -> Result<BatchShape, GenerationError> {
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
        let completion = lane.completion(reason);
        let publish = if self.config.serving.prefix_reuse {
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
        match publish.and(release.map(|_| ())) {
            Ok(()) => lane.reply(completion).await,
            Err(error) => lane.fail(error),
        }
    }

    fn fail_batch_lane(&mut self, lane: BatchLane, error: GenerationError) {
        let _ = self.runner.prepare_paged_lane(lane.slot, None, 0);
        lane.fail(error);
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
    lanes: &[BatchLane],
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
