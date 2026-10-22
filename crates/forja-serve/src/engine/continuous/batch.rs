use std::collections::VecDeque;

use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineLaneCheckpoint,
};
use tokio::sync::mpsc;

use super::lane::BatchLane;
use super::{RequestEnvelope, batch_admission_error, read_batch_tokens, repeated_error};
use crate::engine::{GenerationError, ServingEngine};
use crate::{http::ServiceCommand, metrics::observe_phase};

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
        self.admit_pending(receiver, pending, &mut active).await;
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
                return;
            }
            let shape = match self.decode_shape(&active) {
                Ok(shape) => shape,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            let checkpoints = match self.checkpoint_lanes(&active) {
                Ok(checkpoints) => checkpoints,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            let inputs = match batch_inputs(shape, &active) {
                Ok(inputs) => inputs,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            let output = self
                .runner
                .batch_step(EngineBatch {
                    shape,
                    lanes: inputs,
                })
                .await;
            let tokens = match output {
                Ok(Ok(output)) => match self.runner.read(&output.tokens).await {
                    Ok(bytes) => read_batch_tokens(&bytes),
                    Err(error) => Err(GenerationError::unavailable(format!(
                        "decode batch read failed: {error}"
                    ))),
                },
                Ok(Err(error)) => Err(GenerationError::unavailable(format!(
                    "decode batch failed: {error:?}"
                ))),
                Err(error) => Err(GenerationError::unavailable(format!(
                    "decode batch failed: {error}"
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
            active = self.apply_batch_tokens(active, checkpoints, &tokens).await;
            self.admit_pending(receiver, pending, &mut active).await;
        }
    }

    async fn apply_batch_tokens(
        &mut self,
        active: Vec<BatchLane>,
        checkpoints: Vec<EngineLaneCheckpoint<()>>,
        tokens: &[u32],
    ) -> Vec<BatchLane> {
        let mut next = Vec::new();
        for (mut lane, checkpoint) in active.into_iter().zip(checkpoints) {
            if lane.control.is_cancelled() {
                let rollback = self
                    .runner
                    .rollback_paged_lane(lane.slot, checkpoint.paged)
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
    ) {
        while let Ok(command) = receiver.try_recv() {
            pending.push_back(RequestEnvelope::from_command(command));
        }
        while active.len() < usize::try_from(self.scheduler.active_lanes).unwrap_or(usize::MAX) {
            if pending
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

fn batch_inputs(
    shape: BatchShape,
    lanes: &[BatchLane],
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
            tokens: Some(vec![lane.pending_token]),
            start_pos: lane.position,
            sampling: lane.sampling,
        });
    }
    Ok(inputs)
}
