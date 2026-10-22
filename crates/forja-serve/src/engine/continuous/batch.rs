use forja_core::Backend;
use forja_host::{BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane};

use super::lane::BatchLane;
use super::{batch_admission_error, read_batch_tokens, repeated_error};
use crate::engine::{GenerationError, ServingEngine};

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    pub(super) async fn run_batch_lanes(&mut self, lanes: Vec<BatchLane>) {
        let mut active = Vec::new();
        for mut lane in lanes {
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
        while !active.is_empty() {
            let shape = match self.decode_shape(&active) {
                Ok(shape) => shape,
                Err(error) => {
                    for lane in active {
                        self.fail_batch_lane(lane, repeated_error(&error));
                    }
                    return;
                }
            };
            let mut inputs = vec![None; usize::try_from(shape.bucket).unwrap_or(usize::MAX)];
            for lane in &active {
                inputs[usize::try_from(lane.slot).unwrap_or(usize::MAX)] = Some(EngineBatchLane {
                    tokens: Some(vec![lane.pending_token]),
                    start_pos: lane.position,
                    sampling: lane.sampling,
                });
            }
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
            let mut next = Vec::new();
            for mut lane in active {
                lane.position = lane.position.saturating_add(1);
                let token = tokens[usize::try_from(lane.slot).unwrap_or(usize::MAX)];
                match lane
                    .accept(token, &self.chat_info.stop_token_ids, &mut self.metrics)
                    .await
                {
                    Ok(Some(reason)) => self.finish_batch_lane(lane, reason).await,
                    Ok(None) => next.push(lane),
                    Err(error) => self.fail_batch_lane(lane, error),
                }
            }
            active = next;
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
