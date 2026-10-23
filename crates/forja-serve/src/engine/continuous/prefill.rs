use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, SamplingParams,
};

use super::lane::BatchLane;
use super::{PreparedRequest, batch_admission_error, read_batch_tokens};
use crate::engine::{GenerationError, ServingEngine, sampling_seed};
use crate::metrics::observe;

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    pub(super) async fn start_batch_lane(
        &mut self,
        request: PreparedRequest,
        slot: u32,
    ) -> Option<BatchLane> {
        let sampling = match sampling_seed(request.params.seed, request.params.temperature).await {
            Ok(seed) => SamplingParams {
                temperature: request.params.temperature,
                top_k: request.params.top_k,
                top_p: request.params.top_p,
                seed,
            },
            Err(error) => {
                request.sink.fail(error);
                return None;
            }
        };
        let tokenizer = match self.tokenizer() {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                request.sink.fail(error);
                return None;
            }
        };
        match self.prefill_batch_lane(&request, slot, sampling).await {
            Ok((cached_tokens, first)) => Some(BatchLane::new(
                request,
                slot,
                cached_tokens,
                first,
                sampling,
                tokenizer,
            )),
            Err(error) => {
                let _ = self.runner.prepare_paged_lane(slot, None, 0);
                request.sink.fail(error);
                None
            }
        }
    }

    async fn prefill_batch_lane(
        &mut self,
        request: &PreparedRequest,
        slot: u32,
        sampling: SamplingParams,
    ) -> Result<(u32, u32), GenerationError> {
        let reserved = u64::from(request.prompt_tokens)
            .checked_add(u64::from(request.params.max_tokens))
            .ok_or_else(|| GenerationError::new("paged reservation overflowed"))?;
        let match_started = std::time::Instant::now();
        let hit = self
            .prefix_index
            .as_mut()
            .filter(|_| self.config.serving.prefix_reuse)
            .map(|index| index.lookup(self.prefix_namespace, &request.tokens, reserved))
            .transpose()
            .map_err(|error| GenerationError::unavailable(error.to_string()))?
            .flatten();
        let cached = hit.as_ref().map_or(0, |hit| hit.cached_tokens);
        self.metrics.requests = self.metrics.requests.saturating_add(1);
        observe(&mut self.metrics.match_time, match_started.elapsed());
        if cached == 0 {
            self.metrics.misses = self.metrics.misses.saturating_add(1);
        } else {
            self.metrics.hits = self.metrics.hits.saturating_add(1);
            self.metrics.reused_tokens = self
                .metrics
                .reused_tokens
                .saturating_add(u64::try_from(cached).unwrap_or(u64::MAX));
        }
        let saved = if slot == 0 {
            None
        } else {
            self.runner
                .paged_lane_prefix(0)
                .ok()
                .flatten()
                .map(|prefix| {
                    let reserved = prefix.reserved_tokens().map_err(|error| {
                        GenerationError::unavailable(format!("paged save failed: {error}"))
                    })?;
                    Ok((prefix, reserved))
                })
                .transpose()?
        };
        let suffix = request
            .tokens
            .get(cached..)
            .ok_or_else(|| GenerationError::new("cached prefix exceeds the prepared prompt"))?;
        self.runner
            .prepare_paged_lane(0, hit.map(|hit| hit.sequence), reserved)
            .map_err(|error| batch_admission_error(&error))?;
        let cached = u32::try_from(cached)
            .map_err(|_| GenerationError::new("cached prefix length overflowed"))?;
        let prefill_started = std::time::Instant::now();
        let output = self.run_prompt_chunks(suffix, cached, sampling).await;
        observe(
            &mut self.metrics.suffix_prefill_time,
            prefill_started.elapsed(),
        );
        if slot == 0 {
            return output.map(|token| (cached, token));
        }
        let transfer = output.and_then(|token| {
            self.runner
                .paged_lane_prefix(0)
                .map_err(|error| batch_admission_error(&error))
                .and_then(|prefix| {
                    prefix.ok_or_else(|| GenerationError::new("prefilled lane is unavailable"))
                })
                .and_then(|prefix| {
                    self.runner
                        .prepare_paged_lane(slot, Some(prefix), reserved)
                        .map_err(|error| batch_admission_error(&error))
                })
                .map(|_| (cached, token))
        });
        let restore = match saved {
            Some((prefix, reserved)) => self.runner.prepare_paged_lane(0, Some(prefix), reserved),
            None => self.runner.prepare_paged_lane(0, None, 0),
        }
        .map_err(|error| batch_admission_error(&error));
        match (transfer, restore) {
            (Ok(result), Ok(_)) => Ok(result),
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        }
    }

    async fn run_prompt_chunks(
        &mut self,
        mut tokens: &[u32],
        mut start_pos: u32,
        sampling: SamplingParams,
    ) -> Result<u32, GenerationError> {
        let shapes = self.prompt_shapes();
        let largest = usize::try_from(shapes.last().map_or(0, |shape| shape.tokens)).unwrap_or(0);
        let budget = usize::try_from(self.scheduler.prompt_token_budget).unwrap_or(usize::MAX);
        let mut selected = None;
        while !tokens.is_empty() {
            let count = budget.min(largest).min(tokens.len());
            let shape = shapes
                .iter()
                .copied()
                .find(|shape| usize::try_from(shape.tokens).is_ok_and(|size| size >= count))
                .ok_or_else(|| GenerationError::new("no prompt shape can accept the suffix"))?;
            let output = self
                .runner
                .batch_step(EngineBatch {
                    shape,
                    lanes: vec![Some(EngineBatchLane {
                        tokens: Some(tokens[..count].to_vec()),
                        start_pos,
                        sampling,
                    })],
                })
                .await
                .map_err(|error| {
                    GenerationError::unavailable(format!("prompt batch failed: {error}"))
                })?
                .map_err(|error| {
                    GenerationError::unavailable(format!("prompt batch failed: {error:?}"))
                })?;
            let bytes = self.runner.read(&output.tokens).await.map_err(|error| {
                GenerationError::unavailable(format!("prompt batch read failed: {error}"))
            })?;
            selected = read_batch_tokens(&bytes)?.first().copied();
            let consumed = count;
            let count = u32::try_from(consumed)
                .map_err(|_| GenerationError::new("prompt chunk overflowed"))?;
            start_pos = start_pos
                .checked_add(count)
                .ok_or_else(|| GenerationError::new("prompt position overflowed"))?;
            tokens = tokens
                .get(consumed..)
                .ok_or_else(|| GenerationError::new("prompt chunk exceeds its suffix"))?;
        }
        selected.ok_or_else(|| GenerationError::new("prompt suffix is empty"))
    }

    fn prompt_shapes(&self) -> Vec<BatchShape> {
        let mut shapes = self
            .info
            .serving
            .iter()
            .flat_map(|serving| &serving.batch_shapes)
            .filter(|shape| {
                shape.phase == BatchPhase::Prompt
                    && shape.bucket == 1
                    && shape.selection == BatchSelection::Combined
            })
            .copied()
            .collect::<Vec<_>>();
        shapes.sort_unstable_by_key(|shape| shape.tokens);
        shapes
    }
}
