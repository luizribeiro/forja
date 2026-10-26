use forja_core::Backend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchOutput,
    EngineBatchSelect, EngineBatchSelectLane, SamplingParams,
};
use tokenizers::Tokenizer;

use super::lane::{BatchConstraint, BatchLane};
use super::{PreparedRequest, batch_admission_error, read_batch_tokens};
use crate::engine::{
    GenerationError, ServingEngine, compile_constraint, next_mask, sampling_seed, start_tool_parser,
};
use crate::metrics::{observe, observe_phase};

pub(super) struct PendingPrefill {
    request: PreparedRequest,
    slot: u32,
    sampling: SamplingParams,
    tokenizer: Tokenizer,
    cached: u32,
    reserved: u64,
    consumed: usize,
    position: u32,
    selected: Option<u32>,
    constraint: Option<BatchConstraint>,
}

impl PendingPrefill {
    pub(super) fn remaining(&self) -> u32 {
        u32::try_from(self.request.tokens.len().saturating_sub(self.consumed)).unwrap_or(u32::MAX)
    }

    pub(super) const fn slot(&self) -> u32 {
        self.slot
    }

    pub(super) fn reserved_tokens(&self) -> u64 {
        self.reserved
    }

    #[allow(
        clippy::result_large_err,
        reason = "the owned prefill state is required to release its lane after a refusal"
    )]
    fn finish(self) -> Result<BatchLane, (Self, GenerationError)> {
        let Some(first) = self.selected else {
            return Err((self, GenerationError::new("prompt suffix is empty")));
        };
        Ok(BatchLane::new(
            self.request,
            self.slot,
            self.cached,
            first,
            self.sampling,
            self.tokenizer,
            self.constraint,
        ))
    }
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    #[allow(
        clippy::result_large_err,
        clippy::too_many_lines,
        reason = "ownership returns the prepared request so every construction failure can reply"
    )]
    pub(super) async fn begin_batch_prefill(
        &mut self,
        request: PreparedRequest,
        slot: u32,
    ) -> Result<PendingPrefill, (PreparedRequest, GenerationError)> {
        let sampling = match sampling_seed(request.params.seed, request.params.temperature).await {
            Ok(seed) => SamplingParams {
                temperature: request.params.temperature,
                top_k: request.params.top_k,
                top_p: request.params.top_p,
                seed,
            },
            Err(error) => return Err((request, error)),
        };
        let tokenizer = match self.tokenizer() {
            Ok(tokenizer) => tokenizer,
            Err(error) => return Err((request, error)),
        };
        let parser_started = std::time::Instant::now();
        let constraint = if request
            .constraint
            .as_ref()
            .is_some_and(|constraint| constraint.forced)
        {
            let guidance =
                match compile_constraint(&self.guidance, request.constraint.as_ref()).await {
                    Ok(guidance) => guidance,
                    Err(error) => return Err((request, error)),
                };
            let allow = match next_mask(&guidance).await {
                Ok(allow) => allow,
                Err(error) => return Err((request, error)),
            };
            let parser = match start_tool_parser(
                request.constraint.as_ref(),
                std::sync::Arc::clone(&self.prompt),
                true,
                request.sink.events(),
                &request.control,
            )
            .await
            {
                Ok(parser) => parser,
                Err(error) => return Err((request, error)),
            };
            Some(BatchConstraint {
                guidance,
                allow,
                parser,
            })
        } else {
            None
        };
        if constraint.is_some() {
            observe_phase(
                &mut self.metrics.parser_mask_time,
                &mut self.metrics.parser_mask_time_ns,
                parser_started.elapsed(),
            );
        }
        let Some(reserved) =
            u64::from(request.prompt_tokens).checked_add(u64::from(request.params.max_tokens))
        else {
            return Err((
                request,
                GenerationError::new("paged reservation overflowed"),
            ));
        };
        let match_started = std::time::Instant::now();
        let hit = match self
            .prefix_index
            .as_mut()
            .filter(|_| request.prefix_reuse)
            .map(|index| index.lookup(self.prefix_namespace, &request.tokens, reserved))
            .transpose()
        {
            Ok(hit) => hit.flatten(),
            Err(error) => {
                return Err((request, GenerationError::unavailable(error.to_string())));
            }
        };
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
        if request.tokens.get(cached..).is_none() || cached == request.tokens.len() {
            return Err((
                request,
                GenerationError::new("cached prefix leaves no prompt suffix"),
            ));
        }
        let prefix = hit.map(|hit| hit.sequence);
        loop {
            #[cfg(test)]
            let refuse_bind = self
                .batch_test_barrier
                .as_ref()
                .is_some_and(|barrier| barrier.refuse_bind_once());
            #[cfg(not(test))]
            let refuse_bind = false;
            let binding = if refuse_bind {
                Err(forja_host::bindings::l9o::gpu::compute::Error::Quota(
                    forja_host::PAGED_LANE_RELEASE_REQUIRED.to_owned(),
                ))
            } else {
                self.runner
                    .prepare_paged_lane(slot, prefix.clone(), reserved)
            };
            match binding {
                Ok(_) => break,
                Err(error) if forja_host::is_paged_lane_release_required(&error) => {
                    // WHY: Fixed-bucket graph capture records padded lanes as paged users, so first admission must release their capture occupant.
                    if let Err(error) = self.runner.prepare_paged_lane(slot, None, 0) {
                        return Err((request, batch_admission_error(&error)));
                    }
                }
                Err(error) if prefix.is_none() => {
                    let evicted = match self
                        .prefix_index
                        .as_mut()
                        .map(|index| index.evict_lru(None))
                        .transpose()
                    {
                        Ok(evicted) => evicted.unwrap_or(false),
                        Err(failure) => {
                            return Err((
                                request,
                                GenerationError::unavailable(failure.to_string()),
                            ));
                        }
                    };
                    if !evicted {
                        return Err((request, batch_admission_error(&error)));
                    }
                }
                Err(error) => return Err((request, batch_admission_error(&error))),
            }
        }
        let Ok(cached) = u32::try_from(cached) else {
            return Err((
                request,
                GenerationError::new("cached prefix length overflowed"),
            ));
        };
        Ok(PendingPrefill {
            request,
            slot,
            sampling,
            tokenizer,
            cached,
            reserved,
            consumed: usize::try_from(cached).unwrap_or(usize::MAX),
            position: cached,
            selected: None,
            constraint,
        })
    }

    pub(super) async fn advance_batch_prefill(
        &mut self,
        prefill: &mut PendingPrefill,
        budget: u32,
    ) -> Result<bool, GenerationError> {
        if self.scheduler.prefill_chunk_multiple.is_some() {
            return self.advance_batch_prefill_piece(prefill, budget).await;
        }
        let piece = self
            .scheduler
            .prompt_token_budget
            .min(self.scheduler.idle_prompt_token_budget);
        let mut remaining = budget;
        while remaining > 0 {
            let count = remaining.min(piece);
            if self.advance_batch_prefill_piece(prefill, count).await? {
                return Ok(true);
            }
            remaining -= count;
        }
        Ok(false)
    }

    async fn advance_batch_prefill_piece(
        &mut self,
        prefill: &mut PendingPrefill,
        budget: u32,
    ) -> Result<bool, GenerationError> {
        let remaining = prefill
            .request
            .tokens
            .get(prefill.consumed..)
            .ok_or_else(|| GenerationError::new("prompt chunk exceeds its suffix"))?;
        if remaining.is_empty() {
            return Ok(true);
        }
        let shapes = self.prompt_shapes();
        let largest = usize::try_from(shapes.last().map_or(0, |shape| shape.tokens)).unwrap_or(0);
        let count = usize::try_from(budget)
            .unwrap_or(usize::MAX)
            .min(largest)
            .min(remaining.len());
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.prompt_pieces.lock().unwrap().push(count);
        }
        let shape = shapes
            .iter()
            .copied()
            .find(|shape| usize::try_from(shape.tokens).is_ok_and(|size| size >= count))
            .ok_or_else(|| GenerationError::new("no prompt shape can accept the suffix"))?;
        let started = std::time::Instant::now();
        let output = self
            .runner
            .enqueue_batch_step(EngineBatch {
                shape,
                lanes: vec![Some(EngineBatchLane {
                    slot: prefill.slot,
                    tokens: Some(remaining[..count].to_vec()),
                    start_pos: prefill.position,
                    sampling: prefill.sampling,
                })],
            })
            .await
            .map_err(|error| GenerationError::unavailable(format!("prompt batch failed: {error}")))?
            .map_err(|error| {
                GenerationError::unavailable(format!("prompt batch failed: {error:?}"))
            })?;
        let output = self.select_prefill_output(prefill, output).await?;
        let bytes = self
            .runner
            .read_queued_batch_tokens(output)
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("prompt batch read failed: {error}"))
            })?;
        prefill.selected = read_batch_tokens(&bytes)?.first().copied();
        let count =
            u32::try_from(count).map_err(|_| GenerationError::new("prompt chunk overflowed"))?;
        prefill.position = prefill
            .position
            .checked_add(count)
            .ok_or_else(|| GenerationError::new("prompt position overflowed"))?;
        prefill.consumed = usize::try_from(prefill.position).unwrap_or(usize::MAX);
        observe(&mut self.metrics.suffix_prefill_time, started.elapsed());
        Ok(prefill.consumed == prefill.request.tokens.len())
    }

    #[allow(
        clippy::result_large_err,
        reason = "the owned prefill state is required to release its lane after a refusal"
    )]
    pub(super) fn finish_batch_prefill(
        prefill: PendingPrefill,
    ) -> Result<BatchLane, (PendingPrefill, GenerationError)> {
        prefill.finish()
    }

    pub(super) fn fail_batch_prefill(&mut self, prefill: PendingPrefill, error: GenerationError) {
        let error = self.release_failed_lane(prefill.slot()).unwrap_or(error);
        prefill.request.sink.fail(error);
    }

    pub(super) fn release_failed_lane(&mut self, slot: u32) -> Option<GenerationError> {
        #[cfg(test)]
        if self
            .batch_test_barrier
            .as_ref()
            .is_some_and(|barrier| barrier.release_is_blocked())
        {
            return Some(batch_admission_error(
                &forja_host::bindings::l9o::gpu::compute::Error::Quota(
                    forja_core::PagePoolError::BankInFlight.to_string(),
                ),
            ));
        }
        self.runner
            .prepare_paged_lane(slot, None, 0)
            .err()
            .map(|error| batch_admission_error(&error))
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

    async fn select_prefill_output(
        &mut self,
        prefill: &PendingPrefill,
        output: EngineBatchOutput,
    ) -> Result<EngineBatchOutput, GenerationError> {
        let Some(allow) = prefill
            .constraint
            .as_ref()
            .map(|constraint| constraint.allow.clone())
        else {
            return Ok(output);
        };
        let needed = prefill.slot.saturating_add(1);
        let shape = self
            .info
            .serving
            .iter()
            .flat_map(|serving| &serving.batch_shapes)
            .filter(|shape| {
                shape.phase == BatchPhase::Decode
                    && shape.selection == BatchSelection::Masked
                    && shape.bucket >= needed
            })
            .min_by_key(|shape| shape.bucket)
            .copied()
            .ok_or_else(|| GenerationError::new("no masked batch can accept the prompt lane"))?;
        let count = usize::try_from(shape.bucket)
            .map_err(|_| GenerationError::new("masked bucket exceeds usize"))?;
        let slot = usize::try_from(prefill.slot)
            .map_err(|_| GenerationError::new("prompt lane exceeds usize"))?;
        let mut lanes = vec![None; count];
        *lanes
            .get_mut(slot)
            .ok_or_else(|| GenerationError::new("prompt lane exceeds its masked bucket"))? =
            Some(EngineBatchSelectLane {
                start_pos: prefill
                    .position
                    .checked_add(
                        u32::try_from(
                            prefill
                                .request
                                .tokens
                                .len()
                                .saturating_sub(prefill.consumed),
                        )
                        .map_err(|_| GenerationError::new("prompt suffix exceeds u32"))?,
                    )
                    .and_then(|position| position.checked_sub(1))
                    .ok_or_else(|| GenerationError::new("prompt selection position overflowed"))?,
                sampling: prefill.sampling,
                allow,
            });
        self.runner
            .batch_select(&output.logits, EngineBatchSelect { shape, lanes })
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("prompt selection failed: {error}"))
            })?
            .map_err(|error| {
                GenerationError::unavailable(format!("prompt selection failed: {error:?}"))
            })
    }
}
