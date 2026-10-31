use super::{GenerationControl, GenerationError, ServingEngine, cache_error};
use crate::prefix::PrefixLedger;
use forja_core::Backend;
use forja_host::bindings::l9o::gpu::compute::Error as ComputeError;
use forja_host::{
    BatchPhase, BatchSelection, CanonicalPrompt, EngineBatch, EngineBatchLane, SamplingParams,
};
use std::sync::Arc;

pub(super) struct Recompute {
    prompt: Arc<CanonicalPrompt>,
    control: GenerationControl,
    position: Option<u32>,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
    pub(in crate::engine) async fn advance_canonical_recompute(&mut self) -> bool {
        let job = self
            .canonical_cache
            .as_mut()
            .and_then(|cache| cache.recompute.take());
        let Some(mut job) = job else {
            return false;
        };
        let result = if job.control.is_cancelled() {
            Ok(true)
        } else {
            self.recompute_one_block(&mut job).await
        };
        if job.control.is_cancelled() {
            self.metrics.recompute_cancelled = self.metrics.recompute_cancelled.saturating_add(1);
        } else {
            match result {
                Ok(false) => {
                    if let Some(cache) = &mut self.canonical_cache {
                        cache.recompute = Some(job);
                        return true;
                    }
                }
                Ok(true) => {
                    self.metrics.recompute_completed =
                        self.metrics.recompute_completed.saturating_add(1);
                }
                Err(error) => {
                    if matches!(error, ComputeError::Quota(_)) {
                        self.metrics.snapshot_refusals =
                            self.metrics.snapshot_refusals.saturating_add(1);
                    }
                    self.metrics.recompute_failed = self.metrics.recompute_failed.saturating_add(1);
                    tracing::warn!(%error, "background canonical recompute refused");
                }
            }
        }
        if let Err(error) = self.release_recompute(&job) {
            tracing::warn!(%error, "cannot release background canonical state");
        }
        true
    }

    async fn recompute_one_block(&mut self, job: &mut Recompute) -> Result<bool, ComputeError> {
        let cache = self
            .canonical_cache
            .as_ref()
            .ok_or_else(|| ComputeError::InvalidHandle("canonical cache unavailable".into()))?;
        let store = cache
            .store
            .as_ref()
            .ok_or_else(|| ComputeError::InvalidHandle("canonical store unavailable".into()))?;
        let block = cache.namespace.block().get();
        let length = u32::try_from(job.prompt.tokens().len())
            .map_err(|_| ComputeError::Quota("context length overflow".into()))?;
        let target = length / block * block;
        if job.position.is_none() {
            let hit = store.lookup(cache.namespace, job.prompt.tokens())?;
            if hit
                .as_ref()
                .is_some_and(|hit| u32::try_from(hit.tokens().len()) == Ok(target))
            {
                return Ok(true);
            }
            self.runner.prepare_paged_lane(0, None, 0)?;
            let start = self.runner.prepare_canonical_lane(
                0,
                job.prompt.clone(),
                hit.as_ref(),
                u64::from(length),
            )?;
            job.position = Some(
                u32::try_from(start)
                    .map_err(|_| ComputeError::Quota("context position overflow".into()))?,
            );
        }
        let position = job
            .position
            .ok_or_else(|| ComputeError::InvalidHandle("background lane unavailable".into()))?;
        if position == target {
            let image = self.runner.capture_canonical_prefix(0, cache.allowance)?;
            store.publish(image, &job.control.cancellation.cancelled)?;
            return Ok(true);
        }
        let end = position
            .checked_add(block)
            .filter(|end| *end <= target)
            .ok_or_else(|| ComputeError::Quota("canonical block extent overflow".into()))?;
        let shape = self
            .info
            .serving
            .as_ref()
            .into_iter()
            .flat_map(|info| &info.batch_shapes)
            .find(|shape| {
                shape.phase == BatchPhase::Prompt
                    && shape.selection == BatchSelection::Combined
                    && shape.bucket == 1
                    && shape.tokens == block
            })
            .copied()
            .ok_or_else(|| {
                ComputeError::InvalidHandle("canonical block shape unavailable".into())
            })?;
        let range = usize::try_from(position)
            .map_err(|_| ComputeError::Quota("context position overflow".into()))?
            ..usize::try_from(end)
                .map_err(|_| ComputeError::Quota("context extent overflow".into()))?;
        let output = self
            .runner
            .enqueue_batch_step(EngineBatch {
                shape,
                lanes: vec![Some(EngineBatchLane {
                    slot: 0,
                    tokens: Some(job.prompt.tokens()[range].to_vec()),
                    start_pos: position,
                    sampling: SamplingParams::default(),
                })],
            })
            .await
            .map_err(|error| ComputeError::BackendExecution(error.to_string()))??;
        #[cfg(test)]
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.pause_once().await;
        }
        self.runner.read_queued_batch_tokens(output).await?;
        self.metrics.recompute_blocks = self.metrics.recompute_blocks.saturating_add(1);
        job.position = Some(end);
        Ok(false)
    }

    pub(in crate::engine) fn stop_canonical_recompute(&mut self) -> Result<(), GenerationError> {
        let job = self
            .canonical_cache
            .as_mut()
            .and_then(|cache| cache.recompute.take());
        let Some(job) = job else {
            return Ok(());
        };
        if job.control.is_cancelled() {
            self.metrics.recompute_cancelled = self.metrics.recompute_cancelled.saturating_add(1);
        } else {
            self.metrics.recompute_preempted = self.metrics.recompute_preempted.saturating_add(1);
        }
        self.release_recompute(&job)
    }

    fn release_recompute(&mut self, job: &Recompute) -> Result<(), GenerationError> {
        self.metrics.recompute_pending = false;
        if job.position.is_some() {
            self.runner
                .prepare_paged_lane(0, None, 0)
                .map_err(cache_error)?;
        }
        Ok(())
    }

    pub(in crate::engine) fn queue_canonical_recompute(
        &mut self,
        ledger: &PrefixLedger,
        prompt_tokens: u32,
        reply: &[u32],
        control: &GenerationControl,
        reuse: bool,
    ) {
        let Some(cache) = self.canonical_cache.as_mut().filter(|cache| {
            reuse
                && self.config.serving.prefix_reuse
                && cache.allowance.bytes() > 0
                && cache.store.is_some()
                && !control.is_cancelled()
        }) else {
            return;
        };
        let Some(tokens) = ledger.completed_context(prompt_tokens, reply) else {
            self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
            return;
        };
        if tokens.len() < cache.namespace.block().get() as usize
            || cache
                .recompute
                .as_ref()
                .is_some_and(|job| !job.control.is_cancelled() && job.prompt.tokens() == tokens)
        {
            return;
        }
        let prompt = self.runner.canonical_context(cache.namespace, tokens);
        match prompt {
            Ok(prompt) if prompt.charged_bytes() <= cache.allowance.bytes() => {
                self.metrics.recompute_queued = self.metrics.recompute_queued.saturating_add(1);
                self.metrics.recompute_pending = true;
                cache.recompute = Some(Recompute {
                    prompt,
                    control: control.clone(),
                    position: None,
                });
            }
            Ok(_) | Err(_) => {
                self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
            }
        }
    }
}
