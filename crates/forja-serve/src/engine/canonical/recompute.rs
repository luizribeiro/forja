use super::{GenerationControl, GenerationError, ServingEngine, cache_error};
use crate::prefix::PrefixLedger;
use forja_core::Backend;
use forja_host::bindings::l9o::gpu::compute::Error as ComputeError;
use forja_host::{
    BatchPhase, BatchSelection, CanonicalPrompt, EngineBatch, EngineBatchLane, HybridPrefix,
    SamplingParams,
};
use std::sync::Arc;

pub(super) struct Recompute<B: Backend> {
    prompt: Arc<CanonicalPrompt>,
    control: GenerationControl,
    position: Option<u32>,
    resume: Option<Arc<HybridPrefix<B>>>,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
    pub(in crate::engine) fn canonical_recompute_running(&self) -> bool {
        self.canonical_cache
            .as_ref()
            .is_some_and(|cache| cache.recompute.is_some())
    }
    pub(in crate::engine) async fn advance_canonical_recompute(&mut self) -> bool {
        let job = self.canonical_cache.as_mut().and_then(|cache| {
            cache
                .recompute
                .take()
                .or_else(|| cache.pending_recomputes.pop_front())
        });
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

    async fn recompute_one_block(&mut self, job: &mut Recompute<B>) -> Result<bool, ComputeError> {
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
            let hit = job
                .resume
                .take()
                .or(store.lookup(cache.namespace, job.prompt.tokens())?);
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

    pub(in crate::engine) fn pause_canonical_recompute(&mut self) -> Result<(), GenerationError> {
        let Some(mut job) = self
            .canonical_cache
            .as_mut()
            .and_then(|cache| cache.recompute.take())
        else {
            return Ok(());
        };
        let complete = self.canonical_cache.as_ref().is_some_and(|cache| {
            let block = cache.namespace.block().get() as usize;
            job.position.map(|position| position as usize)
                == Some(job.prompt.tokens().len() / block * block)
        });
        let checkpoint = if job.control.is_cancelled() {
            Ok(None)
        } else {
            self.checkpoint_recompute(&job, complete)
        };
        let released = self.release_recompute(&job);
        job.position = None;
        match checkpoint {
            Ok(_) if job.control.is_cancelled() => {
                self.metrics.recompute_cancelled =
                    self.metrics.recompute_cancelled.saturating_add(1);
            }
            Ok(_) if complete => {
                self.metrics.recompute_completed =
                    self.metrics.recompute_completed.saturating_add(1);
            }
            Ok(Some(resume)) => {
                job.resume = Some(resume);
                if let Some(cache) = &mut self.canonical_cache {
                    cache.pending_recomputes.push_front(job);
                    self.metrics.recompute_pending = true;
                }
                self.metrics.recompute_preempted =
                    self.metrics.recompute_preempted.saturating_add(1);
            }
            Ok(None) => {
                self.metrics.recompute_cancelled =
                    self.metrics.recompute_cancelled.saturating_add(1);
            }
            Err(error) => {
                self.metrics.recompute_failed = self.metrics.recompute_failed.saturating_add(1);
                self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
                tracing::warn!(%error, "cannot checkpoint background canonical state");
            }
        }
        released
    }

    fn checkpoint_recompute(
        &self,
        job: &Recompute<B>,
        complete: bool,
    ) -> Result<Option<Arc<HybridPrefix<B>>>, ComputeError> {
        let cache = self
            .canonical_cache
            .as_ref()
            .ok_or_else(|| ComputeError::InvalidHandle("canonical cache unavailable".into()))?;
        let store = cache
            .store
            .as_ref()
            .ok_or_else(|| ComputeError::InvalidHandle("canonical store unavailable".into()))?;
        let image = self.runner.capture_canonical_prefix(0, cache.allowance)?;
        if complete {
            store.publish(image, &job.control.cancellation.cancelled)?;
            Ok(None)
        } else {
            Ok(Some(image))
        }
    }

    pub(in crate::engine) fn stop_canonical_recompute(&mut self) -> Result<(), GenerationError> {
        let Some(cache) = &mut self.canonical_cache else {
            return Ok(());
        };
        let mut jobs = std::mem::take(&mut cache.pending_recomputes);
        if let Some(active) = cache.recompute.take() {
            jobs.push_front(active);
        }
        let mut result = Ok(());
        for job in jobs {
            if job.control.is_cancelled() {
                self.metrics.recompute_cancelled =
                    self.metrics.recompute_cancelled.saturating_add(1);
            } else {
                self.metrics.recompute_preempted =
                    self.metrics.recompute_preempted.saturating_add(1);
            }
            result = result.and(self.release_recompute(&job));
        }
        result
    }

    fn release_recompute(&mut self, job: &Recompute<B>) -> Result<(), GenerationError> {
        self.metrics.recompute_pending = self
            .canonical_cache
            .as_ref()
            .is_some_and(|cache| cache.recompute.is_some() || !cache.pending_recomputes.is_empty());
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
                .iter()
                .chain(&cache.pending_recomputes)
                .any(|job| !job.control.is_cancelled() && job.prompt.tokens() == tokens)
        {
            return;
        }
        let prompt = self.runner.canonical_context(cache.namespace, tokens);
        match prompt {
            Ok(prompt)
                if cache
                    .snapshot_bytes
                    .checked_add(prompt.charged_bytes())
                    .is_some_and(|bytes| bytes <= cache.allowance.bytes()) =>
            {
                let bound = usize::try_from(self.scheduler.active_lanes).unwrap_or(usize::MAX);
                debug_assert!(bound >= 1);
                if bound == 0 {
                    self.metrics.snapshot_refusals =
                        self.metrics.snapshot_refusals.saturating_add(1);
                    return;
                }
                self.metrics.recompute_queued = self.metrics.recompute_queued.saturating_add(1);
                self.metrics.recompute_pending = true;
                while cache.pending_recomputes.len() >= bound {
                    cache.pending_recomputes.pop_front();
                    self.metrics.recompute_superseded =
                        self.metrics.recompute_superseded.saturating_add(1);
                }
                cache.pending_recomputes.push_back(Recompute {
                    prompt,
                    control: control.clone(),
                    position: None,
                    resume: None,
                });
            }
            Ok(_) | Err(_) => {
                self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
            }
        }
    }
}
