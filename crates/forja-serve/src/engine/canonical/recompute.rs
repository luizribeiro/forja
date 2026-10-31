use super::{GenerationControl, GenerationError, ServingEngine, cache_error};
use crate::prefix::PrefixLedger;
use forja_core::Backend;
use forja_host::CanonicalPrompt;
use std::sync::Arc;

pub(super) struct Recompute {
    prompt: Arc<CanonicalPrompt>,
    control: GenerationControl,
    position: Option<u32>,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
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
