use super::{GenerationControl, ServingEngine};
use crate::prefix::PrefixLedger;
use forja_core::Backend;
use forja_host::CanonicalPrompt;
use std::sync::Arc;

pub(super) struct Recompute {
    prompt: Arc<CanonicalPrompt>,
    control: GenerationControl,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
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
                cache.recompute = Some(Recompute {
                    prompt,
                    control: control.clone(),
                });
            }
            Ok(_) | Err(_) => {
                self.metrics.snapshot_refusals = self.metrics.snapshot_refusals.saturating_add(1);
            }
        }
    }
}
