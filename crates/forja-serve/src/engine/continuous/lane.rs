use tokio::sync::{mpsc, oneshot};

use forja_host::SamplingParams;

use super::{CompletionSink, PreparedRequest};
use crate::engine::{BoundaryAction, BoundaryState, GenerationError};
use crate::metrics::{ServingMetrics, observe, observe_phase};
use crate::prefix::PrefixLedger;
use crate::{Completion, FinishReason, GenerationControl, Usage};

struct DecodeRequest {
    token: u32,
    reply: oneshot::Sender<Result<Option<String>, String>>,
}

struct Detokenizer(mpsc::Sender<DecodeRequest>);

impl Detokenizer {
    fn spawn(tokenizer: tokenizers::Tokenizer) -> Self {
        let (sender, mut receiver) = mpsc::channel::<DecodeRequest>(1);
        tokio::spawn(async move {
            let mut decoder = tokenizer.decode_stream(true);
            while let Some(request) = receiver.recv().await {
                drop(
                    request.reply.send(
                        decoder
                            .step(request.token)
                            .map_err(|error| error.to_string()),
                    ),
                );
            }
        });
        Self(sender)
    }

    async fn step(&self, token: u32) -> Result<Option<String>, GenerationError> {
        let (reply, receiver) = oneshot::channel();
        self.0
            .send(DecodeRequest { token, reply })
            .await
            .map_err(|_| GenerationError::new("detokenizer stopped"))?;
        receiver
            .await
            .map_err(|_| GenerationError::new("detokenizer stopped"))?
            .map_err(|error| GenerationError::new(format!("cannot decode token: {error}")))
    }
}

pub(super) struct BatchLane {
    pub(super) slot: u32,
    pub(super) pending_token: u32,
    pub(super) position: u32,
    pub(super) reserved_tokens: u64,
    pub(super) sampling: SamplingParams,
    pub(super) control: GenerationControl,
    params: crate::ResolvedParams,
    request_params: crate::CompletionParams,
    sink: CompletionSink,
    prompt_tokens: u32,
    cached_tokens: u32,
    started: std::time::Instant,
    consumed: u32,
    decoder: Detokenizer,
    boundary: BoundaryState,
    text: String,
    token_ids: Vec<u32>,
    ledger: PrefixLedger,
}

impl BatchLane {
    pub(super) fn new(
        request: PreparedRequest,
        slot: u32,
        cached_tokens: u32,
        pending_token: u32,
        sampling: SamplingParams,
        tokenizer: tokenizers::Tokenizer,
    ) -> Self {
        Self {
            slot,
            pending_token,
            position: request.prompt_tokens,
            reserved_tokens: u64::from(request.prompt_tokens)
                + u64::from(request.params.max_tokens),
            sampling,
            control: request.control,
            boundary: BoundaryState::new(&request.params.stops, None),
            decoder: Detokenizer::spawn(tokenizer),
            ledger: PrefixLedger::new(request.tokens),
            params: request.params,
            request_params: request.request_params,
            sink: request.sink,
            prompt_tokens: request.prompt_tokens,
            cached_tokens,
            started: request.started,
            consumed: 0,
            text: String::new(),
            token_ids: Vec::new(),
        }
    }

    pub(super) async fn accept(
        &mut self,
        token: u32,
        stop_tokens: &[u32],
        metrics: &mut ServingMetrics,
    ) -> Result<Option<FinishReason>, GenerationError> {
        self.token_ids.push(token);
        if self.consumed == 0 {
            observe(&mut metrics.ttft, self.started.elapsed());
            if self.cached_tokens == 0 {
                observe(&mut metrics.uncached_ttft, self.started.elapsed());
            }
        }
        self.consumed += 1;
        if stop_tokens.contains(&token) {
            return Ok(Some(FinishReason::Stop));
        }
        let piece = self.decoder.step(token).await?;
        let sse_started = self.sink.events().map(|_| std::time::Instant::now());
        let action = self
            .boundary
            .process(
                token,
                piece,
                &mut self.text,
                self.sink.events(),
                &self.control,
            )
            .await?;
        if let Some(started) = sse_started {
            observe_phase(
                &mut metrics.sse_time,
                &mut metrics.sse_time_ns,
                started.elapsed(),
            );
        }
        if matches!(action, BoundaryAction::Stop) {
            return Ok(Some(FinishReason::Stop));
        }
        if matches!(action, BoundaryAction::Activate) {
            return Err(GenerationError::new("unexpected constraint activation"));
        }
        if self.consumed >= self.params.max_tokens {
            self.boundary
                .finish(&mut self.text, self.sink.events(), &self.control)
                .await?;
            return Ok(Some(FinishReason::Length));
        }
        self.ledger.record(token, true);
        self.pending_token = token;
        Ok(None)
    }

    pub(super) fn completion(&self, reason: FinishReason) -> Completion {
        Completion {
            text: self.text.clone(),
            tool_call: None,
            finish_reason: reason,
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                cached_tokens: self.cached_tokens,
                completion_tokens: self.consumed,
            },
            token_ids: self.token_ids.clone(),
        }
    }

    pub(super) fn needs_successor(&self) -> bool {
        self.consumed
            .checked_add(1)
            .is_some_and(|consumed| consumed < self.params.max_tokens)
    }

    pub(super) async fn reply(self, completion: Completion) {
        self.sink
            .finish(&self.request_params, completion, &self.control)
            .await;
    }

    pub(super) fn fail(self, error: GenerationError) {
        self.sink.fail(error);
    }

    pub(super) fn take_ledger(&mut self) -> PrefixLedger {
        std::mem::take(&mut self.ledger)
    }
}
