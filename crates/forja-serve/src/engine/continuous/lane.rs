use tokio::sync::{mpsc, oneshot};

use forja_host::{
    SamplingParams,
    speculative::{VerifyAcceptance, VerifyCut, VerifyDecision},
};

use super::{CompletionSink, PreparedRequest};
use crate::engine::{
    BoundaryState, GenerationError, ToolParser, advance_speculative_constraint, finish_tool_parser,
    parse_tool_piece,
};
use crate::guidance::GuidanceSession;
use crate::metrics::{ServingMetrics, observe, observe_phase};
use crate::prefix::PrefixLedger;
use crate::{Completion, FinishReason, GenerationControl, Usage};

struct DecodeRequest {
    token: u32,
    reply: oneshot::Sender<Result<Option<String>, String>>,
}

struct Detokenizer(mpsc::Sender<DecodeRequest>);

impl Detokenizer {
    fn spawn(tokenizer: tokenizers::Tokenizer, skip_special_tokens: bool) -> Self {
        let (sender, mut receiver) = mpsc::channel::<DecodeRequest>(1);
        tokio::spawn(async move {
            let mut decoder = tokenizer.decode_stream(skip_special_tokens);
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
    pub(super) draft_mode: crate::draft::DraftMode,
    pub(super) draft_refused: bool,
    pub(super) drafting: crate::DraftMetrics,
    pub(super) draft_probe_position: Option<u32>,
    lookup: Option<crate::PromptLookup>,
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
    decode_started: std::time::Instant,
    consumed: u32,
    decoder: Detokenizer,
    boundary: BoundaryState,
    text: String,
    token_ids: Vec<u32>,
    ledger: PrefixLedger,
    constraint: Option<BatchConstraint>,
    prefix_reuse: bool,
    plain_only: bool,
}

pub(super) struct BatchConstraint {
    pub(super) guidance: GuidanceSession,
    pub(super) allow: Vec<u32>,
    pub(super) parser: Option<ToolParser>,
}

impl BatchLane {
    pub(super) fn new(
        request: PreparedRequest,
        slot: u32,
        cached_tokens: u32,
        pending_token: u32,
        sampling: SamplingParams,
        tokenizer: tokenizers::Tokenizer,
        constraint: Option<BatchConstraint>,
    ) -> Self {
        Self {
            slot,
            pending_token,
            draft_mode: crate::draft::DraftMode::Plain,
            draft_refused: false,
            drafting: crate::DraftMetrics::default(),
            draft_probe_position: None,
            lookup: None,
            position: request.prompt_tokens,
            reserved_tokens: u64::from(request.prompt_tokens)
                + u64::from(request.params.max_tokens),
            sampling,
            control: request.control,
            plain_only: request.constraint.is_some() || request.request_params.tools.is_some(),
            boundary: BoundaryState::new(&request.params.stops, None),
            decoder: Detokenizer::spawn(tokenizer, constraint.is_none()),
            ledger: PrefixLedger::new(request.tokens),
            params: request.params,
            request_params: request.request_params,
            sink: request.sink,
            prompt_tokens: request.prompt_tokens,
            cached_tokens,
            started: request.started,
            decode_started: std::time::Instant::now(),
            consumed: 0,
            text: String::new(),
            token_ids: Vec::new(),
            constraint,
            prefix_reuse: request.prefix_reuse,
        }
    }

    pub(super) const fn prefix_reuse(&self) -> bool {
        self.prefix_reuse
    }

    pub(super) fn lane_source<'a>(
        &'a mut self,
        external: Option<&'a mut (dyn crate::DraftSource + 'static)>,
        policy: Option<&forja_config::LookupPolicy>,
    ) -> Option<(&'a mut (dyn crate::DraftSource + 'static), &'a [u32])> {
        let source = if let Some(source) = external {
            source
        } else {
            let policy = policy?;
            self.lookup
                .get_or_insert_with(|| crate::PromptLookup::new(policy.clone()))
        };
        Some((source, self.ledger.tokens()))
    }

    pub(super) async fn accept_tokens(
        &mut self,
        tokens: &[u32],
        stop_tokens: &[u32],
        metrics: &mut ServingMetrics,
    ) -> Result<Option<FinishReason>, GenerationError> {
        for &token in tokens {
            if let Some(reason) = self.accept(token, stop_tokens, metrics).await? {
                return Ok(Some(reason));
            }
        }
        Ok(None)
    }

    pub(super) async fn accept_verified<'a>(
        &mut self,
        acceptance: VerifyAcceptance<'a>,
        stop_tokens: &[u32],
        metrics: &mut ServingMetrics,
    ) -> Result<VerifyDecision<'a>, GenerationError> {
        acceptance
            .scan_async(&mut (self, metrics), |(lane, metrics), token| {
                Box::pin(async move {
                    if lane.control.is_cancelled() {
                        return Err(GenerationError::cancelled());
                    }
                    let reason = lane.accept(token, stop_tokens, metrics).await?;
                    Ok(reason.map(|reason| match reason {
                        FinishReason::Stop if stop_tokens.contains(&token) => VerifyCut::Eos,
                        FinishReason::Stop => VerifyCut::Stop,
                        FinishReason::Length => VerifyCut::MaxTokens,
                        FinishReason::ToolCalls => VerifyCut::Activate,
                    }))
                })
            })
            .await
    }

    pub(super) fn remaining_tokens(&self) -> u64 {
        u64::from(self.params.max_tokens.saturating_sub(self.consumed))
    }

    pub(super) async fn accept(
        &mut self,
        token: u32,
        stop_tokens: &[u32],
        metrics: &mut ServingMetrics,
    ) -> Result<Option<FinishReason>, GenerationError> {
        self.token_ids.push(token);
        if self.consumed == 0 {
            observe_phase(
                &mut metrics.ttft,
                &mut metrics.first_token_time_ns,
                self.started.elapsed(),
            );
            if self.cached_tokens == 0 {
                observe(&mut metrics.uncached_ttft, self.started.elapsed());
            }
        }
        self.consumed += 1;
        if stop_tokens.contains(&token) {
            return Ok(Some(FinishReason::Stop));
        }
        let piece = self.decoder.step(token).await?;
        if let Some(constraint) = &mut self.constraint {
            let parser_started = std::time::Instant::now();
            parse_tool_piece(
                &mut constraint.parser,
                piece,
                self.sink.events(),
                &self.control,
            )
            .await?;
            let (complete, allow) =
                advance_speculative_constraint(Some(&constraint.guidance), token).await?;
            if let Some(allow) = allow {
                constraint.allow = allow;
            }
            observe_phase(
                &mut metrics.parser_mask_time,
                &mut metrics.parser_mask_time_ns,
                parser_started.elapsed(),
            );
            if complete {
                return Ok(Some(FinishReason::ToolCalls));
            }
            if self.consumed >= self.params.max_tokens {
                return Ok(Some(FinishReason::Length));
            }
            self.ledger.record(token, true);
            self.pending_token = token;
            return Ok(None);
        }
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
        if matches!(action, Some(VerifyCut::Stop)) {
            return Ok(Some(FinishReason::Stop));
        }
        if matches!(action, Some(VerifyCut::Activate)) {
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

    pub(super) async fn completion(
        &mut self,
        reason: FinishReason,
    ) -> Result<Completion, GenerationError> {
        let parser = self
            .constraint
            .take()
            .and_then(|constraint| constraint.parser);
        let tool_call = finish_tool_parser(&Ok(reason), parser).await?;
        let reason = crate::engine::tool_finish_reason(reason, &tool_call);
        self.drafting.max_decode_wall_ns =
            u64::try_from(self.decode_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.drafting.exit(match reason {
            FinishReason::Length => "max-tokens",
            FinishReason::Stop => "stop-or-eos",
            FinishReason::ToolCalls => "tool-call",
        });
        Ok(Completion {
            text: self.text.clone(),
            tool_calls: tool_call,
            finish_reason: reason,
            usage: Usage {
                prompt_tokens: self.prompt_tokens,
                cached_tokens: self.cached_tokens,
                completion_tokens: self.consumed,
            },
            token_ids: self.token_ids.clone(),
            drafting: self.drafting.clone(),
        })
    }

    pub(super) fn needs_successor(&self) -> bool {
        self.remaining_tokens() > 1
    }

    pub(super) fn can_draft(&self) -> bool {
        !self.plain_only && !self.is_constrained()
    }

    pub(super) const fn is_constrained(&self) -> bool {
        self.constraint.is_some()
    }

    pub(super) fn constraint_allow(&self) -> Option<&[u32]> {
        self.constraint
            .as_ref()
            .map(|constraint| constraint.allow.as_slice())
    }

    pub(super) async fn reply(self, completion: Completion) {
        self.sink
            .finish(&self.request_params, completion, &self.control)
            .await;
    }

    pub(super) fn fail(self, error: GenerationError) {
        self.sink.fail(error);
    }

    pub(super) const fn ledger(&self) -> &PrefixLedger {
        &self.ledger
    }

    pub(super) fn take_ledger(&mut self) -> PrefixLedger {
        std::mem::take(&mut self.ledger)
    }
}

#[cfg(test)]
mod tests {
    mod boundary;
    use super::*;

    fn streaming_lane(stops: &[&str]) -> (BatchLane, mpsc::Receiver<crate::GenerationEvent>) {
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0), ("b".to_owned(), 1)], Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        let mut request = crate::engine::continuous::prefill::tests::request();
        request.params.max_tokens = 7;
        request.params.stops = stops.iter().map(|stop| (*stop).to_owned()).collect();
        let (events, received) = mpsc::channel(16);
        let (reply, _receiver) = oneshot::channel();
        request.sink = CompletionSink::Stream { events, reply };
        let lane = BatchLane::new(request, 0, 0, 0, SamplingParams::default(), tokenizer, None);
        (lane, received)
    }

    fn acceptance<'a>(samples: &'a [u32], drafts: &[u32]) -> VerifyAcceptance<'a> {
        use forja_host::{
            BatchPhase, BatchSelection, BatchShape,
            speculative::{VerifyBudget, select_verify_shape},
        };
        let shapes = [BatchShape {
            phase: BatchPhase::Verify,
            selection: BatchSelection::Combined,
            bucket: 1,
            tokens: u32::try_from(samples.len()).unwrap(),
        }];
        let fed = select_verify_shape(&shapes, drafts, VerifyBudget::new(7, 2, 7, 32, 32).unwrap())
            .unwrap();
        VerifyAcceptance::new(fed, samples, 2).unwrap()
    }

    #[tokio::test]
    async fn live_stop_strings_span_rounds_without_scanning_or_streaming_the_tail() {
        let (mut lane, mut events) = streaming_lane(&["bba"]);
        let mut metrics = ServingMetrics::default();
        let first = lane
            .accept_verified(acceptance(&[0, 1], &[0]), &[], &mut metrics)
            .await
            .unwrap();
        assert_eq!(first.cut(), None);
        let second = lane
            .accept_verified(acceptance(&[1, 0, 1], &[1, 0]), &[], &mut metrics)
            .await
            .unwrap();
        assert_eq!(second.cut(), Some(VerifyCut::Stop));
        assert_eq!(second.accepted(), 1);
        assert_eq!(second.tokens(), [1, 0]);
        assert_eq!(lane.token_ids, [0, 1, 1, 0]);
        assert_eq!(lane.text, "a");
        assert_eq!(lane.remaining_tokens(), 3);
        assert_eq!(
            events.try_recv().unwrap(),
            crate::GenerationEvent::Content("a".to_owned())
        );
        assert!(events.try_recv().is_err());
        assert_eq!(&lane.ledger().tokens()[65..], &[0, 1, 1]);
    }

    #[tokio::test]
    async fn accepted_rounds_stream_in_order_with_one_shared_output_budget() {
        let (mut lane, mut received) = streaming_lane(&[]);
        let mut metrics = ServingMetrics::default();
        assert_eq!(
            lane.accept_tokens(&[0, 1, 0], &[], &mut metrics)
                .await
                .unwrap(),
            None
        );
        assert_eq!(lane.remaining_tokens(), 4);
        assert!(lane.needs_successor());
        assert_eq!(lane.token_ids, [0, 1, 0]);
        assert_eq!(lane.text, "aba");
        for text in ["a", "b", "a"] {
            assert_eq!(
                received.try_recv().unwrap(),
                crate::GenerationEvent::Content(text.to_owned())
            );
        }
        assert_eq!(
            lane.accept_tokens(&[1, 1, 0, 1], &[], &mut metrics)
                .await
                .unwrap(),
            Some(FinishReason::Length)
        );
        assert_eq!(lane.remaining_tokens(), 0);
        assert!(!lane.needs_successor());
        let completion = lane.completion(FinishReason::Length).await.unwrap();
        assert_eq!(completion.usage.completion_tokens, 7);
        assert!(completion.drafting.max_decode_wall_ns > 0);
        assert_eq!(completion.token_ids, [0, 1, 0, 1, 1, 0, 1]);
        assert_eq!(completion.text, "ababbab");
    }

    #[tokio::test]
    async fn constrained_lanes_preserve_special_token_delimiters() {
        let vocab = [
            ("a".to_owned(), 0),
            ("<close>".to_owned(), 1),
            ("<eos>".to_owned(), 2),
        ];
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("<close>", true)]);
        let worker = crate::guidance::GuidanceWorker::new(&tokenizer, 3, &[2]).unwrap();
        for constrained in [false, true] {
            let constraint = if constrained {
                let guidance = worker.compile(r#"start: "a""#.to_owned()).await.unwrap();
                let allow = guidance.mask().await.unwrap().unwrap();
                Some(BatchConstraint {
                    guidance,
                    allow,
                    parser: None,
                })
            } else {
                None
            };
            let lane = BatchLane::new(
                crate::engine::continuous::prefill::tests::request(),
                0,
                0,
                0,
                SamplingParams::default(),
                tokenizer.clone(),
                constraint,
            );
            let piece = lane.decoder.step(1).await.unwrap();
            assert_eq!(piece.as_deref(), constrained.then_some("<close>"));
        }
    }
}
