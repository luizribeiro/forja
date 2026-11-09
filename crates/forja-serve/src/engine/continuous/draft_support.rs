use super::{PreparedRequest, RequestEnvelope, lane::BatchLane};
use crate::{
    CompletionParams, CompletionRequest, DraftProposal, DraftSource, FinishReason,
    GenerationControl, GenerationError, ServingEngine, engine::BatchTestBarrier,
    http::AdmissionPolicy,
};
use forja_draft_testing::{Target, TestDrafter, Trace};
use forja_host::SamplingParams;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};
pub(super) type TestError = Box<dyn std::error::Error + Send + Sync>;
pub(super) struct Source {
    pub(super) drafter: TestDrafter,
    pub(super) count: usize,
    pub(super) cap: u32,
    pub(super) calls: Arc<AtomicUsize>,
}

impl DraftSource for Source {
    fn propose(&mut self, prefix: &[u32], _: u32, sampling: SamplingParams) -> DraftProposal {
        self.calls.fetch_add(1, Ordering::Relaxed);
        DraftProposal {
            tokens: self
                .drafter
                .propose(prefix, self.count, 4, |history| target(history, sampling)),
            rule_cap: self.cap,
        }
    }
}

pub(super) fn target(prefix: &[u32], sampling: SamplingParams) -> u32 {
    let mut logits = [1.0; 4];
    logits[usize::try_from((prefix.last().unwrap() + 1) % 4).unwrap()] = 4.0;
    Target::new(4, sampling).sample_logits(&logits, u32::try_from(prefix.len() - 1).unwrap())
}

pub(super) async fn engine() -> Result<ServingEngine<forja_cpu::CpuBackend>, TestError> {
    crate::engine::continuous::prefill::tests::engine_with_component(
        32,
        16,
        test_guests::verify_smoke(),
    )
    .await
}

pub(super) fn envelope(
    sampling: SamplingParams,
    max_tokens: u32,
) -> (
    RequestEnvelope,
    oneshot::Receiver<Result<crate::Completion, GenerationError>>,
) {
    let (reply, receiver) = oneshot::channel();
    (
        RequestEnvelope::Text {
            policy: AdmissionPolicy {
                config: Arc::new(forja_config::HostConfig::default()),
            },
            request: CompletionRequest {
                model: "hybrid".to_owned(),
                prompt: "bbbbbbb".to_owned(),
                params: CompletionParams {
                    max_tokens: Some(max_tokens),
                    temperature: Some(sampling.temperature),
                    top_k: Some(sampling.top_k),
                    top_p: Some(sampling.top_p),
                    seed: Some(sampling.seed),
                    ..Default::default()
                },
            },
            control: GenerationControl::new(Duration::from_secs(30)),
            reply,
            queued: Instant::now(),
        },
        receiver,
    )
}

pub(super) async fn run(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    max_tokens: u32,
) -> Result<Trace, TestError> {
    Box::pin(run_with_stop(
        sampling, drafter, count, cap, max_tokens, None,
    ))
    .await
}

pub(super) async fn run_with_stop(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    max_tokens: u32,
    stop: Option<u32>,
) -> Result<Trace, TestError> {
    Box::pin(run_with_boundaries(
        sampling, drafter, count, cap, max_tokens, stop, None,
    ))
    .await
}

pub(super) async fn run_with_boundaries(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    max_tokens: u32,
    stop: Option<u32>,
    stop_text: Option<String>,
) -> Result<Trace, TestError> {
    let mut engine = engine().await?;
    engine.chat_info.stop_token_ids = stop.into_iter().collect();
    if let Some(drafter) = drafter {
        engine = engine.with_drafter(Source {
            drafter,
            count,
            cap,
            calls: Arc::default(),
        });
    }
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let (mut request, reply) = envelope(sampling, max_tokens);
    if let RequestEnvelope::Text { request, .. } = &mut request {
        request.params.stop = stop_text.clone().map(crate::Stop::One);
    }
    let (_sender, mut receiver) = mpsc::channel(2);
    engine
        .run_batch_lanes(&mut receiver, &mut VecDeque::from([request]))
        .await;
    let completion = reply.await??;
    if let Some(stop_text) = &stop_text {
        let decoded: String = completion
            .token_ids
            .iter()
            .map(|id| char::from(b'a' + u8::try_from(*id).unwrap()))
            .collect();
        assert_eq!(completion.text, decoded.split(stop_text).next().unwrap());
    }
    let verified = barrier.verified.lock().unwrap();
    Ok(Trace {
        output: None,
        tokens: completion.token_ids,
        shapes: verified.iter().map(|(shape, _)| *shape).collect(),
        accepted: verified.iter().map(|(_, accepted)| *accepted).collect(),
        plain: usize::try_from(barrier.decoded_batches.load(Ordering::Acquire))?,
        cut: match completion.finish_reason {
            FinishReason::Length => Some(forja_host::speculative::VerifyCut::MaxTokens),
            FinishReason::Stop => Some(if stop_text.is_some() {
                forja_host::speculative::VerifyCut::Stop
            } else {
                forja_host::speculative::VerifyCut::Eos
            }),
            FinishReason::ToolCalls => None,
        },
    })
}

pub(super) async fn prepared_lane(
    engine: &mut ServingEngine<forja_cpu::CpuBackend>,
    prompt: usize,
    slot: u32,
) -> Result<BatchLane, TestError> {
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.tokens = vec![1; prompt];
    request.prompt_tokens = u32::try_from(prompt)?;
    request.params.max_tokens = 7;
    prepare_request(engine, request, slot).await
}

pub(super) async fn prepare_request(
    engine: &mut ServingEngine<forja_cpu::CpuBackend>,
    request: PreparedRequest,
    slot: u32,
) -> Result<BatchLane, TestError> {
    let mut prefill = engine
        .begin_batch_prefill(request, slot)
        .await
        .map_err(|(_, error)| error)?;
    while !engine.advance_batch_prefill(&mut prefill, 16).await? {}
    let mut lane = ServingEngine::<forja_cpu::CpuBackend>::finish_batch_prefill(prefill)
        .map_err(|(_, error)| error)?;
    let _ = lane
        .accept(lane.pending_token, &[], &mut engine.metrics)
        .await?;
    Ok(lane)
}

pub(super) fn source(calls: Arc<AtomicUsize>) -> Source {
    Source {
        drafter: TestDrafter::Oracle,
        count: 2,
        cap: 2,
        calls,
    }
}

#[tokio::test]
async fn protected_request_fixture_matches_its_prefix_target() -> Result<(), TestError> {
    let sampling = SamplingParams::default();
    let trace = Box::pin(run(sampling, None, 0, 0, 5)).await?;
    let mut history = vec![1; 7];
    let mut expected = Vec::new();
    for _ in 0..5 {
        let token = target(&history, sampling);
        history.push(token);
        expected.push(token);
    }
    assert_eq!(trace.tokens, expected);
    assert!(trace.plain > 0);
    assert_eq!(trace.shapes, [] as [u32; 0]);
    let mut engine = engine().await?;
    let lane = prepared_lane(&mut engine, 7, 0).await?;
    assert_eq!(lane.position, 7);
    assert_eq!(lane.pending_token, expected[0]);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut provider = source(Arc::clone(&calls));
    let proposal = provider.propose(&history, 12, sampling);
    assert_eq!(proposal.tokens[0], target(&history, sampling));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    Ok(())
}
