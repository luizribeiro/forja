use super::*;
use crate::{Completion, GenerationEvent, engine::continuous::CompletionSink};
use forja_draft_testing::{RequestOutput, Trace};
use forja_host::{BatchSelection, BatchShape};

pub(super) struct Case {
    pub(super) prompt: Vec<u32>,
    pub(super) max_tokens: u32,
    pub(super) stop_token: Option<u32>,
    pub(super) stop_text: Option<String>,
}

pub(super) async fn run(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    max_tokens: u32,
) -> Result<Trace, TestError> {
    Box::pin(run_case(
        sampling,
        drafter,
        count,
        cap,
        Case {
            prompt: vec![1; 7],
            max_tokens,
            stop_token: None,
            stop_text: None,
        },
    ))
    .await
}

pub(super) async fn run_case(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    case: Case,
) -> Result<Trace, TestError> {
    let mut engine = engine().await?;
    if let Some(drafter) = drafter {
        engine = engine.with_drafter(Source {
            drafter,
            count,
            cap,
            calls: Arc::default(),
        });
    }
    Box::pin(run_engine(engine, sampling, case)).await
}

pub(super) async fn run_engine(
    mut engine: ServingEngine<forja_cpu::CpuBackend>,
    sampling: SamplingParams,
    case: Case,
) -> Result<Trace, TestError> {
    engine.chat_info.stop_token_ids = case.stop_token.into_iter().collect();
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.prompt_tokens = u32::try_from(case.prompt.len())?;
    request.tokens = case.prompt;
    request.params.max_tokens = case.max_tokens;
    request.params.stops = case.stop_text.into_iter().collect();
    request.params.temperature = sampling.temperature;
    request.params.top_k = sampling.top_k;
    request.params.top_p = sampling.top_p;
    request.params.seed = Some(sampling.seed);
    request.control = crate::GenerationControl::new(Duration::from_secs(30));
    let (events, mut received) = mpsc::channel(64);
    let (reply, _receiver) = oneshot::channel();
    request.sink = CompletionSink::Stream { events, reply };
    let mut lane = prepare_request(&mut engine, request, 0).await?;
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let mut plain = 0;
    let mut reason = FinishReason::Length;
    let mut verified_finish = false;
    while lane.remaining_tokens() != 0 {
        let finish = if let Some(candidate) = engine.draft_candidate(&mut lane) {
            assert!(engine.runner.try_reserve_draft_lane(0)?);
            lane.draft_mode = DraftMode::Drafting;
            verified_finish = true;
            engine.run_draft_round(&mut lane, candidate).await?
        } else {
            plain += 1;
            verified_finish = false;
            plain_step(&mut engine, &mut lane).await?
        };
        if let Some(finish) = finish {
            reason = finish;
            break;
        }
    }
    let (completion, output) = observe(&engine, &mut lane, reason, &mut received).await?;
    if verified_finish {
        assert_eq!(lane.pending_token, *completion.token_ids.last().unwrap());
    }
    let verified = barrier.verified.lock().unwrap();
    Ok(Trace {
        output: Some(output),
        tokens: completion.token_ids,
        shapes: verified.iter().map(|(shape, _)| *shape).collect(),
        accepted: verified.iter().map(|(_, accepted)| *accepted).collect(),
        plain,
        cut: Some(match reason {
            FinishReason::Length => VerifyCut::MaxTokens,
            FinishReason::Stop if engine.chat_info.stop_token_ids.is_empty() => VerifyCut::Stop,
            FinishReason::Stop => VerifyCut::Eos,
            FinishReason::ToolCalls => unreachable!(),
        }),
    })
}

pub(super) async fn plain_step(
    engine: &mut ServingEngine<forja_cpu::CpuBackend>,
    lane: &mut BatchLane,
) -> Result<Option<FinishReason>, GenerationError> {
    let output = engine
        .runner
        .enqueue_batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                selection: BatchSelection::Combined,
                bucket: 1,
                tokens: 1,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: lane.slot,
                tokens: Some(vec![lane.pending_token]),
                start_pos: lane.position,
                sampling: lane.sampling,
            })],
        })
        .await
        .map_err(draft_error)?
        .map_err(draft_error)?;
    let tokens = read_batch_tokens(
        &engine
            .runner
            .read_queued_batch_tokens(output)
            .await
            .map_err(draft_error)?,
    )?;
    lane.position += 1;
    lane.accept_tokens(
        &tokens,
        &engine.chat_info.stop_token_ids,
        &mut engine.metrics,
    )
    .await
}

pub(super) async fn observe(
    engine: &ServingEngine<forja_cpu::CpuBackend>,
    lane: &mut BatchLane,
    reason: FinishReason,
    events: &mut mpsc::Receiver<GenerationEvent>,
) -> Result<(Completion, RequestOutput), TestError> {
    let completion = lane.completion(reason).await?;
    let ledger_len = u64::try_from(lane.ledger().tokens().len())?;
    let position = engine
        .runner
        .checkpoint_state_lane(lane.slot)?
        .unwrap()
        .position();
    let cancelled = lane.control.is_cancelled();
    assert_eq!(position.tokens + u64::from(cancelled), ledger_len);
    assert_eq!(position.sampler, position.tokens - 1);
    assert_eq!(position.grammar, position.tokens);
    assert_eq!(position.parser, position.tokens);
    assert_eq!(u64::from(lane.position), position.tokens);
    let kv_logical_len = engine
        .runner
        .checkpoint_paged_lane(lane.slot)?
        .unwrap()
        .logical_len();
    assert_eq!(kv_logical_len, position.tokens);
    let prompt = usize::try_from(completion.usage.prompt_tokens)?;
    assert_eq!(
        &lane.ledger().tokens()[prompt..],
        &completion.token_ids[..completion.token_ids.len() - usize::from(!cancelled)]
    );
    let mut streamed_text = String::new();
    while let Ok(event) = events.try_recv() {
        let GenerationEvent::Content(piece) = event else {
            panic!("unexpected event")
        };
        streamed_text.push_str(&piece);
    }
    assert_eq!(streamed_text, completion.text);
    let output = RequestOutput {
        streamed_text,
        usage: [
            completion.usage.prompt_tokens,
            completion.usage.cached_tokens,
            completion.usage.completion_tokens,
        ],
        ledger_len,
        kv_logical_len,
    };
    Ok((completion, output))
}
