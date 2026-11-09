use super::*;
use forja_draft_testing::Trace;
use forja_host::{BatchSelection, BatchShape};

pub(super) async fn run(
    sampling: SamplingParams,
    drafter: Option<TestDrafter>,
    count: usize,
    cap: u32,
    max_tokens: u32,
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
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.tokens = vec![1; 7];
    request.prompt_tokens = 7;
    request.params.max_tokens = max_tokens;
    request.params.temperature = sampling.temperature;
    request.params.top_k = sampling.top_k;
    request.params.top_p = sampling.top_p;
    request.params.seed = Some(sampling.seed);
    request.control = crate::GenerationControl::new(Duration::from_secs(30));
    let mut lane = prepare_request(&mut engine, request, 0).await?;
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let mut plain = 0;
    while lane.remaining_tokens() != 0 {
        let finish = if let Some(candidate) = engine.draft_candidate(&mut lane) {
            assert!(engine.runner.try_reserve_draft_lane(0)?);
            lane.draft_mode = DraftMode::Drafting;
            engine.run_draft_round(&mut lane, candidate).await?
        } else {
            plain += 1;
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
                        slot: 0,
                        tokens: Some(vec![lane.pending_token]),
                        start_pos: lane.position,
                        sampling,
                    })],
                })
                .await??;
            let tokens = read_batch_tokens(&engine.runner.read_queued_batch_tokens(output).await?)?;
            lane.position += 1;
            lane.accept_tokens(&tokens, &[], &mut engine.metrics)
                .await?
        };
        if finish.is_some() {
            break;
        }
    }
    let completion = lane.completion(FinishReason::Length).await?;
    let verified = barrier.verified.lock().unwrap();
    Ok(Trace {
        tokens: completion.token_ids,
        shapes: verified.iter().map(|(shape, _)| *shape).collect(),
        accepted: verified.iter().map(|(_, accepted)| *accepted).collect(),
        plain,
        cut: Some(forja_host::speculative::VerifyCut::MaxTokens),
    })
}
