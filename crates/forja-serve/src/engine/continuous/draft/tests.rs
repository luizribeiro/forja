mod cuts;
mod direct;
mod lookup;
use super::super::{RequestEnvelope, draft_support::*};
use super::*;
use crate::engine::BatchTestBarrier;
use forja_draft_testing::{TestDrafter, assert_draft_exact};
use forja_host::SamplingParams;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};

fn exact_suite(
    mut run: impl FnMut(
        SamplingParams,
        Option<TestDrafter>,
        usize,
        u32,
        u32,
    ) -> forja_draft_testing::Trace,
) {
    for (count, cap, max_tokens, verify) in [
        (2, 2, 17, true),
        (2, 2, 25, true),
        (1, 2, 7, true),
        (2, 1, 7, true),
        (0, 2, 7, false),
        (2, 0, 7, false),
        (2, 2, 2, false),
        (2, 2, 1, false),
    ] {
        assert_draft_exact(verify, |sampling, drafter| {
            let trace = run(sampling, drafter, count, cap, max_tokens);
            assert_eq!(
                trace.cut,
                Some(forja_host::speculative::VerifyCut::MaxTokens)
            );
            trace
        });
    }
}

#[test]
fn direct_rounds_match_plain_streams_for_every_test_drafter() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    exact_suite(|sampling, drafter, count, cap, max_tokens| {
        runtime
            .block_on(direct::run(sampling, drafter, count, cap, max_tokens))
            .unwrap()
    });
}

#[test]
fn serving_draft_rounds_match_plain_streams_for_every_test_drafter() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    exact_suite(|sampling, drafter, count, cap, max_tokens| {
        runtime
            .block_on(run(sampling, drafter, count, cap, max_tokens))
            .unwrap()
    });
}

#[tokio::test]
async fn a_second_request_drains_the_round_before_returning_to_normal_buckets()
-> Result<(), TestError> {
    let sampling = SamplingParams::default();
    let expected = Box::pin(run(sampling, None, 0, 0, 11)).await?.tokens;
    let mut engine = engine().await?.with_drafter(Source {
        drafter: TestDrafter::Oracle,
        count: 2,
        cap: 2,
        calls: Arc::default(),
    });
    let barrier = Arc::new(BatchTestBarrier::new());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let (request, first) = envelope(sampling, 11);
    let (sender, mut receiver) = mpsc::channel(2);
    let inject = async {
        tokio::time::timeout(Duration::from_secs(10), barrier.reached.notified())
            .await
            .unwrap();
        let (request, reply) = envelope(sampling, 3);
        let RequestEnvelope::Text {
            policy,
            request,
            control,
            reply: sender_reply,
            queued,
        } = request
        else {
            unreachable!()
        };
        sender
            .send(crate::http::ServiceCommand::Text {
                policy,
                request,
                control,
                reply: sender_reply,
                queued,
            })
            .await
            .unwrap();
        barrier.resume.notify_one();
        reply.await
    };
    let mut pending = VecDeque::from([request]);
    let ((), second) = tokio::join!(engine.run_batch_lanes(&mut receiver, &mut pending), inject);
    assert_eq!(first.await??.token_ids, expected);
    assert_eq!(
        second??.token_ids,
        Box::pin(run(sampling, None, 0, 0, 3)).await?.tokens
    );
    assert!(
        engine
            .metrics
            .decode_batches
            .get(&2)
            .is_some_and(|count| *count > 0)
    );
    assert!(barrier.draft_exit_steps.load(Ordering::Acquire) >= 2);
    Ok(())
}

#[tokio::test]
async fn failed_or_cancelled_verify_restores_the_pre_round_boundary() -> Result<(), TestError> {
    for cancel in [false, true] {
        let mut engine = engine().await?.with_drafter(source(Arc::default()));
        let mut lane = prepared_lane(&mut engine, if cancel { 7 } else { 2 }, 0).await?;
        let before = engine.runner.checkpoint_state_lane(0)?.unwrap().position();
        let fingerprint = engine.runner.state_lane_fingerprint(0)?;
        assert!(engine.runner.try_reserve_draft_lane(0)?);
        let candidate = engine.draft_candidate(&mut lane).unwrap();
        let result = if cancel {
            let barrier = Arc::new(BatchTestBarrier::new());
            engine.set_batch_test_barrier(Arc::clone(&barrier));
            let control = lane.control.clone();
            let interrupt = async {
                tokio::time::timeout(Duration::from_secs(10), barrier.reached.notified())
                    .await
                    .unwrap();
                control.cancel();
                barrier.resume.notify_one();
            };
            tokio::join!(engine.run_draft_round(&mut lane, candidate), interrupt).0
        } else {
            engine.run_draft_round(&mut lane, candidate).await
        };
        assert!(result.is_err());
        assert_eq!(
            engine.runner.checkpoint_state_lane(0)?.unwrap().position(),
            before
        );
        assert_eq!(engine.runner.state_lane_fingerprint(0)?, fingerprint);
        assert_eq!(
            engine.runner.paged_lane_prefix(0)?.unwrap().logical_len()?,
            before.tokens
        );
        assert!(engine.runner.try_release_draft_lane(0)?);
        assert_eq!(lane.position, u32::try_from(before.tokens)?);
    }
    Ok(())
}

#[tokio::test]
async fn routing_refusals_preserve_ordinary_lanes_without_invoking_constraints()
-> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = engine().await?.with_drafter(source(Arc::clone(&calls)));
    let first = prepared_lane(&mut engine, 7, 0).await?;
    let second = prepared_lane(&mut engine, 7, 1).await?;
    let mut active = vec![first, second];
    assert!(!engine.advance_drafting(&mut active, true).await);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let peer = active.pop().unwrap();
    assert!(engine.runner.try_reserve_draft_lane(peer.slot)?);
    assert!(!engine.advance_drafting(&mut active, true).await);
    assert_eq!(active[0].draft_mode, DraftMode::Plain);
    assert!(engine.runner.try_release_draft_lane(peer.slot)?);
    engine.runner.prepare_paged_lane(peer.slot, None, 0)?;
    reset_draft_refusals(&mut active);
    assert!(!engine.advance_drafting(&mut active, false).await);
    let calls_before = calls.load(Ordering::Relaxed);
    engine
        .info
        .serving
        .as_mut()
        .unwrap()
        .batch_shapes
        .retain(|shape| shape.phase != BatchPhase::Verify);
    assert!(engine.draft_candidate(&mut active[0]).is_none());
    assert_eq!(calls.load(Ordering::Relaxed), calls_before);
    Ok(())
}

#[tokio::test]
async fn constrained_lanes_never_consult_the_draft_source() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = engine().await?.with_drafter(source(Arc::clone(&calls)));
    let guidance = engine.guidance.compile(r#"start: "a""#.to_owned()).await?;
    let allow = guidance.mask().await?.unwrap();
    let mut lane = BatchLane::new(
        crate::engine::continuous::prefill::tests::request(),
        0,
        0,
        0,
        SamplingParams::default(),
        engine.tokenizer()?,
        Some(crate::engine::continuous::lane::BatchConstraint {
            guidance,
            allow,
            parser: None,
        }),
    );
    assert!(engine.draft_candidate(&mut lane).is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    Ok(())
}

#[tokio::test]
async fn a_serial_barrier_stops_draft_entry_probes_after_the_exit_drain() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = engine().await?.with_drafter(source(Arc::clone(&calls)));
    let barrier = Arc::new(BatchTestBarrier::new());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let (request, first) = envelope(SamplingParams::default(), 11);
    let (sender, mut receiver) = mpsc::channel(2);
    let inject = async {
        tokio::time::timeout(Duration::from_secs(10), barrier.reached.notified())
            .await
            .unwrap();
        let before = calls.load(Ordering::Relaxed);
        let (reply, _response) = oneshot::channel();
        sender
            .send(crate::http::ServiceCommand::Quiesce { reply })
            .await
            .unwrap();
        barrier.resume.notify_one();
        before
    };
    let mut pending = VecDeque::from([request]);
    let ((), before) = tokio::join!(engine.run_batch_lanes(&mut receiver, &mut pending), inject);
    assert_eq!(
        first.await??.token_ids,
        Box::pin(run(SamplingParams::default(), None, 0, 0, 11))
            .await?
            .tokens
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert_eq!(barrier.verified.lock().unwrap().len(), 1);
    assert!(matches!(
        pending.front(),
        Some(RequestEnvelope::Quiesce { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn refused_spares_keep_ordinary_batches_pipelined() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = engine().await?.with_drafter(source(Arc::clone(&calls)));
    engine.runner.prepare_paged_lane(1, None, 1)?;
    assert!(engine.runner.try_reserve_draft_lane(1)?);
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let (request, reply) = envelope(SamplingParams::default(), 11);
    let (_sender, mut receiver) = mpsc::channel(2);
    engine
        .run_batch_lanes(&mut receiver, &mut VecDeque::from([request]))
        .await;
    assert_eq!(
        reply.await??.token_ids,
        Box::pin(run(SamplingParams::default(), None, 0, 0, 11))
            .await?
            .tokens
    );
    assert!(barrier.pipelined_batches.load(Ordering::Acquire) > 0);
    assert!(barrier.verified.lock().unwrap().is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert!(engine.runner.try_release_draft_lane(1)?);
    Ok(())
}

#[tokio::test]
async fn zero_acceptance_keeps_drafting_while_proposals_continue() -> Result<(), TestError> {
    let mut provider = source(Arc::default());
    provider.drafter = TestDrafter::Adversary;
    let mut engine = engine().await?.with_drafter(provider);
    let lane = prepared_lane(&mut engine, 7, 0).await?;
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let mut active = vec![lane];
    assert!(engine.advance_drafting(&mut active, true).await);
    assert_eq!(active[0].draft_mode, DraftMode::Drafting);
    assert!(engine.advance_drafting(&mut active, true).await);
    assert_eq!(active[0].draft_mode, DraftMode::Drafting);
    assert_eq!(*barrier.verified.lock().unwrap(), [(3, 0), (3, 0)]);
    assert_eq!(barrier.draft_exit_steps.load(Ordering::Acquire), 0);
    Ok(())
}

#[test]
fn stop_tokens_inside_accepted_drafts_match_the_plain_stream() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    assert_draft_exact(true, |sampling, drafter| {
        let mut prefix = vec![1; 7];
        let mut plain = Vec::new();
        for _ in 0..4 {
            let token = target(&prefix, sampling);
            plain.push(token);
            prefix.push(token);
        }
        let stop = *plain[2..]
            .iter()
            .find(|token| !plain[..2].contains(token))
            .unwrap();
        let trace = runtime
            .block_on(run_with_stop(sampling, drafter, 2, 2, 17, Some(stop)))
            .unwrap();
        assert_eq!(trace.tokens.last(), Some(&stop));
        assert_eq!(trace.cut, Some(forja_host::speculative::VerifyCut::Eos));
        assert!(trace.tokens.len() <= 4);
        trace
    });
}

#[tokio::test]
async fn a_finished_drafting_lane_releases_spares_before_slot_reuse() -> Result<(), TestError> {
    let mut engine = engine().await?.with_drafter(source(Arc::default()));
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let (_sender, mut receiver) = mpsc::channel(2);
    let (request, reply) = envelope(SamplingParams::default(), 5);
    engine
        .run_batch_lanes(&mut receiver, &mut VecDeque::from([request]))
        .await;
    let first = reply.await??;
    assert_eq!(*barrier.verified.lock().unwrap(), [(3, 2)]);
    engine.runner.prepare_paged_lane(1, None, 1)?;
    assert!(engine.runner.try_reserve_draft_lane(1)?);
    assert!(engine.runner.try_release_draft_lane(1)?);
    engine.runner.prepare_paged_lane(1, None, 0)?;
    let (request, reply) = envelope(SamplingParams::default(), 5);
    engine
        .run_batch_lanes(&mut receiver, &mut VecDeque::from([request]))
        .await;
    assert_eq!(reply.await??.token_ids, first.token_ids);
    assert_eq!(*barrier.verified.lock().unwrap(), [(3, 2), (3, 2)]);
    Ok(())
}

#[tokio::test]
async fn draft_drain_predicate_requires_a_fresh_retryable_lane() -> Result<(), TestError> {
    let mut engine = engine().await?;
    let mut lane = prepared_lane(&mut engine, 7, 0).await?;
    assert!(wants_draft_drain(true, true, std::slice::from_ref(&lane)));
    assert!(!wants_draft_drain(false, true, std::slice::from_ref(&lane)));
    assert!(!wants_draft_drain(true, false, std::slice::from_ref(&lane)));
    lane.draft_refused = true;
    lane.position += 1;
    assert!(!wants_draft_drain(true, true, std::slice::from_ref(&lane)));
    reset_draft_refusals(std::slice::from_mut(&mut lane));
    assert!(wants_draft_drain(true, true, std::slice::from_ref(&lane)));
    lane.draft_probe_position = Some(lane.position);
    assert!(!wants_draft_drain(true, true, std::slice::from_ref(&lane)));
    Ok(())
}

#[tokio::test]
async fn candidates_bind_one_owned_prefix_and_refuse_ineligible_lanes() -> Result<(), TestError> {
    let mut engine = engine().await?;
    let mut lane = prepared_lane(&mut engine, 7, 0).await?;
    assert!(engine.draft_candidate(&mut lane).is_none());
    let calls = Arc::new(AtomicUsize::new(0));
    engine = engine.with_drafter(source(Arc::clone(&calls)));
    let candidate = engine.draft_candidate(&mut lane).unwrap();
    assert_eq!(candidate.fed.shape().tokens, 3);
    assert_eq!(candidate.fed.tokens(), [3, 0]);
    assert!(engine.draft_candidate(&mut lane).is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    reset_draft_refusals(std::slice::from_mut(&mut lane));
    lane.draft_mode = DraftMode::Exiting;
    assert!(engine.draft_candidate(&mut lane).is_none());
    lane.draft_mode = DraftMode::Drafting;
    engine
        .info
        .serving
        .as_mut()
        .unwrap()
        .batch_shapes
        .retain(|shape| shape.phase != BatchPhase::Verify);
    assert!(engine.draft_candidate(&mut lane).is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn multi_token_stop_strings_match_plain_decode_for_every_test_drafter() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for length in [3, 4, 5] {
        assert_draft_exact(true, |sampling, drafter| {
            let mut prefix = vec![1; 7];
            let stop: String = (0..length)
                .map(|_| {
                    let token = target(&prefix, sampling);
                    prefix.push(token);
                    char::from(b'a' + u8::try_from(token).unwrap())
                })
                .collect();
            let trace = runtime
                .block_on(run_with_boundaries(
                    sampling,
                    drafter,
                    2,
                    2,
                    17,
                    None,
                    Some(stop),
                ))
                .unwrap();
            assert_eq!(trace.tokens.len(), length);
            assert_eq!(trace.cut, Some(forja_host::speculative::VerifyCut::Stop));
            trace
        });
    }
}

#[tokio::test]
async fn lazy_constraint_and_tool_lanes_never_consult_the_draft_source() -> Result<(), TestError> {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = engine().await?.with_drafter(source(Arc::clone(&calls)));
    for tools in [false, true] {
        let mut request = crate::engine::continuous::prefill::tests::request();
        if tools {
            request.request_params.tools = Some(Vec::new());
        } else {
            request.constraint = Some(crate::engine::ConstraintPlan {
                grammar: r#"start: "a""#.to_owned(),
                trigger: "ab".to_owned(),
                trigger_tokens: vec![0, 1],
                forced: false,
                tools: Vec::new(),
                selected: None,
                parallel: true,
            });
        }
        let mut lane = BatchLane::new(
            request,
            0,
            0,
            0,
            SamplingParams::default(),
            engine.tokenizer()?,
            None,
        );
        assert!(!lane.is_constrained());
        assert!(engine.draft_candidate(&mut lane).is_none());
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    Ok(())
}

#[tokio::test]
async fn unexpected_verify_activation_rolls_back_without_publication() -> Result<(), TestError> {
    let mut engine = engine().await?.with_drafter(source(Arc::default()));
    let mut prepared = prepared_lane(&mut engine, 7, 0).await?;
    let candidate = engine.draft_candidate(&mut prepared).unwrap();
    assert!(engine.runner.try_reserve_draft_lane(0)?);
    let guidance = engine.guidance.compile(r#"start: "d""#.to_owned()).await?;
    let allow = guidance.mask().await?.unwrap();
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.tokens = vec![1; 7];
    request.prompt_tokens = 7;
    request.params.max_tokens = 7;
    let mut lane = BatchLane::new(
        request,
        0,
        0,
        prepared.pending_token,
        SamplingParams::default(),
        engine.tokenizer()?,
        Some(crate::engine::continuous::lane::BatchConstraint {
            guidance,
            allow,
            parser: None,
        }),
    );
    assert!(lane.is_constrained());
    let before = engine.runner.checkpoint_state_lane(0)?.unwrap().position();
    let fingerprint = engine.runner.state_lane_fingerprint(0)?;
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let error = engine
        .run_draft_round(&mut lane, candidate)
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "verify round cannot activate a tool constraint"
    );
    assert_eq!(
        engine.runner.checkpoint_state_lane(0)?.unwrap().position(),
        before
    );
    assert_eq!(engine.runner.state_lane_fingerprint(0)?, fingerprint);
    assert_eq!(
        engine
            .runner
            .checkpoint_paged_lane(0)?
            .unwrap()
            .logical_len(),
        before.tokens
    );
    assert_eq!(*barrier.verified.lock().unwrap(), []);
    assert_eq!(lane.pending_token, prepared.pending_token);
    assert_eq!(u64::from(lane.position), before.tokens);
    assert!(engine.runner.try_release_draft_lane(0)?);
    Ok(())
}
