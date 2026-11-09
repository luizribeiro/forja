mod direct;
use super::super::draft_support::*;
use super::*;
use crate::engine::BatchTestBarrier;
use forja_draft_testing::{TestDrafter, assert_draft_exact};
use forja_host::SamplingParams;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
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
