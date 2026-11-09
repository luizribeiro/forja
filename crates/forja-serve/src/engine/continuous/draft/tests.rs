use super::super::draft_support::*;
use super::*;
use forja_host::SamplingParams;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
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
