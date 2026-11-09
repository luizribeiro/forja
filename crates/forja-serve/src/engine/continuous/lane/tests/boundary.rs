use super::*;
use crate::engine::ConstraintPlan;
use forja_draft_testing::{Target, TestDrafter, Trace, assert_draft_exact};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape,
    speculative::{VerifyAcceptance, VerifyBudget, select_verify_shape},
};

async fn push(lane: &mut BatchLane, token: u32) -> Result<Option<VerifyCut>, GenerationError> {
    let piece = lane.decoder.step(token).await?;
    lane.boundary
        .process(
            token,
            piece,
            &mut lane.text,
            lane.sink.events(),
            &lane.control,
        )
        .await
}

async fn activation_trace(
    sampling: SamplingParams,
    mut drafter: Option<TestDrafter>,
    end: usize,
) -> Trace {
    let (mut live, mut received) = streaming_lane(&[]);
    let target = Target::new(2, sampling);
    let mut prefix = vec![1];
    let mut trigger = Vec::new();
    for _ in 0..end {
        let token = target.sample(&prefix);
        trigger.push(token);
        prefix.push(token);
    }
    let plan = ConstraintPlan {
        grammar: r#"start: "a""#.to_owned(),
        trigger: trigger
            .iter()
            .map(|id| char::from(b'a' + u8::try_from(*id).unwrap()))
            .collect(),
        trigger_tokens: trigger,
        forced: false,
        tools: Vec::new(),
        selected: None,
        parallel: false,
    };
    live.boundary = BoundaryState::new(&[], Some(&plan));
    let mut history = vec![1];
    let mut trace = Trace::default();
    loop {
        let (tokens, cut) = if let Some(drafter) = &mut drafter {
            let drafts = drafter.propose(&history, 2, 2, |prefix| target.sample(prefix));
            let shapes = [BatchShape {
                phase: BatchPhase::Verify,
                selection: BatchSelection::Combined,
                bucket: 1,
                tokens: 3,
            }];
            let fed = select_verify_shape(
                &shapes,
                &drafts,
                VerifyBudget::new(u64::try_from(history.len() - 1).unwrap(), 2, 7, 32, 32).unwrap(),
            )
            .unwrap();
            let mut verified = history.clone();
            let mut samples = vec![target.sample(&verified)];
            for draft in fed.tokens() {
                verified.push(*draft);
                samples.push(target.sample(&verified));
            }
            let decision = VerifyAcceptance::new(fed, &samples, 2)
                .unwrap()
                .scan_async(&mut live, |live, token| Box::pin(push(live, token)))
                .await
                .unwrap();
            assert_eq!(decision.pending(), *decision.tokens().last().unwrap());
            trace.shapes.push(3);
            trace.accepted.push(decision.accepted());
            (decision.tokens().to_vec(), decision.cut())
        } else {
            trace.plain += 1;
            let token = target.sample(&history);
            let cut = push(&mut live, token).await.unwrap();
            (vec![token], cut)
        };
        history.extend_from_slice(&tokens);
        trace.tokens.extend(tokens);
        trace.cut = cut;
        if cut.is_some() {
            break;
        }
        assert!(trace.tokens.len() < end);
    }
    assert_eq!(trace.tokens.len(), end);
    assert_eq!(trace.cut, Some(VerifyCut::Activate));
    assert_eq!(live.text, "");
    assert!(received.try_recv().is_err());
    trace
}

#[test]
fn live_activation_keeps_the_trigger_pending_for_every_test_drafter() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for end in 1..=4 {
        assert_draft_exact(true, |sampling, drafter| {
            runtime.block_on(activation_trace(sampling, drafter, end))
        });
    }
}
