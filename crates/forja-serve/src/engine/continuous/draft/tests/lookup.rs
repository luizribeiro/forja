use super::*;
use forja_draft_testing::{ExactnessChecks, Trace};

fn profile(k: u32, last: u32) -> forja_config::Profile {
    let mut value =
        serde_json::to_value(super::super::super::prefill::tests::canonical_profile().unwrap())
            .unwrap();
    let final_position =
        u32::try_from(value["workload"]["max-context"].as_u64().unwrap()).unwrap() - 1;
    let last = last.min(final_position);
    let mut arms = vec![serde_json::json!({"lo": 0, "hi": last, "k": k})];
    if last < final_position {
        arms.push(serde_json::json!({"lo": last + 1, "hi": final_position, "k": 0}));
    }
    value["prompt-lookup"] = serde_json::json!({
        "min-match": 8, "max-suffix": 32, "max-drafts": 2,
        "k-rule": arms
    });
    serde_json::from_value(value).unwrap()
}

#[derive(Clone, Copy)]
enum LookupCase {
    Periodic,
    SecondDraftStop,
    ContextCap,
}

#[test]
fn lookup_boundaries_and_mixed_rounds_match_plain_decode() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for case in [
        LookupCase::Periodic,
        LookupCase::SecondDraftStop,
        LookupCase::ContextCap,
    ] {
        assert_draft_exact(ExactnessChecks::source(true), |sampling, enabled| {
            runtime.block_on(async {
                let mut engine = super::super::super::prefill::tests::engine_with_component(
                    128,
                    16,
                    test_guests::verify_smoke(),
                )
                .await
                .unwrap();
                let mut prompt = matched_prompt(sampling);
                let mut stop_token = None;
                let (last, max_tokens) = match case {
                    LookupCase::Periodic => {
                        let mut period: Vec<_> = (0..22).map(|i| i % 4).collect();
                        period[12] = 1;
                        prompt = period.repeat(3);
                        (u32::MAX, 49)
                    }
                    LookupCase::SecondDraftStop => {
                        let mut history = prompt.clone();
                        for _ in 0..3 {
                            let token = target(&history, sampling);
                            history.push(token);
                        }
                        stop_token = history.last().copied();
                        assert!(
                            !history[prompt.len()..history.len() - 1]
                                .contains(&stop_token.unwrap())
                        );
                        (u32::MAX, 7)
                    }
                    LookupCase::ContextCap => {
                        prompt = copied_prompt(sampling, 8);
                        (u32::try_from(prompt.len()).unwrap() + 3, 13)
                    }
                };
                let probe_prefix = prompt.clone();
                let policy = profile(2, last);
                if enabled.is_some() {
                    engine = engine.with_profile_lookup(&policy);
                }
                let trace = Box::pin(direct::run_engine(
                    engine,
                    sampling,
                    direct::Case {
                        prompt,
                        max_tokens,
                        stop_token,
                        stop_text: None,
                    },
                ))
                .await
                .unwrap();
                if enabled.is_some() {
                    match case {
                        LookupCase::Periodic => {
                            assert!(trace.shapes.len() > 1);
                            if sampling.temperature == 0.0 {
                                for accepted in 0..=2 {
                                    assert!(
                                        trace
                                            .shapes
                                            .iter()
                                            .zip(&trace.accepted)
                                            .any(|row| row == (&3, &accepted))
                                    );
                                }
                            }
                        }
                        LookupCase::SecondDraftStop => {
                            assert_eq!(trace.shapes, [3]);
                            assert_eq!(trace.accepted, [1]);
                            assert_eq!(trace.cut, Some(VerifyCut::Eos));
                            assert_eq!(trace.tokens.last(), stop_token.as_ref());
                        }
                        LookupCase::ContextCap => {
                            assert_context_cap(&trace, &policy, last, probe_prefix, sampling);
                        }
                    }
                }
                trace
            })
        });
    }
}

#[test]
fn cancelling_an_outstanding_lookup_preserves_the_plain_boundary() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    assert_draft_exact(ExactnessChecks::source(false), |sampling, enabled| {
        runtime.block_on(async {
            let mut engine = engine().await.unwrap();
            if enabled.is_some() {
                engine = engine.with_profile_lookup(&profile(2, 31));
            }
            let mut request = crate::engine::continuous::prefill::tests::request();
            request.tokens = matched_prompt(sampling);
            request.prompt_tokens = u32::try_from(request.tokens.len()).unwrap();
            request.params.max_tokens = 7;
            request.control = crate::GenerationControl::new(Duration::from_secs(30));
            request.params.temperature = sampling.temperature;
            request.params.top_k = sampling.top_k;
            request.params.top_p = sampling.top_p;
            request.params.seed = Some(sampling.seed);
            let (events, mut received) = mpsc::channel(16);
            let (reply, _receiver) = oneshot::channel();
            request.sink = crate::engine::continuous::CompletionSink::Stream { events, reply };
            let mut lane = prepare_request(&mut engine, request, 0).await.unwrap();
            let before = engine
                .runner
                .checkpoint_state_lane(0)
                .unwrap()
                .unwrap()
                .position();
            let fingerprint = engine.runner.state_lane_fingerprint(0).unwrap();
            let pending = lane.pending_token;
            let barrier = Arc::new(BatchTestBarrier::new());
            engine.set_batch_test_barrier(Arc::clone(&barrier));
            let control = lane.control.clone();
            if enabled.is_some() {
                let candidate = engine.draft_candidate(&mut lane).unwrap();
                assert!(engine.runner.try_reserve_draft_lane(0).unwrap());
                let interrupt = async {
                    tokio::time::timeout(Duration::from_secs(10), barrier.reached.notified())
                        .await
                        .unwrap();
                    control.cancel();
                    barrier.resume.notify_one();
                };
                let result =
                    tokio::join!(engine.run_draft_round(&mut lane, candidate), interrupt).0;
                assert!(result.is_err());
                assert!(engine.runner.try_release_draft_lane(0).unwrap());
            } else {
                control.cancel();
            }
            assert_eq!(
                engine
                    .runner
                    .checkpoint_state_lane(0)
                    .unwrap()
                    .unwrap()
                    .position(),
                before
            );
            assert_eq!(
                engine.runner.state_lane_fingerprint(0).unwrap(),
                fingerprint
            );
            assert_eq!(lane.pending_token, pending);
            assert_eq!(*barrier.verified.lock().unwrap(), []);
            let (completion, output) =
                direct::observe(&engine, &mut lane, FinishReason::Length, &mut received)
                    .await
                    .unwrap();
            assert_eq!(output.kv_logical_len, before.tokens);
            Trace {
                output: Some(output),
                tokens: completion.token_ids,
                ..Trace::default()
            }
        })
    });
}

fn matched_prompt(sampling: SamplingParams) -> Vec<u32> {
    copied_prompt(sampling, 2)
}

fn copied_prompt(sampling: SamplingParams, continuation: usize) -> Vec<u32> {
    let suffix = [0, 0, 0, 1, 2, 3, 1];
    let prompt_len = suffix.len() * 2 + 1 + continuation;
    let mut history = vec![1; prompt_len];
    for _ in 0..=continuation {
        let token = target(&history, sampling);
        history.push(token);
    }
    suffix
        .into_iter()
        .chain(history[prompt_len..].iter().copied())
        .chain(suffix)
        .collect()
}

fn assert_context_cap(
    trace: &Trace,
    policy: &forja_config::Profile,
    last: u32,
    mut prefix: Vec<u32>,
    sampling: SamplingParams,
) {
    use crate::{DraftSource, PromptLookup};
    assert_eq!(policy.prompt_lookup().unwrap().cap(last), 2);
    assert_eq!(policy.prompt_lookup().unwrap().cap(last + 1), 0);
    assert_eq!(trace.shapes, [3, 3]);
    assert_eq!(trace.accepted, [2, 2]);
    assert!(trace.plain > 0);
    let outputs = 1 + trace
        .accepted
        .iter()
        .map(|accepted| accepted + 1)
        .sum::<u32>();
    prefix.extend_from_slice(&trace.tokens[..usize::try_from(outputs).unwrap()]);
    let position = u32::try_from(prefix.len() - 1).unwrap();
    assert_eq!(policy.prompt_lookup().unwrap().cap(position), 0);
    let mut uncapped = PromptLookup::new(profile(2, u32::MAX).prompt_lookup().unwrap().clone());
    assert_eq!(
        uncapped.propose(&prefix, position, sampling).tokens.len(),
        2
    );
}

#[test]
fn real_lookup_rounds_match_plain_streams_and_serving_observables() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for k in [1, 2] {
        assert_draft_exact(true, |sampling, enabled| {
            runtime.block_on(async {
                let mut engine = engine().await.unwrap();
                if enabled.is_some() {
                    engine = engine.with_profile_lookup(&profile(k, 17));
                }
                let trace = Box::pin(direct::run_engine(
                    engine,
                    sampling,
                    direct::Case {
                        prompt: matched_prompt(sampling),
                        max_tokens: 7,
                        stop_token: None,
                        stop_text: None,
                    },
                ))
                .await
                .unwrap();
                if enabled.is_some() {
                    assert_eq!(trace.shapes, [k + 1]);
                    assert_eq!(trace.accepted, [k]);
                    assert!(trace.plain > 0);
                }
                trace
            })
        });
    }
}

#[tokio::test]
async fn a_lookup_miss_preserves_the_plain_pipeline() -> Result<(), TestError> {
    let mut engine = engine().await?.with_profile_lookup(&profile(2, 31));
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
    assert_eq!(*barrier.verified.lock().unwrap(), []);
    assert!(barrier.pipelined_batches.load(Ordering::Acquire) > 0);
    Ok(())
}

#[tokio::test]
async fn lookup_indexes_belong_to_the_request_even_at_equal_positions() -> Result<(), TestError> {
    let mut engine = engine().await?.with_profile_lookup(&profile(2, 31));
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.tokens = matched_prompt(SamplingParams::default());
    request.prompt_tokens = u32::try_from(request.tokens.len())?;
    request.params.max_tokens = 7;
    let mut matched = prepare_request(&mut engine, request, 0).await?;
    let position = matched.position;
    assert!(engine.draft_candidate(&mut matched).is_some());
    engine
        .finish_batch_lane(matched, FinishReason::Length)
        .await;
    let mut unmatched = prepared_lane(&mut engine, 17, 0).await?;
    assert_eq!(position, unmatched.position);
    assert!(engine.draft_candidate(&mut unmatched).is_none());
    Ok(())
}

#[tokio::test]
async fn zero_acceptance_exits_even_when_the_new_suffix_has_a_continuation() -> Result<(), TestError>
{
    use crate::{DraftSource, PromptLookup};
    let profile = profile(2, 31);
    let mut engine = engine().await?.with_profile_lookup(&profile);
    let pattern = [0, 0, 0, 1, 2, 3, 1, 2];
    let mut request = crate::engine::continuous::prefill::tests::request();
    request.tokens = pattern
        .into_iter()
        .chain([1])
        .chain(pattern[1..].iter().copied())
        .chain([3])
        .chain(pattern[..7].iter().copied())
        .collect();
    request.prompt_tokens = u32::try_from(request.tokens.len())?;
    request.params.max_tokens = 7;
    let lane = prepare_request(&mut engine, request, 0).await?;
    let barrier = Arc::new(BatchTestBarrier::passive());
    engine.set_batch_test_barrier(Arc::clone(&barrier));
    let mut active = vec![lane];
    assert!(engine.advance_drafting(&mut active, true).await);
    assert_eq!(*barrier.verified.lock().unwrap(), [(3, 0)]);
    assert_eq!(active[0].drafting.rounds, 1);
    assert_eq!(active[0].drafting.acceptance_by_depth[&0], 1);
    assert_eq!(active[0].drafting.k_eff[&2], 1);
    assert_eq!(active[0].drafting.episodes_entered, 1);
    let mut fresh = PromptLookup::new(profile.prompt_lookup().unwrap().clone());
    let proposal = fresh.propose(
        active[0].ledger().tokens(),
        active[0].position,
        active[0].sampling,
    );
    assert_eq!(proposal.tokens, [0, 0]);
    assert!(!engine.advance_drafting(&mut active, true).await);
    assert_eq!(active[0].draft_mode, DraftMode::Plain);
    assert_eq!(active[0].drafting.episodes_exited["zero-acceptance"], 1);
    engine.runner.prepare_paged_lane(1, None, 1)?;
    assert!(engine.runner.try_reserve_draft_lane(1)?);
    assert!(engine.runner.try_release_draft_lane(1)?);
    assert!(
        direct::plain_step(&mut engine, &mut active[0])
            .await?
            .is_none()
    );
    let expected = fresh.propose(
        active[0].ledger().tokens(),
        active[0].position,
        active[0].sampling,
    );
    let candidate = engine.draft_candidate(&mut active[0]).unwrap();
    assert_eq!(candidate.fed.tokens(), expected.tokens);
    assert_eq!(candidate.fed.shape().tokens, 3);
    Ok(())
}
