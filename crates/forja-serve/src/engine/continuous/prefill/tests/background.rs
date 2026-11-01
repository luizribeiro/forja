use super::*;
use crate::prefix::PrefixLedger;
use forja_cpu::CpuBackend;

type Engine = ServingEngine<CpuBackend>;

async fn configured() -> Result<Engine, TestError> {
    Ok(engine_with_context(256, 64)
        .await?
        .with_profile_prefixes(&canonical_profile()?)?)
}

async fn finish_recompute(engine: &mut Engine) {
    for _ in 0..5 {
        if !engine.advance_canonical_recompute().await {
            break;
        }
    }
    assert!(!engine.metrics.recompute_pending);
}

async fn evaluate(
    engine: &mut Engine,
    tokens: Vec<u32>,
    reuse: bool,
) -> Result<(Option<u32>, [u8; 32], u32), TestError> {
    let mut input = request();
    input.prefix_reuse = reuse;
    input.prompt_tokens = u32::try_from(tokens.len())?;
    input.tokens = tokens;
    let mut pending = engine
        .begin_batch_prefill(input, 0)
        .await
        .map_err(|(_, error)| error)?;
    while !engine.advance_batch_prefill(&mut pending, 64).await? {}
    let result = (
        pending.selected,
        engine.runner.state_lane_fingerprint(0)?.1,
        pending.cached,
    );
    assert!(engine.release_failed_lane(0).is_none());
    Ok(result)
}

#[tokio::test]
async fn next_turn_after_reply_recompute_matches_cold_bits_including_short_replies()
-> Result<(), TestError> {
    for (prompt_tokens, reply_tokens) in [(65, 63), (63, 1), (65, 1), (1, 7)] {
        let prompt = (0..prompt_tokens).map(|i| i % 4).collect::<Vec<_>>();
        let reply = (0..reply_tokens).map(|i| (i + 1) % 4).collect::<Vec<_>>();
        let mut conversation = prompt.clone();
        conversation.extend(&reply);
        let boundary = u32::try_from(conversation.len() / 64 * 64)?;
        let mut next = conversation.clone();
        next.push(2);
        let mut candidate = configured().await?;
        let control = GenerationControl::new(Duration::from_secs(2));
        candidate.queue_canonical_recompute(
            &PrefixLedger::new(prompt),
            prompt_tokens,
            &reply,
            &control,
            true,
        );
        finish_recompute(&mut candidate).await;
        assert_eq!(
            candidate.metrics.recompute_completed,
            u64::from(boundary > 0)
        );
        let cached = evaluate(&mut candidate, next.clone(), true).await?;
        let cold = evaluate(&mut configured().await?, next, false).await?;
        assert_eq!((cached.0, cached.1), (cold.0, cold.1));
        assert_eq!(cached.2, boundary);
        assert_eq!(cold.2, 0);
    }
    Ok(())
}

#[tokio::test]
async fn reply_recompute_respects_disabled_and_insufficient_snapshot_allowances()
-> Result<(), TestError> {
    for allowance in [0, 1, 1024] {
        let mut engine = configured().await?;
        let mut config = engine.config.clone();
        config.serving.state_snapshot_bytes =
            forja_config::Choice::Fixed(forja_config::ByteSize::new(allowance));
        engine.update_policy(1, config)?;
        let control = GenerationControl::new(Duration::from_secs(2));
        engine.queue_canonical_recompute(
            &PrefixLedger::new(vec![0; 65]),
            65,
            &[1; 63],
            &control,
            true,
        );
        finish_recompute(&mut engine).await;
        assert_eq!(
            engine
                .canonical_cache
                .as_ref()
                .unwrap()
                .store
                .as_ref()
                .unwrap()
                .usage()?
                .entries,
            0
        );
        assert_eq!(engine.metrics.snapshot_refusals, u64::from(allowance > 0));
        assert_eq!(engine.metrics.recompute_completed, 0);
        assert_eq!(engine.metrics.recompute_blocks, 0);
        assert_eq!(engine.metrics.recompute_queued, 0);
        assert_eq!(evaluate(&mut engine, vec![0; 65], true).await?.2, 0);
    }
    Ok(())
}

#[tokio::test]
async fn arriving_request_preempts_reply_recompute_after_at_most_one_block() -> Result<(), TestError>
{
    assert_preemption(false).await
}

#[tokio::test]
async fn quiescence_preemption_publishes_idle_metrics_before_reply() -> Result<(), TestError> {
    assert_preemption(true).await
}

async fn assert_preemption(quiesce_only: bool) -> Result<(), TestError> {
    use crate::{
        ServingMetrics,
        http::{AdmissionPolicy, ServiceCommand},
    };
    use std::sync::{Arc, Mutex};
    use tokio::sync::{mpsc, oneshot};
    let mut engine = configured().await?;
    let policy = AdmissionPolicy {
        config: Arc::new(engine.config.clone()),
    };
    let barrier = Arc::new(crate::engine::BatchTestBarrier::new());
    engine.set_batch_test_barrier(barrier.clone());
    let control = GenerationControl::new(Duration::from_secs(2));
    engine.queue_canonical_recompute(
        &PrefixLedger::new(vec![0; 65]),
        65,
        &[0; 127],
        &control,
        true,
    );
    let metrics = Arc::new(Mutex::new(ServingMetrics::default()));
    let (sender, receiver) = mpsc::channel(8);
    let worker = tokio::spawn(engine.run_continuous(receiver, metrics.clone()));
    tokio::time::timeout(Duration::from_secs(5), barrier.reached()).await?;
    if quiesce_only {
        let (reply, quiesced) = oneshot::channel();
        sender.send(ServiceCommand::Quiesce { reply }).await?;
        barrier.release();
        quiesced.await??;
    } else {
        let (reply, completion) = oneshot::channel();
        sender.send(ServiceCommand::Text {
            policy,
            request: serde_json::from_value(serde_json::json!({"model":"hybrid", "prompt":"a", "max_tokens":2, "temperature":0.0}))?,
            control: GenerationControl::new(Duration::from_secs(2)),
            reply,
            queued: Instant::now(),
        }).await?;
        barrier.release();
        let completion = tokio::time::timeout(Duration::from_secs(5), completion).await???;
        assert_eq!(completion.usage.cached_tokens, 0);
        assert_eq!(completion.token_ids.len(), 2);
        let (reply, quiesced) = oneshot::channel();
        sender.send(ServiceCommand::Quiesce { reply }).await?;
        quiesced.await??;
    }
    let observed = metrics.lock().unwrap().clone();
    assert_eq!(observed.recompute_blocks, 1);
    assert_eq!(observed.recompute_preempted, 1);
    assert_eq!(observed.recompute_completed, 0);
    assert!(!observed.recompute_pending);
    drop(sender);
    worker.await?;
    Ok(())
}

#[tokio::test]
async fn serial_generation_keeps_the_canonical_idle_worker() -> Result<(), TestError> {
    use crate::{
        ServingMetrics,
        http::{AdmissionPolicy, ChatService, ServiceCommand},
    };
    use std::sync::{Arc, Mutex};
    use tokio::sync::{mpsc, oneshot};
    let mut engine = configured().await?;
    engine.scheduler.decode_buckets.clear();
    let policy = AdmissionPolicy {
        config: Arc::new(engine.config.clone()),
    };
    let control = GenerationControl::new(Duration::from_secs(2));
    engine.queue_canonical_recompute(
        &PrefixLedger::new(vec![0; 65]),
        65,
        &[1; 63],
        &control,
        true,
    );
    let metrics = Arc::new(Mutex::new(ServingMetrics::default()));
    let (sender, receiver) = mpsc::channel(8);
    let worker = tokio::spawn(engine.run(receiver, metrics.clone()));
    tokio::time::timeout(Duration::from_secs(5), async {
        while metrics.lock().unwrap().recompute_completed == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    let (reply, completion) = oneshot::channel();
    sender.send(ServiceCommand::Text {
        policy,
        request: serde_json::from_value(serde_json::json!({"model":"hybrid", "prompt":"a", "max_tokens":2, "temperature":0.0}))?,
        control: GenerationControl::new(Duration::from_secs(2)), reply, queued: Instant::now(),
    }).await?;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), completion)
            .await???
            .token_ids
            .len(),
        2
    );
    drop(sender);
    worker.await?;
    let observed = metrics.lock().unwrap().clone();
    assert_eq!(observed.recompute_blocks, 2);
    assert_eq!(observed.recompute_completed, 1);
    assert!(!observed.recompute_pending);
    Ok(())
}

#[tokio::test]
async fn pending_reply_fifo_supersedes_only_its_oldest_entry_at_the_lane_bound()
-> Result<(), TestError> {
    let mut engine = configured().await?;
    let mut config = engine.config.clone();
    config.serving.max_active_sequences =
        forja_config::Choice::Fixed(std::num::NonZeroU32::new(2).unwrap());
    engine.update_policy(1, config)?;
    let control = GenerationControl::new(Duration::from_secs(2));
    for token in 0..3 {
        engine.queue_canonical_recompute(
            &PrefixLedger::new(vec![token; 63]),
            63,
            &[3],
            &control,
            true,
        );
    }
    assert_eq!(engine.metrics.recompute_superseded, 1);
    finish_recompute(&mut engine).await;
    assert_eq!(engine.metrics.recompute_completed, 2);
    for token in 0..3 {
        let mut next = vec![token; 63];
        next.extend([3, 2]);
        assert_eq!(
            engine
                .canonical_admission(&next, true)?
                .unwrap()
                .1
                .is_some(),
            token > 0
        );
    }
    Ok(())
}

#[tokio::test]
async fn published_reply_boundary_short_circuits_without_another_canonical_block()
-> Result<(), TestError> {
    let mut engine = configured().await?;
    let ledger = PrefixLedger::new(vec![0; 63]);
    let control = GenerationControl::new(Duration::from_secs(2));
    for completed in 1..=2 {
        engine.queue_canonical_recompute(&ledger, 63, &[1, 2], &control, true);
        finish_recompute(&mut engine).await;
        assert_eq!(engine.metrics.recompute_completed, completed);
        assert_eq!(engine.metrics.recompute_blocks, 1);
        assert_eq!(
            engine
                .canonical_cache
                .as_ref()
                .unwrap()
                .store
                .as_ref()
                .unwrap()
                .usage()?
                .entries,
            1
        );
    }
    Ok(())
}
