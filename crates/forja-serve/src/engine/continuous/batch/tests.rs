use super::*;
use crate::engine::continuous::prefill::tests::{canonical_profile, engine_with_context, request};

type TestError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::test]
async fn lanes_finishing_in_one_decode_batch_both_retain_reply_snapshots() -> Result<(), TestError>
{
    let mut engine = engine_with_context(256, 64)
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut active = Vec::new();
    let mut replies = Vec::new();
    for slot in 0..2 {
        let mut input = request();
        input.prefix_reuse = true;
        input.tokens = vec![slot; 63];
        input.prompt_tokens = 63;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        input.sink = super::super::CompletionSink::Complete(sender);
        replies.push(receiver);
        let mut prefill = engine
            .begin_batch_prefill(input, slot)
            .await
            .map_err(|(_, error)| error)?;
        assert!(engine.advance_batch_prefill(&mut prefill, 64).await?);
        let mut lane = ServingEngine::<forja_cpu::CpuBackend>::finish_batch_prefill(prefill)
            .map_err(|(_, error)| error)?;
        assert!(
            lane.accept(lane.pending_token, &[], &mut engine.metrics)
                .await?
                .is_none()
        );
        active.push(lane);
    }
    let policy = SchedulerPolicy::new(vec![1, 2], 2, 2, 128, 128)?;
    let batch = engine
        .submit_batch(&policy, &active.iter().collect::<Vec<_>>(), &[])
        .await?
        .unwrap();
    let tokens = engine.read_completed_batch(batch.output).await?;
    engine
        .apply_batch_tokens(
            batch.lanes,
            batch.deferred_retirements,
            &mut active,
            &mut None,
            &read_batch_tokens(&tokens)?,
        )
        .await;
    assert!(active.is_empty());
    assert_eq!(engine.metrics.recompute_queued, 2);
    assert_eq!(engine.metrics.recompute_superseded, 0);
    for _ in 0..6 {
        if !engine.advance_canonical_recompute().await {
            break;
        }
    }
    assert_eq!(engine.metrics.recompute_completed, 2);
    assert!(!engine.metrics.recompute_pending);
    for (slot, reply) in replies.into_iter().enumerate() {
        let completion = reply.await??;
        let mut next = vec![u32::try_from(slot)?; 63];
        next.extend(completion.token_ids);
        next.push(2);
        let (_, hit) = engine.canonical_admission(&next, true)?.unwrap();
        assert_eq!(hit.unwrap().tokens().len(), 64);
    }
    Ok(())
}
