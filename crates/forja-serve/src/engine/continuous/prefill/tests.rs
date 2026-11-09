mod background;

use super::{BatchConstraint, PendingPrefill, PreparedRequest};
use crate::{CompletionParams, GenerationControl, ResolvedParams, ServingEngine};
use forja_host::{EngineLoadConfig, EngineRunner, KvCacheFormat, Limits};
use std::{
    path::Path,
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, decoders::byte_level::ByteLevel, models::bpe::BPE};

type TestError = Box<dyn std::error::Error + Send + Sync>;

async fn engine() -> Result<ServingEngine<forja_cpu::CpuBackend>, TestError> {
    engine_with_context(128, 64).await
}

pub(in crate::engine::continuous) async fn engine_with_context(
    context: u32,
    chunk: u32,
) -> Result<ServingEngine<forja_cpu::CpuBackend>, TestError> {
    engine_with_component(context, chunk, test_guests::qwen38_tiny()).await
}

pub(in crate::engine::continuous) async fn engine_with_component(
    context: u32,
    chunk: u32,
    component: &Path,
) -> Result<ServingEngine<forja_cpu::CpuBackend>, TestError> {
    let weights = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../forja-host/tests/fixtures/qwen38/model.safetensors");
    let mut runner = EngineRunner::new(
        component,
        forja_cpu::CpuBackend::new(),
        Limits::new(1 << 30, 8, 1 << 30, 20_000, 1 << 27),
        weights,
    )
    .await?;
    let info = runner.describe().await?;
    runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                max_context: Some(context),
                replay: true,
                paged_cache: true,
                kv_cache: KvCacheFormat::F16,
                ..Default::default()
            },
        )
        .await??;
    let vocab = [
        ("a".to_owned(), 0),
        ("b".to_owned(), 1),
        ("c".to_owned(), 2),
        ("d".to_owned(), 3),
    ];
    let model = BPE::builder().vocab_and_merges(vocab, Vec::new()).build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    Ok(ServingEngine::new(
        "hybrid".to_owned(),
        runner,
        tokenizer,
        crate::ChatFormat::load(
            test_guests::continuous_chat_format(),
            crate::ChatFormatLimits::default(),
        )?,
        forja_config::HostConfig::default(),
        info,
        context,
        chunk,
        chunk,
        crate::PrefixIdentity::new(["engine", "profile", "weights", "tokenizer", "chat"])?,
    )?)
}

pub(in crate::engine::continuous) fn request() -> PreparedRequest {
    let (reply, _receiver) = tokio::sync::oneshot::channel();
    PreparedRequest {
        prefix_reuse: false,
        tokens: vec![0; 65],
        prompt_tokens: 65,
        params: ResolvedParams {
            max_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: Some(0),
            stops: Vec::new(),
            stream: false,
        },
        request_params: CompletionParams::default(),
        constraint: None,
        control: GenerationControl::new(Duration::from_secs(2)),
        sink: super::super::CompletionSink::Complete(reply),
        started: Instant::now(),
    }
}

#[tokio::test]
async fn chunked_constrained_prefill_selects_only_the_final_physical_lane() -> Result<(), TestError>
{
    let mut engine = engine().await?;
    for slot in [0, 7] {
        engine.runner.prepare_paged_lane(slot, None, 0)?;
        engine.runner.prepare_paged_lane(slot, None, 67)?;
    }
    let peer = engine
        .runner
        .checkpoint_state_lane(0)?
        .map(|checkpoint| checkpoint.position());
    let guidance = engine.guidance.compile(r#"start: "b""#.to_owned()).await?;
    let allow = guidance.mask().await?.ok_or("missing grammar mask")?;
    assert_eq!(allow, [2, 0]);
    let mut prefill = PendingPrefill {
        request: request(),
        slot: 7,
        sampling: forja_host::SamplingParams::default(),
        tokenizer: engine.tokenizer()?,
        cached: 0,
        reserved: 67,
        consumed: 0,
        position: 0,
        selected: None,
        constraint: Some(BatchConstraint {
            guidance,
            allow,
            parser: None,
        }),
    };
    assert!(!engine.advance_batch_prefill(&mut prefill, 64).await?);
    assert_eq!(
        (prefill.position, prefill.consumed, prefill.selected),
        (64, 64, None)
    );
    assert_eq!(
        engine
            .runner
            .checkpoint_state_lane(7)?
            .ok_or("missing carry")?
            .position()
            .tokens,
        64
    );
    assert!(engine.advance_batch_prefill(&mut prefill, 64).await?);
    assert_eq!(
        (prefill.position, prefill.consumed, prefill.selected),
        (65, 65, Some(1))
    );
    let position = engine
        .runner
        .checkpoint_state_lane(7)?
        .ok_or("missing carry")?
        .position();
    assert_eq!(
        (
            position.tokens,
            position.sampler,
            position.grammar,
            position.parser
        ),
        (65, 64, 65, 65)
    );
    assert_eq!(
        engine
            .runner
            .checkpoint_state_lane(0)?
            .map(|checkpoint| checkpoint.position()),
        peer
    );
    Ok(())
}

pub(in crate::engine::continuous) fn canonical_profile() -> Result<forja_config::Profile, TestError>
{
    Ok(serde_json::from_value(serde_json::json!({
        "schema-version": 1, "family": "hybrid",
        "model": { "id": "test/hybrid", "revision": "0".repeat(40), "weights-sha256": format!("sha256:{}", "0".repeat(64)) },
        "target": { "component": "wasm32-wasip2", "backend": "metal", "device-family": "test" },
        "numerics": { "activations": "f32", "weights": "f32" },
        "workload": { "kind": "causal-lm", "batch": 1, "max-context": 128, "prefill-chunk": 64 },
        "serving": { "prompt-token-budget": 64, "paged-cache": true, "state-snapshot-bytes": 268_435_456_u64, "state-snapshot-interval": 4096 },
        "variants": { "fixed": { "recurrent": "fixed" } }
    }))?)
}

#[tokio::test]
async fn paired_cache_configuration_honors_operator_disable_and_restore() -> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    assert!(engine.canonical_cache.is_some());
    assert_eq!(engine.canonical_piece(0, 65, 65)?, 64);
    let mut config = engine.config.clone();
    config.serving.state_snapshot_bytes =
        forja_config::Choice::Fixed(forja_config::ByteSize::new(0));
    engine.update_policy(1, config.clone())?;
    assert_eq!(engine.canonical_piece(0, 65, 65)?, 65);
    config.serving.state_snapshot_bytes = forja_config::Choice::Auto;
    engine.update_policy(2, config.clone())?;
    assert_eq!(engine.canonical_piece(0, 65, 65)?, 64);
    config.serving.state_snapshot_bytes =
        forja_config::Choice::Fixed(forja_config::ByteSize::new(268_435_457));
    assert!(engine.update_policy(3, config).is_err());
    assert_eq!(engine.config_generation, 2);
    Ok(())
}

#[tokio::test]
async fn paired_serving_admission_resumes_the_longest_boundary_and_replays_tail()
-> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut initial = request();
    initial.prefix_reuse = true;
    let mut cold = engine
        .begin_batch_prefill(initial, 0)
        .await
        .map_err(|(_, error)| error)?;
    assert!(!engine.advance_batch_prefill(&mut cold, 64).await?);
    assert!(engine.advance_batch_prefill(&mut cold, 64).await?);
    let expected = engine.runner.state_lane_fingerprint(0)?.1;
    assert!(engine.release_failed_lane(0).is_none());
    let mut next = request();
    next.prefix_reuse = true;
    let mut hit = engine
        .begin_batch_prefill(next, 7)
        .await
        .map_err(|(_, error)| error)?;
    assert_eq!((hit.cached, hit.position, hit.consumed), (64, 64, 64));
    assert!(engine.advance_batch_prefill(&mut hit, 64).await?);
    assert_eq!(hit.selected, cold.selected);
    assert_eq!(engine.runner.state_lane_fingerprint(7)?.1, expected);
    assert_eq!((engine.metrics.hits, engine.metrics.reused_tokens), (1, 64));
    Ok(())
}

#[tokio::test]
async fn cancelled_publication_leaves_the_paired_cache_empty() -> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut pending = engine
        .begin_batch_prefill(request(), 0)
        .await
        .map_err(|(_, error)| error)?;
    assert!(!engine.advance_batch_prefill(&mut pending, 64).await?);
    pending.request.control.cancel();
    engine.publish_canonical_boundary(0, 64, 65, true, &pending.request.control)?;
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
    let control = GenerationControl::new(Duration::from_secs(2));
    engine.publish_canonical_boundary(0, 64, 65, true, &control)?;
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
    Ok(())
}

#[tokio::test]
async fn serial_prefill_uses_the_same_canonical_tail_as_continuous_serving() -> Result<(), TestError>
{
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let input = request();
    let cold = engine
        .prefill(
            input.tokens.clone(),
            65,
            input.params.clone(),
            None,
            &input.control,
        )
        .await?;
    let expected = engine.runner.read_queued_token(cold.first).await?;
    let expected_state = engine.runner.state_lane_fingerprint(0)?.1;
    let hit = engine
        .prefill(input.tokens, 65, input.params, None, &input.control)
        .await?;
    assert_eq!(hit.cached_tokens, 64);
    assert_eq!(engine.runner.read_queued_token(hit.first).await?, expected);
    assert_eq!(engine.runner.state_lane_fingerprint(0)?.1, expected_state);
    Ok(())
}

async fn complete_prompt(
    engine: &mut ServingEngine<forja_cpu::CpuBackend>,
    token: u32,
) -> Result<(Option<u32>, [u8; 32], u32), TestError> {
    let mut input = request();
    input.prefix_reuse = true;
    input.tokens.fill(token);
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
async fn snapshot_quota_refusals_preserve_successful_serving_and_cold_state()
-> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut config = engine.config.clone();
    config.serving.state_snapshot_bytes =
        forja_config::Choice::Fixed(forja_config::ByteSize::new(1));
    engine.update_policy(1, config)?;
    let cold = complete_prompt(&mut engine, 0).await?;
    let again = complete_prompt(&mut engine, 0).await?;
    assert_eq!(cold, again);
    assert_eq!(again.2, 0);
    assert_eq!(engine.metrics.snapshot_refusals, 2);
    Ok(())
}

#[tokio::test]
async fn serving_entry_pressure_evicts_the_unpinned_paired_boundary() -> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut config = engine.config.clone();
    config.serving.prefix_cache_entries = forja_config::Choice::Fixed(std::num::NonZeroU32::MIN);
    engine.update_policy(1, config)?;
    let original = complete_prompt(&mut engine, 0).await?;
    assert_eq!(complete_prompt(&mut engine, 1).await?.2, 0);
    let evicted = complete_prompt(&mut engine, 0).await?;
    assert_eq!(evicted, original);
    let hit = complete_prompt(&mut engine, 0).await?;
    assert_eq!(hit.2, 64);
    assert_eq!((hit.0, hit.1), (original.0, original.1));
    Ok(())
}

#[tokio::test]
async fn undeclared_snapshot_allowances_refuse_operator_growth_at_startup_and_reload()
-> Result<(), TestError> {
    let mut value = serde_json::to_value(canonical_profile()?)?;
    value["serving"]
        .as_object_mut()
        .unwrap()
        .remove("state-snapshot-bytes");
    value["serving"]
        .as_object_mut()
        .unwrap()
        .remove("state-snapshot-interval");
    let profile: forja_config::Profile = serde_json::from_value(value)?;
    let mut engine = engine().await?;
    let mut config = engine.config.clone();
    config.serving.state_snapshot_bytes =
        forja_config::Choice::Fixed(forja_config::ByteSize::new(1));
    assert!(engine.update_policy(1, config.clone()).is_err());
    assert_eq!(engine.config_generation, 0);
    engine.config = config;
    assert!(engine.with_profile_prefixes(&profile).is_err());
    Ok(())
}

#[tokio::test]
async fn retained_runtime_memory_includes_the_paired_prefix_index() -> Result<(), TestError> {
    let engine = engine().await?;
    let before = engine.memory_usage().accounted_bytes().unwrap();
    let engine = engine.with_profile_prefixes(&canonical_profile()?)?;
    assert!(engine.memory_usage().accounted_bytes().unwrap() > before);
    Ok(())
}

#[tokio::test]
async fn invalid_canonical_lane_admission_preserves_unpinned_cached_prefixes()
-> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    complete_prompt(&mut engine, 0).await?;
    let mut invalid = request();
    invalid.prefix_reuse = true;
    invalid.tokens.fill(1);
    assert!(engine.begin_batch_prefill(invalid, 99).await.is_err());
    assert_eq!(complete_prompt(&mut engine, 0).await?.2, 64);
    Ok(())
}

#[tokio::test]
async fn sparse_boundaries_consume_the_allocated_prompt_budget() -> Result<(), TestError> {
    let mut value = serde_json::to_value(canonical_profile()?)?;
    value["workload"]["max-context"] = serde_json::json!(512);
    value["workload"]["prefill-chunk"] = serde_json::json!(128);
    value["serving"]["prompt-token-budget"] = serde_json::json!(128);
    value["serving"]["state-snapshot-interval"] = serde_json::json!(128);
    let profile: forja_config::Profile = serde_json::from_value(value)?;
    let mut engine = engine_with_context(512, 128)
        .await?
        .with_profile_prefixes(&profile)?;
    complete_prompt(&mut engine, 0).await?;
    let mut input = request();
    input.prefix_reuse = true;
    input.tokens = vec![0; 257];
    input.prompt_tokens = 257;
    let mut pending = engine
        .begin_batch_prefill(input, 0)
        .await
        .map_err(|(_, error)| error)?;
    assert_eq!(pending.cached, 64);
    let id = crate::scheduler::RequestId(1);
    let mut policy = crate::scheduler::SchedulerPolicy::new(vec![1], 1, 1, 128, 128)?
        .with_prompt_chunk_multiple(std::num::NonZeroU32::new(64));
    policy.ready_prompt(id, pending.remaining());
    let mut rounds = Vec::new();
    loop {
        let batch = policy.prompt_batch(false);
        assert_eq!(batch.len(), 1);
        let (scheduled, budget) = batch[0];
        assert_eq!(scheduled, id);
        let before = pending.position;
        let finished = engine.advance_batch_prefill(&mut pending, budget).await?;
        assert_eq!(pending.position - before, budget);
        rounds.push(budget);
        if finished {
            break;
        }
    }
    assert_eq!(rounds, [128, 64, 1]);
    assert_eq!(pending.position, 257);
    Ok(())
}

#[tokio::test]
async fn paired_hit_admission_supports_immediate_constrained_selection_rewind()
-> Result<(), TestError> {
    use forja_host::{
        BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchSelect,
        EngineBatchSelectLane, EngineLaneCheckpoint, SamplingParams, TokenIds,
    };
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let mut initial = request();
    initial.prefix_reuse = true;
    let _cold = engine
        .begin_batch_prefill(initial, 0)
        .await
        .map_err(|(_, error)| error)?;
    let _parent = engine
        .runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: 64,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 0,
                tokens: Some(vec![0; 64]),
                start_pos: 0,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    let control = GenerationControl::new(Duration::from_secs(2));
    engine.publish_canonical_boundary(0, 64, 65, true, &control)?;
    let parent_bits = engine.runner.state_lane_fingerprint(0)?;
    let mut input = request();
    input.prefix_reuse = true;
    let hit = engine
        .begin_batch_prefill(input, 7)
        .await
        .map_err(|(_, error)| error)?;
    assert_eq!((hit.cached, hit.position), (64, 64));
    let before = engine.runner.state_lane_fingerprint(7)?;
    let mut checkpoint = EngineLaneCheckpoint::new(
        63,
        0,
        (),
        hit.reserved,
        engine.runner.checkpoint_paged_lane(7)?.unwrap(),
    );
    checkpoint.state_version = engine.runner.checkpoint_state_lane(7)?;
    let tail = engine
        .runner
        .batch_step(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: 64,
                selection: BatchSelection::Combined,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 7,
                tokens: Some(vec![0]),
                start_pos: 64,
                sampling: SamplingParams::default(),
            })],
        })
        .await??;
    let mut lanes = vec![None; 8];
    lanes[7] = Some(EngineBatchSelectLane {
        start_pos: 64,
        sampling: SamplingParams::default(),
        allow: TokenIds::new(vec![1]),
    });
    let selected = engine
        .runner
        .batch_select(
            &tail.logits,
            EngineBatchSelect {
                shape: BatchShape {
                    phase: BatchPhase::Decode,
                    bucket: 8,
                    tokens: 1,
                    selection: BatchSelection::Masked,
                },
                lanes,
            },
        )
        .await??;
    let selected = engine.runner.read_queued_batch_tokens(selected).await?;
    assert_eq!(super::super::read_batch_tokens(&selected)?[7], 1);
    engine.runner.rollback_lane(7, checkpoint)?;
    assert_eq!(engine.runner.state_lane_fingerprint(7)?, before);
    assert_eq!(
        engine.runner.paged_lane_prefix(7)?.unwrap().logical_len()?,
        64
    );
    assert_eq!(engine.runner.state_lane_fingerprint(0)?, parent_bits);
    Ok(())
}

#[tokio::test]
async fn short_prompts_bypass_canonical_admission_without_ledger_charges() -> Result<(), TestError>
{
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let before = engine.memory_usage().live_bytes;
    for tokens in [vec![], vec![0]] {
        assert!(engine.canonical_admission(&tokens, true)?.is_none());
        assert_eq!(engine.memory_usage().live_bytes, before);
    }
    let mut input = request();
    input.prefix_reuse = true;
    input.tokens = vec![0];
    input.prompt_tokens = 1;
    let mut prefill = engine
        .begin_batch_prefill(input, 0)
        .await
        .map_err(|(_, error)| error)?;
    assert_eq!((prefill.cached, prefill.position), (0, 0));
    assert!(engine.advance_batch_prefill(&mut prefill, 64).await?);
    assert!(prefill.selected.is_some());
    assert_eq!(engine.metrics.snapshot_refusals, 0);
    Ok(())
}

#[tokio::test]
async fn foreground_admission_preserves_the_queued_reply_ledger() -> Result<(), TestError> {
    let mut engine = engine()
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let control = GenerationControl::new(Duration::from_secs(2));
    let ledger = crate::prefix::PrefixLedger::new(vec![0; 65]);
    engine.queue_canonical_recompute(&ledger, 65, &[0; 63], &control, true);
    assert!(engine.metrics.recompute_pending);
    let input = request();
    let _prefill = engine
        .begin_batch_prefill(input, 0)
        .await
        .map_err(|(_, error)| error)?;
    assert!(engine.metrics.recompute_pending);
    assert_eq!(engine.metrics.recompute_preempted, 0);
    assert_eq!(engine.metrics.recompute_blocks, 0);
    Ok(())
}

#[tokio::test]
async fn cancelled_reply_recompute_releases_the_lane_without_partial_publication()
-> Result<(), TestError> {
    let mut engine = engine_with_context(256, 64)
        .await?
        .with_profile_prefixes(&canonical_profile()?)?;
    let control = GenerationControl::new(Duration::from_secs(2));
    let ledger = crate::prefix::PrefixLedger::new(vec![0; 65]);
    engine.queue_canonical_recompute(&ledger, 65, &[0; 127], &control, true);
    assert!(engine.advance_canonical_recompute().await);
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
        0
    );
    control.cancel();
    assert!(engine.advance_canonical_recompute().await);
    assert!(!engine.metrics.recompute_pending);
    assert_eq!(engine.metrics.recompute_cancelled, 1);
    assert_eq!(engine.metrics.recompute_completed, 0);
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
    engine.runner.prepare_paged_lane(0, None, 0)?;
    Ok(())
}
