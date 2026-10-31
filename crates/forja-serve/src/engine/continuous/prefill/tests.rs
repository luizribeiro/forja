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
    let weights = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../forja-host/tests/fixtures/qwen38/model.safetensors");
    let mut runner = EngineRunner::new(
        test_guests::qwen38_tiny(),
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
                max_context: Some(128),
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
        128,
        64,
        64,
        crate::PrefixIdentity::new(["engine", "profile", "weights", "tokenizer", "chat"])?,
    )?)
}

fn request() -> PreparedRequest {
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
