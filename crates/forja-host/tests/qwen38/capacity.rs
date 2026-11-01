use super::{Error, fixture};
use forja_host::bindings::l9o::gpu::compute::Error as ComputeError;
use forja_host::{
    BatchPhase, EngineBeginDecode, EngineDecode, EngineLoadConfig, EngineRunner, KvCacheFormat,
    Limits, SamplingParams,
};

const CONTEXT: u32 = 262_144;
const TINY_LIVE_QUOTA_BYTES: u64 = 48 << 20;

async fn load(
    storage: KvCacheFormat,
) -> Result<EngineRunner<forja_cpu::CpuBackend>, Box<dyn Error>> {
    let mut runner = EngineRunner::new(
        test_guests::qwen38_tiny(),
        forja_cpu::CpuBackend::new(),
        Limits::new(TINY_LIVE_QUOTA_BYTES, 8, 1 << 30, 20_000, 1 << 27),
        fixture("model.safetensors"),
    )
    .await?;
    runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                max_context: Some(CONTEXT),
                replay: false,
                paged_cache: true,
                kv_cache: storage,
                ..Default::default()
            },
        )
        .await??;
    Ok(runner)
}

async fn piece(
    runner: &mut EngineRunner<forja_cpu::CpuBackend>,
    start_pos: u32,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let output = runner
        .decode(EngineDecode {
            phase: BatchPhase::Prompt,
            tokens: Some(vec![1; 64]),
            start_pos,
            sampling: SamplingParams::default(),
        })
        .await??;
    Ok(runner.read(&output.logits).await?)
}

pub(super) fn boundary<B: forja_core::Backend + Send + Sync + 'static>(
    runner: &EngineRunner<B>,
) -> Result<(forja_host::LanePosition, u64), Box<dyn Error>> {
    Ok((
        runner
            .checkpoint_state_lane(0)?
            .ok_or("missing state")?
            .position(),
        runner
            .checkpoint_paged_lane(0)?
            .ok_or("missing pages")?
            .logical_len(),
    ))
}

#[tokio::test]
async fn long_context_capacity_refusals_preserve_admitted_state() -> Result<(), Box<dyn Error>> {
    let mut runner = load(KvCacheFormat::F16).await?;
    runner.prepare_paged_lane(0, None, 0)?;
    runner.prepare_paged_lane(0, None, u64::from(CONTEXT))?;
    let ledger = runner
        .memory_usage()
        .accounted_bytes()
        .ok_or("ledger overflow")?;
    let paged = runner
        .paged_memory_usage()?
        .ok_or("missing paged storage")?;
    assert!(paged.total().ok_or("paged ledger overflow")? < ledger);
    assert!(ledger <= TINY_LIVE_QUOTA_BYTES);
    assert_eq!(piece(&mut runner, 0).await?.len(), 33 * 4);
    let initial = boundary(&runner)?;
    for (lane, tokens) in [(0, u64::from(CONTEXT) + 1), (1, u64::from(CONTEXT))] {
        let memory = runner.memory_usage().accounted_bytes();
        assert!(runner.prepare_paged_lane(lane, None, tokens).is_err());
        assert_eq!(runner.memory_usage().accounted_bytes(), memory);
        assert_eq!(boundary(&runner)?, initial);
    }
    for (tokens, start_pos, phase) in [
        (vec![1; 2049], 64, BatchPhase::Prompt),
        (vec![1], CONTEXT, BatchPhase::Decode),
    ] {
        assert!(
            runner
                .begin_decode(EngineBeginDecode {
                    phase,
                    tokens: Some(tokens),
                    start_pos
                })
                .await?
                .is_err()
        );
        assert_eq!(boundary(&runner)?, initial);
    }
    let mut control = load(KvCacheFormat::F16).await?;
    control.prepare_paged_lane(0, None, 0)?;
    control.prepare_paged_lane(0, None, 128)?;
    piece(&mut control, 0).await?;
    assert_eq!(
        piece(&mut runner, 64).await?,
        piece(&mut control, 64).await?
    );
    // The quota fits the full KV pool, but cannot fit context-wide logits as well.
    assert!(TINY_LIVE_QUOTA_BYTES < paged.slab_bytes + u64::from(CONTEXT) * 33 * 4);
    Ok(())
}

#[tokio::test]
async fn long_float_cache_does_not_raise_the_live_quota() -> Result<(), Box<dyn Error>> {
    let error = load(KvCacheFormat::F32)
        .await
        .err()
        .ok_or("float cache exceeded the quota")?;
    let quota = match error.downcast_ref::<ComputeError>() {
        Some(ComputeError::Quota(_)) => true,
        Some(ComputeError::OpSignature(message)) => message.contains("Error::Quota("),
        _ => false,
    };
    assert!(quota, "{error:?}");
    Ok(())
}
