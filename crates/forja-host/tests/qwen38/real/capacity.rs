use super::super::capacity::boundary;
use super::{Error, load_profiled};
use forja_host::{BatchPhase, EngineBeginDecode, EngineDecode, SamplingParams};

fn state_versions_bytes() -> Result<u64, Box<dyn Error>> {
    let mut bytes = 0_u64;
    for plane in qwen38::Config::model().state_planes()? {
        let width = match plane.dtype {
            forja_sdk::DType::BF16 => 2_u64,
            forja_sdk::DType::F32 => 4,
            _ => return Err("unexpected state precision".into()),
        };
        let size = plane
            .shape
            .iter()
            .try_fold(width, |size, dimension| {
                size.checked_mul(u64::from(*dimension))
            })
            .ok_or("state byte overflow")?;
        bytes = bytes.checked_add(size).ok_or("state byte overflow")?;
    }
    Ok(bytes
        .checked_mul(3)
        .and_then(|size| size.checked_mul(8))
        .ok_or("state version byte overflow")?)
}

async fn capacity(replay: bool) -> Result<(), Box<dyn Error>> {
    let (mut runner, _) =
        load_profiled(replay, None, None, test_guests::qwen38_long().to_path_buf()).await?;
    assert_eq!(runner.metrics().live_graphs > 0, replay);
    let paged = runner.paged_memory_usage()?.ok_or("missing paged ledger")?;
    let memory = runner.memory_usage();
    let charged = memory.accounted_bytes().ok_or("ledger overflow")?;
    assert!(charged <= 48 << 30);
    assert!(charged >= 15_132_802_048 + paged.slab_bytes + state_versions_bytes()?);
    assert!(paged.slab_bytes >= 16 << 30);
    for lane in 0..8 {
        runner.prepare_paged_lane(lane, None, 0)?;
    }
    runner.prepare_paged_lane(0, None, 262_144)?;
    let output = runner
        .decode(EngineDecode {
            phase: BatchPhase::Prompt,
            tokens: Some(vec![1; 64]),
            start_pos: 0,
            sampling: SamplingParams::default(),
        })
        .await??;
    assert_eq!(runner.read(&output.logits).await?.len(), 248_320 * 4);
    assert_eq!(runner.read(&output.token).await?.len(), 4);
    let initial = boundary(&runner)?;
    let submissions = runner.metrics().submissions;
    for (lane, tokens) in [(0, 262_145), (1, 262_144)] {
        let before = runner.memory_usage().accounted_bytes();
        assert!(runner.prepare_paged_lane(lane, None, tokens).is_err());
        assert_eq!(runner.memory_usage().accounted_bytes(), before);
        assert_eq!(boundary(&runner)?, initial);
    }
    assert!(
        runner
            .begin_decode(EngineBeginDecode {
                phase: BatchPhase::Prompt,
                tokens: Some(vec![1; 2049]),
                start_pos: 64
            })
            .await?
            .is_err()
    );
    assert_eq!(boundary(&runner)?, initial);
    assert_eq!(runner.metrics().submissions, submissions);
    assert!(
        runner
            .memory_usage()
            .accounted_bytes()
            .ok_or("ledger overflow")?
            <= 48 << 30
    );
    eprintln!(
        "correctness capacity diagnostic: replay={replay} memory={:?} paged={:?}",
        runner.memory_usage(),
        runner.paged_memory_usage()?
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_long_profile_admission_uses_full_ledger() -> Result<(), Box<dyn Error>> {
    capacity(false).await
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_long_profile_replay_fits_the_unchanged_quota() -> Result<(), Box<dyn Error>> {
    capacity(true).await
}

async fn float_cache(name: &str, replay: bool) -> Result<(), Box<dyn Error>> {
    let (mut runner, fixtures) = load_profiled(
        replay,
        None,
        None,
        test_guests::qwen38_float_kv().to_path_buf(),
    )
    .await?;
    assert_eq!(runner.metrics().live_graphs > 0, replay);
    super::verification::verify(&mut runner, fixtures.prompt(name).ok_or("missing prompt")?).await
}

macro_rules! float_cases {
    ($($name:ident: $prompt:literal),+ $(,)?) => { $(
        #[tokio::test]
        #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
        async fn $name() -> Result<(), Box<dyn Error>> { float_cache($prompt, false).await }
    )+ };
}
float_cases! {
    metal_real_float_cache_hello_matches_transformers: "hello",
    metal_real_float_cache_arithmetic_matches_transformers: "arithmetic",
    metal_real_float_cache_completion_matches_transformers: "completion",
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_float_cache_replay_matches_transformers() -> Result<(), Box<dyn Error>> {
    float_cache("arithmetic", true).await
}
