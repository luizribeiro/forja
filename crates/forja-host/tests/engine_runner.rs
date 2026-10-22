//! Engine component integration checks.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use forja_cpu::CpuBackend;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineBatchSelect,
    EngineBatchSelectLane, EngineLoadConfig, EngineRunner, EngineStep, FixedVariantPick, Limits,
    SamplingParams, VariantPickArm, VariantRulePick, bindings::l9o::gpu::compute::Error,
};
use golden_fixtures::decode_f32_le;

const LIMITS: Limits = Limits::new(1024 * 1024, 4, 1024, 128, 1024 * 1024);
static NEXT_WEIGHT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[tokio::test]
async fn runs_an_engine_and_reads_its_logits() -> wasmtime::Result<()> {
    run_engine(test_guests::engine_smoke(), false, "engine").await
}

#[tokio::test]
async fn runs_an_sdk_exported_engine() -> wasmtime::Result<()> {
    run_engine(test_guests::engine_sdk_smoke(), true, "sdk-engine").await
}

#[tokio::test]
async fn runs_a_combined_batch_with_inert_padding() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("combined-batch").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let output = runner.batch_step(combined_batch(2)).await??;
    assert_eq!(runner.read(&output.tokens).await?, [3, 0, 0, 0, 0, 0, 0, 0]);
    Ok(())
}

#[tokio::test]
async fn queued_batches_retain_earlier_outputs() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("queued-combined-batch").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let first = runner.enqueue_batch_step(combined_batch(2)).await??;
    let second = runner.enqueue_batch_step(combined_batch(3)).await??;
    assert_eq!(
        runner.read_queued_batch_tokens(first).await?,
        [3, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        runner.read_queued_batch_tokens(second).await?,
        [0, 0, 0, 0, 0, 0, 0, 0]
    );
    Ok(())
}

#[tokio::test]
async fn combined_batch_refuses_corrupt_outputs() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("corrupt-batch").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let error = runner
        .batch_step(combined_batch(100))
        .await?
        .expect_err("malformed batch output must be rejected");
    assert!(matches!(error, Error::Layout(_)));
    assert_eq!(runner.paged_prefix()?.unwrap().logical_len()?, 0);
    Ok(())
}

#[tokio::test]
async fn combined_batch_obeys_guest_call_quota() -> wasmtime::Result<()> {
    let weights = weight_file("batch-deadline")?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS.with_guest_call_timeout(Duration::from_millis(50)),
        weights.path(),
    )
    .await?;
    runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                paged_cache: true,
                ..EngineLoadConfig::default()
            },
        )
        .await??;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let started = Instant::now();
    let error = runner
        .batch_step(combined_batch(u32::MAX))
        .await?
        .expect_err("batch call must time out");
    assert!(matches!(error, Error::Quota(_)));
    assert!(started.elapsed() < Duration::from_secs(1));
    Ok(())
}

fn combined_batch(token: u32) -> EngineBatch {
    EngineBatch {
        shape: BatchShape {
            phase: BatchPhase::Decode,
            bucket: 2,
            tokens: 1,
            selection: BatchSelection::Combined,
        },
        lanes: vec![
            Some(EngineBatchLane {
                tokens: Some(vec![token]),
                start_pos: 0,
                sampling: SamplingParams::default(),
            }),
            None,
        ],
    }
}

#[tokio::test]
async fn runs_a_forward_only_batch() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("forward-batch").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let logits = runner
        .batch_forward(EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Decode,
                bucket: 2,
                tokens: 1,
                selection: BatchSelection::Forward,
            },
            lanes: vec![
                Some(EngineBatchLane {
                    tokens: Some(vec![1]),
                    start_pos: 0,
                    sampling: SamplingParams::default(),
                }),
                None,
            ],
        })
        .await??;
    assert_eq!(runner.read(&logits).await?.len(), 8 * size_of::<f32>());
    let selected = runner
        .batch_select(
            &logits,
            EngineBatchSelect {
                shape: BatchShape {
                    phase: BatchPhase::Decode,
                    bucket: 2,
                    tokens: 1,
                    selection: BatchSelection::Masked,
                },
                lanes: vec![
                    Some(EngineBatchSelectLane {
                        start_pos: 0,
                        sampling: SamplingParams::default(),
                        allow: vec![2],
                    }),
                    None,
                ],
            },
        )
        .await??;
    assert_eq!(
        runner.read(&selected.tokens).await?,
        [2, 0, 0, 0, 0, 0, 0, 0]
    );
    Ok(())
}

#[tokio::test]
async fn paged_load_requires_a_serving_descriptor() -> wasmtime::Result<()> {
    let weights = weight_file("paged-without-descriptor")?;
    let mut runner = EngineRunner::new(
        test_guests::prefill_replay_smoke(),
        CpuBackend::new(),
        LIMITS,
        weights.path(),
    )
    .await?;
    let error = runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                paged_cache: true,
                ..EngineLoadConfig::default()
            },
        )
        .await?
        .unwrap_err();
    assert!(matches!(
        error,
        forja_host::bindings::l9o::gpu::compute::Error::Layout(_)
    ));
    Ok(())
}

#[tokio::test]
async fn prepares_independent_paged_lanes() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("paged-lanes").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    assert!(runner.prepare_paged_lane(1, None, 8)?);
    assert!(runner.prepare_paged_lane(2, None, 8).is_err());
    let memory = runner.paged_memory_usage()?.unwrap();
    assert!(memory.slab_bytes > 0);
    assert!(memory.slot_bytes > 0);
    Ok(())
}

#[tokio::test]
async fn rolls_back_only_the_selected_paged_lane() -> wasmtime::Result<()> {
    let (_weights, mut runner) = paged_smoke_runner("lane-rollback").await?;
    assert!(runner.prepare_paged_lane(0, None, 8)?);
    let checkpoint = runner.checkpoint_paged_lane(0)?.unwrap();
    assert!(runner.prepare_paged_lane(1, None, 4)?);

    runner.rollback_paged_lane(0, checkpoint)?;

    assert_eq!(runner.paged_lane_prefix(0)?.unwrap().reserved_tokens()?, 8);
    assert_eq!(runner.paged_lane_prefix(1)?.unwrap().reserved_tokens()?, 4);
    Ok(())
}

#[tokio::test]
async fn passes_engine_selections_through_load_config() -> wasmtime::Result<()> {
    let weights = weight_file("load-config")?;
    let mut runner = EngineRunner::new(
        test_guests::engine_sdk_smoke(),
        CpuBackend::new(),
        LIMITS,
        weights.path(),
    )
    .await?;
    let config = EngineLoadConfig {
        replay: true,
        paged_cache: false,
        kv_cache: forja_host::KvCacheFormat::F16,
        tunings: vec!["fused".to_owned()],
        fixed_variant_picks: vec![FixedVariantPick {
            site: "dense.decode".to_owned(),
            name: "matmul.gemv-transposed".to_owned(),
        }],
        variant_rule_picks: vec![VariantRulePick {
            site: "attention.decode".to_owned(),
            parameter: "position".to_owned(),
            arms: vec![VariantPickArm {
                lo: 0,
                hi: 1023,
                name: "sdpa.decomposed".to_owned(),
            }],
        }],
    };
    runner.load_with_selections(None, &config).await??;
    Ok(())
}

#[tokio::test]
async fn guest_step_stops_at_its_cpu_deadline() -> wasmtime::Result<()> {
    let limits = LIMITS.with_guest_call_timeout(Duration::from_millis(50));
    let weights = weight_file("deadline")?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        limits,
        weights.path(),
    )
    .await?;
    runner.load().await??;
    let started = Instant::now();
    let error = runner
        .step(EngineStep {
            tokens: vec![u32::MAX],
            start_pos: 0,
            taps: false,
        })
        .await?
        .expect_err("an infinite guest step must time out");
    assert!(matches!(error, Error::Quota(_)));
    assert!(started.elapsed() < Duration::from_secs(1));

    run_engine(test_guests::engine_smoke(), false, "deadline-followup").await
}

#[tokio::test]
async fn preceding_step_outputs_are_released() -> wasmtime::Result<()> {
    let limits = Limits::new(1024 * 1024, 4, 1024, 4, 1024 * 1024);
    let weights = weight_file("output-release")?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        limits,
        weights.path(),
    )
    .await?;
    runner.load().await??;
    for _ in 0..40 {
        runner
            .step(EngineStep {
                tokens: vec![7],
                start_pos: 0,
                taps: false,
            })
            .await??;
    }
    Ok(())
}

#[tokio::test]
async fn rejects_outputs_that_disagree_with_metadata() -> wasmtime::Result<()> {
    let weights = weight_file("metadata")?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        weights.path(),
    )
    .await?;
    runner.load().await??;
    for (token, taps) in [(100, false), (101, true), (102, true), (103, false)] {
        let error = runner
            .step(EngineStep {
                tokens: vec![token],
                start_pos: 0,
                taps,
            })
            .await?
            .expect_err("malformed engine output must be rejected");
        assert!(matches!(error, Error::Layout(_)));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
async fn metal_runner_rejects_a_cpu_runner_tensor() -> wasmtime::Result<()> {
    let weights = weight_file("foreign-tensor")?;
    let mut cpu = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        weights.path(),
    )
    .await?;
    cpu.load().await??;
    let output = cpu
        .step(EngineStep {
            tokens: vec![7],
            start_pos: 0,
            taps: false,
        })
        .await??;
    let backend = forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?;
    let mut metal =
        EngineRunner::new(test_guests::engine_smoke(), backend, LIMITS, weights.path()).await?;
    let error = metal
        .read(&output.logits)
        .await
        .expect_err("a foreign engine tensor must be rejected");
    assert!(matches!(error, Error::InvalidHandle(_)));
    Ok(())
}

async fn run_engine(component: &Path, taps: bool, fixture: &str) -> wasmtime::Result<()> {
    let weights = weight_file(fixture)?;
    let mut runner =
        EngineRunner::new(component, CpuBackend::new(), LIMITS, weights.path()).await?;

    let info = runner.describe().await?;
    assert_eq!(info.vocab, 4);
    let serving = info.serving.expect("engine smoke declares paged serving");
    assert_eq!(serving.page_tokens, 4);
    assert_eq!(serving.planes[0].label, "cache");
    assert_eq!(serving.active_lanes, 2);
    assert_eq!(serving.max_context, 32);
    runner.load().await??;
    let output = runner
        .step(EngineStep {
            tokens: vec![7],
            start_pos: 0,
            taps,
        })
        .await??;
    let bytes = runner.read(&output.logits).await?;
    let logits = decode_f32_le(&bytes)?;
    assert_eq!(logits, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(output.taps.len(), usize::from(taps));
    Ok(())
}

async fn paged_smoke_runner(
    fixture: &str,
) -> wasmtime::Result<(WeightFile, EngineRunner<CpuBackend>)> {
    let weights = weight_file(fixture)?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        weights.path(),
    )
    .await?;
    runner
        .load_with_selections(
            None,
            &EngineLoadConfig {
                paged_cache: true,
                ..EngineLoadConfig::default()
            },
        )
        .await??;
    Ok((weights, runner))
}

struct WeightFile {
    directory: PathBuf,
    path: PathBuf,
}

impl WeightFile {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for WeightFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn weight_file(test: &str) -> wasmtime::Result<WeightFile> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(wasmtime::Error::msg)?
        .as_nanos();
    let nonce = NEXT_WEIGHT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "forja-engine-runner-{test}-{}-{timestamp}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&directory).map_err(wasmtime::Error::msg)?;
    let path = directory.join("weights.safetensors");
    let weights = WeightFile { directory, path };
    fs::write(weights.path(), empty_safetensors()).map_err(wasmtime::Error::msg)?;
    Ok(weights)
}

fn empty_safetensors() -> Vec<u8> {
    let mut header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend_from_slice(&0_f32.to_le_bytes());
    bytes
}
