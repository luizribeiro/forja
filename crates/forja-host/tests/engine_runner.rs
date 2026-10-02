//! Engine component integration checks.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use forja_cpu::CpuBackend;
use forja_host::{
    EngineLoadConfig, EngineRunner, EngineStep, FixedVariantPick, Limits, VariantPickArm,
    VariantRulePick, bindings::l9o::gpu::compute::Error,
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

    assert_eq!(runner.describe().await?.vocab, 4);
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
