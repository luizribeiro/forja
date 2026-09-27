//! Engine component integration checks.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use forja_cpu::CpuBackend;
use forja_host::{EngineRunner, EngineStep, Limits, bindings::l9o::gpu::compute::Error};

const LIMITS: Limits = Limits::new(1024 * 1024, 4, 1024, 128, 1024 * 1024);

#[tokio::test]
async fn runs_an_engine_and_reads_its_logits() -> wasmtime::Result<()> {
    run_engine(test_guests::engine_smoke(), false).await
}

#[tokio::test]
async fn runs_an_sdk_exported_engine() -> wasmtime::Result<()> {
    run_engine(test_guests::engine_sdk_smoke(), true).await
}

#[tokio::test]
async fn guest_step_stops_at_its_cpu_deadline() -> wasmtime::Result<()> {
    let limits = LIMITS.with_guest_call_timeout(Duration::from_millis(50));
    let weights = weight_file()?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        limits,
        &weights,
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

    run_engine(test_guests::engine_smoke(), false).await
}

#[tokio::test]
async fn preceding_step_outputs_are_released() -> wasmtime::Result<()> {
    let limits = Limits::new(1024 * 1024, 4, 1024, 4, 1024 * 1024);
    let weights = weight_file()?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        limits,
        &weights,
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
    let weights = weight_file()?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        &weights,
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
    let weights = weight_file()?;
    let mut cpu = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        &weights,
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
        EngineRunner::new(test_guests::engine_smoke(), backend, LIMITS, &weights).await?;
    let error = metal
        .read(&output.logits)
        .await
        .expect_err("a foreign engine tensor must be rejected");
    assert!(matches!(error, Error::InvalidHandle(_)));
    Ok(())
}

async fn run_engine(component: &Path, taps: bool) -> wasmtime::Result<()> {
    let weights = weight_file()?;
    let mut runner = EngineRunner::new(component, CpuBackend::new(), LIMITS, &weights).await?;

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
    let logits = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(logits, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(output.taps.len(), usize::from(taps));
    Ok(())
}

fn weight_file() -> wasmtime::Result<PathBuf> {
    let weights =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/engine-empty.safetensors");
    std::fs::write(&weights, empty_safetensors()).map_err(wasmtime::Error::msg)?;
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
