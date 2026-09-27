//! Engine component integration checks.

use std::path::Path;

use forja_cpu::CpuBackend;
use forja_host::{EngineRunner, EngineStep, Limits};

const LIMITS: Limits = Limits::new(1024 * 1024, 4, 1024, 128, 1024 * 1024);

#[tokio::test]
async fn runs_an_engine_and_reads_its_logits() -> wasmtime::Result<()> {
    let weights =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/engine-empty.safetensors");
    std::fs::write(&weights, empty_safetensors()).map_err(wasmtime::Error::msg)?;
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        LIMITS,
        &weights,
    )
    .await?;

    assert_eq!(runner.describe().await?.vocab, 4);
    runner.load().await??;
    let output = runner
        .step(EngineStep {
            tokens: vec![7],
            start_pos: 0,
            taps: false,
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
    Ok(())
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
