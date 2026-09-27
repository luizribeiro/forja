//! End-to-end component tests for the guest tensor surface.

use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_host::{Host, Limits, add_to_linker};
use std::path::Path;
use wasmtime::component::{Component, Instance, Linker};
use wasmtime::{Config, Engine, Store};

const LIMITS: Limits = Limits::new(64 * 1024, 8, 16 * 1024, 8, 64 * 1024);
const COMMAND_LIMITS: Limits = Limits::new(128 * 1024, 8, 16 * 1024, 8, 64 * 1024);

#[tokio::test(flavor = "multi_thread")]
async fn cpu_tensor_smoke() -> wasmtime::Result<()> {
    let checksum = run(forja_cpu::CpuBackend::new()).await?;
    assert_eq!(checksum, expected_checksum());
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_tensor_smoke() -> wasmtime::Result<()> {
    let backend = forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?;
    let checksum = run(backend).await?;
    assert_eq!(checksum, expected_checksum());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn cpu_tensor_smoke_submits_commands() -> wasmtime::Result<()> {
    let expected = direct_checksums(&forja_cpu::CpuBackend::new())?;
    let actual = run_commands(forja_cpu::CpuBackend::new()).await?;
    assert_eq!(actual, expected);
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_tensor_smoke_submits_commands() -> wasmtime::Result<()> {
    let expected =
        direct_checksums(&forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?)?;
    let actual =
        run_commands(forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?).await?;
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn adversarial_tensor_calls_return_errors() -> wasmtime::Result<()> {
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::tensor_abuse(),
        LIMITS,
    )
    .await?;
    let run = instance.get_typed_func::<(), (Result<u64, String>,)>(&mut store, "run")?;
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    assert_eq!(result.map_err(wasmtime::Error::msg)?, expected_checksum());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn large_broadcast_is_only_usable_without_reading() -> wasmtime::Result<()> {
    let limits = Limits::new(4, 8, 4_000_000_000, 4, 4);
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::tensor_abuse(),
        limits,
    )
    .await?;
    let input =
        instance.get_typed_func::<(), (Result<(), String>,)>(&mut store, "large-dispatch-input")?;
    let (result,) = store
        .run_concurrent(async move |accessor| input.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)?;

    let read =
        instance.get_typed_func::<(), (Result<(), String>,)>(&mut store, "large-read-refused")?;
    let (result,) = store
        .run_concurrent(async move |accessor| read.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

#[tokio::test(flavor = "multi_thread")]
async fn guest_memory_growth_stops_at_the_store_limit() -> wasmtime::Result<()> {
    let limits = LIMITS.with_store_limits(16 * 1024 * 1024, 10_000, 10_000);
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::tensor_abuse(),
        limits,
    )
    .await?;
    let grow = instance.get_typed_func::<(u32,), (bool,)>(&mut store, "grow-memory")?;
    let (failed,) = store
        .run_concurrent(async move |accessor| {
            grow.call_concurrent(accessor, (64 * 1024 * 1024,)).await
        })
        .await??;
    assert!(failed);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn canonical_handle_misuse_traps() -> wasmtime::Result<()> {
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::tensor_abuse(),
        LIMITS,
    )
    .await?;
    let misuse = instance.get_typed_func::<(), ()>(&mut store, "misuse-handle")?;
    let result = store
        .run_concurrent(async move |accessor| misuse.call_concurrent(accessor, ()).await)
        .await;
    let trap = result.expect_err("reusing a consumed handle must trap");
    assert!(trap.to_string().contains("handle"));
    Ok(())
}

async fn run<B>(backend: B) -> wasmtime::Result<u64>
where
    B: Backend + Send + Sync + 'static,
{
    let (mut store, instance) = instantiate(backend, test_guests::tensor_smoke(), LIMITS).await?;
    let run = instance.get_typed_func::<(), (Result<u64, String>,)>(&mut store, "run")?;
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

async fn run_commands<B>(backend: B) -> wasmtime::Result<(u64, u64)>
where
    B: Backend + Send + Sync + 'static,
{
    let (mut store, instance) =
        instantiate(backend, test_guests::rmsnorm_smoke(), COMMAND_LIMITS).await?;
    let run = instance.get_typed_func::<(), (Result<(u64, u64), String>,)>(&mut store, "run")?;
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

fn direct_checksums<B: Backend>(backend: &B) -> wasmtime::Result<(u64, u64)> {
    let input = backend
        .alloc(DType::F32, &[7, 1024])
        .map_err(backend_error)?;
    let weight = backend.alloc(DType::F32, &[1024]).map_err(backend_error)?;
    let normalized = backend
        .alloc(DType::F32, &[7, 1024])
        .map_err(backend_error)?;
    let activated = backend
        .alloc(DType::F32, &[7, 1024])
        .map_err(backend_error)?;
    backend
        .write(&input, &command_input_bytes())
        .map_err(backend_error)?;
    backend
        .write(&weight, &command_weight_bytes())
        .map_err(backend_error)?;
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::RmsNorm { eps: 0.00001 },
            &[&input, &weight],
            &normalized,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(Op::SiluMul, &[&normalized, &input], &activated)
        .map_err(wasmtime::Error::msg)?;
    let submission = backend.submit(commands).map_err(backend_error)?;
    submission.wait().map_err(backend_error)?;
    let normalized = backend.read(&normalized).map_err(backend_error)?;
    let activated = backend.read(&activated).map_err(backend_error)?;
    Ok((checksum(&normalized), checksum(&activated)))
}

fn command_input_bytes() -> Vec<u8> {
    (0_u16..7 * 1024)
        .flat_map(|index| ((f32::from(index % 257) - 128.0) / 37.0).to_le_bytes())
        .collect()
}

fn command_weight_bytes() -> Vec<u8> {
    (0_u16..1024)
        .flat_map(|index| (0.5 + f32::from(index) / 2048.0).to_le_bytes())
        .collect()
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn backend_error(error: impl std::fmt::Display) -> wasmtime::Error {
    wasmtime::Error::msg(error.to_string())
}

async fn instantiate<B>(
    backend: B,
    component_path: &Path,
    limits: Limits,
) -> wasmtime::Result<(Store<Host<B>>, Instance)>
where
    B: Backend + Send + Sync + 'static,
{
    let mut config = Config::new();
    config.wasm_component_model_async(true);
    config.concurrency_support(true);
    let engine = Engine::new(&config)?;
    let component = Component::from_file(&engine, component_path)?;
    let mut linker = Linker::new(&engine);
    add_to_linker(&mut linker)?;
    let mut store = Host::new_store(&engine, backend, limits);
    let instance = linker.instantiate_async(&mut store, &component).await?;
    Ok((store, instance))
}

fn expected_checksum() -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for column in 0_u16..1024 {
        for row in 0_u16..7 {
            let value = f32::from(row * 1024 + column) / 251.0;
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    hash
}
