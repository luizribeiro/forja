//! End-to-end component tests for the guest tensor surface.

use forja_core::Backend;
use forja_host::{Host, Limits, add_to_linker};
use wasmtime::component::{Component, Instance, Linker};
use wasmtime::{Config, Engine, Store};

const LIMITS: Limits = Limits::new(64 * 1024, 8, 16 * 1024, 8, 64 * 1024);

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

async fn run<B>(backend: B) -> wasmtime::Result<u64>
where
    B: Backend + Send + Sync + 'static,
{
    let (mut store, instance) = instantiate(backend).await?;
    let run = instance.get_typed_func::<(), (Result<u64, String>,)>(&mut store, "run")?;
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

async fn instantiate<B>(backend: B) -> wasmtime::Result<(Store<Host<B>>, Instance)>
where
    B: Backend + Send + Sync + 'static,
{
    let mut config = Config::new();
    config.wasm_component_model_async(true);
    config.concurrency_support(true);
    let engine = Engine::new(&config)?;
    let component = Component::from_file(&engine, test_guests::tensor_smoke())?;
    let mut linker = Linker::new(&engine);
    add_to_linker(&mut linker)?;
    let mut store = Host::new_store(&engine, backend, LIMITS);
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
