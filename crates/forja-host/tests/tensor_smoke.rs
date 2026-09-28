//! End-to-end component tests for the guest tensor surface.

use forja_core::{Backend, CommandList, DType, Op, Submission, ViewOp};
use forja_host::{Grants, Host, Limits, add_to_linker, component_engine};
use forja_testing::{DeterministicValues, F32_TOLERANCE, normwise_relative_error};
use golden_fixtures::decode_f32_le;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use wasmtime::Store;
use wasmtime::component::{Component, Instance, Linker};

const LIMITS: Limits = Limits::new(64 * 1024, 8, 16 * 1024, 8, 64 * 1024);
const COMMAND_LIMITS: Limits = Limits::new(128 * 1024, 8, 16 * 1024, 8, 64 * 1024);
const WEIGHT_LIMITS: Limits = Limits::new(20, 8, 16 * 1024, 4, 64 * 1024);
const SDK_LIMITS: Limits = Limits::new(64 * 1024 * 1024, 8, 2_100_000, 128, 7 * 1024 * 4)
    .with_command_limits(32, 64 * 1024 * 1024)
    .with_gpu_limits(Duration::from_secs(30), Duration::MAX)
    .with_store_limits(512 * 1024 * 1024, 10_000, 10_000);
static NEXT_WEIGHT_FILE: AtomicU64 = AtomicU64::new(0);

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

#[tokio::test(flavor = "multi_thread")]
async fn cpu_guest_fused_program_matches_trusted_operations() -> wasmtime::Result<()> {
    let expected = direct_fused_values()?;
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::program_smoke(),
        COMMAND_LIMITS,
    )
    .await?;
    let run = instance.get_typed_func::<(), (Result<Vec<f32>, String>,)>(&mut store, "run")?;
    Host::reset_guest_deadline(&mut store);
    let (actual,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    let actual = actual.map_err(wasmtime::Error::msg)?;
    let error = normwise_relative_error(&expected, &actual);
    assert!(error <= F32_TOLERANCE, "relative error {error}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn compatibility_cache_releases_evicted_kernel_leases() -> wasmtime::Result<()> {
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::program_smoke(),
        COMMAND_LIMITS.with_kernel_limit(8),
    )
    .await?;
    let churn =
        instance.get_typed_func::<(), (Result<(), String>,)>(&mut store, "churn-program-cache")?;
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| churn.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

#[tokio::test(flavor = "multi_thread")]
async fn explicit_kernels_and_compatibility_cache_share_quota() -> wasmtime::Result<()> {
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::program_smoke(),
        COMMAND_LIMITS.with_kernel_limit(8),
    )
    .await?;
    let exhaust =
        instance.get_typed_func::<(), (Result<(), String>,)>(&mut store, "exhaust-shared-quota")?;
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| exhaust.call_concurrent(accessor, ()).await)
        .await??;
    assert_eq!(
        result,
        Err("Error::Quota(\"live kernels exceed the guest limit\")".to_owned())
    );
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
async fn cpu_guest_uses_granted_weights() -> wasmtime::Result<()> {
    run_weights(forja_cpu::CpuBackend::new()).await
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_guest_uses_granted_weights() -> wasmtime::Result<()> {
    run_weights(forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?).await
}

#[tokio::test(flavor = "multi_thread")]
async fn cpu_sdk_attention_smoke() -> wasmtime::Result<()> {
    run_sdk_attention(forja_cpu::CpuBackend::new(), forja_cpu::CpuBackend::new()).await
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_tensor_smoke_runs_sdk_attention_block() -> wasmtime::Result<()> {
    run_sdk_attention(
        forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?,
        forja_metal::MetalBackend::new().map_err(wasmtime::Error::msg)?,
    )
    .await
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
    Host::reset_guest_deadline(&mut store);
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
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| input.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)?;

    let read =
        instance.get_typed_func::<(), (Result<(), String>,)>(&mut store, "large-read-refused")?;
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| read.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

#[tokio::test(flavor = "multi_thread")]
async fn kernel_churn_returns_a_quota_error() -> wasmtime::Result<()> {
    let limits = LIMITS.with_kernel_limit(4);
    let (mut store, instance) = instantiate(
        forja_cpu::CpuBackend::new(),
        test_guests::tensor_abuse(),
        limits,
    )
    .await?;
    let churn =
        instance.get_typed_func::<(), (Result<u32, String>,)>(&mut store, "churn-kernels")?;
    Host::reset_guest_deadline(&mut store);
    let (created,) = store
        .run_concurrent(async move |accessor| churn.call_concurrent(accessor, ()).await)
        .await??;
    assert_eq!(created.map_err(wasmtime::Error::msg)?, 4);
    Ok(())
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
    Host::reset_guest_deadline(&mut store);
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
    Host::reset_guest_deadline(&mut store);
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
    Host::reset_guest_deadline(&mut store);
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
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
        .await??;
    result.map_err(wasmtime::Error::msg)
}

async fn run_weights<B>(backend: B) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    let path = weight_file();
    let grants = Grants::new().with_weights("model", &path);
    let result = async {
        let (mut store, instance) =
            instantiate_with_grants(backend, test_guests::weights_smoke(), WEIGHT_LIMITS, grants)
                .await?;
        let run = instance.get_typed_func::<(), (Result<Vec<u8>, String>,)>(&mut store, "run")?;
        Host::reset_guest_deadline(&mut store);
        let (result,) = store
            .run_concurrent(async move |accessor| run.call_concurrent(accessor, ()).await)
            .await??;
        let actual = result.map_err(wasmtime::Error::msg)?;
        let expected = [22.0_f32, 28.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(wasmtime::Error::msg("weight matmul result differed"));
        }
        Ok(())
    }
    .await;
    fs::remove_file(path)?;
    result
}

async fn run_sdk_attention<B>(reference: B, guest: B) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    let values = AttentionValues::new();
    let expected = direct_attention(&reference, &values)?;
    drop(reference);
    let (mut store, instance) = instantiate(guest, test_guests::sdk_smoke(), SDK_LIMITS).await?;
    let run = instance
        .get_typed_func::<AttentionParams, (Result<Vec<f32>, String>,)>(&mut store, "run")?;
    let parameters = values.into_parameters();
    Host::reset_guest_deadline(&mut store);
    let (result,) = store
        .run_concurrent(async move |accessor| run.call_concurrent(accessor, parameters).await)
        .await??;
    let actual = result.map_err(wasmtime::Error::msg)?;
    let error = normwise_relative_error(&expected, &actual);
    if error > F32_TOLERANCE {
        return Err(wasmtime::Error::msg(format!(
            "SDK attention relative error {error} exceeded {F32_TOLERANCE}"
        )));
    }
    Ok(())
}

type AttentionParams = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

struct AttentionValues {
    input: Vec<f32>,
    norm_weight: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    v_weight: Vec<f32>,
    output_weight: Vec<f32>,
}

struct AttentionTensors {
    input: forja_core::Tensor,
    norm_weight: forja_core::Tensor,
    normalized: forja_core::Tensor,
    q_weight: forja_core::Tensor,
    k_weight: forja_core::Tensor,
    v_weight: forja_core::Tensor,
    q_projection: forja_core::Tensor,
    k_projection: forja_core::Tensor,
    v_projection: forja_core::Tensor,
    q_rows: forja_core::Tensor,
    k_rows: forja_core::Tensor,
    q_rotated: forja_core::Tensor,
    k_rotated: forja_core::Tensor,
    q_heads: forja_core::Tensor,
    k_heads: forja_core::Tensor,
    v_heads: forja_core::Tensor,
    positions: forja_core::Tensor,
    attended: forja_core::Tensor,
    attended_rows: forja_core::Tensor,
    contiguous: forja_core::Tensor,
    flattened: forja_core::Tensor,
    output_weight: forja_core::Tensor,
    projected: forja_core::Tensor,
    result: forja_core::Tensor,
}

impl AttentionValues {
    fn new() -> Self {
        let mut generator = DeterministicValues::new(0xa54f_f53a_5f1d_36f1);
        let mut next = |len| {
            (0..len)
                .map(|_| generator.next_f32() * 0.02)
                .collect::<Vec<_>>()
        };
        Self {
            input: next(7 * 1024),
            norm_weight: next(1024),
            q_weight: next(2048 * 1024),
            k_weight: next(1024 * 1024),
            v_weight: next(1024 * 1024),
            output_weight: next(1024 * 2048),
        }
    }

    fn into_parameters(self) -> AttentionParams {
        (
            self.input,
            self.norm_weight,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.output_weight,
        )
    }
}

fn direct_attention<B: Backend>(
    backend: &B,
    values: &AttentionValues,
) -> wasmtime::Result<Vec<f32>> {
    let input = f32_tensor(backend, &[7, 1024], &values.input)?;
    let norm_weight = f32_tensor(backend, &[1024], &values.norm_weight)?;
    let q_weight = f32_tensor(backend, &[2048, 1024], &values.q_weight)?;
    let k_weight = f32_tensor(backend, &[1024, 1024], &values.k_weight)?;
    let v_weight = f32_tensor(backend, &[1024, 1024], &values.v_weight)?;
    let output_weight = f32_tensor(backend, &[1024, 2048], &values.output_weight)?;
    let positions = backend.alloc(DType::U32, &[7]).map_err(backend_error)?;
    backend
        .write(
            &positions,
            &(0_u32..7).flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
        )
        .map_err(backend_error)?;

    let normalized = f32_output(backend, &[7, 1024])?;
    let q_projection = f32_output(backend, &[7, 2048])?;
    let k_projection = f32_output(backend, &[7, 1024])?;
    let v_projection = f32_output(backend, &[7, 1024])?;
    let q_weight = backend
        .view(&q_weight, ViewOp::Permute(vec![1, 0]))
        .map_err(backend_error)?;
    let k_weight = backend
        .view(&k_weight, ViewOp::Permute(vec![1, 0]))
        .map_err(backend_error)?;
    let v_weight = backend
        .view(&v_weight, ViewOp::Permute(vec![1, 0]))
        .map_err(backend_error)?;
    let q_rows = backend
        .view(&q_projection, ViewOp::Reshape(vec![7, 16, 128]))
        .map_err(backend_error)?;
    let k_rows = backend
        .view(&k_projection, ViewOp::Reshape(vec![7, 8, 128]))
        .map_err(backend_error)?;
    let v_rows = backend
        .view(&v_projection, ViewOp::Reshape(vec![7, 8, 128]))
        .map_err(backend_error)?;
    let q_rotated = f32_output(backend, &[7, 16, 128])?;
    let k_rotated = f32_output(backend, &[7, 8, 128])?;
    let q_heads = backend
        .view(&q_rotated, ViewOp::Permute(vec![1, 0, 2]))
        .map_err(backend_error)?;
    let k_heads = backend
        .view(&k_rotated, ViewOp::Permute(vec![1, 0, 2]))
        .map_err(backend_error)?;
    let v_heads = backend
        .view(&v_rows, ViewOp::Permute(vec![1, 0, 2]))
        .map_err(backend_error)?;
    let attended = f32_output(backend, &[16, 7, 128])?;
    let attended_rows = backend
        .view(&attended, ViewOp::Permute(vec![1, 0, 2]))
        .map_err(backend_error)?;
    let contiguous = f32_output(backend, &[7, 16, 128])?;
    let flattened = backend
        .view(&contiguous, ViewOp::Reshape(vec![7, 2048]))
        .map_err(backend_error)?;
    let output_weight = backend
        .view(&output_weight, ViewOp::Permute(vec![1, 0]))
        .map_err(backend_error)?;
    let projected = f32_output(backend, &[7, 1024])?;
    let result = f32_output(backend, &[7, 1024])?;

    let tensors = AttentionTensors {
        input,
        norm_weight,
        normalized,
        q_weight,
        k_weight,
        v_weight,
        q_projection,
        k_projection,
        v_projection,
        q_rows,
        k_rows,
        q_rotated,
        k_rotated,
        q_heads,
        k_heads,
        v_heads,
        positions,
        attended,
        attended_rows,
        contiguous,
        flattened,
        output_weight,
        projected,
        result,
    };
    let commands = record_attention(&tensors)?;
    backend
        .submit(commands)
        .map_err(backend_error)?
        .wait()
        .map_err(backend_error)?;
    let bytes = backend.read(&tensors.result).map_err(backend_error)?;
    Ok(decode_f32_le(&bytes)?)
}

fn record_attention(tensors: &AttentionTensors) -> wasmtime::Result<CommandList> {
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::RmsNorm { eps: 1.0e-6 },
            &[&tensors.input, &tensors.norm_weight],
            &tensors.normalized,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Matmul,
            &[&tensors.normalized, &tensors.q_weight],
            &tensors.q_projection,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Matmul,
            &[&tensors.normalized, &tensors.k_weight],
            &tensors.k_projection,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Matmul,
            &[&tensors.normalized, &tensors.v_weight],
            &tensors.v_projection,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Rope { theta: 1_000_000.0 },
            &[&tensors.q_rows, &tensors.positions],
            &tensors.q_rotated,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Rope { theta: 1_000_000.0 },
            &[&tensors.k_rows, &tensors.positions],
            &tensors.k_rotated,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: 0,
            },
            &[&tensors.q_heads, &tensors.k_heads, &tensors.v_heads],
            &tensors.attended,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(Op::Copy, &[&tensors.attended_rows], &tensors.contiguous)
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Matmul,
            &[&tensors.flattened, &tensors.output_weight],
            &tensors.projected,
        )
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(
            Op::Add,
            &[&tensors.input, &tensors.projected],
            &tensors.result,
        )
        .map_err(wasmtime::Error::msg)?;
    Ok(commands)
}

fn f32_tensor<B: Backend>(
    backend: &B,
    shape: &[u32],
    values: &[f32],
) -> wasmtime::Result<forja_core::Tensor> {
    let tensor = backend.alloc(DType::F32, shape).map_err(backend_error)?;
    backend
        .write(
            &tensor,
            &values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(backend_error)?;
    Ok(tensor)
}

fn f32_output<B: Backend>(backend: &B, shape: &[u32]) -> wasmtime::Result<forja_core::Tensor> {
    backend.alloc(DType::F32, shape).map_err(backend_error)
}

fn weight_file() -> PathBuf {
    let mut header =
        br#"{"projection":{"dtype":"F32","shape":[3,2],"data_offsets":[0,24]}}"#.to_vec();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(
        [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes),
    );
    let path = std::env::temp_dir().join(format!(
        "forja-weight-guest-{}-{}",
        std::process::id(),
        NEXT_WEIGHT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, bytes).unwrap();
    path
}

fn direct_fused_values() -> wasmtime::Result<Vec<f32>> {
    let backend = forja_cpu::CpuBackend::new();
    let residual_values = fused_values(0.03125);
    let update_values = fused_values(-0.015_625);
    let weight_values = (0_u16..1024)
        .map(|index| 0.5 + f32::from(index) / 2048.0)
        .collect::<Vec<_>>();
    let residual = f32_tensor(&backend, &[7, 1024], &residual_values)?;
    let update = f32_tensor(&backend, &[7, 1024], &update_values)?;
    let weight = f32_tensor(&backend, &[1024], &weight_values)?;
    let sum = f32_output(&backend, &[7, 1024])?;
    let normalized = f32_output(&backend, &[7, 1024])?;
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::Add, &[&residual, &update], &sum)
        .map_err(wasmtime::Error::msg)?;
    commands
        .dispatch(Op::RmsNorm { eps: 1.0e-6 }, &[&sum, &weight], &normalized)
        .map_err(wasmtime::Error::msg)?;
    backend
        .submit(commands)
        .map_err(backend_error)?
        .wait()
        .map_err(backend_error)?;
    let bytes = backend.read(&normalized).map_err(backend_error)?;
    Ok(decode_f32_le(&bytes)?)
}

fn fused_values(scale: f32) -> Vec<f32> {
    (0_u16..7 * 1024)
        .map(|index| (f32::from(index % 257) - 128.0) * scale)
        .collect()
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
    instantiate_with_grants(backend, component_path, limits, Grants::new()).await
}

async fn instantiate_with_grants<B>(
    backend: B,
    component_path: &Path,
    limits: Limits,
    grants: Grants,
) -> wasmtime::Result<(Store<Host<B>>, Instance)>
where
    B: Backend + Send + Sync + 'static,
{
    let engine = component_engine()?;
    let component = Component::from_file(&engine, component_path)?;
    let mut linker = Linker::new(&engine);
    add_to_linker(&mut linker)?;
    let mut store = Host::new_store_with_grants(&engine, backend, limits, grants);
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
