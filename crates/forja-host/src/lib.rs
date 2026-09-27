//! Trusted host integration for running Forja guest components.

#![allow(clippy::manual_async_fn)]

#[cfg(test)]
mod fuzz_tests;
mod native;
mod weights;

use std::time::{Duration, Instant};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use forja_core::{
    Backend, BackendError, CommandList, DType, Layout, LayoutError, Op, OpError, Slice, Submission,
    SubmissionProfile, Tensor, ViewOp,
    program::{
        BinOp, Inst, MAX_INSTRUCTIONS, MAX_OUTPUTS, Program, ProgramError, ProgramKind, RedOp,
        UnOp, ValidatedProgram, ValueType,
    },
};
use wasmtime::component::{Accessor, Component, HasData, Linker, Resource, ResourceTable};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder, Trap};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

pub use native::{NativeCommandList, NativeHost, NativeTensor};
pub use weights::{Safetensors, WeightError, WeightSource, WeightTensor};

/// Host bindings for the guest-facing compute interface.
#[allow(missing_docs)]
pub mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "host",
        imports: { default: async | trappable },
        require_store_data_send: true,
        with: {
            "l9o:gpu/compute.tensor": crate::TensorEntry,
            "l9o:gpu/compute.command-list": crate::CommandListEntry,
            "l9o:gpu/compute.weights": crate::WeightsEntry,
        },
    });
}

/// Bindings for components implementing the engine contract.
#[allow(missing_docs)]
pub mod engine_bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "engine-component",
        imports: { default: async | trappable },
        exports: { default: async },
        require_store_data_send: true,
        with: {
            "l9o:gpu/compute": crate::bindings::l9o::gpu::compute,
        },
    });
}

use bindings::l9o::gpu::compute;

/// Static metadata declared by an engine component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineInfo {
    /// Logit vector length.
    pub vocab: u32,
    /// Largest accepted token position.
    pub max_context: u32,
    /// Layers whose hidden states can be returned by a step.
    pub tap_layers: Vec<u32>,
}

/// Input to one unbatched engine invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineStep {
    /// Token ids processed by this invocation.
    pub tokens: Vec<u32>,
    /// Position of the first token.
    pub start_pos: u32,
    /// Whether declared hidden-state taps should be returned.
    pub taps: bool,
}

/// A tensor returned by a component and owned by its runner.
///
/// The handle remains valid until the runner starts its next step, which releases every output
/// from the preceding step before invoking the guest.
#[derive(Debug)]
pub struct EngineTensor {
    handle: u32,
    runner_id: u64,
}

/// Device-resident outputs from one engine invocation.
#[derive(Debug)]
pub struct EngineOutput {
    /// Last-position logits with shape `[vocab]`.
    pub logits: EngineTensor,
    /// Requested per-layer hidden states.
    pub taps: Vec<EngineTensor>,
}

/// Cumulative execution counters for an engine runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineMetrics {
    /// Successfully completed backend submissions.
    pub submissions: u64,
    /// Completed submissions that supplied device timestamps.
    pub timed_submissions: u64,
    /// Sum of device execution time reported by timed submissions.
    pub gpu_time: Duration,
}

/// Count and wall time for calls to one guest import.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImportProfile {
    /// Number of calls.
    pub count: u64,
    /// Total host wall time in the calls.
    pub time: Duration,
}

/// Guest-import timings collected during one engine step.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImportProfiles {
    /// Tensor allocation calls.
    pub alloc: ImportProfile,
    /// Slice-view calls.
    pub view_slice: ImportProfile,
    /// Reshape-view calls.
    pub view_reshape: ImportProfile,
    /// Permute-view calls.
    pub view_permute: ImportProfile,
    /// Broadcast-view calls.
    pub view_broadcast: ImportProfile,
    /// Tensor write calls.
    pub write: ImportProfile,
    /// Dispatch-recording calls.
    pub dispatch: ImportProfile,
    /// Submission calls.
    pub submit: ImportProfile,
    /// Tensor read calls.
    pub read: ImportProfile,
    /// Command-list construction calls.
    pub command_list: ImportProfile,
    /// Tensor and command-list resource-drop calls.
    pub resource_drop: ImportProfile,
}

impl ImportProfiles {
    fn total_time(&self) -> Duration {
        [
            self.alloc,
            self.view_slice,
            self.view_reshape,
            self.view_permute,
            self.view_broadcast,
            self.write,
            self.dispatch,
            self.submit,
            self.read,
            self.command_list,
            self.resource_drop,
        ]
        .into_iter()
        .fold(Duration::ZERO, |total, profile| total + profile.time)
    }
}

/// Timing detail for one engine step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineStepProfile {
    /// Complete host wall time for the step.
    pub wall_time: Duration,
    /// Step wall time outside the measured imports.
    pub guest_time: Duration,
    /// Guest-import counts and timings.
    pub imports: ImportProfiles,
    /// Backend output-buffer allocations.
    pub allocations: ImportProfile,
    /// Backend buffer releases.
    pub releases: ImportProfile,
    /// Detailed timing for the step's submission.
    pub submission: Option<SubmissionProfile>,
}

#[derive(Clone, Copy)]
enum ImportKind {
    Alloc,
    ViewSlice,
    ViewReshape,
    ViewPermute,
    ViewBroadcast,
    Write,
    Dispatch,
    Submit,
    Read,
    CommandList,
    ResourceDrop,
}

struct ImportTimer {
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
    kind: ImportKind,
    started: Option<Instant>,
}

impl ImportTimer {
    fn start(profile: Option<Arc<Mutex<EngineStepProfile>>>, kind: ImportKind) -> Self {
        let started = profile.as_ref().map(|_| Instant::now());
        Self {
            profile,
            kind,
            started,
        }
    }
}

impl Drop for ImportTimer {
    fn drop(&mut self) {
        let (Some(profile), Some(started)) = (&self.profile, self.started) else {
            return;
        };
        let Ok(mut profile) = profile.lock() else {
            return;
        };
        let timing = match self.kind {
            ImportKind::Alloc => &mut profile.imports.alloc,
            ImportKind::ViewSlice => &mut profile.imports.view_slice,
            ImportKind::ViewReshape => &mut profile.imports.view_reshape,
            ImportKind::ViewPermute => &mut profile.imports.view_permute,
            ImportKind::ViewBroadcast => &mut profile.imports.view_broadcast,
            ImportKind::Write => &mut profile.imports.write,
            ImportKind::Dispatch => &mut profile.imports.dispatch,
            ImportKind::Submit => &mut profile.imports.submit,
            ImportKind::Read => &mut profile.imports.read,
            ImportKind::CommandList => &mut profile.imports.command_list,
            ImportKind::ResourceDrop => &mut profile.imports.resource_drop,
        };
        timing.count = timing.count.saturating_add(1);
        timing.time = timing.time.saturating_add(started.elapsed());
    }
}

#[derive(Clone, Copy)]
enum BackendEvent {
    Allocation,
    Release,
}

struct BackendTimer {
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
    event: BackendEvent,
    started: Option<Instant>,
}

impl BackendTimer {
    fn start(profile: Option<&Arc<Mutex<EngineStepProfile>>>, event: BackendEvent) -> Self {
        let started = profile.map(|_| Instant::now());
        Self {
            profile: profile.cloned(),
            event,
            started,
        }
    }
}

impl Drop for BackendTimer {
    fn drop(&mut self) {
        let (Some(profile), Some(started)) = (&self.profile, self.started) else {
            return;
        };
        let Ok(mut profile) = profile.lock() else {
            return;
        };
        let timing = match self.event {
            BackendEvent::Allocation => &mut profile.allocations,
            BackendEvent::Release => &mut profile.releases,
        };
        timing.count = timing.count.saturating_add(1);
        timing.time = timing.time.saturating_add(started.elapsed());
    }
}

/// An instantiated engine component and its host resources.
pub struct EngineRunner<B: Backend + Send + Sync + 'static> {
    store: Store<Host<B>>,
    instance: engine_bindings::EngineComponent,
    id: u64,
    info: Option<EngineInfo>,
    output_handles: Vec<u32>,
    profiling: bool,
    last_profile: Option<EngineStepProfile>,
}

const EPOCH_TICK: Duration = Duration::from_millis(10);
static EPOCH_ENGINES: Mutex<Vec<EpochEngine>> = Mutex::new(Vec::new());
static EPOCH_TICKER: OnceLock<()> = OnceLock::new();
static NEXT_RUNNER_ID: AtomicU64 = AtomicU64::new(1);

struct EpochEngine {
    engine: Engine,
    live: Weak<()>,
}

/// Builds an engine configured for asynchronous components and epoch interruption.
///
/// # Errors
///
/// Returns an error when Wasmtime cannot create the engine.
pub fn component_engine() -> wasmtime::Result<Engine> {
    let mut config = Config::new();
    config.wasm_component_model_async(true);
    config.concurrency_support(true);
    config.epoch_interruption(true);
    Engine::new(&config)
}

fn register_epoch_engine(engine: &Engine) -> Arc<()> {
    let live = Arc::new(());
    if let Ok(mut engines) = EPOCH_ENGINES.lock() {
        engines.push(EpochEngine {
            engine: engine.clone(),
            live: Arc::downgrade(&live),
        });
    }
    EPOCH_TICKER.get_or_init(|| {
        drop(thread::spawn(|| {
            loop {
                thread::sleep(EPOCH_TICK);
                if let Ok(mut engines) = EPOCH_ENGINES.lock() {
                    engines.retain(|entry| {
                        if entry.live.strong_count() == 0 {
                            false
                        } else {
                            entry.engine.increment_epoch();
                            true
                        }
                    });
                }
            }
        }));
    });
    live
}

fn epoch_ticks(timeout: Duration) -> u64 {
    let tick_ns = EPOCH_TICK.as_nanos();
    let ticks = timeout.as_nanos().saturating_add(tick_ns - 1) / tick_ns;
    u64::try_from(ticks).map_or(u64::MAX, |ticks| ticks.max(1))
}

impl<B> EngineRunner<B>
where
    B: Backend + Send + Sync + 'static,
{
    /// Instantiates a component and gives it access to one safetensors file.
    ///
    /// # Errors
    ///
    /// Returns component compilation, linking, instantiation, or grant errors.
    pub async fn new(
        component_path: &Path,
        backend: B,
        limits: Limits,
        weights_path: impl Into<PathBuf>,
    ) -> wasmtime::Result<Self> {
        let engine = component_engine()?;
        let component = Component::from_file(&engine, component_path)?;
        let mut linker = Linker::new(&engine);
        add_engine_to_linker(&mut linker)?;
        let grants = Grants::new().with_weights("engine", weights_path);
        let mut store = Host::new_store_with_grants(&engine, backend, limits, grants);
        let instance =
            engine_bindings::EngineComponent::instantiate_async(&mut store, &component, &linker)
                .await?;
        let id = NEXT_RUNNER_ID
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
            .map_err(|_| wasmtime::Error::msg("engine runner id space is exhausted"))?;
        Ok(Self {
            store,
            instance,
            id,
            info: None,
            output_handles: Vec::new(),
            profiling: false,
            last_profile: None,
        })
    }

    /// Returns the engine's static metadata.
    ///
    /// # Errors
    ///
    /// Returns a component execution error.
    pub async fn describe(&mut self) -> wasmtime::Result<EngineInfo> {
        self.set_guest_deadline();
        let info = self
            .instance
            .l9o_gpu_engine()
            .call_describe(&mut self.store)
            .await?;
        let info = EngineInfo {
            vocab: info.vocab,
            max_context: info.max_context,
            tap_layers: info.tap_layers,
        };
        self.info = Some(info.clone());
        Ok(info)
    }

    /// Opens the configured weight grant and asks the engine to load it.
    ///
    /// # Errors
    ///
    /// Returns a host, component, or engine loading error.
    pub async fn load(&mut self) -> wasmtime::Result<Result<(), compute::Error>> {
        self.set_guest_deadline();
        let weights = self
            .store
            .data_mut()
            .open_weights("engine")
            .map_err(wasmtime::Error::msg)?;
        let engine = self.instance.l9o_gpu_engine();
        match self
            .store
            .run_concurrent(async move |accessor| engine.call_load(accessor, weights).await)
            .await
        {
            Ok(result) => result,
            Err(error) if is_epoch_timeout(&error) => Ok(Err(guest_timeout())),
            Err(error) => Err(error),
        }
    }

    /// Runs one engine invocation and retains its device tensors.
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn step(
        &mut self,
        input: EngineStep,
    ) -> wasmtime::Result<Result<EngineOutput, compute::Error>> {
        if !self.profiling {
            return self.step_inner(input).await;
        }
        self.store.data_mut().begin_profile_step();
        let started = Instant::now();
        let result = self.step_inner(input).await;
        self.last_profile = self.store.data_mut().finish_profile_step(started.elapsed());
        result
    }

    async fn step_inner(
        &mut self,
        input: EngineStep,
    ) -> wasmtime::Result<Result<EngineOutput, compute::Error>> {
        use engine_bindings::exports::l9o::gpu::engine::{StepIn, StepOut};
        if let Err(error) = self.release_outputs() {
            return Ok(Err(error));
        }
        let info = match &self.info {
            Some(info) => info.clone(),
            None => self.describe().await?,
        };
        let Ok(sequence) = u32::try_from(input.tokens.len()) else {
            return Ok(Err(quota("token count exceeds u32")));
        };
        let taps_requested = input.taps;
        self.set_guest_deadline();
        let engine = self.instance.l9o_gpu_engine();
        let output = match self
            .store
            .run_concurrent(async move |accessor| {
                engine
                    .call_step(
                        accessor,
                        StepIn {
                            tokens: input.tokens,
                            start_pos: input.start_pos,
                            taps: input.taps,
                        },
                    )
                    .await
            })
            .await
        {
            Ok(output) => output?,
            Err(error) if is_epoch_timeout(&error) => return Ok(Err(guest_timeout())),
            Err(error) => return Err(error),
        }?;
        let StepOut { logits, taps } = output;
        let handles = std::iter::once(logits.rep())
            .chain(taps.iter().map(Resource::rep))
            .collect::<Vec<_>>();
        if let Err(error) = self.validate_output(&info, sequence, taps_requested, &logits, &taps) {
            return Ok(Err(match self.release_handles(handles) {
                Ok(()) => error,
                Err(release_error) => release_error,
            }));
        }
        self.output_handles = handles;
        Ok(Ok(EngineOutput {
            logits: EngineTensor {
                handle: logits.rep(),
                runner_id: self.id,
            },
            taps: taps
                .into_iter()
                .map(|tensor| EngineTensor {
                    handle: tensor.rep(),
                    runner_id: self.id,
                })
                .collect(),
        }))
    }

    /// Reads a returned tensor through the selected backend.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or backend read failure.
    pub async fn read(&mut self, tensor: &EngineTensor) -> Result<Vec<u8>, compute::Error> {
        if tensor.runner_id != self.id {
            return Err(invalid_handle("engine tensor belongs to another runner"));
        }
        let resource = Resource::new_borrow(tensor.handle);
        match self.store.data().prepare_read(&resource) {
            Ok(request) => request.run().await.map_err(guest_error),
            Err(error) => Err(error),
        }
    }

    /// Returns cumulative backend execution counters.
    #[must_use]
    pub fn metrics(&self) -> EngineMetrics {
        self.store.data().engine_metrics()
    }

    /// Enables or disables detailed profiling for subsequent steps.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    /// Takes the most recently completed step profile.
    pub fn take_profile(&mut self) -> Option<EngineStepProfile> {
        self.last_profile.take()
    }

    fn set_guest_deadline(&mut self) {
        Host::reset_guest_deadline(&mut self.store);
    }

    fn release_outputs(&mut self) -> Result<(), compute::Error> {
        let handles = std::mem::take(&mut self.output_handles);
        self.release_handles(handles)
    }

    fn release_handles(&mut self, handles: Vec<u32>) -> Result<(), compute::Error> {
        let mut failure = None;
        for handle in handles {
            if let Err(error) = self.store.data_mut().drop_tensor(Resource::new_own(handle))
                && failure.is_none()
            {
                failure = Some(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn validate_output(
        &self,
        info: &EngineInfo,
        sequence: u32,
        taps_requested: bool,
        logits: &Resource<TensorEntry>,
        taps: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        let logits = self.store.data().entry(logits)?;
        if logits.tensor.layout().dtype() != DType::F32
            || logits.tensor.layout().shape() != [info.vocab]
        {
            return Err(compute::Error::Layout(format!(
                "engine logits must be f32 [{}]",
                info.vocab
            )));
        }
        let expected_taps = if taps_requested {
            info.tap_layers.len()
        } else {
            0
        };
        if taps.len() != expected_taps {
            return Err(compute::Error::Layout(format!(
                "engine returned {} taps, expected {expected_taps}",
                taps.len()
            )));
        }
        for tap in taps {
            let tap = self.store.data().entry(tap)?;
            let shape = tap.tensor.layout().shape();
            if tap.tensor.layout().dtype() != DType::F32
                || shape.len() != 2
                || shape.first() != Some(&sequence)
            {
                return Err(compute::Error::Layout(format!(
                    "engine taps must be f32 [{sequence}, hidden]"
                )));
            }
        }
        Ok(())
    }
}

/// Resource limits applied before backend work or component allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    live_bytes: u64,
    tensor_rank: usize,
    tensor_elements: u64,
    live_tensor_handles: usize,
    read_bytes: u64,
    guest_memory_bytes: usize,
    table_elements: usize,
    instances: usize,
    dispatches_per_list: usize,
    work_per_dispatch: u64,
    guest_call_timeout: Duration,
    submission_timeout: Duration,
    gpu_time_budget: Duration,
}

/// Host-configured capabilities available to a guest component.
#[derive(Clone, Debug, Default)]
pub struct Grants {
    weights: HashMap<String, PathBuf>,
}

impl Grants {
    /// Creates an empty grant set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grants one opaque key access to a safetensors file.
    #[must_use]
    pub fn with_weights(mut self, key: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.weights.insert(key.into(), path.into());
        self
    }

    fn weights_path(&self, key: &str) -> Option<&Path> {
        self.weights.get(key).map(PathBuf::as_path)
    }
}

impl Limits {
    /// Creates tensor limits with a 4 GiB guest-memory default.
    #[must_use]
    pub const fn new(
        max_live_bytes: u64,
        max_tensor_rank: usize,
        max_tensor_elements: u64,
        max_live_tensor_handles: usize,
        max_read_bytes: u64,
    ) -> Self {
        Self {
            live_bytes: max_live_bytes,
            tensor_rank: max_tensor_rank,
            tensor_elements: max_tensor_elements,
            live_tensor_handles: max_live_tensor_handles,
            read_bytes: max_read_bytes,
            guest_memory_bytes: 4 * 1024 * 1024 * 1024,
            table_elements: 10_000,
            instances: 10_000,
            dispatches_per_list: usize::MAX,
            work_per_dispatch: u64::MAX,
            guest_call_timeout: Duration::from_secs(30),
            submission_timeout: Duration::from_secs(10),
            gpu_time_budget: Duration::MAX,
        }
    }

    /// Overrides command recording limits.
    #[must_use]
    pub const fn with_command_limits(
        mut self,
        max_dispatches_per_list: usize,
        max_work_per_dispatch: u64,
    ) -> Self {
        self.dispatches_per_list = max_dispatches_per_list;
        self.work_per_dispatch = max_work_per_dispatch;
        self
    }

    /// Overrides the maximum CPU time allowed for one guest export call.
    #[must_use]
    pub const fn with_guest_call_timeout(mut self, timeout: Duration) -> Self {
        self.guest_call_timeout = timeout;
        self
    }

    /// Overrides the per-submission deadline and cumulative device-time budget.
    ///
    /// Timeouts and exhausted budgets are reported through the guest quota error.
    #[must_use]
    pub const fn with_gpu_limits(
        mut self,
        submission_timeout: Duration,
        gpu_time_budget: Duration,
    ) -> Self {
        self.submission_timeout = submission_timeout;
        self.gpu_time_budget = gpu_time_budget;
        self
    }

    /// Overrides limits applied to WebAssembly store resources.
    #[must_use]
    pub const fn with_store_limits(
        mut self,
        guest_memory_bytes: usize,
        table_elements: usize,
        instances: usize,
    ) -> Self {
        self.guest_memory_bytes = guest_memory_bytes;
        self.table_elements = table_elements;
        self.instances = instances;
        self
    }
}

#[derive(Debug)]
struct BufferHandle {
    owner: Tensor,
    kind: BufferKind,
}

#[derive(Debug)]
enum BufferKind {
    Allocated {
        byte_len: u64,
        live_bytes: Arc<AtomicU64>,
    },
    Weights(Safetensors),
}

impl BufferHandle {
    fn release<B: Backend>(
        &self,
        backend: &B,
        profile: Option<&Arc<Mutex<EngineStepProfile>>>,
    ) -> Result<(), BackendError> {
        let _timer = BackendTimer::start(profile, BackendEvent::Release);
        backend.release(&self.owner)?;
        match &self.kind {
            BufferKind::Allocated {
                byte_len,
                live_bytes,
            } => live_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                    bytes.checked_sub(*byte_len)
                })
                .map(|_| ())
                .map_err(|_| BackendError::ExecutionFailed),
            BufferKind::Weights(_) => Ok(()),
        }
    }

    fn weights(&self) -> Option<&Safetensors> {
        match &self.kind {
            BufferKind::Allocated { .. } => None,
            BufferKind::Weights(source) => Some(source),
        }
    }
}

/// Host-owned state behind a guest tensor resource.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    tensor: Tensor,
    buffer: Arc<BufferHandle>,
}

/// Host-owned state behind a guest weights resource.
#[derive(Debug)]
pub struct WeightsEntry {
    buffer: Arc<BufferHandle>,
}

/// Host-owned state behind a guest command-list resource.
#[derive(Debug)]
pub struct CommandListEntry {
    commands: CommandList,
    retained: Vec<TensorEntry>,
}

/// Store state implementing the guest compute interface over a trusted backend.
pub struct Host<B: Backend> {
    backend: Arc<B>,
    table: ResourceTable,
    wasi: WasiCtx,
    limits: Limits,
    grants: Grants,
    weight_files: HashMap<String, Weak<BufferHandle>>,
    store_limits: StoreLimits,
    live_bytes: Arc<AtomicU64>,
    live_handles: usize,
    gpu_time_ns: Arc<AtomicU64>,
    completed_submissions: Arc<AtomicU64>,
    timed_submissions: Arc<AtomicU64>,
    completed_gpu_time_ns: Arc<AtomicU64>,
    active_profile: Option<Arc<Mutex<EngineStepProfile>>>,
    epoch_registration: Option<Arc<()>>,
}

impl<B: Backend> Host<B> {
    fn new(backend: B, limits: Limits) -> Self {
        Self::with_grants(backend, limits, Grants::new())
    }

    fn with_grants(backend: B, limits: Limits, grants: Grants) -> Self {
        let store_limits = StoreLimitsBuilder::new()
            .memory_size(limits.guest_memory_bytes)
            .table_elements(limits.table_elements)
            .instances(limits.instances)
            .build();
        Self {
            backend: Arc::new(backend),
            table: ResourceTable::new(),
            wasi: WasiCtxBuilder::new().build(),
            limits,
            grants,
            weight_files: HashMap::new(),
            store_limits,
            live_bytes: Arc::new(AtomicU64::new(0)),
            live_handles: 0,
            gpu_time_ns: Arc::new(AtomicU64::new(0)),
            completed_submissions: Arc::new(AtomicU64::new(0)),
            timed_submissions: Arc::new(AtomicU64::new(0)),
            completed_gpu_time_ns: Arc::new(AtomicU64::new(0)),
            active_profile: None,
            epoch_registration: None,
        }
    }

    /// Creates a Wasmtime store with the configured resource limiter installed.
    #[must_use]
    pub fn new_store(engine: &Engine, backend: B, limits: Limits) -> Store<Self>
    where
        B: Send + 'static,
    {
        let mut store = Store::new(engine, Self::new(backend, limits));
        store.data_mut().epoch_registration = Some(register_epoch_engine(engine));
        store.limiter(|host| &mut host.store_limits);
        Self::reset_guest_deadline(&mut store);
        store
    }

    /// Creates a Wasmtime store with host-configured capabilities and resource limits.
    #[must_use]
    pub fn new_store_with_grants(
        engine: &Engine,
        backend: B,
        limits: Limits,
        grants: Grants,
    ) -> Store<Self>
    where
        B: Send + 'static,
    {
        let mut store = Store::new(engine, Self::with_grants(backend, limits, grants));
        store.data_mut().epoch_registration = Some(register_epoch_engine(engine));
        store.limiter(|host| &mut host.store_limits);
        Self::reset_guest_deadline(&mut store);
        store
    }

    /// Starts a fresh CPU-time budget for the next guest export call.
    pub fn reset_guest_deadline(store: &mut Store<Self>) {
        let ticks = epoch_ticks(store.data().limits.guest_call_timeout);
        store.set_epoch_deadline(ticks);
    }

    fn engine_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            submissions: self.completed_submissions.load(Ordering::Acquire),
            timed_submissions: self.timed_submissions.load(Ordering::Acquire),
            gpu_time: Duration::from_nanos(self.completed_gpu_time_ns.load(Ordering::Acquire)),
        }
    }

    fn begin_profile_step(&mut self) {
        self.active_profile = Some(Arc::new(Mutex::new(EngineStepProfile::default())));
    }

    fn finish_profile_step(&mut self, wall_time: Duration) -> Option<EngineStepProfile> {
        let profile = self.active_profile.take()?;
        let mut profile = profile.lock().ok()?.clone();
        profile.wall_time = wall_time;
        profile.guest_time = wall_time.saturating_sub(profile.imports.total_time());
        Some(profile)
    }

    fn import_timer(&self, kind: ImportKind) -> ImportTimer {
        ImportTimer::start(self.active_profile.clone(), kind)
    }

    /// Allocates a contiguous tensor after enforcing all guest quotas.
    ///
    /// # Errors
    ///
    /// Returns a quota or backend error without creating a guest handle.
    pub fn alloc(
        &mut self,
        dtype: compute::Dtype,
        shape: &[u32],
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        let dtype = core_dtype(dtype);
        let byte_len = self.check_allocation(dtype, shape)?;
        let timer = BackendTimer::start(self.active_profile.as_ref(), BackendEvent::Allocation);
        let tensor = self.backend.alloc(dtype, shape).map_err(guest_error)?;
        drop(timer);
        let entry = TensorEntry {
            tensor: tensor.clone(),
            buffer: Arc::new(BufferHandle {
                owner: tensor.clone(),
                kind: BufferKind::Allocated {
                    byte_len,
                    live_bytes: Arc::clone(&self.live_bytes),
                },
            }),
        };
        let resource = match self.table.push(entry) {
            Ok(resource) => resource,
            Err(error) => {
                self.backend.release(&tensor).map_err(guest_error)?;
                return Err(invalid_handle(error));
            }
        };
        self.live_bytes.fetch_add(byte_len, Ordering::AcqRel);
        self.live_handles += 1;
        Ok(resource)
    }

    /// Creates a validated metadata-only tensor view.
    ///
    /// # Errors
    ///
    /// Returns a layout, quota, backend, or invalid-handle error.
    pub fn view(
        &mut self,
        resource: &Resource<TensorEntry>,
        operation: compute::ViewOp,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.entry(resource)?.clone();
        let operation = core_view(operation).map_err(guest_error)?;
        let layout = validate_view(&entry.tensor, &operation).map_err(guest_error)?;
        self.check_tensor_shape(layout.shape())?;
        let tensor = self
            .backend
            .view(&entry.tensor, operation)
            .map_err(guest_error)?;
        let view = self
            .table
            .push(TensorEntry {
                tensor,
                buffer: Arc::clone(&entry.buffer),
            })
            .map_err(invalid_handle)?;
        self.live_handles += 1;
        Ok(view)
    }

    fn open_weights(&mut self, grant: &str) -> Result<Resource<WeightsEntry>, compute::Error> {
        self.check_handle_quota()?;
        let path = self
            .grants
            .weights_path(grant)
            .ok_or_else(|| invalid_handle("weight grant is not configured"))?;
        let buffer = if let Some(buffer) = self.weight_files.get(grant).and_then(Weak::upgrade) {
            buffer
        } else {
            let source = Safetensors::open(path).map_err(weight_error)?;
            let first = source.tensors().first().ok_or_else(|| {
                compute::Error::Layout("weight file contains no tensors".to_owned())
            })?;
            let region = source.mapped_region().map_err(weight_error)?;
            let buffer_len = u64::try_from(region.len())
                .map_err(|_| guest_error(BackendError::AllocationFailed))?;
            let owner_layout = first.layout(buffer_len).map_err(guest_error)?;
            let buffer_id = self.backend.import_readonly(region).map_err(guest_error)?;
            let owner = self
                .backend
                .tensor(buffer_id, owner_layout)
                .map_err(guest_error)?;
            let buffer = Arc::new(BufferHandle {
                owner,
                kind: BufferKind::Weights(source),
            });
            self.weight_files
                .insert(grant.to_owned(), Arc::downgrade(&buffer));
            buffer
        };
        match self.table.push(WeightsEntry {
            buffer: Arc::clone(&buffer),
        }) {
            Ok(resource) => {
                self.live_handles += 1;
                Ok(resource)
            }
            Err(error) => {
                release_buffer(self.backend.as_ref(), buffer, self.active_profile.as_ref())
                    .map_err(guest_error)?;
                Err(invalid_handle(error))
            }
        }
    }

    fn weight_tensor(
        &mut self,
        resource: &Resource<WeightsEntry>,
        name: &str,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        let source = entry
            .buffer
            .weights()
            .ok_or_else(|| invalid_handle("weight handle has no metadata"))?;
        let metadata = source
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == name)
            .ok_or_else(|| invalid_handle("weight tensor is not present"))?;
        self.check_tensor_shape(metadata.shape())?;
        let layout = metadata
            .layout(entry.buffer.owner.buffer().byte_len())
            .map_err(guest_error)?;
        let tensor = self
            .backend
            .tensor(entry.buffer.owner.buffer(), layout)
            .map_err(guest_error)?;
        let buffer = Arc::clone(&entry.buffer);
        let tensor = self
            .table
            .push(TensorEntry { tensor, buffer })
            .map_err(invalid_handle)?;
        self.live_handles += 1;
        Ok(tensor)
    }

    fn weight_info(
        &self,
        resource: &Resource<WeightsEntry>,
        name: &str,
    ) -> Result<compute::TensorInfo, compute::Error> {
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        let source = entry
            .buffer
            .weights()
            .ok_or_else(|| invalid_handle("weight handle has no metadata"))?;
        let tensor = source
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == name)
            .ok_or_else(|| invalid_handle("weight tensor is not present"))?;
        Ok(compute::TensorInfo {
            dtype: guest_dtype(tensor.dtype()),
            shape: tensor.shape().to_vec(),
        })
    }

    fn weight_names(
        &self,
        resource: &Resource<WeightsEntry>,
    ) -> Result<Vec<String>, wasmtime::Error> {
        let entry = self.table.get(resource).map_err(wasmtime::Error::msg)?;
        let source = entry
            .buffer
            .weights()
            .ok_or_else(|| wasmtime::Error::msg("weight handle has no metadata"))?;
        Ok(source
            .tensors()
            .iter()
            .map(|tensor| tensor.name().to_owned())
            .collect())
    }

    fn drop_weights(&mut self, resource: Resource<WeightsEntry>) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        let live_handles = self
            .live_handles
            .checked_sub(1)
            .ok_or_else(|| invalid_handle("live handle accounting underflowed"))?;
        release_buffer(
            self.backend.as_ref(),
            entry.buffer,
            self.active_profile.as_ref(),
        )
        .map_err(guest_error)?;
        self.live_handles = live_handles;
        Ok(())
    }

    /// Writes bytes to a contiguous tensor with an exact logical byte count.
    ///
    /// # Errors
    ///
    /// Returns a layout, backend, or invalid-handle error.
    pub fn write(
        &self,
        resource: &Resource<TensorEntry>,
        bytes: &[u8],
    ) -> Result<(), compute::Error> {
        let tensor = &self.entry(resource)?.tensor;
        if !tensor.layout().is_contiguous() {
            return Err(compute::Error::Layout(
                "writes require a contiguous tensor".to_owned(),
            ));
        }
        let expected = tensor
            .layout()
            .element_count()
            .checked_mul(tensor.layout().dtype().byte_size())
            .ok_or_else(|| compute::Error::Layout("tensor byte size overflowed".to_owned()))?;
        if u64::try_from(bytes.len()) != Ok(expected) {
            return Err(compute::Error::Layout(format!(
                "write has {} bytes but tensor requires {expected}",
                bytes.len()
            )));
        }
        self.backend.write(tensor, bytes).map_err(guest_error)
    }

    /// Creates an empty command list.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle error if the resource table refuses the entry.
    pub fn command_list(&mut self) -> Result<Resource<CommandListEntry>, compute::Error> {
        self.table
            .push(CommandListEntry {
                commands: CommandList::new(),
                retained: Vec::new(),
            })
            .map_err(invalid_handle)
    }

    /// Validates and records one trusted operation.
    ///
    /// # Errors
    ///
    /// Returns an operation, quota, or invalid-handle error before recording invalid work.
    pub fn dispatch(
        &mut self,
        commands: &Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: &[Resource<TensorEntry>],
        output: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        let input_entries = inputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output_entry = self.entry(output)?.clone();
        let input_tensors = input_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let operation = core_op(operation);
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.commands.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        self.check_dispatch_work(operation, &input_tensors, &output_entry.tensor)?;

        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry
            .commands
            .dispatch(operation, &input_tensors, &output_entry.tensor)
            .map_err(guest_error)?;
        entry.retained.extend(input_entries);
        entry.retained.push(output_entry);
        Ok(())
    }

    /// Validates and records one guest-authored scalar program.
    ///
    /// # Errors
    ///
    /// Returns an operation, quota, or invalid-handle error before recording invalid work.
    pub fn dispatch_program(
        &mut self,
        commands: &Resource<CommandListEntry>,
        program: compute::Program,
        inputs: &[Resource<TensorEntry>],
        outputs: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        let input_entries = inputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output_entries = outputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let program = core_program(program)?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.commands.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        let input_tensors = input_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let output_tensors = output_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry
            .commands
            .dispatch_program(&program, &input_tensors, &output_tensors)
            .map_err(guest_error)?;
        entry.retained.extend(input_entries);
        entry.retained.extend(output_entries);
        Ok(())
    }

    fn drop_command_list(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        self.release_retained(entry.retained)
    }

    fn release_retained(&self, entries: Vec<TensorEntry>) -> Result<(), compute::Error> {
        release_retained(self.backend.as_ref(), entries, self.active_profile.as_ref())
            .map_err(guest_error)
    }

    fn prepare_submit(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<SubmitRequest<B>, compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        Ok(SubmitRequest {
            backend: Arc::clone(&self.backend),
            commands: entry.commands,
            retained: entry.retained,
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
            completed_submissions: Arc::clone(&self.completed_submissions),
            timed_submissions: Arc::clone(&self.timed_submissions),
            completed_gpu_time_ns: Arc::clone(&self.completed_gpu_time_ns),
            profile: self.active_profile.clone(),
        })
    }

    /// Drops a guest tensor handle and releases its buffer after the last view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle error, or a backend error on final release.
    pub fn drop_tensor(&mut self, resource: Resource<TensorEntry>) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        let live_handles = self
            .live_handles
            .checked_sub(1)
            .ok_or_else(|| invalid_handle("live handle accounting underflowed"))?;
        release_buffer(
            self.backend.as_ref(),
            entry.buffer,
            self.active_profile.as_ref(),
        )
        .map_err(guest_error)?;
        self.live_handles = live_handles;
        Ok(())
    }

    fn entry(&self, resource: &Resource<TensorEntry>) -> Result<&TensorEntry, compute::Error> {
        self.table.get(resource).map_err(invalid_handle)
    }

    fn check_allocation(&self, dtype: DType, shape: &[u32]) -> Result<u64, compute::Error> {
        self.check_handle_quota()?;
        let elements = self.check_tensor_shape(shape)?;
        let byte_len = elements
            .checked_mul(dtype.byte_size())
            .ok_or_else(|| quota("tensor byte size exceeds the guest limit"))?;
        if self
            .live_bytes
            .load(Ordering::Acquire)
            .checked_add(byte_len)
            .is_none_or(|bytes| bytes > self.limits.live_bytes)
        {
            return Err(quota("live tensor bytes exceed the guest limit"));
        }
        Ok(byte_len)
    }

    fn check_tensor_shape(&self, shape: &[u32]) -> Result<u64, compute::Error> {
        if shape.len() > self.limits.tensor_rank {
            return Err(quota("tensor rank exceeds the guest limit"));
        }
        let elements = shape
            .iter()
            .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
        let Some(elements) = elements else {
            return Err(quota("tensor element count exceeds the guest limit"));
        };
        if elements == 0 {
            return Err(compute::Error::Layout(
                "tensor must contain at least one element".to_owned(),
            ));
        }
        if elements > self.limits.tensor_elements {
            return Err(quota("tensor element count exceeds the guest limit"));
        }
        Ok(elements)
    }

    fn check_handle_quota(&self) -> Result<(), compute::Error> {
        if self.live_handles >= self.limits.live_tensor_handles {
            return Err(quota("live tensor handles exceed the guest limit"));
        }
        Ok(())
    }

    fn check_dispatch_work(
        &self,
        operation: Op,
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Result<(), compute::Error> {
        let mut work = self.check_tensor_shape(output.layout().shape())?;
        for input in inputs {
            work = work
                .checked_add(self.check_tensor_shape(input.layout().shape())?)
                .ok_or_else(dispatch_work_quota)?;
        }
        let flops = match operation {
            Op::Matmul => matmul_flops(inputs),
            Op::Sdpa { .. } => sdpa_flops(inputs),
            _ => Some(0),
        }
        .ok_or_else(dispatch_work_quota)?;
        work = work.checked_add(flops).ok_or_else(dispatch_work_quota)?;
        if work > self.limits.work_per_dispatch {
            return Err(dispatch_work_quota());
        }
        Ok(())
    }

    fn prepare_read(
        &self,
        resource: &Resource<TensorEntry>,
    ) -> Result<ReadRequest<B>, compute::Error> {
        let entry = self.entry(resource)?;
        let byte_len = entry
            .tensor
            .layout()
            .element_count()
            .checked_mul(entry.tensor.layout().dtype().byte_size())
            .ok_or_else(|| quota("read byte size exceeds the guest limit"))?;
        if byte_len > self.limits.read_bytes {
            return Err(quota("read byte size exceeds the guest limit"));
        }
        Ok(ReadRequest {
            backend: Arc::clone(&self.backend),
            tensor: entry.tensor.clone(),
            buffer: Arc::clone(&entry.buffer),
            profile: self.active_profile.clone(),
        })
    }
}

struct ReadRequest<B: Backend> {
    backend: Arc<B>,
    tensor: Tensor,
    buffer: Arc<BufferHandle>,
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
}

struct SubmitRequest<B: Backend> {
    backend: Arc<B>,
    commands: CommandList,
    retained: Vec<TensorEntry>,
    timeout: Duration,
    gpu_time_budget_ns: u64,
    gpu_time_ns: Arc<AtomicU64>,
    completed_submissions: Arc<AtomicU64>,
    timed_submissions: Arc<AtomicU64>,
    completed_gpu_time_ns: Arc<AtomicU64>,
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
}

struct GpuReservation {
    total: Arc<AtomicU64>,
    reserved: u64,
}

impl GpuReservation {
    fn new(total: Arc<AtomicU64>, budget: u64, reserved: u64) -> Result<Self, BackendError> {
        total
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(reserved).filter(|&next| next <= budget)
            })
            .map_err(|_| BackendError::QuotaExceeded)?;
        Ok(Self { total, reserved })
    }

    const fn amount(&self) -> u64 {
        self.reserved
    }

    fn settle(mut self, charged: u64) -> Result<(), BackendError> {
        let charged = charged.min(self.reserved);
        self.total
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_sub(self.reserved)
                    .and_then(|remaining| remaining.checked_add(charged))
            })
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.reserved = 0;
        Ok(())
    }
}

impl Drop for GpuReservation {
    fn drop(&mut self) {
        if self.reserved != 0 {
            let _ = self
                .total
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    current.checked_sub(self.reserved)
                });
        }
    }
}

impl<B> SubmitRequest<B>
where
    B: Backend + Send + Sync + 'static,
{
    async fn run(self) -> Result<Option<u64>, compute::Error> {
        let reservation = GpuReservation::new(
            Arc::clone(&self.gpu_time_ns),
            self.gpu_time_budget_ns,
            duration_ns(self.timeout).max(1),
        );
        tokio::task::spawn_blocking(move || {
            let Self {
                backend,
                commands,
                retained,
                timeout,
                gpu_time_budget_ns: _,
                gpu_time_ns: _,
                completed_submissions,
                timed_submissions,
                completed_gpu_time_ns,
                profile,
            } = self;
            let result = match reservation {
                Err(error) => Err(error),
                Ok(reservation) => match if profile.is_some() {
                    backend.submit_profiled(commands)
                } else {
                    backend.submit(commands)
                } {
                    Err(error) => Err(error),
                    Ok(submission) => {
                        let started = Instant::now();
                        let wait = submission.wait_timeout(timeout);
                        let wall_time = duration_ns(started.elapsed()).max(1);
                        let gpu_time = submission.gpu_time().map(duration_ns);
                        let submission_profile = submission.profile();
                        let charged = if wait == Err(BackendError::Timeout) {
                            reservation.amount()
                        } else {
                            gpu_time.unwrap_or(wall_time).max(1)
                        };
                        let accounting = reservation.settle(charged);
                        match (wait, accounting) {
                            (Err(error), _) | (Ok(()), Err(error)) => Err(error),
                            (Ok(()), Ok(())) => {
                                if let (Some(target), Some(submission_profile)) =
                                    (&profile, submission_profile)
                                    && let Ok(mut target) = target.lock()
                                {
                                    target.submission = Some(submission_profile);
                                }
                                saturating_increment(&completed_submissions);
                                if let Some(gpu_time) = gpu_time {
                                    saturating_increment(&timed_submissions);
                                    saturating_add(&completed_gpu_time_ns, gpu_time);
                                }
                                Ok(gpu_time)
                            }
                        }
                    }
                },
            };
            let release = release_retained(backend.as_ref(), retained, profile.as_ref());
            match (result, release) {
                (Err(error), _) | (Ok(_), Err(error)) => Err(error),
                (Ok(gpu_time), Ok(())) => Ok(gpu_time),
            }
        })
        .await
        .map_err(|_| guest_error(BackendError::ExecutionFailed))?
        .map_err(guest_error)
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
        Some(value.saturating_add(1))
    });
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(value))
    });
}

impl<B> ReadRequest<B>
where
    B: Backend + Send + Sync + 'static,
{
    async fn run(self) -> Result<Vec<u8>, BackendError> {
        tokio::task::spawn_blocking(move || {
            let Self {
                backend,
                tensor,
                buffer,
                profile,
            } = self;
            let result = backend.read(&tensor);
            let release = release_buffer(backend.as_ref(), buffer, profile.as_ref());
            match (result, release) {
                (Err(error), _) | (Ok(_), Err(error)) => Err(error),
                (Ok(bytes), Ok(())) => Ok(bytes),
            }
        })
        .await
        .map_err(|_| BackendError::ExecutionFailed)?
    }
}

impl<B> compute::HostTensor for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn alloc(
        &mut self,
        dtype: compute::Dtype,
        shape: Vec<u32>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<TensorEntry>, compute::Error>>> + Send
    {
        let _timer = self.import_timer(ImportKind::Alloc);
        std::future::ready(Ok(Host::alloc(self, dtype, &shape)))
    }

    fn view(
        &mut self,
        resource: Resource<TensorEntry>,
        operation: compute::ViewOp,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<TensorEntry>, compute::Error>>> + Send
    {
        let kind = match &operation {
            compute::ViewOp::Slice(_) => ImportKind::ViewSlice,
            compute::ViewOp::Reshape(_) => ImportKind::ViewReshape,
            compute::ViewOp::Permute(_) => ImportKind::ViewPermute,
            compute::ViewOp::Broadcast(_) => ImportKind::ViewBroadcast,
        };
        let _timer = self.import_timer(kind);
        std::future::ready(Ok(Host::view(self, &resource, operation)))
    }

    fn write(
        &mut self,
        resource: Resource<TensorEntry>,
        bytes: Vec<u8>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Write);
        std::future::ready(Ok(Host::write(self, &resource, &bytes)))
    }

    fn drop(
        &mut self,
        resource: Resource<TensorEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        let _timer = self.import_timer(ImportKind::ResourceDrop);
        std::future::ready(Host::drop_tensor(self, resource).map_err(wasmtime::Error::msg))
    }
}

impl<B> compute::HostCommandList for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn new(&mut self) -> impl Future<Output = wasmtime::Result<Resource<CommandListEntry>>> + Send {
        let _timer = self.import_timer(ImportKind::CommandList);
        std::future::ready(Host::command_list(self).map_err(wasmtime::Error::msg))
    }

    fn dispatch(
        &mut self,
        resource: Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: Vec<Resource<TensorEntry>>,
        output: Resource<TensorEntry>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(Host::dispatch(
            self, &resource, operation, &inputs, &output,
        )))
    }

    fn dispatch_program(
        &mut self,
        resource: Resource<CommandListEntry>,
        program: compute::Program,
        inputs: Vec<Resource<TensorEntry>>,
        outputs: Vec<Resource<TensorEntry>>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(Host::dispatch_program(
            self, &resource, program, &inputs, &outputs,
        )))
    }

    fn drop(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        let _timer = self.import_timer(ImportKind::ResourceDrop);
        std::future::ready(
            self.drop_command_list(resource)
                .map_err(wasmtime::Error::msg),
        )
    }
}

impl<B> compute::HostWeights for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn info(
        &mut self,
        resource: Resource<WeightsEntry>,
        name: String,
    ) -> impl Future<Output = wasmtime::Result<Result<compute::TensorInfo, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.weight_info(&resource, &name)))
    }

    fn tensor(
        &mut self,
        resource: Resource<WeightsEntry>,
        name: String,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<TensorEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.weight_tensor(&resource, &name)))
    }

    fn names(
        &mut self,
        resource: Resource<WeightsEntry>,
    ) -> impl Future<Output = wasmtime::Result<Vec<String>>> + Send {
        std::future::ready(self.weight_names(&resource))
    }

    fn drop(
        &mut self,
        resource: Resource<WeightsEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(self.drop_weights(resource).map_err(wasmtime::Error::msg))
    }
}

impl<B> compute::Host for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn open_weights(
        &mut self,
        grant: String,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<WeightsEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.open_weights(&grant)))
    }
}

impl<B> WasiView for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

/// Adds the WASI Preview 2 imports emitted by Rust guests.
///
/// # Errors
///
/// Returns an error if a WASI import cannot be defined on the linker.
pub fn add_wasi_to_linker<B>(linker: &mut Linker<Host<B>>) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    wasmtime_wasi::p2::add_to_linker_async(linker)
}

struct HostBindings<B>(std::marker::PhantomData<B>);

impl<B: Backend + 'static> HasData for HostBindings<B> {
    type Data<'a> = &'a mut Host<B>;
}

impl<B> compute::HostTensorWithStore<Host<B>> for HostBindings<B>
where
    B: Backend + Send + Sync + 'static,
{
    async fn read(
        accessor: &Accessor<Host<B>, Self>,
        resource: Resource<TensorEntry>,
    ) -> wasmtime::Result<Result<Vec<u8>, compute::Error>> {
        let timer = accessor.with(|mut access| access.get().import_timer(ImportKind::Read));
        let request = accessor.with(|mut access| access.get().prepare_read(&resource));
        let request = match request {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let result = Ok(request.run().await.map_err(guest_error));
        drop(timer);
        result
    }
}

impl<B> compute::HostWithStore<Host<B>> for HostBindings<B>
where
    B: Backend + Send + Sync + 'static,
{
    async fn submit(
        accessor: &Accessor<Host<B>, Self>,
        resource: Resource<CommandListEntry>,
    ) -> wasmtime::Result<Result<Option<u64>, compute::Error>> {
        let timer = accessor.with(|mut access| access.get().import_timer(ImportKind::Submit));
        let request = accessor.with(|mut access| access.get().prepare_submit(resource));
        let request = match request {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let result = Ok(request.run().await);
        drop(timer);
        result
    }
}

/// Adds compute and WASI Preview 2 imports to a component linker.
///
/// # Errors
///
/// Returns an error if an import cannot be defined on the linker.
pub fn add_to_linker<B>(linker: &mut Linker<Host<B>>) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    add_wasi_to_linker(linker)?;
    bindings::Host_::add_to_linker::<Host<B>, HostBindings<B>>(linker, |host| host)
}

fn add_engine_to_linker<B>(linker: &mut Linker<Host<B>>) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    add_wasi_to_linker(linker)?;
    engine_bindings::EngineComponent::add_to_linker::<Host<B>, HostBindings<B>>(linker, |host| host)
}

fn core_dtype(dtype: compute::Dtype) -> DType {
    match dtype {
        compute::Dtype::F32 => DType::F32,
        compute::Dtype::F16 => DType::F16,
        compute::Dtype::Bf16 => DType::BF16,
        compute::Dtype::I32 => DType::I32,
        compute::Dtype::U32 => DType::U32,
    }
}

fn guest_dtype(dtype: DType) -> compute::Dtype {
    match dtype {
        DType::F32 => compute::Dtype::F32,
        DType::F16 => compute::Dtype::F16,
        DType::BF16 => compute::Dtype::Bf16,
        DType::I32 => compute::Dtype::I32,
        DType::U32 => compute::Dtype::U32,
    }
}

fn core_view(operation: compute::ViewOp) -> Result<ViewOp, LayoutError> {
    Ok(match operation {
        compute::ViewOp::Slice(specs) => ViewOp::Slice(
            specs
                .into_iter()
                .map(|spec| Slice::new(spec.start, spec.len, spec.step))
                .collect::<Result<_, _>>()?,
        ),
        compute::ViewOp::Reshape(shape) => ViewOp::Reshape(shape),
        compute::ViewOp::Permute(axes) => ViewOp::Permute(axes),
        compute::ViewOp::Broadcast(shape) => ViewOp::Broadcast(shape),
    })
}

fn core_op(operation: compute::Op) -> Op {
    match operation {
        compute::Op::Copy => Op::Copy,
        compute::Op::Add => Op::Add,
        compute::Op::SiluMul => Op::SiluMul,
        compute::Op::RmsNorm(eps) => Op::RmsNorm { eps },
        compute::Op::Softmax => Op::Softmax,
        compute::Op::Rope(config) => Op::Rope {
            theta: config.theta,
        },
        compute::Op::Embed => Op::Embed,
        compute::Op::Matmul => Op::Matmul,
        compute::Op::Sdpa(config) => Op::Sdpa {
            scale: config.scale,
            causal: config.causal,
            q_start: config.q_start,
        },
    }
}

fn core_program(program: compute::Program) -> Result<ValidatedProgram, compute::Error> {
    if program.insts.len() > MAX_INSTRUCTIONS {
        return Err(program_error(&ProgramError::TooManyInstructions));
    }
    if program.outputs.len() > MAX_OUTPUTS {
        return Err(program_error(&ProgramError::TooManyOutputs));
    }
    Program {
        kind: match program.kind {
            compute::ProgramKind::Map => ProgramKind::Map,
            compute::ProgramKind::Row => ProgramKind::Row,
        },
        insts: program.insts.into_iter().map(core_inst).collect(),
        outputs: program.outputs,
    }
    .validate()
    .map_err(|error| program_error(&error))
}

fn core_inst(instruction: compute::Inst) -> Inst {
    match instruction {
        compute::Inst::Input(slot) => Inst::Input(slot),
        compute::Inst::Const(value) => Inst::Const(value),
        compute::Inst::Index(axis) => Inst::Index(axis),
        compute::Inst::Extent(axis) => Inst::Extent(axis),
        compute::Inst::Unary((op, value)) => Inst::Unary(core_unop(op), value),
        compute::Inst::Binary((op, left, right)) => Inst::Binary(core_binop(op), left, right),
        compute::Inst::Select((condition, accepted, rejected)) => {
            Inst::Select(condition, accepted, rejected)
        }
        compute::Inst::Cast((to, value)) => Inst::Cast(core_value_type(to), value),
        compute::Inst::Reduce((op, value)) => Inst::Reduce(core_redop(op), value),
    }
}

fn core_unop(op: compute::Unop) -> UnOp {
    match op {
        compute::Unop::Neg => UnOp::Neg,
        compute::Unop::Abs => UnOp::Abs,
        compute::Unop::Exp => UnOp::Exp,
        compute::Unop::Log => UnOp::Log,
        compute::Unop::Sqrt => UnOp::Sqrt,
        compute::Unop::Rsqrt => UnOp::Rsqrt,
        compute::Unop::Sin => UnOp::Sin,
        compute::Unop::Cos => UnOp::Cos,
        compute::Unop::Tanh => UnOp::Tanh,
        compute::Unop::Sigmoid => UnOp::Sigmoid,
        compute::Unop::Recip => UnOp::Recip,
        compute::Unop::Floor => UnOp::Floor,
    }
}

fn core_binop(op: compute::Binop) -> BinOp {
    match op {
        compute::Binop::Add => BinOp::Add,
        compute::Binop::Sub => BinOp::Sub,
        compute::Binop::Mul => BinOp::Mul,
        compute::Binop::Div => BinOp::Div,
        compute::Binop::Min => BinOp::Min,
        compute::Binop::Max => BinOp::Max,
        compute::Binop::Pow => BinOp::Pow,
        compute::Binop::Lt => BinOp::Lt,
        compute::Binop::Le => BinOp::Le,
        compute::Binop::Eq => BinOp::Eq,
        compute::Binop::Ne => BinOp::Ne,
        compute::Binop::Ge => BinOp::Ge,
        compute::Binop::Gt => BinOp::Gt,
    }
}

const fn core_redop(op: compute::Redop) -> RedOp {
    match op {
        compute::Redop::Sum => RedOp::Sum,
        compute::Redop::Max => RedOp::Max,
        compute::Redop::Min => RedOp::Min,
    }
}

const fn core_value_type(value_type: compute::ValueType) -> ValueType {
    match value_type {
        compute::ValueType::F32 => ValueType::F32,
        compute::ValueType::U32 => ValueType::U32,
        compute::ValueType::Bool => ValueType::Bool,
    }
}

fn program_error(error: &ProgramError) -> compute::Error {
    let message = match error {
        ProgramError::OperandNotEarlier { instruction, .. }
        | ProgramError::NonFiniteConstant { instruction }
        | ProgramError::AxisOutOfRange { instruction, .. }
        | ProgramError::ReduceInMap { instruction }
        | ProgramError::InvalidUnaryType { instruction }
        | ProgramError::InvalidBinaryType { instruction }
        | ProgramError::InvalidSelectCondition { instruction }
        | ProgramError::InvalidSelectBranch { instruction }
        | ProgramError::InvalidReduceType { instruction } => {
            format!("instruction {instruction}: {error}")
        }
        ProgramError::InvalidOutputValue { instruction, .. } => {
            format!("instruction {instruction}: {error}")
        }
        _ => error.to_string(),
    };
    compute::Error::OpSignature(message)
}

fn matmul_flops(inputs: &[&Tensor]) -> Option<u64> {
    let left = inputs.first()?.layout().shape();
    let right = inputs.get(1)?.layout().shape();
    let batch = if left.len() == 3 {
        u64::from(*left.first()?)
    } else {
        1
    };
    let rank = left.len();
    batch
        .checked_mul(u64::from(*left.get(rank.checked_sub(2)?)?))?
        .checked_mul(u64::from(*left.last()?))?
        .checked_mul(u64::from(*right.last()?))?
        .checked_mul(2)
}

fn sdpa_flops(inputs: &[&Tensor]) -> Option<u64> {
    let query = inputs.first()?.layout().shape();
    let key = inputs.get(1)?.layout().shape();
    let value = inputs.get(2)?.layout().shape();
    u64::from(*query.first()?)
        .checked_mul(u64::from(*query.get(1)?))?
        .checked_mul(u64::from(*key.get(1)?))?
        .checked_mul(u64::from(*query.get(2)?).checked_add(u64::from(*value.get(2)?))?)?
        .checked_mul(2)
}

fn validate_view(tensor: &Tensor, operation: &ViewOp) -> Result<Layout, LayoutError> {
    match operation {
        ViewOp::Slice(specs) => tensor.layout().slice(specs),
        ViewOp::Reshape(shape) => tensor.layout().reshape(shape),
        ViewOp::Permute(axes) => tensor.layout().permute(axes),
        ViewOp::Broadcast(shape) => tensor.layout().broadcast(shape),
    }
}

fn release_retained<B: Backend>(
    backend: &B,
    entries: Vec<TensorEntry>,
    profile: Option<&Arc<Mutex<EngineStepProfile>>>,
) -> Result<(), BackendError> {
    let mut failure = None;
    for entry in entries {
        if let Err(error) = release_buffer(backend, entry.buffer, profile)
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), Err)
}

fn release_buffer<B: Backend>(
    backend: &B,
    buffer: Arc<BufferHandle>,
    profile: Option<&Arc<Mutex<EngineStepProfile>>>,
) -> Result<(), BackendError> {
    Arc::into_inner(buffer).map_or(Ok(()), |buffer| buffer.release(backend, profile))
}

fn quota(message: &str) -> compute::Error {
    compute::Error::Quota(message.to_owned())
}

fn guest_timeout() -> compute::Error {
    quota("guest call exceeded the CPU time limit")
}

fn is_epoch_timeout(error: &wasmtime::Error) -> bool {
    error.downcast_ref::<Trap>() == Some(&Trap::Interrupt)
}

fn dispatch_work_quota() -> compute::Error {
    quota("dispatch work exceeds the guest limit")
}

fn invalid_handle(error: impl ToString) -> compute::Error {
    let message = error.to_string();
    drop(error);
    compute::Error::InvalidHandle(message)
}

fn weight_error(error: WeightError) -> compute::Error {
    let message = error.to_string();
    drop(error);
    compute::Error::Layout(message)
}

fn guest_error(error: impl Into<GuestFailure>) -> compute::Error {
    match error.into() {
        GuestFailure::Layout(error) => compute::Error::Layout(error.to_string()),
        GuestFailure::Op(error) => compute::Error::OpSignature(error.to_string()),
        GuestFailure::Backend(BackendError::QuotaExceeded) => quota("backend quota exceeded"),
        GuestFailure::Backend(BackendError::AllocationFailed) => {
            compute::Error::BackendExecution("backend allocation failed".to_owned())
        }
        GuestFailure::Backend(BackendError::ExecutionFailed) => {
            compute::Error::BackendExecution("backend execution failed".to_owned())
        }
        GuestFailure::Backend(BackendError::UnsupportedOperation) => {
            compute::Error::BackendExecution("backend operation is unsupported".to_owned())
        }
        GuestFailure::Backend(BackendError::Timeout) => {
            quota("backend submission exceeded its time limit")
        }
        GuestFailure::Backend(BackendError::InvalidInput) => {
            invalid_handle("backend rejected the tensor handle")
        }
        GuestFailure::Backend(BackendError::IndexOutOfRange { index }) => {
            compute::Error::OpSignature(format!("backend index {index} is out of range"))
        }
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

enum GuestFailure {
    Layout(LayoutError),
    Op(OpError),
    Backend(BackendError),
}

impl From<LayoutError> for GuestFailure {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<OpError> for GuestFailure {
    fn from(error: OpError) -> Self {
        Self::Op(error)
    }
}

impl From<BackendError> for GuestFailure {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};
    use std::{
        fs,
        sync::{
            Arc, Condvar, Mutex,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        },
    };

    use forja_core::{
        Backend, BackendError, BufferId, CommandList, DType, Layout, LayoutError, MappedRegion,
        OpError, Submission, Tensor, ViewOp as CoreViewOp,
    };
    use forja_cpu::CpuBackend;
    use wasmtime::component::Resource;

    use super::{
        BackendEvent, BackendTimer, EngineMetrics, EngineStepProfile, Grants, Host, ImportKind,
        ImportTimer, Limits, MAX_INSTRUCTIONS, MAX_OUTPUTS, bindings::l9o::gpu::compute,
        core_program,
    };

    const GENEROUS: Limits = Limits::new(u64::MAX, 8, u64::MAX, 32, u64::MAX);
    static NEXT_WEIGHT_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn converts_and_records_wit_programs() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();
        let program = compute::Program {
            kind: compute::ProgramKind::Map,
            insts: vec![
                compute::Inst::Input(0),
                compute::Inst::Const(2.0),
                compute::Inst::Binary((compute::Binop::Mul, 0, 1)),
            ],
            outputs: vec![(0, 2)],
        };

        host.dispatch_program(
            &Resource::new_borrow(commands.rep()),
            program,
            &[Resource::new_borrow(input.rep())],
            &[Resource::new_borrow(output.rep())],
        )
        .unwrap();

        assert!(
            host.table
                .get(&commands)
                .unwrap()
                .commands
                .clone()
                .into_dispatches()[0]
                .bound_program()
                .is_some()
        );
    }

    #[test]
    fn program_validation_errors_name_the_instruction() {
        let error = core_program(compute::Program {
            kind: compute::ProgramKind::Map,
            insts: vec![
                compute::Inst::Const(1.0),
                compute::Inst::Unary((compute::Unop::Exp, 1)),
            ],
            outputs: vec![(0, 0)],
        })
        .unwrap_err();

        assert!(matches!(
            error,
            compute::Error::OpSignature(message) if message.contains("instruction 1")
        ));
    }

    #[test]
    fn refuses_oversized_wit_lists_before_conversion() {
        let instructions = vec![compute::Inst::Const(1.0); MAX_INSTRUCTIONS + 1];
        let error = core_program(compute::Program {
            kind: compute::ProgramKind::Map,
            insts: instructions,
            outputs: vec![(0, 0)],
        })
        .unwrap_err();
        assert!(
            matches!(error, compute::Error::OpSignature(message) if message.contains("TooManyInstructions"))
        );

        let error = core_program(compute::Program {
            kind: compute::ProgramKind::Map,
            insts: vec![compute::Inst::Const(1.0)],
            outputs: vec![(0, 0); MAX_OUTPUTS + 1],
        })
        .unwrap_err();
        assert!(
            matches!(error, compute::Error::OpSignature(message) if message.contains("TooManyOutputs"))
        );
    }

    #[test]
    fn invalid_output_errors_name_the_instruction() {
        let error = core_program(compute::Program {
            kind: compute::ProgramKind::Map,
            insts: vec![compute::Inst::Const(1.0)],
            outputs: vec![(0, 7)],
        })
        .unwrap_err();

        assert!(matches!(
            error,
            compute::Error::OpSignature(message) if message.contains("instruction 7")
        ));
    }

    #[test]
    fn import_timer_records_only_the_selected_kind() {
        let profile = Arc::new(Mutex::new(EngineStepProfile::default()));
        drop(ImportTimer::start(
            Some(Arc::clone(&profile)),
            ImportKind::Dispatch,
        ));
        let profile = profile.lock().unwrap();
        assert_eq!(profile.imports.dispatch.count, 1);
        assert_eq!(profile.imports.alloc.count, 0);
    }

    #[test]
    fn backend_timer_counts_allocations_and_releases() {
        let profile = Arc::new(Mutex::new(EngineStepProfile::default()));
        drop(BackendTimer::start(
            Some(&profile),
            BackendEvent::Allocation,
        ));
        drop(BackendTimer::start(Some(&profile), BackendEvent::Release));
        let profile = profile.lock().unwrap();
        assert_eq!(profile.allocations.count, 1);
        assert_eq!(profile.releases.count, 1);
    }

    #[test]
    fn repeated_weight_opens_share_one_import_and_obey_handle_quota() {
        let path = test_weight_file();
        let imports = Arc::new(AtomicUsize::new(0));
        let releases = Arc::new(AtomicUsize::new(0));
        let backend = ImportCountingBackend {
            inner: CpuBackend::new(),
            imports: Arc::clone(&imports),
            releases: Arc::clone(&releases),
        };
        let limits = Limits::new(u64::MAX, 8, u64::MAX, 8, u64::MAX);
        let grants = Grants::new().with_weights("model", &path);
        let mut host = Host::with_grants(backend, limits, grants);
        let mut handles = Vec::new();
        let mut quota_errors = 0;

        for _ in 0..10_000 {
            match host.open_weights("model") {
                Ok(handle) => handles.push(handle),
                Err(compute::Error::Quota(_)) => quota_errors += 1,
                Err(error) => panic!("unexpected open error: {error:?}"),
            }
        }

        assert_eq!(handles.len(), 8);
        assert_eq!(quota_errors, 9_992);
        assert_eq!(imports.load(Ordering::Acquire), 1);

        host.drop_weights(handles.pop().unwrap()).unwrap();
        let tensor = host
            .weight_tensor(&Resource::new_borrow(handles[0].rep()), "value")
            .unwrap();
        for handle in handles {
            host.drop_weights(handle).unwrap();
        }
        let reopened = host.open_weights("model").unwrap();
        assert_eq!(imports.load(Ordering::Acquire), 1);
        host.drop_weights(reopened).unwrap();
        host.drop_tensor(tensor).unwrap();
        assert_eq!(releases.load(Ordering::Acquire), 1);
        fs::remove_file(path).unwrap();
    }

    fn test_weight_file() -> std::path::PathBuf {
        let mut header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(1.0_f32.to_le_bytes());
        let path = std::env::temp_dir().join(format!(
            "forja-weight-cache-{}-{}",
            std::process::id(),
            NEXT_WEIGHT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    struct ImportCountingBackend {
        inner: CpuBackend,
        imports: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl Backend for ImportCountingBackend {
        type Submission = <CpuBackend as Backend>::Submission;

        fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
            self.inner.alloc(dtype, shape)
        }

        fn import_readonly(&self, bytes: MappedRegion) -> Result<BufferId, BackendError> {
            self.imports.fetch_add(1, Ordering::AcqRel);
            self.inner.import_readonly(bytes)
        }

        fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
            self.inner.tensor(buffer, layout)
        }

        fn view(&self, tensor: &Tensor, op: CoreViewOp) -> Result<Tensor, BackendError> {
            self.inner.view(tensor, op)
        }

        fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
            self.inner.write(tensor, bytes)
        }

        fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
            self.inner.read(tensor)
        }

        fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
            self.releases.fetch_add(1, Ordering::AcqRel);
            self.inner.release(tensor)
        }

        fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
            self.inner.submit(commands)
        }
    }

    #[test]
    fn refuses_each_quota_before_allocation() {
        let cases = [
            (Limits::new(3, 8, u64::MAX, 8, u64::MAX), vec![1]),
            (Limits::new(u64::MAX, 1, u64::MAX, 8, u64::MAX), vec![1, 1]),
            (Limits::new(u64::MAX, 8, 6, 8, u64::MAX), vec![7]),
            (Limits::new(u64::MAX, 8, u64::MAX, 0, u64::MAX), vec![1]),
        ];
        for (limits, shape) in cases {
            let mut host = Host::new(CpuBackend::new(), limits);
            assert!(matches!(
                host.alloc(compute::Dtype::F32, &shape),
                Err(compute::Error::Quota(_))
            ));
        }

        let mut host = Host::new(CpuBackend::new(), Limits::new(3, 8, 1, 1, u64::MAX));
        assert!(matches!(
            host.alloc(compute::Dtype::F32, &[1]),
            Err(compute::Error::Quota(_))
        ));
        host.limits.live_bytes = 4;
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert_eq!(
            host.table
                .get(&tensor)
                .unwrap()
                .tensor
                .buffer()
                .allocation(),
            0
        );
    }

    fn assert_empty_tensors_are_layout_errors<B: Backend>(backend: B) {
        let mut host = Host::new(backend, GENEROUS);
        assert!(matches!(
            host.alloc(compute::Dtype::F32, &[0]),
            Err(compute::Error::Layout(_))
        ));
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert!(matches!(
            host.view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Slice(vec![compute::SliceSpec {
                    start: 0,
                    len: 0,
                    step: 1,
                }]),
            ),
            Err(compute::Error::Layout(_))
        ));
    }

    #[test]
    fn cpu_rejects_empty_tensors_at_the_host_boundary() {
        assert_empty_tensors_are_layout_errors(CpuBackend::new());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_tensor_smoke_rejects_empty_tensors_at_the_host_boundary() {
        assert_empty_tensors_are_layout_errors(forja_metal::MetalBackend::new().unwrap());
    }

    #[test]
    fn refuses_views_over_handle_and_element_limits() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(16, 8, 1, 1, u64::MAX));
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert!(matches!(
            host.view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Reshape(vec![1]),
            ),
            Err(compute::Error::Quota(_))
        ));

        host.limits.live_tensor_handles = 8;
        assert!(matches!(
            host.view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Broadcast(vec![4_000_000_000]),
            ),
            Err(compute::Error::Quota(_))
        ));
    }

    #[test]
    fn writes_require_contiguous_exactly_sized_bytes() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let tensor = host.alloc(compute::Dtype::F32, &[7, 1024]).unwrap();
        assert!(matches!(
            host.write(&tensor, &[]),
            Err(compute::Error::Layout(_))
        ));

        let view = host
            .view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Permute(vec![1, 0]),
            )
            .unwrap();
        assert!(matches!(
            host.write(&view, &vec![0; 7 * 1024 * 4]),
            Err(compute::Error::Layout(_))
        ));
    }

    #[test]
    fn records_every_guest_operation() {
        let cases = [
            (
                compute::Op::Copy,
                vec![(compute::Dtype::F32, vec![7])],
                (compute::Dtype::F32, vec![7]),
            ),
            (
                compute::Op::Add,
                vec![
                    (compute::Dtype::F16, vec![7]),
                    (compute::Dtype::F16, vec![7]),
                ],
                (compute::Dtype::F16, vec![7]),
            ),
            (
                compute::Op::SiluMul,
                vec![
                    (compute::Dtype::Bf16, vec![33]),
                    (compute::Dtype::Bf16, vec![33]),
                ],
                (compute::Dtype::Bf16, vec![33]),
            ),
            (
                compute::Op::RmsNorm(0.00001),
                vec![
                    (compute::Dtype::F32, vec![7, 1024]),
                    (compute::Dtype::F32, vec![1024]),
                ],
                (compute::Dtype::F32, vec![7, 1024]),
            ),
            (
                compute::Op::Softmax,
                vec![(compute::Dtype::F32, vec![7, 33])],
                (compute::Dtype::F32, vec![7, 33]),
            ),
            (
                compute::Op::Rope(compute::RopeCfg { theta: 10_000.0 }),
                vec![
                    (compute::Dtype::F32, vec![7, 16, 128]),
                    (compute::Dtype::U32, vec![7]),
                ],
                (compute::Dtype::F32, vec![7, 16, 128]),
            ),
            (
                compute::Op::Embed,
                vec![
                    (compute::Dtype::F32, vec![33, 128]),
                    (compute::Dtype::U32, vec![7]),
                ],
                (compute::Dtype::F32, vec![7, 128]),
            ),
            (
                compute::Op::Matmul,
                vec![
                    (compute::Dtype::F32, vec![7, 33]),
                    (compute::Dtype::F32, vec![33, 1024]),
                ],
                (compute::Dtype::F32, vec![7, 1024]),
            ),
            (
                compute::Op::Sdpa(compute::SdpaCfg {
                    scale: 0.088,
                    causal: false,
                    q_start: 0,
                }),
                vec![
                    (compute::Dtype::F32, vec![16, 7, 128]),
                    (compute::Dtype::F32, vec![8, 33, 128]),
                    (compute::Dtype::F32, vec![8, 33, 128]),
                ],
                (compute::Dtype::F32, vec![16, 7, 128]),
            ),
        ];

        for (operation, input_specs, (output_dtype, output_shape)) in cases {
            let mut host = Host::new(CpuBackend::new(), GENEROUS);
            let inputs = input_specs
                .iter()
                .map(|(dtype, shape)| host.alloc(*dtype, shape).unwrap())
                .collect::<Vec<_>>();
            let input_borrows = inputs
                .iter()
                .map(|resource| Resource::new_borrow(resource.rep()))
                .collect::<Vec<_>>();
            let output = host.alloc(output_dtype, &output_shape).unwrap();
            let commands = host.command_list().unwrap();

            host.dispatch(&commands, operation, &input_borrows, &output)
                .unwrap();

            assert_eq!(
                host.table
                    .get(&commands)
                    .unwrap()
                    .commands
                    .clone()
                    .into_dispatches()
                    .len(),
                1
            );
        }
    }

    #[test]
    fn invalid_dispatches_preserve_the_command_list() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::RmsNorm(f32::NAN),
                &[Resource::new_borrow(input.rep())],
                &output,
            ),
            Err(compute::Error::OpSignature(_))
        ));
        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &Resource::new_borrow(input.rep()),
            ),
            Err(compute::Error::OpSignature(_))
        ));
        assert!(
            host.table
                .get(&commands)
                .unwrap()
                .commands
                .clone()
                .into_dispatches()
                .is_empty()
        );

        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
        assert!(matches!(
            host.dispatch(
                &Resource::new_borrow(42),
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            ),
            Err(compute::Error::InvalidHandle(_))
        ));
        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(43)],
                &output,
            ),
            Err(compute::Error::InvalidHandle(_))
        ));
    }

    #[test]
    fn refuses_dispatches_after_the_command_count_limit() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        host.limits.dispatches_per_list = 1;
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();

        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            ),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.table.get(&commands).unwrap().commands.len(), 1);
    }

    #[test]
    fn refuses_dispatches_over_the_work_limit() {
        let limits = GENEROUS.with_command_limits(usize::MAX, 13);
        let mut host = Host::new(CpuBackend::new(), limits);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();

        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            ),
            Err(compute::Error::Quota(_))
        ));
        assert!(host.table.get(&commands).unwrap().commands.is_empty());

        host.limits.work_per_dispatch = 14;
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
    }

    #[test]
    fn failed_dispatch_leaves_existing_commands_unchanged() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();

        host.limits.work_per_dispatch = 13;
        assert!(matches!(
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            ),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.table.get(&commands).unwrap().commands.len(), 1);
    }

    #[test]
    fn dispatch_recording_scales_roughly_linearly() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();

        let started = Instant::now();
        for _ in 0..1_000 {
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            )
            .unwrap();
        }
        let first_thousand = started.elapsed();
        let started = Instant::now();
        for _ in 1_000..10_000 {
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &output,
            )
            .unwrap();
        }
        assert!(started.elapsed() < first_thousand.saturating_mul(30));
    }

    #[test]
    fn dropping_a_command_list_releases_unreferenced_buffers() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let input_tensor = host.entry(&input).unwrap().tensor.clone();
        let output_tensor = host.entry(&output).unwrap().tensor.clone();
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();

        host.drop_tensor(input).unwrap();
        host.drop_tensor(output).unwrap();
        assert!(host.backend.read(&input_tensor).is_ok());
        assert!(host.backend.read(&output_tensor).is_ok());

        host.drop_command_list(commands).unwrap();
        assert_eq!(
            host.backend.read(&input_tensor),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            host.backend.read(&output_tensor),
            Err(BackendError::InvalidInput)
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn submit_retains_buffers_until_execution_finishes() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        host.write(&input, &[0; 7 * 4]).unwrap();
        let input_tensor = host.entry(&input).unwrap().tensor.clone();
        let output_tensor = host.entry(&output).unwrap().tensor.clone();
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();

        let request = host.prepare_submit(commands).unwrap();
        host.drop_tensor(input).unwrap();
        host.drop_tensor(output).unwrap();
        assert!(host.backend.read(&input_tensor).is_ok());
        assert!(host.backend.read(&output_tensor).is_ok());

        request.run().await.unwrap();
        assert_eq!(
            host.backend.read(&input_tensor),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            host.backend.read(&output_tensor),
            Err(BackendError::InvalidInput)
        );
    }

    #[test]
    fn views_keep_their_buffer_alive_until_the_last_drop() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let base = host.alloc(compute::Dtype::F32, &[7, 1024]).unwrap();
        let view = host
            .view(
                &Resource::new_borrow(base.rep()),
                compute::ViewOp::Permute(vec![1, 0]),
            )
            .unwrap();
        let tensor = host.entry(&view).unwrap().tensor.clone();

        host.drop_tensor(base).unwrap();
        assert_eq!(host.backend.read(&tensor).unwrap().len(), 7 * 1024 * 4);
        host.drop_tensor(view).unwrap();
        assert_eq!(host.backend.read(&tensor), Err(BackendError::InvalidInput));
    }

    #[test]
    fn maps_every_host_error_kind() {
        assert!(matches!(
            super::guest_error(LayoutError::ZeroSliceStep),
            compute::Error::Layout(_)
        ));
        assert!(matches!(
            super::guest_error(OpError::InvalidScale),
            compute::Error::OpSignature(_)
        ));
        assert!(matches!(
            super::guest_error(BackendError::QuotaExceeded),
            compute::Error::Quota(_)
        ));
        assert!(matches!(
            super::guest_error(BackendError::ExecutionFailed),
            compute::Error::BackendExecution(_)
        ));
        assert!(matches!(
            super::guest_error(BackendError::Timeout),
            compute::Error::Quota(_)
        ));

        let host = Host::new(CpuBackend::new(), GENEROUS);
        assert!(matches!(
            host.entry(&Resource::new_borrow(42)),
            Err(compute::Error::InvalidHandle(_))
        ));
    }

    #[test]
    fn refuses_reads_over_the_byte_limit_before_gathering() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(4, 8, 4097, 8, 4));
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let view = host
            .view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Broadcast(vec![4097]),
            )
            .unwrap();
        assert!(matches!(
            host.prepare_read(&view),
            Err(compute::Error::Quota(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_in_flight_read_keeps_the_buffer_alive() {
        let gate = Arc::new(ReadGate::default());
        let backend = BlockingBackend::new(Arc::clone(&gate));
        let mut host = Host::new(backend, GENEROUS);
        let base = host.alloc(compute::Dtype::U32, &[1]).unwrap();
        host.write(&base, &7_u32.to_le_bytes()).unwrap();
        let view = host
            .view(
                &Resource::new_borrow(base.rep()),
                compute::ViewOp::Reshape(vec![1]),
            )
            .unwrap();
        let read = host.prepare_read(&view).unwrap();
        let task = tokio::spawn(read.run());

        while !gate.started() {
            tokio::task::yield_now().await;
        }
        host.drop_tensor(base).unwrap();
        host.drop_tensor(view).unwrap();
        assert_eq!(gate.releases.load(Ordering::Acquire), 0);

        gate.finish();
        assert_eq!(task.await.unwrap().unwrap(), 7_u32.to_le_bytes());
        assert_eq!(gate.releases.load(Ordering::Acquire), 1);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_tensor_smoke_enforces_gpu_time_budget() {
        let limits = GENEROUS.with_gpu_limits(
            std::time::Duration::from_secs(10),
            std::time::Duration::from_secs(10),
        );
        let backend = forja_metal::MetalBackend::new().unwrap();
        let mut host = Host::new(backend, limits);
        let input = host.alloc(compute::Dtype::F32, &[4097]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[4097]).unwrap();

        let first = host.command_list().unwrap();
        host.dispatch(
            &first,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
        assert!(host.prepare_submit(first).unwrap().run().await.unwrap() > Some(0));

        let later = host.command_list().unwrap();
        host.dispatch(
            &later,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
        assert!(matches!(
            host.prepare_submit(later).unwrap().run().await,
            Err(compute::Error::Quota(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_submits_reserve_their_deadlines() {
        const SUBMITS: usize = 4;
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(
            Some(Arc::clone(&gate)),
            Some(Duration::from_nanos(50)),
            Duration::ZERO,
        );
        let limits = GENEROUS.with_gpu_limits(Duration::from_nanos(100), Duration::from_nanos(300));
        let mut host = Host::new(backend, limits);
        let mut requests = Vec::new();
        for _ in 0..SUBMITS {
            let commands = host.command_list().unwrap();
            requests.push(host.prepare_submit(commands).unwrap());
        }
        let tasks = requests
            .into_iter()
            .map(|request| tokio::spawn(request.run()))
            .collect::<Vec<_>>();

        gate.wait_for(SUBMITS - 1);
        gate.release();
        let mut accepted = 0;
        let mut refused = 0;
        for task in tasks {
            match task.await.unwrap() {
                Ok(_) => accepted += 1,
                Err(compute::Error::Quota(_)) => refused += 1,
                Err(error) => panic!("unexpected submit error: {error:?}"),
            }
        }
        assert_eq!(accepted, SUBMITS - 1);
        assert_eq!(refused, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cumulative_gpu_charge_never_exceeds_the_budget() {
        let backend = AccountingBackend::new(None, Some(Duration::from_nanos(30)), Duration::ZERO);
        let limits =
            GENEROUS.with_gpu_limits(Duration::from_nanos(100), Duration::from_nanos(1_000));
        let mut host = Host::new(backend, limits);
        let mut accepted = 0;
        let mut refused = 0;
        for _ in 0..64 {
            let commands = host.command_list().unwrap();
            match host.prepare_submit(commands).unwrap().run().await {
                Ok(_) => accepted += 1,
                Err(compute::Error::Quota(_)) => refused += 1,
                Err(error) => panic!("unexpected submit error: {error:?}"),
            }
            assert!(host.gpu_time_ns.load(Ordering::Acquire) <= 1_000);
        }
        assert!(accepted > 10);
        assert!(refused > 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn missing_gpu_timestamps_charge_wall_time() {
        let backend = AccountingBackend::new(None, None, Duration::from_millis(1));
        let limits = GENEROUS.with_gpu_limits(Duration::from_secs(1), Duration::from_secs(2));
        let mut host = Host::new(backend, limits);
        let commands = host.command_list().unwrap();
        assert_eq!(
            host.prepare_submit(commands).unwrap().run().await.unwrap(),
            None
        );
        assert!(host.gpu_time_ns.load(Ordering::Acquire) > 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn completed_submissions_report_device_time() {
        let backend = AccountingBackend::new(None, Some(Duration::from_nanos(33)), Duration::ZERO);
        let mut host = Host::new(backend, GENEROUS);
        for _ in 0..2 {
            let commands = host.command_list().unwrap();
            host.prepare_submit(commands).unwrap().run().await.unwrap();
        }
        assert_eq!(
            host.engine_metrics(),
            EngineMetrics {
                submissions: 2,
                timed_submissions: 2,
                gpu_time: Duration::from_nanos(66),
            }
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn timed_out_submissions_charge_the_full_deadline() {
        let backend = AccountingBackend::timing_out();
        let limits =
            GENEROUS.with_gpu_limits(Duration::from_nanos(100), Duration::from_nanos(1_000));
        let mut host = Host::new(backend, limits);
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.prepare_submit(commands).unwrap().run().await,
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.gpu_time_ns.load(Ordering::Acquire), 100);
    }

    #[derive(Default)]
    struct ReadGate {
        state: Mutex<(bool, bool)>,
        ready: Condvar,
        releases: AtomicUsize,
    }

    impl ReadGate {
        fn started(&self) -> bool {
            self.state.lock().unwrap().0
        }

        fn wait(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 = true;
            self.ready.notify_all();
            while !state.1 {
                state = self.ready.wait(state).unwrap();
            }
        }

        fn finish(&self) {
            self.state.lock().unwrap().1 = true;
            self.ready.notify_all();
        }
    }

    struct BlockingBackend {
        inner: CpuBackend,
        gate: Arc<ReadGate>,
    }

    #[derive(Default)]
    struct SubmitGate {
        state: Mutex<(usize, bool)>,
        ready: Condvar,
    }

    impl SubmitGate {
        fn wait(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            self.ready.notify_all();
            while !state.1 {
                state = self.ready.wait(state).unwrap();
            }
        }

        fn wait_for(&self, count: usize) {
            let mut state = self.state.lock().unwrap();
            while state.0 < count {
                state = self.ready.wait(state).unwrap();
            }
        }

        fn release(&self) {
            self.state.lock().unwrap().1 = true;
            self.ready.notify_all();
        }
    }

    struct AccountingBackend {
        inner: CpuBackend,
        gate: Option<Arc<SubmitGate>>,
        gpu_time: Option<Duration>,
        delay: Duration,
        wait_error: Option<BackendError>,
    }

    impl AccountingBackend {
        fn new(gate: Option<Arc<SubmitGate>>, gpu_time: Option<Duration>, delay: Duration) -> Self {
            Self {
                inner: CpuBackend::new(),
                gate,
                gpu_time,
                delay,
                wait_error: None,
            }
        }

        fn timing_out() -> Self {
            Self {
                inner: CpuBackend::new(),
                gate: None,
                gpu_time: None,
                delay: Duration::ZERO,
                wait_error: Some(BackendError::Timeout),
            }
        }
    }

    struct AccountingSubmission {
        gate: Option<Arc<SubmitGate>>,
        gpu_time: Option<Duration>,
        delay: Duration,
        wait_error: Option<BackendError>,
    }

    impl Submission for AccountingSubmission {
        fn wait(&self) -> Result<(), BackendError> {
            if let Some(gate) = &self.gate {
                gate.wait();
            }
            std::thread::sleep(self.delay);
            self.wait_error.map_or(Ok(()), Err)
        }

        fn gpu_time(&self) -> Option<Duration> {
            self.gpu_time
        }
    }

    impl Backend for AccountingBackend {
        type Submission = AccountingSubmission;

        fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
            self.inner.alloc(dtype, shape)
        }

        fn view(&self, tensor: &Tensor, op: CoreViewOp) -> Result<Tensor, BackendError> {
            self.inner.view(tensor, op)
        }

        fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
            self.inner.write(tensor, bytes)
        }

        fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
            self.inner.read(tensor)
        }

        fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
            self.inner.release(tensor)
        }

        fn submit(&self, _commands: CommandList) -> Result<Self::Submission, BackendError> {
            Ok(AccountingSubmission {
                gate: self.gate.clone(),
                gpu_time: self.gpu_time,
                delay: self.delay,
                wait_error: self.wait_error,
            })
        }
    }

    impl BlockingBackend {
        fn new(gate: Arc<ReadGate>) -> Self {
            Self {
                inner: CpuBackend::new(),
                gate,
            }
        }
    }

    impl Backend for BlockingBackend {
        type Submission = <CpuBackend as Backend>::Submission;

        fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
            self.inner.alloc(dtype, shape)
        }

        fn view(&self, tensor: &Tensor, op: CoreViewOp) -> Result<Tensor, BackendError> {
            self.inner.view(tensor, op)
        }

        fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
            self.inner.write(tensor, bytes)
        }

        fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
            self.gate.wait();
            self.inner.read(tensor)
        }

        fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
            self.gate.releases.fetch_add(1, Ordering::AcqRel);
            self.inner.release(tensor)
        }

        fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
            self.inner.submit(commands)
        }
    }
}
