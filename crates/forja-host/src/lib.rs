//! Trusted host integration for running Forja guest components.

#![allow(clippy::manual_async_fn)]

#[cfg(test)]
mod fuzz_tests;
mod native;
mod weights;

use std::time::{Duration, Instant};
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use forja_core::{
    Affine, Backend, BackendDispatchData, BackendError, BufferId, CommandList, DType, DeviceInfo,
    GraphLimits, GraphTemplate, Layout, LayoutError, MemoryUsage, Op, OpError, PagePool,
    PagePoolError, PagedCopyDirection, PagedLayout, PagedPlane, PagedSequence, PagedSlotBanks,
    ParamSpace, PreparedGraph, RuleArm, SequenceCheckpoint, Slice, Submission, SubmissionProfile,
    SymbolicLayout, SymbolicLayoutError, TemplateOp, TemplateTensor, Tensor, ViewOp,
    gather_matmul_flops, gather_quant_matmul_dispatch_flops, matmul_flops,
    program::{
        BinOp, EpilogueError, EpilogueSignature, Inst, KernelSignature, MAX_INSTRUCTIONS,
        MAX_OUTPUTS, PrepareError, PreparedEpilogue, PreparedProgram, Program, ProgramError,
        ProgramKind, RedOp, UnOp, ValidatedProgram, ValueType, prepare_epilogue_retained,
        prepare_program_retained,
    },
    quant_matmul_flops, sdpa_flops,
};
use wasmtime::component::{Accessor, Component, HasData, Linker, Resource, ResourceTable};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder, Trap};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

/// Caps guest-controlled allocation for a backend variant name.
const MAX_VARIANT_NAME_BYTES: usize = 64;
/// Caps validation and replay work for one guest-provided variant rule.
const MAX_VARIANT_RULE_ARMS: usize = 64;

pub use native::{NativeCommandList, NativeEpilogue, NativeHost, NativeKernel, NativeTensor};
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
            "l9o:gpu/compute.kernel": crate::KernelEntry,
            "l9o:gpu/compute.epilogue": crate::EpilogueEntry,
            "l9o:gpu/compute.command-list": crate::CommandListEntry,
            "l9o:gpu/compute.weights": crate::WeightsEntry,
            "l9o:gpu/compute.params": crate::ParamsEntry,
            "l9o:gpu/compute.graph": crate::GraphEntry,
            "l9o:gpu/paged-storage.page-pool": crate::PagePoolEntry,
            "l9o:gpu/paged-storage.sequence": crate::PagedSequenceEntry,
            "l9o:gpu/paged-storage.batch-slot": crate::BatchSlotEntry,
        },
    });
}

/// Host bindings for the optional Metal algorithm interface.
#[cfg(target_os = "macos")]
#[allow(missing_docs)]
pub mod metal_bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "metal-host",
        imports: { default: async | trappable },
        require_store_data_send: true,
        with: {
            "l9o:gpu/compute": crate::bindings::l9o::gpu::compute,
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
            "l9o:gpu/paged-storage": crate::bindings::l9o::gpu::paged_storage,
        },
    });
}

use bindings::l9o::gpu::compute;
use bindings::l9o::gpu::paged_storage;
use compute::{Binop as WitBinOp, Redop as WitRedOp, Unop as WitUnOp, ValueType as WitValueType};

/// Static metadata declared by an engine component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineInfo {
    /// Logit vector length.
    pub vocab: u32,
    /// Largest accepted token position.
    pub max_context: u32,
    /// Layers whose hidden states can be returned by a step.
    pub tap_layers: Vec<u32>,
    /// Layers whose router logits can be returned by a step.
    pub router_layers: Vec<u32>,
    /// Optional geometry for host-owned paged serving storage.
    pub serving: Option<ServingInfo>,
}

/// Model-independent geometry for host-owned paged serving storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServingInfo {
    /// Tokens stored in one logical page.
    pub page_tokens: u32,
    /// Cache planes stored for each logical page.
    pub planes: Vec<CachePlane>,
    /// Largest number of simultaneously active lanes.
    pub active_lanes: u32,
    /// Largest context accepted through paged serving.
    pub max_context: u32,
}

/// One cache plane whose dtype is selected by the engine load profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachePlane {
    /// Scalar elements stored for each token.
    pub elements_per_token: u64,
    /// Required byte alignment.
    pub alignment: u64,
    /// Engine-owned diagnostic label.
    pub label: String,
}

/// Effective engine-side selections supplied during load.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadConfig {
    /// Whether the guest captures and replays reusable graphs.
    pub replay: bool,
    /// Whether the host supplies paged cache storage to the guest.
    pub paged_cache: bool,
    /// Key/value cache storage format selected by the profile.
    pub kv_cache: KvCacheFormat,
    /// Registered tuning names.
    pub tunings: Vec<String>,
    /// Fixed variant selections.
    pub fixed_variant_picks: Vec<FixedVariantPick>,
    /// Parameter-dependent variant selections.
    pub variant_rule_picks: Vec<VariantRulePick>,
}

impl Default for EngineLoadConfig {
    fn default() -> Self {
        Self {
            replay: true,
            paged_cache: false,
            kv_cache: KvCacheFormat::F32,
            tunings: Vec::new(),
            fixed_variant_picks: Vec::new(),
            variant_rule_picks: Vec::new(),
        }
    }
}

/// Key/value cache storage format selected by an engine profile.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvCacheFormat {
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point.
    Bf16,
    /// IEEE 754 binary32.
    #[default]
    F32,
}

/// One fixed engine dispatch-site selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedVariantPick {
    /// Stable engine-owned dispatch site.
    pub site: String,
    /// Stable backend algorithm name.
    pub name: String,
}

/// One parameter-dependent engine dispatch-site selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantRulePick {
    /// Stable engine-owned dispatch site.
    pub site: String,
    /// Replay parameter name.
    pub parameter: String,
    /// Complete ordered rule arms.
    pub arms: Vec<VariantPickArm>,
}

/// One inclusive variant-rule interval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantPickArm {
    /// Inclusive lower bound.
    pub lo: u32,
    /// Inclusive upper bound.
    pub hi: u32,
    /// Stable backend algorithm name.
    pub name: String,
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

/// Sampling parameters applied during decode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// Logit temperature, where zero selects greedily.
    pub temperature: f32,
    /// Number of greatest logits retained, where zero disables top-k.
    pub top_k: u32,
    /// Cumulative probability retained after top-k.
    pub top_p: f32,
    /// Counter-based random seed.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Input to decode with an engine-retained feedback token.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineDecode {
    /// Tokens to process, or `None` to reuse the preceding selected token.
    pub tokens: Option<Vec<u32>>,
    /// Position assigned to the first token, including a reused feedback token.
    pub start_pos: u32,
    /// Token sampling parameters.
    pub sampling: SamplingParams,
}

/// A tensor returned by a component and owned by its runner.
///
/// The handle remains valid until the runner consumes it or starts a step that replaces outputs.
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
    /// Requested per-layer router logits.
    pub router_logits: Vec<EngineTensor>,
}

/// Device-resident outputs from greedy decode.
#[derive(Debug)]
pub struct EngineDecodeOutput {
    /// Last-position logits with shape `[vocab]`.
    pub logits: EngineTensor,
    /// Selected token id with shape `[1]`.
    pub token: EngineTensor,
}

/// Input to the forward phase of constrained decode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineBeginDecode {
    /// Tokens to process, or `None` to reuse the retained feedback token.
    pub tokens: Option<Vec<u32>>,
    /// Position assigned to the first token.
    pub start_pos: u32,
}

/// Cumulative execution counters for an engine runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineMetrics {
    /// Bytes held by live guest tensor resources.
    pub live_bytes: u64,
    /// Command graphs currently retained by the guest.
    pub live_graphs: usize,
    /// Successfully completed backend submissions.
    pub submissions: u64,
    /// Completed submissions that supplied device timestamps.
    pub timed_submissions: u64,
    /// Sum of device execution time reported by timed submissions.
    pub gpu_time: Duration,
}

/// Memory observed after one profiled engine step.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EngineMemory {
    /// Bytes held by live guest tensor resources.
    pub live_bytes: u64,
    /// Additional bytes charged by the backend.
    pub charged_bytes: u64,
    /// Bytes currently allocated by the device API, when available.
    pub device_bytes: Option<u64>,
    /// Current process physical footprint, when available.
    pub physical_footprint: Option<u64>,
    /// Peak process physical footprint for this run, when available.
    pub peak_physical_footprint: Option<u64>,
}

impl EngineMemory {
    /// Returns the total bytes charged to the guest quota.
    #[must_use]
    pub fn accounted_bytes(self) -> Option<u64> {
        self.live_bytes.checked_add(self.charged_bytes)
    }

    /// Returns device allocation not covered by quota accounting.
    #[must_use]
    pub fn unaccounted_bytes(self) -> Option<u64> {
        self.device_bytes
            .zip(self.accounted_bytes())
            .map(|(actual, accounted)| actual.saturating_sub(accounted))
    }
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
    /// Host reads of returned tensors after the guest call.
    pub output_read: ImportProfile,
    /// Detailed timing for the step's submission.
    pub submission: Option<SubmissionProfile>,
    /// Memory observed after the step completed.
    pub memory: EngineMemory,
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
    loaded_num_hidden_layers: Option<u32>,
    output_handles: Vec<u32>,
    discarded_speculation: bool,
    profiling: bool,
    last_profile: Option<EngineStepProfile>,
    page_pool: Option<u32>,
    paged_origin: Option<SequenceCheckpoint>,
    paged_checkpoint: Option<(u32, SequenceCheckpoint)>,
}

/// A shareable host-owned paged sequence used for prefix reuse.
#[derive(Clone)]
pub struct PagedPrefix {
    sequence: Arc<Mutex<PagedSequence>>,
}

impl PagedPrefix {
    /// Wraps an owned sequence for indexed prefix retention.
    #[doc(hidden)]
    #[must_use]
    pub fn new(sequence: PagedSequence) -> Self {
        Self {
            sequence: Arc::new(Mutex::new(sequence)),
        }
    }

    /// Forks a committed prefix and reserves its complete request capacity.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid lengths, exhausted quota, or corrupt state.
    pub fn fork_reserved(
        &self,
        prefix_tokens: u64,
        reserved_tokens: u64,
    ) -> Result<Self, PagePoolError> {
        let sequence = self
            .sequence
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .fork_reserved(prefix_tokens, reserved_tokens)?;
        Ok(Self::new(sequence))
    }

    /// Returns the number of physical pages referenced by this sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence state is corrupt.
    pub fn page_count(&self) -> Result<usize, PagePoolError> {
        Ok(self
            .sequence
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .table()?
            .entries()
            .len())
    }

    /// Returns the committed logical token length.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence state is corrupt.
    pub fn logical_len(&self) -> Result<u64, PagePoolError> {
        Ok(self
            .sequence
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .logical_len())
    }

    /// Returns the reserved token capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence state is corrupt.
    pub fn reserved_tokens(&self) -> Result<u64, PagePoolError> {
        Ok(self
            .sequence
            .lock()
            .map_err(|_| PagePoolError::CorruptState)?
            .reserved_tokens())
    }
}

type EngineCallFuture<'a, T> =
    Pin<Box<dyn Future<Output = wasmtime::Result<Result<T, compute::Error>>> + Send + 'a>>;

/// Bounds timeout-check latency without dedicating a thread to each store.
const EPOCH_TICK: Duration = Duration::from_millis(10);
static EPOCH_ENGINES: Mutex<Vec<EpochEngine>> = Mutex::new(Vec::new());
static EPOCH_TICKER: OnceLock<()> = OnceLock::new();
static NEXT_RUNNER_ID: AtomicU64 = AtomicU64::new(1);

fn expected_layer_outputs(layers: &[u32], loaded: Option<u32>, requested: bool) -> usize {
    if !requested {
        return 0;
    }
    loaded.map_or(layers.len(), |loaded| {
        layers.iter().filter(|&&layer| layer <= loaded).count()
    })
}

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
        add_engine_to_linker(&mut linker, &backend)?;
        let grants = Grants::new().with_weights("engine", weights_path);
        let mut store = Host::new_store_with_grants(&engine, backend, limits, grants);
        Host::configure_component_mode(&mut store, &engine, &component);
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
            loaded_num_hidden_layers: None,
            output_handles: Vec::new(),
            discarded_speculation: false,
            profiling: false,
            last_profile: None,
            page_pool: None,
            paged_origin: None,
            paged_checkpoint: None,
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
            router_layers: info.router_layers,
            serving: info.serving.map(|serving| ServingInfo {
                page_tokens: serving.page_tokens,
                planes: serving
                    .planes
                    .into_iter()
                    .map(|plane| CachePlane {
                        elements_per_token: plane.elements_per_token,
                        alignment: plane.alignment,
                        label: plane.label,
                    })
                    .collect(),
                active_lanes: serving.active_lanes,
                max_context: serving.max_context,
            }),
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
        self.load_with_config(None).await
    }

    /// Opens the configured weight grant and asks the engine to load a layer prefix.
    ///
    /// # Errors
    ///
    /// Returns a host, component, or engine loading error.
    pub async fn load_with_config(
        &mut self,
        num_hidden_layers: Option<u32>,
    ) -> wasmtime::Result<Result<(), compute::Error>> {
        self.load_with_selections(num_hidden_layers, &EngineLoadConfig::default())
            .await
    }

    /// Opens the configured weight grant and loads effective engine-side selections.
    ///
    /// # Errors
    ///
    /// Returns a host, component, or engine loading error.
    pub async fn load_with_selections(
        &mut self,
        num_hidden_layers: Option<u32>,
        config: &EngineLoadConfig,
    ) -> wasmtime::Result<Result<(), compute::Error>> {
        let info = match &self.info {
            Some(info) => info.clone(),
            None => self.describe().await?,
        };
        let pool = if config.paged_cache {
            match self.bind_page_pool(&info, config.kv_cache) {
                Ok(pool) => pool,
                Err(error) => return Ok(Err(error)),
            }
        } else {
            None
        };
        self.set_guest_deadline();
        let weights = self
            .store
            .data_mut()
            .open_weights("engine")
            .map_err(wasmtime::Error::msg)?;
        let engine = self.instance.l9o_gpu_engine();
        let result = match self
            .store
            .run_concurrent(async move |accessor| {
                engine
                    .call_load(
                        accessor,
                        weights,
                        pool.map(Resource::new_borrow),
                        engine_bindings::exports::l9o::gpu::engine::LoadConfig {
                            num_hidden_layers,
                            replay: config.replay,
                            kv_cache: match config.kv_cache {
                                KvCacheFormat::F16 => {
                                    engine_bindings::exports::l9o::gpu::engine::KvCacheFormat::Float16
                                }
                                KvCacheFormat::Bf16 => {
                                    engine_bindings::exports::l9o::gpu::engine::KvCacheFormat::Bfloat16
                                }
                                KvCacheFormat::F32 => {
                                    engine_bindings::exports::l9o::gpu::engine::KvCacheFormat::Float32
                                }
                            },
                            tunings: config.tunings.clone(),
                            fixed_variant_picks: config
                                .fixed_variant_picks
                                .iter()
                                .map(|pick| {
                                    engine_bindings::exports::l9o::gpu::engine::FixedVariantPick {
                                        site: pick.site.clone(),
                                        name: pick.name.clone(),
                                    }
                                })
                                .collect(),
                            variant_rule_picks: config
                                .variant_rule_picks
                                .iter()
                                .map(|pick| {
                                    engine_bindings::exports::l9o::gpu::engine::VariantRulePick {
                                    site: pick.site.clone(),
                                    parameter: pick.parameter.clone(),
                                    arms: pick.arms.iter().map(|arm| {
                                        engine_bindings::exports::l9o::gpu::engine::VariantArm {
                                            lo: arm.lo,
                                            hi: arm.hi,
                                            name: arm.name.clone(),
                                        }
                                    }).collect(),
                                }
                                })
                                .collect(),
                        },
                    )
                    .await
            })
            .await
        {
            Ok(result) => result,
            Err(error) if is_epoch_timeout(&error) => Ok(Err(guest_timeout())),
            Err(error) => Err(error),
        };
        if matches!(result, Ok(Ok(()))) {
            self.loaded_num_hidden_layers = num_hidden_layers;
        }
        result
    }

    fn bind_page_pool(
        &mut self,
        info: &EngineInfo,
        kv_cache: KvCacheFormat,
    ) -> Result<Option<u32>, compute::Error> {
        let Some(serving) = &info.serving else {
            return Err(compute::Error::Layout(
                "paged cache requires an engine serving descriptor".to_owned(),
            ));
        };
        if serving.active_lanes == 0
            || serving.max_context == 0
            || serving.max_context > info.max_context
        {
            return Err(compute::Error::Layout(
                "paged serving ceilings are invalid".to_owned(),
            ));
        }
        let dtype = match kv_cache {
            KvCacheFormat::F16 => DType::F16,
            KvCacheFormat::Bf16 => DType::BF16,
            KvCacheFormat::F32 => DType::F32,
        };
        if let Some(resource) = self.page_pool {
            let entry = self
                .store
                .data()
                .table
                .get(&Resource::<PagePoolEntry>::new_borrow(resource))
                .map_err(invalid_handle)?;
            if entry
                .pool
                .layout()
                .planes()
                .iter()
                .any(|plane| plane.dtype != dtype)
            {
                return Err(compute::Error::Layout(
                    "paged pool dtype cannot change after load".to_owned(),
                ));
            }
            return Ok(Some(resource));
        }
        let planes = serving
            .planes
            .iter()
            .map(|plane| PagedPlane {
                dtype,
                elements_per_token: plane.elements_per_token,
                alignment: plane.alignment,
                label: plane.label.clone(),
            })
            .collect();
        let layout = PagedLayout::new(serving.page_tokens, planes)
            .map_err(|error| compute::Error::Layout(error.to_string()))?;
        let pages_per_sequence = serving.max_context.div_ceil(serving.page_tokens);
        let pool = PagePool::new(
            layout,
            pages_per_sequence,
            self.store.data().limits.live_bytes,
        )
        .map_err(|error| compute::Error::Quota(error.to_string()))?;
        let entry = PagePoolEntry {
            pool,
            active_lanes: serving.active_lanes,
            max_context: serving.max_context,
            plane_count: u32::try_from(serving.planes.len())
                .map_err(|_| compute::Error::Layout("paged plane count exceeds u32".to_owned()))?,
            slot_banks: None,
            slot: None,
            charged_bytes: AtomicU64::new(0),
            live_bytes: Arc::clone(&self.store.data().live_bytes),
        };
        let resource = self
            .store
            .data_mut()
            .table
            .push(entry)
            .map_err(invalid_handle)?;
        self.page_pool = Some(resource.rep());
        Ok(self.page_pool)
    }

    async fn prepare_paged_append(
        &mut self,
        start: u32,
        tokens: u32,
    ) -> Result<Option<SequenceCheckpoint>, compute::Error> {
        if start == 0
            && let (Some(pool), Some(checkpoint)) = (self.page_pool, self.paged_origin.take())
        {
            self.store
                .data_mut()
                .rollback_paged_append(pool, checkpoint)?;
            self.paged_checkpoint = None;
        }
        if let (Some(pool), Some((checkpoint_start, checkpoint))) =
            (self.page_pool, self.paged_checkpoint.take())
        {
            if checkpoint_start == start {
                self.store
                    .data_mut()
                    .rollback_paged_append(pool, checkpoint)?;
            } else {
                self.paged_checkpoint = Some((checkpoint_start, checkpoint));
            }
        }
        match self.page_pool {
            Some(pool) => {
                let (checkpoint, copy) = self
                    .store
                    .data_mut()
                    .begin_paged_append(pool, start, tokens)?;
                if let Some(copy) = copy
                    && let Err(error) = copy.run().await
                {
                    self.rollback_paged_append(checkpoint)?;
                    return Err(error);
                }
                Ok(checkpoint)
            }
            None => Ok(None),
        }
    }

    fn rollback_paged_append(
        &mut self,
        checkpoint: Option<SequenceCheckpoint>,
    ) -> Result<(), compute::Error> {
        match (self.page_pool, checkpoint) {
            (Some(pool), Some(checkpoint)) => self
                .store
                .data_mut()
                .rollback_paged_append(pool, checkpoint),
            _ => Ok(()),
        }
    }

    fn finish_paged_append(
        &mut self,
        start: u32,
        checkpoint: Option<SequenceCheckpoint>,
    ) -> Result<(), compute::Error> {
        match (self.page_pool, checkpoint) {
            (Some(pool), Some(checkpoint)) => {
                let checkpoint = self
                    .store
                    .data_mut()
                    .finish_paged_append(pool, checkpoint)?;
                if start == 0 {
                    self.paged_origin = checkpoint;
                    self.paged_checkpoint = None;
                } else {
                    self.paged_checkpoint = checkpoint.map(|checkpoint| (start, checkpoint));
                }
                Ok(())
            }
            _ => Ok(()),
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
        self.with_profile(move |runner| Box::pin(runner.step_inner(input)))
            .await
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
        let checkpoint = match self.prepare_paged_append(input.start_pos, sequence).await {
            Ok(checkpoint) => checkpoint,
            Err(error) => return Ok(Err(error)),
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
            Ok(output) => match output? {
                Ok(output) => output,
                Err(error) => {
                    self.rollback_paged_append(checkpoint)?;
                    return Ok(Err(error));
                }
            },
            Err(error) if is_epoch_timeout(&error) => {
                self.rollback_paged_append(checkpoint)?;
                return Ok(Err(guest_timeout()));
            }
            Err(error) => {
                self.rollback_paged_append(checkpoint)?;
                return Err(error);
            }
        };
        let StepOut {
            logits,
            taps,
            router_logits,
        } = output;
        let handles = std::iter::once(logits.rep())
            .chain(taps.iter().map(Resource::rep))
            .chain(router_logits.iter().map(Resource::rep))
            .collect::<Vec<_>>();
        if let Err(error) = self.validate_output(
            &info,
            sequence,
            taps_requested,
            &logits,
            &taps,
            &router_logits,
        ) {
            self.rollback_paged_append(checkpoint)?;
            return Ok(Err(match self.release_handles(handles) {
                Ok(()) => error,
                Err(release_error) => release_error,
            }));
        }
        self.finish_paged_append(input.start_pos, checkpoint)?;
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
            router_logits: router_logits
                .into_iter()
                .map(|tensor| EngineTensor {
                    handle: tensor.rep(),
                    runner_id: self.id,
                })
                .collect(),
        }))
    }

    /// Runs greedy decode and retains its logits and selected-token tensors.
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn decode(
        &mut self,
        input: EngineDecode,
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        self.with_profile(move |runner| Box::pin(runner.decode_inner(input, true)))
            .await
    }

    /// Enqueues greedy decode while retaining outputs from earlier queued calls.
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn enqueue_decode(
        &mut self,
        input: EngineDecode,
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        self.with_profile(move |runner| Box::pin(runner.decode_inner(input, false)))
            .await
    }

    /// Runs the forward phase of constrained decode and retains its logits.
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn begin_decode(
        &mut self,
        input: EngineBeginDecode,
    ) -> wasmtime::Result<Result<EngineTensor, compute::Error>> {
        self.with_profile(move |runner| Box::pin(runner.begin_decode_inner(input)))
            .await
    }

    async fn begin_decode_inner(
        &mut self,
        input: EngineBeginDecode,
    ) -> wasmtime::Result<Result<EngineTensor, compute::Error>> {
        let appended = input
            .tokens
            .as_ref()
            .map_or(1, |tokens| u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        let checkpoint = match self.prepare_paged_append(input.start_pos, appended).await {
            Ok(checkpoint) => checkpoint,
            Err(error) => return Ok(Err(error)),
        };
        let start_pos = input.start_pos;
        self.set_guest_deadline();
        let engine = self.instance.l9o_gpu_engine();
        let logits = match self
            .store
            .run_concurrent(async move |accessor| {
                engine
                    .call_begin_decode(accessor, input.tokens, input.start_pos)
                    .await
            })
            .await
        {
            Ok(output) => output?,
            Err(error) if is_epoch_timeout(&error) => {
                self.rollback_paged_append(checkpoint)?;
                return Ok(Err(guest_timeout()));
            }
            Err(error) => {
                self.rollback_paged_append(checkpoint)?;
                return Err(error);
            }
        };
        let logits = match logits {
            Ok(logits) => logits,
            Err(error) => {
                self.rollback_paged_append(checkpoint)?;
                return Ok(Err(error));
            }
        };
        let info = match &self.info {
            Some(info) => info.clone(),
            None => self.describe().await?,
        };
        if let Err(error) = self.validate_logits(&info, &logits) {
            self.rollback_paged_append(checkpoint)?;
            return Ok(Err(error));
        }
        self.finish_paged_append(start_pos, checkpoint)?;
        self.output_handles.push(logits.rep());
        Ok(Ok(EngineTensor {
            handle: logits.rep(),
            runner_id: self.id,
        }))
    }

    /// Releases logits returned by [`Self::begin_decode`] without selecting a token.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or guest resource-release failure.
    pub fn discard_tensor(&mut self, tensor: &EngineTensor) -> Result<(), compute::Error> {
        if tensor.runner_id != self.id {
            return Err(invalid_handle("engine tensor belongs to another runner"));
        }
        self.output_handles
            .retain(|handle| *handle != tensor.handle);
        self.release_handles(vec![tensor.handle])
    }

    /// Selects a constrained token from logits returned by [`Self::begin_decode`].
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn finish_decode(
        &mut self,
        logits: &EngineTensor,
        start_pos: u32,
        sampling: SamplingParams,
        allow: &[u32],
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        let logits = EngineTensor {
            handle: logits.handle,
            runner_id: logits.runner_id,
        };
        let allow = allow.to_vec();
        self.with_profile(move |runner| {
            Box::pin(runner.finish_decode_inner(logits, start_pos, sampling, allow))
        })
        .await
    }

    async fn finish_decode_inner(
        &mut self,
        logits: EngineTensor,
        start_pos: u32,
        sampling: SamplingParams,
        allow: Vec<u32>,
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        if logits.runner_id != self.id {
            return Ok(Err(invalid_handle(
                "engine tensor belongs to another runner",
            )));
        }
        self.set_guest_deadline();
        let engine = self.instance.l9o_gpu_engine();
        let borrowed = Resource::new_borrow(logits.handle);
        let output = match self
            .store
            .run_concurrent(async move |accessor| {
                engine
                    .call_finish_decode(
                        accessor,
                        borrowed,
                        start_pos,
                        engine_bindings::exports::l9o::gpu::engine::SamplingParams {
                            temperature: sampling.temperature,
                            top_k: sampling.top_k,
                            top_p: sampling.top_p,
                            seed: sampling.seed,
                        },
                        allow,
                    )
                    .await
            })
            .await
        {
            Ok(output) => output?,
            Err(error) if is_epoch_timeout(&error) => return Ok(Err(guest_timeout())),
            Err(error) => return Err(error),
        };
        let output = match output {
            Ok(output) => output,
            Err(error) => return Ok(Err(error)),
        };
        let info = match &self.info {
            Some(info) => info.clone(),
            None => self.describe().await?,
        };
        if let Err(error) = self.validate_decode_output(&info, &output.logits, &output.token) {
            return Ok(Err(error));
        }
        if let Err(error) = self.release_outputs() {
            return Ok(Err(error));
        }
        self.discarded_speculation = false;
        self.output_handles = vec![output.logits.rep(), output.token.rep()];
        Ok(Ok(EngineDecodeOutput {
            logits: EngineTensor {
                handle: output.logits.rep(),
                runner_id: self.id,
            },
            token: EngineTensor {
                handle: output.token.rep(),
                runner_id: self.id,
            },
        }))
    }

    /// Restores engine feedback for a constrained decode rewind.
    ///
    /// # Errors
    ///
    /// Returns a component execution error or the engine's structured failure.
    pub async fn rewind_decode(
        &mut self,
        next_pos: u32,
    ) -> wasmtime::Result<Result<(), compute::Error>> {
        self.with_profile(move |runner| Box::pin(runner.rewind_decode_inner(next_pos)))
            .await
    }

    /// Replaces the active paged request with a retained prefix or a fresh private sequence.
    ///
    /// Returns `false` when this runner was loaded without paged storage.
    ///
    /// # Errors
    ///
    /// Returns an error when reservation or binding fails.
    pub fn prepare_paged_request(
        &mut self,
        prefix: Option<PagedPrefix>,
        reserved_tokens: u64,
    ) -> Result<bool, compute::Error> {
        let Some(pool) = self.page_pool else {
            return Ok(false);
        };
        self.store
            .data_mut()
            .bind_paged_request(pool, prefix, reserved_tokens)?;
        self.paged_origin = None;
        self.paged_checkpoint = None;
        Ok(true)
    }

    /// Returns the active paged sequence as a shareable prefix capability.
    ///
    /// # Errors
    ///
    /// Returns an error when paged host state is unavailable or corrupt.
    pub fn paged_prefix(&self) -> Result<Option<PagedPrefix>, compute::Error> {
        self.page_pool
            .map(|pool| self.store.data().paged_prefix(pool))
            .transpose()
    }

    async fn rewind_decode_inner(
        &mut self,
        next_pos: u32,
    ) -> wasmtime::Result<Result<(), compute::Error>> {
        if let Err(error) = self.release_outputs() {
            return Ok(Err(error));
        }
        self.set_guest_deadline();
        let engine = self.instance.l9o_gpu_engine();
        let result = match self
            .store
            .run_concurrent(async move |accessor| {
                engine.call_rewind_decode(accessor, next_pos).await
            })
            .await
        {
            Ok(output) => output,
            Err(error) if is_epoch_timeout(&error) => Ok(Err(guest_timeout())),
            Err(error) => Err(error),
        };
        if matches!(result, Ok(Ok(()))) {
            if let Some((start, checkpoint)) = self.paged_checkpoint.take() {
                if start == next_pos {
                    self.rollback_paged_append(Some(checkpoint))?;
                } else if let Some(pool) = self.page_pool {
                    self.store
                        .data_mut()
                        .rewind_private_paged_sequence(pool, next_pos)?;
                }
            }
            self.discarded_speculation = false;
        }
        result
    }

    async fn decode_inner(
        &mut self,
        input: EngineDecode,
        release_previous: bool,
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        use engine_bindings::exports::l9o::gpu::engine::{DecodeIn, DecodeOut};
        if input.tokens.is_some() {
            self.discarded_speculation = false;
        } else if self.discarded_speculation {
            return Ok(Err(compute::Error::OpSignature(
                "discarded speculative decode invalidated this runner".to_owned(),
            )));
        }
        if release_previous && let Err(error) = self.release_outputs() {
            return Ok(Err(error));
        }
        let info = match &self.info {
            Some(info) => info.clone(),
            None => self.describe().await?,
        };
        let appended = input
            .tokens
            .as_ref()
            .map_or(1, |tokens| u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        let checkpoint = match self.prepare_paged_append(input.start_pos, appended).await {
            Ok(checkpoint) => checkpoint,
            Err(error) => return Ok(Err(error)),
        };
        self.set_guest_deadline();
        let engine = self.instance.l9o_gpu_engine();
        let output = match self
            .store
            .run_concurrent(async move |accessor| {
                engine
                    .call_decode(
                        accessor,
                        DecodeIn {
                            tokens: input.tokens,
                            start_pos: input.start_pos,
                            sampling: engine_bindings::exports::l9o::gpu::engine::SamplingParams {
                                temperature: input.sampling.temperature,
                                top_k: input.sampling.top_k,
                                top_p: input.sampling.top_p,
                                seed: input.sampling.seed,
                            },
                        },
                    )
                    .await
            })
            .await
        {
            Ok(output) => match output? {
                Ok(output) => output,
                Err(error) => {
                    self.rollback_paged_append(checkpoint)?;
                    return Ok(Err(error));
                }
            },
            Err(error) if is_epoch_timeout(&error) => {
                self.rollback_paged_append(checkpoint)?;
                return Ok(Err(guest_timeout()));
            }
            Err(error) => {
                self.rollback_paged_append(checkpoint)?;
                return Err(error);
            }
        };
        let DecodeOut { logits, token } = output;
        let handles = vec![logits.rep(), token.rep()];
        if let Err(error) = self.validate_decode_output(&info, &logits, &token) {
            self.rollback_paged_append(checkpoint)?;
            return Ok(Err(match self.release_handles(handles) {
                Ok(()) => error,
                Err(release_error) => release_error,
            }));
        }
        self.finish_paged_append(input.start_pos, checkpoint)?;
        if release_previous {
            self.output_handles = handles;
        } else {
            self.output_handles.extend(handles);
        }
        Ok(Ok(EngineDecodeOutput {
            logits: EngineTensor {
                handle: logits.rep(),
                runner_id: self.id,
            },
            token: EngineTensor {
                handle: token.rep(),
                runner_id: self.id,
            },
        }))
    }

    /// Reads a queued decode token and releases that call's returned tensor handles.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or backend read failure.
    pub async fn read_queued_token(
        &mut self,
        output: EngineDecodeOutput,
    ) -> Result<Vec<u8>, compute::Error> {
        let result = self.read_queued_token_retained(&output).await;
        let release = self.release_decode_output(output);
        match (result, release) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(bytes), Ok(())) => Ok(bytes),
        }
    }

    /// Reads a queued decode token while retaining that call's returned tensors.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or backend read failure.
    pub async fn read_queued_token_retained(
        &mut self,
        output: &EngineDecodeOutput,
    ) -> Result<Vec<u8>, compute::Error> {
        if output.token.runner_id != self.id {
            return Err(invalid_handle("engine tensor belongs to another runner"));
        }
        let resource = Resource::new_borrow(output.token.handle);
        match self.store.data().prepare_replay_read(&resource) {
            Ok(request) => request.run().await.map_err(guest_error),
            Err(error) => Err(error),
        }
    }

    /// Releases an unread queued decode output.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle, replay, or backend release failure.
    pub async fn discard_queued_decode(
        &mut self,
        output: EngineDecodeOutput,
    ) -> Result<(), compute::Error> {
        self.discarded_speculation = true;
        self.read_queued_token(output).await.map(drop)
    }

    /// Releases an already-read queued decode output without invalidating speculation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or guest resource-release failure.
    pub fn release_queued_decode(
        &mut self,
        output: EngineDecodeOutput,
    ) -> Result<(), compute::Error> {
        self.release_decode_output(output)
    }

    fn release_decode_output(&mut self, output: EngineDecodeOutput) -> Result<(), compute::Error> {
        let EngineDecodeOutput { logits, token } = output;
        let handles = [logits.handle, token.handle];
        self.output_handles
            .retain(|handle| !handles.contains(handle));
        self.release_handles(handles.into())
    }

    /// Reads a returned tensor through the selected backend.
    ///
    /// # Errors
    ///
    /// Returns an invalid-handle or backend read failure.
    pub async fn read(&mut self, tensor: &EngineTensor) -> Result<Vec<u8>, compute::Error> {
        let started = self.last_profile.as_ref().map(|_| Instant::now());
        if tensor.runner_id != self.id {
            return Err(invalid_handle("engine tensor belongs to another runner"));
        }
        let resource = Resource::new_borrow(tensor.handle);
        let result = match self.store.data().prepare_read(&resource) {
            Ok(request) => request.run().await.map_err(guest_error),
            Err(error) => Err(error),
        };
        if let (Some(started), Some(profile)) = (started, self.last_profile.as_mut()) {
            let elapsed = started.elapsed();
            profile.output_read.count = profile.output_read.count.saturating_add(1);
            profile.output_read.time = profile.output_read.time.saturating_add(elapsed);
            profile.wall_time = profile.wall_time.saturating_add(elapsed);
        }
        result
    }

    /// Returns cumulative backend execution counters.
    #[must_use]
    pub fn metrics(&self) -> EngineMetrics {
        self.store.data().engine_metrics()
    }

    /// Returns memory currently attributed to this engine runner.
    #[must_use]
    pub fn memory_usage(&self) -> EngineMemory {
        self.store.data().engine_memory()
    }

    /// Waits until the requested number of backend submissions has completed successfully.
    ///
    /// # Errors
    ///
    /// Returns a timeout if completion accounting does not advance within the backend deadline.
    pub fn wait_for_submissions(
        &self,
        expected: u64,
    ) -> impl Future<Output = Result<(), BackendError>> + Send + 'static {
        let timeout = self.store.data().limits.submission_timeout;
        let notify = Arc::clone(&self.store.data().submission_notify);
        let completed = Arc::clone(&self.store.data().completed_submissions);
        async move {
            let started = Instant::now();
            loop {
                let notified = notify.notified();
                if completed.load(Ordering::Acquire) >= expected {
                    return Ok(());
                }
                let remaining = timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() || tokio::time::timeout(remaining, notified).await.is_err() {
                    return Err(BackendError::Timeout);
                }
            }
        }
    }

    /// Waits for every backend submission queued before this call to complete.
    ///
    /// # Errors
    ///
    /// Returns a timeout if completion accounting does not advance within the backend deadline.
    pub async fn wait_for_queued_submissions(&self) -> Result<(), BackendError> {
        let expected = self.store.data().queued_submissions.load(Ordering::Acquire);
        self.wait_for_submissions(expected).await
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

    async fn with_profile<T>(
        &mut self,
        call: impl for<'a> FnOnce(&'a mut Self) -> EngineCallFuture<'a, T>,
    ) -> wasmtime::Result<Result<T, compute::Error>> {
        if !self.profiling {
            return call(self).await;
        }
        self.store.data_mut().begin_profile_step();
        let started = Instant::now();
        let result = call(self).await;
        self.last_profile = self.store.data_mut().finish_profile_step(started.elapsed());
        result
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
        router_logits: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        self.validate_logits(info, logits)?;
        let expected_taps = expected_layer_outputs(
            &info.tap_layers,
            self.loaded_num_hidden_layers,
            taps_requested,
        );
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
        let expected_routers = expected_layer_outputs(
            &info.router_layers,
            self.loaded_num_hidden_layers,
            taps_requested,
        );
        if router_logits.len() != expected_routers {
            return Err(compute::Error::Layout(format!(
                "engine returned {} router taps, expected {expected_routers}",
                router_logits.len()
            )));
        }
        for router in router_logits {
            let router = self.store.data().entry(router)?;
            let shape = router.tensor.layout().shape();
            if router.tensor.layout().dtype() != DType::F32
                || shape.len() != 2
                || shape.first() != Some(&sequence)
                || shape.last() == Some(&0)
            {
                return Err(compute::Error::Layout(format!(
                    "engine router taps must be f32 [{sequence}, experts]"
                )));
            }
        }
        Ok(())
    }

    fn validate_decode_output(
        &self,
        info: &EngineInfo,
        logits: &Resource<TensorEntry>,
        token: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        self.validate_logits(info, logits)?;
        let token = self.store.data().entry(token)?;
        if token.tensor.layout().dtype() != DType::U32 || token.tensor.layout().shape() != [1] {
            return Err(compute::Error::Layout(
                "engine selected token must be u32 [1]".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_logits(
        &self,
        info: &EngineInfo,
        logits: &Resource<TensorEntry>,
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
    live_kernels: usize,
    live_graphs: usize,
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
            live_kernels: 64,
            live_graphs: 16,
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

    /// Overrides the number of prepared programs that may remain alive.
    #[must_use]
    pub const fn with_kernel_limit(mut self, max_live_kernels: usize) -> Self {
        self.live_kernels = max_live_kernels;
        self
    }

    /// Overrides the number of graphs that may remain alive.
    #[must_use]
    pub const fn with_graph_limit(mut self, max_live_graphs: usize) -> Self {
        self.live_graphs = max_live_graphs;
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
    taints: Arc<BufferTaints>,
    guest_handles: AtomicUsize,
}

#[derive(Debug)]
enum BufferKind {
    Allocated {
        byte_len: u64,
        live_bytes: Arc<AtomicU64>,
    },
    Weights {
        source: Safetensors,
        copied_byte_len: Option<u64>,
        live_bytes: Arc<AtomicU64>,
    },
    CopiedWeight {
        byte_len: u64,
        live_bytes: Arc<AtomicU64>,
    },
}

impl BufferHandle {
    fn retain_guest_handle(&self) {
        self.guest_handles.fetch_add(1, Ordering::AcqRel);
    }

    fn release_guest_handle(&self) -> Result<(), compute::Error> {
        self.guest_handles
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_sub(1)
            })
            .map(|_| ())
            .map_err(|_| invalid_handle("guest tensor handle accounting underflowed"))
    }

    fn guest_reachable(&self) -> bool {
        self.guest_handles.load(Ordering::Acquire) != 0
    }

    fn release<B: Backend>(
        &self,
        backend: &B,
        profile: Option<&Arc<Mutex<EngineStepProfile>>>,
    ) -> Result<(), BackendError> {
        let _timer = BackendTimer::start(profile, BackendEvent::Release);
        backend.release(&self.owner)?;
        self.taints.release(self.owner.buffer());
        match &self.kind {
            BufferKind::CopiedWeight {
                byte_len,
                live_bytes,
            }
            | BufferKind::Allocated {
                byte_len,
                live_bytes,
            } => live_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                    bytes.checked_sub(*byte_len)
                })
                .map(|_| ())
                .map_err(|_| BackendError::ExecutionFailed),
            BufferKind::Weights {
                copied_byte_len,
                live_bytes,
                ..
            } => copied_byte_len.map_or(Ok(()), |byte_len| {
                live_bytes
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bytes| {
                        bytes.checked_sub(byte_len)
                    })
                    .map(|_| ())
                    .map_err(|_| BackendError::ExecutionFailed)
            }),
        }
    }

    fn weights(&self) -> Option<&Safetensors> {
        match &self.kind {
            BufferKind::Allocated { .. } | BufferKind::CopiedWeight { .. } => None,
            BufferKind::Weights { source, .. } => Some(source),
        }
    }
}

/// Host-owned state behind a guest tensor resource.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    tensor: Tensor,
    symbolic: Option<SymbolicLayout>,
    buffer: Arc<BufferHandle>,
}

/// Host-owned state behind an opaque guest page-pool resource.
pub struct PagePoolEntry {
    pool: PagePool,
    active_lanes: u32,
    max_context: u32,
    plane_count: u32,
    slot_banks: Option<Arc<PagedSlotBanks>>,
    slot: Option<u32>,
    charged_bytes: AtomicU64,
    live_bytes: Arc<AtomicU64>,
}

impl Drop for PagePoolEntry {
    fn drop(&mut self) {
        self.live_bytes
            .fetch_sub(self.charged_bytes.load(Ordering::Acquire), Ordering::AcqRel);
    }
}

/// Host-owned state behind an opaque guest paged-sequence resource.
#[derive(Debug)]
pub struct PagedSequenceEntry;

/// Host-owned state behind an opaque guest batch-slot resource.
pub struct BatchSlotEntry {
    binding: Arc<PagedBinding>,
    planes: Vec<u32>,
    tables: [Vec<u32>; 2],
    generations: u32,
}

struct PagedBinding {
    pool: PagePool,
    banks: Arc<PagedSlotBanks>,
    sequence: Mutex<Option<Arc<Mutex<PagedSequence>>>>,
    used: AtomicBool,
    table_dirty: AtomicBool,
}

struct PagedBackingHandles {
    planes: Vec<u32>,
    tables: [Vec<u32>; 2],
    generations: u32,
}

impl TensorEntry {
    fn template(&self) -> Result<TemplateTensor, compute::Error> {
        self.symbolic.as_ref().map_or_else(
            || Ok(self.tensor.clone().into()),
            |layout| {
                TemplateTensor::symbolic(self.tensor.clone(), layout.clone())
                    .map_err(|error| compute::Error::Layout(error.to_string()))
            },
        )
    }
}

/// Host-owned state behind a guest weights resource.
#[derive(Debug)]
pub struct WeightsEntry {
    buffers: Vec<Arc<BufferHandle>>,
}

/// Host-owned state behind a guest kernel resource.
#[derive(Clone, Debug)]
pub struct KernelEntry {
    program: Arc<PreparedProgram>,
}

/// Host-owned state behind a guest epilogue resource.
#[derive(Clone, Debug)]
pub struct EpilogueEntry {
    epilogue: Arc<PreparedEpilogue>,
}

impl EpilogueEntry {
    /// Returns the validated epilogue retained by this resource.
    #[doc(hidden)]
    #[must_use]
    pub const fn prepared(&self) -> &Arc<PreparedEpilogue> {
        &self.epilogue
    }
}

/// Host-owned state behind a guest parameter-space resource.
#[derive(Clone, Debug)]
pub struct ParamsEntry {
    space: ParamSpace,
}

#[derive(Debug)]
struct KernelLease {
    live: Arc<AtomicUsize>,
}

impl Drop for KernelLease {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }
}

impl KernelLease {
    fn acquire(live: Arc<AtomicUsize>, limit: usize) -> Result<Self, compute::Error> {
        live.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_add(1).filter(|&next| next <= limit)
        })
        .map_err(|_| quota("live kernels exceed the guest limit"))?;
        Ok(Self { live })
    }
}

#[derive(Debug)]
struct GraphLease {
    live: Arc<AtomicUsize>,
}

impl Drop for GraphLease {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }
}

impl GraphLease {
    fn acquire(live: Arc<AtomicUsize>, limit: usize) -> Result<Self, compute::Error> {
        live.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            count.checked_add(1).filter(|&next| next <= limit)
        })
        .map_err(|_| quota("live graphs exceed the guest limit"))?;
        Ok(Self { live })
    }
}

#[derive(Debug)]
enum RecordedDispatch {
    Static(Box<forja_core::Dispatch>),
    Operation {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
    Variant {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
        data: BackendDispatchData,
    },
    VariantRule {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    },
    Program {
        program: Arc<PreparedProgram>,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
    Epilogue {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        output: Box<TemplateTensor>,
        epilogue: Arc<PreparedEpilogue>,
        extras: Vec<TemplateTensor>,
    },
    EpilogueVariant {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        output: Box<TemplateTensor>,
        epilogue: Arc<PreparedEpilogue>,
        extras: Vec<TemplateTensor>,
        data: BackendDispatchData,
    },
    EpilogueVariantRule {
        op: TemplateOp,
        inputs: Vec<TemplateTensor>,
        output: Box<TemplateTensor>,
        epilogue: Arc<PreparedEpilogue>,
        extras: Vec<TemplateTensor>,
        parameter: u8,
        arms: Vec<(u32, BackendDispatchData)>,
    },
}

fn record_graph_dispatch(
    graph: &mut GraphTemplate,
    dispatch: &RecordedDispatch,
) -> Result<(), forja_core::GraphError> {
    match dispatch {
        RecordedDispatch::Static(dispatch) => graph.record_validated(dispatch.as_ref().clone()),
        RecordedDispatch::Operation {
            op,
            inputs,
            outputs,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let outputs = outputs.iter().collect::<Vec<_>>();
            graph.dispatch_many(*op, &inputs, &outputs)
        }
        RecordedDispatch::Variant {
            op,
            inputs,
            outputs,
            data,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let outputs = outputs.iter().collect::<Vec<_>>();
            graph.dispatch_variant_many(*op, &inputs, &outputs, data.clone())
        }
        RecordedDispatch::VariantRule {
            op,
            inputs,
            outputs,
            parameter,
            arms,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let outputs = outputs.iter().collect::<Vec<_>>();
            graph.dispatch_variant_rule_many(*op, &inputs, &outputs, *parameter, arms.clone())
        }
        RecordedDispatch::Program {
            program,
            inputs,
            outputs,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let outputs = outputs.iter().collect::<Vec<_>>();
            graph.dispatch_kernel(program, &inputs, &outputs)
        }
        RecordedDispatch::Epilogue {
            op,
            inputs,
            output,
            epilogue,
            extras,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let extras = extras.iter().collect::<Vec<_>>();
            graph.dispatch_epilogue(*op, &inputs, output.as_ref(), epilogue, &extras)
        }
        RecordedDispatch::EpilogueVariant {
            op,
            inputs,
            output,
            epilogue,
            extras,
            data,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let extras = extras.iter().collect::<Vec<_>>();
            graph.dispatch_epilogue_variant(
                *op,
                &inputs,
                output.as_ref(),
                epilogue,
                &extras,
                data.clone(),
            )
        }
        RecordedDispatch::EpilogueVariantRule {
            op,
            inputs,
            output,
            epilogue,
            extras,
            parameter,
            arms,
        } => {
            let inputs = inputs.iter().collect::<Vec<_>>();
            let extras = extras.iter().collect::<Vec<_>>();
            graph.dispatch_epilogue_variant_rule(
                *op,
                &inputs,
                output.as_ref(),
                epilogue,
                &extras,
                *parameter,
                arms.clone(),
            )
        }
    }
}

/// Host-owned state behind a guest command-list resource.
#[derive(Debug)]
pub struct CommandListEntry {
    commands: CommandList,
    recorded: Vec<RecordedDispatch>,
    access: SubmissionAccess,
    protected: HashSet<BufferId>,
    space: Option<ParamSpace>,
    parameterized: bool,
    retained: Vec<TensorEntry>,
    retained_kernels: Vec<KernelEntry>,
    retained_epilogues: Vec<EpilogueEntry>,
    paged: Vec<PagedUse>,
}

/// Host-owned state behind a guest graph resource.
#[derive(Debug)]
pub struct GraphEntry {
    graph: Arc<PreparedGraph>,
    access: SubmissionAccess,
    retained: Vec<TensorEntry>,
    retained_kernels: Vec<KernelEntry>,
    retained_epilogues: Vec<EpilogueEntry>,
    paged: Vec<PagedUse>,
    _lease: GraphLease,
}

/// Store state implementing the guest compute interface over a trusted backend.
pub struct Host<B: Backend> {
    backend: Arc<B>,
    table: ResourceTable,
    wasi: WasiCtx,
    limits: Limits,
    grants: Grants,
    weight_files: HashMap<String, Vec<Weak<BufferHandle>>>,
    store_limits: StoreLimits,
    live_bytes: Arc<AtomicU64>,
    live_handles: usize,
    live_kernels: Arc<AtomicUsize>,
    live_graphs: Arc<AtomicUsize>,
    graph_arena_bytes: u64,
    gpu_time_ns: Arc<AtomicU64>,
    queued_submissions: Arc<AtomicU64>,
    completed_submissions: Arc<AtomicU64>,
    timed_submissions: Arc<AtomicU64>,
    completed_gpu_time_ns: Arc<AtomicU64>,
    submission_notify: Arc<tokio::sync::Notify>,
    replay: Arc<ReplayState>,
    taints: Arc<BufferTaints>,
    active_profile: Option<Arc<Mutex<EngineStepProfile>>>,
    epoch_registration: Option<Arc<()>>,
    explicit_metal_variants: bool,
}

impl<B: Backend + 'static> Host<B> {
    fn new(backend: B, limits: Limits) -> Self {
        Self::with_grants(backend, limits, Grants::new())
    }

    fn with_grants(backend: B, limits: Limits, grants: Grants) -> Self {
        backend.set_memory_limit(limits.live_bytes);
        let replay = Arc::new(ReplayState::new(backend.graph_memory_depth()));
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
            live_kernels: Arc::new(AtomicUsize::new(0)),
            live_graphs: Arc::new(AtomicUsize::new(0)),
            graph_arena_bytes: 0,
            gpu_time_ns: Arc::new(AtomicU64::new(0)),
            queued_submissions: Arc::new(AtomicU64::new(0)),
            completed_submissions: Arc::new(AtomicU64::new(0)),
            timed_submissions: Arc::new(AtomicU64::new(0)),
            completed_gpu_time_ns: Arc::new(AtomicU64::new(0)),
            submission_notify: Arc::new(tokio::sync::Notify::new()),
            replay,
            taints: Arc::new(BufferTaints::default()),
            active_profile: None,
            epoch_registration: None,
            explicit_metal_variants: false,
        }
    }

    fn open_paged_slot(
        &mut self,
        resource: &Resource<PagePoolEntry>,
        planes: u32,
    ) -> Result<Resource<BatchSlotEntry>, compute::Error> {
        let (pool, active_lanes, table_pages, page_tokens, plane_specs, before) = {
            let entry = self.table.get(resource).map_err(invalid_handle)?;
            if planes == 0 || planes > entry.plane_count || entry.slot_banks.is_some() {
                return Err(compute::Error::Layout(
                    "paged slot planes are invalid or already open".to_owned(),
                ));
            }
            let table_pages = entry
                .max_context
                .div_ceil(entry.pool.layout().page_tokens());
            (
                entry.pool.clone(),
                usize::try_from(entry.active_lanes)
                    .map_err(|_| compute::Error::Layout("lane count exceeds usize".to_owned()))?,
                usize::try_from(table_pages).map_err(|_| {
                    compute::Error::Layout("page table length exceeds usize".to_owned())
                })?,
                entry.pool.layout().page_tokens(),
                entry
                    .pool
                    .layout()
                    .planes()
                    .iter()
                    .take(usize::try_from(planes).map_err(|_| {
                        compute::Error::Layout("paged plane count exceeds usize".to_owned())
                    })?)
                    .map(|plane| {
                        u32::try_from(plane.elements_per_token)
                            .map(|elements| (guest_dtype(plane.dtype), elements))
                            .map_err(|_| {
                                compute::Error::Layout("paged plane width exceeds u32".to_owned())
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                entry
                    .pool
                    .memory_usage()
                    .map_err(|error| compute::Error::Quota(error.to_string()))?
                    .0
                    .total()
                    .ok_or_else(|| {
                        compute::Error::Quota("paged pool byte size overflowed".to_owned())
                    })?,
            )
        };
        let banks = Arc::new(
            PagedSlotBanks::new(pool.clone(), active_lanes, table_pages)
                .map_err(|error| compute::Error::Quota(error.to_string()))?,
        );
        let backing =
            self.allocate_paged_backing(plane_specs, page_tokens, table_pages, active_lanes)?;
        let after = pool
            .memory_usage()
            .map_err(|error| compute::Error::Quota(error.to_string()))?
            .0
            .total()
            .ok_or_else(|| compute::Error::Quota("paged pool byte size overflowed".to_owned()))?;
        let charge = after.checked_sub(before).ok_or_else(|| {
            compute::Error::BackendExecution("paged pool charge regressed".to_owned())
        })?;
        self.check_byte_allocation(charge)?;
        let binding = Arc::new(PagedBinding {
            pool,
            banks: Arc::clone(&banks),
            sequence: Mutex::new(None),
            used: AtomicBool::new(false),
            table_dirty: AtomicBool::new(false),
        });
        let mut cleanup = backing.planes.clone();
        cleanup.extend(backing.tables.iter().flatten().copied());
        cleanup.push(backing.generations);
        let slot = match self.table.push(BatchSlotEntry {
            binding,
            planes: backing.planes,
            tables: backing.tables,
            generations: backing.generations,
        }) {
            Ok(slot) => slot,
            Err(error) => {
                self.drop_paged_handles(cleanup);
                return Err(invalid_handle(error));
            }
        };
        let entry = self.table.get_mut(resource).map_err(invalid_handle)?;
        entry.slot_banks = Some(banks);
        entry.slot = Some(slot.rep());
        entry.charged_bytes.fetch_add(charge, Ordering::AcqRel);
        entry.live_bytes.fetch_add(charge, Ordering::AcqRel);
        Ok(slot)
    }

    fn allocate_paged_backing(
        &mut self,
        plane_specs: Vec<(compute::Dtype, u32)>,
        page_tokens: u32,
        table_pages: usize,
        active_lanes: usize,
    ) -> Result<PagedBackingHandles, compute::Error> {
        let physical_pages = u32::try_from(table_pages)
            .map_err(|_| compute::Error::Layout("physical page count exceeds u32".to_owned()))?;
        let mut backing = Vec::new();
        let mut plane_handles = Vec::new();
        for (dtype, elements) in plane_specs {
            match self.alloc_paged_tensor(dtype, &[physical_pages, page_tokens, elements]) {
                Ok(tensor) => {
                    plane_handles.push(tensor);
                    backing.push(tensor);
                }
                Err(error) => {
                    self.drop_paged_handles(backing);
                    return Err(error);
                }
            }
        }
        let generations = match self.alloc_paged_tensor(compute::Dtype::U32, &[physical_pages]) {
            Ok(tensor) => {
                backing.push(tensor);
                tensor
            }
            Err(error) => {
                self.drop_paged_handles(backing);
                return Err(error);
            }
        };
        let mut table_handles = [Vec::new(), Vec::new()];
        for bank in &mut table_handles {
            for _ in 0..active_lanes {
                match self.alloc_paged_tensor(compute::Dtype::U32, &[physical_pages, 2]) {
                    Ok(tensor) => {
                        bank.push(tensor);
                        backing.push(tensor);
                    }
                    Err(error) => {
                        self.drop_paged_handles(backing);
                        return Err(error);
                    }
                }
            }
        }
        let generation_bytes = usize::try_from(physical_pages)
            .ok()
            .and_then(|pages| pages.checked_mul(4))
            .ok_or_else(|| {
                compute::Error::Layout("paged generation bytes overflowed".to_owned())
            })?;
        let zeros = vec![0; generation_bytes];
        if let Err(error) = self.write(&Resource::new_borrow(generations), &zeros) {
            self.drop_paged_handles(backing);
            return Err(error);
        }
        let poison = [0_u32.to_ne_bytes(), u32::MAX.to_ne_bytes()].concat();
        let poison = poison.repeat(table_pages);
        for &table in table_handles.iter().flatten() {
            if let Err(error) = self.write(&Resource::new_borrow(table), &poison) {
                self.drop_paged_handles(backing);
                return Err(error);
            }
        }
        Ok(PagedBackingHandles {
            planes: plane_handles,
            tables: table_handles,
            generations,
        })
    }

    fn alloc_paged_tensor(
        &mut self,
        dtype: compute::Dtype,
        shape: &[u32],
    ) -> Result<u32, compute::Error> {
        let dtype = core_dtype(dtype);
        let byte_len = self.check_allocation(dtype, shape)?;
        let tensor = self.backend.alloc(dtype, shape).map_err(guest_error)?;
        let buffer = Arc::new(BufferHandle {
            owner: tensor.clone(),
            taints: Arc::clone(&self.taints),
            guest_handles: AtomicUsize::new(0),
            kind: BufferKind::Allocated {
                byte_len,
                live_bytes: Arc::clone(&self.live_bytes),
            },
        });
        let entry = TensorEntry {
            tensor: tensor.clone(),
            symbolic: None,
            buffer: Arc::clone(&buffer),
        };
        self.live_bytes.fetch_add(byte_len, Ordering::AcqRel);
        let resource = self.table.push(entry).map_err(|error| {
            let _ = release_buffer(self.backend.as_ref(), buffer, self.active_profile.as_ref());
            invalid_handle(error)
        })?;
        Ok(resource.rep())
    }

    fn drop_paged_handles(&mut self, resources: Vec<u32>) {
        for resource in resources {
            if let Ok(entry) = self
                .table
                .delete(Resource::<TensorEntry>::new_own(resource))
            {
                let _ = release_buffer(
                    self.backend.as_ref(),
                    entry.buffer,
                    self.active_profile.as_ref(),
                );
            }
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "checkpoint rollback must cover table rebinding and COW preparation"
    )]
    fn begin_paged_append(
        &mut self,
        pool_rep: u32,
        start: u32,
        tokens: u32,
    ) -> Result<(Option<SequenceCheckpoint>, Option<SubmitRequest<B>>), compute::Error> {
        let (pool, max_context, slot_rep) = {
            let entry = self
                .table
                .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
                .map_err(invalid_handle)?;
            let Some(slot) = entry.slot else {
                return Ok((None, None));
            };
            (entry.pool.clone(), entry.max_context, slot)
        };
        let (binding, table_rep, generation_rep) = {
            let slot = self
                .table
                .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                .map_err(invalid_handle)?;
            (
                Arc::clone(&slot.binding),
                *slot
                    .tables
                    .first()
                    .and_then(|bank| bank.first())
                    .ok_or_else(|| invalid_handle("paged lane zero is missing"))?,
                slot.generations,
            )
        };
        binding.used.store(false, Ordering::Release);
        let sequence = {
            let mut active = binding
                .sequence
                .lock()
                .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
            if active.is_none() {
                *active = Some(Arc::new(Mutex::new(
                    pool.reserve(u64::from(max_context))
                        .map_err(|error| compute::Error::Quota(error.to_string()))?,
                )));
                binding.table_dirty.store(true, Ordering::Release);
            }
            Arc::clone(
                active
                    .as_ref()
                    .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?,
            )
        };
        let (checkpoint, table, copy, upload) = {
            let mut sequence = sequence
                .lock()
                .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
            let checkpoint = sequence
                .checkpoint()
                .map_err(|error| compute::Error::Quota(error.to_string()))?;
            if sequence.logical_len() == 0 && start != 0 {
                sequence
                    .commit_write(u64::from(start))
                    .map_err(|error| compute::Error::Quota(error.to_string()))?;
            }
            if sequence.logical_len() != u64::from(start) {
                return Err(compute::Error::Layout(
                    "paged append start does not match the sequence".to_owned(),
                ));
            }
            let copy = sequence
                .cow_tail()
                .map_err(|error| compute::Error::Quota(error.to_string()))?;
            sequence
                .commit_write(u64::from(tokens))
                .map_err(|error| compute::Error::Quota(error.to_string()))?;
            if copy.is_some() {
                binding.table_dirty.store(true, Ordering::Release);
            }
            if binding.table_dirty.load(Ordering::Acquire) {
                binding
                    .banks
                    .bind(0, 0, &sequence)
                    .map_err(|error| compute::Error::Quota(error.to_string()))?;
            }
            binding
                .banks
                .set_logical_len(0, 0, sequence.logical_len())
                .map_err(|error| compute::Error::Quota(error.to_string()))?;
            let table = binding
                .banks
                .slot(0, 0)
                .map_err(|error| compute::Error::Quota(error.to_string()))?
                .table()
                .clone();
            let upload = binding.table_dirty.swap(false, Ordering::AcqRel);
            (checkpoint, table, copy, upload)
        };
        let prepared = (|| {
            if upload {
                self.upload_paged_table(table_rep, generation_rep, &table)?;
            }
            copy.map(|copy| self.prepare_cow_copy(slot_rep, Arc::clone(&binding), copy))
                .transpose()
        })();
        match prepared {
            Ok(copy) => Ok((Some(checkpoint), copy)),
            Err(error) => {
                sequence
                    .lock()
                    .map_err(|_| guest_error(BackendError::ExecutionFailed))?
                    .rollback(checkpoint)
                    .map_err(|rollback| compute::Error::Quota(rollback.to_string()))?;
                Err(error)
            }
        }
    }

    fn upload_paged_table(
        &self,
        table_rep: u32,
        generation_rep: u32,
        table: &forja_core::PageTable,
    ) -> Result<(), compute::Error> {
        let mut table_bytes = Vec::with_capacity(table.entries().len() * 8);
        for entry in table.entries() {
            table_bytes.extend_from_slice(&entry.page().to_ne_bytes());
            table_bytes.extend_from_slice(&entry.generation().to_ne_bytes());
        }
        let mut generation_bytes = Vec::with_capacity(table.generations().len() * 4);
        for generation in table.generations() {
            generation_bytes.extend_from_slice(&generation.to_ne_bytes());
        }
        self.write(&Resource::new_borrow(table_rep), &table_bytes)?;
        self.write(&Resource::new_borrow(generation_rep), &generation_bytes)
    }

    fn prepare_cow_copy(
        &self,
        slot_rep: u32,
        binding: Arc<PagedBinding>,
        copy: forja_core::CowPageCopy,
    ) -> Result<SubmitRequest<B>, compute::Error> {
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
            .map_err(invalid_handle)?;
        let mut commands = CommandList::new();
        let mut access = SubmissionAccess::default();
        for plane in &slot.planes {
            let tensor = &self
                .table
                .get(&Resource::<TensorEntry>::new_borrow(*plane))
                .map_err(invalid_handle)?
                .tensor;
            let shape = tensor.layout().shape();
            let slices = |page| {
                Ok::<_, compute::Error>(ViewOp::Slice(vec![
                    Slice::new(page, 1, 1).map_err(guest_error)?,
                    Slice::new(0, copy.token_count(), 1).map_err(guest_error)?,
                    Slice::new(0, shape[2], 1).map_err(guest_error)?,
                ]))
            };
            let source = self
                .backend
                .view(tensor, slices(copy.source())?)
                .map_err(guest_error)?;
            let destination = self
                .backend
                .view(tensor, slices(copy.destination())?)
                .map_err(guest_error)?;
            commands
                .dispatch(Op::Copy, &[&source], &destination)
                .map_err(guest_error)?;
            access.reads.insert(tensor.buffer());
            access.writes.insert(tensor.buffer());
        }
        Ok(SubmitRequest {
            backend: Arc::clone(&self.backend),
            work: SubmissionWork::Commands(commands),
            access,
            retained: Vec::new(),
            paged: vec![PagedUse { binding, bank: 0 }],
            flight: None,
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
            queued_submissions: Arc::clone(&self.queued_submissions),
            completed_submissions: Arc::clone(&self.completed_submissions),
            timed_submissions: Arc::clone(&self.timed_submissions),
            completed_gpu_time_ns: Arc::clone(&self.completed_gpu_time_ns),
            submission_notify: Arc::clone(&self.submission_notify),
            taints: Arc::clone(&self.taints),
            profile: self.active_profile.clone(),
        })
    }

    fn finish_paged_append(
        &mut self,
        pool_rep: u32,
        checkpoint: SequenceCheckpoint,
    ) -> Result<Option<SequenceCheckpoint>, compute::Error> {
        let slot_rep = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("paged slot is not open"))?;
        let used = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
            .map_err(invalid_handle)?
            .binding
            .used
            .load(Ordering::Acquire);
        if used {
            Ok(Some(checkpoint))
        } else {
            self.rollback_paged_append(pool_rep, checkpoint)?;
            Ok(None)
        }
    }

    fn rollback_paged_append(
        &mut self,
        pool_rep: u32,
        checkpoint: SequenceCheckpoint,
    ) -> Result<(), compute::Error> {
        let slot_rep = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("paged slot is not open"))?;
        let binding = Arc::clone(
            &self
                .table
                .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                .map_err(invalid_handle)?
                .binding,
        );
        let sequence = binding
            .sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid_handle("paged sequence is not active"))?;
        sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .rollback(checkpoint)
            .map_err(|error| compute::Error::Quota(error.to_string()))
    }

    fn rewind_private_paged_sequence(
        &mut self,
        pool_rep: u32,
        logical_len: u32,
    ) -> Result<(), compute::Error> {
        let slot_rep = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("paged slot is not open"))?;
        let binding = Arc::clone(
            &self
                .table
                .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                .map_err(invalid_handle)?
                .binding,
        );
        let sequence = binding
            .sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid_handle("paged sequence is not active"))?;
        sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .rewind_private(u64::from(logical_len))
            .map_err(|error| compute::Error::Quota(error.to_string()))
    }

    fn paged_prefix(&self, pool_rep: u32) -> Result<PagedPrefix, compute::Error> {
        let slot_rep = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("paged slot is not open"))?;
        let binding = &self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
            .map_err(invalid_handle)?
            .binding;
        let sequence = binding
            .sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .as_ref()
            .cloned()
            .ok_or_else(|| invalid_handle("paged sequence is not active"))?;
        Ok(PagedPrefix { sequence })
    }

    fn bind_paged_request(
        &mut self,
        pool_rep: u32,
        prefix: Option<PagedPrefix>,
        reserved_tokens: u64,
    ) -> Result<(), compute::Error> {
        let (pool, max_context, slot_rep) = {
            let entry = self
                .table
                .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
                .map_err(invalid_handle)?;
            (
                entry.pool.clone(),
                entry.max_context,
                entry
                    .slot
                    .ok_or_else(|| invalid_handle("paged slot is not open"))?,
            )
        };
        if reserved_tokens > u64::from(max_context) {
            return Err(compute::Error::Quota(
                "paged request exceeds the context ceiling".to_owned(),
            ));
        }
        let binding = Arc::clone(
            &self
                .table
                .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                .map_err(invalid_handle)?
                .binding,
        );
        let sequence = match prefix {
            Some(prefix) => prefix.sequence,
            None => Arc::new(Mutex::new(
                pool.reserve(reserved_tokens)
                    .map_err(|error| compute::Error::Quota(error.to_string()))?,
            )),
        };
        {
            let sequence_guard = sequence
                .lock()
                .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
            binding
                .banks
                .bind(0, 0, &sequence_guard)
                .map_err(|error| compute::Error::Quota(error.to_string()))?;
        }
        *binding
            .sequence
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))? = Some(sequence);
        binding.used.store(false, Ordering::Release);
        binding.table_dirty.store(true, Ordering::Release);
        Ok(())
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

    /// Configures component-specific recording policy after creating a store.
    pub fn configure_component_mode(
        store: &mut Store<Self>,
        engine: &Engine,
        component: &Component,
    ) {
        store.data_mut().explicit_metal_variants = component
            .component_type()
            .get_import(engine, "l9o:gpu/metal-variants@0.1.0")
            .is_some();
    }

    /// Starts a fresh CPU-time budget for the next guest export call.
    pub fn reset_guest_deadline(store: &mut Store<Self>) {
        let ticks = epoch_ticks(store.data().limits.guest_call_timeout);
        store.set_epoch_deadline(ticks);
    }

    fn engine_metrics(&self) -> EngineMetrics {
        EngineMetrics {
            live_bytes: self.live_bytes.load(Ordering::Acquire),
            live_graphs: self.live_graphs.load(Ordering::Acquire),
            submissions: self.completed_submissions.load(Ordering::Acquire),
            timed_submissions: self.timed_submissions.load(Ordering::Acquire),
            gpu_time: Duration::from_nanos(self.completed_gpu_time_ns.load(Ordering::Acquire)),
        }
    }

    fn engine_memory(&self) -> EngineMemory {
        let MemoryUsage {
            charged_bytes,
            device_bytes,
            physical_footprint,
            peak_physical_footprint,
        } = self.backend.memory_usage();
        let live_bytes = self.live_bytes.load(Ordering::Acquire);
        EngineMemory {
            live_bytes,
            charged_bytes: charged_bytes.saturating_sub(live_bytes),
            device_bytes,
            physical_footprint,
            peak_physical_footprint,
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
        profile.memory = self.engine_memory();
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
            symbolic: None,
            buffer: Arc::new(BufferHandle {
                owner: tensor.clone(),
                taints: Arc::clone(&self.taints),
                guest_handles: AtomicUsize::new(0),
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
        self.table
            .get(&resource)
            .map_err(invalid_handle)?
            .buffer
            .retain_guest_handle();
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
        if let Some(layout) = &entry.symbolic {
            let symbolic =
                symbolic_view(layout, &operation).map_err(|error| graph_error(&error))?;
            let view = self
                .table
                .push(TensorEntry {
                    tensor: entry.tensor,
                    symbolic: Some(symbolic),
                    buffer: entry.buffer,
                })
                .map_err(invalid_handle)?;
            self.table
                .get(&view)
                .map_err(invalid_handle)?
                .buffer
                .retain_guest_handle();
            self.live_handles += 1;
            return Ok(view);
        }
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
                symbolic: None,
                buffer: Arc::clone(&entry.buffer),
            })
            .map_err(invalid_handle)?;
        self.table
            .get(&view)
            .map_err(invalid_handle)?
            .buffer
            .retain_guest_handle();
        self.live_handles += 1;
        Ok(view)
    }

    fn view_param(
        &mut self,
        resource: &Resource<TensorEntry>,
        space: &Resource<ParamsEntry>,
        slices: Vec<compute::ParamSlice>,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.entry(resource)?.clone();
        let space = self.table.get(space).map_err(invalid_handle)?.space.clone();
        if slices.len() != entry.tensor.layout().shape().len() {
            return Err(compute::Error::Layout(
                "parameterized slice rank does not match tensor rank".to_owned(),
            ));
        }
        let mut layout = entry
            .symbolic
            .clone()
            .unwrap_or_else(|| SymbolicLayout::new(entry.tensor.layout().clone(), space.clone()));
        if layout.space() != &space {
            return Err(compute::Error::Layout(
                "parameterized view mixes parameter spaces".to_owned(),
            ));
        }
        for (axis, slice) in slices.into_iter().enumerate() {
            let axis = u8::try_from(axis)
                .map_err(|_| compute::Error::Layout("tensor rank exceeds u8".to_owned()))?;
            layout = layout
                .slice(
                    axis,
                    core_affine(slice.start),
                    core_affine(slice.len),
                    slice.step,
                )
                .map_err(|error| graph_error(&error))?;
        }
        let view = self
            .table
            .push(TensorEntry {
                tensor: entry.tensor,
                symbolic: Some(layout),
                buffer: entry.buffer,
            })
            .map_err(invalid_handle)?;
        self.table
            .get(&view)
            .map_err(invalid_handle)?
            .buffer
            .retain_guest_handle();
        self.live_handles += 1;
        Ok(view)
    }

    fn params(
        &mut self,
        ranges: Vec<compute::ParamRange>,
    ) -> wasmtime::Result<Resource<ParamsEntry>> {
        let ranges = ranges
            .into_iter()
            .map(|range| range.lo..=range.hi)
            .collect();
        let space = ParamSpace::new(ranges).map_err(wasmtime::Error::msg)?;
        self.table
            .push(ParamsEntry { space })
            .map_err(wasmtime::Error::msg)
    }

    fn drop_params(&mut self, resource: Resource<ParamsEntry>) -> wasmtime::Result<()> {
        self.table
            .delete(resource)
            .map(|_| ())
            .map_err(wasmtime::Error::msg)
    }

    fn open_weights(&mut self, grant: &str) -> Result<Resource<WeightsEntry>, compute::Error> {
        self.check_handle_quota()?;
        let path = self
            .grants
            .weights_path(grant)
            .ok_or_else(|| invalid_handle("weight grant is not configured"))?
            .to_owned();
        let cached = self.weight_files.get(grant).cloned().unwrap_or_default();
        let live_cached = (!cached.is_empty())
            .then(|| cached.iter().map(Weak::upgrade).collect())
            .flatten();
        let buffers = if let Some(buffers) = live_cached {
            buffers
        } else {
            let sources = Safetensors::open_all(path).map_err(weight_error)?;
            let mut buffers = Vec::with_capacity(sources.len());
            let can_reuse = cached.len() == sources.len();
            for (index, source) in sources.into_iter().enumerate() {
                let imported = can_reuse
                    .then(|| cached[index].upgrade())
                    .flatten()
                    .map_or_else(|| self.import_weight_source(source), Ok);
                match imported {
                    Ok(buffer) => buffers.push(buffer),
                    Err(error) => {
                        release_buffers(
                            self.backend.as_ref(),
                            buffers,
                            self.active_profile.as_ref(),
                        )
                        .map_err(guest_error)?;
                        return Err(error);
                    }
                }
            }
            self.weight_files.insert(
                grant.to_owned(),
                buffers.iter().map(Arc::downgrade).collect(),
            );
            buffers
        };
        match self.table.push(WeightsEntry {
            buffers: buffers.clone(),
        }) {
            Ok(resource) => {
                self.live_handles += 1;
                Ok(resource)
            }
            Err(error) => {
                release_buffers(self.backend.as_ref(), buffers, self.active_profile.as_ref())
                    .map_err(guest_error)?;
                Err(invalid_handle(error))
            }
        }
    }

    fn import_weight_source(
        &self,
        source: Safetensors,
    ) -> Result<Arc<BufferHandle>, compute::Error> {
        let first = source
            .tensors()
            .iter()
            .find(|tensor| tensor.is_dtype_aligned())
            .ok_or_else(|| compute::Error::Layout("weight file contains no tensors".to_owned()))?;
        let region = source.mapped_region().map_err(weight_error)?;
        let buffer_len =
            u64::try_from(region.len()).map_err(|_| guest_error(BackendError::AllocationFailed))?;
        let owner_layout = first.layout(buffer_len).map_err(guest_error)?;
        let imported = self.backend.import_readonly(region).map_err(guest_error)?;
        let buffer_id = imported.buffer();
        let owner = self
            .backend
            .tensor(buffer_id, owner_layout)
            .map_err(guest_error)?;
        let copied_byte_len = imported.copied_byte_len();
        if let Some(byte_len) = copied_byte_len
            && let Err(error) = self.check_byte_allocation(byte_len)
        {
            self.backend.release(&owner).map_err(guest_error)?;
            return Err(error);
        }
        let buffer = Arc::new(BufferHandle {
            owner,
            kind: BufferKind::Weights {
                source,
                copied_byte_len,
                live_bytes: Arc::clone(&self.live_bytes),
            },
            taints: Arc::clone(&self.taints),
            guest_handles: AtomicUsize::new(0),
        });
        if let Some(byte_len) = copied_byte_len {
            self.live_bytes.fetch_add(byte_len, Ordering::AcqRel);
        }
        Ok(buffer)
    }

    fn weight_tensor(
        &mut self,
        resource: &Resource<WeightsEntry>,
        name: &str,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        let (buffer, metadata) = entry
            .buffers
            .iter()
            .find_map(|buffer| {
                let source = buffer.weights()?;
                source
                    .tensors()
                    .iter()
                    .find(|tensor| tensor.name() == name)
                    .map(|metadata| (buffer, metadata))
            })
            .ok_or_else(|| invalid_handle("weight tensor is not present"))?;
        self.check_tensor_shape(metadata.shape())?;
        let (tensor, buffer) = if metadata.is_dtype_aligned() {
            let layout = metadata
                .layout(buffer.owner.buffer().byte_len())
                .map_err(guest_error)?;
            let tensor = self
                .backend
                .tensor(buffer.owner.buffer(), layout)
                .map_err(guest_error)?;
            (tensor, Arc::clone(buffer))
        } else {
            let byte_len = self.check_allocation(metadata.dtype(), metadata.shape())?;
            let start = metadata.byte_offset();
            let end = start
                .checked_add(metadata.byte_len())
                .ok_or_else(|| guest_error(BackendError::AllocationFailed))?;
            let bytes = self
                .backend
                .read_buffer_range(buffer.owner.buffer(), start..end)
                .map_err(guest_error)?;
            let tensor = self
                .backend
                .alloc(metadata.dtype(), metadata.shape())
                .map_err(guest_error)?;
            if let Err(error) = self.backend.write(&tensor, &bytes) {
                let _ = self.backend.release(&tensor);
                return Err(guest_error(error));
            }
            let buffer = Arc::new(BufferHandle {
                owner: tensor.clone(),
                kind: BufferKind::CopiedWeight {
                    byte_len,
                    live_bytes: Arc::clone(&self.live_bytes),
                },
                taints: Arc::clone(&self.taints),
                guest_handles: AtomicUsize::new(0),
            });
            self.live_bytes.fetch_add(byte_len, Ordering::AcqRel);
            (tensor, buffer)
        };
        let retained_buffer = Arc::clone(&buffer);
        let tensor = match self.table.push(TensorEntry {
            tensor,
            symbolic: None,
            buffer,
        }) {
            Ok(resource) => resource,
            Err(error) => {
                release_buffer(
                    self.backend.as_ref(),
                    retained_buffer,
                    self.active_profile.as_ref(),
                )
                .map_err(guest_error)?;
                return Err(invalid_handle(error));
            }
        };
        self.table
            .get(&tensor)
            .map_err(invalid_handle)?
            .buffer
            .retain_guest_handle();
        self.live_handles += 1;
        Ok(tensor)
    }

    fn weight_info(
        &self,
        resource: &Resource<WeightsEntry>,
        name: &str,
    ) -> Result<compute::TensorInfo, compute::Error> {
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        let tensor = entry
            .buffers
            .iter()
            .filter_map(|buffer| buffer.weights())
            .flat_map(WeightSource::tensors)
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
        Ok(entry
            .buffers
            .iter()
            .filter_map(|buffer| buffer.weights())
            .flat_map(WeightSource::tensors)
            .map(|tensor| tensor.name().to_owned())
            .collect())
    }

    fn drop_weights(&mut self, resource: Resource<WeightsEntry>) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        let live_handles = self
            .live_handles
            .checked_sub(1)
            .ok_or_else(|| invalid_handle("live handle accounting underflowed"))?;
        release_buffers(
            self.backend.as_ref(),
            entry.buffers,
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
        let entry = self.entry(resource)?;
        if entry.symbolic.is_some() {
            return Err(compute::Error::Layout(
                "cannot write a symbolic tensor".to_owned(),
            ));
        }
        let tensor = &entry.tensor;
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
        entry
            .buffer
            .taints
            .wait_for_writes(tensor.buffer())
            .map_err(guest_error)?;
        self.backend.write(tensor, bytes).map_err(guest_error)
    }

    fn record_paged_dispatch(
        &mut self,
        commands: &Resource<CommandListEntry>,
        operation: (TemplateOp, Option<Op>),
        inputs: Vec<TensorEntry>,
        output: TensorEntry,
        protected: Vec<BufferId>,
        usage: PagedUse,
    ) -> Result<(), compute::Error> {
        let (template_op, concrete_op) = operation;
        let input_tensors = inputs.iter().map(|entry| &entry.tensor).collect::<Vec<_>>();
        let output_tensors = [&output.tensor];
        let dispatch_space = tensor_space(inputs.iter().chain([&output]))?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        merge_parameter_space(entry.space.as_ref(), dispatch_space.as_ref())?;
        if let Some(operation) = concrete_op {
            self.check_dispatch_work(operation, &input_tensors, &output_tensors)?;
        }
        let template_inputs = inputs
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_output = output.template()?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        if let Some(operation) = concrete_op
            && dispatch_space.is_none()
        {
            entry
                .commands
                .dispatch(operation, &input_tensors, &output.tensor)
                .map_err(guest_error)?;
            let dispatch = entry
                .commands
                .last_dispatch()
                .cloned()
                .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
            entry
                .recorded
                .push(RecordedDispatch::Static(Box::new(dispatch)));
        } else {
            entry.parameterized = true;
            entry.recorded.push(RecordedDispatch::Operation {
                op: template_op,
                inputs: template_inputs,
                outputs: vec![template_output],
            });
        }
        entry.space = entry.space.take().or(dispatch_space);
        entry.access.record(&inputs, std::slice::from_ref(&output));
        entry.protected.extend(protected);
        entry.retained.extend(inputs);
        entry.retained.push(output);
        usage.binding.used.store(true, Ordering::Release);
        if !entry
            .paged
            .iter()
            .any(|existing| Arc::ptr_eq(&existing.binding, &usage.binding))
        {
            entry.paged.push(usage);
        }
        Ok(())
    }

    fn paged_slot_entries(
        &self,
        slot: &Resource<BatchSlotEntry>,
        lane: u32,
        plane: u32,
    ) -> Result<
        (
            TensorEntry,
            TensorEntry,
            TensorEntry,
            forja_core::PagedSlot,
            PagedUse,
        ),
        compute::Error,
    > {
        let slot = self.table.get(slot).map_err(invalid_handle)?;
        let lane = usize::try_from(lane)
            .map_err(|_| compute::Error::Layout("paged lane exceeds usize".to_owned()))?;
        let plane = usize::try_from(plane)
            .map_err(|_| compute::Error::Layout("paged plane exceeds usize".to_owned()))?;
        let page_tensor = self
            .entry(&Resource::new_borrow(*slot.planes.get(plane).ok_or_else(
                || invalid_handle("paged plane is out of range"),
            )?))?
            .clone();
        let table_tensor = self
            .entry(&Resource::new_borrow(
                *slot
                    .tables
                    .first()
                    .and_then(|bank| bank.get(lane))
                    .ok_or_else(|| invalid_handle("paged lane is out of range"))?,
            ))?
            .clone();
        let generations = self.entry(&Resource::new_borrow(slot.generations))?.clone();
        let snapshot = slot
            .binding
            .banks
            .slot(0, lane)
            .map_err(|error| compute::Error::Layout(error.to_string()))?;
        if !snapshot.is_active() {
            return Err(compute::Error::Layout("paged lane is inactive".to_owned()));
        }
        Ok((
            page_tensor,
            table_tensor,
            generations,
            snapshot,
            PagedUse {
                binding: Arc::clone(&slot.binding),
                bank: 0,
            },
        ))
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
                recorded: Vec::new(),
                access: SubmissionAccess::default(),
                protected: HashSet::new(),
                space: None,
                parameterized: false,
                retained: Vec::new(),
                retained_kernels: Vec::new(),
                retained_epilogues: Vec::new(),
                paged: Vec::new(),
            })
            .map_err(invalid_handle)
    }

    fn graph_output(
        &mut self,
        commands: &Resource<CommandListEntry>,
        tensor: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        let buffer = self.entry(tensor)?.tensor.buffer();
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        if !entry
            .retained
            .iter()
            .any(|retained| retained.tensor.buffer() == buffer)
        {
            return Err(compute::Error::OpSignature(
                "graph output was not used by the command list".to_owned(),
            ));
        }
        entry.protected.insert(buffer);
        Ok(())
    }

    fn create_kernel(
        &mut self,
        source: compute::ProgramSource,
        signature: compute::KernelSignature,
    ) -> Result<Resource<KernelEntry>, compute::Error> {
        let validated = core_program(source)?;
        let signature = core_kernel_signature(signature);
        let lease = KernelLease::acquire(Arc::clone(&self.live_kernels), self.limits.live_kernels)?;
        let program =
            prepare_program_retained(self.backend.as_ref(), validated, signature, Box::new(lease))
                .map_err(prepare_error)?;
        self.table
            .push(KernelEntry { program })
            .map_err(invalid_handle)
    }

    fn drop_kernel(&mut self, resource: Resource<KernelEntry>) -> Result<(), compute::Error> {
        self.table
            .delete(resource)
            .map(|_| ())
            .map_err(invalid_handle)
    }

    fn create_epilogue(
        &mut self,
        source: compute::ProgramSource,
        signature: compute::EpilogueSignature,
    ) -> Result<Resource<EpilogueEntry>, compute::Error> {
        let validated = core_program(source)?;
        let signature = EpilogueSignature::new(
            signature.accumulator_count,
            signature.logical_rank,
            signature.extras.into_iter().map(core_dtype).collect(),
            core_dtype(signature.output),
        );
        let lease = KernelLease::acquire(Arc::clone(&self.live_kernels), self.limits.live_kernels)?;
        let epilogue = prepare_epilogue_retained(validated, signature, Box::new(lease))
            .map_err(|error| epilogue_error(&error))?;
        self.table
            .push(EpilogueEntry { epilogue })
            .map_err(invalid_handle)
    }

    fn drop_epilogue(&mut self, resource: Resource<EpilogueEntry>) -> Result<(), compute::Error> {
        self.table
            .delete(resource)
            .map(|_| ())
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
        self.dispatch_many(commands, operation, inputs, std::slice::from_ref(output))
    }

    /// Validates and records one trusted operation with multiple outputs.
    ///
    /// # Errors
    ///
    /// Returns an operation, quota, or invalid-handle error before recording invalid work.
    pub fn dispatch_many(
        &mut self,
        commands: &Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: &[Resource<TensorEntry>],
        outputs: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        let (template_op, concrete_op) = core_op(operation);
        #[cfg(target_os = "macos")]
        if (self.backend.as_ref() as &dyn Any).is::<forja_metal::MetalBackend>()
            && let Some(name) = forja_metal::MetalBackend::portable_variant_name(
                template_operation_kind(template_op),
            )
        {
            if self.explicit_metal_variants {
                return Err(compute::Error::OpSignature(format!(
                    "metal {} requires an explicit variant",
                    template_operation_kind(template_op)
                )));
            }
            return self.record_metal_variant(commands, operation, inputs, outputs, name);
        }
        self.dispatch_core(commands, template_op, concrete_op, inputs, outputs)
    }

    fn dispatch_core(
        &mut self,
        commands: &Resource<CommandListEntry>,
        template_op: TemplateOp,
        concrete_op: Option<Op>,
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
        let input_tensors = input_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let template_inputs = input_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_outputs = output_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let output_tensors = output_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let dispatch_space = tensor_space(input_entries.iter().chain(&output_entries))?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        merge_parameter_space(entry.space.as_ref(), dispatch_space.as_ref())?;
        if let Some(operation) = concrete_op {
            self.check_dispatch_work(operation, &input_tensors, &output_tensors)?;
        }

        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        if let Some(operation) = concrete_op
            && dispatch_space.is_none()
        {
            entry
                .commands
                .dispatch_many(operation, &input_tensors, &output_tensors)
                .map_err(guest_error)?;
            let dispatch = entry
                .commands
                .last_dispatch()
                .cloned()
                .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
            entry
                .recorded
                .push(RecordedDispatch::Static(Box::new(dispatch)));
        } else {
            entry.parameterized = true;
            entry.recorded.push(RecordedDispatch::Operation {
                op: template_op,
                inputs: template_inputs,
                outputs: template_outputs,
            });
        }
        entry.space = entry.space.take().or(dispatch_space);
        entry.access.record(&input_entries, &output_entries);
        entry.retained.extend(input_entries);
        entry.retained.extend(output_entries);
        Ok(())
    }

    /// Validates and records one guest-authored scalar program.
    ///
    /// # Errors
    ///
    /// Returns an operation, quota, or invalid-handle error before recording invalid work.
    pub fn dispatch_kernel(
        &mut self,
        commands: &Resource<CommandListEntry>,
        kernel: &Resource<KernelEntry>,
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
        let program = Arc::clone(&self.table.get(kernel).map_err(invalid_handle)?.program);
        let retained_kernel = KernelEntry {
            program: Arc::clone(&program),
        };
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
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
        let dispatch_space = tensor_space(input_entries.iter().chain(&output_entries))?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        merge_parameter_space(entry.space.as_ref(), dispatch_space.as_ref())?;
        if dispatch_space.is_none() {
            self.check_program_work(program.validated(), &input_tensors, &output_tensors)?;
        }
        let template_inputs = input_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_outputs = output_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        if dispatch_space.is_none() {
            entry
                .commands
                .dispatch_kernel(&program, &input_tensors, &output_tensors)
                .map_err(guest_error)?;
            let dispatch = entry
                .commands
                .last_dispatch()
                .cloned()
                .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
            entry
                .recorded
                .push(RecordedDispatch::Static(Box::new(dispatch)));
        } else {
            entry.parameterized = true;
            entry.recorded.push(RecordedDispatch::Program {
                program: Arc::clone(&program),
                inputs: template_inputs,
                outputs: template_outputs,
            });
        }
        entry.space = entry.space.take().or(dispatch_space);
        entry.access.record(&input_entries, &output_entries);
        entry.retained.extend(input_entries);
        entry.retained.extend(output_entries);
        entry.retained_kernels.push(retained_kernel);
        Ok(())
    }

    /// Validates and records a primitive with a guest-authored scalar epilogue.
    ///
    /// # Errors
    ///
    /// Returns an operation, quota, or invalid-handle error before recording invalid work.
    pub fn dispatch_epilogue(
        &mut self,
        commands: &Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: &[Resource<TensorEntry>],
        output: &Resource<TensorEntry>,
        epilogue: &Resource<EpilogueEntry>,
        extras: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        let (template_op, concrete_op) = core_op(operation);
        let operation = concrete_op.ok_or_else(|| {
            compute::Error::OpSignature("epilogue operation must be static".to_owned())
        })?;
        let input_entries = inputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let extra_entries = extras
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output_entry = self.entry(output)?.clone();
        let prepared = Arc::clone(self.table.get(epilogue).map_err(invalid_handle)?.prepared());
        let input_tensors = input_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let extra_tensors = extra_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let mut all_entries = input_entries.clone();
        all_entries.extend(extra_entries.clone());
        let dispatch_space = tensor_space(all_entries.iter().chain([&output_entry]))?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        merge_parameter_space(entry.space.as_ref(), dispatch_space.as_ref())?;
        self.check_dispatch_work(operation, &input_tensors, &[&output_entry.tensor])?;
        let template_inputs = input_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_extras = extra_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_output = output_entry.template()?;

        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        if dispatch_space.is_none() {
            entry
                .commands
                .dispatch_epilogue(
                    operation,
                    &input_tensors,
                    &output_entry.tensor,
                    &prepared,
                    &extra_tensors,
                )
                .map_err(guest_error)?;
            let dispatch = entry
                .commands
                .last_dispatch()
                .cloned()
                .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
            entry
                .recorded
                .push(RecordedDispatch::Static(Box::new(dispatch)));
        } else {
            entry.parameterized = true;
            entry.recorded.push(RecordedDispatch::Epilogue {
                op: template_op,
                inputs: template_inputs,
                output: Box::new(template_output),
                epilogue: Arc::clone(&prepared),
                extras: template_extras,
            });
        }
        entry.space = entry.space.take().or(dispatch_space);
        entry
            .access
            .record(&all_entries, std::slice::from_ref(&output_entry));
        entry.retained.extend(all_entries);
        entry.retained.push(output_entry);
        entry
            .retained_epilogues
            .push(EpilogueEntry { epilogue: prepared });
        Ok(())
    }

    fn drop_command_list(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        self.release_retained(entry.retained)
    }

    fn prepare_captured_graph(
        &mut self,
        graph: GraphTemplate,
        entry: &mut CommandListEntry,
    ) -> Result<PreparedGraph, compute::Error> {
        let reachable = entry
            .retained
            .iter()
            .filter(|retained| {
                retained.buffer.guest_reachable()
                    || retained.symbolic.is_some()
                    || entry.protected.contains(&retained.tensor.buffer())
            })
            .map(|entry| entry.tensor.buffer())
            .collect::<HashSet<_>>();
        let depth = self.backend.graph_memory_depth();
        if depth == 0 {
            return self.backend.prepare_graph(graph).map_err(guest_error);
        }
        let plan = graph
            .plan_memory(&reachable)
            .map_err(|error| graph_error(&error))?;
        if plan.arena_bytes() == 0 {
            return self.backend.prepare_graph(graph).map_err(guest_error);
        }
        let next_arena_bytes = self.graph_arena_bytes.max(plan.arena_bytes());
        let depth =
            u64::try_from(depth).map_err(|_| quota("graph arena depth exceeds the guest limit"))?;
        let old_charge = self
            .graph_arena_bytes
            .checked_mul(depth)
            .ok_or_else(|| quota("graph arena bytes exceed the guest limit"))?;
        let new_charge = next_arena_bytes
            .checked_mul(depth)
            .ok_or_else(|| quota("graph arena bytes exceed the guest limit"))?;
        let projected = self
            .live_bytes
            .load(Ordering::Acquire)
            .checked_sub(plan.transient_bytes())
            .and_then(|bytes| bytes.checked_sub(old_charge))
            .and_then(|bytes| bytes.checked_add(new_charge))
            .ok_or_else(|| quota("graph arena bytes exceed the guest limit"))?;
        if projected > self.limits.live_bytes {
            return Err(quota("graph arena bytes exceed the guest limit"));
        }
        let prepared = self
            .backend
            .prepare_graph_memory(graph, plan.clone())
            .map_err(guest_error)?;
        let (transient, retained): (Vec<_>, Vec<_>) = std::mem::take(&mut entry.retained)
            .into_iter()
            .partition(|entry| plan.contains(entry.tensor.buffer()));
        self.release_retained(transient)?;
        self.live_bytes
            .fetch_add(new_charge - old_charge, Ordering::AcqRel);
        self.graph_arena_bytes = next_arena_bytes;
        entry.retained = retained;
        Ok(prepared)
    }

    fn create_graph(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<Resource<GraphEntry>, compute::Error> {
        let mut entry = self.table.delete(resource).map_err(invalid_handle)?;
        let space = match entry.space.clone() {
            Some(space) => space,
            None => ParamSpace::new(Vec::new()).map_err(|error| graph_error(&error))?,
        };
        let mut graph = GraphTemplate::new(
            space,
            GraphLimits::new(
                self.limits.dispatches_per_list,
                self.limits.tensor_elements,
                self.limits.work_per_dispatch,
            ),
        );
        let built = entry
            .recorded
            .iter()
            .try_for_each(|dispatch| record_graph_dispatch(&mut graph, dispatch));
        let graph = match built {
            Ok(()) => self.prepare_captured_graph(graph, &mut entry),
            Err(error) => Err(graph_error(&error)),
        };
        let graph = match graph {
            Ok(graph) => graph,
            Err(error) => {
                self.release_retained(entry.retained)?;
                return Err(error);
            }
        };
        let lease =
            match GraphLease::acquire(Arc::clone(&self.live_graphs), self.limits.live_graphs) {
                Ok(lease) => lease,
                Err(error) => {
                    self.release_retained(entry.retained)?;
                    return Err(error);
                }
            };
        let graph_entry = GraphEntry {
            graph: Arc::new(graph),
            access: entry.access,
            retained: Vec::new(),
            retained_kernels: Vec::new(),
            retained_epilogues: Vec::new(),
            paged: entry.paged.clone(),
            _lease: lease,
        };
        let resource = match self.table.push(graph_entry) {
            Ok(resource) => resource,
            Err(error) => {
                self.release_retained(entry.retained)?;
                return Err(invalid_handle(error));
            }
        };
        let graph = self.table.get_mut(&resource).map_err(invalid_handle)?;
        graph.retained = entry.retained;
        graph.retained_kernels = entry.retained_kernels;
        graph.retained_epilogues = entry.retained_epilogues;
        graph.paged = entry.paged;
        Ok(resource)
    }

    fn drop_graph(&mut self, resource: Resource<GraphEntry>) -> Result<(), compute::Error> {
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
        if entry.parameterized {
            self.release_retained(entry.retained)?;
            return Err(compute::Error::OpSignature(
                "submit does not accept parameterized command lists".to_owned(),
            ));
        }
        Ok(SubmitRequest {
            backend: Arc::clone(&self.backend),
            work: SubmissionWork::Commands(entry.commands),
            access: entry.access,
            retained: entry.retained,
            paged: entry.paged,
            flight: None,
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
            queued_submissions: Arc::clone(&self.queued_submissions),
            completed_submissions: Arc::clone(&self.completed_submissions),
            timed_submissions: Arc::clone(&self.timed_submissions),
            completed_gpu_time_ns: Arc::clone(&self.completed_gpu_time_ns),
            submission_notify: Arc::clone(&self.submission_notify),
            taints: Arc::clone(&self.taints),
            profile: self.active_profile.clone(),
        })
    }

    #[cfg(test)]
    fn prepare_replay(
        &self,
        resource: &Resource<GraphEntry>,
        values: Vec<u32>,
    ) -> Result<SubmitRequest<B>, compute::Error> {
        let replay = self.replay_state(resource)?;
        let flight = ReplayFlight::acquire(replay)?;
        self.prepare_replay_with_flight(resource, values, flight)
    }

    fn replay_state(
        &self,
        resource: &Resource<GraphEntry>,
    ) -> Result<Arc<ReplayState>, compute::Error> {
        self.table.get(resource).map_err(invalid_handle)?;
        Ok(Arc::clone(&self.replay))
    }

    fn prepare_replay_with_flight(
        &self,
        resource: &Resource<GraphEntry>,
        values: Vec<u32>,
        flight: ReplayFlight,
    ) -> Result<SubmitRequest<B>, compute::Error> {
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        entry
            .graph
            .values(values.clone())
            .map_err(|error| graph_error(&error))?;
        Ok(SubmitRequest {
            backend: Arc::clone(&self.backend),
            work: SubmissionWork::Replay {
                graph: Arc::clone(&entry.graph),
                values,
            },
            access: entry.access.clone(),
            retained: Vec::new(),
            paged: entry.paged.clone(),
            flight: Some(flight),
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
            queued_submissions: Arc::clone(&self.queued_submissions),
            completed_submissions: Arc::clone(&self.completed_submissions),
            timed_submissions: Arc::clone(&self.timed_submissions),
            completed_gpu_time_ns: Arc::clone(&self.completed_gpu_time_ns),
            submission_notify: Arc::clone(&self.submission_notify),
            taints: Arc::clone(&self.taints),
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
        entry.buffer.release_guest_handle()?;
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
        self.check_byte_allocation(byte_len)?;
        Ok(byte_len)
    }

    fn check_byte_allocation(&self, byte_len: u64) -> Result<(), compute::Error> {
        if self
            .live_bytes
            .load(Ordering::Acquire)
            .checked_add(byte_len)
            .is_none_or(|bytes| bytes > self.limits.live_bytes)
        {
            return Err(quota("live tensor bytes exceed the guest limit"));
        }
        Ok(())
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
        outputs: &[&Tensor],
    ) -> Result<(), compute::Error> {
        let mut work = 0_u64;
        for tensor in inputs.iter().chain(outputs) {
            work = work
                .checked_add(self.check_tensor_shape(tensor.layout().shape())?)
                .ok_or_else(dispatch_work_quota)?;
        }
        let flops = match operation {
            Op::Matmul => inputs.first().zip(inputs.get(1)).and_then(|(left, right)| {
                matmul_flops(left.layout().shape(), right.layout().shape())
            }),
            Op::GatherMatmul => inputs
                .first()
                .zip(inputs.get(1))
                .zip(inputs.get(2))
                .and_then(|((input, weights), indices)| {
                    gather_matmul_flops(
                        input.layout().shape(),
                        weights.layout().shape(),
                        indices.layout().shape(),
                    )
                }),
            Op::QuantMatmul { .. } => {
                inputs
                    .first()
                    .zip(inputs.get(1))
                    .and_then(|(input, packed)| {
                        quant_matmul_flops(input.layout().shape(), packed.layout().shape())
                    })
            }
            Op::GatherQuantMatmul { .. } => {
                let indices = if inputs.len() == 8 { 7 } else { 4 };
                inputs
                    .first()
                    .zip(inputs.get(1))
                    .zip(inputs.get(indices))
                    .and_then(|((input, packed), indices)| {
                        gather_quant_matmul_dispatch_flops(
                            input.layout().shape(),
                            packed.layout().shape(),
                            indices.layout().shape(),
                            inputs.len(),
                        )
                    })
            }
            Op::Sdpa { .. } => inputs
                .first()
                .zip(inputs.get(1))
                .zip(inputs.get(2))
                .and_then(|((query, key), value)| {
                    sdpa_flops(
                        query.layout().shape(),
                        key.layout().shape(),
                        value.layout().shape(),
                    )
                }),
            _ => Some(0),
        }
        .ok_or_else(dispatch_work_quota)?;
        work = work.checked_add(flops).ok_or_else(dispatch_work_quota)?;
        if work > self.limits.work_per_dispatch {
            return Err(dispatch_work_quota());
        }
        Ok(())
    }

    fn check_program_work(
        &self,
        program: &ValidatedProgram,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), compute::Error> {
        let elements = outputs
            .first()
            .ok_or_else(dispatch_work_quota)
            .and_then(|output| self.check_tensor_shape(output.layout().shape()))?;
        let mut work = inputs
            .iter()
            .chain(outputs)
            .try_fold(0_u64, |work, tensor| {
                work.checked_add(self.check_tensor_shape(tensor.layout().shape())?)
                    .ok_or_else(dispatch_work_quota)
            })?;
        let reduction_passes = if program.program().kind == ProgramKind::Row {
            program
                .program()
                .insts
                .iter()
                .filter(|inst| matches!(inst, Inst::Reduce(_, _)))
                .count()
        } else {
            0
        };
        let passes = program
            .program()
            .insts
            .len()
            .checked_add(reduction_passes)
            .and_then(|passes| u64::try_from(passes).ok())
            .ok_or_else(dispatch_work_quota)?;
        work = work
            .checked_add(
                passes
                    .checked_mul(elements)
                    .ok_or_else(dispatch_work_quota)?,
            )
            .ok_or_else(dispatch_work_quota)?;
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
        if entry.symbolic.is_some() {
            return Err(compute::Error::Layout(
                "cannot read a symbolic tensor".to_owned(),
            ));
        }
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
            replay_output: false,
        })
    }

    fn prepare_replay_read(
        &self,
        resource: &Resource<TensorEntry>,
    ) -> Result<ReadRequest<B>, compute::Error> {
        let mut request = self.prepare_read(resource)?;
        request.replay_output = true;
        Ok(request)
    }
}

#[cfg(target_os = "macos")]
impl<B: Backend + 'static> Host<B> {
    fn metal_backend(&self) -> Result<&forja_metal::MetalBackend, compute::Error> {
        (self.backend.as_ref() as &dyn Any)
            .downcast_ref()
            .ok_or_else(|| compute::Error::OpSignature("Metal variants are unavailable".to_owned()))
    }

    fn record_metal_variant(
        &mut self,
        commands: &Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: &[Resource<TensorEntry>],
        outputs: &[Resource<TensorEntry>],
        name: &str,
    ) -> Result<(), compute::Error> {
        validate_variant_name_length(name)?;
        let input_entries = inputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output_entries = outputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let input_tensors = input_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let output_tensors = output_entries
            .iter()
            .map(|entry| &entry.tensor)
            .collect::<Vec<_>>();
        let template_inputs = input_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_outputs = output_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let dispatch_space = tensor_space(input_entries.iter().chain(&output_entries))?;
        let (template_op, concrete_op) = core_op(operation);
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        let command_space = entry.space.as_ref();
        merge_parameter_space(command_space, dispatch_space.as_ref())?;
        if let Some(operation) = concrete_op {
            self.check_dispatch_work(operation, &input_tensors, &output_tensors)?;
        }
        let variant = if let Some(operation) = concrete_op
            && dispatch_space.is_none()
        {
            self.metal_backend()?
                .validate_variant_for_op(operation, &input_tensors, &output_tensors, name)
                .map_err(|error| match error {
                    forja_metal::VariantValidationError::Operation(error) => guest_error(error),
                    forja_metal::VariantValidationError::Variant(error) => variant_error(&error),
                })?
        } else {
            let space = dispatch_space.as_ref().or(command_space).ok_or_else(|| {
                compute::Error::OpSignature(
                    "variant operation requires replay parameters".to_owned(),
                )
            })?;
            let mut graph = GraphTemplate::new(
                space.clone(),
                GraphLimits::new(
                    self.limits.dispatches_per_list,
                    self.limits.tensor_elements,
                    self.limits.work_per_dispatch,
                ),
            );
            let input_refs = template_inputs.iter().collect::<Vec<_>>();
            let output_refs = template_outputs.iter().collect::<Vec<_>>();
            graph
                .dispatch_many(template_op, &input_refs, &output_refs)
                .map_err(|error| graph_error(&error))?;
            self.metal_backend()?
                .validate_template_variant(space, template_op, &input_refs, &output_refs, name)
                .map_err(|error| variant_error(&error))?
        };
        self.dispatch_core(commands, template_op, concrete_op, inputs, outputs)?;
        let data = BackendDispatchData::new(variant);
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        let recorded = entry
            .recorded
            .pop()
            .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
        match recorded {
            RecordedDispatch::Static(mut dispatch) => {
                dispatch.set_backend_data(data.clone());
                if !entry.commands.set_last_backend_data(data) {
                    return Err(guest_error(BackendError::ExecutionFailed));
                }
                entry.recorded.push(RecordedDispatch::Static(dispatch));
            }
            RecordedDispatch::Operation {
                op,
                inputs,
                outputs,
            } => entry.recorded.push(RecordedDispatch::Variant {
                op,
                inputs,
                outputs,
                data,
            }),
            _ => return Err(guest_error(BackendError::ExecutionFailed)),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_metal_variant_rule(
        &mut self,
        commands: &Resource<CommandListEntry>,
        params: &Resource<ParamsEntry>,
        parameter: u8,
        arms: &[metal_bindings::l9o::gpu::metal_variants::Arm],
        operation: compute::Op,
        inputs: &[Resource<TensorEntry>],
        outputs: &[Resource<TensorEntry>],
    ) -> Result<(), compute::Error> {
        validate_variant_rule_boundary(arms)?;
        let space = self
            .table
            .get(params)
            .map_err(invalid_handle)?
            .space
            .clone();
        let rule = forja_core::VariantRule::new(
            &space,
            parameter,
            arms.iter()
                .map(|arm| RuleArm::new(arm.lo, arm.hi, &arm.name))
                .collect(),
        )
        .map_err(|error| variant_error(&error))?;
        let input_entries = inputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output_entries = outputs
            .iter()
            .map(|resource| self.entry(resource).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let input_tensors = entry_tensors(&input_entries);
        let output_tensors = entry_tensors(&output_entries);
        validate_variant_rule_space(
            tensor_space(input_entries.iter().chain(&output_entries))?,
            &space,
        )?;
        let template_inputs = input_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let template_outputs = output_entries
            .iter()
            .map(TensorEntry::template)
            .collect::<Result<Vec<_>, _>>()?;
        let input_refs = template_inputs.iter().collect::<Vec<_>>();
        let output_refs = template_outputs.iter().collect::<Vec<_>>();
        let (template_op, concrete_op) = core_op(operation);
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(quota("command list dispatch count exceeds the guest limit"));
        }
        merge_parameter_space(entry.space.as_ref(), Some(&space))?;
        if let Some(operation) = concrete_op {
            self.check_dispatch_work(operation, &input_tensors, &output_tensors)?;
        }
        let mut graph = GraphTemplate::new(
            space.clone(),
            GraphLimits::new(
                self.limits.dispatches_per_list,
                self.limits.tensor_elements,
                self.limits.work_per_dispatch,
            ),
        );
        graph
            .dispatch_many(template_op, &input_refs, &output_refs)
            .map_err(|error| graph_error(&error))?;
        let variants = self
            .metal_backend()?
            .validate_variant_rule(&space, &rule, template_op, &input_refs, &output_refs)
            .map_err(|error| variant_error(&error))?;
        self.dispatch_core(commands, template_op, concrete_op, inputs, outputs)?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        merge_parameter_space(entry.space.as_ref(), Some(&space))?;
        let recorded = entry
            .recorded
            .pop()
            .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
        let (op, inputs, outputs) = match recorded {
            RecordedDispatch::Static(dispatch) => (
                TemplateOp::from(dispatch.op()),
                template_inputs,
                template_outputs,
            ),
            RecordedDispatch::Operation {
                op,
                inputs,
                outputs,
            } => (op, inputs, outputs),
            _ => return Err(guest_error(BackendError::ExecutionFailed)),
        };
        entry.recorded.push(RecordedDispatch::VariantRule {
            op,
            inputs,
            outputs,
            parameter,
            arms: rule
                .arms()
                .iter()
                .zip(variants)
                .map(|(arm, variant)| (arm.hi(), BackendDispatchData::new(variant)))
                .collect(),
        });
        entry.parameterized = true;
        entry.space = Some(space);
        Ok(())
    }

    fn select_metal_epilogue_variant(
        &mut self,
        commands: &Resource<CommandListEntry>,
        name: &str,
    ) -> Result<(), compute::Error> {
        validate_variant_name_length(name)?;
        let variant = {
            let entry = self.table.get(commands).map_err(invalid_handle)?;
            match entry.recorded.last() {
                Some(RecordedDispatch::Static(dispatch))
                    if dispatch.prepared_epilogue().is_some() =>
                {
                    self.metal_backend()?
                        .validate_variant_for_op(
                            dispatch.op(),
                            &dispatch.primitive_inputs().iter().collect::<Vec<_>>(),
                            &dispatch.outputs().iter().collect::<Vec<_>>(),
                            name,
                        )
                        .map_err(|error| match error {
                            forja_metal::VariantValidationError::Operation(error) => {
                                guest_error(error)
                            }
                            forja_metal::VariantValidationError::Variant(error) => {
                                variant_error(&error)
                            }
                        })?
                }
                Some(RecordedDispatch::Epilogue {
                    op, inputs, output, ..
                }) => {
                    let space = entry.space.as_ref().ok_or_else(|| {
                        compute::Error::OpSignature(
                            "epilogue variant requires replay parameters".to_owned(),
                        )
                    })?;
                    self.metal_backend()?
                        .validate_template_variant(
                            space,
                            *op,
                            &inputs.iter().collect::<Vec<_>>(),
                            &[output.as_ref()],
                            name,
                        )
                        .map_err(|error| variant_error(&error))?
                }
                _ => return Err(guest_error(BackendError::InvalidInput)),
            }
        };
        let data = BackendDispatchData::new(variant);
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        let recorded = entry
            .recorded
            .pop()
            .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
        match recorded {
            RecordedDispatch::Static(mut dispatch) => {
                dispatch.set_backend_data(data.clone());
                if !entry.commands.set_last_backend_data(data) {
                    return Err(guest_error(BackendError::ExecutionFailed));
                }
                entry.recorded.push(RecordedDispatch::Static(dispatch));
            }
            RecordedDispatch::Epilogue {
                op,
                inputs,
                output,
                epilogue,
                extras,
            } => entry.recorded.push(RecordedDispatch::EpilogueVariant {
                op,
                inputs,
                output,
                epilogue,
                extras,
                data,
            }),
            _ => return Err(guest_error(BackendError::ExecutionFailed)),
        }
        Ok(())
    }

    fn select_metal_epilogue_variant_rule(
        &mut self,
        commands: &Resource<CommandListEntry>,
        params: &Resource<ParamsEntry>,
        parameter: u8,
        arms: &[metal_bindings::l9o::gpu::metal_variants::Arm],
    ) -> Result<(), compute::Error> {
        validate_variant_rule_boundary(arms)?;
        let space = self
            .table
            .get(params)
            .map_err(invalid_handle)?
            .space
            .clone();
        let rule = forja_core::VariantRule::new(
            &space,
            parameter,
            arms.iter()
                .map(|arm| RuleArm::new(arm.lo, arm.hi, &arm.name))
                .collect(),
        )
        .map_err(|error| variant_error(&error))?;
        let variants = {
            let entry = self.table.get(commands).map_err(invalid_handle)?;
            validate_variant_rule_space(entry.space.clone(), &space)?;
            let Some(RecordedDispatch::Epilogue {
                op, inputs, output, ..
            }) = entry.recorded.last()
            else {
                return Err(guest_error(BackendError::InvalidInput));
            };
            self.metal_backend()?
                .validate_variant_rule(
                    &space,
                    &rule,
                    *op,
                    &inputs.iter().collect::<Vec<_>>(),
                    &[output.as_ref()],
                )
                .map_err(|error| variant_error(&error))?
        };
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        let recorded = entry
            .recorded
            .pop()
            .ok_or_else(|| guest_error(BackendError::ExecutionFailed))?;
        let RecordedDispatch::Epilogue {
            op,
            inputs,
            output,
            epilogue,
            extras,
        } = recorded
        else {
            return Err(guest_error(BackendError::ExecutionFailed));
        };
        entry.recorded.push(RecordedDispatch::EpilogueVariantRule {
            op,
            inputs,
            output,
            epilogue,
            extras,
            parameter,
            arms: rule
                .arms()
                .iter()
                .zip(variants)
                .map(|(arm, variant)| (arm.hi(), BackendDispatchData::new(variant)))
                .collect(),
        });
        entry.parameterized = true;
        Ok(())
    }
}

struct ReadRequest<B: Backend> {
    backend: Arc<B>,
    tensor: Tensor,
    buffer: Arc<BufferHandle>,
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
    replay_output: bool,
}

struct SubmitRequest<B: Backend> {
    backend: Arc<B>,
    work: SubmissionWork,
    access: SubmissionAccess,
    retained: Vec<TensorEntry>,
    paged: Vec<PagedUse>,
    flight: Option<ReplayFlight>,
    timeout: Duration,
    gpu_time_budget_ns: u64,
    gpu_time_ns: Arc<AtomicU64>,
    queued_submissions: Arc<AtomicU64>,
    completed_submissions: Arc<AtomicU64>,
    timed_submissions: Arc<AtomicU64>,
    completed_gpu_time_ns: Arc<AtomicU64>,
    submission_notify: Arc<tokio::sync::Notify>,
    taints: Arc<BufferTaints>,
    profile: Option<Arc<Mutex<EngineStepProfile>>>,
}

enum SubmissionWork {
    Commands(CommandList),
    Replay {
        graph: Arc<PreparedGraph>,
        values: Vec<u32>,
    },
}

#[derive(Clone)]
struct PagedUse {
    binding: Arc<PagedBinding>,
    bank: usize,
}

impl std::fmt::Debug for PagedUse {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PagedUse")
            .field("bank", &self.bank)
            .finish_non_exhaustive()
    }
}

impl PagedUse {
    fn mark_all(usages: &[Self], serial: Option<u64>) -> Result<(), BackendError> {
        serial.map_or(Ok(()), |serial| {
            usages
                .iter()
                .try_for_each(|usage| usage.mark_submitted(serial))
        })
    }

    fn observe_all(
        usages: &[Self],
        serial: Option<u64>,
        completed: bool,
    ) -> Result<(), BackendError> {
        if !completed {
            return Ok(());
        }
        serial.map_or(Ok(()), |serial| {
            usages
                .iter()
                .try_for_each(|usage| usage.observe_completion(serial))
        })
    }

    fn mark_submitted(&self, serial: u64) -> Result<(), BackendError> {
        self.binding.used.store(true, Ordering::Release);
        let sequence = self
            .binding
            .sequence
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .as_ref()
            .cloned()
            .ok_or(BackendError::ExecutionFailed)?;
        sequence
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .mark_submitted(serial)
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.binding
            .banks
            .mark_submitted(self.bank, serial)
            .map_err(|_| BackendError::ExecutionFailed)
    }

    fn observe_completion(&self, serial: u64) -> Result<(), BackendError> {
        self.binding
            .pool
            .observe_completion(serial)
            .map_err(|_| BackendError::ExecutionFailed)
    }
}

#[derive(Clone, Debug, Default)]
struct SubmissionAccess {
    reads: HashSet<BufferId>,
    writes: HashSet<BufferId>,
}

impl SubmissionAccess {
    fn record(&mut self, inputs: &[TensorEntry], outputs: &[TensorEntry]) {
        self.reads
            .extend(inputs.iter().map(|entry| entry.tensor.buffer()));
        self.writes
            .extend(outputs.iter().map(|entry| entry.tensor.buffer()));
    }
}

#[derive(Debug, Default)]
struct BufferTaints {
    state: Mutex<BufferTaintState>,
}

#[derive(Debug, Default)]
struct BufferTaintState {
    tainted: HashMap<BufferId, BackendError>,
    pending_writes: HashMap<BufferId, Vec<Weak<SubmissionOutcome>>>,
}

#[derive(Debug, Default)]
struct SubmissionOutcome {
    result: Mutex<Option<Result<(), BackendError>>>,
    ready: Condvar,
}

impl SubmissionOutcome {
    fn wait(&self) -> Result<(), BackendError> {
        let mut result = self
            .result
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        while result.is_none() {
            result = self
                .ready
                .wait(result)
                .map_err(|_| BackendError::ExecutionFailed)?;
        }
        result.unwrap_or(Err(BackendError::ExecutionFailed))
    }

    fn complete(&self, result: Result<(), BackendError>) {
        if let Ok(mut target) = self.result.lock() {
            *target = Some(result);
        }
        self.ready.notify_all();
    }
}

impl BufferTaints {
    fn begin(self: &Arc<Self>, access: &SubmissionAccess) -> Result<TaintFlight, BackendError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if let Some(error) = access
            .reads
            .iter()
            .find_map(|buffer| state.tainted.get(buffer).copied())
        {
            return Err(error);
        }
        let mut seen = HashSet::new();
        let mut dependencies = Vec::new();
        for buffer in &access.reads {
            if let Some(pending) = state.pending_writes.get_mut(buffer) {
                pending.retain(|outcome| outcome.strong_count() != 0);
                dependencies.extend(
                    pending
                        .iter()
                        .filter_map(Weak::upgrade)
                        .filter(|outcome| seen.insert(Arc::as_ptr(outcome) as usize)),
                );
            }
        }
        let outcome = Arc::new(SubmissionOutcome::default());
        for buffer in &access.writes {
            state
                .pending_writes
                .entry(*buffer)
                .or_default()
                .push(Arc::downgrade(&outcome));
        }
        Ok(TaintFlight {
            owner: Arc::clone(self),
            outcome,
            dependencies,
            writes: access.writes.iter().copied().collect(),
            finished: false,
        })
    }

    fn wait_for_writes(&self, buffer: BufferId) -> Result<(), BackendError> {
        let pending = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            if let Some(error) = state.tainted.get(&buffer) {
                return Err(*error);
            }
            let Some(pending) = state.pending_writes.get_mut(&buffer) else {
                return Ok(());
            };
            pending.retain(|outcome| outcome.strong_count() != 0);
            pending.iter().filter_map(Weak::upgrade).collect::<Vec<_>>()
        };
        for outcome in pending {
            outcome.wait()?;
        }
        self.state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .tainted
            .get(&buffer)
            .copied()
            .map_or(Ok(()), Err)
    }

    fn check(&self, buffer: BufferId) -> Result<(), BackendError> {
        self.state
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .tainted
            .get(&buffer)
            .copied()
            .map_or(Ok(()), Err)
    }

    fn release(&self, buffer: BufferId) {
        if let Ok(mut state) = self.state.lock() {
            state.tainted.remove(&buffer);
            state.pending_writes.remove(&buffer);
        }
    }
}

struct TaintFlight {
    owner: Arc<BufferTaints>,
    outcome: Arc<SubmissionOutcome>,
    dependencies: Vec<Arc<SubmissionOutcome>>,
    writes: Vec<BufferId>,
    finished: bool,
}

impl TaintFlight {
    fn finish<T>(
        mut self,
        mut result: Result<T, BackendError>,
        committed: bool,
    ) -> Result<T, BackendError> {
        if result.is_ok() {
            for dependency in &self.dependencies {
                if let Err(error) = dependency.wait() {
                    result = Err(error);
                    break;
                }
            }
        }
        if committed
            && let Err(error) = result
            && let Ok(mut state) = self.owner.state.lock()
        {
            for buffer in &self.writes {
                state.tainted.entry(*buffer).or_insert(error);
            }
        }
        self.outcome
            .complete(result.as_ref().map(|_| ()).map_err(|error| *error));
        self.finished = true;
        result
    }
}

impl Drop for TaintFlight {
    fn drop(&mut self) {
        if !self.finished {
            if let Ok(mut state) = self.owner.state.lock() {
                for buffer in &self.writes {
                    state
                        .tainted
                        .entry(*buffer)
                        .or_insert(BackendError::ExecutionFailed);
                }
            }
            self.outcome.complete(Err(BackendError::ExecutionFailed));
        }
    }
}

#[derive(Debug)]
struct ReplayState {
    limit: usize,
    status: Mutex<ReplayStatus>,
    available: tokio::sync::Notify,
    #[cfg(test)]
    drop_barrier: Mutex<Option<Arc<std::sync::Barrier>>>,
}

impl ReplayState {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            status: Mutex::new(ReplayStatus::default()),
            available: tokio::sync::Notify::new(),
            #[cfg(test)]
            drop_barrier: Mutex::new(None),
        }
    }
}

#[derive(Debug, Default)]
struct ReplayStatus {
    in_flight: usize,
}

struct ReplayFlight(Arc<ReplayState>);

impl ReplayFlight {
    #[cfg(test)]
    fn acquire(replay: Arc<ReplayState>) -> Result<Self, compute::Error> {
        Self::try_acquire(replay)?.ok_or_else(|| {
            compute::Error::OpSignature("graph replay depth exceeds host limit".to_owned())
        })
    }

    async fn acquire_wait(replay: Arc<ReplayState>) -> Result<Self, compute::Error> {
        loop {
            let notified = replay.available.notified();
            if let Some(flight) = Self::try_acquire(Arc::clone(&replay))? {
                return Ok(flight);
            }
            notified.await;
        }
    }

    fn try_acquire(replay: Arc<ReplayState>) -> Result<Option<Self>, compute::Error> {
        let mut status = replay
            .status
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
        if replay.limit != 0 && status.in_flight >= replay.limit {
            return Ok(None);
        }
        status.in_flight = status.in_flight.checked_add(1).ok_or_else(|| {
            compute::Error::OpSignature("graph replay depth exceeds host limit".to_owned())
        })?;
        drop(status);
        Ok(Some(Self(replay)))
    }

    fn finish<T>(&self, result: Result<T, BackendError>) -> Result<T, BackendError> {
        let status = self
            .0
            .status
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        drop(status);
        result
    }
}

impl Drop for ReplayFlight {
    fn drop(&mut self) {
        #[cfg(test)]
        if let Ok(mut hook) = self.0.drop_barrier.lock()
            && let Some(barrier) = hook.take()
        {
            drop(hook);
            barrier.wait();
            barrier.wait();
        }
        if let Ok(mut status) = self.0.status.lock() {
            status.in_flight = status.in_flight.saturating_sub(1);
            drop(status);
            self.0.available.notify_one();
        }
    }
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
        self.spawn(None)
            .await
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?
            .map_err(guest_error)
    }

    async fn run_deferred(self) -> Result<Option<u64>, compute::Error> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = self.spawn(Some(sender));
        let started = receiver
            .await
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
        drop(task);
        started.map(|()| None).map_err(guest_error)
    }

    fn spawn(
        self,
        started: Option<tokio::sync::oneshot::Sender<Result<(), BackendError>>>,
    ) -> tokio::task::JoinHandle<Result<Option<u64>, BackendError>> {
        let reservation = GpuReservation::new(
            Arc::clone(&self.gpu_time_ns),
            self.gpu_time_budget_ns,
            duration_ns(self.timeout).max(1),
        );
        tokio::task::spawn_blocking(move || {
            let Self {
                backend,
                work,
                access,
                retained,
                paged,
                flight,
                timeout,
                gpu_time_budget_ns: _,
                gpu_time_ns: _,
                queued_submissions,
                completed_submissions,
                timed_submissions,
                completed_gpu_time_ns,
                submission_notify,
                taints,
                profile,
            } = self;
            let submitted = match reservation {
                Err(error) => Err(error),
                Ok(reservation) => match taints.begin(&access) {
                    Err(error) => Err(error),
                    Ok(taint) => match submit_work(backend.as_ref(), work, profile.is_some()) {
                        Err(error) => {
                            let error = taint
                                .finish::<()>(Err(error), false)
                                .err()
                                .unwrap_or(BackendError::ExecutionFailed);
                            Err(error)
                        }
                        Ok(submission) => {
                            let serial = submission.completion_serial();
                            let marked = PagedUse::mark_all(&paged, serial);
                            Ok((submission, reservation, taint, serial, marked))
                        }
                    },
                },
            };
            let (submission, reservation, taint, serial, marked) = match submitted {
                Ok(submitted) => {
                    publish_queued_submission(&queued_submissions, started);
                    submitted
                }
                Err(error) => {
                    let release = release_retained(backend.as_ref(), retained, profile.as_ref());
                    let result = combine_submission_release(Err(error), release);
                    let result = match &flight {
                        Some(flight) => flight.finish(result),
                        None => result,
                    };
                    drop(flight);
                    if let Some(started) = started {
                        let _ = started.send(result.as_ref().map(|_| ()).map_err(|error| *error));
                    }
                    return result;
                }
            };
            let wait_started = Instant::now();
            let wait = submission.wait_timeout(timeout);
            let observed =
                PagedUse::observe_all(&paged, serial, wait != Err(BackendError::Timeout));
            let wall_time = duration_ns(wait_started.elapsed()).max(1);
            let gpu_time = submission.gpu_time().map(duration_ns);
            let submission_profile = submission.profile();
            let charged = submission_charge(&wait, reservation.amount(), gpu_time, wall_time);
            let accounting = reservation.settle(charged);
            let result = combine_submission_results(wait, accounting, marked, observed).map(|()| {
                if let (Some(target), Some(submission_profile)) = (&profile, submission_profile)
                    && let Ok(mut target) = target.lock()
                {
                    target.submission = Some(submission_profile);
                }
                gpu_time
            });
            let release = release_retained(backend.as_ref(), retained, profile.as_ref());
            let result = combine_submission_release(result, release);
            let result = match &flight {
                Some(flight) => flight.finish(result),
                None => result,
            };
            let result = taint.finish(result, true);
            drop(flight);
            if let Ok(gpu_time) = result {
                if let Some(gpu_time) = gpu_time {
                    saturating_increment(&timed_submissions);
                    saturating_add(&completed_gpu_time_ns, gpu_time);
                }
                saturating_increment(&completed_submissions);
                submission_notify.notify_waiters();
            }
            result
        })
    }
}

fn submit_work<B: Backend>(
    backend: &B,
    work: SubmissionWork,
    profiled: bool,
) -> Result<B::Submission, BackendError> {
    match work {
        SubmissionWork::Commands(commands) if profiled => backend.submit_profiled(commands),
        SubmissionWork::Commands(commands) => backend.submit(commands),
        SubmissionWork::Replay { graph, values } if profiled => {
            backend.replay_profiled(&graph, values)
        }
        SubmissionWork::Replay { graph, values } => backend.replay(&graph, values),
    }
}

fn combine_submission_results(
    wait: Result<(), BackendError>,
    accounting: Result<(), BackendError>,
    marked: Result<(), BackendError>,
    observed: Result<(), BackendError>,
) -> Result<(), BackendError> {
    match (wait, accounting, marked, observed) {
        (Err(error), _, _, _)
        | (Ok(()), Err(error), _, _)
        | (Ok(()), Ok(()), Err(error), _)
        | (Ok(()), Ok(()), Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(()), Ok(()), Ok(())) => Ok(()),
    }
}

fn submission_charge(
    wait: &Result<(), BackendError>,
    reserved: u64,
    gpu_time: Option<u64>,
    wall_time: u64,
) -> u64 {
    if *wait == Err(BackendError::Timeout) {
        reserved
    } else {
        gpu_time.unwrap_or(wall_time).max(1)
    }
}

fn combine_submission_release<T>(
    result: Result<T, BackendError>,
    release: Result<(), BackendError>,
) -> Result<T, BackendError> {
    match (result, release) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
        Some(value.saturating_add(1))
    });
}

fn publish_queued_submission(
    queued_submissions: &AtomicU64,
    started: Option<tokio::sync::oneshot::Sender<Result<(), BackendError>>>,
) {
    saturating_increment(queued_submissions);
    if let Some(started) = started {
        let _ = started.send(Ok(()));
    }
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
                replay_output,
            } = self;
            let taint = if replay_output {
                buffer.taints.check(tensor.buffer())
            } else {
                buffer.taints.wait_for_writes(tensor.buffer())
            };
            if let Err(error) = taint {
                let release = release_buffer(backend.as_ref(), buffer, profile.as_ref());
                return combine_submission_release(Err(error), release);
            }
            let result = if replay_output {
                backend.read_replay_output(&tensor)
            } else {
                backend.read(&tensor)
            };
            let result = match result {
                Ok(bytes) => buffer.taints.check(tensor.buffer()).map(|()| bytes),
                Err(error) => Err(error),
            };
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

    fn view_param(
        &mut self,
        resource: Resource<TensorEntry>,
        space: Resource<ParamsEntry>,
        slices: Vec<compute::ParamSlice>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<TensorEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(Host::view_param(self, &resource, &space, slices)))
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

impl<B> compute::HostParams for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn new(
        &mut self,
        ranges: Vec<compute::ParamRange>,
    ) -> impl Future<Output = wasmtime::Result<Resource<ParamsEntry>>> + Send {
        std::future::ready(self.params(ranges))
    }

    fn drop(
        &mut self,
        resource: Resource<ParamsEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(self.drop_params(resource))
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

    fn output(
        &mut self,
        resource: Resource<CommandListEntry>,
        value: Resource<TensorEntry>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        std::future::ready(Ok(self.graph_output(&resource, &value)))
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

    fn dispatch_many(
        &mut self,
        resource: Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: Vec<Resource<TensorEntry>>,
        outputs: Vec<Resource<TensorEntry>>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(Host::dispatch_many(
            self, &resource, operation, &inputs, &outputs,
        )))
    }

    fn dispatch_kernel(
        &mut self,
        resource: Resource<CommandListEntry>,
        kernel: Resource<KernelEntry>,
        inputs: Vec<Resource<TensorEntry>>,
        outputs: Vec<Resource<TensorEntry>>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(Host::dispatch_kernel(
            self, &resource, &kernel, &inputs, &outputs,
        )))
    }

    fn dispatch_epilogue(
        &mut self,
        resource: Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: Vec<Resource<TensorEntry>>,
        output: Resource<TensorEntry>,
        epilogue: Resource<EpilogueEntry>,
        extras: Vec<Resource<TensorEntry>>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(Host::dispatch_epilogue(
            self, &resource, operation, &inputs, &output, &epilogue, &extras,
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

#[cfg(target_os = "macos")]
impl<B> metal_bindings::l9o::gpu::metal_variants::Host for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn record_variant(
        &mut self,
        commands: Resource<CommandListEntry>,
        operation: compute::Op,
        inputs: Vec<Resource<TensorEntry>>,
        outputs: Vec<Resource<TensorEntry>>,
        name: String,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(
            self.record_metal_variant(&commands, operation, &inputs, &outputs, &name)
        ))
    }

    fn record_variant_rule(
        &mut self,
        commands: Resource<CommandListEntry>,
        space: Resource<ParamsEntry>,
        parameter: u8,
        arms: Vec<metal_bindings::l9o::gpu::metal_variants::Arm>,
        operation: compute::Op,
        inputs: Vec<Resource<TensorEntry>>,
        outputs: Vec<Resource<TensorEntry>>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(self.record_metal_variant_rule(
            &commands, &space, parameter, &arms, operation, &inputs, &outputs,
        )))
    }

    fn select_epilogue_variant(
        &mut self,
        commands: Resource<CommandListEntry>,
        name: String,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(self.select_metal_epilogue_variant(&commands, &name)))
    }

    fn select_epilogue_variant_rule(
        &mut self,
        commands: Resource<CommandListEntry>,
        space: Resource<ParamsEntry>,
        parameter: u8,
        arms: Vec<metal_bindings::l9o::gpu::metal_variants::Arm>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let _timer = self.import_timer(ImportKind::Dispatch);
        std::future::ready(Ok(
            self.select_metal_epilogue_variant_rule(&commands, &space, parameter, &arms)
        ))
    }
}

impl<B> compute::HostKernel for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn create(
        &mut self,
        source: compute::ProgramSource,
        signature: compute::KernelSignature,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<KernelEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.create_kernel(source, signature)))
    }

    fn drop(
        &mut self,
        resource: Resource<KernelEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(self.drop_kernel(resource).map_err(wasmtime::Error::msg))
    }
}

impl<B> compute::HostEpilogue for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn create(
        &mut self,
        source: compute::ProgramSource,
        signature: compute::EpilogueSignature,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<EpilogueEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.create_epilogue(source, signature)))
    }

    fn drop(
        &mut self,
        resource: Resource<EpilogueEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(self.drop_epilogue(resource).map_err(wasmtime::Error::msg))
    }
}

impl<B> compute::HostGraph for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn create(
        &mut self,
        commands: Resource<CommandListEntry>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<GraphEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.create_graph(commands)))
    }

    fn drop(
        &mut self,
        resource: Resource<GraphEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(self.drop_graph(resource).map_err(wasmtime::Error::msg))
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
    fn describe_device(
        &mut self,
    ) -> impl Future<Output = wasmtime::Result<Result<compute::DeviceInfo, compute::Error>>> + Send
    {
        std::future::ready(Ok(self
            .backend
            .device_info()
            .map(guest_device_info)
            .map_err(guest_error)))
    }

    fn open_weights(
        &mut self,
        grant: String,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<WeightsEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.open_weights(&grant)))
    }
}

impl<B> paged_storage::HostPagePool for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn open_slot(
        &mut self,
        resource: Resource<PagePoolEntry>,
        planes: u32,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<BatchSlotEntry>, compute::Error>>> + Send
    {
        std::future::ready(Ok(self.open_paged_slot(&resource, planes)))
    }

    fn drop(
        &mut self,
        resource: Resource<PagePoolEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(
            self.table
                .delete(resource)
                .map(|_| ())
                .map_err(wasmtime::Error::msg),
        )
    }
}

impl<B> paged_storage::HostSequence for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn drop(
        &mut self,
        resource: Resource<PagedSequenceEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        std::future::ready(
            self.table
                .delete(resource)
                .map(|_| ())
                .map_err(wasmtime::Error::msg),
        )
    }
}

impl<B> paged_storage::HostBatchSlot for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn drop(
        &mut self,
        resource: Resource<BatchSlotEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        let result = self
            .table
            .delete(resource)
            .map_err(invalid_handle)
            .map(|entry| {
                let mut resources = entry.planes;
                resources.extend(entry.tables.into_iter().flatten());
                resources.push(entry.generations);
                self.drop_paged_handles(resources);
            });
        std::future::ready(result.map_err(wasmtime::Error::msg))
    }
}

impl<B> paged_storage::Host for Host<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn record_paged_copy(
        &mut self,
        commands: Resource<CommandListEntry>,
        slot: Resource<BatchSlotEntry>,
        lane: u32,
        span: paged_storage::PagedSpan,
        ordinary: Resource<TensorEntry>,
        direction: paged_storage::CopyDirection,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let result = (|| {
            let (pages, table, generations, _, usage) =
                self.paged_slot_entries(&slot, lane, span.plane)?;
            let ordinary = self.entry(&ordinary)?.clone();
            let start = core_affine(span.start);
            let tokens = ordinary
                .tensor
                .layout()
                .shape()
                .first()
                .copied()
                .ok_or_else(|| compute::Error::Layout("paged copy input is scalar".to_owned()))?;
            if u64::from(tokens) != span.len {
                return Err(compute::Error::Layout(
                    "paged copy span length does not match the tensor".to_owned(),
                ));
            }
            let direction = match direction {
                paged_storage::CopyDirection::IntoPages => PagedCopyDirection::IntoPages,
                paged_storage::CopyDirection::OutOfPages => PagedCopyDirection::OutOfPages,
            };
            let page_tokens = pages.tensor.layout().shape()[1];
            let template = TemplateOp::paged_copy(direction, start, page_tokens);
            let concrete = start.is_constant().then_some(Op::PagedCopy {
                direction,
                start: span.start.offset,
                page_tokens,
            });
            match direction {
                PagedCopyDirection::IntoPages => {
                    let protected = vec![
                        pages.tensor.buffer(),
                        table.tensor.buffer(),
                        generations.tensor.buffer(),
                    ];
                    self.record_paged_dispatch(
                        &commands,
                        (template, concrete),
                        vec![ordinary, table, generations],
                        pages,
                        protected,
                        usage,
                    )
                }
                PagedCopyDirection::OutOfPages => {
                    let protected = vec![
                        pages.tensor.buffer(),
                        table.tensor.buffer(),
                        generations.tensor.buffer(),
                    ];
                    self.record_paged_dispatch(
                        &commands,
                        (template, concrete),
                        vec![pages, table, generations],
                        ordinary,
                        protected,
                        usage,
                    )
                }
            }
        })();
        std::future::ready(Ok(result))
    }

    fn record_paged_sdpa(
        &mut self,
        commands: Resource<CommandListEntry>,
        slot: Resource<BatchSlotEntry>,
        lane: u32,
        query: Resource<TensorEntry>,
        output: Resource<TensorEntry>,
        config: paged_storage::SdpaCfg,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let result = (|| {
            let (key, table, generations, snapshot, usage) =
                self.paged_slot_entries(&slot, lane, config.key_plane)?;
            let (value, _, _, _, _) = self.paged_slot_entries(&slot, lane, config.value_plane)?;
            let query = self.entry(&query)?.clone();
            let output = self.entry(&output)?.clone();
            let [_query_heads, query_tokens, width]: [u32; 3] =
                query.tensor.layout().shape().try_into().map_err(|_| {
                    compute::Error::Layout("paged query must have rank three".to_owned())
                })?;
            let page_shape: [u32; 3] = key.tensor.layout().shape().try_into().map_err(|_| {
                compute::Error::Layout("paged plane must have rank three".to_owned())
            })?;
            let key_heads = page_shape[2]
                .checked_div(width)
                .filter(|&heads| heads != 0)
                .ok_or_else(|| compute::Error::Layout("paged plane width is invalid".to_owned()))?;
            let reshape_plane = |entry: TensorEntry| -> Result<TensorEntry, compute::Error> {
                let layout = entry
                    .tensor
                    .layout()
                    .reshape(&[page_shape[0], page_shape[1], key_heads, width])
                    .map_err(|error| compute::Error::Layout(error.to_string()))?;
                let tensor = self
                    .backend
                    .tensor(entry.tensor.buffer(), layout)
                    .map_err(guest_error)?;
                Ok(TensorEntry { tensor, ..entry })
            };
            let key_length = u32::try_from(snapshot.logical_len())
                .map_err(|_| compute::Error::Layout("paged key length exceeds u32".to_owned()))?;
            let q_start = key_length.checked_sub(query_tokens).ok_or_else(|| {
                compute::Error::Layout("paged query exceeds the key length".to_owned())
            })?;
            let affine_q_start = core_affine(config.q_start);
            let affine_key_length = core_affine(config.key_length);
            if config.q_start.param != config.key_length.param
                || config.q_start.scale != config.key_length.scale
                || config.q_start.offset.checked_add(query_tokens) != Some(config.key_length.offset)
                || (affine_q_start.is_constant()
                    && (config.q_start.offset != q_start || config.key_length.offset != key_length))
            {
                return Err(compute::Error::Layout(
                    "paged attention positions do not match the bound sequence".to_owned(),
                ));
            }
            let page_tokens = page_shape[1];
            let contiguous_entries = snapshot.table().is_contiguous();
            let template = TemplateOp::paged_sdpa(
                config.scale,
                config.causal,
                affine_q_start,
                affine_key_length,
                page_tokens,
                contiguous_entries,
            );
            let concrete = affine_q_start.is_constant().then_some(Op::PagedSdpa {
                scale: config.scale,
                causal: config.causal,
                q_start,
                key_length,
                page_tokens,
                contiguous_entries,
            });
            let protected = vec![
                key.tensor.buffer(),
                value.tensor.buffer(),
                table.tensor.buffer(),
                generations.tensor.buffer(),
            ];
            self.record_paged_dispatch(
                &commands,
                (template, concrete),
                vec![
                    query,
                    reshape_plane(key)?,
                    reshape_plane(value)?,
                    table,
                    generations,
                ],
                output,
                protected,
                usage,
            )
        })();
        std::future::ready(Ok(result))
    }
}

fn guest_device_info(info: DeviceInfo) -> compute::DeviceInfo {
    compute::DeviceInfo {
        family: info.family,
        recommended_max_working_set_size: info.recommended_max_working_set_size,
        max_threadgroup_memory_length: info.max_threadgroup_memory_length,
        gpu_core_count: info.gpu_core_count,
        simd_width: info.simd_width,
        streaming_read_gigabytes_per_second: info.streaming_read_gigabytes_per_second,
        streaming_copy_gigabytes_per_second: info.streaming_copy_gigabytes_per_second,
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

    async fn replay(
        accessor: &Accessor<Host<B>, Self>,
        resource: Resource<GraphEntry>,
        values: Vec<u32>,
    ) -> wasmtime::Result<Result<Option<u64>, compute::Error>> {
        let timer = accessor.with(|mut access| access.get().import_timer(ImportKind::Submit));
        let replay = accessor.with(|mut access| access.get().replay_state(&resource));
        let replay = match replay {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let flight = match ReplayFlight::acquire_wait(replay).await {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let request = accessor.with(|mut access| {
            access
                .get()
                .prepare_replay_with_flight(&resource, values, flight)
        });
        let request = match request {
            Ok(value) => value,
            Err(error) => return Ok(Err(error)),
        };
        let overlap = request.profile.is_none() && request.backend.supports_replay_overlap();
        let result = Ok(if overlap {
            request.run_deferred().await
        } else {
            request.run().await
        });
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

/// Adds the core, optional Metal-variant, and WASI imports to a component linker.
///
/// # Errors
///
/// Returns an error if an import cannot be defined on the linker.
#[cfg(target_os = "macos")]
pub fn add_metal_to_linker(
    linker: &mut Linker<Host<forja_metal::MetalBackend>>,
) -> wasmtime::Result<()> {
    add_wasi_to_linker(linker)?;
    metal_bindings::MetalHost::add_to_linker::<
        Host<forja_metal::MetalBackend>,
        HostBindings<forja_metal::MetalBackend>,
    >(linker, |host| host)
}

fn add_engine_to_linker<B>(linker: &mut Linker<Host<B>>, backend: &B) -> wasmtime::Result<()>
where
    B: Backend + Send + Sync + 'static,
{
    add_wasi_to_linker(linker)?;
    engine_bindings::EngineComponent::add_to_linker::<Host<B>, HostBindings<B>>(linker, |host| {
        host
    })?;
    #[cfg(target_os = "macos")]
    if (backend as &dyn Any).is::<forja_metal::MetalBackend>() {
        metal_bindings::l9o::gpu::metal_variants::add_to_linker::<Host<B>, HostBindings<B>>(
            linker,
            |host| host,
        )?;
    }
    Ok(())
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

fn core_affine(affine: compute::Affine) -> Affine {
    affine.param.map_or_else(
        || Affine::constant(affine.offset),
        |parameter| {
            if affine.scale == 0 {
                Affine::constant(affine.offset)
            } else {
                Affine::parameter(parameter, affine.offset, affine.scale)
            }
        },
    )
}

fn template_operation_kind(operation: TemplateOp) -> forja_core::OperationKind {
    match operation {
        TemplateOp::Static(operation) => forja_core::OperationKind::of(operation),
        TemplateOp::Sdpa { .. } => forja_core::OperationKind::Sdpa,
        TemplateOp::PagedSdpa { .. } => forja_core::OperationKind::PagedSdpa,
        TemplateOp::Sample { .. } => forja_core::OperationKind::Sample,
        TemplateOp::PagedCopy { .. } | TemplateOp::SampleMasked { .. } => {
            forja_core::OperationKind::Other
        }
    }
}

fn core_op(operation: compute::Op) -> (TemplateOp, Option<Op>) {
    match operation {
        compute::Op::Copy => concrete_template(Op::Copy),
        compute::Op::RmsNorm(eps) => concrete_template(Op::RmsNorm { eps }),
        compute::Op::Argmax(config) => concrete_template(Op::Argmax {
            chunk_width: config.chunk_width,
        }),
        compute::Op::TopK(config) => concrete_template(Op::TopK {
            k: config.k,
            normalize: config.normalize,
            chunk_width: config.chunk_width,
        }),
        compute::Op::Sample(config) => {
            let position = core_affine(config.position);
            let concrete = position.is_constant().then_some(Op::Sample {
                position: config.position.offset,
                chunk_width: config.chunk_width,
                fast_top_k: config.fast_top_k,
                fast_max_chunks: config.fast_max_chunks,
                rejection_rounds: config.rejection_rounds,
            });
            (
                TemplateOp::sample(
                    position,
                    config.chunk_width,
                    config.fast_top_k,
                    config.fast_max_chunks,
                    config.rejection_rounds,
                ),
                concrete,
            )
        }
        compute::Op::SampleMasked(config) => {
            let position = core_affine(config.position);
            let concrete = position.is_constant().then_some(Op::SampleMasked {
                position: config.position.offset,
                chunk_width: config.chunk_width,
                fast_top_k: config.fast_top_k,
                fast_max_chunks: config.fast_max_chunks,
                rejection_rounds: config.rejection_rounds,
            });
            (
                TemplateOp::sample_masked(
                    position,
                    config.chunk_width,
                    config.fast_top_k,
                    config.fast_max_chunks,
                    config.rejection_rounds,
                ),
                concrete,
            )
        }
        compute::Op::Rope(config) => concrete_template(Op::Rope {
            theta: config.theta,
        }),
        compute::Op::Embed => concrete_template(Op::Embed),
        compute::Op::QuantEmbed(config) => concrete_template(Op::QuantEmbed {
            bits: config.bits,
            group_size: config.group_size,
        }),
        compute::Op::Matmul => concrete_template(Op::Matmul),
        compute::Op::GatherMatmul => concrete_template(Op::GatherMatmul),
        compute::Op::QuantMatmul(config) => concrete_template(Op::QuantMatmul {
            bits: config.bits,
            group_size: config.group_size,
        }),
        compute::Op::GatherQuantMatmul(config) => concrete_template(Op::GatherQuantMatmul {
            bits: config.bits,
            group_size: config.group_size,
        }),
        compute::Op::Sdpa(config) => {
            let q_start = core_affine(config.q_start);
            let concrete = q_start.is_constant().then_some(Op::Sdpa {
                scale: config.scale,
                causal: config.causal,
                q_start: config.q_start.offset,
            });
            (
                TemplateOp::sdpa(config.scale, config.causal, q_start),
                concrete,
            )
        }
    }
}

fn concrete_template(op: Op) -> (TemplateOp, Option<Op>) {
    (op.into(), Some(op))
}

fn symbolic_view(
    layout: &SymbolicLayout,
    operation: &ViewOp,
) -> Result<SymbolicLayout, SymbolicLayoutError> {
    match operation {
        ViewOp::Slice(slices) => {
            let mut result = layout.clone();
            for (axis, slice) in slices.iter().enumerate() {
                let axis = u8::try_from(axis)
                    .map_err(|_| SymbolicLayoutError::AxisOutOfRange { axis: u8::MAX })?;
                result =
                    result.slice(axis, slice.start().into(), slice.len().into(), slice.step())?;
            }
            Ok(result)
        }
        ViewOp::Reshape(shape) => layout.reshape(shape.clone()),
        ViewOp::Permute(axes) => layout.permute(axes),
        ViewOp::Broadcast(shape) => layout.broadcast(shape.clone()),
    }
}

fn entry_tensors(entries: &[TensorEntry]) -> Vec<&Tensor> {
    entries.iter().map(|entry| &entry.tensor).collect()
}

fn tensor_space<'a>(
    entries: impl Iterator<Item = &'a TensorEntry>,
) -> Result<Option<ParamSpace>, compute::Error> {
    let mut space = None;
    for candidate in entries.filter_map(|entry| entry.symbolic.as_ref().map(SymbolicLayout::space))
    {
        merge_parameter_space(space.as_ref(), Some(candidate))?;
        space = Some(candidate.clone());
    }
    Ok(space)
}

fn merge_parameter_space(
    current: Option<&ParamSpace>,
    candidate: Option<&ParamSpace>,
) -> Result<(), compute::Error> {
    if let (Some(current), Some(candidate)) = (current, candidate)
        && current != candidate
    {
        return Err(compute::Error::Layout(
            "command list mixes parameter spaces".to_owned(),
        ));
    }
    Ok(())
}

fn graph_error(error: &impl ToString) -> compute::Error {
    compute::Error::OpSignature(error.to_string())
}

fn variant_error(error: &impl ToString) -> compute::Error {
    compute::Error::OpSignature(error.to_string())
}

fn validate_variant_name_length(name: &str) -> Result<(), compute::Error> {
    if name.len() > MAX_VARIANT_NAME_BYTES {
        return Err(compute::Error::OpSignature(
            "Metal variant name exceeds 64 bytes".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_variant_rule_boundary(
    arms: &[metal_bindings::l9o::gpu::metal_variants::Arm],
) -> Result<(), compute::Error> {
    if arms.len() > MAX_VARIANT_RULE_ARMS {
        return Err(quota("Metal variant rule has too many arms"));
    }
    for arm in arms {
        validate_variant_name_length(&arm.name)?;
    }
    Ok(())
}

fn validate_variant_rule_space(
    candidate: Option<ParamSpace>,
    expected: &ParamSpace,
) -> Result<(), compute::Error> {
    let candidate = candidate.ok_or_else(|| {
        compute::Error::OpSignature(
            "variant rule requires tensors from its parameter space".to_owned(),
        )
    })?;
    if &candidate != expected {
        return Err(compute::Error::OpSignature(
            "variant rule parameter space does not match its tensors".to_owned(),
        ));
    }
    Ok(())
}

fn core_program(program: compute::ProgramSource) -> Result<ValidatedProgram, compute::Error> {
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

fn core_kernel_signature(signature: compute::KernelSignature) -> KernelSignature {
    KernelSignature::new(
        signature.rank,
        signature.inputs.into_iter().map(core_dtype).collect(),
        signature.outputs.into_iter().map(core_dtype).collect(),
        signature.scalars,
    )
}

fn prepare_error(error: PrepareError) -> compute::Error {
    match error {
        PrepareError::Backend(error) => guest_error(error),
        error => compute::Error::OpSignature(error.to_string()),
    }
}

fn epilogue_error(error: &EpilogueError) -> compute::Error {
    compute::Error::OpSignature(error.to_string())
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

forja_program_conversions::program_op_conversions! {
    fn core_unop(WitUnOp => UnOp);
    fn core_binop(WitBinOp => BinOp);
    fn core_redop(WitRedOp => RedOp);
}

forja_program_conversions::program_value_type_conversion! {
    fn core_value_type(WitValueType => ValueType);
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

fn release_buffers<B: Backend>(
    backend: &B,
    buffers: Vec<Arc<BufferHandle>>,
    profile: Option<&Arc<Mutex<EngineStepProfile>>>,
) -> Result<(), BackendError> {
    let mut failure = None;
    for buffer in buffers {
        if let Err(error) = release_buffer(backend, buffer, profile)
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), Err)
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
        GuestFailure::Backend(error @ BackendError::InvalidDeviceWorkingSet { .. }) => {
            compute::Error::BackendExecution(error.to_string())
        }
        GuestFailure::Backend(
            error @ (BackendError::DeviceExecutionFailed { .. }
            | BackendError::DeviceErrorFlag { .. }),
        ) => compute::Error::BackendExecution(error.to_string()),
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
            Arc, Barrier, Condvar, Mutex,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        },
    };

    use forja_core::{
        Backend, BackendError, BufferId, CommandList, DType, Layout, LayoutError, MappedRegion,
        MemoryUsage, Op, OpError, PagePool, PagedLayout, PagedPlane, Submission, Tensor,
        ViewOp as CoreViewOp,
        program::{KernelSignature, ValidatedProgram},
    };
    use forja_cpu::CpuBackend;
    use wasmtime::component::Resource;

    use super::{
        BackendEvent, BackendTimer, BatchSlotEntry, EngineMemory, EngineMetrics, EngineRunner,
        EngineStep, EngineStepProfile, Grants, Host, ImportKind, ImportTimer, Limits,
        MAX_INSTRUCTIONS, MAX_OUTPUTS, PagePoolEntry, PagedSequenceEntry, ParamsEntry,
        ReplayFlight, ReplayState, TensorEntry,
        bindings::l9o::gpu::{compute, paged_storage},
        core_op, core_program, expected_layer_outputs,
    };

    const GENEROUS: Limits = Limits::new(u64::MAX, 8, u64::MAX, 32, u64::MAX);
    static NEXT_WEIGHT_FILE: AtomicU64 = AtomicU64::new(0);

    fn insert_test_pool<B: Backend>(
        host: &mut Host<B>,
        plane_count: u32,
        max_context: u32,
    ) -> Resource<PagePoolEntry> {
        let planes = (0..plane_count)
            .map(|plane| PagedPlane {
                dtype: DType::F32,
                elements_per_token: 2,
                alignment: 4,
                label: format!("plane-{plane}"),
            })
            .collect();
        let pool = PagePool::new(
            PagedLayout::new(4, planes).unwrap(),
            max_context.div_ceil(4),
            4_096,
        )
        .unwrap();
        let charged_bytes = pool.memory_usage().unwrap().0.total().unwrap();
        host.live_bytes.fetch_add(charged_bytes, Ordering::AcqRel);
        host.table
            .push(PagePoolEntry {
                pool,
                active_lanes: 1,
                max_context,
                plane_count,
                slot_banks: None,
                slot: None,
                charged_bytes: AtomicU64::new(charged_bytes),
                live_bytes: Arc::clone(&host.live_bytes),
            })
            .unwrap()
    }

    #[test]
    fn paged_pool_fixture_tracks_its_initial_charge() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let pool = insert_test_pool(&mut host, 1, 8);
        let charged = host
            .table
            .get(&pool)
            .unwrap()
            .charged_bytes
            .load(Ordering::Acquire);

        assert_eq!(host.live_bytes.load(Ordering::Acquire), charged);
    }

    #[test]
    fn paged_prefix_reports_committed_and_reserved_lengths() {
        let pool = PagePool::new(
            PagedLayout::new(
                4,
                vec![PagedPlane {
                    dtype: DType::F32,
                    elements_per_token: 1,
                    alignment: 4,
                    label: "test".to_owned(),
                }],
            )
            .unwrap(),
            2,
            1_024,
        )
        .unwrap();
        let mut sequence = pool.reserve(8).unwrap();
        sequence.commit_write(3).unwrap();
        let prefix = super::PagedPrefix::new(sequence);

        assert_eq!(prefix.logical_len().unwrap(), 3);
        assert_eq!(prefix.reserved_tokens().unwrap(), 8);
    }

    #[test]
    fn loaded_layer_count_limits_expected_taps() {
        let layers = [1, 2, 4, 8];
        assert_eq!(expected_layer_outputs(&layers, Some(4), true), 3);
        assert_eq!(expected_layer_outputs(&layers, None, true), 4);
        assert_eq!(expected_layer_outputs(&layers, Some(4), false), 0);
    }

    #[tokio::test]
    async fn paged_wasmtime_calls_require_an_active_lane() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let commands = host.command_list().unwrap();
        let paged = PagePool::new(
            PagedLayout::new(
                4,
                vec![PagedPlane {
                    dtype: DType::F32,
                    elements_per_token: 1,
                    alignment: 4,
                    label: "test".to_owned(),
                }],
            )
            .unwrap(),
            1,
            4_096,
        )
        .unwrap();
        let charged_bytes = paged.memory_usage().unwrap().0.total().unwrap();
        host.live_bytes.fetch_add(charged_bytes, Ordering::AcqRel);
        let pool = host
            .table
            .push(PagePoolEntry {
                pool: paged,
                active_lanes: 1,
                max_context: 4,
                plane_count: 1,
                slot_banks: None,
                slot: None,
                charged_bytes: AtomicU64::new(charged_bytes),
                live_bytes: Arc::clone(&host.live_bytes),
            })
            .unwrap();
        let sequence = host.table.push(PagedSequenceEntry).unwrap();
        let slot = host.open_paged_slot(&pool, 1).unwrap();
        let ordinary = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let copy = <Host<CpuBackend> as paged_storage::Host>::record_paged_copy(
            &mut host,
            Resource::new_borrow(commands.rep()),
            Resource::new_borrow(slot.rep()),
            0,
            paged_storage::PagedSpan {
                plane: 0,
                start: wit_affine(None, 0, 0),
                len: 1,
            },
            Resource::new_borrow(ordinary.rep()),
            paged_storage::CopyDirection::IntoPages,
        )
        .await
        .unwrap();
        assert!(matches!(copy, Err(compute::Error::Layout(_))));
        let sdpa = <Host<CpuBackend> as paged_storage::Host>::record_paged_sdpa(
            &mut host,
            Resource::new_borrow(commands.rep()),
            Resource::new_borrow(slot.rep()),
            0,
            Resource::new_borrow(ordinary.rep()),
            Resource::new_borrow(output.rep()),
            paged_storage::SdpaCfg {
                key_plane: 0,
                value_plane: 1,
                scale: 1.0,
                causal: false,
                q_start: wit_affine(None, 0, 0),
                key_length: wit_affine(None, 0, 1),
            },
        )
        .await
        .unwrap();
        assert!(matches!(sdpa, Err(compute::Error::Layout(_))));

        let pool_rep = pool.rep();
        let sequence_rep = sequence.rep();
        let slot_rep = slot.rep();
        <Host<CpuBackend> as paged_storage::HostPagePool>::drop(&mut host, pool)
            .await
            .unwrap();
        <Host<CpuBackend> as paged_storage::HostSequence>::drop(&mut host, sequence)
            .await
            .unwrap();
        <Host<CpuBackend> as paged_storage::HostBatchSlot>::drop(&mut host, slot)
            .await
            .unwrap();
        assert!(
            host.table
                .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
                .is_err()
        );
        assert!(
            host.table
                .get(&Resource::<PagedSequenceEntry>::new_borrow(sequence_rep))
                .is_err()
        );
        assert!(
            host.table
                .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                .is_err()
        );
    }

    #[tokio::test]
    async fn paged_wasmtime_calls_record_copy_and_attention_with_one_usage() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let pool = insert_test_pool(&mut host, 2, 8);
        let slot = host.open_paged_slot(&pool, 2).unwrap();
        let _checkpoint = host.begin_paged_append(pool.rep(), 0, 1).unwrap();
        let commands = host.command_list().unwrap();
        let ordinary = host.alloc(compute::Dtype::F32, &[1, 2]).unwrap();
        let query = host.alloc(compute::Dtype::F32, &[1, 1, 2]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1, 1, 2]).unwrap();

        <Host<CpuBackend> as paged_storage::Host>::record_paged_copy(
            &mut host,
            Resource::new_borrow(commands.rep()),
            Resource::new_borrow(slot.rep()),
            0,
            paged_storage::PagedSpan {
                plane: 0,
                start: wit_affine(None, 0, 0),
                len: 1,
            },
            Resource::new_borrow(ordinary.rep()),
            paged_storage::CopyDirection::IntoPages,
        )
        .await
        .unwrap()
        .unwrap();
        <Host<CpuBackend> as paged_storage::Host>::record_paged_sdpa(
            &mut host,
            Resource::new_borrow(commands.rep()),
            Resource::new_borrow(slot.rep()),
            0,
            Resource::new_borrow(query.rep()),
            Resource::new_borrow(output.rep()),
            paged_storage::SdpaCfg {
                key_plane: 0,
                value_plane: 1,
                scale: 1.0,
                causal: true,
                q_start: wit_affine(None, 0, 0),
                key_length: wit_affine(None, 0, 1),
            },
        )
        .await
        .unwrap()
        .unwrap();

        let recorded = host.table.get(&commands).unwrap();
        assert_eq!(recorded.recorded.len(), 2);
        assert_eq!(recorded.paged.len(), 1);
    }

    #[test]
    fn paged_backing_allocation_failure_releases_every_partial_resource() {
        let limits = Limits::new(130, 8, u64::MAX, 32, u64::MAX);
        let mut host = Host::new(CpuBackend::new(), limits);
        let pool = insert_test_pool(&mut host, 1, 8);
        let pool_state = host.table.get(&pool).unwrap().pool.clone();
        let before_memory = pool_state.memory_usage().unwrap();
        let before_live = host.live_bytes.load(Ordering::Acquire);

        assert!(matches!(
            host.open_paged_slot(&pool, 1),
            Err(compute::Error::Quota(_))
        ));

        let entry = host.table.get(&pool).unwrap();
        assert!(entry.slot_banks.is_none());
        assert_eq!(pool_state.memory_usage().unwrap(), before_memory);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before_live);
    }

    #[test]
    fn paged_append_rollback_restores_the_retained_sequence() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let pool = insert_test_pool(&mut host, 1, 8);
        host.open_paged_slot(&pool, 1).unwrap();
        let initial = host
            .begin_paged_append(pool.rep(), 0, 0)
            .unwrap()
            .0
            .unwrap();
        host.rollback_paged_append(pool.rep(), initial).unwrap();
        let before_memory = host.table.get(&pool).unwrap().pool.memory_usage().unwrap();
        let checkpoint = host
            .begin_paged_append(pool.rep(), 0, 3)
            .unwrap()
            .0
            .unwrap();

        host.rollback_paged_append(pool.rep(), checkpoint).unwrap();

        let slot = host.table.get(&pool).unwrap().slot.unwrap();
        let binding = &host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot))
            .unwrap()
            .binding;
        let active = binding.sequence.lock().unwrap();
        let sequence = active.as_ref().unwrap().lock().unwrap();
        assert_eq!(sequence.logical_len(), 0);
        assert_eq!(sequence.reserved_tokens(), 8);
        assert_eq!(sequence.page_reference_counts().unwrap(), [1, 1]);
        assert_eq!(
            host.table.get(&pool).unwrap().pool.memory_usage().unwrap(),
            before_memory
        );
    }

    #[test]
    fn paged_submission_completion_releases_retired_pages_but_timeout_does_not() {
        let pool = PagePool::new(
            PagedLayout::new(
                4,
                vec![PagedPlane {
                    dtype: DType::F32,
                    elements_per_token: 1,
                    alignment: 4,
                    label: "test".to_owned(),
                }],
            )
            .unwrap(),
            1,
            1_024,
        )
        .unwrap();
        let mut sequence = pool.reserve(4).unwrap();
        sequence.commit_write(4).unwrap();
        let old_page = sequence.table().unwrap().resolve(0).unwrap();
        let sequence = Arc::new(Mutex::new(sequence));
        let banks = Arc::new(forja_core::PagedSlotBanks::new(pool.clone(), 1, 1).unwrap());
        banks.bind(0, 0, &sequence.lock().unwrap()).unwrap();
        let binding = Arc::new(super::PagedBinding {
            pool: pool.clone(),
            banks,
            sequence: Mutex::new(Some(Arc::clone(&sequence))),
            used: std::sync::atomic::AtomicBool::new(false),
            table_dirty: std::sync::atomic::AtomicBool::new(false),
        });
        let usage = super::PagedUse { binding, bank: 0 };
        super::PagedUse::mark_all(std::slice::from_ref(&usage), Some(7)).unwrap();
        *usage.binding.sequence.lock().unwrap() = None;
        drop(sequence);

        super::PagedUse::observe_all(std::slice::from_ref(&usage), Some(7), false).unwrap();
        let replacement = pool.reserve(4).unwrap();
        assert_ne!(replacement.table().unwrap().resolve(0).unwrap(), old_page);
        drop(replacement);

        super::PagedUse::observe_all(std::slice::from_ref(&usage), Some(7), true).unwrap();
        let reused = pool.reserve(4).unwrap();
        assert_eq!(reused.table().unwrap().resolve(0).unwrap(), old_page);
    }

    #[test]
    fn paged_request_binding_accepts_a_reserved_prefix_fork() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let pool = insert_test_pool(&mut host, 1, 16);
        host.open_paged_slot(&pool, 1).unwrap();
        host.bind_paged_request(pool.rep(), None, 8).unwrap();
        let active = host.paged_prefix(pool.rep()).unwrap();
        active.sequence.lock().unwrap().commit_write(7).unwrap();
        let fork = active.fork_reserved(7, 12).unwrap();

        host.bind_paged_request(pool.rep(), Some(fork), 12).unwrap();

        let rebound = host.paged_prefix(pool.rep()).unwrap();
        let sequence = rebound.sequence.lock().unwrap();
        assert_eq!(sequence.logical_len(), 7);
        assert_eq!(sequence.reserved_tokens(), 12);
    }

    #[test]
    fn paged_request_binding_refuses_a_foreign_prefix_without_rebinding() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let local = insert_test_pool(&mut host, 1, 8);
        let foreign = insert_test_pool(&mut host, 1, 8);
        host.open_paged_slot(&local, 1).unwrap();
        host.bind_paged_request(local.rep(), None, 8).unwrap();
        let before = host.paged_prefix(local.rep()).unwrap();
        let before_logical = before.sequence.lock().unwrap().logical_len();
        let mut foreign_sequence = host.table.get(&foreign).unwrap().pool.reserve(8).unwrap();
        foreign_sequence.commit_write(3).unwrap();

        assert!(matches!(
            host.bind_paged_request(
                local.rep(),
                Some(super::PagedPrefix::new(foreign_sequence)),
                8,
            ),
            Err(compute::Error::Quota(_))
        ));

        let after = host.paged_prefix(local.rep()).unwrap();
        let sequence = after.sequence.lock().unwrap();
        assert_eq!(sequence.logical_len(), before_logical);
        assert_eq!(sequence.reserved_tokens(), 8);
    }

    #[tokio::test]
    async fn failed_cow_submission_restores_host_paged_state() {
        let mut backend = AccountingBackend::new(None, None, Duration::ZERO);
        backend.fail_submission = Some(1);
        let weights = test_weight_file();
        let mut runner =
            EngineRunner::new(test_guests::engine_smoke(), backend, GENEROUS, &weights)
                .await
                .unwrap();
        runner
            .load_with_selections(
                None,
                &super::EngineLoadConfig {
                    paged_cache: true,
                    ..super::EngineLoadConfig::default()
                },
            )
            .await
            .unwrap()
            .unwrap();
        runner.prepare_paged_request(None, 8).unwrap();
        let checkpoint = runner.prepare_paged_append(0, 3).await.unwrap();
        let pool_rep = runner.page_pool.unwrap();
        let binding = {
            let host = runner.store.data();
            let slot_rep = host
                .table
                .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
                .unwrap()
                .slot
                .unwrap();
            Arc::clone(
                &host
                    .table
                    .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
                    .unwrap()
                    .binding,
            )
        };
        binding.used.store(true, Ordering::Release);
        runner.finish_paged_append(0, checkpoint).unwrap();
        let retained = runner.paged_prefix().unwrap().unwrap();
        let fork = retained.fork_reserved(3, 8).unwrap();
        runner.prepare_paged_request(Some(fork), 8).unwrap();
        let before = runner.paged_prefix().unwrap().unwrap();
        let (before_table, before_refs) = {
            let sequence = before.sequence.lock().unwrap();
            (
                sequence.table().unwrap(),
                sequence.page_reference_counts().unwrap(),
            )
        };
        let (before_memory, before_charged, before_live) = {
            let host = runner.store.data();
            let entry = host
                .table
                .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
                .unwrap();
            (
                entry.pool.memory_usage().unwrap(),
                entry.charged_bytes.load(Ordering::Acquire),
                host.live_bytes.load(Ordering::Acquire),
            )
        };

        assert!(matches!(
            runner.prepare_paged_append(3, 1).await,
            Err(compute::Error::BackendExecution(_))
        ));

        let after = runner.paged_prefix().unwrap().unwrap();
        let sequence = after.sequence.lock().unwrap();
        assert_eq!(sequence.logical_len(), 3);
        assert_eq!(sequence.reserved_tokens(), 8);
        assert_eq!(sequence.table().unwrap().entries(), before_table.entries());
        assert_eq!(sequence.page_reference_counts().unwrap(), before_refs);
        drop(sequence);
        let host = runner.store.data();
        let entry = host
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool_rep))
            .unwrap();
        assert_eq!(entry.pool.memory_usage().unwrap(), before_memory);
        assert_eq!(entry.charged_bytes.load(Ordering::Acquire), before_charged);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before_live);
        fs::remove_file(weights).unwrap();
    }

    fn doubling_program(value: f32) -> compute::ProgramSource {
        compute::ProgramSource {
            kind: compute::ProgramKind::Map,
            insts: vec![
                compute::Inst::Input(0),
                compute::Inst::Const(value),
                compute::Inst::Binary((compute::Binop::Mul, 0, 1)),
            ],
            outputs: vec![(0, 2)],
        }
    }

    fn unary_f32_signature(rank: u8) -> compute::KernelSignature {
        compute::KernelSignature {
            rank,
            inputs: vec![compute::Dtype::F32],
            outputs: vec![compute::Dtype::F32],
            scalars: 0,
        }
    }

    fn unary_f32_epilogue(rank: u8) -> compute::EpilogueSignature {
        compute::EpilogueSignature {
            accumulator_count: 1,
            logical_rank: rank,
            extras: vec![],
            output: compute::Dtype::F32,
        }
    }

    fn wit_affine(param: Option<u8>, scale: u32, offset: u32) -> compute::Affine {
        compute::Affine {
            param,
            scale,
            offset,
        }
    }

    #[cfg(target_os = "macos")]
    fn metal_matmul_tensors(
        host: &mut Host<forja_metal::MetalBackend>,
        rows: u32,
    ) -> ([Resource<TensorEntry>; 2], Resource<TensorEntry>) {
        (
            [
                host.alloc(compute::Dtype::F32, &[rows, 33]).unwrap(),
                host.alloc(compute::Dtype::F32, &[33, 7]).unwrap(),
            ],
            host.alloc(compute::Dtype::F32, &[rows, 7]).unwrap(),
        )
    }

    #[cfg(target_os = "macos")]
    fn metal_symbolic_matmul(
        host: &mut Host<forja_metal::MetalBackend>,
    ) -> (
        Resource<ParamsEntry>,
        [Resource<TensorEntry>; 2],
        Resource<TensorEntry>,
    ) {
        let params = host
            .params(vec![compute::ParamRange { lo: 2, hi: 3 }])
            .unwrap();
        let left = host.alloc(compute::Dtype::F32, &[3, 33]).unwrap();
        let right = host.alloc(compute::Dtype::F32, &[33, 7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[3, 7]).unwrap();
        let slices = |width| {
            vec![
                compute::ParamSlice {
                    start: wit_affine(None, 0, 0),
                    len: wit_affine(Some(0), 1, 0),
                    step: 1,
                },
                compute::ParamSlice {
                    start: wit_affine(None, 0, 0),
                    len: wit_affine(None, 0, width),
                    step: 1,
                },
            ]
        };
        let left = host
            .view_param(
                &Resource::new_borrow(left.rep()),
                &Resource::new_borrow(params.rep()),
                slices(33),
            )
            .unwrap();
        let output = host
            .view_param(
                &Resource::new_borrow(output.rep()),
                &Resource::new_borrow(params.rep()),
                slices(7),
            )
            .unwrap();
        (params, [left, right], output)
    }

    #[cfg(target_os = "macos")]
    fn metal_rule_arm() -> super::metal_bindings::l9o::gpu::metal_variants::Arm {
        super::metal_bindings::l9o::gpu::metal_variants::Arm {
            lo: 2,
            hi: 3,
            name: "matmul.steel-64x64x16-2x2".to_owned(),
        }
    }

    #[cfg(target_os = "macos")]
    fn borrowed_tensors(tensors: &[Resource<TensorEntry>]) -> Vec<Resource<TensorEntry>> {
        tensors
            .iter()
            .map(|tensor| Resource::new_borrow(tensor.rep()))
            .collect()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_variant_entry_points_bound_guest_collections_first() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let invalid_commands = Resource::new_borrow(41);
        assert!(matches!(
            host.record_metal_variant(
                &invalid_commands,
                compute::Op::Matmul,
                &[],
                &[],
                &"a".repeat(65),
            ),
            Err(compute::Error::OpSignature(_))
        ));
        let oversized = (0..65).map(|_| metal_rule_arm()).collect::<Vec<_>>();
        assert!(matches!(
            host.record_metal_variant_rule(
                &invalid_commands,
                &Resource::new_borrow(42),
                0,
                &oversized,
                compute::Op::Matmul,
                &[],
                &[],
            ),
            Err(compute::Error::Quota(_))
        ));
        let oversized_name = [super::metal_bindings::l9o::gpu::metal_variants::Arm {
            lo: 2,
            hi: 3,
            name: "a".repeat(65),
        }];
        assert!(matches!(
            host.record_metal_variant_rule(
                &invalid_commands,
                &Resource::new_borrow(42),
                0,
                &oversized_name,
                compute::Op::Matmul,
                &[],
                &[],
            ),
            Err(compute::Error::OpSignature(_))
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_variant_rules_require_the_tensor_parameter_space() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let (params, inputs, output) = metal_symbolic_matmul(&mut host);
        let other = host
            .params(vec![compute::ParamRange { lo: 2, hi: 3 }])
            .unwrap();
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.record_metal_variant_rule(
                &commands,
                &other,
                0,
                &[metal_rule_arm()],
                compute::Op::Matmul,
                &borrowed_tensors(&inputs),
                &[Resource::new_borrow(output.rep())],
            ),
            Err(compute::Error::OpSignature(_))
        ));

        let (inputs, output) = metal_matmul_tensors(&mut host, 7);
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.record_metal_variant_rule(
                &commands,
                &params,
                0,
                &[metal_rule_arm()],
                compute::Op::Matmul,
                &borrowed_tensors(&inputs),
                &[Resource::new_borrow(output.rep())],
            ),
            Err(compute::Error::OpSignature(_))
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_affine_variant_uses_the_command_parameter_space() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let (_params, inputs, output) = metal_symbolic_matmul(&mut host);
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &Resource::new_borrow(output.rep()),
        )
        .unwrap();

        let query = host.alloc(compute::Dtype::F32, &[16, 1, 128]).unwrap();
        let key = host.alloc(compute::Dtype::F32, &[8, 4, 128]).unwrap();
        let value = host.alloc(compute::Dtype::F32, &[8, 4, 128]).unwrap();
        let attended = host.alloc(compute::Dtype::F32, &[16, 1, 128]).unwrap();
        host.record_metal_variant(
            &commands,
            compute::Op::Sdpa(compute::SdpaCfg {
                scale: 0.088,
                causal: true,
                q_start: wit_affine(Some(0), 1, 0),
            }),
            &[
                Resource::new_borrow(query.rep()),
                Resource::new_borrow(key.rep()),
                Resource::new_borrow(value.rep()),
            ],
            &[Resource::new_borrow(attended.rep())],
            "sdpa.decomposed",
        )
        .unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_epilogue_variants_preserve_the_epilogue_binding() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let (inputs, output) = metal_matmul_tensors(&mut host, 7);
        let epilogue = host
            .create_epilogue(doubling_program(2.0), unary_f32_epilogue(2))
            .unwrap();
        let prepared = Arc::clone(host.table.get(&epilogue).unwrap().prepared());
        let commands = host.command_list().unwrap();
        host.dispatch_epilogue(
            &commands,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &Resource::new_borrow(output.rep()),
            &epilogue,
            &[],
        )
        .unwrap();
        host.select_metal_epilogue_variant(&commands, "matmul.steel-64x64x16-2x2")
            .unwrap();

        let super::RecordedDispatch::Static(dispatch) =
            host.table.get(&commands).unwrap().recorded.last().unwrap()
        else {
            panic!("expected a static epilogue dispatch");
        };
        assert!(Arc::ptr_eq(
            dispatch.prepared_epilogue().unwrap(),
            &prepared
        ));
        assert_eq!(
            dispatch
                .backend_data::<forja_metal::ValidatedVariant>()
                .unwrap()
                .name(),
            "matmul.steel-64x64x16-2x2"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_epilogue_variant_rules_preserve_every_arm() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let (params, inputs, output) = metal_symbolic_matmul(&mut host);
        let epilogue = host
            .create_epilogue(doubling_program(2.0), unary_f32_epilogue(2))
            .unwrap();
        let prepared = Arc::clone(host.table.get(&epilogue).unwrap().prepared());
        let commands = host.command_list().unwrap();
        host.dispatch_epilogue(
            &commands,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &Resource::new_borrow(output.rep()),
            &epilogue,
            &[],
        )
        .unwrap();
        host.select_metal_epilogue_variant_rule(&commands, &params, 0, &[metal_rule_arm()])
            .unwrap();

        let super::RecordedDispatch::EpilogueVariantRule {
            epilogue,
            extras,
            parameter,
            arms,
            ..
        } = host.table.get(&commands).unwrap().recorded.last().unwrap()
        else {
            panic!("expected an epilogue variant rule");
        };
        assert!(Arc::ptr_eq(epilogue, &prepared));
        assert!(extras.is_empty());
        assert_eq!(*parameter, 0);
        assert_eq!(arms.len(), 1);
        assert_eq!(
            arms[0]
                .1
                .get::<forja_metal::ValidatedVariant>()
                .unwrap()
                .name(),
            "matmul.steel-64x64x16-2x2"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_epilogue_variant_selection_refuses_invalid_bindings() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let (inputs, output) = metal_matmul_tensors(&mut host, 7);
        let commands = host.command_list().unwrap();
        host.dispatch_many(
            &commands,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &[Resource::new_borrow(output.rep())],
        )
        .unwrap();
        assert!(matches!(
            host.select_metal_epilogue_variant(&commands, "matmul.steel-64x64x16-2x2"),
            Err(compute::Error::InvalidHandle(_))
        ));

        let epilogue = host
            .create_epilogue(doubling_program(2.0), unary_f32_epilogue(2))
            .unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch_epilogue(
            &commands,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &Resource::new_borrow(output.rep()),
            &epilogue,
            &[],
        )
        .unwrap();
        assert!(matches!(
            host.select_metal_epilogue_variant(&commands, "quant-matmul.tiled"),
            Err(compute::Error::OpSignature(_))
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_variant_entry_points_enforce_dispatch_and_work_quotas() {
        for limits in [
            GENEROUS.with_command_limits(0, u64::MAX),
            GENEROUS.with_command_limits(usize::MAX, 1),
        ] {
            let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), limits);
            let (inputs, output) = metal_matmul_tensors(&mut host, 7);
            let commands = host.command_list().unwrap();
            assert!(matches!(
                host.record_metal_variant(
                    &commands,
                    compute::Op::Matmul,
                    &borrowed_tensors(&inputs),
                    &[Resource::new_borrow(output.rep())],
                    "matmul.steel-64x64x16-2x2",
                ),
                Err(compute::Error::Quota(_))
            ));
        }

        for limits in [
            GENEROUS.with_command_limits(0, u64::MAX),
            GENEROUS.with_command_limits(usize::MAX, 1),
        ] {
            let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), limits);
            let (params, inputs, output) = metal_symbolic_matmul(&mut host);
            let commands = host.command_list().unwrap();
            assert!(matches!(
                host.record_metal_variant_rule(
                    &commands,
                    &params,
                    0,
                    &[metal_rule_arm()],
                    compute::Op::Matmul,
                    &borrowed_tensors(&inputs),
                    &[Resource::new_borrow(output.rep())],
                ),
                Err(compute::Error::Quota(_))
            ));
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn empty_graph<B>(host: &mut Host<B>) -> Resource<super::GraphEntry>
    where
        B: Backend + 'static,
    {
        let commands = host.command_list().unwrap();
        host.create_graph(commands).unwrap()
    }

    #[cfg(target_os = "macos")]
    fn captured_copy_chain(
        host: &mut Host<forja_metal::MetalBackend>,
        input: &Resource<TensorEntry>,
        bridge: &Resource<TensorEntry>,
        output: &Resource<TensorEntry>,
    ) -> Resource<super::GraphEntry> {
        let first = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let second = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let commands = host.command_list().unwrap();
        for (source, target) in [
            (input, &first),
            (&first, bridge),
            (bridge, &second),
            (&second, output),
        ] {
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(source.rep())],
                &Resource::new_borrow(target.rep()),
            )
            .unwrap();
        }
        host.drop_tensor(first).unwrap();
        host.drop_tensor(second).unwrap();
        host.create_graph(commands).unwrap()
    }

    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_captured_graphs_share_two_transient_arenas_and_match_lazy_execution() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let bridge = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        host.write(&input, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();

        let mut graphs = Vec::new();
        for _ in 0..8 {
            graphs.push(captured_copy_chain(&mut host, &input, &bridge, &output));
            assert_eq!(host.live_bytes.load(Ordering::Acquire), 80);
        }
        host.prepare_replay(&graphs[0], Vec::new())
            .unwrap()
            .run()
            .await
            .unwrap();
        let planned = host.prepare_read(&output).unwrap().run().await.unwrap();

        let lazy_output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let lazy_first = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let lazy_second = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let commands = host.command_list().unwrap();
        for (source, target) in [
            (&input, &lazy_first),
            (&lazy_first, &bridge),
            (&bridge, &lazy_second),
            (&lazy_second, &lazy_output),
        ] {
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(source.rep())],
                &Resource::new_borrow(target.rep()),
            )
            .unwrap();
        }
        host.prepare_submit(commands).unwrap().run().await.unwrap();
        let unplanned = host
            .prepare_read(&lazy_output)
            .unwrap()
            .run()
            .await
            .unwrap();
        assert_eq!(planned, unplanned);
    }

    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_dropped_graph_output_remains_stable_across_replays() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let transient = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let sink = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let expected = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        host.write(&input, &expected).unwrap();
        let output_tensor = host.entry(&output).unwrap().tensor.clone();
        let commands = host.command_list().unwrap();
        for (source, target) in [
            (&input, &transient),
            (&transient, &output),
            (&output, &sink),
        ] {
            host.dispatch(
                &commands,
                compute::Op::Copy,
                &[Resource::new_borrow(source.rep())],
                &Resource::new_borrow(target.rep()),
            )
            .unwrap();
        }
        host.graph_output(&commands, &output).unwrap();
        host.drop_tensor(transient).unwrap();
        host.drop_tensor(output).unwrap();
        let graph = host.create_graph(commands).unwrap();

        for _ in 0..16 {
            host.prepare_replay(&graph, Vec::new())
                .unwrap()
                .run()
                .await
                .unwrap();
            assert_eq!(host.backend.read(&output_tensor).unwrap(), expected);
        }
    }

    #[cfg(target_os = "macos")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_interleaved_planned_graphs_wait_for_shared_arena_slots() {
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), GENEROUS);
        let first_input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let first_bridge = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let first_output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let second_input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let second_bridge = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let second_output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        host.write(&first_input, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        host.write(&second_input, &f32_bytes(&[5.0, 6.0, 7.0, 8.0]))
            .unwrap();
        let first = captured_copy_chain(&mut host, &first_input, &first_bridge, &first_output);
        let second = captured_copy_chain(&mut host, &second_input, &second_bridge, &second_output);

        host.prepare_replay(&first, Vec::new())
            .unwrap()
            .run_deferred()
            .await
            .unwrap();
        host.prepare_replay(&second, Vec::new())
            .unwrap()
            .run_deferred()
            .await
            .unwrap();
        for graph in [&first, &second] {
            let flight = ReplayFlight::acquire_wait(Arc::clone(&host.replay))
                .await
                .unwrap();
            host.prepare_replay_with_flight(graph, Vec::new(), flight)
                .unwrap()
                .run_deferred()
                .await
                .unwrap();
        }
        let first_bytes = host
            .prepare_replay_read(&first_output)
            .unwrap()
            .run()
            .await
            .unwrap();
        let second_bytes = host
            .prepare_replay_read(&second_output)
            .unwrap()
            .run()
            .await
            .unwrap();

        let lazy_outputs = [
            host.alloc(compute::Dtype::F32, &[4]).unwrap(),
            host.alloc(compute::Dtype::F32, &[4]).unwrap(),
        ];
        let lazy_commands = host.command_list().unwrap();
        for (input, output) in [
            (&first_input, &lazy_outputs[0]),
            (&second_input, &lazy_outputs[1]),
        ] {
            host.dispatch(
                &lazy_commands,
                compute::Op::Copy,
                &[Resource::new_borrow(input.rep())],
                &Resource::new_borrow(output.rep()),
            )
            .unwrap();
        }
        host.prepare_submit(lazy_commands)
            .unwrap()
            .run()
            .await
            .unwrap();
        let lazy_first = host
            .prepare_read(&lazy_outputs[0])
            .unwrap()
            .run()
            .await
            .unwrap();
        let lazy_second = host
            .prepare_read(&lazy_outputs[1])
            .unwrap()
            .run()
            .await
            .unwrap();
        assert_eq!(first_bytes, lazy_first);
        assert_eq!(second_bytes, lazy_second);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_graph_creation_refuses_two_arenas_over_the_quota() {
        let limits = Limits::new(80, 8, u64::MAX, 32, u64::MAX);
        let mut host = Host::new(forja_metal::MetalBackend::new().unwrap(), limits);
        host.backend.set_memory_limit(u64::MAX);
        let input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let first = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let second = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let final_output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &Resource::new_borrow(first.rep()),
        )
        .unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(first.rep())],
            &Resource::new_borrow(second.rep()),
        )
        .unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(first.rep())],
            &Resource::new_borrow(output.rep()),
        )
        .unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(second.rep())],
            &Resource::new_borrow(final_output.rep()),
        )
        .unwrap();
        host.drop_tensor(first).unwrap();
        host.drop_tensor(second).unwrap();

        assert!(matches!(
            host.create_graph(commands),
            Err(compute::Error::Quota(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn symbolic_graph_retains_tensors_and_replays_checked_values() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[4]).unwrap();
        host.write(&input, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let output_tensor = host.entry(&output).unwrap().tensor.clone();
        let params = host
            .params(vec![compute::ParamRange { lo: 0, hi: 3 }])
            .unwrap();
        let slice = compute::ParamSlice {
            start: wit_affine(None, 0, 0),
            len: wit_affine(Some(0), 1, 1),
            step: 1,
        };
        let symbolic_input = host
            .view_param(
                &Resource::new_borrow(input.rep()),
                &Resource::new_borrow(params.rep()),
                vec![slice],
            )
            .unwrap();
        let symbolic_output = host
            .view_param(
                &Resource::new_borrow(output.rep()),
                &Resource::new_borrow(params.rep()),
                vec![slice],
            )
            .unwrap();
        assert!(matches!(
            host.write(&symbolic_input, &[0; 4]),
            Err(compute::Error::Layout(_))
        ));
        assert!(matches!(
            host.prepare_read(&symbolic_input),
            Err(compute::Error::Layout(_))
        ));

        let rejected = host.command_list().unwrap();
        host.dispatch(
            &rejected,
            compute::Op::Copy,
            &[Resource::new_borrow(symbolic_input.rep())],
            &Resource::new_borrow(symbolic_output.rep()),
        )
        .unwrap();
        assert!(matches!(
            host.prepare_submit(rejected),
            Err(compute::Error::OpSignature(_))
        ));

        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(symbolic_input.rep())],
            &Resource::new_borrow(symbolic_output.rep()),
        )
        .unwrap();
        let graph = host.create_graph(commands).unwrap();
        host.drop_params(params).unwrap();
        host.drop_tensor(input).unwrap();
        host.drop_tensor(output).unwrap();
        host.drop_tensor(symbolic_input).unwrap();
        host.drop_tensor(symbolic_output).unwrap();

        assert!(matches!(
            host.prepare_replay(&graph, Vec::new()),
            Err(compute::Error::OpSignature(_))
        ));
        assert!(matches!(
            host.prepare_replay(&graph, vec![4]),
            Err(compute::Error::OpSignature(_))
        ));
        host.prepare_replay(&graph, vec![2])
            .unwrap()
            .run()
            .await
            .unwrap();
        assert_eq!(
            host.backend.read(&output_tensor).unwrap(),
            f32_bytes(&[1.0, 2.0, 3.0, 0.0])
        );

        host.drop_graph(graph).unwrap();
        assert_eq!(
            host.backend.read(&output_tensor),
            Err(BackendError::InvalidInput)
        );
    }

    #[test]
    fn graph_quota_counts_live_graphs() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_graph_limit(1));
        let graph = empty_graph(&mut host);
        assert_eq!(host.engine_metrics().live_graphs, 1);
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.create_graph(commands),
            Err(compute::Error::Quota(_))
        ));
        host.drop_graph(graph).unwrap();
        assert_eq!(host.engine_metrics().live_graphs, 0);
        empty_graph(&mut host);
    }

    #[test]
    fn created_graph_uses_command_validation_limits() {
        let limits = Limits::new(u64::MAX, 8, 17, 32, u64::MAX).with_command_limits(3, 29);
        let mut host = Host::new(CpuBackend::new(), limits);
        let commands = host.command_list().unwrap();
        let graph = host.create_graph(commands).unwrap();
        let entry = host.table.get(&graph).unwrap();

        assert_eq!(
            entry.graph.limits(),
            forja_core::GraphLimits::new(
                limits.dispatches_per_list,
                limits.tensor_elements,
                limits.work_per_dispatch,
            )
        );
    }

    #[test]
    fn graph_retains_prepared_kernels() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_kernel_limit(1));
        let input = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let kernel = host
            .create_kernel(doubling_program(2.0), unary_f32_signature(1))
            .unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch_kernel(&commands, &kernel, &[input], &[output])
            .unwrap();
        let graph = host.create_graph(commands).unwrap();
        host.drop_kernel(kernel).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);
        assert!(matches!(
            host.create_kernel(doubling_program(3.0), unary_f32_signature(1)),
            Err(compute::Error::Quota(_))
        ));
        host.drop_graph(graph).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replay_waits_for_depth_to_become_available() {
        let replay = Arc::new(ReplayState::new(2));
        let first = ReplayFlight::acquire(Arc::clone(&replay)).unwrap();
        let _second = ReplayFlight::acquire(Arc::clone(&replay)).unwrap();
        let third = tokio::spawn(ReplayFlight::acquire_wait(Arc::clone(&replay)));
        tokio::task::yield_now().await;
        assert!(!third.is_finished());
        drop(first);
        let _third = third.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replays_share_backend_depth_across_graphs() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO)
            .with_graph_memory_depth(2);
        let mut host = Host::new(backend, GENEROUS);
        let first_graph = empty_graph(&mut host);
        let second_graph = empty_graph(&mut host);
        let first = host.prepare_replay(&first_graph, Vec::new()).unwrap();
        let second = host.prepare_replay(&second_graph, Vec::new()).unwrap();
        let first_task = tokio::spawn(first.run());
        let second_task = tokio::spawn(second.run());
        gate.wait_for(2);
        let replay = Arc::clone(&host.replay);
        let third = tokio::spawn(ReplayFlight::acquire_wait(replay));
        tokio::task::yield_now().await;
        assert!(!third.is_finished());
        gate.release();
        first_task.await.unwrap().unwrap();
        second_task.await.unwrap().unwrap();
        drop(third.await.unwrap().unwrap());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn completion_releases_replay_depth_before_waking_waiters() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let mut runner = super::EngineRunner::new(
            test_guests::engine_smoke(),
            backend,
            GENEROUS,
            "unused.safetensors",
        )
        .await
        .unwrap();
        let graph = empty_graph(runner.store.data_mut());
        let replay = Arc::clone(&runner.store.data().replay);
        let barrier = Arc::new(Barrier::new(2));
        *replay.drop_barrier.lock().unwrap() = Some(Arc::clone(&barrier));

        let first = runner
            .store
            .data()
            .prepare_replay(&graph, Vec::new())
            .unwrap();
        let second = runner
            .store
            .data()
            .prepare_replay(&graph, Vec::new())
            .unwrap();
        let first_task = tokio::spawn(first.run());
        gate.wait_for(1);
        let completed = Arc::clone(&runner.store.data().completed_submissions);

        gate.release();
        barrier.wait();
        let published_before_release = completed.load(Ordering::Acquire);
        barrier.wait();

        assert_eq!(published_before_release, 0);
        runner.wait_for_submissions(1).await.unwrap();
        let third = runner
            .store
            .data()
            .prepare_replay(&graph, Vec::new())
            .unwrap();
        first_task.await.unwrap().unwrap();
        drop(third);
        drop(second);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn pipelined_replays_reserve_gpu_budget_individually() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(
            Some(Arc::clone(&gate)),
            Some(Duration::from_nanos(10)),
            Duration::ZERO,
        );
        let limits = GENEROUS.with_gpu_limits(Duration::from_nanos(100), Duration::from_nanos(150));
        let mut host = Host::new(backend, limits);
        let graph = empty_graph(&mut host);
        let first = host.prepare_replay(&graph, Vec::new()).unwrap();
        let task = tokio::spawn(first.run());
        gate.wait_for(1);

        assert!(matches!(
            host.prepare_replay(&graph, Vec::new()).unwrap().run().await,
            Err(compute::Error::Quota(_))
        ));
        gate.release();
        task.await.unwrap().unwrap();
        host.prepare_replay(&graph, Vec::new())
            .unwrap()
            .run()
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_a_graph_with_replays_in_flight_is_safe() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let mut host = Host::new(backend, GENEROUS);
        let graph = empty_graph(&mut host);
        let request = host.prepare_replay(&graph, Vec::new()).unwrap();
        let task = tokio::spawn(request.run());
        gate.wait_for(1);

        host.drop_graph(graph).unwrap();
        assert_eq!(host.live_graphs.load(Ordering::Acquire), 0);
        gate.release();
        task.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deferred_replay_returns_after_queue_submission() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let mut host = Host::new(backend, GENEROUS);
        let graph = empty_graph(&mut host);
        let replay = Arc::clone(&host.replay);

        assert_eq!(
            host.prepare_replay(&graph, Vec::new())
                .unwrap()
                .run_deferred()
                .await
                .unwrap(),
            None
        );
        gate.wait_for(1);
        assert_eq!(replay.status.lock().unwrap().in_flight, 1);
        assert_eq!(host.queued_submissions.load(Ordering::Acquire), 1);
        assert_eq!(host.engine_metrics().submissions, 0);
        gate.release();
        tokio::time::timeout(Duration::from_secs(1), async {
            while replay.status.lock().unwrap().in_flight != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(host.engine_metrics().submissions, 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn chunked_prefill_queues_without_intervening_host_access() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let weights = test_weight_file();
        let mut runner =
            EngineRunner::new(test_guests::engine_sdk_smoke(), backend, GENEROUS, &weights)
                .await
                .unwrap();
        runner.load().await.unwrap().unwrap();
        let task = tokio::spawn(async move {
            let result = runner
                .step(EngineStep {
                    tokens: (0..1_024).map(|value| value % 4).collect(),
                    start_pos: 0,
                    taps: false,
                })
                .await;
            (runner, result)
        });
        let queued = tokio::time::timeout(Duration::from_secs(1), async {
            while gate.count() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await;
        gate.release();
        let (mut runner, output) = task.await.unwrap();
        let output = output.unwrap().unwrap();
        runner.wait_for_queued_submissions().await.unwrap();
        assert!(queued.is_ok(), "second prefill chunk waited on the host");
        assert_eq!(runner.read(&output.logits).await.unwrap().len(), 16);
        drop(runner);
        fs::remove_file(weights).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn failed_writer_taints_same_and_shared_graph_replays() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::failing_nth(Arc::clone(&gate), 1);
        let mut host = Host::new(backend, GENEROUS);
        let source = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let shared = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let temporary = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let sink = host.alloc(compute::Dtype::F32, &[1]).unwrap();

        let first_commands = host.command_list().unwrap();
        host.dispatch(
            &first_commands,
            compute::Op::Copy,
            &[Resource::new_borrow(shared.rep())],
            &temporary,
        )
        .unwrap();
        host.dispatch(
            &first_commands,
            compute::Op::Copy,
            &[Resource::new_borrow(source.rep())],
            &shared,
        )
        .unwrap();
        let first_graph = host.create_graph(first_commands).unwrap();

        let second_commands = host.command_list().unwrap();
        host.dispatch(
            &second_commands,
            compute::Op::Copy,
            &[Resource::new_borrow(shared.rep())],
            &sink,
        )
        .unwrap();
        let second_graph = host.create_graph(second_commands).unwrap();

        let failed = host.prepare_replay(&first_graph, Vec::new()).unwrap();
        let failed_task = tokio::spawn(failed.run());
        gate.wait_for(1);
        let same_task = tokio::spawn(host.prepare_replay(&first_graph, Vec::new()).unwrap().run());
        let shared_task = tokio::spawn(
            host.prepare_replay(&second_graph, Vec::new())
                .unwrap()
                .run(),
        );
        gate.wait_for(3);
        gate.release();

        assert!(failed_task.await.unwrap().is_err());
        assert!(same_task.await.unwrap().is_err());
        assert!(shared_task.await.unwrap().is_err());
        assert!(host.prepare_read(&shared).unwrap().run().await.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn replay_uses_gpu_budget_and_observes_intervening_writes() {
        let backend = AccountingBackend::new(None, Some(Duration::from_nanos(60)), Duration::ZERO);
        let limits = GENEROUS.with_gpu_limits(Duration::from_nanos(100), Duration::from_nanos(200));
        let mut host = Host::new(backend, limits);
        let input = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &output,
        )
        .unwrap();
        let graph = host.create_graph(commands).unwrap();

        host.write(&input, &1.0_f32.to_le_bytes()).unwrap();
        host.prepare_replay(&graph, Vec::new())
            .unwrap()
            .run()
            .await
            .unwrap();
        assert_eq!(
            host.backend
                .read(&host.entry(&output).unwrap().tensor)
                .unwrap(),
            1.0_f32.to_le_bytes()
        );
        host.write(&input, &2.0_f32.to_le_bytes()).unwrap();
        host.prepare_replay(&graph, Vec::new())
            .unwrap()
            .run()
            .await
            .unwrap();
        assert_eq!(
            host.backend
                .read(&host.entry(&output).unwrap().tensor)
                .unwrap(),
            2.0_f32.to_le_bytes()
        );
        assert!(matches!(
            host.prepare_replay(&graph, Vec::new()).unwrap().run().await,
            Err(compute::Error::Quota(_))
        ));
    }

    #[test]
    fn converts_and_records_wit_programs() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = host.command_list().unwrap();
        let program = doubling_program(2.0);

        let kernel = host.create_kernel(program, unary_f32_signature(1)).unwrap();
        host.dispatch_kernel(
            &Resource::new_borrow(commands.rep()),
            &kernel,
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
    fn parameterized_views_preserve_their_parameter_space() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let tensor = host.alloc(compute::Dtype::F32, &[2, 4]).unwrap();
        let params = host
            .params(vec![compute::ParamRange { lo: 0, hi: 2 }])
            .unwrap();
        let view = host
            .view_param(
                &tensor,
                &params,
                vec![
                    compute::ParamSlice {
                        start: compute::Affine {
                            param: None,
                            scale: 0,
                            offset: 0,
                        },
                        len: compute::Affine {
                            param: None,
                            scale: 0,
                            offset: 2,
                        },
                        step: 1,
                    },
                    compute::ParamSlice {
                        start: compute::Affine {
                            param: Some(0),
                            scale: 1,
                            offset: 0,
                        },
                        len: compute::Affine {
                            param: None,
                            scale: 0,
                            offset: 2,
                        },
                        step: 1,
                    },
                ],
            )
            .unwrap();
        let permuted = host
            .view(&view, compute::ViewOp::Permute(vec![1, 0]))
            .unwrap();

        let symbolic = host.entry(&permuted).unwrap().symbolic.as_ref().unwrap();
        let values = symbolic.space().values(vec![2]).unwrap();
        assert_eq!(symbolic.instantiate(&values).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn affine_sdpa_positions_remain_parameterized_when_recorded() {
        let (operation, concrete) = core_op(compute::Op::Sdpa(compute::SdpaCfg {
            scale: 0.088,
            causal: true,
            q_start: wit_affine(Some(0), 2, 1),
        }));

        assert!(operation.is_parameter_dependent());
        assert!(concrete.is_none());

        let (operation, concrete) = core_op(compute::Op::Sdpa(compute::SdpaCfg {
            scale: 0.088,
            causal: true,
            q_start: wit_affine(None, 0, 7),
        }));
        assert!(!operation.is_parameter_dependent());
        assert_eq!(
            concrete,
            Some(Op::Sdpa {
                scale: 0.088,
                causal: true,
                q_start: 7,
            })
        );
    }

    #[test]
    fn kernel_quota_counts_live_preparations() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_kernel_limit(2));
        let first = host
            .create_kernel(doubling_program(2.0), unary_f32_signature(1))
            .unwrap();
        let duplicate = host
            .create_kernel(doubling_program(2.0), unary_f32_signature(1))
            .unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 2);
        assert!(matches!(
            host.create_kernel(doubling_program(3.0), unary_f32_signature(1)),
            Err(compute::Error::Quota(_))
        ));

        host.drop_kernel(first).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);
        host.drop_kernel(duplicate).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
        host.create_kernel(doubling_program(3.0), unary_f32_signature(1))
            .unwrap();
    }

    #[test]
    fn epilogues_share_the_prepared_program_quota() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_kernel_limit(1));
        let epilogue = host
            .create_epilogue(doubling_program(2.0), unary_f32_epilogue(1))
            .unwrap();
        let prepared = host.table.get(&epilogue).unwrap().prepared();
        assert_eq!(prepared.signature().accumulator_count(), 1);
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);
        assert!(matches!(
            host.create_kernel(doubling_program(3.0), unary_f32_signature(1)),
            Err(compute::Error::Quota(_))
        ));

        host.drop_epilogue(epilogue).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
        host.create_kernel(doubling_program(3.0), unary_f32_signature(1))
            .unwrap();
    }

    #[test]
    fn rejected_epilogue_releases_its_quota_lease() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_kernel_limit(1));
        let mut signature = unary_f32_epilogue(1);
        signature.accumulator_count = 0;

        assert!(matches!(
            host.create_epilogue(doubling_program(2.0), signature),
            Err(compute::Error::OpSignature(_))
        ));
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
    }

    #[test]
    fn command_list_retains_bound_epilogues() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS.with_kernel_limit(1));
        let left = host.alloc(compute::Dtype::F32, &[7, 33]).unwrap();
        let right = host.alloc(compute::Dtype::F32, &[33, 17]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7, 17]).unwrap();
        let commands = host.command_list().unwrap();
        let epilogue = host
            .create_epilogue(doubling_program(2.0), unary_f32_epilogue(2))
            .unwrap();

        host.dispatch_epilogue(
            &commands,
            compute::Op::Matmul,
            &[left, right],
            &output,
            &epilogue,
            &[],
        )
        .unwrap();
        assert!(
            host.table
                .get(&commands)
                .unwrap()
                .commands
                .last_dispatch()
                .unwrap()
                .prepared_epilogue()
                .is_some()
        );

        host.drop_epilogue(epilogue).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);
        host.drop_command_list(commands).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
    }

    #[test]
    fn kernel_dispatch_refuses_signature_mismatches() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let kernel = host
            .create_kernel(doubling_program(2.0), unary_f32_signature(2))
            .unwrap();
        let commands = host.command_list().unwrap();

        assert!(matches!(
            host.dispatch_kernel(&commands, &kernel, &[input], &[output]),
            Err(compute::Error::OpSignature(_))
        ));
        assert!(host.table.get(&commands).unwrap().commands.is_empty());
    }

    #[test]
    fn kernel_creation_refuses_reserved_scalars() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let mut signature = unary_f32_signature(1);
        signature.scalars = 1;
        assert!(matches!(
            host.create_kernel(doubling_program(2.0), signature),
            Err(compute::Error::OpSignature(_))
        ));
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
    }

    #[test]
    fn kernel_from_another_store_is_invalid() {
        let mut first = Host::new(CpuBackend::new(), GENEROUS);
        let kernel = first
            .create_kernel(doubling_program(2.0), unary_f32_signature(1))
            .unwrap();
        let mut second = Host::new(CpuBackend::new(), GENEROUS);
        let input = second.alloc(compute::Dtype::F32, &[7]).unwrap();
        let output = second.alloc(compute::Dtype::F32, &[7]).unwrap();
        let commands = second.command_list().unwrap();

        assert!(matches!(
            second.dispatch_kernel(
                &commands,
                &Resource::new_borrow(kernel.rep()),
                &[input],
                &[output],
            ),
            Err(compute::Error::InvalidHandle(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropped_kernel_is_retained_until_submission_completes() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let mut host = Host::new(backend, GENEROUS.with_kernel_limit(1));
        let input = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        host.write(&input, &3.0_f32.to_le_bytes()).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let kernel = host
            .create_kernel(doubling_program(2.0), unary_f32_signature(1))
            .unwrap();
        let commands = host.command_list().unwrap();
        host.dispatch_kernel(
            &commands,
            &kernel,
            &[input],
            &[Resource::new_borrow(output.rep())],
        )
        .unwrap();
        host.drop_kernel(kernel).unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);

        let request = host.prepare_submit(commands).unwrap();
        let task = tokio::spawn(request.run());
        gate.wait_for(1);
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 1);
        gate.release();
        task.await.unwrap().unwrap();
        assert_eq!(host.live_kernels.load(Ordering::Acquire), 0);
        let bytes = host.prepare_read(&output).unwrap().run().await.unwrap();
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()).to_bits(),
            6.0_f32.to_bits()
        );
    }

    #[test]
    fn program_validation_errors_name_the_instruction() {
        let error = core_program(compute::ProgramSource {
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
        let error = core_program(compute::ProgramSource {
            kind: compute::ProgramKind::Map,
            insts: instructions,
            outputs: vec![(0, 0)],
        })
        .unwrap_err();
        assert!(
            matches!(error, compute::Error::OpSignature(message) if message.contains("TooManyInstructions"))
        );

        let error = core_program(compute::ProgramSource {
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
        let error = core_program(compute::ProgramSource {
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
    fn memory_profile_combines_quota_charges_and_reports_the_gap() {
        let memory = EngineMemory {
            live_bytes: 4,
            charged_bytes: 7,
            device_bytes: Some(17),
            physical_footprint: Some(23),
            peak_physical_footprint: Some(29),
        };
        assert_eq!(memory.accounted_bytes(), Some(11));
        assert_eq!(memory.unaccounted_bytes(), Some(6));

        let overflow = EngineMemory {
            live_bytes: u64::MAX,
            charged_bytes: 1,
            ..EngineMemory::default()
        };
        assert_eq!(overflow.accounted_bytes(), None);
    }

    #[test]
    fn host_configures_the_backend_memory_limit() {
        let backend = AccountingBackend::new(None, None, Duration::ZERO);
        let configured = Arc::clone(&backend.memory_limit);
        let limits = Limits::new(4097, 8, u64::MAX, 32, u64::MAX);

        let _host = Host::new(backend, limits);

        assert_eq!(configured.load(Ordering::Acquire), 4097);
    }

    #[test]
    fn engine_memory_subtracts_live_storage_from_backend_charges() {
        let backend = AccountingBackend::with_memory_usage(MemoryUsage {
            charged_bytes: 13,
            ..MemoryUsage::default()
        });
        let mut host = Host::new(backend, GENEROUS);
        host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert_eq!(
            host.engine_memory(),
            EngineMemory {
                live_bytes: 4,
                charged_bytes: 9,
                ..EngineMemory::default()
            }
        );

        let backend = AccountingBackend::with_memory_usage(MemoryUsage {
            charged_bytes: 3,
            ..MemoryUsage::default()
        });
        let mut host = Host::new(backend, GENEROUS);
        host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert_eq!(host.engine_memory().charged_bytes, 0);
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

    #[test]
    fn copied_weights_obey_and_release_the_byte_quota() {
        let path = test_unaligned_weight_file();
        let grants = Grants::new().with_weights("model", &path);
        let limits = Limits::new(4, 8, u64::MAX, 8, u64::MAX);
        let mut host = Host::with_grants(CpuBackend::new(), limits, grants);
        let weights = host.open_weights("model").unwrap();

        let first = host
            .weight_tensor(&Resource::new_borrow(weights.rep()), "unaligned")
            .unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 4);
        assert!(matches!(
            host.weight_tensor(&Resource::new_borrow(weights.rep()), "unaligned"),
            Err(compute::Error::Quota(_))
        ));

        host.drop_tensor(first).unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        let second = host
            .weight_tensor(&Resource::new_borrow(weights.rep()), "unaligned")
            .unwrap();
        host.drop_tensor(second).unwrap();
        host.drop_weights(weights).unwrap();
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

    fn test_unaligned_weight_file() -> std::path::PathBuf {
        let mut header = br#"{"aligned":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"unaligned":{"dtype":"U32","shape":[1],"data_offsets":[2,6]}}"#.to_vec();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0; 6]);
        let path = std::env::temp_dir().join(format!(
            "forja-unaligned-weight-{}-{}",
            std::process::id(),
            NEXT_WEIGHT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn test_forced_copy_weight_file() -> std::path::PathBuf {
        let mut header = br#"{"aligned":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"unaligned":{"dtype":"U32","shape":[1],"data_offsets":[2,6]}}"#.to_vec();
        while (header.len() + 8).is_multiple_of(16) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0; 6]);
        let path = std::env::temp_dir().join(format!(
            "forja-copied-weight-{}-{}",
            std::process::id(),
            NEXT_WEIGHT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_copied_weight_imports_share_charge_and_release() {
        let path = test_forced_copy_weight_file();
        let imports = Arc::new(AtomicUsize::new(0));
        let releases = Arc::new(AtomicUsize::new(0));
        let backend = ImportCountingBackend {
            inner: forja_metal::MetalBackend::new().unwrap(),
            imports: Arc::clone(&imports),
            releases: Arc::clone(&releases),
        };
        let grants = Grants::new().with_weights("model", &path);
        let mut host =
            Host::with_grants(backend, Limits::new(10, 8, u64::MAX, 8, u64::MAX), grants);

        let first = host.open_weights("model").unwrap();
        let second = host.open_weights("model").unwrap();
        assert_eq!(imports.load(Ordering::Acquire), 1);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 6);
        fs::remove_file(path).unwrap();

        let tensor = host
            .weight_tensor(&Resource::new_borrow(first.rep()), "unaligned")
            .unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 10);
        host.drop_tensor(tensor).unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 6);
        host.drop_weights(first).unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 6);
        host.drop_weights(second).unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        assert_eq!(releases.load(Ordering::Acquire), 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_copied_weight_import_refuses_byte_quota() {
        let path = test_forced_copy_weight_file();
        let imports = Arc::new(AtomicUsize::new(0));
        let releases = Arc::new(AtomicUsize::new(0));
        let backend = ImportCountingBackend {
            inner: forja_metal::MetalBackend::new().unwrap(),
            imports: Arc::clone(&imports),
            releases: Arc::clone(&releases),
        };
        let grants = Grants::new().with_weights("model", &path);
        let mut host = Host::with_grants(backend, Limits::new(5, 8, u64::MAX, 8, u64::MAX), grants);

        assert!(matches!(
            host.open_weights("model"),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        assert_eq!(imports.load(Ordering::Acquire), 1);
        assert_eq!(releases.load(Ordering::Acquire), 1);
        fs::remove_file(path).unwrap();
    }

    struct ImportCountingBackend<B> {
        inner: B,
        imports: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl<B: Backend> Backend for ImportCountingBackend<B> {
        type Submission = B::Submission;
        type ProgramHandle = B::ProgramHandle;

        fn prepare_program(
            &self,
            program: &ValidatedProgram,
            signature: &KernelSignature,
        ) -> Result<Self::ProgramHandle, BackendError> {
            self.inner.prepare_program(program, signature)
        }

        fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
            self.inner.alloc(dtype, shape)
        }

        fn import_readonly(
            &self,
            bytes: MappedRegion,
        ) -> Result<forja_core::ReadonlyImport, BackendError> {
            self.imports.fetch_add(1, Ordering::AcqRel);
            self.inner.import_readonly(bytes)
        }

        fn read_buffer_range(
            &self,
            buffer: BufferId,
            range: std::ops::Range<u64>,
        ) -> Result<Vec<u8>, BackendError> {
            self.inner.read_buffer_range(buffer, range)
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

    fn assert_empty_tensors_are_layout_errors<B: Backend + 'static>(backend: B) {
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
                compute::Op::RmsNorm(0.00001),
                vec![
                    (compute::Dtype::F32, vec![7, 1024]),
                    (compute::Dtype::F32, vec![1024]),
                ],
                (compute::Dtype::F32, vec![7, 1024]),
            ),
            (
                compute::Op::Argmax(compute::ReductionCfg { chunk_width: 2_048 }),
                vec![(compute::Dtype::F32, vec![7, 33])],
                (compute::Dtype::U32, vec![7]),
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
                    q_start: wit_affine(None, 0, 0),
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

            let recorded = host.table.get(&commands).unwrap();
            assert_eq!(recorded.commands.clone().into_dispatches().len(), 1);
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
    fn refuses_large_program_iteration_work() {
        let limits = GENEROUS.with_command_limits(usize::MAX, 10_000_000);
        let mut host = Host::new(CpuBackend::new(), limits);
        let scalar = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        let input = host
            .view(
                &Resource::new_borrow(scalar.rep()),
                compute::ViewOp::Broadcast(vec![1_000_000]),
            )
            .unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1_000_000]).unwrap();
        let commands = host.command_list().unwrap();
        let mut insts = vec![compute::Inst::Input(0)];
        for operand in 0..255 {
            insts.push(compute::Inst::Unary((compute::Unop::Neg, operand)));
        }

        assert!(matches!(
            {
                let kernel = host
                    .create_kernel(
                        compute::ProgramSource {
                            kind: compute::ProgramKind::Map,
                            insts,
                            outputs: vec![(0, 255)],
                        },
                        compute::KernelSignature {
                            rank: 1,
                            inputs: vec![compute::Dtype::F32],
                            outputs: vec![compute::Dtype::F32],
                            scalars: 0,
                        },
                    )
                    .unwrap();
                host.dispatch_kernel(&commands, &kernel, &[input], &[output])
            },
            Err(compute::Error::Quota(_))
        ));
        assert!(host.table.get(&commands).unwrap().commands.is_empty());
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
        let diagnostic = super::guest_error(BackendError::DeviceErrorFlag {
            submission: 17,
            dispatch: 3,
            operation: "gather-matmul",
            flag: 1,
            value: 7,
        });
        assert!(matches!(
            diagnostic,
            compute::Error::BackendExecution(message)
                if message == "backend error: DeviceErrorFlag { submission: 17, dispatch: 3, operation: \"gather-matmul\", flag: 1, value: 7 }"
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
        let mut tasks = tokio::task::JoinSet::new();
        for request in requests {
            tasks.spawn(request.run());
        }

        gate.wait_for(SUBMITS - 1);
        let refused = tasks.join_next().await.unwrap().unwrap();
        gate.release();
        assert!(matches!(refused, Err(compute::Error::Quota(_))));
        let mut accepted = 0;
        while let Some(task) = tasks.join_next().await {
            match task.unwrap() {
                Ok(_) => accepted += 1,
                Err(error) => panic!("unexpected submit error: {error:?}"),
            }
        }
        assert_eq!(accepted, SUBMITS - 1);
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
                live_bytes: 0,
                live_graphs: 0,
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
        fn count(&self) -> usize {
            self.state.lock().unwrap().0
        }

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
        fail_submission: Option<u64>,
        submissions: AtomicU64,
        memory_usage: MemoryUsage,
        memory_limit: Arc<AtomicU64>,
        graph_memory_depth: usize,
    }

    impl AccountingBackend {
        fn new(gate: Option<Arc<SubmitGate>>, gpu_time: Option<Duration>, delay: Duration) -> Self {
            Self {
                inner: CpuBackend::new(),
                gate,
                gpu_time,
                delay,
                wait_error: None,
                fail_submission: None,
                submissions: AtomicU64::new(0),
                memory_usage: MemoryUsage::default(),
                memory_limit: Arc::new(AtomicU64::new(u64::MAX)),
                graph_memory_depth: 0,
            }
        }

        fn with_graph_memory_depth(mut self, graph_memory_depth: usize) -> Self {
            self.graph_memory_depth = graph_memory_depth;
            self
        }

        fn with_memory_usage(memory_usage: MemoryUsage) -> Self {
            Self {
                memory_usage,
                ..Self::new(None, None, Duration::ZERO)
            }
        }

        fn failing_nth(gate: Arc<SubmitGate>, submission: u64) -> Self {
            Self {
                inner: CpuBackend::new(),
                gate: Some(gate),
                gpu_time: None,
                delay: Duration::ZERO,
                wait_error: None,
                fail_submission: Some(submission),
                submissions: AtomicU64::new(0),
                memory_usage: MemoryUsage::default(),
                memory_limit: Arc::new(AtomicU64::new(u64::MAX)),
                graph_memory_depth: 0,
            }
        }

        fn timing_out() -> Self {
            Self {
                inner: CpuBackend::new(),
                gate: None,
                gpu_time: None,
                delay: Duration::ZERO,
                wait_error: Some(BackendError::Timeout),
                fail_submission: None,
                submissions: AtomicU64::new(0),
                memory_usage: MemoryUsage::default(),
                memory_limit: Arc::new(AtomicU64::new(u64::MAX)),
                graph_memory_depth: 0,
            }
        }
    }

    struct AccountingSubmission {
        inner: <CpuBackend as Backend>::Submission,
        _commands: CommandList,
        gate: Option<Arc<SubmitGate>>,
        gpu_time: Option<Duration>,
        delay: Duration,
        wait_error: Option<BackendError>,
        sequence: u64,
        fail_submission: Option<u64>,
    }

    impl Submission for AccountingSubmission {
        fn wait(&self) -> Result<(), BackendError> {
            if let Some(gate) = &self.gate {
                gate.wait();
            }
            self.inner.wait()?;
            std::thread::sleep(self.delay);
            if self.fail_submission == Some(self.sequence) {
                return Err(BackendError::ExecutionFailed);
            }
            self.wait_error.map_or(Ok(()), Err)
        }

        fn gpu_time(&self) -> Option<Duration> {
            self.gpu_time
        }
    }

    impl Backend for AccountingBackend {
        type Submission = AccountingSubmission;
        type ProgramHandle = <CpuBackend as Backend>::ProgramHandle;

        fn memory_usage(&self) -> MemoryUsage {
            self.memory_usage
        }

        fn set_memory_limit(&self, bytes: u64) {
            self.memory_limit.store(bytes, Ordering::Release);
        }

        fn prepare_program(
            &self,
            program: &ValidatedProgram,
            signature: &KernelSignature,
        ) -> Result<Self::ProgramHandle, BackendError> {
            self.inner.prepare_program(program, signature)
        }

        fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
            self.inner.alloc(dtype, shape)
        }

        fn import_readonly(
            &self,
            bytes: MappedRegion,
        ) -> Result<forja_core::ReadonlyImport, BackendError> {
            self.inner.import_readonly(bytes)
        }

        fn read_buffer_range(
            &self,
            buffer: BufferId,
            range: std::ops::Range<u64>,
        ) -> Result<Vec<u8>, BackendError> {
            self.inner.read_buffer_range(buffer, range)
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
            self.inner.release(tensor)
        }

        fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
            let retained = commands.clone();
            let inner = self.inner.submit(commands)?;
            let sequence = self
                .submissions
                .fetch_add(1, Ordering::AcqRel)
                .saturating_add(1);
            Ok(AccountingSubmission {
                inner,
                _commands: retained,
                gate: self.gate.clone(),
                gpu_time: self.gpu_time,
                delay: self.delay,
                wait_error: self.wait_error,
                sequence,
                fail_submission: self.fail_submission,
            })
        }

        fn supports_replay_overlap(&self) -> bool {
            true
        }

        fn graph_memory_depth(&self) -> usize {
            self.graph_memory_depth
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
        type ProgramHandle = <CpuBackend as Backend>::ProgramHandle;

        fn prepare_program(
            &self,
            program: &ValidatedProgram,
            signature: &KernelSignature,
        ) -> Result<Self::ProgramHandle, BackendError> {
            self.inner.prepare_program(program, signature)
        }

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
