//! Trusted host integration for running Forja guest components.

#![allow(clippy::manual_async_fn)]

#[cfg(test)]
mod fuzz_tests;
mod native;
mod weights;

use std::time::{Duration, Instant};
use std::{
    collections::{HashMap, HashSet},
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

use forja_core::{
    Affine, Backend, BackendError, BufferId, CommandList, DType, GraphLimits, GraphTemplate,
    Layout, LayoutError, Op, OpError, ParamSpace, PreparedGraph, Slice, Submission,
    SubmissionProfile, SymbolicLayout, SymbolicLayoutError, TemplateOp, TemplateTensor, Tensor,
    ViewOp, gather_matmul_flops, gather_quant_matmul_flops, gather_quant_silu_mul_flops,
    matmul_flops,
    program::{
        BinOp, Inst, KernelSignature, MAX_INSTRUCTIONS, MAX_OUTPUTS, PrepareError, PreparedProgram,
        Program, ProgramError, ProgramKind, RedOp, UnOp, ValidatedProgram, ValueType,
        prepare_program_retained,
    },
    quant_matmul_flops, sdpa_flops,
};
use wasmtime::component::{Accessor, Component, HasData, Linker, Resource, ResourceTable};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder, Trap};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

pub use native::{NativeCommandList, NativeHost, NativeKernel, NativeTensor};
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
            "l9o:gpu/compute.command-list": crate::CommandListEntry,
            "l9o:gpu/compute.weights": crate::WeightsEntry,
            "l9o:gpu/compute.params": crate::ParamsEntry,
            "l9o:gpu/compute.graph": crate::GraphEntry,
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

/// Cumulative execution counters for an engine runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineMetrics {
    /// Bytes held by live guest tensor resources.
    pub live_bytes: u64,
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
    /// Host reads of returned tensors after the guest call.
    pub output_read: ImportProfile,
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
    loaded_num_hidden_layers: Option<u32>,
    output_handles: Vec<u32>,
    discarded_speculation: bool,
    profiling: bool,
    last_profile: Option<EngineStepProfile>,
}

type EngineCallFuture<'a, T> =
    Pin<Box<dyn Future<Output = wasmtime::Result<Result<T, compute::Error>>> + Send + 'a>>;

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
            loaded_num_hidden_layers: None,
            output_handles: Vec::new(),
            discarded_speculation: false,
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
            router_layers: info.router_layers,
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
                        engine_bindings::exports::l9o::gpu::engine::LoadConfig {
                            num_hidden_layers,
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

    async fn decode_inner(
        &mut self,
        input: EngineDecode,
        release_previous: bool,
    ) -> wasmtime::Result<Result<EngineDecodeOutput, compute::Error>> {
        use engine_bindings::exports::l9o::gpu::engine::{DecodeIn, DecodeOut};
        if self.discarded_speculation {
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
            Ok(output) => output?,
            Err(error) if is_epoch_timeout(&error) => return Ok(Err(guest_timeout())),
            Err(error) => return Err(error),
        }?;
        let DecodeOut { logits, token } = output;
        let handles = vec![logits.rep(), token.rep()];
        if let Err(error) = self.validate_decode_output(&info, &logits, &token) {
            return Ok(Err(match self.release_handles(handles) {
                Ok(()) => error,
                Err(release_error) => release_error,
            }));
        }
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
        let result = if output.token.runner_id == self.id {
            let resource = Resource::new_borrow(output.token.handle);
            match self.store.data().prepare_replay_read(&resource) {
                Ok(request) => request.run().await.map_err(guest_error),
                Err(error) => Err(error),
            }
        } else {
            Err(invalid_handle("engine tensor belongs to another runner"))
        };
        let release = self.release_decode_output(output);
        match (result, release) {
            (Err(error), _) | (Ok(_), Err(error)) => Err(error),
            (Ok(bytes), Ok(())) => Ok(bytes),
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

    /// Waits until the requested number of backend submissions has completed successfully.
    ///
    /// # Errors
    ///
    /// Returns a timeout if completion accounting does not advance within the backend deadline.
    pub async fn wait_for_submissions(&self, expected: u64) -> Result<(), BackendError> {
        let timeout = self.store.data().limits.submission_timeout;
        let notify = Arc::clone(&self.store.data().submission_notify);
        let started = Instant::now();
        loop {
            let notified = notify.notified();
            if self.metrics().submissions >= expected {
                return Ok(());
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() || tokio::time::timeout(remaining, notified).await.is_err() {
                return Err(BackendError::Timeout);
            }
        }
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
}

#[derive(Debug)]
enum BufferKind {
    Allocated {
        byte_len: u64,
        live_bytes: Arc<AtomicU64>,
    },
    Weights(Safetensors),
    CopiedWeight {
        byte_len: u64,
        live_bytes: Arc<AtomicU64>,
    },
}

impl BufferHandle {
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
            BufferKind::Weights(_) => Ok(()),
        }
    }

    fn weights(&self) -> Option<&Safetensors> {
        match &self.kind {
            BufferKind::Allocated { .. } | BufferKind::CopiedWeight { .. } => None,
            BufferKind::Weights(source) => Some(source),
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
    Program {
        program: Arc<PreparedProgram>,
        inputs: Vec<TemplateTensor>,
        outputs: Vec<TemplateTensor>,
    },
}

/// Host-owned state behind a guest command-list resource.
#[derive(Debug)]
pub struct CommandListEntry {
    commands: CommandList,
    recorded: Vec<RecordedDispatch>,
    access: SubmissionAccess,
    space: Option<ParamSpace>,
    parameterized: bool,
    retained: Vec<TensorEntry>,
    retained_kernels: Vec<KernelEntry>,
}

/// Host-owned state behind a guest graph resource.
#[derive(Debug)]
pub struct GraphEntry {
    graph: Arc<PreparedGraph>,
    access: SubmissionAccess,
    retained: Vec<TensorEntry>,
    retained_kernels: Vec<KernelEntry>,
    replay: Arc<ReplayState>,
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
    gpu_time_ns: Arc<AtomicU64>,
    completed_submissions: Arc<AtomicU64>,
    timed_submissions: Arc<AtomicU64>,
    completed_gpu_time_ns: Arc<AtomicU64>,
    submission_notify: Arc<tokio::sync::Notify>,
    taints: Arc<BufferTaints>,
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
            live_kernels: Arc::new(AtomicUsize::new(0)),
            live_graphs: Arc::new(AtomicUsize::new(0)),
            gpu_time_ns: Arc::new(AtomicU64::new(0)),
            completed_submissions: Arc::new(AtomicU64::new(0)),
            timed_submissions: Arc::new(AtomicU64::new(0)),
            completed_gpu_time_ns: Arc::new(AtomicU64::new(0)),
            submission_notify: Arc::new(tokio::sync::Notify::new()),
            taints: Arc::new(BufferTaints::default()),
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
            live_bytes: self.live_bytes.load(Ordering::Acquire),
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
            symbolic: None,
            buffer: Arc::new(BufferHandle {
                owner: tensor.clone(),
                taints: Arc::clone(&self.taints),
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
        let buffers = if let Some(buffers) = self
            .weight_files
            .get(grant)
            .and_then(|buffers| buffers.iter().map(Weak::upgrade).collect())
        {
            buffers
        } else {
            let sources = Safetensors::open_all(path).map_err(weight_error)?;
            let mut buffers = Vec::with_capacity(sources.len());
            for source in sources {
                match self.import_weight_source(source) {
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
            .find(|tensor| tensor.is_aligned())
            .ok_or_else(|| compute::Error::Layout("weight file contains no tensors".to_owned()))?;
        let region = source.mapped_region().map_err(weight_error)?;
        let buffer_len =
            u64::try_from(region.len()).map_err(|_| guest_error(BackendError::AllocationFailed))?;
        let owner_layout = first.layout(buffer_len).map_err(guest_error)?;
        let buffer_id = self.backend.import_readonly(region).map_err(guest_error)?;
        let owner = self
            .backend
            .tensor(buffer_id, owner_layout)
            .map_err(guest_error)?;
        Ok(Arc::new(BufferHandle {
            owner,
            kind: BufferKind::Weights(source),
            taints: Arc::clone(&self.taints),
        }))
    }

    fn weight_tensor(
        &mut self,
        resource: &Resource<WeightsEntry>,
        name: &str,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        let (buffer, source, metadata) = entry
            .buffers
            .iter()
            .find_map(|buffer| {
                let source = buffer.weights()?;
                source
                    .tensors()
                    .iter()
                    .find(|tensor| tensor.name() == name)
                    .map(|metadata| (buffer, source, metadata))
            })
            .ok_or_else(|| invalid_handle("weight tensor is not present"))?;
        self.check_tensor_shape(metadata.shape())?;
        let (tensor, buffer) = if metadata.is_aligned() {
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
            let region = source.mapped_region().map_err(weight_error)?;
            let start = usize::try_from(metadata.byte_offset())
                .map_err(|_| guest_error(BackendError::AllocationFailed))?;
            let len = usize::try_from(metadata.byte_len())
                .map_err(|_| guest_error(BackendError::AllocationFailed))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| guest_error(BackendError::AllocationFailed))?;
            let bytes = region
                .bytes()
                .get(start..end)
                .ok_or_else(|| guest_error(BackendError::InvalidInput))?;
            let tensor = self
                .backend
                .alloc(metadata.dtype(), metadata.shape())
                .map_err(guest_error)?;
            if let Err(error) = self.backend.write(&tensor, bytes) {
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
                space: None,
                parameterized: false,
                retained: Vec::new(),
                retained_kernels: Vec::new(),
            })
            .map_err(invalid_handle)
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
        let (template_op, concrete_op) = core_op(operation);
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

    fn drop_command_list(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<(), compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
        self.release_retained(entry.retained)
    }

    fn create_graph(
        &mut self,
        resource: Resource<CommandListEntry>,
    ) -> Result<Resource<GraphEntry>, compute::Error> {
        let entry = self.table.delete(resource).map_err(invalid_handle)?;
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
            .try_for_each(|dispatch| match dispatch {
                RecordedDispatch::Static(dispatch) => {
                    graph.record_validated(dispatch.as_ref().clone())
                }
                RecordedDispatch::Operation {
                    op,
                    inputs,
                    outputs,
                } => {
                    let inputs = inputs.iter().collect::<Vec<_>>();
                    let outputs = outputs.iter().collect::<Vec<_>>();
                    graph.dispatch_many(*op, &inputs, &outputs)
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
            });
        let graph = match built {
            Ok(()) => self.backend.prepare_graph(graph).map_err(guest_error),
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
            replay: Arc::new(ReplayState::default()),
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
            flight: None,
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
            completed_submissions: Arc::clone(&self.completed_submissions),
            timed_submissions: Arc::clone(&self.timed_submissions),
            completed_gpu_time_ns: Arc::clone(&self.completed_gpu_time_ns),
            submission_notify: Arc::clone(&self.submission_notify),
            taints: Arc::clone(&self.taints),
            profile: self.active_profile.clone(),
        })
    }

    fn prepare_replay(
        &self,
        resource: &Resource<GraphEntry>,
        values: Vec<u32>,
    ) -> Result<SubmitRequest<B>, compute::Error> {
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        entry
            .graph
            .values(values.clone())
            .map_err(|error| graph_error(&error))?;
        let flight = ReplayFlight::acquire(Arc::clone(&entry.replay))?;
        Ok(SubmitRequest {
            backend: Arc::clone(&self.backend),
            work: SubmissionWork::Replay {
                graph: Arc::clone(&entry.graph),
                values,
            },
            access: entry.access.clone(),
            retained: Vec::new(),
            flight: Some(flight),
            timeout: self.limits.submission_timeout,
            gpu_time_budget_ns: duration_ns(self.limits.gpu_time_budget),
            gpu_time_ns: Arc::clone(&self.gpu_time_ns),
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
            Op::GatherQuantMatmul { .. } => inputs
                .first()
                .zip(inputs.get(1))
                .zip(inputs.get(4))
                .and_then(|((input, packed), indices)| {
                    gather_quant_matmul_flops(
                        input.layout().shape(),
                        packed.layout().shape(),
                        indices.layout().shape(),
                    )
                }),
            Op::GatherQuantSiluMul { .. } => inputs
                .first()
                .zip(inputs.get(1))
                .zip(inputs.get(7))
                .and_then(|((input, packed), indices)| {
                    gather_quant_silu_mul_flops(
                        input.layout().shape(),
                        packed.layout().shape(),
                        indices.layout().shape(),
                    )
                }),
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
    flight: Option<ReplayFlight>,
    timeout: Duration,
    gpu_time_budget_ns: u64,
    gpu_time_ns: Arc<AtomicU64>,
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

const DEFAULT_REPLAY_DEPTH: usize = 2;

#[derive(Debug, Default)]
struct ReplayState {
    status: Mutex<ReplayStatus>,
    #[cfg(test)]
    drop_barrier: Mutex<Option<Arc<std::sync::Barrier>>>,
}

#[derive(Debug, Default)]
struct ReplayStatus {
    in_flight: usize,
}

struct ReplayFlight(Arc<ReplayState>);

impl ReplayFlight {
    fn acquire(replay: Arc<ReplayState>) -> Result<Self, compute::Error> {
        let mut status = replay
            .status
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
        status.in_flight = status
            .in_flight
            .checked_add(1)
            .filter(|&depth| depth <= DEFAULT_REPLAY_DEPTH)
            .ok_or_else(|| {
                compute::Error::OpSignature("graph replay depth exceeds host limit".to_owned())
            })?;
        drop(status);
        Ok(Self(replay))
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
                flight,
                timeout,
                gpu_time_budget_ns: _,
                gpu_time_ns: _,
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
                        Ok(submission) => Ok((submission, reservation, taint)),
                    },
                },
            };
            let (submission, reservation, taint) = match submitted {
                Ok(submitted) => {
                    if let Some(started) = started {
                        let _ = started.send(Ok(()));
                    }
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
            let wall_time = duration_ns(wait_started.elapsed()).max(1);
            let gpu_time = submission.gpu_time().map(duration_ns);
            let submission_profile = submission.profile();
            let charged = if wait == Err(BackendError::Timeout) {
                reservation.amount()
            } else {
                gpu_time.unwrap_or(wall_time).max(1)
            };
            let accounting = reservation.settle(charged);
            let result = match (wait, accounting) {
                (Err(error), _) | (Ok(()), Err(error)) => Err(error),
                (Ok(()), Ok(())) => {
                    if let (Some(target), Some(submission_profile)) = (&profile, submission_profile)
                        && let Ok(mut target) = target.lock()
                    {
                        target.submission = Some(submission_profile);
                    }
                    Ok(gpu_time)
                }
            };
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

    async fn replay(
        accessor: &Accessor<Host<B>, Self>,
        resource: Resource<GraphEntry>,
        values: Vec<u32>,
    ) -> wasmtime::Result<Result<Option<u64>, compute::Error>> {
        let timer = accessor.with(|mut access| access.get().import_timer(ImportKind::Submit));
        let request = accessor.with(|mut access| access.get().prepare_replay(&resource, values));
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

fn core_op(operation: compute::Op) -> (TemplateOp, Option<Op>) {
    match operation {
        compute::Op::Copy => concrete_template(Op::Copy),
        compute::Op::Add => concrete_template(Op::Add),
        compute::Op::SiluMul => concrete_template(Op::SiluMul),
        compute::Op::RmsNorm(eps) => concrete_template(Op::RmsNorm { eps }),
        compute::Op::Softmax => concrete_template(Op::Softmax),
        compute::Op::Argmax => concrete_template(Op::Argmax),
        compute::Op::TopK(config) => concrete_template(Op::TopK {
            k: config.k,
            normalize: config.normalize,
        }),
        compute::Op::Sample(config) => {
            let position = core_affine(config.position);
            let concrete = position.is_constant().then_some(Op::Sample {
                position: config.position.offset,
            });
            (TemplateOp::sample(position), concrete)
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
        compute::Op::GatherQuantSiluMul(config) => concrete_template(Op::GatherQuantSiluMul {
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
        Backend, BackendError, BufferId, CommandList, DType, Layout, LayoutError, MappedRegion, Op,
        OpError, Submission, Tensor, ViewOp as CoreViewOp,
        program::{KernelSignature, ValidatedProgram},
    };
    use forja_cpu::CpuBackend;
    use wasmtime::component::Resource;

    use super::{
        BackendEvent, BackendTimer, EngineMetrics, EngineStepProfile, Grants, Host, ImportKind,
        ImportTimer, Limits, MAX_INSTRUCTIONS, MAX_OUTPUTS, bindings::l9o::gpu::compute, core_op,
        core_program, expected_layer_outputs,
    };

    const GENEROUS: Limits = Limits::new(u64::MAX, 8, u64::MAX, 32, u64::MAX);
    static NEXT_WEIGHT_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn loaded_layer_count_limits_expected_taps() {
        let layers = [1, 2, 4, 8];
        assert_eq!(expected_layer_outputs(&layers, Some(4), true), 3);
        assert_eq!(expected_layer_outputs(&layers, None, true), 4);
        assert_eq!(expected_layer_outputs(&layers, Some(4), false), 0);
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

    fn wit_affine(param: Option<u8>, scale: u32, offset: u32) -> compute::Affine {
        compute::Affine {
            param,
            scale,
            offset,
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
        B: Backend,
    {
        let commands = host.command_list().unwrap();
        host.create_graph(commands).unwrap()
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
        assert_eq!(host.live_graphs.load(Ordering::Acquire), 1);
        let commands = host.command_list().unwrap();
        assert!(matches!(
            host.create_graph(commands),
            Err(compute::Error::Quota(_))
        ));
        host.drop_graph(graph).unwrap();
        assert_eq!(host.live_graphs.load(Ordering::Acquire), 0);
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
    async fn replay_refuses_depth_above_host_limit() {
        let gate = Arc::new(SubmitGate::default());
        let backend = AccountingBackend::new(Some(Arc::clone(&gate)), None, Duration::ZERO);
        let mut host = Host::new(backend, GENEROUS);
        let graph = empty_graph(&mut host);
        let first = host.prepare_replay(&graph, Vec::new()).unwrap();
        let second = host.prepare_replay(&graph, Vec::new()).unwrap();
        let first_task = tokio::spawn(first.run());
        let second_task = tokio::spawn(second.run());
        gate.wait_for(2);
        assert!(matches!(
            host.prepare_replay(&graph, Vec::new()),
            Err(compute::Error::OpSignature(_))
        ));
        gate.release();
        first_task.await.unwrap().unwrap();
        second_task.await.unwrap().unwrap();
        host.prepare_replay(&graph, Vec::new())
            .unwrap()
            .run()
            .await
            .unwrap();
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
        let replay = Arc::clone(&runner.store.data().table.get(&graph).unwrap().replay);
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
        let replay = Arc::clone(&host.table.get(&graph).unwrap().replay);

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

    struct ImportCountingBackend {
        inner: CpuBackend,
        imports: Arc<AtomicUsize>,
        releases: Arc<AtomicUsize>,
    }

    impl Backend for ImportCountingBackend {
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
                compute::Op::Argmax,
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
                live_bytes: 0,
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
        fail_submission: Option<u64>,
        submissions: AtomicU64,
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
