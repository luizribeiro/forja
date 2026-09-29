use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    ffi::c_void,
    fmt::Write as _,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex, OnceLock, Weak},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use block2::RcBlock;
use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, DispatchProfile, GraphTemplate, Layout,
    Op, PreparedGraph, ProfileCount, Slice, Submission, SubmissionProfile, Tensor,
    program::{KernelSignature, PreparedProgram, ProgramHash, ProgramKind, ValidatedProgram},
    required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue,
    MTL4CommitFeedback, MTL4CommitOptions, MTL4CounterHeap, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTL4TimestampGranularity, MTLAllocation, MTLBuffer, MTLCompileOptions,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLEvent, MTLFunctionConstantValues,
    MTLLibrary, MTLMathMode, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions,
    MTLSharedEvent, MTLSharedEventListener, MTLSize,
};

use crate::{
    map_codegen,
    matmul::{classify, select_gemm},
    storage::MetalBackend,
};

type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;
type EncodedEmbed = (Vec<BufferBinding>, BufferBinding);

#[derive(Clone)]
struct BufferBinding {
    raw: MetalBufferRef,
    offset: usize,
    len: usize,
    address: u64,
}

impl BufferBinding {
    fn whole(raw: MetalBufferRef) -> Self {
        let address = raw.gpuAddress();
        let len = raw.length();
        Self {
            raw,
            offset: 0,
            len,
            address,
        }
    }
}

struct ArgumentBuffer {
    raw: MetalBufferRef,
    capacity: usize,
}

struct ArgumentWriter {
    raw: MetalBufferRef,
    capacity: usize,
    offset: usize,
    written: Vec<(usize, usize)>,
}

impl ArgumentWriter {
    fn new(buffer: &ArgumentBuffer) -> Self {
        Self {
            raw: buffer.raw.clone(),
            capacity: buffer.capacity,
            offset: 0,
            written: Vec::new(),
        }
    }

    fn write(&mut self, bytes: &[u8]) -> Result<BufferBinding, BackendError> {
        let started = TEMPORARY_PROFILE.with(|profile| profile.get().map(|_| Instant::now()));
        let offset = reserve_argument(&mut self.offset, bytes.len(), self.capacity)?;
        // SAFETY: The checked range lies within the live shared allocation and argument writes are
        // completed before the command buffer can execute.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.raw.contents().cast::<u8>().as_ptr().add(offset),
                bytes.len(),
            );
        }
        self.written.push((offset, bytes.len()));
        if let Some(started) = started {
            TEMPORARY_PROFILE.with(|profile| {
                if let Some(mut timing) = profile.get() {
                    timing.count = 1;
                    timing.time = timing.time.saturating_add(started.elapsed());
                    profile.set(Some(timing));
                }
            });
        }
        let address = self
            .raw
            .gpuAddress()
            .checked_add(u64::try_from(offset).map_err(|_| BackendError::AllocationFailed)?)
            .ok_or(BackendError::AllocationFailed)?;
        Ok(BufferBinding {
            raw: self.raw.clone(),
            offset,
            len: bytes.len(),
            address,
        })
    }

    fn copy_image(&mut self, image: &[u8]) -> Result<(), BackendError> {
        let started = TEMPORARY_PROFILE.with(|profile| profile.get().map(|_| Instant::now()));
        if image.len() > self.capacity {
            return Err(BackendError::AllocationFailed);
        }
        // SAFETY: The image length was checked against the live shared allocation capacity, and
        // the copy completes before the command buffer can execute.
        unsafe {
            std::ptr::copy_nonoverlapping(
                image.as_ptr(),
                self.raw.contents().cast::<u8>().as_ptr(),
                image.len(),
            );
        }
        self.offset = image.len();
        if let Some(started) = started {
            TEMPORARY_PROFILE.with(|profile| {
                if let Some(mut timing) = profile.get() {
                    timing.count = timing.count.saturating_add(1);
                    timing.time = timing.time.saturating_add(started.elapsed());
                    profile.set(Some(timing));
                }
            });
        }
        Ok(())
    }

    fn validate_capacity(&self, expected: usize, empty: bool) -> Result<(), BackendError> {
        if !empty && self.offset != expected {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(())
    }
}

#[derive(Default)]
struct ArgumentSizer {
    offset: usize,
}

impl ArgumentSizer {
    fn write(&mut self, len: usize) -> Result<(), BackendError> {
        reserve_argument(&mut self.offset, len, usize::MAX).map(|_| ())
    }
}

fn reserve_argument(
    offset: &mut usize,
    len: usize,
    capacity: usize,
) -> Result<usize, BackendError> {
    let start = offset
        .checked_next_multiple_of(16)
        .ok_or(BackendError::AllocationFailed)?;
    *offset = start
        .checked_add(len)
        .filter(|&end| end <= capacity)
        .ok_or(BackendError::AllocationFailed)?;
    Ok(start)
}

#[derive(Clone)]
struct EncoderTensor {
    buffer: BufferBinding,
    layout: Layout,
}

#[derive(Clone, Copy)]
struct MatmulShape {
    left_dtype: DType,
    right_dtype: DType,
    output_dtype: DType,
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    left_column_major: bool,
    right_column_major: bool,
}

struct MatmulLaunch {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    kernel: &'static str,
    block_rows: u32,
    block_columns: u32,
    thread_count: usize,
}

struct PreparedMatmulInput {
    tensor: EncoderTensor,
    column_major: u32,
    leading_dimension: u64,
    batch_stride: u64,
    copied: bool,
}

struct PreparedMatmulOutput {
    tensor: EncoderTensor,
    leading_dimension: u64,
    batch_stride: u64,
    copied: bool,
}

fn log_matmul_route(
    left: &PreparedMatmulInput,
    right: &PreparedMatmulInput,
    output: &PreparedMatmulOutput,
    launch: &MatmulLaunch,
    shape: MatmulShape,
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !ENABLED.get_or_init(|| std::env::var_os("FORJA_METAL_LOG_MATMUL").is_some()) {
        return;
    }
    let layout = |column_major, copied| match (copied, column_major) {
        (true, _) => "needs-copy->row-major",
        (false, 0) => "row-major",
        (false, _) => "column-major",
    };
    eprintln!(
        "matmul route: [{}, {}, {}]x[{}, {}] path={} kernel={} layouts={}/{}/{} scratch={}/{}/{} staging-casts=none threads={} groups={}x{}x{} dtypes={:?}/{:?}/{:?}",
        shape.batch,
        shape.rows,
        shape.inner,
        shape.inner,
        shape.columns,
        if shape.rows == 1 { "gemv" } else { "gemm" },
        launch.kernel,
        layout(left.column_major, left.copied),
        layout(right.column_major, right.copied),
        layout(0, output.copied),
        left.copied,
        right.copied,
        output.copied,
        launch.thread_count,
        shape.columns.div_ceil(launch.block_columns),
        shape.rows.div_ceil(launch.block_rows),
        shape.batch,
        shape.left_dtype,
        shape.right_dtype,
        shape.output_dtype,
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SdpaKernel {
    Vector,
    Steel,
    Decomposed,
}

const VECTOR_TWO_PASS_MIN_KEY_LENGTH: u32 = 1024;
const VECTOR_SELECTED_MIN_KEY_LENGTH: u32 = 512;
const VECTOR_MAX_KEY_LENGTH: u32 = 65_536;
const STEEL_SELECTED_MIN_QUERY_LENGTH: u32 = 512;
const ARGMAX_CHUNK_WIDTH: u32 = 2048;

struct EncodedDispatches {
    temporaries: Vec<BufferBinding>,
    error_flags: Vec<BufferBinding>,
    bindings: ArgumentBindings,
    arguments: Option<ArgumentUsage>,
    program_encoding: ProfileCount,
    program_compile_fallbacks: u64,
}

#[derive(Default)]
struct ArgumentBindings {
    ranges: HashSet<BoundRange>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct BoundRange {
    base: u64,
    offset: usize,
    len: usize,
    address: u64,
}

struct ArgumentUsage {
    base: u64,
    written: HashSet<(usize, usize)>,
}

#[derive(Clone)]
enum PlannedRange {
    Fixed(BoundRange),
    Arena { offset: usize, len: usize },
}

#[derive(Clone)]
enum PlanCall {
    Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
    Bind {
        index: usize,
        range: PlannedRange,
    },
    ArgumentTable,
    Dispatch {
        threadgroups: MTLSize,
        threads_per_threadgroup: MTLSize,
    },
    Barrier,
}

struct PlanRecording {
    arena_base: u64,
    calls: Vec<PlanCall>,
}

struct PlanRecorderScope(bool);

impl PlanRecorderScope {
    fn enter(arena_base: u64) -> Self {
        PLAN_RECORDER.with(|recorder| {
            debug_assert!(recorder.borrow().is_none());
            *recorder.borrow_mut() = Some(PlanRecording {
                arena_base,
                calls: Vec::new(),
            });
        });
        Self(true)
    }

    fn finish(mut self) -> Vec<PlanCall> {
        self.0 = false;
        PLAN_RECORDER.with(|recorder| {
            recorder
                .borrow_mut()
                .take()
                .map_or_else(Vec::new, |recording| recording.calls)
        })
    }
}

impl Drop for PlanRecorderScope {
    fn drop(&mut self) {
        if self.0 {
            PLAN_RECORDER.with(|recorder| {
                recorder.borrow_mut().take();
            });
        }
    }
}

fn record_plan_call(call: PlanCall) {
    PLAN_RECORDER.with(|recorder| {
        if let Some(recording) = recorder.borrow_mut().as_mut() {
            recording.calls.push(call);
        }
    });
}

fn record_plan_binding(index: usize, range: BoundRange) {
    PLAN_RECORDER.with(|recorder| {
        let mut recorder = recorder.borrow_mut();
        let Some(recording) = recorder.as_mut() else {
            return;
        };
        let range = if range.base == recording.arena_base {
            PlannedRange::Arena {
                offset: range.offset,
                len: range.len,
            }
        } else {
            PlannedRange::Fixed(range)
        };
        recording.calls.push(PlanCall::Bind { index, range });
    });
}

impl ArgumentBindings {
    fn bind(
        &mut self,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        index: usize,
        buffer: &BufferBinding,
    ) {
        // SAFETY: Each caller uses an index within its argument-table descriptor and registers
        // the bound buffer in the command resource owner before submission.
        unsafe {
            table.setAddress_atIndex(buffer.address, index);
        }
        let range = BoundRange {
            base: buffer.raw.gpuAddress(),
            offset: buffer.offset,
            len: buffer.len,
            address: buffer.address,
        };
        record_plan_binding(index, range);
        self.ranges.insert(range);
    }

    fn bind_raw(
        &mut self,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        index: usize,
        buffer: &MetalBufferRef,
    ) {
        let address = buffer.gpuAddress();
        // SAFETY: Each caller uses an index within its argument-table descriptor and registers
        // the bound buffer in the command resource owner before submission.
        unsafe {
            table.setAddress_atIndex(address, index);
        }
        let range = BoundRange {
            base: address,
            offset: 0,
            len: buffer.length(),
            address,
        };
        record_plan_binding(index, range);
        self.ranges.insert(range);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    name: String,
    constants: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProgramPipelineKey {
    hash: ProgramHash,
    input_dtypes: Vec<u8>,
    output_dtypes: Vec<u8>,
    rank: u8,
    row: bool,
    resident: bool,
}

impl ProgramPipelineKey {
    fn new(program: &ValidatedProgram, signature: &KernelSignature, resident: bool) -> Self {
        let row = program.program().kind == ProgramKind::Row;
        Self {
            hash: program.content_hash(),
            input_dtypes: signature
                .input_dtypes()
                .iter()
                .copied()
                .map(dtype_key)
                .collect(),
            output_dtypes: signature
                .output_dtypes()
                .iter()
                .copied()
                .map(dtype_key)
                .collect(),
            rank: signature.rank(),
            row,
            resident: row && resident,
        }
    }
}

struct DispatchEncoding<'a> {
    dispatches: &'a [Dispatch],
    barriers: &'a [bool],
}

struct ProgramEncodingState<'a> {
    bindings: &'a mut ArgumentBindings,
    arguments: &'a mut ArgumentWriter,
    compile_fallbacks: &'a mut u64,
}

struct ProgramPipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    resident: bool,
    compile_fallback: bool,
}

pub(super) struct MetalProgram {
    rereading: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    resident: Option<LazyProgramPipeline>,
}

// SAFETY: Compiled pipeline states are immutable, and Metal permits encoding them from multiple
// host threads.
unsafe impl Send for MetalProgram {}
// SAFETY: Compiled pipeline states are immutable, and lazy initialization is synchronized by
// `OnceLock`.
unsafe impl Sync for MetalProgram {}

struct LazyProgramPipeline {
    key: ProgramPipelineKey,
    source: String,
    state: OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    compilation: Mutex<()>,
}

/// Opaque state retained by a prepared Metal scalar program.
pub struct MetalProgramHandle(pub(super) Arc<MetalProgram>);

impl MetalProgram {
    fn rereading(&self, compile_fallback: bool) -> ProgramPipeline {
        ProgramPipeline {
            state: self.rereading.clone(),
            resident: false,
            compile_fallback,
        }
    }

    fn select(&self, backend: &MetalBackend, width: u32) -> Result<ProgramPipeline, BackendError> {
        let Some(resident) = self
            .resident
            .as_ref()
            .filter(|_| width <= map_codegen::REGISTER_RESIDENT_WIDTH)
        else {
            return Ok(self.rereading(false));
        };
        let pipeline = if let Some(pipeline) = resident.state.get() {
            pipeline
        } else {
            let _compilation = resident
                .compilation
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            if let Some(pipeline) = resident.state.get() {
                pipeline
            } else {
                let compile_allowed = backend
                    .program_compile_tokens
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .consume(1);
                if !compile_allowed {
                    return Ok(self.rereading(true));
                }
                let pipeline = backend
                    .pipelines
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .compile_program_source(&resident.key, &resident.source)?;
                resident
                    .state
                    .set(pipeline)
                    .map_err(|_| BackendError::ExecutionFailed)?;
                resident.state.get().ok_or(BackendError::ExecutionFailed)?
            }
        };
        if pipeline.maxTotalThreadsPerThreadgroup()
            >= usize::try_from(width).map_err(|_| BackendError::ExecutionFailed)?
        {
            return Ok(ProgramPipeline {
                state: pipeline.clone(),
                resident: true,
                compile_fallback: false,
            });
        }
        Ok(self.rereading(false))
    }
}

struct CompletionState {
    feedback: Option<CommitResult>,
    event_signaled: bool,
    event_value: Option<u64>,
    committed: Option<Instant>,
    feedback_elapsed: Option<Duration>,
    event_elapsed: Option<Duration>,
    result: Option<Result<(), BackendError>>,
    resolving: bool,
}

#[cfg(test)]
static LAST_SUBMIT_NANOS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static LAST_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(super) fn last_submission_timing() -> (Duration, Duration) {
    (
        Duration::from_nanos(LAST_SUBMIT_NANOS.load(Ordering::Relaxed)),
        Duration::from_nanos(LAST_WAIT_NANOS.load(Ordering::Relaxed)),
    )
}

#[cfg(test)]
fn store_duration(target: &AtomicU64, duration: Duration) {
    target.store(
        u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

struct CommitFeedback(Retained<ProtocolObject<dyn MTL4CommitFeedback>>);

// SAFETY: Commit feedback is immutable after Metal invokes the handler, and ownership moves to
// the waiting thread before its properties are read or it is released.
unsafe impl Send for CommitFeedback {}

enum CommitResult {
    Feedback(CommitFeedback),
}

thread_local! {
    static IN_METAL_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static TEMPORARY_PROFILE: Cell<Option<ProfileCount>> = const { Cell::new(None) };
    static PLAN_RECORDER: RefCell<Option<PlanRecording>> = const { RefCell::new(None) };
}

struct TemporaryProfileScope {
    previous: Option<ProfileCount>,
    enabled: bool,
}

impl TemporaryProfileScope {
    fn enter(enabled: bool) -> Self {
        let previous =
            TEMPORARY_PROFILE.with(|profile| profile.replace(enabled.then(ProfileCount::default)));
        Self { previous, enabled }
    }

    fn finish(&mut self) -> ProfileCount {
        if !self.enabled {
            return ProfileCount::default();
        }
        self.enabled = false;
        TEMPORARY_PROFILE.with(|profile| {
            let current = profile.replace(self.previous);
            current.unwrap_or_default()
        })
    }
}

impl Drop for TemporaryProfileScope {
    fn drop(&mut self) {
        if self.enabled {
            TEMPORARY_PROFILE.with(|profile| profile.set(self.previous));
        }
    }
}

#[cfg(test)]
thread_local! {
    static FORCED_SDPA_KERNEL: Cell<Option<SdpaKernel>> = const { Cell::new(None) };
}

struct MetalCallbackScope(bool);

impl MetalCallbackScope {
    fn enter() -> Self {
        let previous = IN_METAL_CALLBACK.with(|active| active.replace(true));
        debug_assert!(!previous);
        Self(previous)
    }
}

impl Drop for MetalCallbackScope {
    fn drop(&mut self) {
        IN_METAL_CALLBACK.with(|active| active.set(self.0));
    }
}

fn assert_not_in_metal_callback() {
    IN_METAL_CALLBACK.with(|active| {
        #[cfg(test)]
        assert!(!active.get(), "Metal method called from a callback");
        #[cfg(not(test))]
        debug_assert!(!active.get(), "Metal method called from a callback");
    });
}

fn commit_result(feedback: &CommitResult) -> Result<(), BackendError> {
    match feedback {
        CommitResult::Feedback(feedback) => {
            assert_not_in_metal_callback();
            if feedback.0.error().is_some() {
                Err(BackendError::ExecutionFailed)
            } else {
                Ok(())
            }
        }
    }
}

pub(super) struct InFlightBuffer {
    pub(super) raw: MetalBufferRef,
    pool_resident: bool,
}

// SAFETY: Metal buffer resources support concurrent retain and release, and this wrapper never
// exposes CPU access to their contents.
unsafe impl Send for InFlightBuffer {}

// SAFETY: The wrapper only keeps a Metal buffer alive and provides no access to its contents.
unsafe impl Sync for InFlightBuffer {}

pub(super) struct InFlightEvent {
    pub(super) raw: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// SAFETY: Shared events are designed for cross-thread signaling and waiting, and the wrapper
// exposes only the thread-safe wait operation.
unsafe impl Send for InFlightEvent {}

// SAFETY: Concurrent waits do not mutate the retained shared event through Rust references.
unsafe impl Sync for InFlightEvent {}

pub(super) struct InFlightResidency {
    pub(super) _raw: Vec<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}

struct CommitResidency {
    sets: Vec<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    order_on_queue: bool,
}

// SAFETY: The residency set is committed before submission and remains immutable while shared
// across completion and waiting threads.
unsafe impl Send for InFlightResidency {}

// SAFETY: The wrapper exposes no operations on the retained residency set.
unsafe impl Sync for InFlightResidency {}

struct ReusableSubmissionObjects {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    argument_buffer: ArgumentBuffer,
}

// SAFETY: Ownership moves between the encoding thread and the completion tracker, and the
// allocator and argument table are not reused until GPU completion.
unsafe impl Send for ReusableSubmissionObjects {}

struct SubmissionObjects {
    reusable: Option<ReusableSubmissionObjects>,
    owner: Weak<InFlightTracker>,
    programs: Vec<Arc<PreparedProgram>>,
}

impl SubmissionObjects {
    fn allocator(&self) -> Result<&ProtocolObject<dyn MTL4CommandAllocator>, BackendError> {
        self.reusable
            .as_ref()
            .map(|objects| objects.allocator.as_ref())
            .ok_or(BackendError::ExecutionFailed)
    }

    fn argument_table(&self) -> Result<&ProtocolObject<dyn MTL4ArgumentTable>, BackendError> {
        self.reusable
            .as_ref()
            .map(|objects| objects.argument_table.as_ref())
            .ok_or(BackendError::ExecutionFailed)
    }

    fn argument_buffer(&self) -> Result<&ArgumentBuffer, BackendError> {
        self.reusable
            .as_ref()
            .map(|objects| &objects.argument_buffer)
            .ok_or(BackendError::ExecutionFailed)
    }

    fn take(&mut self) -> Result<ReusableSubmissionObjects, BackendError> {
        self.reusable.take().ok_or(BackendError::ExecutionFailed)
    }
}

impl Drop for SubmissionObjects {
    fn drop(&mut self) {
        if let (Some(objects), Some(owner)) = (self.reusable.take(), self.owner.upgrade()) {
            owner.return_dirty(objects);
        }
    }
}

fn reset_allocator(allocator: &ProtocolObject<dyn MTL4CommandAllocator>) {
    assert_not_in_metal_callback();
    allocator.reset();
}

struct GpuTimestamps {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    frequency: u64,
    entry_size: usize,
    count: usize,
}

// SAFETY: Counter heaps remain immutable on the CPU while the retained submission is in flight,
// and timestamp resolution happens only after the shared event establishes GPU completion.
unsafe impl Send for GpuTimestamps {}

// SAFETY: Shared access only resolves completed timestamp data and does not mutate the heap.
unsafe impl Sync for GpuTimestamps {}

impl GpuTimestamps {
    fn elapsed(&self) -> Option<Duration> {
        self.elapsed_pairs(&[(0, 1)])?.pop()
    }

    fn elapsed_pairs(&self, pairs: &[(usize, usize)]) -> Option<Vec<Duration>> {
        // SAFETY: The requested range covers the heap and resolution follows GPU completion.
        let data = unsafe { self.heap.resolveCounterRange(NSRange::new(0, self.count)) }?;
        let bytes = data.to_vec();
        pairs
            .iter()
            .map(|&(start, end)| {
                let read = |index: usize| {
                    let offset = self.entry_size.checked_mul(index)?;
                    Some(u64::from_ne_bytes(
                        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
                    ))
                };
                let ticks = read(end)?.checked_sub(read(start)?)?;
                let seconds = ticks / self.frequency;
                let nanos = u128::from(ticks % self.frequency).checked_mul(1_000_000_000)?
                    / u128::from(self.frequency);
                Some(Duration::new(seconds, u32::try_from(nanos).ok()?))
            })
            .collect()
    }
}

struct CommandResources {
    buffers: Vec<InFlightBuffer>,
    error_flags: Vec<(usize, usize)>,
}

pub(super) struct PreparedMetalGraph {
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: HashSet<BufferId>,
    encoding: Option<MetalEncodingPlan>,
}

#[derive(Clone)]
struct StaticEncodingPlan {
    calls: Vec<PlanCall>,
}

struct MetalEncodingPlan {
    dispatches: Vec<Option<StaticEncodingPlan>>,
    arena: Vec<u8>,
    argument_offset: usize,
}

fn dedupe_plan_calls(plans: &mut [Option<StaticEncodingPlan>]) {
    let mut pipeline = None::<Retained<ProtocolObject<dyn MTLComputePipelineState>>>;
    let mut argument_table_set = false;
    for plan in plans {
        let Some(plan) = plan else {
            pipeline = None;
            argument_table_set = false;
            continue;
        };
        plan.calls.retain(|call| match call {
            PlanCall::Pipeline(next) => {
                let repeated = pipeline
                    .as_ref()
                    .is_some_and(|current| std::ptr::eq(&raw const **current, &raw const **next));
                pipeline = Some(next.clone());
                !repeated
            }
            PlanCall::ArgumentTable => {
                let repeated = argument_table_set;
                argument_table_set = true;
                !repeated
            }
            PlanCall::Bind { .. } | PlanCall::Dispatch { .. } | PlanCall::Barrier => true,
        });
    }
}

// SAFETY: The committed residency set remains immutable, and its allocations are retained by the
// host graph for at least as long as this state.
unsafe impl Send for PreparedMetalGraph {}

// SAFETY: Shared access only submits the immutable committed residency set to command buffers.
unsafe impl Sync for PreparedMetalGraph {}

pub(super) struct Completion {
    state: Mutex<CompletionState>,
    ready: Condvar,
    event: InFlightEvent,
    resources: CommandResources,
    _residency: InFlightResidency,
    timestamps: Option<GpuTimestamps>,
    tracker: Weak<InFlightTracker>,
    dependencies: Mutex<Vec<Arc<Completion>>>,
    commit: CommitRetention,
}

type FeedbackHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTL4CommitFeedback>>)>;

struct CommitRetention {
    _handler: FeedbackHandler,
    options: Retained<MTL4CommitOptions>,
}

// SAFETY: The commit options are immutable after registration, and the block captures only a
// thread-safe weak completion reference. Objective-C blocks and objects may be retained and
// released from Metal callback queues.
unsafe impl Send for CommitRetention {}

// SAFETY: Shared access can only retain the immutable commit objects; callback state is protected
// by the completion mutex.
unsafe impl Sync for CommitRetention {}

impl Completion {
    fn new(
        resources: CommandResources,
        event: InFlightEvent,
        residency: InFlightResidency,
        timestamps: Option<GpuTimestamps>,
        tracker: Weak<InFlightTracker>,
        dependencies: Vec<Arc<Completion>>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|completion: &Weak<Self>| {
            let callback_completion = completion.clone();
            let handler: FeedbackHandler = RcBlock::new(
                move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    let _scope = MetalCallbackScope::enter();
                    if let Some(completion) = callback_completion.upgrade() {
                        // SAFETY: Metal supplies a live feedback object for the callback. Retaining
                        // it keeps the immutable result alive for inspection on the waiting thread.
                        if let Some(feedback) = unsafe { Retained::retain(feedback.as_ptr()) } {
                            completion.finish(CommitResult::Feedback(CommitFeedback(feedback)));
                        }
                    }
                },
            );
            let options = MTL4CommitOptions::new();
            // SAFETY: `handler` is a live heap block. This completion owns both the block and the
            // options, and the in-flight tracker retains the completion until the queue event and
            // feedback callback have both completed.
            unsafe {
                options.addFeedbackHandler(RcBlock::as_ptr(&handler));
            }
            Self {
                state: Mutex::new(CompletionState {
                    feedback: None,
                    event_signaled: false,
                    event_value: None,
                    committed: None,
                    feedback_elapsed: None,
                    event_elapsed: None,
                    result: None,
                    resolving: false,
                }),
                ready: Condvar::new(),
                event,
                resources,
                _residency: residency,
                timestamps,
                tracker,
                dependencies: Mutex::new(dependencies),
                commit: CommitRetention {
                    _handler: handler,
                    options,
                },
            }
        })
    }

    fn finish(self: &Arc<Self>, result: CommitResult) {
        let complete = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.feedback = Some(result);
            state.feedback_elapsed = state.committed.map(|started| started.elapsed());
            state.event_signaled
        };
        if complete {
            self.reap();
        }
        self.ready.notify_all();
    }

    pub(super) fn wait(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(result) = state.result {
                return result;
            }
            if state.resolving {
                let remaining = timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Err(BackendError::Timeout);
                }
                let (next, wait) = self
                    .ready
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state = next;
                if wait.timed_out() && state.result.is_none() {
                    return Err(BackendError::Timeout);
                }
                continue;
            }
            if state.event_signaled
                && let Some(feedback) = &state.feedback
            {
                let mut result = commit_result(feedback);
                state.resolving = true;
                drop(state);
                if let Some(tracker) = self.tracker.upgrade() {
                    tracker.drain_done();
                }
                if result.is_ok() {
                    result = self.check_error_flags();
                }
                let dependencies = std::mem::take(
                    &mut *self
                        .dependencies
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                if result.is_ok() {
                    for dependency in dependencies {
                        let remaining = timeout.saturating_sub(started.elapsed());
                        if remaining.is_zero() {
                            result = Err(BackendError::Timeout);
                            break;
                        }
                        if let Err(error) = dependency.wait(remaining) {
                            result = Err(error);
                            break;
                        }
                    }
                }
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let result = *state.result.get_or_insert(result);
                state.resolving = false;
                self.ready.notify_all();
                return result;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                #[cfg(test)]
                self.log_timeout(&state);
                return Err(BackendError::Timeout);
            }
            let (next, wait) = self
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && (!state.event_signaled || state.feedback.is_none()) {
                #[cfg(test)]
                self.log_timeout(&state);
                return Err(BackendError::Timeout);
            }
        }
    }

    fn signal_event(self: &Arc<Self>) {
        let complete = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.event_signaled = true;
            state.event_elapsed = state.committed.map(|started| started.elapsed());
            state.feedback.is_some()
        };
        if complete {
            self.reap();
        }
        self.ready.notify_all();
    }

    fn mark_committed(&self, event_value: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.committed = Some(Instant::now());
        state.event_value = Some(event_value);
    }

    pub(super) fn event_value(&self) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .event_value
    }

    pub(super) fn result(&self) -> Option<Result<(), BackendError>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .result
    }

    #[cfg(test)]
    fn log_timeout(&self, state: &CompletionState) {
        eprintln!(
            "Metal wait timed out: submission={:?} expected_event={:?} signaled_event={} listener_notified={} feedback_received={}",
            state.event_value,
            state.event_value,
            self.event.raw.signaledValue(),
            state.event_signaled,
            state.feedback.is_some(),
        );
    }

    fn reap(self: &Arc<Self>) {
        if let Some(tracker) = self.tracker.upgrade() {
            tracker.remove(self);
        }
    }

    fn completion_timing(&self) -> Option<(Duration, Duration)> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Some((state.feedback_elapsed?, state.event_elapsed?))
    }

    fn check_error_flags(&self) -> Result<(), BackendError> {
        for &(flag, offset) in &self.resources.error_flags {
            // SAFETY: The queue event has signaled GPU completion, and each indexed retained
            // shared buffer contains two aligned u32 values initialized by the host.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.resources.buffers[flag]
                        .raw
                        .contents()
                        .cast::<u8>()
                        .as_ptr()
                        .add(offset),
                    8,
                )
            };
            let has_error = u32::from_ne_bytes(
                bytes[..4]
                    .try_into()
                    .map_err(|_| BackendError::ExecutionFailed)?,
            );
            let index = u32::from_ne_bytes(
                bytes[4..]
                    .try_into()
                    .map_err(|_| BackendError::ExecutionFailed)?,
            );
            if has_error != 0 {
                return if has_error == 1 {
                    Err(BackendError::IndexOutOfRange { index })
                } else {
                    Err(BackendError::InvalidInput)
                };
            }
        }
        Ok(())
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.timestamps.as_ref()?.elapsed()
    }

    fn dispatch_times(&self, operations: &[Op]) -> Option<Vec<DispatchProfile>> {
        let pairs = (0..operations.len())
            .map(|index| (2 + index * 2, 3 + index * 2))
            .collect::<Vec<_>>();
        Some(
            operations
                .iter()
                .copied()
                .zip(self.timestamps.as_ref()?.elapsed_pairs(&pairs)?)
                .map(|(op, gpu_time)| DispatchProfile { op, gpu_time })
                .collect(),
        )
    }
}

type NotificationHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;

struct ListenerRegistration {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    listener: Retained<MTLSharedEventListener>,
    notification: NotificationHandler,
}

impl ListenerRegistration {
    fn register(&self, value: u64) {
        assert_not_in_metal_callback();
        // SAFETY: The in-flight tracker owns the listener and heap block before registration, and
        // Metal invokes the notification even when the shared event has already reached `value`.
        unsafe {
            self.event.notifyListener_atValue_block(
                &self.listener,
                value,
                RcBlock::as_ptr(&self.notification),
            );
        }
    }
}

struct InFlightCompletion {
    completion: Arc<Completion>,
    objects: ReusableSubmissionObjects,
    _programs: Vec<Arc<PreparedProgram>>,
    _listener: Retained<MTLSharedEventListener>,
    _notification: NotificationHandler,
}

// SAFETY: The listener is thread-safe, and the immutable notification block captures only
// thread-safe `Arc` values. Objective-C blocks may be retained and released on the listener queue.
unsafe impl Send for InFlightCompletion {}

// SAFETY: Shared access only retains the listener, block, and completion; their mutable state is
// protected by mutexes.
unsafe impl Sync for InFlightCompletion {}

pub(super) struct InFlightTracker {
    completions: Mutex<Vec<InFlightCompletion>>,
    done: Mutex<Vec<InFlightCompletion>>,
    pool: Mutex<Vec<ReusableSubmissionObjects>>,
}

impl InFlightTracker {
    pub(super) const fn new() -> Self {
        Self {
            completions: Mutex::new(Vec::new()),
            done: Mutex::new(Vec::new()),
            pool: Mutex::new(Vec::new()),
        }
    }

    fn checkout(
        self: &Arc<Self>,
        device: &ProtocolObject<dyn MTLDevice>,
        argument_capacity: usize,
    ) -> Result<SubmissionObjects, BackendError> {
        use objc2_metal::MTL4ArgumentTableDescriptor;

        self.drain_done();
        let reusable = self
            .pool
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .pop();
        let mut reusable = if let Some(objects) = reusable {
            reset_allocator(&objects.allocator);
            objects
        } else {
            let allocator = device
                .newCommandAllocator()
                .ok_or(BackendError::ExecutionFailed)?;
            let descriptor = MTL4ArgumentTableDescriptor::new();
            descriptor.setMaxBufferBindCount(24);
            let argument_table = device
                .newArgumentTableWithDescriptor_error(&descriptor)
                .map_err(|_| BackendError::ExecutionFailed)?;
            let raw = device
                .newBufferWithLength_options(
                    argument_capacity,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or(BackendError::AllocationFailed)?;
            ReusableSubmissionObjects {
                allocator,
                argument_table,
                argument_buffer: ArgumentBuffer {
                    raw,
                    capacity: argument_capacity,
                },
            }
        };
        if reusable.argument_buffer.capacity != argument_capacity {
            let raw = device
                .newBufferWithLength_options(
                    argument_capacity,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or(BackendError::AllocationFailed)?;
            reusable.argument_buffer = ArgumentBuffer {
                raw,
                capacity: argument_capacity,
            };
        }
        Ok(SubmissionObjects {
            reusable: Some(reusable),
            owner: Arc::downgrade(self),
            programs: Vec::new(),
        })
    }

    fn track(
        self: &Arc<Self>,
        completion: &Arc<Completion>,
        listener: &Retained<MTLSharedEventListener>,
        objects: &mut SubmissionObjects,
    ) -> Result<ListenerRegistration, BackendError> {
        let pending = Arc::clone(completion);
        let owner = Arc::clone(self);
        let notification: NotificationHandler = RcBlock::new(move |_event, _value| {
            let _scope = MetalCallbackScope::enter();
            let active_completion = Arc::clone(&pending);
            let active_owner = Arc::clone(&owner);
            active_completion.signal_event();
            drop(active_owner);
        });
        self.completions
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .push(InFlightCompletion {
                completion: Arc::clone(completion),
                objects: objects.take()?,
                _programs: std::mem::take(&mut objects.programs),
                _listener: listener.clone(),
                _notification: notification.clone(),
            });
        Ok(ListenerRegistration {
            event: completion.event.raw.clone(),
            listener: listener.clone(),
            notification,
        })
    }

    fn remove(&self, completion: &Arc<Completion>) {
        let Some(completed) = self.take(completion) else {
            return;
        };
        self.done
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(completed);
    }

    fn take(&self, completion: &Arc<Completion>) -> Option<InFlightCompletion> {
        let mut completions = self
            .completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = completions
            .iter()
            .position(|candidate| Arc::ptr_eq(&candidate.completion, completion))?;
        Some(completions.swap_remove(index))
    }

    fn cancel(&self, completion: &Arc<Completion>) {
        if let Some(completed) = self.take(completion) {
            self.return_dirty(completed.objects);
        }
    }

    pub(super) fn drain_done(&self) {
        assert_not_in_metal_callback();
        let done = std::mem::take(
            &mut *self
                .done
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for completed in done {
            pool.push(completed.objects);
        }
    }

    fn return_dirty(&self, objects: ReusableSubmissionObjects) {
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(objects);
    }

    pub(super) fn drain(&self, timeout: Duration) {
        let completions = self
            .completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|entry| Arc::clone(&entry.completion))
            .collect::<Vec<_>>();
        for completion in completions {
            let _ = completion.wait(timeout);
        }
        self.drain_done();
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    #[cfg(test)]
    pub(super) fn pooled_len(&self) -> usize {
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

/// Completion state for a Metal command buffer.
pub struct MetalSubmission {
    completion: Arc<Completion>,
    timeout: Duration,
    profile: Option<Mutex<SubmissionProfile>>,
    dispatch_operations: Vec<Op>,
}

impl Submission for MetalSubmission {
    fn wait(&self) -> Result<(), BackendError> {
        self.wait_profiled(self.timeout)
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<(), BackendError> {
        self.wait_profiled(timeout)
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.completion.gpu_time()
    }

    fn profile(&self) -> Option<SubmissionProfile> {
        let mut profile = self.profile.as_ref()?.lock().ok()?.clone();
        profile.per_dispatch = self.completion.dispatch_times(&self.dispatch_operations)?;
        Some(profile)
    }
}

impl MetalSubmission {
    fn wait_profiled(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        let result = self.completion.wait(timeout);
        #[cfg(test)]
        store_duration(&LAST_WAIT_NANOS, started.elapsed());
        if result.is_ok()
            && let Some(profile) = &self.profile
            && let Ok(mut profile) = profile.lock()
        {
            profile.wait = started.elapsed();
            profile.gpu_time = self.completion.gpu_time().unwrap_or_default();
        }
        result
    }

    /// Returns feedback and shared-event callback latency from queue commit after a successful wait.
    #[must_use]
    pub fn completion_timing(&self) -> Option<(Duration, Duration)> {
        self.completion.completion_timing()
    }
}

impl MetalBackend {
    pub(super) fn prepare_scalar_program(
        &self,
        program: &ValidatedProgram,
        signature: &KernelSignature,
    ) -> Result<MetalProgramHandle, BackendError> {
        let cache_key = (program.content_hash(), signature.clone());
        let mut prepared = self
            .prepared_programs
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        prepared.retain(|_, program| program.strong_count() > 0);
        if let Some(existing) = prepared.get(&cache_key).and_then(Weak::upgrade) {
            return Ok(MetalProgramHandle(existing));
        }

        self.program_compile_tokens
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .consume(1)
            .then_some(())
            .ok_or(BackendError::QuotaExceeded)?;
        let rereading_key = ProgramPipelineKey::new(program, signature, false);
        let rereading_source = map_codegen::generate(program, signature, false);
        let rereading = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .compile_program_source(&rereading_key, &rereading_source)?;
        // The bound row width is unavailable at preparation. The rereading pipeline works for
        // every width; the resident candidate is compiled only when a dispatch can select it.
        let resident = (program.program().kind == ProgramKind::Row).then(|| {
            let key = ProgramPipelineKey::new(program, signature, true);
            LazyProgramPipeline {
                source: map_codegen::generate(program, signature, true),
                key,
                state: OnceLock::new(),
                compilation: Mutex::new(()),
            }
        });
        let program = Arc::new(MetalProgram {
            rereading,
            resident,
        });
        prepared.insert(cache_key, Arc::downgrade(&program));
        Ok(MetalProgramHandle(program))
    }

    pub(super) fn submit_commands(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
        self.submit_commands_inner::<false>(commands, None)
    }

    pub(super) fn submit_commands_profiled(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
        self.submit_commands_inner::<true>(commands, None)
    }

    pub(super) fn prepare_metal_graph(
        &self,
        graph: &GraphTemplate,
    ) -> Result<PreparedMetalGraph, BackendError> {
        let encoding = (self.graph_replay == crate::storage::MetalGraphReplay::Tier2)
            .then(|| self.prepare_encoding_plan(graph))
            .transpose()?;
        let residency = self
            .device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let tensors = graph.retained_tensors();
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut retained = HashSet::with_capacity(tensors.len());
        for tensor in &tensors {
            if retained.insert(tensor.buffer()) {
                let buffer: &ProtocolObject<dyn MTLBuffer> = &buffers.get(tensor)?.raw;
                let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
                residency.addAllocation(allocation);
            }
        }
        residency.commit();
        Ok(PreparedMetalGraph {
            residency,
            buffers: retained,
            encoding,
        })
    }

    fn prepare_encoding_plan(
        &self,
        graph: &GraphTemplate,
    ) -> Result<MetalEncodingPlan, BackendError> {
        let static_dispatches = graph
            .static_dispatches()
            .filter(|(_, dispatch)| reusable_dispatch(dispatch))
            .collect::<Vec<_>>();
        let mut plans = vec![None; graph.len()];
        if static_dispatches.is_empty() {
            return Ok(MetalEncodingPlan {
                dispatches: plans,
                arena: Vec::new(),
                argument_offset: 0,
            });
        }
        let dispatches = static_dispatches
            .iter()
            .map(|(_, dispatch)| (*dispatch).clone())
            .collect::<Vec<_>>();
        let capacity = self.argument_capacity(&dispatches)?;
        let objects = self.in_flight.checkout(&self.device, capacity)?;
        let command_buffer = self.begin_command_buffer(objects.allocator()?)?;
        let mut arguments = ArgumentWriter::new(objects.argument_buffer()?);
        for (index, dispatch) in static_dispatches {
            let recorder = PlanRecorderScope::enter(arguments.raw.gpuAddress());
            let encoded = self.encode_dispatches(
                &command_buffer,
                objects.argument_table()?,
                &DispatchEncoding {
                    dispatches: std::slice::from_ref(dispatch),
                    barriers: &[false],
                },
                None,
                &mut arguments,
                None,
            )?;
            let calls = recorder.finish();
            if !encoded.error_flags.is_empty()
                || encoded
                    .temporaries
                    .iter()
                    .any(|buffer| buffer.raw.gpuAddress() != arguments.raw.gpuAddress())
            {
                return Err(BackendError::ExecutionFailed);
            }
            plans[index] = Some(StaticEncodingPlan { calls });
        }
        dedupe_plan_calls(&mut plans);
        command_buffer.endCommandBuffer();
        let mut arena = vec![0_u8; arguments.offset];
        // SAFETY: The arena length is the initialized prefix of the live shared argument buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                arguments.raw.contents().cast::<u8>().as_ptr(),
                arena.as_mut_ptr(),
                arena.len(),
            );
        }
        Ok(MetalEncodingPlan {
            dispatches: plans,
            argument_offset: arena.len(),
            arena,
        })
    }

    pub(super) fn replay_graph(
        &self,
        graph: &PreparedGraph,
        values: Vec<u32>,
        profile: bool,
    ) -> Result<MetalSubmission, BackendError> {
        let state = graph
            .backend_state::<PreparedMetalGraph>()
            .ok_or(BackendError::InvalidInput)?;
        let values = graph
            .values(values)
            .map_err(|_| BackendError::InvalidInput)?;
        let commands = graph
            .template()
            .instantiate(&values)
            .map_err(|_| BackendError::InvalidInput)?;
        if profile {
            self.submit_commands_inner::<true>(commands, Some(state))
        } else {
            self.submit_commands_inner::<false>(commands, Some(state))
        }
    }

    #[allow(clippy::too_many_lines)]
    fn submit_commands_inner<const PROFILE: bool>(
        &self,
        commands: CommandList,
        graph: Option<&PreparedMetalGraph>,
    ) -> Result<MetalSubmission, BackendError> {
        #[cfg(test)]
        let submit_started = Instant::now();
        self.in_flight.drain_done();
        let validation_started = PROFILE.then(Instant::now);
        let program_recording = commands.program_recording();
        let retained_tensors_validated = commands.retained_tensors_validated();
        let barriers = required_barriers(&commands);
        let dispatches = commands.into_dispatches();
        let prepared_programs = dispatches
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                graph
                    .and_then(|prepared| prepared.encoding.as_ref())
                    .and_then(|encoding| encoding.dispatches.get(*index))
                    .is_none_or(Option::is_none)
            })
            .filter_map(|(_, dispatch)| dispatch.prepared_program())
            .cloned()
            .collect::<Vec<_>>();
        if dispatches.iter().enumerate().any(|(index, dispatch)| {
            !graph
                .and_then(|prepared| prepared.encoding.as_ref())
                .and_then(|encoding| encoding.dispatches.get(index))
                .is_some_and(Option::is_some)
                && !supported_dispatch(dispatch)
        }) {
            return Err(BackendError::UnsupportedOperation);
        }
        let accesses = dispatches
            .iter()
            .flat_map(|dispatch| {
                dispatch
                    .inputs()
                    .iter()
                    .cloned()
                    .map(|tensor| (tensor, false))
                    .chain(
                        dispatch
                            .outputs()
                            .iter()
                            .cloned()
                            .map(|tensor| (tensor, true)),
                    )
            })
            .collect::<Vec<_>>();
        let tensors = accesses
            .iter()
            .map(|(tensor, _)| tensor.clone())
            .collect::<Vec<_>>();
        if !retained_tensors_validated {
            for tensor in &tensors {
                self.validate(tensor)?;
            }
        }
        let mut profile = PROFILE.then(|| SubmissionProfile {
            program_recording,
            validation: validation_started.map_or(Duration::ZERO, |started| started.elapsed()),
            dispatches: u64::try_from(dispatches.len()).unwrap_or(u64::MAX),
            barriers: u64::try_from(barriers.iter().filter(|&&barrier| barrier).count())
                .unwrap_or(u64::MAX),
            ..SubmissionProfile::default()
        });
        let argument_capacity = self.argument_capacity_with_plan(
            &dispatches,
            graph.and_then(|prepared| prepared.encoding.as_ref()),
        )?;
        let mut objects = self.in_flight.checkout(&self.device, argument_capacity)?;
        let command_buffer = self.begin_command_buffer(objects.allocator()?)?;
        let timestamp_count = if PROFILE {
            dispatches
                .len()
                .checked_mul(2)
                .and_then(|count| count.checked_add(2))
                .ok_or(BackendError::ExecutionFailed)?
        } else {
            2
        };
        let timestamps = self.make_timestamps(timestamp_count)?;
        // SAFETY: `make_timestamps` always reserves at least two entries, so indices 0 and 1 are
        // in range.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 0);
        }
        let encoding_started = PROFILE.then(Instant::now);
        let mut temporary_profile = TemporaryProfileScope::enter(PROFILE);
        let mut argument_writer = ArgumentWriter::new(objects.argument_buffer()?);
        if let Some(encoding) = graph.and_then(|graph| graph.encoding.as_ref()) {
            argument_writer.copy_image(&encoding.arena)?;
        }
        let encoded = self.encode_dispatches(
            &command_buffer,
            objects.argument_table()?,
            &DispatchEncoding {
                dispatches: &dispatches,
                barriers: &barriers,
            },
            PROFILE.then_some(&timestamps),
            &mut argument_writer,
            graph.and_then(|prepared| {
                prepared
                    .encoding
                    .as_ref()
                    .map(|encoding| encoding.dispatches.as_slice())
            }),
        )?;
        argument_writer.validate_capacity(argument_capacity, dispatches.is_empty())?;
        if let Some(profile) = &mut profile {
            profile.metadata_buffers = temporary_profile.finish();
            profile.program_encoding = encoded.program_encoding;
            profile.program_compile_fallbacks = encoded.program_compile_fallbacks;
            let encoding = encoding_started.map_or(Duration::ZERO, |started| started.elapsed());
            profile.encoding = encoding.saturating_sub(profile.metadata_buffers.time);
        }
        let resources =
            self.command_resources(&tensors, encoded, graph.map(|prepared| &prepared.buffers))?;
        objects.programs = prepared_programs;
        let residency_started = PROFILE.then(Instant::now);
        let residency = self.make_resident(&command_buffer, &resources)?;
        if let Some(graph) = graph {
            command_buffer.useResidencySet(&graph.residency);
        }
        if let Some(profile) = &mut profile {
            profile.residency =
                residency_started.map_or(Duration::ZERO, |started| started.elapsed());
        }
        // SAFETY: `make_timestamps` always reserves at least two entries, so indices 0 and 1 are
        // in range.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 1);
        }
        command_buffer.endCommandBuffer();
        let operations = dispatches.iter().map(Dispatch::op).collect::<Vec<_>>();
        let commit_started = PROFILE.then(Instant::now);
        let mut submission = self.commit(
            &command_buffer,
            &accesses,
            resources,
            CommitResidency {
                sets: graph.map_or_else(
                    || vec![residency.clone()],
                    |graph| vec![residency.clone(), graph.residency.clone()],
                ),
                order_on_queue: graph.is_some(),
            },
            timestamps,
            &mut objects,
        )?;
        if let Some(mut profile) = profile {
            profile.commit = commit_started.map_or(Duration::ZERO, |started| started.elapsed());
            submission.profile = Some(Mutex::new(profile));
            submission.dispatch_operations = operations;
        }
        #[cfg(test)]
        store_duration(&LAST_SUBMIT_NANOS, submit_started.elapsed());
        Ok(submission)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        plan: &DispatchEncoding<'_>,
        timestamps: Option<&GpuTimestamps>,
        arguments: &mut ArgumentWriter,
        prepared: Option<&[Option<StaticEncodingPlan>]>,
    ) -> Result<EncodedDispatches, BackendError> {
        use objc2_metal::MTL4CommandEncoder;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(BackendError::ExecutionFailed)?;
        let mut temporaries = Vec::with_capacity(plan.dispatches.len().saturating_mul(3));
        let mut error_flags = Vec::new();
        let mut bindings = ArgumentBindings::default();
        let mut program_encoding = ProfileCount::default();
        let mut program_compile_fallbacks = 0_u64;
        for (index, (dispatch, &barrier)) in plan.dispatches.iter().zip(plan.barriers).enumerate() {
            if index > 0 {
                write_dispatch_timestamp(&encoder, timestamps, 1 + index * 2);
            }
            if barrier {
                encode_dispatch_barrier(&encoder);
            }
            write_dispatch_timestamp(&encoder, timestamps, 2 + index * 2);
            if let Some(plan) = prepared
                .and_then(|plans| plans.get(index))
                .and_then(Option::as_ref)
            {
                Self::encode_plan(&encoder, table, plan, &mut bindings, arguments)?;
                continue;
            }
            self.encode_dispatch(
                &encoder,
                table,
                dispatch,
                timestamps.is_some(),
                &mut temporaries,
                &mut error_flags,
                &mut bindings,
                arguments,
                &mut program_encoding,
                &mut program_compile_fallbacks,
            )?;
        }
        if !plan.dispatches.is_empty() {
            write_dispatch_timestamp(&encoder, timestamps, 1 + plan.dispatches.len() * 2);
        }
        encoder.endEncoding();
        let argument_usage = (arguments.offset > 0).then(|| ArgumentUsage {
            base: arguments.raw.gpuAddress(),
            written: arguments.written.iter().copied().collect(),
        });
        if argument_usage.is_some() {
            temporaries.push(BufferBinding::whole(arguments.raw.clone()));
        }
        Ok(EncodedDispatches {
            temporaries,
            error_flags,
            bindings,
            arguments: argument_usage,
            program_encoding,
            program_compile_fallbacks,
        })
    }

    fn encode_plan(
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        plan: &StaticEncodingPlan,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<(), BackendError> {
        use objc2_metal::MTL4ComputeCommandEncoder;

        for call in &plan.calls {
            match call {
                PlanCall::Pipeline(pipeline) => encoder.setComputePipelineState(pipeline),
                PlanCall::Bind { index, range } => {
                    let range = match range {
                        PlannedRange::Fixed(range) => *range,
                        PlannedRange::Arena { offset, len } => {
                            offset
                                .checked_add(*len)
                                .filter(|&end| end <= arguments.offset)
                                .ok_or(BackendError::ExecutionFailed)?;
                            let address = arguments
                                .raw
                                .gpuAddress()
                                .checked_add(
                                    u64::try_from(*offset)
                                        .map_err(|_| BackendError::ExecutionFailed)?,
                                )
                                .ok_or(BackendError::ExecutionFailed)?;
                            arguments.written.push((*offset, *len));
                            BoundRange {
                                base: arguments.raw.gpuAddress(),
                                offset: *offset,
                                len: *len,
                                address,
                            }
                        }
                    };
                    // SAFETY: Prepared binding indices and ranges were captured from a successful
                    // encoding, and arena addresses are rebased into the current live allocation.
                    unsafe {
                        table.setAddress_atIndex(range.address, *index);
                    }
                    bindings.ranges.insert(range);
                }
                PlanCall::ArgumentTable => encoder.setArgumentTable(Some(table)),
                PlanCall::Dispatch {
                    threadgroups,
                    threads_per_threadgroup,
                } => encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    *threadgroups,
                    *threads_per_threadgroup,
                ),
                PlanCall::Barrier => encode_dispatch_barrier(encoder),
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_dispatch(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        profile: bool,
        temporaries: &mut Vec<BufferBinding>,
        error_flags: &mut Vec<BufferBinding>,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
        program_encoding: &mut ProfileCount,
        program_compile_fallbacks: &mut u64,
    ) -> Result<(), BackendError> {
        if matches!(dispatch.op(), Op::Program(_)) {
            let started = profile.then(Instant::now);
            let mut state = ProgramEncodingState {
                bindings,
                arguments,
                compile_fallbacks: program_compile_fallbacks,
            };
            temporaries.extend(self.encode_program(encoder, table, dispatch, &mut state)?);
            if let Some(started) = started {
                program_encoding.count = program_encoding.count.saturating_add(1);
                program_encoding.time = program_encoding.time.saturating_add(started.elapsed());
            }
            return Ok(());
        }
        let buffers = match dispatch.op() {
            Op::RmsNorm { eps } => {
                self.encode_rms_norm(encoder, table, dispatch, eps, bindings, arguments)?
            }
            Op::Softmax => self.encode_softmax(encoder, table, dispatch, bindings, arguments)?,
            Op::Argmax => self.encode_argmax(encoder, table, dispatch, bindings, arguments)?,
            Op::Sample { position } => {
                let (buffers, flag) =
                    self.encode_sample(encoder, table, dispatch, position, bindings, arguments)?;
                error_flags.push(flag);
                buffers
            }
            Op::Rope { theta } => {
                self.encode_rope(encoder, table, dispatch, theta, bindings, arguments)?
            }
            Op::Embed => {
                let (buffers, flag) =
                    self.encode_embed(encoder, table, dispatch, bindings, arguments)?;
                error_flags.push(flag);
                buffers
            }
            Op::Copy => self.encode_copy(encoder, table, dispatch, bindings, arguments)?,
            Op::Matmul => self.encode_matmul(encoder, table, dispatch, bindings, arguments)?,
            Op::Sdpa { .. } => {
                self.encode_sdpa_dispatch(encoder, table, dispatch, bindings, arguments)?
            }
            Op::Add | Op::SiluMul => {
                let kernel = match dispatch.op() {
                    Op::Add
                        if dispatch
                            .inputs()
                            .iter()
                            .chain(std::iter::once(dispatch.output()))
                            .all(|tensor| tensor.layout().is_contiguous()) =>
                    {
                        "add_contiguous"
                    }
                    Op::Add => "add_strided",
                    Op::SiluMul => "silu_mul",
                    _ => return Err(BackendError::InvalidInput),
                };
                self.encode_elementwise(encoder, table, dispatch, kernel, bindings, arguments)?
            }
            Op::Program(_) => return Err(BackendError::InvalidInput),
        };
        temporaries.extend(buffers);
        Ok(())
    }

    fn encode_copy(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let output = self.encoder_tensor(dispatch.output())?;
        self.encode_copy_tensors(encoder, table, &input, &output, bindings, arguments)
    }

    fn encode_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [left_tensor, right_tensor] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output_tensor = dispatch.output();
        let dtype = output_tensor.layout().dtype();
        let mut temporaries = Vec::new();
        let left = self.prepare_matmul_input(
            encoder,
            table,
            left_tensor,
            bindings,
            &mut temporaries,
            arguments,
        )?;
        let right = self.prepare_matmul_input(
            encoder,
            table,
            right_tensor,
            bindings,
            &mut temporaries,
            arguments,
        )?;
        if left.copied || right.copied {
            encode_dispatch_barrier(encoder);
        }
        let final_output = self.encoder_tensor(output_tensor)?;
        let direct_output = classify(&final_output.layout)
            .kernel_strides()
            .filter(|&(column_major, _, _)| column_major == 0);
        let (output, copy_output) =
            if let Some((_, leading_dimension, batch_stride)) = direct_output {
                (
                    PreparedMatmulOutput {
                        tensor: self.encoder_tensor(output_tensor)?,
                        leading_dimension,
                        batch_stride,
                        copied: false,
                    },
                    false,
                )
            } else {
                let scratch = self.scratch_tensor(dtype, output_tensor.layout().shape())?;
                let (_, leading_dimension, batch_stride) = classify(&scratch.layout)
                    .kernel_strides()
                    .ok_or(BackendError::ExecutionFailed)?;
                temporaries.push(scratch.buffer.clone());
                (
                    PreparedMatmulOutput {
                        tensor: scratch,
                        leading_dimension,
                        batch_stride,
                        copied: true,
                    },
                    true,
                )
            };
        let parameter_buffer =
            self.encode_matmul_kernel(encoder, table, &left, &right, &output, bindings, arguments)?;
        temporaries.push(parameter_buffer);
        if copy_output {
            encode_dispatch_barrier(encoder);
            temporaries.extend(self.encode_copy_tensors(
                encoder,
                table,
                &output.tensor,
                &final_output,
                bindings,
                arguments,
            )?);
        }
        Ok(temporaries)
    }

    fn encode_sdpa_dispatch(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let Op::Sdpa {
            scale,
            causal,
            q_start,
        } = dispatch.op()
        else {
            return Err(BackendError::InvalidInput);
        };
        match select_sdpa(dispatch)? {
            SdpaKernel::Vector => self.encode_vector_sdpa(
                encoder, table, dispatch, scale, causal, q_start, bindings, arguments,
            ),
            SdpaKernel::Steel => self.encode_steel_sdpa(
                encoder, table, dispatch, scale, causal, q_start, bindings, arguments,
            ),
            SdpaKernel::Decomposed => self.encode_decomposed_sdpa(
                encoder, table, dispatch, scale, causal, q_start, bindings, arguments,
            ),
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_vector_sdpa(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        scale: f32,
        causal: bool,
        q_start: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        use objc2_metal::MTLSize;

        let [query_tensor, key_tensor, value_tensor] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let query = self.encoder_tensor(query_tensor)?;
        let key = self.encoder_tensor(key_tensor)?;
        let value = self.encoder_tensor(value_tensor)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let [query_heads, query_length, width] = shape3(&query.layout)?;
        let [kv_heads, key_length, _] = shape3(&key.layout)?;
        let [_, _, value_width] = shape3(&value.layout)?;
        let two_pass = key_length >= VECTOR_TWO_PASS_MIN_KEY_LENGTH;
        let heads_per_group = query_heads / kv_heads;
        let blocks = if two_pass {
            self.vector_block_count(key_length, heads_per_group, query_length)?
        } else {
            1
        };
        let params = Self::sdpa_params(
            &query, &key, &value, &output, scale, causal, q_start, blocks, arguments,
        )?;
        let constants = [
            (0, dtype_code(query.layout.dtype())),
            (1, dtype_code(key.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (3, dtype_code(value.layout.dtype())),
        ];
        let mut temporaries = vec![params.clone()];
        if !two_pass {
            let pipeline = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .get(vector_kernel("mlx_sdpa_vector", width)?, &constants)?;
            set_pipeline(encoder, &pipeline);
            for (index, tensor) in [&query, &key, &value, &output].into_iter().enumerate() {
                bindings.bind(table, index, &tensor.buffer);
            }
            bindings.bind(table, 4, &params);
            set_argument_table(encoder, table);
            dispatch_threadgroups(
                encoder,
                MTLSize {
                    width: usize::try_from(query_heads).map_err(|_| BackendError::InvalidInput)?,
                    height: usize::try_from(query_length)
                        .map_err(|_| BackendError::InvalidInput)?,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
            return Ok(temporaries);
        }

        let intermediate = self.scratch_tensor(
            query.layout.dtype(),
            &[query_heads, query_length, blocks, value_width],
        )?;
        let sums = self.scratch_tensor(DType::F32, &[query_heads, query_length, blocks])?;
        let maxs = self.scratch_tensor(DType::F32, &[query_heads, query_length, blocks])?;
        let first = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(vector_kernel("mlx_sdpa_vector_2pass_1", width)?, &constants)?;
        set_pipeline(encoder, &first);
        for (index, tensor) in [&query, &key, &value, &intermediate, &sums, &maxs]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 6, &params);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(kv_heads).map_err(|_| BackendError::InvalidInput)?,
                height: 1,
                depth: usize::try_from(blocks).map_err(|_| BackendError::InvalidInput)?,
            },
            MTLSize {
                width: usize::try_from(
                    32_u32
                        .checked_mul(heads_per_group)
                        .and_then(|count| count.checked_mul(query_length))
                        .ok_or(BackendError::InvalidInput)?,
                )
                .map_err(|_| BackendError::InvalidInput)?,
                height: 1,
                depth: 1,
            },
        );
        encode_dispatch_barrier(encoder);
        let second = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(vector_kernel("mlx_sdpa_vector_2pass_2", width)?, &constants)?;
        set_pipeline(encoder, &second);
        for (index, tensor) in [&intermediate, &sums, &maxs, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &params);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(query_heads).map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(query_length).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        temporaries.extend([intermediate.buffer, sums.buffer, maxs.buffer]);
        Ok(temporaries)
    }

    fn vector_block_count(
        &self,
        key_length: u32,
        heads_per_group: u32,
        query_length: u32,
    ) -> Result<u32, BackendError> {
        let simdgroups = heads_per_group
            .checked_mul(query_length)
            .ok_or(BackendError::InvalidInput)?;
        let architecture = self.device.architecture().name().to_string();
        let class = architecture.chars().next_back();
        let blocks = match class {
            Some('s') => {
                if key_length > 1024 && simdgroups > 4 {
                    match key_length {
                        ..=8192 => 128,
                        8193..=32_768 => 256,
                        32_769..=65_536 => 512,
                        _ => 1024,
                    }
                } else {
                    64
                }
            }
            Some('d') => {
                if simdgroups <= 2 && key_length > 8192 {
                    256
                } else if simdgroups >= 6 && key_length >= 65_536 {
                    1024
                } else if simdgroups >= 6 && key_length >= 16_384 {
                    512
                } else {
                    128
                }
            }
            _ if simdgroups >= 4 => 64,
            _ => 32,
        };
        Ok(blocks)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_steel_sdpa(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        scale: f32,
        causal: bool,
        q_start: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        use objc2_metal::MTLSize;

        let [query_tensor, key_tensor, value_tensor] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let query = self.encoder_tensor(query_tensor)?;
        let key = self.encoder_tensor(key_tensor)?;
        let value = self.encoder_tensor(value_tensor)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let [query_heads, query_length, width] = shape3(&query.layout)?;
        let [_, key_length, _] = shape3(&key.layout)?;
        let block_query = 32;
        let block_key = if width == 64 { 32 } else { 16 };
        let params = Self::sdpa_params(
            &query, &key, &value, &output, scale, causal, q_start, 1, arguments,
        )?;
        let constants = [
            (0, dtype_code(query.layout.dtype())),
            (1, dtype_code(key.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (3, dtype_code(value.layout.dtype())),
            (210, u32::from(query_length % block_query == 0)),
            (211, u32::from(key_length % block_key == 0)),
            (212, u32::from(causal)),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(steel_attention_kernel(width)?, &constants)?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&query, &key, &value, &output].into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &params);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(query_length.div_ceil(block_query))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(query_heads).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        Ok(vec![params])
    }

    #[allow(clippy::too_many_arguments)]
    fn sdpa_params(
        query: &EncoderTensor,
        key: &EncoderTensor,
        value: &EncoderTensor,
        output: &EncoderTensor,
        scale: f32,
        causal: bool,
        q_start: u32,
        blocks: u32,
        arguments: &mut ArgumentWriter,
    ) -> Result<BufferBinding, BackendError> {
        let [query_heads, query_length, width] = shape3(&query.layout)?;
        let [kv_heads, key_length, _] = shape3(&key.layout)?;
        let [_, _, value_width] = shape3(&value.layout)?;
        let mut bytes = Vec::with_capacity(172);
        for item in [
            query.layout.offset(),
            key.layout.offset(),
            value.layout.offset(),
            output.layout.offset(),
        ]
        .into_iter()
        .chain(query.layout.strides().iter().copied())
        .chain(key.layout.strides().iter().copied())
        .chain(value.layout.strides().iter().copied())
        .chain(output.layout.strides().iter().copied())
        {
            bytes.extend_from_slice(&item.to_ne_bytes());
        }
        bytes.extend_from_slice(&scale.to_ne_bytes());
        for item in [
            query_heads,
            query_length,
            width,
            kv_heads,
            key_length,
            value_width,
            query_heads / kv_heads,
            q_start,
            u32::from(causal),
            blocks,
        ] {
            bytes.extend_from_slice(&item.to_ne_bytes());
        }
        arguments.write(&bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_decomposed_sdpa(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        scale: f32,
        causal: bool,
        q_start: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [query, key, value] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let query = self.encoder_tensor(query)?;
        let key = self.encoder_tensor(key)?;
        let value = self.encoder_tensor(value)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let [query_heads, query_length, _] = shape3(&query.layout)?;
        let [kv_heads, key_length, _] = shape3(&key.layout)?;
        let [_, _, value_width] = shape3(&value.layout)?;
        let heads_per_group = query_heads / kv_heads;
        let scores = self.scratch_tensor(DType::F32, &[query_heads, query_length, key_length])?;
        let mut temporaries = vec![scores.buffer.clone()];
        temporaries.extend(self.encode_sdpa_query_key(
            encoder,
            table,
            &query,
            &key,
            &scores,
            kv_heads,
            heads_per_group,
            bindings,
            arguments,
        )?);
        encode_dispatch_barrier(encoder);
        temporaries.push(self.encode_sdpa_scale_mask(
            encoder, table, &scores, scale, causal, q_start, bindings, arguments,
        )?);
        encode_dispatch_barrier(encoder);
        temporaries.extend(
            self.encode_softmax_tensors(encoder, table, &scores, &scores, bindings, arguments)?,
        );
        encode_dispatch_barrier(encoder);

        temporaries.extend(self.encode_sdpa_probability_value(
            encoder,
            table,
            &scores,
            &value,
            &output,
            kv_heads,
            heads_per_group,
            query_length,
            value_width,
            bindings,
            arguments,
        )?);
        Ok(temporaries)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_sdpa_query_key(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        query: &EncoderTensor,
        key: &EncoderTensor,
        scores: &EncoderTensor,
        kv_heads: u32,
        heads_per_group: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let mut temporaries = Vec::new();
        for kv_head in 0..kv_heads {
            let first_head = kv_head
                .checked_mul(heads_per_group)
                .ok_or(BackendError::InvalidInput)?;
            let query_group = head_group(query, first_head, heads_per_group)?;
            let key_head = head_matrix(key, kv_head, true)?;
            let score_group = head_group(scores, first_head, heads_per_group)?;
            let left = self.prepare_matmul_encoder_input(
                encoder,
                table,
                query_group,
                bindings,
                &mut temporaries,
                arguments,
            )?;
            let right = self.prepare_matmul_encoder_input(
                encoder,
                table,
                key_head,
                bindings,
                &mut temporaries,
                arguments,
            )?;
            if left.copied || right.copied {
                encode_dispatch_barrier(encoder);
            }
            let (_, leading_dimension, batch_stride) = classify(&score_group.layout)
                .kernel_strides()
                .ok_or(BackendError::ExecutionFailed)?;
            temporaries.push(self.encode_matmul_kernel(
                encoder,
                table,
                &left,
                &right,
                &PreparedMatmulOutput {
                    tensor: score_group,
                    leading_dimension,
                    batch_stride,
                    copied: false,
                },
                bindings,
                arguments,
            )?);
        }
        Ok(temporaries)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_sdpa_probability_value(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        scores: &EncoderTensor,
        value: &EncoderTensor,
        output: &EncoderTensor,
        kv_heads: u32,
        heads_per_group: u32,
        query_length: u32,
        value_width: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let mut temporaries = Vec::new();
        for kv_head in 0..kv_heads {
            let first_head = kv_head
                .checked_mul(heads_per_group)
                .ok_or(BackendError::InvalidInput)?;
            let score_group = head_group(scores, first_head, heads_per_group)?;
            let value_head = head_matrix(value, kv_head, false)?;
            let output_group = head_group(output, first_head, heads_per_group)?;
            let left = self.prepare_matmul_encoder_input(
                encoder,
                table,
                score_group,
                bindings,
                &mut temporaries,
                arguments,
            )?;
            let right = self.prepare_matmul_encoder_input(
                encoder,
                table,
                value_head,
                bindings,
                &mut temporaries,
                arguments,
            )?;
            let direct = classify(&output_group.layout)
                .kernel_strides()
                .filter(|&(column_major, _, _)| column_major == 0);
            let (prepared_output, copy_output) = if let Some((_, ld, stride)) = direct {
                (
                    PreparedMatmulOutput {
                        tensor: output_group.clone(),
                        leading_dimension: ld,
                        batch_stride: stride,
                        copied: false,
                    },
                    false,
                )
            } else {
                let scratch = self.scratch_tensor(
                    output_group.layout.dtype(),
                    &[heads_per_group, query_length, value_width],
                )?;
                let (_, ld, stride) = classify(&scratch.layout)
                    .kernel_strides()
                    .ok_or(BackendError::ExecutionFailed)?;
                temporaries.push(scratch.buffer.clone());
                (
                    PreparedMatmulOutput {
                        tensor: scratch,
                        leading_dimension: ld,
                        batch_stride: stride,
                        copied: true,
                    },
                    true,
                )
            };
            temporaries.push(self.encode_matmul_kernel(
                encoder,
                table,
                &left,
                &right,
                &prepared_output,
                bindings,
                arguments,
            )?);
            if copy_output {
                encode_dispatch_barrier(encoder);
                temporaries.extend(self.encode_copy_tensors(
                    encoder,
                    table,
                    &prepared_output.tensor,
                    &output_group,
                    bindings,
                    arguments,
                )?);
            }
        }
        Ok(temporaries)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_sdpa_scale_mask(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        scores: &EncoderTensor,
        scale: f32,
        causal: bool,
        q_start: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<BufferBinding, BackendError> {
        use objc2_metal::MTLSize;

        let [_, query_length, key_length] = shape3(&scores.layout)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("sdpa_scale_mask", &[])?;
        let score_count =
            u32::try_from(scores.layout.element_count()).map_err(|_| BackendError::InvalidInput)?;
        let mut params = Vec::with_capacity(24);
        params.extend_from_slice(&scale.to_ne_bytes());
        for value in [
            query_length,
            key_length,
            q_start,
            u32::from(causal),
            score_count,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let params = arguments.write(&params)?;
        set_pipeline(encoder, &pipeline);
        bindings.bind(table, 0, &scores.buffer);
        bindings.bind(table, 1, &params);
        set_argument_table(encoder, table);
        let count = usize::try_from(scores.layout.element_count())
            .map_err(|_| BackendError::InvalidInput)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: count.div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(params)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_matmul_kernel(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        left: &PreparedMatmulInput,
        right: &PreparedMatmulInput,
        output: &PreparedMatmulOutput,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<BufferBinding, BackendError> {
        use objc2_metal::MTLSize;

        let shape = left.tensor.layout.shape();
        let rank = shape.len();
        let rows = shape[rank - 2];
        let inner = shape[rank - 1];
        let columns = *right
            .tensor
            .layout
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let batch = if rank == 3 { shape[0] } else { 1 };
        let dtype = output.tensor.layout.dtype();
        let mut params = Vec::with_capacity(96);
        for value in [
            left.tensor.layout.offset(),
            right.tensor.layout.offset(),
            output.tensor.layout.offset(),
            left.leading_dimension,
            right.leading_dimension,
            output.leading_dimension,
            left.batch_stride,
            right.batch_stride,
            output.batch_stride,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        for value in [
            rows,
            columns,
            inner,
            left.column_major,
            right.column_major,
            0,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let parameter_buffer = arguments.write(&params)?;
        let shape = MatmulShape {
            left_dtype: left.tensor.layout.dtype(),
            right_dtype: right.tensor.layout.dtype(),
            output_dtype: dtype,
            batch,
            rows,
            columns,
            inner,
            left_column_major: left.column_major != 0,
            right_column_major: right.column_major != 0,
        };
        let launch = self.matmul_launch(shape)?;
        log_matmul_route(left, right, output, &launch, shape);
        set_pipeline(encoder, &launch.pipeline);
        bindings.bind(table, 0, &left.tensor.buffer);
        bindings.bind(table, 1, &right.tensor.buffer);
        bindings.bind(table, 2, &output.tensor.buffer);
        bindings.bind(table, 3, &parameter_buffer);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(launch.block_columns))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(rows.div_ceil(launch.block_rows))
                    .map_err(|_| BackendError::InvalidInput)?,
                depth: usize::try_from(batch).map_err(|_| BackendError::InvalidInput)?,
            },
            MTLSize {
                width: launch.thread_count,
                height: 1,
                depth: 1,
            },
        );
        Ok(parameter_buffer)
    }

    fn prepare_matmul_input(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        tensor: &Tensor,
        bindings: &mut ArgumentBindings,
        temporaries: &mut Vec<BufferBinding>,
        arguments: &mut ArgumentWriter,
    ) -> Result<PreparedMatmulInput, BackendError> {
        let source = self.encoder_tensor(tensor)?;
        self.prepare_matmul_encoder_input(encoder, table, source, bindings, temporaries, arguments)
    }

    fn prepare_matmul_encoder_input(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        source: EncoderTensor,
        bindings: &mut ArgumentBindings,
        temporaries: &mut Vec<BufferBinding>,
        arguments: &mut ArgumentWriter,
    ) -> Result<PreparedMatmulInput, BackendError> {
        if let Some((column_major, leading_dimension, batch_stride)) =
            classify(&source.layout).kernel_strides()
        {
            return Ok(PreparedMatmulInput {
                tensor: source,
                column_major,
                leading_dimension,
                batch_stride,
                copied: false,
            });
        }
        let scratch = self.scratch_tensor(source.layout.dtype(), source.layout.shape())?;
        temporaries.push(scratch.buffer.clone());
        temporaries.extend(
            self.encode_copy_tensors(encoder, table, &source, &scratch, bindings, arguments)?,
        );
        let (column_major, leading_dimension, batch_stride) = classify(&scratch.layout)
            .kernel_strides()
            .ok_or(BackendError::ExecutionFailed)?;
        Ok(PreparedMatmulInput {
            tensor: scratch,
            column_major,
            leading_dimension,
            batch_stride,
            copied: true,
        })
    }

    fn matmul_launch(&self, shape: MatmulShape) -> Result<MatmulLaunch, BackendError> {
        let (kernel, block_rows, block_columns, thread_count, constants) = if shape.rows == 1 {
            let (kernel, block_columns, thread_count) =
                if shape.right_column_major && !shape.left_column_major {
                    ("gemv_transposed", 32, 256)
                } else {
                    ("gemv", 4, 32)
                };
            (
                kernel,
                1,
                block_columns,
                thread_count,
                vec![
                    (0, dtype_code(shape.left_dtype)),
                    (1, dtype_code(shape.right_dtype)),
                    (2, dtype_code(shape.output_dtype)),
                ],
            )
        } else {
            let config = select_gemm(
                shape.left_dtype,
                shape.batch,
                shape.rows,
                shape.columns,
                shape.inner,
                shape.left_column_major,
                shape.right_column_major,
            )
            .ok_or(BackendError::InvalidInput)?;
            (
                config.kernel,
                config.block_rows,
                config.block_columns,
                config.thread_count,
                vec![
                    (0, dtype_code(shape.left_dtype)),
                    (1, dtype_code(shape.right_dtype)),
                    (2, dtype_code(shape.output_dtype)),
                    (200, u32::from(shape.rows.is_multiple_of(config.block_rows))),
                    (
                        201,
                        u32::from(shape.columns.is_multiple_of(config.block_columns)),
                    ),
                    (
                        202,
                        u32::from(shape.inner.is_multiple_of(config.block_inner)),
                    ),
                ],
            )
        };
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        Ok(MatmulLaunch {
            pipeline,
            kernel,
            block_rows,
            block_columns,
            thread_count,
        })
    }

    fn encode_copy_tensors(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        input: &EncoderTensor,
        output: &EncoderTensor,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        use objc2_metal::MTLSize;

        let kernel = if input.layout.is_contiguous() && output.layout.is_contiguous() {
            "copy_contiguous"
        } else {
            "copy_strided"
        };
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                kernel,
                &[
                    (0, dtype_code(input.layout.dtype())),
                    (2, dtype_code(output.layout.dtype())),
                ],
            )?;
        set_pipeline(encoder, &pipeline);
        let layouts = vec![
            Self::layout_buffer(&input.layout, arguments)?,
            Self::layout_buffer(&output.layout, arguments)?,
        ];
        bindings.bind(table, 0, &input.buffer);
        bindings.bind(table, 1, &output.buffer);
        bindings.bind(table, 2, &layouts[0]);
        bindings.bind(table, 3, &layouts[1]);
        set_argument_table(encoder, table);
        let count = usize::try_from(output.layout.element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: count.div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(layouts)
    }

    fn encoder_tensor(&self, tensor: &Tensor) -> Result<EncoderTensor, BackendError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(EncoderTensor {
            buffer: BufferBinding::whole(buffers.get(tensor)?.raw.clone()),
            layout: tensor.layout().clone(),
        })
    }

    fn scratch_tensor(&self, dtype: DType, shape: &[u32]) -> Result<EncoderTensor, BackendError> {
        use objc2_metal::MTLResourceOptions;

        let elements = shape
            .iter()
            .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
        let byte_len = elements
            .and_then(|count| count.checked_mul(dtype.byte_size()))
            .ok_or(BackendError::AllocationFailed)?;
        let len = usize::try_from(byte_len).map_err(|_| BackendError::AllocationFailed)?;
        let buffer = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or(BackendError::AllocationFailed)?;
        let layout = Layout::contiguous(dtype, 0, shape.to_vec(), byte_len)
            .map_err(|_| BackendError::InvalidInput)?;
        Ok(EncoderTensor {
            buffer: BufferBinding::whole(buffer),
            layout,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn encode_rope(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        theta: f32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input, positions] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let shape = input.layout().shape();
        let heads = shape[1];
        let width = shape[2];
        let half_width = width / 2;
        let constants = [
            (0, dtype_code(input.layout().dtype())),
            (2, dtype_code(output.layout().dtype())),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("rope", &constants)?;
        set_pipeline(encoder, &pipeline);
        let mut params = [0_u8; 12];
        params[..4].copy_from_slice(&heads.to_ne_bytes());
        params[4..8].copy_from_slice(&width.to_ne_bytes());
        params[8..].copy_from_slice(&half_width.to_ne_bytes());
        let frequencies = (0..half_width)
            .flat_map(|index| {
                theta
                    .powf(2.0 * index as f32 / width as f32)
                    .recip()
                    .to_ne_bytes()
            })
            .collect::<Vec<_>>();
        let temporaries = vec![
            Self::layout_buffer(input.layout(), arguments)?,
            Self::layout_buffer(positions.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
            arguments.write(&params)?,
            arguments.write(&frequencies)?,
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [input, positions, output].into_iter().enumerate() {
            bindings.bind_raw(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        bindings.bind(table, 7, &temporaries[4]);
        drop(buffers);
        set_argument_table(encoder, table);
        let pair_count = output
            .layout()
            .element_count()
            .checked_div(2)
            .ok_or(BackendError::InvalidInput)?;
        let thread_count = usize::try_from(pair_count).map_err(|_| BackendError::InvalidInput)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: thread_count.div_ceil(group_width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        Ok(temporaries)
    }

    fn encode_embed(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [embeddings, ids] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let vocab = embeddings.layout().shape()[0];
        let width = embeddings.layout().shape()[1];
        let constants = [
            (0, dtype_code(embeddings.layout().dtype())),
            (2, dtype_code(output.layout().dtype())),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("embed", &constants)?;
        set_pipeline(encoder, &pipeline);
        let mut params = [0_u8; 8];
        params[..4].copy_from_slice(&vocab.to_ne_bytes());
        params[4..].copy_from_slice(&width.to_ne_bytes());
        let mut error_state = [0_u8; 8];
        error_state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
        let error_flag = arguments.write(&error_state)?;
        let temporaries = vec![
            Self::layout_buffer(embeddings.layout(), arguments)?,
            Self::layout_buffer(ids.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
            arguments.write(&params)?,
            error_flag.clone(),
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [embeddings, ids, output].into_iter().enumerate() {
            bindings.bind_raw(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        bindings.bind(table, 7, &temporaries[4]);
        drop(buffers);
        set_argument_table(encoder, table);
        let thread_count = usize::try_from(output.layout().element_count())
            .map_err(|_| BackendError::InvalidInput)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: thread_count.div_ceil(group_width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        Ok((temporaries, error_flag))
    }

    fn encode_softmax(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let output = self.encoder_tensor(dispatch.output())?;
        self.encode_softmax_tensors(encoder, table, &input, &output, bindings, arguments)
    }

    fn encode_softmax_tensors(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        input: &EncoderTensor,
        output: &EncoderTensor,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let width = *input
            .layout
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
        ];
        let kernel = if width <= 1024 {
            "softmax_single"
        } else {
            "softmax_looped"
        };
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        set_pipeline(encoder, &pipeline);
        let temporaries = vec![
            Self::layout_buffer(&input.layout, arguments)?,
            Self::layout_buffer(&output.layout, arguments)?,
            arguments.write(&width.to_ne_bytes())?,
        ];
        bindings.bind(table, 0, &input.buffer);
        bindings.bind(table, 1, &output.buffer);
        bindings.bind(table, 2, &temporaries[0]);
        bindings.bind(table, 3, &temporaries[1]);
        bindings.bind(table, 4, &temporaries[2]);
        set_argument_table(encoder, table);
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, &output.layout, width)?;
        dispatch_threadgroups(encoder, threadgroups, threads);
        Ok(temporaries)
    }

    fn encode_argmax(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let width = *input
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let chunks = width.div_ceil(ARGMAX_CHUNK_WIDTH);
        let rows = u32::try_from(output.layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let scratch = self.scratch_tensor(DType::U32, &[rows, chunks, 2])?;
        let (partials_pipeline, finalize_pipeline) = {
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            (
                pipelines.get(
                    "argmax_partials",
                    &[(0, dtype_code(input.layout().dtype()))],
                )?,
                pipelines.get("argmax_finalize", &[])?,
            )
        };
        let mut temporaries = vec![
            Self::layout_buffer(input.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
            arguments.write(&width.to_ne_bytes())?,
            arguments.write(&chunks.to_ne_bytes())?,
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        bindings.bind_raw(table, 0, &buffers.get(input)?.raw);
        drop(buffers);
        set_pipeline(encoder, &partials_pipeline);
        bindings.bind(table, 1, &scratch.buffer);
        bindings.bind(table, 2, &temporaries[0]);
        bindings.bind(table, 3, &temporaries[2]);
        bindings.bind(table, 4, &temporaries[3]);
        set_argument_table(encoder, table);
        let partial_threads = simd_thread_count(&partials_pipeline, 256)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows)
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .checked_mul(
                        usize::try_from(chunks).map_err(|_| BackendError::ExecutionFailed)?,
                    )
                    .ok_or(BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: partial_threads,
                height: 1,
                depth: 1,
            },
        );
        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &finalize_pipeline);
        bindings.bind(table, 0, &scratch.buffer);
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        bindings.bind_raw(table, 1, &buffers.get(output)?.raw);
        drop(buffers);
        bindings.bind(table, 2, &temporaries[1]);
        bindings.bind(table, 3, &temporaries[3]);
        set_argument_table(encoder, table);
        let requested_threads = usize::try_from(chunks)
            .map_err(|_| BackendError::ExecutionFailed)?
            .next_multiple_of(32)
            .min(256);
        let finalize_threads = simd_thread_count(&finalize_pipeline, requested_threads)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: finalize_threads,
                height: 1,
                depth: 1,
            },
        );
        temporaries.push(scratch.buffer);
        Ok(temporaries)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_sample(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        position: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [logits, sampling] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let width = *logits
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let rows = u32::try_from(output.layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let chunks = width.div_ceil(ARGMAX_CHUNK_WIDTH);
        let partials = self.scratch_tensor(DType::U32, &[rows, chunks, 2])?;
        let (sample_pipeline, partials_pipeline, finalize_pipeline) = {
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            let dtype = [(0, dtype_code(logits.layout().dtype()))];
            (
                pipelines.get("sample", &dtype)?,
                pipelines.get("sample_partials", &dtype)?,
                pipelines.get("sample_reduce_finalize", &[])?,
            )
        };
        let error_flag = arguments.write(&[0_u8; 8])?;
        let mut temporaries = vec![
            Self::layout_buffer(logits.layout(), arguments)?,
            Self::layout_buffer(sampling.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
            arguments.write(&width.to_ne_bytes())?,
            arguments.write(&position.to_ne_bytes())?,
            arguments.write(&chunks.to_ne_bytes())?,
            error_flag.clone(),
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        set_pipeline(encoder, &sample_pipeline);
        for (index, tensor) in [logits, sampling, output].into_iter().enumerate() {
            bindings.bind_raw(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        bindings.bind(table, 7, &temporaries[4]);
        bindings.bind(table, 8, &temporaries[6]);
        set_argument_table(encoder, table);
        let threads = simd_thread_count(&sample_pipeline, 256)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &partials_pipeline);
        bindings.bind_raw(table, 0, &buffers.get(logits)?.raw);
        bindings.bind_raw(table, 1, &buffers.get(sampling)?.raw);
        drop(buffers);
        bindings.bind(table, 2, &partials.buffer);
        bindings.bind(table, 3, &temporaries[0]);
        bindings.bind(table, 4, &temporaries[1]);
        bindings.bind(table, 5, &temporaries[3]);
        bindings.bind(table, 6, &temporaries[4]);
        bindings.bind(table, 7, &temporaries[5]);
        set_argument_table(encoder, table);
        let partial_threads = simd_thread_count(&partials_pipeline, 256)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows)
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .checked_mul(
                        usize::try_from(chunks).map_err(|_| BackendError::ExecutionFailed)?,
                    )
                    .ok_or(BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: partial_threads,
                height: 1,
                depth: 1,
            },
        );

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &finalize_pipeline);
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        bindings.bind_raw(table, 0, &buffers.get(sampling)?.raw);
        bindings.bind(table, 1, &partials.buffer);
        bindings.bind_raw(table, 2, &buffers.get(output)?.raw);
        drop(buffers);
        bindings.bind(table, 3, &temporaries[1]);
        bindings.bind(table, 4, &temporaries[2]);
        bindings.bind(table, 5, &temporaries[3]);
        bindings.bind(table, 6, &temporaries[5]);
        set_argument_table(encoder, table);
        let requested_threads = usize::try_from(chunks)
            .map_err(|_| BackendError::ExecutionFailed)?
            .next_multiple_of(32)
            .min(256);
        let finalize_threads = simd_thread_count(&finalize_pipeline, requested_threads)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: finalize_threads,
                height: 1,
                depth: 1,
            },
        );
        temporaries.push(partials.buffer);
        Ok((temporaries, error_flag))
    }

    fn encode_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        eps: f32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input, weight] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let width = *input
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let constants = [
            (0, dtype_code(input.layout().dtype())),
            (1, dtype_code(weight.layout().dtype())),
            (2, dtype_code(output.layout().dtype())),
        ];
        let kernel = if width <= 1024 {
            "rms_norm_single"
        } else {
            "rms_norm_looped"
        };
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        set_pipeline(encoder, &pipeline);
        let mut temporaries = vec![
            Self::layout_buffer(input.layout(), arguments)?,
            Self::layout_buffer(weight.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
        ];
        let mut params = [0_u8; 8];
        params[..4].copy_from_slice(&eps.to_ne_bytes());
        params[4..].copy_from_slice(&width.to_ne_bytes());
        temporaries.push(arguments.write(&params)?);
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [input, weight, output].into_iter().enumerate() {
            bindings.bind_raw(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        drop(buffers);
        set_argument_table(encoder, table);
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, output.layout(), width)?;
        dispatch_threadgroups(encoder, threadgroups, threads);
        Ok(temporaries)
    }

    fn encode_elementwise(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        kernel: &str,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        use objc2_metal::MTLSize;

        let operands = dispatch
            .inputs()
            .iter()
            .chain(std::iter::once(dispatch.output()))
            .collect::<Vec<_>>();
        let constants = dispatch
            .inputs()
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                Ok((
                    u32::try_from(index).map_err(|_| BackendError::InvalidInput)?,
                    dtype_code(tensor.layout().dtype()),
                ))
            })
            .chain(std::iter::once(Ok((
                2,
                dtype_code(dispatch.output().layout().dtype()),
            ))))
            .collect::<Result<Vec<_>, BackendError>>()?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        set_pipeline(encoder, &pipeline);
        let layouts = operands
            .iter()
            .map(|tensor| Self::layout_buffer(tensor.layout(), arguments))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in operands.iter().enumerate() {
            let buffer = buffers.get(tensor)?;
            bindings.bind_raw(table, index, &buffer.raw);
            bindings.bind(table, index + operands.len(), &layouts[index]);
        }
        drop(buffers);
        set_argument_table(encoder, table);
        let thread_count = usize::try_from(dispatch.output().layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: thread_count.div_ceil(group_width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        Ok(layouts)
    }

    fn encode_program(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        state: &mut ProgramEncodingState<'_>,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let program = dispatch.bound_program().ok_or(BackendError::InvalidInput)?;
        let prepared = dispatch
            .prepared_program()
            .ok_or(BackendError::InvalidInput)?;
        let handle = prepared
            .backend_handle::<MetalProgramHandle>()
            .ok_or(BackendError::InvalidInput)?;
        let width = dispatch
            .output()
            .layout()
            .shape()
            .last()
            .copied()
            .unwrap_or(1);
        let pipeline = handle.0.select(self, width)?;
        if pipeline.compile_fallback {
            *state.compile_fallbacks = state.compile_fallbacks.saturating_add(1);
        }
        set_pipeline(encoder, &pipeline.state);
        let operands = program
            .inputs()
            .iter()
            .chain(program.outputs())
            .collect::<Vec<_>>();
        let layouts = operands
            .iter()
            .map(|tensor| Self::layout_buffer(tensor.layout(), state.arguments))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in operands.iter().enumerate() {
            state
                .bindings
                .bind_raw(table, index, &buffers.get(tensor)?.raw);
            state
                .bindings
                .bind(table, index + operands.len(), &layouts[index]);
        }
        drop(buffers);
        set_argument_table(encoder, table);
        let (threadgroups, threads_per_threadgroup) = match program.program().program().kind {
            ProgramKind::Map => map_dispatch_geometry(&pipeline.state, dispatch.output().layout())?,
            ProgramKind::Row => {
                let width = dispatch
                    .output()
                    .layout()
                    .shape()
                    .last()
                    .copied()
                    .ok_or(BackendError::InvalidInput)?;
                program_row_dispatch_geometry(
                    &pipeline.state,
                    dispatch.output().layout(),
                    width,
                    pipeline.resident,
                )?
            }
        };
        dispatch_threadgroups(encoder, threadgroups, threads_per_threadgroup);
        Ok(layouts)
    }

    fn layout_buffer(
        layout: &Layout,
        arguments: &mut ArgumentWriter,
    ) -> Result<BufferBinding, BackendError> {
        let bytes = encode_layout(layout)?;
        arguments.write(&bytes)
    }

    fn argument_capacity(&self, dispatches: &[Dispatch]) -> Result<usize, BackendError> {
        self.argument_capacity_with_plan(dispatches, None)
    }

    fn argument_capacity_with_plan(
        &self,
        dispatches: &[Dispatch],
        plan: Option<&MetalEncodingPlan>,
    ) -> Result<usize, BackendError> {
        let mut arguments = ArgumentSizer::default();
        if let Some(plan) = plan {
            arguments.offset = plan.argument_offset;
        }
        for (index, dispatch) in dispatches.iter().enumerate() {
            if plan
                .and_then(|plan| plan.dispatches.get(index))
                .is_some_and(Option::is_some)
            {
                continue;
            }
            self.size_dispatch_arguments(dispatch, &mut arguments)?;
        }
        Ok(arguments.offset.max(1))
    }

    fn size_dispatch_arguments(
        &self,
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        match dispatch.op() {
            Op::Program(_) => {
                let program = dispatch.bound_program().ok_or(BackendError::InvalidInput)?;
                for _ in program.inputs().iter().chain(program.outputs()) {
                    arguments.write(112)?;
                }
            }
            Op::Copy => Self::size_copy_arguments(arguments)?,
            Op::Add | Op::SiluMul => {
                for _ in 0..=dispatch.inputs().len() {
                    arguments.write(112)?;
                }
            }
            Op::RmsNorm { .. } => {
                for len in [112, 112, 112, 8] {
                    arguments.write(len)?;
                }
            }
            Op::Softmax => Self::size_softmax_arguments(arguments)?,
            Op::Argmax => {
                Self::size_softmax_arguments(arguments)?;
                arguments.write(size_of::<u32>())?;
            }
            Op::Sample { .. } => {
                for len in [8, 112, 112, 112, 4, 4, 4] {
                    arguments.write(len)?;
                }
            }
            Op::Rope { .. } => {
                for len in [112, 112, 112, 12] {
                    arguments.write(len)?;
                }
                let width = dispatch
                    .output()
                    .layout()
                    .shape()
                    .last()
                    .copied()
                    .ok_or(BackendError::InvalidInput)?;
                let frequency_bytes = usize::try_from(width / 2)
                    .map_err(|_| BackendError::AllocationFailed)?
                    .checked_mul(size_of::<f32>())
                    .ok_or(BackendError::AllocationFailed)?;
                arguments.write(frequency_bytes)?;
            }
            Op::Embed => {
                for len in [8, 112, 112, 112, 8] {
                    arguments.write(len)?;
                }
            }
            Op::Matmul => Self::size_matmul_arguments(dispatch, arguments)?,
            Op::Sdpa { .. } => self.size_sdpa_arguments(dispatch, arguments)?,
        }
        Ok(())
    }

    fn size_copy_arguments(arguments: &mut ArgumentSizer) -> Result<(), BackendError> {
        arguments.write(112)?;
        arguments.write(112)
    }

    fn size_softmax_arguments(arguments: &mut ArgumentSizer) -> Result<(), BackendError> {
        arguments.write(112)?;
        arguments.write(112)?;
        arguments.write(size_of::<u32>())
    }

    fn size_matmul_arguments(
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        let [left, right] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        Self::size_prepared_matmul_input(left.layout(), arguments)?;
        Self::size_prepared_matmul_input(right.layout(), arguments)?;
        arguments.write(96)?;
        if classify(dispatch.output().layout())
            .kernel_strides()
            .is_none_or(|(column_major, _, _)| column_major != 0)
        {
            Self::size_copy_arguments(arguments)?;
        }
        Ok(())
    }

    fn size_prepared_matmul_input(
        layout: &Layout,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        if classify(layout).kernel_strides().is_none() {
            Self::size_copy_arguments(arguments)?;
        }
        Ok(())
    }

    fn size_sdpa_arguments(
        &self,
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        if select_sdpa(dispatch)? != SdpaKernel::Decomposed {
            return arguments.write(172);
        }
        let [query, key, value] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let query = self.encoder_tensor(query)?;
        let key = self.encoder_tensor(key)?;
        let value = self.encoder_tensor(value)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let [query_heads, query_length, _] = shape3(&query.layout)?;
        let [kv_heads, key_length, _] = shape3(&key.layout)?;
        let heads_per_group = query_heads
            .checked_div(kv_heads)
            .ok_or(BackendError::InvalidInput)?;
        let scores_layout = Layout::contiguous(
            DType::F32,
            0,
            vec![query_heads, query_length, key_length],
            u64::from(query_heads)
                .checked_mul(u64::from(query_length))
                .and_then(|count| count.checked_mul(u64::from(key_length)))
                .and_then(|count| count.checked_mul(DType::F32.byte_size()))
                .ok_or(BackendError::AllocationFailed)?,
        )
        .map_err(|_| BackendError::InvalidInput)?;
        let scores = EncoderTensor {
            buffer: query.buffer.clone(),
            layout: scores_layout,
        };
        for kv_head in 0..kv_heads {
            let first_head = kv_head
                .checked_mul(heads_per_group)
                .ok_or(BackendError::InvalidInput)?;
            let query_group = head_group(&query, first_head, heads_per_group)?;
            let key_head = head_matrix(&key, kv_head, true)?;
            Self::size_prepared_matmul_input(&query_group.layout, arguments)?;
            Self::size_prepared_matmul_input(&key_head.layout, arguments)?;
            arguments.write(96)?;
        }
        arguments.write(24)?;
        Self::size_softmax_arguments(arguments)?;
        for kv_head in 0..kv_heads {
            let first_head = kv_head
                .checked_mul(heads_per_group)
                .ok_or(BackendError::InvalidInput)?;
            let score_group = head_group(&scores, first_head, heads_per_group)?;
            let value_head = head_matrix(&value, kv_head, false)?;
            Self::size_prepared_matmul_input(&score_group.layout, arguments)?;
            Self::size_prepared_matmul_input(&value_head.layout, arguments)?;
            arguments.write(96)?;
            let output_group = head_group(&output, first_head, heads_per_group)?;
            if classify(&output_group.layout)
                .kernel_strides()
                .is_none_or(|(column_major, _, _)| column_major != 0)
            {
                Self::size_copy_arguments(arguments)?;
            }
        }
        Ok(())
    }

    fn begin_command_buffer(
        &self,
        allocator: &ProtocolObject<dyn MTL4CommandAllocator>,
    ) -> Result<Retained<ProtocolObject<dyn MTL4CommandBuffer>>, BackendError> {
        let command_buffer = self
            .device
            .newCommandBuffer()
            .ok_or(BackendError::ExecutionFailed)?;
        command_buffer.beginCommandBufferWithAllocator(allocator);
        Ok(command_buffer)
    }

    fn make_timestamps(&self, count: usize) -> Result<GpuTimestamps, BackendError> {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: The descriptor count matches all timestamp indices encoded by the caller.
        unsafe {
            descriptor.setCount(count);
        }
        let heap = self
            .device
            .newCounterHeapWithDescriptor_error(&descriptor)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let frequency = self.device.queryTimestampFrequency();
        let entry_size = self
            .device
            .sizeOfCounterHeapEntry(MTL4CounterHeapType::Timestamp);
        if frequency == 0 || entry_size < size_of::<u64>() {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(GpuTimestamps {
            heap,
            frequency,
            entry_size,
            count,
        })
    }

    fn command_resources(
        &self,
        tensors: &[Tensor],
        encoded: EncodedDispatches,
        graph_buffers: Option<&HashSet<BufferId>>,
    ) -> Result<CommandResources, BackendError> {
        let EncodedDispatches {
            temporaries,
            error_flags,
            bindings,
            arguments,
            program_encoding: _,
            program_compile_fallbacks: _,
        } = encoded;
        let mut indices = HashMap::<u64, usize>::new();
        let mut owned = Vec::<InFlightBuffer>::new();
        let mut add = |raw: MetalBufferRef, pool_resident: bool| {
            let address = raw.gpuAddress();
            *indices.entry(address).or_insert_with(|| {
                let index = owned.len();
                owned.push(InFlightBuffer { raw, pool_resident });
                index
            })
        };
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            for tensor in tensors {
                let buffer = buffers.get(tensor)?;
                add(
                    buffer.raw.clone(),
                    buffer.pool_resident
                        || graph_buffers.is_some_and(|buffers| buffers.contains(&tensor.buffer())),
                );
            }
        }
        for temporary in temporaries {
            add(temporary.raw, false);
        }
        let error_flags = error_flags
            .iter()
            .map(|flag| {
                indices
                    .get(&flag.raw.gpuAddress())
                    .copied()
                    .map(|index| (index, flag.offset))
                    .ok_or(BackendError::ExecutionFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for binding in bindings.ranges {
            let offset =
                u64::try_from(binding.offset).map_err(|_| BackendError::ExecutionFailed)?;
            if !indices.contains_key(&binding.base)
                || binding.base.checked_add(offset) != Some(binding.address)
            {
                return Err(BackendError::ExecutionFailed);
            }
            if arguments
                .as_ref()
                .is_some_and(|usage| usage.base == binding.base)
                && !arguments
                    .as_ref()
                    .is_some_and(|usage| usage.written.contains(&(binding.offset, binding.len)))
            {
                return Err(BackendError::ExecutionFailed);
            }
        }
        Ok(CommandResources {
            buffers: owned,
            error_flags,
        })
    }

    fn make_resident(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        resources: &CommandResources,
    ) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, BackendError> {
        let descriptor = MTLResidencySetDescriptor::new();
        let residency = self
            .device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|_| BackendError::ExecutionFailed)?;
        for buffer in &resources.buffers {
            if buffer.pool_resident {
                continue;
            }
            let buffer: &ProtocolObject<dyn MTLBuffer> = &buffer.raw;
            let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
            residency.addAllocation(allocation);
        }
        residency.commit();
        command_buffer.useResidencySet(&residency);
        Ok(residency)
    }

    fn retain_tensors(
        &self,
        accesses: &[(Tensor, bool)],
        event: InFlightEvent,
        resources: CommandResources,
        residency: InFlightResidency,
        timestamps: GpuTimestamps,
        order_on_queue: bool,
    ) -> Result<(Arc<Completion>, Option<u64>), BackendError> {
        let mut seen = HashSet::<BufferId>::new();
        let unique = accesses
            .iter()
            .map(|(tensor, _)| tensor)
            .filter(|tensor| seen.insert(tensor.buffer()))
            .collect::<Vec<_>>();
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut dependency = None;
        let mut dependencies = Vec::new();
        let mut seen_dependencies = HashSet::new();
        for tensor in &unique {
            let buffer = buffers.get_mut(tensor)?;
            if order_on_queue {
                dependency = dependency.max(buffer.pending_event_value());
            } else {
                buffer.wait_pending(self.gpu_timeout)?;
            }
        }
        for (tensor, writes) in accesses {
            if !writes {
                dependencies.extend(
                    buffers
                        .get_mut(tensor)?
                        .dependencies(&tensor.layout().byte_span())
                        .into_iter()
                        .filter(|completion| {
                            seen_dependencies.insert(Arc::as_ptr(completion) as usize)
                        }),
                );
            }
        }
        let completion = Completion::new(
            resources,
            event,
            residency,
            Some(timestamps),
            Arc::downgrade(&self.in_flight),
            dependencies,
        );
        for (tensor, writes) in accesses {
            buffers.get_mut(tensor)?.track(tensor, &completion, *writes);
        }
        Ok((completion, dependency))
    }

    fn commit(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        accesses: &[(Tensor, bool)],
        resources: CommandResources,
        residency: CommitResidency,
        timestamps: GpuTimestamps,
        objects: &mut SubmissionObjects,
    ) -> Result<MetalSubmission, BackendError> {
        let mut next_event_value = self
            .next_event_value
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let (completion, dependency) = self.retain_tensors(
            accesses,
            InFlightEvent {
                raw: self.shared_event.clone(),
            },
            resources,
            InFlightResidency {
                _raw: residency.sets,
            },
            timestamps,
            residency.order_on_queue,
        )?;
        let registration = self
            .in_flight
            .track(&completion, &self.event_listener, objects)?;
        let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = command_buffer;
        let mut command_buffers = [NonNull::from(command_buffer_ref)];
        let event_value = *next_event_value;
        let Some(following_event_value) = event_value.checked_add(1) else {
            drop(next_event_value);
            self.in_flight.cancel(&completion);
            return Err(BackendError::ExecutionFailed);
        };
        completion.mark_committed(event_value);
        let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &self.shared_event;
        let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
        if let Some(dependency) = dependency {
            self.queue.waitForEvent_value(event, dependency);
        }
        // SAFETY: The pointer names one live command buffer and the count matches the array.
        unsafe {
            self.queue.commit_count_options(
                NonNull::from(&mut command_buffers[0]),
                command_buffers.len(),
                &completion.commit.options,
            );
        }
        self.queue.signalEvent_value(event, event_value);
        *next_event_value = following_event_value;
        drop(next_event_value);
        registration.register(event_value);
        Ok(MetalSubmission {
            completion,
            timeout: self.gpu_timeout,
            profile: None,
            dispatch_operations: Vec::new(),
        })
    }
}

fn shape3(layout: &Layout) -> Result<[u32; 3], BackendError> {
    layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)
}

fn supported_dispatch(dispatch: &Dispatch) -> bool {
    match dispatch.op() {
        Op::Program(_) => dispatch.bound_program().is_some(),
        Op::Copy
        | Op::Add
        | Op::SiluMul
        | Op::RmsNorm { .. }
        | Op::Softmax
        | Op::Argmax
        | Op::Rope { .. }
        | Op::Embed
        | Op::Matmul
        | Op::Sdpa { .. }
        | Op::Sample { .. } => true,
    }
}

fn reusable_dispatch(dispatch: &Dispatch) -> bool {
    match dispatch.op() {
        Op::Argmax | Op::Sample { .. } | Op::Embed | Op::Sdpa { .. } => false,
        Op::Matmul => {
            dispatch
                .inputs()
                .iter()
                .all(|tensor| classify(tensor.layout()).kernel_strides().is_some())
                && classify(dispatch.output().layout())
                    .kernel_strides()
                    .is_some_and(|(column_major, _, _)| column_major == 0)
        }
        _ => true,
    }
}

fn vector_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [query, key, value] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let [query_heads, query_length, width] = shape3(query.layout())?;
    let [kv_heads, key_length, _] = shape3(key.layout())?;
    let [_, _, value_width] = shape3(value.layout())?;
    let simdgroups = query_heads
        .checked_div(kv_heads)
        .and_then(|group| group.checked_mul(query_length));
    Ok(width == value_width
        && matches!(width, 64 | 128)
        && simdgroups.is_some_and(|count| count <= 32)
        && key_length <= VECTOR_MAX_KEY_LENGTH)
}

fn vector_kernel(prefix: &str, width: u32) -> Result<&'static str, BackendError> {
    match (prefix, width) {
        ("mlx_sdpa_vector", 64) => Ok("mlx_sdpa_vector_64"),
        ("mlx_sdpa_vector", 128) => Ok("mlx_sdpa_vector_128"),
        ("mlx_sdpa_vector_2pass_1", 64) => Ok("mlx_sdpa_vector_2pass_1_64"),
        ("mlx_sdpa_vector_2pass_1", 128) => Ok("mlx_sdpa_vector_2pass_1_128"),
        ("mlx_sdpa_vector_2pass_2", 64) => Ok("mlx_sdpa_vector_2pass_2_64"),
        ("mlx_sdpa_vector_2pass_2", 128) => Ok("mlx_sdpa_vector_2pass_2_128"),
        _ => Err(BackendError::InvalidInput),
    }
}

fn steel_attention_kernel(width: u32) -> Result<&'static str, BackendError> {
    match width {
        64 => Ok("steel_attention_64"),
        128 => Ok("steel_attention_128"),
        _ => Err(BackendError::InvalidInput),
    }
}

fn select_sdpa(dispatch: &Dispatch) -> Result<SdpaKernel, BackendError> {
    #[cfg(test)]
    if let Some(kernel) = FORCED_SDPA_KERNEL.with(Cell::get) {
        return Ok(kernel);
    }
    let [query, key, _] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let query_length = shape3(query.layout())?[1];
    let key_length = shape3(key.layout())?[1];
    if query_length == 1
        && key_length >= VECTOR_SELECTED_MIN_KEY_LENGTH
        && vector_sdpa_supported(dispatch)?
    {
        Ok(SdpaKernel::Vector)
    } else if query_length >= STEEL_SELECTED_MIN_QUERY_LENGTH
        && query_length == key_length
        && steel_sdpa_supported(dispatch)?
    {
        Ok(SdpaKernel::Steel)
    } else {
        Ok(SdpaKernel::Decomposed)
    }
}

fn steel_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [query, _, value] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let [_, query_length, width] = shape3(query.layout())?;
    let [_, _, value_width] = shape3(value.layout())?;
    Ok(query_length > 1 && width == value_width && matches!(width, 64 | 128))
}

fn head_group(
    tensor: &EncoderTensor,
    first_head: u32,
    head_count: u32,
) -> Result<EncoderTensor, BackendError> {
    let [heads, rows, columns] = shape3(&tensor.layout)?;
    let slices = [
        Slice::new(first_head, head_count, 1).map_err(|_| BackendError::InvalidInput)?,
        Slice::new(0, rows, 1).map_err(|_| BackendError::InvalidInput)?,
        Slice::new(0, columns, 1).map_err(|_| BackendError::InvalidInput)?,
    ];
    if first_head
        .checked_add(head_count)
        .is_none_or(|end| end > heads)
    {
        return Err(BackendError::InvalidInput);
    }
    let layout = tensor
        .layout
        .slice(&slices)
        .map_err(|_| BackendError::InvalidInput)?;
    Ok(EncoderTensor {
        buffer: tensor.buffer.clone(),
        layout,
    })
}

fn head_matrix(
    tensor: &EncoderTensor,
    head: u32,
    transpose: bool,
) -> Result<EncoderTensor, BackendError> {
    let [heads, rows, columns] = shape3(&tensor.layout)?;
    if head >= heads {
        return Err(BackendError::InvalidInput);
    }
    let offset = u64::from(head)
        .checked_mul(tensor.layout.strides()[0])
        .and_then(|distance| tensor.layout.offset().checked_add(distance))
        .ok_or(BackendError::InvalidInput)?;
    let (shape, strides) = if transpose {
        (
            vec![columns, rows],
            vec![tensor.layout.strides()[2], tensor.layout.strides()[1]],
        )
    } else {
        (
            vec![rows, columns],
            vec![tensor.layout.strides()[1], tensor.layout.strides()[2]],
        )
    };
    let layout = Layout::new(
        tensor.layout.dtype(),
        offset,
        shape,
        strides,
        tensor.layout.buffer_len(),
    )
    .map_err(|_| BackendError::InvalidInput)?;
    Ok(EncoderTensor {
        buffer: tensor.buffer.clone(),
        layout,
    })
}

fn set_pipeline(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    record_plan_call(PlanCall::Pipeline(pipeline.clone()));
    encoder.setComputePipelineState(pipeline);
}

fn set_argument_table(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    table: &ProtocolObject<dyn MTL4ArgumentTable>,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    record_plan_call(PlanCall::ArgumentTable);
    encoder.setArgumentTable(Some(table));
}

fn dispatch_threadgroups(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    record_plan_call(PlanCall::Dispatch {
        threadgroups,
        threads_per_threadgroup,
    });
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
}

fn encode_dispatch_barrier(encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>) {
    use objc2_metal::{MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages};

    record_plan_call(PlanCall::Barrier);
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
}

fn write_dispatch_timestamp(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    timestamps: Option<&GpuTimestamps>,
    index: usize,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    let Some(timestamps) = timestamps else {
        return;
    };
    // SAFETY: Profiled heaps reserve `2 + 2 * dispatches` entries, and callers keep every
    // per-dispatch index below that bound.
    unsafe {
        encoder.writeTimestampWithGranularity_intoHeap_atIndex(
            MTL4TimestampGranularity::Precise,
            &timestamps.heap,
            index,
        );
    }
}

fn row_dispatch_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output: &Layout,
    width: u32,
) -> Result<(MTLSize, MTLSize), BackendError> {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let thread_count = if width <= 1024 {
        usize::try_from(width)
            .map_err(|_| BackendError::ExecutionFailed)?
            .next_multiple_of(32)
            .min(max_threads)
    } else {
        max_threads.min(256)
    };
    let rows = output
        .element_count()
        .checked_div(u64::from(width))
        .ok_or(BackendError::InvalidInput)?;
    Ok((
        MTLSize {
            width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_count,
            height: 1,
            depth: 1,
        },
    ))
}

fn program_row_dispatch_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output: &Layout,
    width: u32,
    resident: bool,
) -> Result<(MTLSize, MTLSize), BackendError> {
    if !resident {
        return row_dispatch_geometry(pipeline, output, width);
    }
    let width = usize::try_from(width).map_err(|_| BackendError::ExecutionFailed)?;
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if width > max_threads {
        return Err(BackendError::ExecutionFailed);
    }
    let preferred = width
        .checked_next_multiple_of(32)
        .ok_or(BackendError::ExecutionFailed)?;
    let thread_count = if preferred <= max_threads {
        preferred
    } else {
        width
    };
    let rows = output
        .element_count()
        .checked_div(u64::try_from(width).map_err(|_| BackendError::InvalidInput)?)
        .ok_or(BackendError::InvalidInput)?;
    Ok((
        MTLSize {
            width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: thread_count,
            height: 1,
            depth: 1,
        },
    ))
}

fn map_dispatch_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output: &Layout,
) -> Result<(MTLSize, MTLSize), BackendError> {
    let shape = output.shape();
    let width = shape.last().copied().unwrap_or(1);
    let height = shape.iter().rev().nth(1).copied().unwrap_or(1);
    let depth = shape[..shape.len().saturating_sub(2)]
        .iter()
        .try_fold(1_usize, |count, &extent| {
            count.checked_mul(usize::try_from(extent).ok()?)
        })
        .ok_or(BackendError::ExecutionFailed)?;
    let width = usize::try_from(width).map_err(|_| BackendError::ExecutionFailed)?;
    let group_width = pipeline
        .maxTotalThreadsPerThreadgroup()
        .clamp(1, 256)
        .min(width.max(1));
    Ok((
        MTLSize {
            width: width.div_ceil(group_width),
            height: usize::try_from(height).map_err(|_| BackendError::ExecutionFailed)?,
            depth,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    ))
}

fn simd_thread_count(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    requested: usize,
) -> Result<usize, BackendError> {
    let available = pipeline.maxTotalThreadsPerThreadgroup().min(requested);
    if available < 32 {
        return Err(BackendError::ExecutionFailed);
    }
    Ok(available - available % 32)
}

const fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
    }
}

const fn dtype_key(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
    }
}

fn encode_layout(layout: &Layout) -> Result<[u8; 112], BackendError> {
    let mut bytes = [0_u8; 112];
    bytes[0..8].copy_from_slice(&layout.offset().to_ne_bytes());
    let rank = u32::try_from(layout.shape().len()).map_err(|_| BackendError::InvalidInput)?;
    bytes[8..12].copy_from_slice(&rank.to_ne_bytes());
    for (axis, &extent) in layout.shape().iter().enumerate() {
        let start = 12_usize
            .checked_add(axis.checked_mul(4).ok_or(BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)?;
        bytes[start..start + 4].copy_from_slice(&extent.to_ne_bytes());
    }
    let element_count =
        u32::try_from(layout.element_count()).map_err(|_| BackendError::InvalidInput)?;
    bytes[44..48].copy_from_slice(&element_count.to_ne_bytes());
    for (axis, &stride) in layout.strides().iter().enumerate() {
        let start = 48_usize
            .checked_add(axis.checked_mul(8).ok_or(BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)?;
        bytes[start..start + 8].copy_from_slice(&stride.to_ne_bytes());
    }
    Ok(bytes)
}

pub(super) struct PipelineCache {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    fast_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    nan_preserving_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    pub(super) fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        source: &str,
    ) -> Result<Self, BackendError> {
        let source = NSString::from_str(source);
        let fast_options = compile_options(MTLMathMode::Fast);
        let fast_library = device
            .newLibraryWithSource_options_error(&source, Some(&fast_options))
            .map_err(|_| BackendError::ExecutionFailed)?;
        let nan_preserving_options = compile_options(MTLMathMode::Relaxed);
        let nan_preserving_library = device
            .newLibraryWithSource_options_error(&source, Some(&nan_preserving_options))
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            device: device.clone(),
            fast_library,
            nan_preserving_library,
            pipelines: HashMap::new(),
        })
    }

    pub(super) fn get(
        &mut self,
        name: &str,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let key = PipelineKey {
            name: name.to_owned(),
            constants: constants.to_vec(),
        };
        if let Some(pipeline) = self.pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let values = MTLFunctionConstantValues::new();
        for &(index, value) in constants {
            // SAFETY: `value` is live for the call, its type matches `MTLDataType::UInt`, and
            // the caller supplies indices declared by the selected embedded kernel.
            unsafe {
                values.setConstantValue_type_atIndex(
                    NonNull::from(&value).cast::<c_void>(),
                    MTLDataType::UInt,
                    usize::try_from(index).map_err(|_| BackendError::ExecutionFailed)?,
                );
            }
        }
        let library = if nan_preserving_kernel(name) {
            &self.nan_preserving_library
        } else {
            &self.fast_library
        };
        let name = NSString::from_str(name);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &values)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn compile_program_source(
        &self,
        key: &ProgramPipelineKey,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let source = NSString::from_str(source);
        let options = compile_options(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|_| program_compile_failed(key.hash))?;
        let kernel = if key.row {
            map_codegen::ROW_KERNEL_NAME
        } else {
            map_codegen::KERNEL_NAME
        };
        let name = NSString::from_str(kernel);
        let values = MTLFunctionConstantValues::new();
        let constants = key
            .input_dtypes
            .iter()
            .enumerate()
            .map(|(slot, &dtype)| ([0, 1, 3, 4, 5, 6, 7, 8][slot], dtype))
            .chain(
                key.output_dtypes
                    .iter()
                    .enumerate()
                    .map(|(slot, &dtype)| ([2, 9, 10, 11][slot], dtype)),
            )
            .map(|(index, dtype)| (index, u32::from(dtype)))
            .collect::<Vec<_>>();
        for &(index, dtype) in &constants {
            // SAFETY: `dtype` is live for the call, its type matches `MTLDataType::UInt`, and
            // program dtype constants occupy the declared indices zero through eleven.
            unsafe {
                values.setConstantValue_type_atIndex(
                    NonNull::from(&dtype).cast::<c_void>(),
                    MTLDataType::UInt,
                    index,
                );
            }
        }
        let function = library
            .newFunctionWithName_constantValues_error(&name, &values)
            .map_err(|_| program_compile_failed(key.hash))?;
        self.device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| program_compile_failed(key.hash))
    }
}

fn nan_preserving_kernel(name: &str) -> bool {
    matches!(
        name,
        "softmax_single" | "softmax_looped" | "argmax_partials"
    )
}

fn compile_options(math_mode: MTLMathMode) -> Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    options.setMathMode(math_mode);
    options
}

fn program_compile_failed(hash: ProgramHash) -> BackendError {
    let mut encoded = String::with_capacity(64);
    for byte in hash.as_bytes() {
        let _ = write!(encoded, "{byte:02x}");
    }
    eprintln!("Metal program compilation failed: {encoded}");
    BackendError::ExecutionFailed
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        thread,
        time::{Duration, Instant},
    };

    use forja_core::{
        Affine, Backend, CommandList, DType, GraphLimits, Op, ParamSpace, Slice, Submission,
        SymbolicLayout, TemplateTensor, ViewOp,
        program::{
            BinOp, BoundProgram, Inst, KernelSignature, Program, ProgramKind, RedOp,
            ValidatedProgram, bind_program, prepare_program,
        },
    };
    use forja_cpu::CpuBackend;
    use forja_testing::{TensorSpec, assert_backends_agree, assert_outputs_agree};

    use super::*;

    fn symbolic_prefix(base: &Tensor, space: ParamSpace, len: Affine) -> TemplateTensor {
        let layout = SymbolicLayout::new(base.layout().clone(), space)
            .slice(0, 0.into(), len, 1)
            .unwrap();
        TemplateTensor::symbolic(base.clone(), layout).unwrap()
    }

    fn sample_params(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> Vec<u8> {
        [
            temperature.to_bits(),
            top_k,
            top_p.to_bits(),
            u32::try_from(seed & u64::from(u32::MAX)).unwrap(),
            u32::try_from(seed >> 32).unwrap(),
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sample(
        backend: &MetalBackend,
        logits: &[u8],
        width: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        position: u32,
    ) -> Result<u32, BackendError> {
        let input = backend.alloc(DType::F32, &[1, width])?;
        backend.write(&input, logits)?;
        let params = backend.alloc(DType::U32, &[5])?;
        backend.write(&params, &sample_params(temperature, top_k, top_p, seed))?;
        let output = backend.alloc(DType::U32, &[1])?;
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Sample { position }, &[&input, &params], &output)
            .map_err(|_| BackendError::InvalidInput)?;
        backend.submit(commands)?.wait()?;
        backend
            .read(&output)?
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| BackendError::ExecutionFailed)
    }

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
    }

    #[test]
    fn plan_recorder_is_scoped_and_preserves_call_order() {
        record_plan_call(PlanCall::ArgumentTable);
        let recorder = PlanRecorderScope::enter(0);
        record_plan_call(PlanCall::ArgumentTable);
        record_plan_call(PlanCall::Dispatch {
            threadgroups: MTLSize {
                width: 7,
                height: 1,
                depth: 1,
            },
            threads_per_threadgroup: MTLSize {
                width: 33,
                height: 1,
                depth: 1,
            },
        });
        record_plan_call(PlanCall::Barrier);
        let calls = recorder.finish();

        assert!(matches!(calls.as_slice(), [
            PlanCall::ArgumentTable,
            PlanCall::Dispatch { threadgroups, threads_per_threadgroup },
            PlanCall::Barrier,
        ] if threadgroups.width == 7 && threadgroups.height == 1 && threadgroups.depth == 1
            && threads_per_threadgroup.width == 33
            && threads_per_threadgroup.height == 1
            && threads_per_threadgroup.depth == 1));
        assert!(PlanRecorderScope::enter(0).finish().is_empty());
    }

    #[test]
    fn program_cache_key_ignores_shapes_and_strides() {
        let backend = CpuBackend::new();
        let contiguous = identity_program(&backend, DType::F16, DType::F32, &[7, 33], false);
        let permuted = identity_program(&backend, DType::F16, DType::F32, &[7, 33], true);
        let other_shape = identity_program(&backend, DType::F16, DType::F32, &[1, 4097], false);
        let other_rank = identity_program(&backend, DType::F16, DType::F32, &[7], false);
        let other_input = identity_program(&backend, DType::BF16, DType::F32, &[7, 33], false);
        let other_output = identity_program(&backend, DType::F16, DType::F16, &[7, 33], false);
        let key = program_pipeline_key(&contiguous, false);
        assert_eq!(key, program_pipeline_key(&permuted, false));
        assert_eq!(key, program_pipeline_key(&other_shape, false));
        assert_ne!(key, program_pipeline_key(&other_rank, false));
        assert_ne!(key, program_pipeline_key(&other_input, false));
        assert_ne!(key, program_pipeline_key(&other_output, false));
    }

    #[test]
    fn row_pipeline_key_distinguishes_register_residency() {
        let backend = CpuBackend::new();
        let resident = row_program(&backend, map_codegen::REGISTER_RESIDENT_WIDTH);
        let resident_key = program_pipeline_key(&resident, true);
        let rereading_key = program_pipeline_key(&resident, false);
        assert!(resident_key.resident);
        assert!(!rereading_key.resident);
        assert_ne!(resident_key, rereading_key);
    }

    #[test]
    fn identical_preparations_share_compilation_until_handles_drop() {
        let budget = crate::ProgramCompileBudget::new(3, Duration::from_hours(24)).unwrap();
        let backend = MetalBackend::with_program_compile_budget(budget).unwrap();
        let signature = KernelSignature::new(1, vec![DType::F32], vec![DType::F32], 0);
        let program = |constant| {
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Const(constant),
                    Inst::Binary(BinOp::Add, 0, 1),
                ],
                outputs: vec![(0, 2)],
            }
            .validate()
            .unwrap()
        };
        let prepared = (0..100)
            .map(|_| prepare_program(&backend, program(1.0), signature.clone()).unwrap())
            .collect::<Vec<_>>();
        let first = prepared[0].backend_handle::<MetalProgramHandle>().unwrap();
        assert!(prepared.iter().all(|candidate| Arc::ptr_eq(
            &first.0,
            &candidate.backend_handle::<MetalProgramHandle>().unwrap().0
        )));
        let second = prepare_program(&backend, program(2.0), signature.clone()).unwrap();
        drop(prepared);
        let replacement = prepare_program(&backend, program(1.0), signature.clone()).unwrap();
        assert!(matches!(
            prepare_program(&backend, program(3.0), signature),
            Err(forja_core::program::PrepareError::Backend(
                BackendError::QuotaExceeded
            ))
        ));
        drop((second, replacement));
    }

    #[test]
    fn prepared_dispatch_only_charges_lazy_row_compilation() {
        let budget = crate::ProgramCompileBudget::new(1, Duration::from_hours(24)).unwrap();
        let map_backend = MetalBackend::with_program_compile_budget(budget).unwrap();
        let map = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Const(1.0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let signature = KernelSignature::new(1, vec![], vec![DType::F32], 0);
        let prepared = prepare_program(&map_backend, map, signature.clone()).unwrap();
        let output = map_backend.alloc(DType::F32, &[33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[], &[&output])
            .unwrap();
        map_backend.submit(commands).unwrap().wait().unwrap();

        let row_backend = MetalBackend::with_program_compile_budget(budget).unwrap();
        let row = Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Const(1.0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(&row_backend, row, signature).unwrap();
        let output = row_backend.alloc(DType::F32, &[33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[], &[&output])
            .unwrap();
        let submission = row_backend.submit_profiled(commands).unwrap();
        submission.wait().unwrap();
        assert_eq!(submission.profile().unwrap().program_compile_fallbacks, 1);
        assert_eq!(
            row_backend.read(&output).unwrap(),
            [1.0_f32.to_le_bytes(); 33].concat()
        );
    }

    #[test]
    fn preparation_prunes_dead_dedup_entries() {
        let backend = MetalBackend::new().unwrap();
        let signature = KernelSignature::new(1, vec![], vec![DType::F32], 0);
        {
            let mut prepared = backend.prepared_programs.lock().unwrap();
            for value in 0_u16..1_000 {
                let program = Program {
                    kind: ProgramKind::Map,
                    insts: vec![Inst::Const(f32::from(value))],
                    outputs: vec![(0, 0)],
                }
                .validate()
                .unwrap();
                prepared.insert(
                    (program.content_hash(), signature.clone()),
                    Weak::<MetalProgram>::new(),
                );
            }
            assert_eq!(prepared.len(), 1_000);
        }
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Const(-1.0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let _prepared = prepare_program(&backend, program, signature).unwrap();

        assert_eq!(backend.prepared_programs.lock().unwrap().len(), 1);
    }

    #[test]
    fn submitted_dispatches_retain_prepared_pipelines_until_completion() {
        let backend = MetalBackend::new().unwrap();
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Const(1.0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(
            &backend,
            program,
            KernelSignature::new(1, vec![], vec![DType::F32], 0),
        )
        .unwrap();
        let pipeline = Arc::downgrade(&prepared.backend_handle::<MetalProgramHandle>().unwrap().0);
        let output = backend.alloc(DType::F32, &[33]).unwrap();
        let mut commands = CommandList::new();
        for _ in 0..100 {
            commands
                .dispatch_kernel(&prepared, &[], &[&output])
                .unwrap();
        }
        drop(prepared);
        let submission = backend.submit(commands).unwrap();
        assert!(pipeline.upgrade().is_some());
        drop(submission);
        assert_eq!(
            backend.read(&output).unwrap(),
            [1.0_f32.to_le_bytes(); 33].concat()
        );
        assert!(pipeline.upgrade().is_none());
    }

    #[test]
    fn prepared_row_program_matches_cpu() {
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Reduce(RedOp::Sum, 0),
                Inst::Binary(BinOp::Div, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
        .validate()
        .unwrap();
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let expected = run_row_program(&cpu, &program, 33);
        let actual = run_row_program(&metal, &program, 33);

        assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
    }

    #[test]
    fn map_pipeline_compile_failure_is_reported() {
        let cpu = CpuBackend::new();
        let program = identity_program(&cpu, DType::F32, DType::F32, &[7], false);
        let backend = MetalBackend::new().unwrap();
        let cache = backend.pipelines.lock().unwrap();
        let key = program_pipeline_key(&program, false);
        assert!(matches!(
            cache.compile_program_source(&key, "kernel void broken("),
            Err(BackendError::ExecutionFailed)
        ));
    }

    #[test]
    fn metal_executes_map_program_with_multiple_outputs() {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let expected = run_two_output_program(&cpu);
        let actual = run_two_output_program(&metal);
        for (expected, actual) in expected.iter().zip(actual) {
            assert_outputs_agree(DType::F32, expected, &actual).unwrap();
        }
    }

    #[test]
    fn metal_executes_dependent_row_reductions() {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Reduce(RedOp::Sum, 0),
                Inst::Binary(BinOp::Div, 0, 1),
                Inst::Reduce(RedOp::Max, 2),
                Inst::Binary(BinOp::Add, 2, 3),
            ],
            outputs: vec![(0, 4)],
        }
        .validate()
        .unwrap();
        let expected = run_single_program(&cpu, &program);
        let actual = run_single_program(&metal, &program);
        assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
    }

    #[test]
    fn register_heavy_rows_cover_every_element_at_residency_boundary() {
        let cpu = CpuBackend::new();
        let metal = MetalBackend::new().unwrap();
        let program = register_heavy_row_program();

        for width in [1024_u32, 1025] {
            let expected = run_row_program(&cpu, &program, width);
            let actual = run_row_program(&metal, &program, width);
            let (expected, expected_remainder) = expected.as_chunks::<4>();
            let (actual, actual_remainder) = actual.as_chunks::<4>();
            assert!(expected_remainder.is_empty());
            assert!(actual_remainder.is_empty());
            for (expected, actual) in expected.iter().zip(actual) {
                assert_outputs_agree(DType::F32, expected, actual).unwrap();
            }
        }
    }

    fn run_single_program<B: Backend>(backend: &B, program: &ValidatedProgram) -> Vec<u8> {
        let input = backend.alloc(DType::F32, &[2, 33]).unwrap();
        let bytes = (0_u16..66)
            .flat_map(|value| (f32::from(value) / 16.0 - 2.0).to_le_bytes())
            .collect::<Vec<_>>();
        backend.write(&input, &bytes).unwrap();
        let output = backend.alloc(DType::F32, &[2, 33]).unwrap();
        let prepared = prepare_program(
            backend,
            program.clone(),
            KernelSignature::new(2, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&input], &[&output])
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        backend.read(&output).unwrap()
    }

    fn run_row_program<B: Backend>(backend: &B, program: &ValidatedProgram, width: u32) -> Vec<u8> {
        let input = backend.alloc(DType::F32, &[width]).unwrap();
        let bytes = (0..width)
            .flat_map(|index| {
                let value = f32::from(u16::try_from(index % 17).unwrap()) / 64.0 - 0.125;
                value.to_le_bytes()
            })
            .collect::<Vec<_>>();
        backend.write(&input, &bytes).unwrap();
        let output = backend.alloc(DType::F32, &[width]).unwrap();
        let prepared = prepare_program(
            backend,
            program.clone(),
            KernelSignature::new(1, vec![DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&input], &[&output])
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        backend.read(&output).unwrap()
    }

    fn register_heavy_row_program() -> ValidatedProgram {
        let mut insts = vec![Inst::Input(0)];
        let mut values = Vec::with_capacity(85);
        for coefficient in 1_u16..=85 {
            let constant = u32::try_from(insts.len()).unwrap();
            insts.push(Inst::Const(f32::from(coefficient) / 128.0));
            let product = u32::try_from(insts.len()).unwrap();
            insts.push(Inst::Binary(BinOp::Mul, 0, constant));
            values.push(product);
        }
        let mut combined = values[0];
        for value in values.into_iter().skip(1) {
            let sum = u32::try_from(insts.len()).unwrap();
            insts.push(Inst::Binary(BinOp::Add, combined, value));
            combined = sum;
        }
        let reduced = u32::try_from(insts.len()).unwrap();
        insts.push(Inst::Reduce(RedOp::Sum, combined));
        assert_eq!(insts.len(), forja_core::program::MAX_INSTRUCTIONS);
        Program {
            kind: ProgramKind::Row,
            insts,
            outputs: vec![(0, reduced)],
        }
        .validate()
        .unwrap()
    }

    fn run_two_output_program<B: Backend>(backend: &B) -> [Vec<u8>; 2] {
        let left = backend.alloc(DType::F32, &[7]).unwrap();
        let right = backend.alloc(DType::F32, &[7]).unwrap();
        let left_bytes = (0_u32..7)
            .flat_map(|value| f32::from(u16::try_from(value).unwrap()).to_le_bytes())
            .collect::<Vec<_>>();
        let right_bytes = (0_u32..7)
            .flat_map(|value| (f32::from(u16::try_from(value).unwrap()) / 2.0).to_le_bytes())
            .collect::<Vec<_>>();
        backend.write(&left, &left_bytes).unwrap();
        backend.write(&right, &right_bytes).unwrap();
        let sum = backend.alloc(DType::F32, &[7]).unwrap();
        let difference = backend.alloc(DType::F32, &[7]).unwrap();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Add, 0, 1),
                Inst::Binary(BinOp::Sub, 0, 1),
            ],
            outputs: vec![(0, 2), (1, 3)],
        }
        .validate()
        .unwrap();
        let prepared = prepare_program(
            backend,
            program,
            KernelSignature::new(
                1,
                vec![DType::F32, DType::F32],
                vec![DType::F32, DType::F32],
                0,
            ),
        )
        .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_kernel(&prepared, &[&left, &right], &[&sum, &difference])
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        [
            backend.read(&sum).unwrap(),
            backend.read(&difference).unwrap(),
        ]
    }

    fn identity_program(
        backend: &CpuBackend,
        input_dtype: DType,
        output_dtype: DType,
        shape: &[u32],
        permuted: bool,
    ) -> BoundProgram {
        let input = if permuted {
            let allocation = backend.alloc(input_dtype, &[shape[1], shape[0]]).unwrap();
            backend
                .view(&allocation, ViewOp::Permute(vec![1, 0]))
                .unwrap()
        } else {
            backend.alloc(input_dtype, shape).unwrap()
        };
        let output = backend.alloc(output_dtype, shape).unwrap();
        let program = Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap();
        bind_program(&program, &[&input], &[&output]).unwrap()
    }

    fn row_program(backend: &CpuBackend, width: u32) -> BoundProgram {
        let input = backend.alloc(DType::F32, &[width]).unwrap();
        let output = backend.alloc(DType::F32, &[width]).unwrap();
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Input(0), Inst::Reduce(RedOp::Sum, 0)],
            outputs: vec![(0, 1)],
        }
        .validate()
        .unwrap();
        bind_program(&program, &[&input], &[&output]).unwrap()
    }

    fn program_pipeline_key(program: &BoundProgram, resident: bool) -> ProgramPipelineKey {
        let signature = KernelSignature::new(
            u8::try_from(program.outputs()[0].layout().shape().len()).unwrap(),
            program
                .inputs()
                .iter()
                .map(|tensor| tensor.layout().dtype())
                .collect(),
            program
                .outputs()
                .iter()
                .map(|tensor| tensor.layout().dtype())
                .collect(),
            0,
        );
        ProgramPipelineKey::new(program.program(), &signature, resident)
    }

    fn run_varying_submissions(backend: &MetalBackend, count: usize, offset: usize) {
        let widths = [1_u32, 7, 33, 4097];
        for sequence in offset..offset + count {
            let width = widths[sequence % widths.len()];
            let len = usize::try_from(width).unwrap();
            let left = backend.alloc(DType::F32, &[width]).unwrap();
            let right = backend.alloc(DType::F32, &[width]).unwrap();
            let output = backend.alloc(DType::F32, &[width]).unwrap();
            let left_bytes = vec![1.0_f32; len]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let right_bytes = vec![2.0_f32; len]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            backend.write(&left, &left_bytes).unwrap();
            backend.write(&right, &right_bytes).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::Add, &[&left, &right], &output)
                .unwrap();
            let submission = backend.submit(commands).unwrap();
            if sequence % 2 == 0 {
                submission.wait().unwrap();
            } else {
                drop(submission);
            }
            let actual = backend.read(&output).unwrap();
            let expected = vec![3.0_f32; len]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
            backend.release(&left).unwrap();
            backend.release(&right).unwrap();
            backend.release(&output).unwrap();
        }
    }

    #[test]
    fn pooled_submission_objects_survive_waits_and_drops() {
        let backend = MetalBackend::new().unwrap();
        run_varying_submissions(&backend, 1000, 0);
        backend.in_flight.drain_done();
        assert_eq!(backend.in_flight.len(), 0);
        assert_eq!(backend.in_flight.pooled_len(), 1);
        let pool = backend.in_flight.pool.lock().unwrap();
        let capacity = pool[0].argument_buffer.capacity;
        let address = pool[0].argument_buffer.raw.gpuAddress();
        drop(pool);
        let objects = backend
            .in_flight
            .checkout(&backend.device, capacity)
            .unwrap();
        assert_eq!(objects.argument_buffer().unwrap().raw.gpuAddress(), address);
    }

    #[test]
    fn command_resources_reject_unowned_and_unwritten_bindings() {
        let backend = MetalBackend::new().unwrap();
        let raw = backend
            .device
            .newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let base = raw.gpuAddress();
        let unowned = EncodedDispatches {
            temporaries: Vec::new(),
            error_flags: Vec::new(),
            bindings: ArgumentBindings {
                ranges: HashSet::from([BoundRange {
                    base,
                    offset: 0,
                    len: 8,
                    address: base,
                }]),
            },
            arguments: None,
            program_encoding: ProfileCount::default(),
            program_compile_fallbacks: 0,
        };
        assert!(matches!(
            backend.command_resources(&[], unowned, None),
            Err(BackendError::ExecutionFailed)
        ));

        let unwritten = EncodedDispatches {
            temporaries: vec![BufferBinding::whole(raw)],
            error_flags: Vec::new(),
            bindings: ArgumentBindings {
                ranges: HashSet::from([BoundRange {
                    base,
                    offset: 16,
                    len: 8,
                    address: base + 16,
                }]),
            },
            arguments: Some(ArgumentUsage {
                base,
                written: HashSet::from([(0, 8)]),
            }),
            program_encoding: ProfileCount::default(),
            program_compile_fallbacks: 0,
        };
        assert!(matches!(
            backend.command_resources(&[], unwritten, None),
            Err(BackendError::ExecutionFailed)
        ));
    }

    #[test]
    fn decomposed_attention_submission_uses_exact_argument_capacity() {
        let _override = SdpaOverride::set(SdpaKernel::Decomposed);
        let backend = MetalBackend::new().unwrap();
        let mut commands = CommandList::new();
        let mut tensors = Vec::new();
        for _ in 0..4 {
            let query = backend.alloc(DType::F32, &[16, 7, 128]).unwrap();
            let key_cache = backend.alloc(DType::F32, &[8, 33, 256]).unwrap();
            let value_cache = backend.alloc(DType::F32, &[8, 33, 256]).unwrap();
            let output = backend.alloc(DType::F32, &[16, 7, 128]).unwrap();
            let slices = vec![
                Slice::new(0, 8, 1).unwrap(),
                Slice::new(0, 33, 1).unwrap(),
                Slice::new(0, 128, 2).unwrap(),
            ];
            let key = backend
                .view(&key_cache, ViewOp::Slice(slices.clone()))
                .unwrap();
            let value = backend.view(&value_cache, ViewOp::Slice(slices)).unwrap();
            commands
                .dispatch(
                    Op::Sdpa {
                        scale: 128.0_f32.sqrt().recip(),
                        causal: true,
                        q_start: 0,
                    },
                    &[&query, &key, &value],
                    &output,
                )
                .unwrap();
            tensors.extend([query, key_cache, value_cache, output]);
        }
        let capacity = backend
            .argument_capacity(&commands.clone().into_dispatches())
            .unwrap();
        assert!(capacity > 4096);
        backend.submit(commands).unwrap().wait().unwrap();
        backend.in_flight.drain_done();
        assert_eq!(
            backend.in_flight.pool.lock().unwrap()[0]
                .argument_buffer
                .capacity,
            capacity
        );
        drop(tensors);
    }

    #[test]
    fn concurrent_submissions_share_one_backend() {
        let backend = Arc::new(MetalBackend::new().unwrap());
        let workers = (0..8)
            .map(|worker| {
                let backend = Arc::clone(&backend);
                thread::spawn(move || run_varying_submissions(&backend, 200, worker * 200))
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
        let started = Instant::now();
        while backend.in_flight.len() != 0 {
            assert!(started.elapsed() < backend.gpu_timeout);
            thread::yield_now();
        }
        backend.in_flight.drain_done();
        assert!(backend.in_flight.pooled_len() >= 1);
    }

    #[test]
    fn metal_calls_reject_callback_context() {
        let result = std::panic::catch_unwind(|| {
            let _scope = MetalCallbackScope::enter();
            assert_not_in_metal_callback();
        });
        assert!(result.is_err());
    }

    #[test]
    fn metal_empty_command_list_completes() {
        let backend = MetalBackend::new().unwrap();
        backend.submit(CommandList::new()).unwrap().wait().unwrap();
    }

    #[test]
    fn metal_submission_reports_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let shape = [1024, 4097];
        let gate = backend.alloc(DType::F32, &shape).unwrap();
        let up = backend.alloc(DType::F32, &shape).unwrap();
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::SiluMul, &[&gate, &up], &output)
            .unwrap();

        let started = Instant::now();
        let submission = backend.submit(commands).unwrap();
        submission.wait().unwrap();
        let wall_time = started.elapsed();
        let gpu_time = submission.gpu_time().unwrap();

        assert!(gpu_time > Duration::ZERO);
        assert!(
            gpu_time <= wall_time,
            "GPU {gpu_time:?}, wall {wall_time:?}"
        );
    }

    #[test]
    fn profiled_submission_reports_phases_and_dispatches() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let right = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Add, &[&left, &right], &output)
            .unwrap();
        let submission = backend.submit_profiled(commands).unwrap();
        submission.wait().unwrap();
        let profile = submission.profile().unwrap();
        assert_eq!(profile.dispatches, 1);
        assert_eq!(profile.per_dispatch.len(), 1);
        assert_eq!(profile.per_dispatch[0].op, Op::Add);
        assert!(profile.gpu_time > Duration::ZERO);
        assert!(profile.per_dispatch[0].gpu_time > Duration::ZERO);
        assert_eq!(profile.metadata_buffers.count, 1);
    }

    #[test]
    fn prepared_graph_owns_one_persistent_residency_set() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let right = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        graph
            .dispatch(Op::Add, &[&left.into(), &right.into()], &output.into())
            .unwrap();

        let graph = backend.prepare_graph(graph).unwrap();
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        assert_eq!(state.residency.allocationCount(), 3);
        assert!(
            state
                .encoding
                .as_ref()
                .unwrap()
                .dispatches
                .iter()
                .any(Option::is_some)
        );
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
    }

    #[test]
    fn tier_two_replays_argmax_with_dynamic_scratch() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .dispatch(Op::Argmax, &[&input.clone().into()], &output.clone().into())
            .unwrap();

        let graph = backend.prepare_graph(template).unwrap();
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        assert!(state.encoding.as_ref().unwrap().dispatches[0].is_none());
        let first = backend.replay(&graph, Vec::new()).unwrap();
        let second = backend.replay(&graph, Vec::new()).unwrap();
        second.wait().unwrap();
        first.wait().unwrap();
        assert_eq!(backend.read(&output).unwrap(), 4096_u32.to_le_bytes());
    }

    #[test]
    fn consecutive_replays_preserve_feedback() {
        const REPLAYS: usize = 128;

        let backend = MetalBackend::new().unwrap();
        let feedback = backend.alloc(DType::F32, &[1]).unwrap();
        let staged = backend.alloc(DType::F32, &[1]).unwrap();
        let one = backend.alloc(DType::F32, &[1]).unwrap();
        backend.write(&one, &1.0_f32.to_le_bytes()).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .dispatch(
                Op::Copy,
                &[&feedback.clone().into()],
                &staged.clone().into(),
            )
            .unwrap();
        template
            .dispatch(
                Op::Add,
                &[&staged.into(), &one.into()],
                &feedback.clone().into(),
            )
            .unwrap();
        let graph = backend.prepare_graph(template).unwrap();

        let mut submissions = VecDeque::with_capacity(2);
        for _ in 0..REPLAYS {
            submissions.push_back(backend.replay(&graph, Vec::new()).unwrap());
            if submissions.len() == 2 {
                submissions.pop_front().unwrap().wait().unwrap();
            }
        }
        for submission in submissions {
            submission.wait().unwrap();
        }
        assert_eq!(backend.read(&feedback).unwrap(), 128.0_f32.to_le_bytes());
    }

    #[test]
    fn tier_two_replay_checks_mixed_plan_argument_capacity() {
        let backend = MetalBackend::new().unwrap();
        let static_input = backend.alloc(DType::F32, &[7]).unwrap();
        let static_output = backend.alloc(DType::F32, &[7]).unwrap();
        let dynamic_input = backend.alloc(DType::F32, &[7]).unwrap();
        let dynamic_output = backend.alloc(DType::F32, &[7]).unwrap();
        let space = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let dynamic_input =
            symbolic_prefix(&dynamic_input, space.clone(), Affine::parameter(0, 0, 1));
        let dynamic_output =
            symbolic_prefix(&dynamic_output, space.clone(), Affine::parameter(0, 0, 1));
        let mut template = GraphTemplate::new(space, GraphLimits::default());
        template
            .dispatch(
                Op::Copy,
                &[&TemplateTensor::from(static_input)],
                &TemplateTensor::from(static_output),
            )
            .unwrap();
        template
            .dispatch(Op::Copy, &[&dynamic_input], &dynamic_output)
            .unwrap();

        let graph = backend.prepare_graph(template.clone()).unwrap();
        backend.replay(&graph, vec![7]).unwrap().wait().unwrap();

        let mut state = backend.prepare_metal_graph(&template).unwrap();
        state.encoding.as_mut().unwrap().argument_offset += 16;
        let corrupted = PreparedGraph::with_backend_state(template, state);
        assert!(matches!(
            backend.replay(&corrupted, vec![7]),
            Err(BackendError::ExecutionFailed)
        ));
    }

    #[test]
    fn static_plan_calls_are_deduplicated_within_dynamic_boundaries() {
        let backend = MetalBackend::new().unwrap();
        let static_input = backend.alloc(DType::F32, &[7]).unwrap();
        let static_output = backend.alloc(DType::F32, &[7]).unwrap();
        let dynamic_input = backend.alloc(DType::F32, &[7]).unwrap();
        let dynamic_output = backend.alloc(DType::F32, &[7]).unwrap();
        let space = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let dynamic_input =
            symbolic_prefix(&dynamic_input, space.clone(), Affine::parameter(0, 0, 1));
        let dynamic_output =
            symbolic_prefix(&dynamic_output, space.clone(), Affine::parameter(0, 0, 1));
        let input = TemplateTensor::from(static_input);
        let output = TemplateTensor::from(static_output);
        let mut template = GraphTemplate::new(space, GraphLimits::default());
        template.dispatch(Op::Copy, &[&input], &output).unwrap();
        template.dispatch(Op::Copy, &[&input], &output).unwrap();
        template
            .dispatch(Op::Copy, &[&dynamic_input], &dynamic_output)
            .unwrap();
        template.dispatch(Op::Copy, &[&input], &output).unwrap();

        let state = backend.prepare_metal_graph(&template).unwrap();
        let plans = &state.encoding.as_ref().unwrap().dispatches;
        let has_pipeline = |index: usize| {
            plans[index]
                .as_ref()
                .unwrap()
                .calls
                .iter()
                .any(|call| matches!(call, PlanCall::Pipeline(_)))
        };
        let has_table = |index: usize| {
            plans[index]
                .as_ref()
                .unwrap()
                .calls
                .iter()
                .any(|call| matches!(call, PlanCall::ArgumentTable))
        };
        assert!(has_pipeline(0) && has_table(0));
        assert!(!has_pipeline(1) && !has_table(1));
        assert!(plans[2].is_none());
        assert!(has_pipeline(3) && has_table(3));
    }

    #[test]
    fn replay_failure_does_not_consume_the_graph() {
        let backend = MetalBackend::new().unwrap();
        let table = backend.alloc(DType::F32, &[6, 7]).unwrap();
        let ids = backend.alloc(DType::U32, &[1]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 7]).unwrap();
        backend.write(&ids, &99_u32.to_le_bytes()).unwrap();
        let table_template = table.clone().into();
        let ids_template = ids.clone().into();
        let output_template = output.clone().into();
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        graph
            .dispatch(
                Op::Embed,
                &[&table_template, &ids_template],
                &output_template,
            )
            .unwrap();
        let graph = backend.prepare_graph(graph).unwrap();

        assert_eq!(
            backend.replay(&graph, Vec::new()).unwrap().wait(),
            Err(BackendError::IndexOutOfRange { index: 99 })
        );
        backend.write(&ids, &1_u32.to_le_bytes()).unwrap();
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
    }

    #[test]
    fn failed_replay_taints_later_graph_reads() {
        let backend = MetalBackend::new().unwrap();
        let table = backend.alloc(DType::F32, &[1, 7]).unwrap();
        let invalid = backend.alloc(DType::U32, &[1]).unwrap();
        let shared = backend.alloc(DType::F32, &[1, 7]).unwrap();
        let sink = backend.alloc(DType::F32, &[1, 7]).unwrap();
        backend.write(&invalid, &99_u32.to_le_bytes()).unwrap();
        let mut failing =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        failing
            .dispatch(
                Op::Embed,
                &[&table.into(), &invalid.into()],
                &shared.clone().into(),
            )
            .unwrap();
        let failing = backend.prepare_graph(failing).unwrap();
        let mut consumer =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        consumer
            .dispatch(Op::Copy, &[&shared.into()], &sink.into())
            .unwrap();
        let consumer = backend.prepare_graph(consumer).unwrap();

        let failed = backend.replay(&failing, Vec::new()).unwrap();
        let same = backend.replay(&failing, Vec::new()).unwrap();
        let shared = backend.replay(&consumer, Vec::new()).unwrap();
        assert_eq!(
            failed.wait(),
            Err(BackendError::IndexOutOfRange { index: 99 })
        );
        assert_eq!(
            same.wait(),
            Err(BackendError::IndexOutOfRange { index: 99 })
        );
        assert_eq!(
            shared.wait(),
            Err(BackendError::IndexOutOfRange { index: 99 })
        );
    }

    #[test]
    fn dropped_submissions_retain_released_buffers() {
        let backend = MetalBackend::new().unwrap();
        for _ in 0..100 {
            let gate = backend.alloc(DType::F32, &[4097]).unwrap();
            let up = backend.alloc(DType::F32, &[4097]).unwrap();
            let output = backend.alloc(DType::F32, &[4097]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::SiluMul, &[&gate, &up], &output)
                .unwrap();
            let submission = backend.submit(commands).unwrap();
            drop(submission);
            backend.release(&gate).unwrap();
            backend.release(&up).unwrap();
            backend.release(&output).unwrap();
        }
        drop(backend);
    }

    #[test]
    fn timed_out_work_never_exposes_an_active_output() {
        let backend = MetalBackend::with_gpu_timeout(Duration::from_micros(1)).unwrap();
        let shape = [1024, 4097];
        let gate = backend.alloc(DType::F32, &shape).unwrap();
        let up = backend.alloc(DType::F32, &shape).unwrap();
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::SiluMul, &[&gate, &up], &output)
            .unwrap();

        assert_eq!(
            backend.submit(commands).unwrap().wait(),
            Err(BackendError::Timeout)
        );
        match backend.read(&output) {
            Ok(bytes) => assert_eq!(bytes.len(), 1024 * 4097 * 4),
            Err(error) => assert_eq!(error, BackendError::Timeout),
        }
    }

    #[test]
    fn metal_silu_mul_matches_cpu_for_dtypes_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for shape in [&[7, 3072][..], &[1, 3072], &[33, 4097]] {
                let inputs = [
                    TensorSpec::contiguous(dtype, shape),
                    TensorSpec::contiguous(dtype, shape),
                ];
                let output = TensorSpec::contiguous(dtype, shape);
                assert_backends_agree(&reference, &candidate, Op::SiluMul, &inputs, &output)
                    .unwrap();
            }
        }

        let permuted = [
            TensorSpec::permuted(DType::F32, &[3072, 7], &[1, 0]),
            TensorSpec::contiguous(DType::F32, &[7, 3072]),
        ];
        assert_backends_agree(
            &reference,
            &candidate,
            Op::SiluMul,
            &permuted,
            &TensorSpec::contiguous(DType::F32, &[7, 3072]),
        )
        .unwrap();

        let mixed = [
            TensorSpec::contiguous(DType::F16, &[7, 3072]),
            TensorSpec::contiguous(DType::BF16, &[7, 3072]),
        ];
        assert_backends_agree(
            &reference,
            &candidate,
            Op::SiluMul,
            &mixed,
            &TensorSpec::contiguous(DType::F32, &[7, 3072]),
        )
        .unwrap();
    }

    #[test]
    fn metal_copy_matches_cpu_for_casts_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let floats = [DType::F32, DType::F16, DType::BF16];
        for input_dtype in floats {
            for output_dtype in floats {
                for width in [1, 7, 33, 4097] {
                    assert_backends_agree(
                        &reference,
                        &candidate,
                        Op::Copy,
                        &[TensorSpec::contiguous(input_dtype, &[width])],
                        &TensorSpec::contiguous(output_dtype, &[width]),
                    )
                    .unwrap();
                }
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Copy,
                    &[TensorSpec::permuted(input_dtype, &[7, 33], &[1, 0])],
                    &TensorSpec::contiguous(output_dtype, &[33, 7]),
                )
                .unwrap();
            }
        }
        for dtype in [DType::I32, DType::U32] {
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Copy,
                &[TensorSpec::permuted(dtype, &[7, 33], &[1, 0])],
                &TensorSpec::contiguous(dtype, &[33, 7]),
            )
            .unwrap();
        }
    }

    #[test]
    fn scratch_copy_is_visible_to_a_following_consumer() {
        use objc2_metal::{MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages};

        let backend = MetalBackend::new().unwrap();
        let source = backend.alloc(DType::F32, &[33]).unwrap();
        let output = backend.alloc(DType::F32, &[33]).unwrap();
        let bytes = (0_u16..33)
            .map(f32::from)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&source, &bytes).unwrap();

        let mut objects = backend.in_flight.checkout(&backend.device, 4096).unwrap();
        let command_buffer = backend
            .begin_command_buffer(objects.allocator().unwrap())
            .unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        let table = objects.argument_table().unwrap();
        let source_buffer = backend.encoder_tensor(&source).unwrap();
        let output_buffer = backend.encoder_tensor(&output).unwrap();
        let scratch = backend.scratch_tensor(DType::F32, &[33]).unwrap();
        let mut bindings = ArgumentBindings::default();
        let mut arguments = ArgumentWriter::new(objects.argument_buffer().unwrap());
        let mut temporaries = vec![scratch.buffer.clone()];
        temporaries.extend(
            backend
                .encode_copy_tensors(
                    &encoder,
                    table,
                    &source_buffer,
                    &scratch,
                    &mut bindings,
                    &mut arguments,
                )
                .unwrap(),
        );
        encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        temporaries.extend(
            backend
                .encode_copy_tensors(
                    &encoder,
                    table,
                    &scratch,
                    &output_buffer,
                    &mut bindings,
                    &mut arguments,
                )
                .unwrap(),
        );
        temporaries.push(BufferBinding::whole(arguments.raw.clone()));
        encoder.endEncoding();
        let tensors = [source.clone(), output.clone()];
        let resources = backend
            .command_resources(
                &tensors,
                EncodedDispatches {
                    temporaries,
                    error_flags: Vec::new(),
                    bindings,
                    arguments: Some(ArgumentUsage {
                        base: arguments.raw.gpuAddress(),
                        written: arguments.written.iter().copied().collect(),
                    }),
                    program_encoding: ProfileCount::default(),
                    program_compile_fallbacks: 0,
                },
                None,
            )
            .unwrap();
        let residency = backend.make_resident(&command_buffer, &resources).unwrap();
        let timestamps = backend.make_timestamps(2).unwrap();
        // SAFETY: The heap has two entries and remains live through completion.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 0);
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 1);
        }
        command_buffer.endCommandBuffer();
        backend
            .commit(
                &command_buffer,
                &[(source, false), (output.clone(), true)],
                resources,
                CommitResidency {
                    sets: vec![residency],
                    order_on_queue: false,
                },
                timestamps,
                &mut objects,
            )
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(backend.read(&output).unwrap(), bytes);
    }

    fn assert_matmul(a: TensorSpec, b: TensorSpec, output: &TensorSpec) {
        assert_matmul_with(&MetalBackend::new().unwrap(), a, b, output);
    }

    fn assert_matmul_with(
        backend: &MetalBackend,
        a: TensorSpec,
        b: TensorSpec,
        output: &TensorSpec,
    ) {
        let shape = (a.allocation_shape().to_vec(), b.allocation_shape().to_vec());
        let started = Instant::now();
        assert_backends_agree(&CpuBackend::new(), backend, Op::Matmul, &[a, b], output).unwrap();
        let state = backend.debug_state();
        eprintln!(
            "matmul {shape:?}: wall={:?} submit={:?} wait={:?} resources={state:?}",
            started.elapsed(),
            state.last_submit,
            state.last_wait,
        );
    }

    #[test]
    fn repeated_backend_gemms_release_owned_resources() {
        for iteration in 0..200 {
            let started = Instant::now();
            {
                let backend = MetalBackend::new().unwrap();
                assert_matmul_with(
                    &backend,
                    TensorSpec::contiguous(DType::F32, &[7, 33]),
                    TensorSpec::contiguous(DType::F32, &[33, 33]),
                    &TensorSpec::contiguous(DType::F32, &[7, 33]),
                );
                let state = backend.debug_state();
                assert_eq!(state.live_backends, 1);
                assert_eq!(state.live_listener_queues, 1);
                assert_eq!(state.in_flight_submissions, 0);
            }
            assert_eq!(MetalBackend::debug_live_counts(), (0, 0));
            eprintln!("backend iteration {iteration}: {:?}", started.elapsed());
        }
    }

    fn assert_sdpa(
        op: Op,
        query: TensorSpec,
        key: TensorSpec,
        value: TensorSpec,
        output: &TensorSpec,
    ) {
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            op,
            &[query, key, value],
            output,
        )
        .unwrap();
    }

    struct SdpaOverride(Option<SdpaKernel>);

    impl SdpaOverride {
        fn set(kernel: SdpaKernel) -> Self {
            let previous = FORCED_SDPA_KERNEL.with(|forced| forced.replace(Some(kernel)));
            Self(previous)
        }
    }

    impl Drop for SdpaOverride {
        fn drop(&mut self) {
            FORCED_SDPA_KERNEL.with(|forced| forced.set(self.0));
        }
    }

    fn assert_sdpa_with(
        kernel: SdpaKernel,
        op: Op,
        query: TensorSpec,
        key: TensorSpec,
        value: TensorSpec,
        output: &TensorSpec,
    ) {
        let _override = SdpaOverride::set(kernel);
        assert_sdpa(op, query, key, value, output);
    }

    #[test]
    fn metal_decomposed_attention_matches_hand_sized_cases() {
        assert_sdpa_with(
            SdpaKernel::Decomposed,
            Op::Sdpa {
                scale: 1.0,
                causal: false,
                q_start: 0,
            },
            TensorSpec::contiguous(DType::F32, &[1, 1, 2]),
            TensorSpec::contiguous(DType::F32, &[1, 2, 2]),
            TensorSpec::contiguous(DType::F32, &[1, 2, 1]),
            &TensorSpec::contiguous(DType::F32, &[1, 1, 1]),
        );
    }

    #[test]
    fn metal_decomposed_attention_matches_grouped_heads() {
        assert_sdpa_with(
            SdpaKernel::Decomposed,
            Op::Sdpa {
                scale: 0.5,
                causal: false,
                q_start: 0,
            },
            TensorSpec::contiguous(DType::F32, &[4, 3, 4]),
            TensorSpec::contiguous(DType::F32, &[2, 7, 4]),
            TensorSpec::contiguous(DType::F32, &[2, 7, 3]),
            &TensorSpec::contiguous(DType::F32, &[4, 3, 3]),
        );
    }

    #[test]
    fn causal_attention_first_row_is_finite_on_every_path() {
        for (kernel, key_length) in [
            (SdpaKernel::Vector, 33),
            (SdpaKernel::Vector, 1024),
            (SdpaKernel::Steel, 33),
            (SdpaKernel::Decomposed, 33),
        ] {
            assert_sdpa_with(
                kernel,
                Op::Sdpa {
                    scale: 0.125,
                    causal: true,
                    q_start: 0,
                },
                TensorSpec::contiguous(DType::F32, &[4, 1, 64]),
                TensorSpec::contiguous(DType::F32, &[2, key_length, 64]),
                TensorSpec::contiguous(DType::F32, &[2, key_length, 64]),
                &TensorSpec::contiguous(DType::F32, &[4, 1, 64]),
            );
        }
    }

    #[test]
    fn metal_attention_reads_strided_qwen_cache_views() {
        let cache_slice = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(0, 7, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            assert_sdpa_with(
                SdpaKernel::Decomposed,
                Op::Sdpa {
                    scale: 128.0_f32.sqrt().recip(),
                    causal: true,
                    q_start: 0,
                },
                TensorSpec::contiguous(dtype, &[16, 7, 128]),
                TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                &TensorSpec::contiguous(dtype, &[16, 7, 128]),
            );
        }
    }

    #[test]
    fn metal_vector_attention_matches_short_decode() {
        let _override = SdpaOverride::set(SdpaKernel::Vector);
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for key_length in [1, 37] {
                let cache_slice = [
                    Slice::new(0, 8, 1).unwrap(),
                    Slice::new(0, key_length, 1).unwrap(),
                    Slice::new(0, 128, 1).unwrap(),
                ];
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Sdpa {
                        scale: 128.0_f32.sqrt().recip(),
                        causal: true,
                        q_start: key_length - 1,
                    },
                    &[
                        TensorSpec::contiguous(dtype, &[16, 1, 128]),
                        TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                        TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                    ],
                    &TensorSpec::contiguous(dtype, &[16, 1, 128]),
                )
                .unwrap();
            }
        }
        assert_backends_agree(
            &reference,
            &candidate,
            Op::Sdpa {
                scale: 0.125,
                causal: true,
                q_start: 29,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[4, 8, 64]),
                TensorSpec::contiguous(DType::F32, &[2, 37, 64]),
                TensorSpec::contiguous(DType::F32, &[2, 37, 64]),
            ],
            &TensorSpec::contiguous(DType::F32, &[4, 8, 64]),
        )
        .unwrap();
    }

    #[test]
    fn metal_vector_attention_matches_two_pass_decode() {
        let _override = SdpaOverride::set(SdpaKernel::Vector);
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for key_length in [1024, 4096] {
                let cache_slice = [
                    Slice::new(0, 8, 1).unwrap(),
                    Slice::new(0, key_length, 1).unwrap(),
                    Slice::new(0, 128, 1).unwrap(),
                ];
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Sdpa {
                        scale: 128.0_f32.sqrt().recip(),
                        causal: true,
                        q_start: key_length - 1,
                    },
                    &[
                        TensorSpec::contiguous(dtype, &[16, 1, 128]),
                        TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                        TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                    ],
                    &TensorSpec::contiguous(dtype, &[16, 1, 128]),
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn metal_decode_matches_the_last_prefill_row() {
        let backend = MetalBackend::new().unwrap();
        let key_length = 37_u32;
        let query = backend.alloc(DType::F32, &[16, key_length, 128]).unwrap();
        let key_cache = backend.alloc(DType::F32, &[8, 4096, 128]).unwrap();
        let value_cache = backend.alloc(DType::F32, &[8, 4096, 128]).unwrap();
        let query_values = (0_u16..31)
            .cycle()
            .take(usize::try_from(16 * key_length * 128).unwrap())
            .map(|index| f32::from(index) / 31.0 - 0.5)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let cache_values = (0_u16..37)
            .cycle()
            .take(8 * 4096 * 128)
            .map(|index| f32::from(index) / 37.0 - 0.5)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&query, &query_values).unwrap();
        backend.write(&key_cache, &cache_values).unwrap();
        backend.write(&value_cache, &cache_values).unwrap();
        let cache_slice = ViewOp::Slice(vec![
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(0, key_length, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ]);
        let key = backend.view(&key_cache, cache_slice.clone()).unwrap();
        let value = backend.view(&value_cache, cache_slice).unwrap();
        let decode_query = backend
            .view(
                &query,
                ViewOp::Slice(vec![
                    Slice::new(0, 16, 1).unwrap(),
                    Slice::new(key_length - 1, 1, 1).unwrap(),
                    Slice::new(0, 128, 1).unwrap(),
                ]),
            )
            .unwrap();
        let prefill = backend.alloc(DType::F32, &[16, key_length, 128]).unwrap();
        let decode = backend.alloc(DType::F32, &[16, 1, 128]).unwrap();
        for (query, output, q_start) in [
            (&query, &prefill, 0),
            (&decode_query, &decode, key_length - 1),
        ] {
            let _override = (q_start > 0).then(|| SdpaOverride::set(SdpaKernel::Vector));
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::Sdpa {
                        scale: 128.0_f32.sqrt().recip(),
                        causal: true,
                        q_start,
                    },
                    &[query, &key, &value],
                    output,
                )
                .unwrap();
            backend.submit(commands).unwrap().wait().unwrap();
        }
        let prefill_row = backend
            .view(
                &prefill,
                ViewOp::Slice(vec![
                    Slice::new(0, 16, 1).unwrap(),
                    Slice::new(key_length - 1, 1, 1).unwrap(),
                    Slice::new(0, 128, 1).unwrap(),
                ]),
            )
            .unwrap();
        assert_outputs_agree(
            DType::F32,
            &backend.read(&prefill_row).unwrap(),
            &backend.read(&decode).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn metal_steel_attention_matches_prefill_shapes() {
        for (dtype, length, causal) in [
            (DType::BF16, 7, false),
            (DType::BF16, 128, false),
            (DType::BF16, 512, true),
        ] {
            let cache_slice = [
                Slice::new(0, 8, 1).unwrap(),
                Slice::new(0, length, 1).unwrap(),
                Slice::new(0, 128, 1).unwrap(),
            ];
            assert_sdpa_with(
                SdpaKernel::Steel,
                Op::Sdpa {
                    scale: 128.0_f32.sqrt().recip(),
                    causal,
                    q_start: 0,
                },
                TensorSpec::contiguous(dtype, &[16, length, 128]),
                TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                TensorSpec::sliced(dtype, &[8, 4096, 128], &cache_slice),
                &TensorSpec::contiguous(dtype, &[16, length, 128]),
            );
        }
        assert_sdpa_with(
            SdpaKernel::Steel,
            Op::Sdpa {
                scale: 0.125,
                causal: false,
                q_start: 0,
            },
            TensorSpec::contiguous(DType::F32, &[4, 128, 64]),
            TensorSpec::contiguous(DType::F32, &[2, 128, 64]),
            TensorSpec::contiguous(DType::F32, &[2, 128, 64]),
            &TensorSpec::contiguous(DType::F32, &[4, 128, 64]),
        );
    }

    #[test]
    fn metal_steel_attention_matches_chunked_prefill() {
        let cache_slice = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(0, 512, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        assert_sdpa_with(
            SdpaKernel::Steel,
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: 384,
            },
            TensorSpec::contiguous(DType::BF16, &[16, 128, 128]),
            TensorSpec::sliced(DType::BF16, &[8, 4096, 128], &cache_slice),
            TensorSpec::sliced(DType::BF16, &[8, 4096, 128], &cache_slice),
            &TensorSpec::contiguous(DType::BF16, &[16, 128, 128]),
        );
        assert_sdpa_with(
            SdpaKernel::Steel,
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: 4,
            },
            TensorSpec::contiguous(DType::BF16, &[16, 33, 128]),
            TensorSpec::contiguous(DType::BF16, &[8, 37, 128]),
            TensorSpec::contiguous(DType::BF16, &[8, 37, 128]),
            &TensorSpec::contiguous(DType::BF16, &[16, 33, 128]),
        );
        assert_sdpa_with(
            SdpaKernel::Steel,
            Op::Sdpa {
                scale: 0.125,
                causal: true,
                q_start: 32,
            },
            TensorSpec::contiguous(DType::F32, &[4, 1, 64]),
            TensorSpec::contiguous(DType::F32, &[2, 33, 64]),
            TensorSpec::contiguous(DType::F32, &[2, 33, 64]),
            &TensorSpec::contiguous(DType::F32, &[4, 1, 64]),
        );
    }

    #[test]
    fn metal_attention_decomposes_unsupported_head_widths() {
        assert_sdpa(
            Op::Sdpa {
                scale: 80.0_f32.sqrt().recip(),
                causal: true,
                q_start: 32,
            },
            TensorSpec::contiguous(DType::BF16, &[4, 1, 80]),
            TensorSpec::contiguous(DType::BF16, &[2, 33, 80]),
            TensorSpec::contiguous(DType::BF16, &[2, 33, 80]),
            &TensorSpec::contiguous(DType::BF16, &[4, 1, 80]),
        );
    }

    #[test]
    fn metal_matrix_multiplication_matches_cpu_at_tile_edges_and_in_batches() {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (m, n, k) in [
                (1, 1, 1),
                (1, 4097, 33),
                (33, 1, 4097),
                (4097, 33, 1),
                (33, 33, 33),
            ] {
                assert_matmul(
                    TensorSpec::contiguous(dtype, &[m, k]),
                    TensorSpec::contiguous(dtype, &[k, n]),
                    &TensorSpec::contiguous(dtype, &[m, n]),
                );
            }
            assert_matmul(
                TensorSpec::contiguous(dtype, &[16, 7, 33]),
                TensorSpec::contiguous(dtype, &[16, 33, 33]),
                &TensorSpec::contiguous(dtype, &[16, 7, 33]),
            );
        }
    }

    #[test]
    fn metal_gemm_matches_qwen_projection_shapes() {
        let backend = MetalBackend::new().unwrap();
        for m in [7, 128, 512] {
            for (k, n) in [
                (1024, 2048),
                (1024, 1024),
                (2048, 1024),
                (1024, 3072),
                (3072, 1024),
            ] {
                assert_matmul_with(
                    &backend,
                    TensorSpec::contiguous(DType::BF16, &[m, k]),
                    TensorSpec::permuted(DType::BF16, &[n, k], &[1, 0]),
                    &TensorSpec::contiguous(DType::BF16, &[m, n]),
                );
            }
        }
    }

    #[test]
    fn metal_gemv_matches_qwen_decode_shapes() {
        for (inner, columns) in [
            (1024, 2048),
            (1024, 1024),
            (2048, 1024),
            (1024, 3072),
            (3072, 1024),
            (1024, 151_936),
        ] {
            assert_matmul(
                TensorSpec::contiguous(DType::BF16, &[1, inner]),
                TensorSpec::permuted(DType::BF16, &[columns, inner], &[1, 0]),
                &TensorSpec::contiguous(DType::BF16, &[1, columns]),
            );
        }
        assert_matmul(
            TensorSpec::contiguous(DType::BF16, &[1, 33]),
            TensorSpec::permuted(DType::BF16, &[4097, 33], &[1, 0]),
            &TensorSpec::contiguous(DType::BF16, &[1, 4097]),
        );
    }

    #[test]
    fn metal_gemv_handles_column_major_decode_input() {
        assert_matmul(
            TensorSpec::sliced_permuted(
                DType::BF16,
                &[1024, 2],
                &[
                    Slice::new(0, 1024, 1).unwrap(),
                    Slice::new(0, 1, 1).unwrap(),
                ],
                &[1, 0],
            ),
            TensorSpec::permuted(DType::BF16, &[2048, 1024], &[1, 0]),
            &TensorSpec::contiguous(DType::BF16, &[1, 2048]),
        );
    }

    #[test]
    fn metal_matmul_copies_irregular_operands_through_scratch() {
        assert_matmul(
            TensorSpec::sliced(
                DType::F32,
                &[7, 66],
                &[Slice::new(0, 7, 1).unwrap(), Slice::new(0, 33, 2).unwrap()],
            ),
            TensorSpec::permuted(DType::F32, &[17, 33], &[1, 0]),
            &TensorSpec::sliced(
                DType::F32,
                &[7, 34],
                &[Slice::new(0, 7, 1).unwrap(), Slice::new(0, 17, 2).unwrap()],
            ),
        );
    }

    #[test]
    fn metal_matmul_casts_inputs_to_the_output_dtype() {
        assert_matmul(
            TensorSpec::contiguous(DType::F16, &[7, 33]),
            TensorSpec::permuted(DType::BF16, &[17, 33], &[1, 0]),
            &TensorSpec::contiguous(DType::F32, &[7, 17]),
        );
    }

    #[test]
    fn metal_copy_writes_a_kv_cache_slice() {
        let slices = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(37, 1, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Copy,
            &[TensorSpec::contiguous(DType::F16, &[8, 1, 128])],
            &TensorSpec::sliced(DType::F16, &[8, 4096, 128], &slices),
        )
        .unwrap();
    }

    #[test]
    fn metal_add_matches_cpu_for_dtypes_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097] {
                let input = TensorSpec::contiguous(dtype, &[width]);
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Add,
                    &[input.clone(), input],
                    &TensorSpec::contiguous(dtype, &[width]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Add,
                &[
                    TensorSpec::permuted(dtype, &[7, 33], &[1, 0]),
                    TensorSpec::broadcast(dtype, &[1, 7], &[33, 7]),
                ],
                &TensorSpec::contiguous(dtype, &[33, 7]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_rms_norm_matches_cpu_for_dtypes_rows_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 128, 1024, 4097] {
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::RmsNorm { eps: 1e-6 },
                    &[
                        TensorSpec::contiguous(dtype, &[7, width]),
                        TensorSpec::contiguous(dtype, &[width]),
                    ],
                    &TensorSpec::contiguous(dtype, &[7, width]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::RmsNorm { eps: 1e-6 },
                &[
                    TensorSpec::permuted(dtype, &[33, 7], &[1, 0]),
                    TensorSpec::sliced(dtype, &[66], &[Slice::new(0, 33, 2).unwrap()]),
                ],
                &TensorSpec::contiguous(dtype, &[7, 33]),
            )
            .unwrap();
        }
        assert_backends_agree(
            &reference,
            &candidate,
            Op::RmsNorm { eps: 1e-6 },
            &[
                TensorSpec::contiguous(DType::F16, &[7, 1024]),
                TensorSpec::contiguous(DType::BF16, &[1024]),
            ],
            &TensorSpec::contiguous(DType::F16, &[7, 1024]),
        )
        .unwrap();
    }

    #[test]
    fn metal_softmax_matches_cpu_for_dtypes_rows_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097] {
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Softmax,
                    &[TensorSpec::contiguous(dtype, &[7, width])],
                    &TensorSpec::contiguous(dtype, &[7, width]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Softmax,
                &[TensorSpec::permuted(dtype, &[33, 7], &[1, 0])],
                &TensorSpec::contiguous(dtype, &[7, 33]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_argmax_matches_cpu_for_dtypes_rows_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 2047, 2048, 2049, 4097, 151_936] {
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Argmax,
                    &[TensorSpec::contiguous(dtype, &[2, width])],
                    &TensorSpec::contiguous(DType::U32, &[2]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Argmax,
                &[TensorSpec::permuted(dtype, &[33, 7], &[1, 0])],
                &TensorSpec::contiguous(DType::U32, &[7]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_argmax_reports_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[1, 151_936]).unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mut timings = Vec::with_capacity(20);
        for iteration in 0..23 {
            let mut commands = CommandList::new();
            commands.dispatch(Op::Argmax, &[&input], &output).unwrap();
            let submission = backend.submit_profiled(commands).unwrap();
            submission.wait().unwrap();
            if iteration >= 3 {
                timings.push(submission.profile().unwrap().per_dispatch[0].gpu_time);
            }
        }
        timings.sort_unstable();
        let median = timings[timings.len() / 2];
        eprintln!("argmax width 151936: median GPU {median:?}");
        assert!(median > Duration::ZERO);
    }

    #[test]
    fn metal_argmax_matches_cpu_total_order_and_last_tie() {
        let cases = [
            (
                DType::F32,
                [
                    0x7fc0_0001_u32,
                    0xffc0_0001,
                    0xff80_0000,
                    0x8000_0000,
                    0x0000_0000,
                    0x7f80_0000,
                    0x7fc0_0001,
                ]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            ),
            (
                DType::F16,
                [0x7e01_u16, 0xfe01, 0xfc00, 0x8000, 0x0000, 0x7c00, 0x7e01]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
            (
                DType::BF16,
                [0x7fc1_u16, 0xffc1, 0xff80, 0x8000, 0x0000, 0x7f80, 0x7fc1]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        ];
        for (dtype, bytes) in cases {
            assert_backends_agree(
                &CpuBackend::new(),
                &MetalBackend::new().unwrap(),
                Op::Argmax,
                &[TensorSpec::initialized(dtype, &[1, 7], bytes)],
                &TensorSpec::contiguous(DType::U32, &[1]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_greedy_sample_matches_argmax_edges_and_qwen_width() {
        let params = sample_params(0.0, 0, 1.0, 7);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097, 151_936] {
                assert_backends_agree(
                    &CpuBackend::new(),
                    &MetalBackend::new().unwrap(),
                    Op::Sample { position: 33 },
                    &[
                        TensorSpec::contiguous(dtype, &[1, width]),
                        TensorSpec::initialized(DType::U32, &[5], params.clone()),
                    ],
                    &TensorSpec::contiguous(DType::U32, &[1]),
                )
                .unwrap();
            }
        }
        let edges = [
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::INFINITY,
            f32::NAN,
            f32::NAN,
            f32::INFINITY,
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Sample { position: 33 },
            &[
                TensorSpec::initialized(DType::F32, &[1, 7], edges),
                TensorSpec::initialized(DType::U32, &[5], params),
            ],
            &TensorSpec::contiguous(DType::U32, &[1]),
        )
        .unwrap();
    }

    #[test]
    fn metal_sample_respects_top_k_and_top_p_exactly() {
        let logits = [10.0_f32, 9.0, 8.0, 1.0, 0.0, -1.0, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for (temperature, top_k, top_p, maximum) in
            [(1.0, 3, 1.0, 2), (1.0, 0, 0.7, 1), (1.0, 2, 0.999, 1)]
        {
            for position in 0..128 {
                let token = run_sample(
                    &MetalBackend::new().unwrap(),
                    &logits,
                    7,
                    temperature,
                    top_k,
                    top_p,
                    11,
                    position,
                )
                .unwrap();
                assert!(token <= maximum, "selected {token} at position {position}");
            }
        }
    }

    #[test]
    fn metal_sample_is_deterministic_and_reports_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let logits = (0_u32..151_936)
            .flat_map(|index| {
                let value = f32::from(u16::try_from(index % 97).unwrap()) / 32.0;
                value.to_le_bytes()
            })
            .collect::<Vec<_>>();
        let first = run_sample(&backend, &logits, 151_936, 0.7, 0, 0.9, 91, 511).unwrap();
        let second = run_sample(&backend, &logits, 151_936, 0.7, 0, 0.9, 91, 511).unwrap();
        assert_eq!(first, second);

        let input = backend.alloc(DType::F32, &[1, 151_936]).unwrap();
        backend.write(&input, &logits).unwrap();
        let params = backend.alloc(DType::U32, &[5]).unwrap();
        backend
            .write(&params, &sample_params(0.7, 0, 0.9, 91))
            .unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Sample { position: 511 }, &[&input, &params], &output)
            .unwrap();
        let submission = backend.submit_profiled(commands).unwrap();
        submission.wait().unwrap();
        let elapsed = submission.profile().unwrap().per_dispatch[0].gpu_time;
        eprintln!("sample width 151936 top-p 0.9: GPU {elapsed:?}");
        assert!(elapsed > Duration::ZERO);
    }

    #[test]
    fn metal_sample_refuses_invalid_live_parameters() {
        let backend = MetalBackend::new().unwrap();
        let logits = vec![0_u8; 7 * 4];
        assert_eq!(
            run_sample(&backend, &logits, 7, f32::NAN, 0, 1.0, 0, 0),
            Err(BackendError::InvalidInput)
        );
    }

    #[test]
    fn metal_softmax_handles_masked_and_all_masked_rows() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let masked = [
            0.0_f32,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            f32::NEG_INFINITY,
            2.0,
            0.5,
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
        assert_backends_agree(
            &reference,
            &candidate,
            Op::Softmax,
            &[TensorSpec::initialized(DType::F32, &[1, 7], masked)],
            &TensorSpec::contiguous(DType::F32, &[1, 7]),
        )
        .unwrap();
        let all_masked = std::iter::repeat_n(f32::NEG_INFINITY, 33)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_backends_agree(
            &reference,
            &candidate,
            Op::Softmax,
            &[TensorSpec::initialized(DType::F32, &[1, 33], all_masked)],
            &TensorSpec::contiguous(DType::F32, &[1, 33]),
        )
        .unwrap();
    }

    #[test]
    fn metal_rope_matches_cpu_for_qwen_shapes_and_position() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let query_positions = (4089_u32..=4095)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let decode_position = 4095_u32.to_le_bytes().to_vec();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let positions = TensorSpec::initialized(DType::U32, &[7], query_positions.clone());
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Rope { theta: 1e6 },
                &[
                    TensorSpec::contiguous(dtype, &[7, 16, 128]),
                    positions.clone(),
                ],
                &TensorSpec::contiguous(dtype, &[7, 16, 128]),
            )
            .unwrap();
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Rope { theta: 1e6 },
                &[
                    TensorSpec::permuted(dtype, &[128, 16, 7], &[2, 1, 0]),
                    positions,
                ],
                &TensorSpec::contiguous(dtype, &[7, 16, 128]),
            )
            .unwrap();
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Rope { theta: 1e6 },
                &[
                    TensorSpec::contiguous(dtype, &[1, 8, 128]),
                    TensorSpec::initialized(DType::U32, &[1], decode_position.clone()),
                ],
                &TensorSpec::contiguous(dtype, &[1, 8, 128]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_embed_matches_cpu_for_dtypes_ids_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let ids = [0_u32, 32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097] {
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::Embed,
                    &[
                        TensorSpec::contiguous(dtype, &[33, width]),
                        TensorSpec::initialized(DType::U32, &[2], ids.clone()),
                    ],
                    &TensorSpec::contiguous(dtype, &[2, width]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                Op::Embed,
                &[
                    TensorSpec::permuted(dtype, &[128, 33], &[1, 0]),
                    TensorSpec::initialized(DType::U32, &[2], ids.clone()),
                ],
                &TensorSpec::contiguous(dtype, &[2, 128]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_embed_zeros_and_reports_an_out_of_range_id() {
        let cases = [
            (33, vec![33_u32], 33),
            (6, vec![5_u32, u32::MAX], u32::MAX),
            (6, vec![u32::MAX], u32::MAX),
        ];
        for (vocab, values, expected) in cases {
            let backend = MetalBackend::new().unwrap();
            let table = backend.alloc(DType::F32, &[vocab, 7]).unwrap();
            let ids = backend
                .alloc(DType::U32, &[u32::try_from(values.len()).unwrap()])
                .unwrap();
            let bytes = values
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            backend.write(&ids, &bytes).unwrap();
            let output = backend
                .alloc(DType::F32, &[u32::try_from(bytes.len() / 4).unwrap(), 7])
                .unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::Embed, &[&table, &ids], &output)
                .unwrap();

            assert_eq!(
                backend.submit(commands).unwrap().wait(),
                Err(BackendError::IndexOutOfRange { index: expected })
            );
            let _completion_error = backend.read(&output);
            assert_eq!(backend.read(&output).unwrap(), vec![0_u8; bytes.len() * 7]);
        }
    }
}
