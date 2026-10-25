use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
    ffi::c_void,
    fmt::Write as _,
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(test)]
use std::sync::atomic::AtomicU64;

use block2::RcBlock;
use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, DispatchProfile, GraphTemplate,
    KernelProfile, Layout, Op, PagedCopyDirection, PreparedGraph, ProfileCount, ProfileTensor,
    Slice, Submission, SubmissionProfile, Tensor,
    program::{
        KernelSignature, PreparedEpilogue, PreparedProgram, ProgramHash, ProgramKind,
        ValidatedProgram,
    },
    required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
#[cfg(test)]
use objc2_metal::MTLResourceOptions;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue,
    MTL4CommitFeedback, MTL4CommitOptions, MTL4CounterHeap, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTL4TimestampGranularity, MTLAllocation, MTLBuffer, MTLCompileOptions,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLEvent, MTLFunctionConstantValues,
    MTLLibrary, MTLMathMode, MTLResidencySet, MTLResidencySetDescriptor, MTLSharedEvent,
    MTLSharedEventListener, MTLSize,
};

use crate::{
    epilogue_codegen, map_codegen,
    matmul::{
        GEMM_32_64, GEMM_64_32, GEMM_64_64_FLOAT, GEMM_64_64_HALF, GemmConfig, classify,
        select_internal_gemm,
    },
    storage::{GraphArenaReservation, MetalBackend, MetalBuffer},
    variants::MetalVariant,
};

type ComputePipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

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

trait ArgumentSink {
    type Written;

    fn write_argument(&mut self, bytes: &[u8]) -> Result<Self::Written, BackendError>;
}

impl ArgumentSink for ArgumentWriter {
    type Written = BufferBinding;

    fn write_argument(&mut self, bytes: &[u8]) -> Result<Self::Written, BackendError> {
        self.write(bytes)
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

impl ArgumentSink for ArgumentSizer {
    type Written = ();

    fn write_argument(&mut self, bytes: &[u8]) -> Result<Self::Written, BackendError> {
        self.write(bytes.len())
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

fn gather_quant_params(
    tensors: [&EncoderTensor; 6],
    config: (u8, u32),
    block_rows: u32,
) -> Result<(Vec<u8>, [u32; 3]), BackendError> {
    let [input, packed, scales, biases, indices, output] = tensors;
    let (bits, group_size) = config;
    let [rows, inner]: [u32; 2] = input
        .layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let [experts, columns, packed_width]: [u32; 3] = packed
        .layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let [_, routes]: [u32; 2] = indices
        .layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let grouped = GroupedRouteShape::new(rows, routes, experts, block_rows)?;
    let mut params = Vec::with_capacity(212);
    for value in [
        input.layout.offset(),
        packed.layout.offset(),
        scales.layout.offset(),
        biases.layout.offset(),
        indices.layout.offset(),
        output.layout.offset(),
    ]
    .into_iter()
    .chain(input.layout.strides().iter().copied())
    .chain(packed.layout.strides().iter().copied())
    .chain(scales.layout.strides().iter().copied())
    .chain(biases.layout.strides().iter().copied())
    .chain(indices.layout.strides().iter().copied())
    .chain(output.layout.strides().iter().copied())
    {
        params.extend_from_slice(&value.to_ne_bytes());
    }
    for value in [
        rows,
        routes,
        inner,
        experts,
        columns,
        packed_width,
        u32::from(bits),
        group_size,
        grouped.block_capacity,
    ] {
        params.extend_from_slice(&value.to_ne_bytes());
    }
    Ok((params, [rows, routes, columns]))
}

fn gather_quant_combine_params(
    tensors: [&EncoderTensor; 7],
    config: (u8, u32),
) -> Result<(Vec<u8>, [u32; 3]), BackendError> {
    let [input, packed, scales, biases, indices, weights, output] = tensors;
    let (bits, group_size) = config;
    let [rows, routes, inner]: [u32; 3] = input
        .layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let [experts, columns, packed_width]: [u32; 3] = packed
        .layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let mut params = Vec::with_capacity(256);
    for value in [
        input.layout.offset(),
        packed.layout.offset(),
        scales.layout.offset(),
        biases.layout.offset(),
        indices.layout.offset(),
        weights.layout.offset(),
        output.layout.offset(),
    ]
    .into_iter()
    .chain(input.layout.strides().iter().copied())
    .chain(packed.layout.strides().iter().copied())
    .chain(scales.layout.strides().iter().copied())
    .chain(biases.layout.strides().iter().copied())
    .chain(indices.layout.strides().iter().copied())
    .chain(weights.layout.strides().iter().copied())
    .chain(output.layout.strides().iter().copied())
    {
        params.extend_from_slice(&value.to_ne_bytes());
    }
    for value in [
        rows,
        routes,
        inner,
        experts,
        columns,
        packed_width,
        u32::from(bits),
        group_size,
    ] {
        params.extend_from_slice(&value.to_ne_bytes());
    }
    Ok((params, [rows, routes, columns]))
}

#[derive(Clone, Copy)]
struct GroupedRouteShape {
    route_count: u32,
    block_capacity: u32,
    block_elements: u32,
}

#[derive(Clone, Copy)]
struct WeightedGatherSchedule {
    implementation: &'static str,
    threadgroup_width: u32,
}

fn weighted_gather_schedule(
    selected: Option<MetalVariant>,
    routes: u32,
) -> Result<WeightedGatherSchedule, BackendError> {
    match selected {
        Some(MetalVariant::GatherQuantMatmulWeightedBroad) => Ok(WeightedGatherSchedule {
            implementation: "gather_quantized_weighted_sum_broad",
            threadgroup_width: 32,
        }),
        Some(
            MetalVariant::GatherQuantMatmulPortable | MetalVariant::GatherQuantMatmulRouteGemv,
        ) if routes <= 8 => Ok(WeightedGatherSchedule {
            implementation: "gather_quantized_weighted_sum",
            threadgroup_width: routes.checked_mul(32).ok_or(BackendError::InvalidInput)?,
        }),
        _ => Err(BackendError::InvalidInput),
    }
}

impl GroupedRouteShape {
    fn new(rows: u32, routes: u32, experts: u32, block_rows: u32) -> Result<Self, BackendError> {
        let route_count = u64::from(rows)
            .checked_mul(u64::from(routes))
            .ok_or(BackendError::InvalidInput)?;
        let block_capacity = route_count
            .div_ceil(u64::from(block_rows))
            .checked_add(u64::from(experts))
            .ok_or(BackendError::InvalidInput)?;
        let block_elements = block_capacity
            .checked_mul(4)
            .ok_or(BackendError::InvalidInput)?;
        Ok(Self {
            route_count: u32::try_from(route_count).map_err(|_| BackendError::InvalidInput)?,
            block_capacity: u32::try_from(block_capacity)
                .map_err(|_| BackendError::InvalidInput)?,
            block_elements: u32::try_from(block_elements)
                .map_err(|_| BackendError::InvalidInput)?,
        })
    }
}

fn grouped_gather_dispatch(dispatch: &Dispatch) -> bool {
    matches!(
        selected_variant(dispatch),
        Some(
            MetalVariant::GatherQuantMatmulGrouped
                | MetalVariant::GatherQuantMatmulGroupedSmallM
                | MetalVariant::GatherQuantMatmulGroupedPairSmallM
        )
    )
}

fn grouped_block_rows(dispatch: &Dispatch) -> u32 {
    if matches!(
        selected_variant(dispatch),
        Some(
            MetalVariant::GatherQuantMatmulGroupedSmallM
                | MetalVariant::GatherQuantMatmulGroupedPairSmallM
        )
    ) {
        8
    } else {
        16
    }
}

fn grouped_error_flag(arguments: &mut ArgumentWriter) -> Result<BufferBinding, BackendError> {
    arguments.write(&error_state())
}

fn error_state() -> [u8; 8] {
    let mut state = [0_u8; 8];
    state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
    state
}

fn paged_attention_state() -> [u8; 16] {
    let mut state = [0_u8; 16];
    state[4..8].copy_from_slice(&u32::MAX.to_ne_bytes());
    state[8..12].copy_from_slice(&1_u32.to_ne_bytes());
    state
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
    VectorGqa,
    Steel,
    SteelHalf,
    Decomposed,
}

#[cfg(test)]
/// Bounds the vector-SDPA test fixture to the registered legality limit.
const VECTOR_MAX_KEY_LENGTH: u32 = 65_536;
/// Specializes how many elements a partial reduction workgroup consumes.
const REDUCTION_CHUNK_WIDTH_CONSTANT: u32 = 17;
/// Specializes the bounded sampling rank arrays without exceeding their capacity of twenty.
const SAMPLE_FAST_TOP_K_CONSTANT: u32 = 18;
/// Function constant selecting mask reads in the parallel sampling kernels.
const SAMPLE_HAS_MASK_CONSTANT: u32 = 20;
/// Specializes the bounded sampling merge without exceeding 2,048 partial candidates.
const SAMPLE_FAST_MAX_CHUNKS_CONSTANT: u32 = 19;
/// Index of the sampling error flag in the prepared argument vector.
const SAMPLE_ERROR: usize = 0;
/// Index of the logits layout in the prepared argument vector.
const SAMPLE_LOGITS_LAYOUT: usize = 1;
/// Index of the sampling-parameter layout in the prepared argument vector.
const SAMPLE_PARAMS_LAYOUT: usize = 2;
/// Index of the output layout in the prepared argument vector.
const SAMPLE_OUTPUT_LAYOUT: usize = 3;
/// Index of the row width in the prepared argument vector.
const SAMPLE_WIDTH: usize = 4;
/// Index of the sequence position in the prepared argument vector.
const SAMPLE_POSITION: usize = 5;
/// Index of the partial-workgroup count in the prepared argument vector.
const SAMPLE_CHUNKS: usize = 6;
/// Index of the rejection-attempt bound in the prepared argument vector.
const SAMPLE_MAX_ROUNDS: usize = 7;
/// First index reserved for per-round sampling arguments.
const SAMPLE_ROUND_ARGUMENTS: usize = 8;
/// Unpinned single-row matrix products use the ordinary GEMV storage traversal.
const DEFAULT_GEMV_TRANSPOSED: bool = false;

#[derive(Clone, Copy)]
struct SampleKernelParams {
    position: u32,
    chunk_width: u32,
    fast_top_k: u32,
    fast_max_chunks: u32,
    rejection_rounds: u32,
}

impl SampleKernelParams {
    const fn function_constants(self, dtype: u32) -> [(u32, u32); 4] {
        [
            (0, dtype),
            (REDUCTION_CHUNK_WIDTH_CONSTANT, self.chunk_width),
            (SAMPLE_FAST_TOP_K_CONSTANT, self.fast_top_k),
            (SAMPLE_FAST_MAX_CHUNKS_CONSTANT, self.fast_max_chunks),
        ]
    }
}

struct EncodedDispatches {
    temporaries: Vec<BufferBinding>,
    error_flags: Vec<EncodedErrorFlag>,
    bindings: ArgumentBindings,
    arguments: Option<ArgumentUsage>,
    program_encoding: ProfileCount,
    program_compile_fallbacks: u64,
    internal_profiles: Vec<InternalDispatchProfile>,
}

#[derive(Clone)]
struct InternalDispatchProfile {
    dispatch: usize,
    kernel: &'static str,
    start: usize,
    end: usize,
}

struct EncodedErrorFlag {
    binding: BufferBinding,
    dispatch: usize,
    operation: &'static str,
}

impl EncodedErrorFlag {
    const fn new(binding: BufferBinding, dispatch: usize, operation: &'static str) -> Self {
        Self {
            binding,
            dispatch,
            operation,
        }
    }
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
    GraphArena { offset: usize, len: usize },
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
    IndirectDispatch {
        address: u64,
        threads_per_threadgroup: MTLSize,
    },
    Barrier,
    InternalBoundary,
}

struct PlanRecording {
    arena_base: u64,
    graph_arena_base: Option<u64>,
    calls: Vec<PlanCall>,
}

struct PlanRecorderScope(bool);

impl PlanRecorderScope {
    fn enter(arena_base: u64, graph_arena_base: Option<u64>) -> Self {
        PLAN_RECORDER.with(|recorder| {
            debug_assert!(recorder.borrow().is_none());
            *recorder.borrow_mut() = Some(PlanRecording {
                arena_base,
                graph_arena_base,
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
        } else if Some(range.base) == recording.graph_arena_base {
            PlannedRange::GraphArena {
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
        buffer: &MetalBuffer,
    ) -> Result<(), BackendError> {
        let base = buffer.raw.gpuAddress();
        let offset = u64::try_from(buffer.data_offset).map_err(|_| BackendError::InvalidInput)?;
        let address = base.checked_add(offset).ok_or(BackendError::InvalidInput)?;
        // SAFETY: Each caller uses an index within its argument-table descriptor and registers
        // the bound buffer in the command resource owner before submission.
        unsafe {
            table.setAddress_atIndex(address, index);
        }
        let range = BoundRange {
            base,
            offset: buffer.data_offset,
            len: buffer.len,
            address,
        };
        record_plan_binding(index, range);
        self.ranges.insert(range);
        Ok(())
    }
}

struct SampleCommonBindings<'a> {
    logits: &'a BufferBinding,
    sampling: &'a BufferBinding,
    state: &'a BufferBinding,
    logits_layout: &'a BufferBinding,
    sampling_layout: &'a BufferBinding,
    width: &'a BufferBinding,
}

impl SampleCommonBindings<'_> {
    fn bind(
        &self,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        bindings: &mut ArgumentBindings,
        workspace: &BufferBinding,
    ) {
        bindings.bind(table, 0, self.logits);
        bindings.bind(table, 1, self.sampling);
        bindings.bind(table, 2, workspace);
        bindings.bind(table, 3, self.state);
        bindings.bind(table, 4, self.logits_layout);
        bindings.bind(table, 5, self.sampling_layout);
        bindings.bind(table, 6, self.width);
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
    static SUBMISSION_SCRATCH: RefCell<Option<Arc<Mutex<ReusableScratch>>>> = const { RefCell::new(None) };
}

/// The primary submission slot keeps its high-water scratch set to avoid allocation and residency
/// churn, while overlap slots retain only decode-sized buffers so quadratic prefill temporaries do
/// not multiply across every in-flight slot.
#[derive(Default)]
struct ReusableScratch {
    buffers: Vec<InFlightBuffer>,
    next: usize,
}

impl ReusableScratch {
    fn acquire(
        &mut self,
        backend: &MetalBackend,
        len: usize,
    ) -> Result<MetalBufferRef, BackendError> {
        let index = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(BackendError::AllocationFailed)?;
        if index == self.buffers.len() {
            self.buffers.push(InFlightBuffer {
                raw: backend.new_buffer(len)?,
                pool_resident: false,
            });
        } else if self.buffers[index].raw.length() < len {
            self.buffers[index].raw = backend.new_buffer(len)?;
        }
        Ok(self.buffers[index].raw.clone())
    }

    fn trim_above(&mut self, limit: usize) -> bool {
        let old_len = self.buffers.len();
        self.buffers.retain(|buffer| buffer.raw.length() <= limit);
        self.next = 0;
        self.buffers.len() != old_len
    }
}

struct ScratchScope(Option<Arc<Mutex<ReusableScratch>>>);

impl ScratchScope {
    fn enter(scratch: Arc<Mutex<ReusableScratch>>) -> Result<Self, BackendError> {
        scratch
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .next = 0;
        let previous = SUBMISSION_SCRATCH.with(|current| current.replace(Some(scratch)));
        Ok(Self(previous))
    }
}

impl Drop for ScratchScope {
    fn drop(&mut self) {
        SUBMISSION_SCRATCH.with(|current| {
            current.replace(self.0.take());
        });
    }
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
    static FORCED_VECTOR_BLOCKS: Cell<Option<u32>> = const { Cell::new(None) };
    static FORCED_SAMPLE_REJECTION_ROUNDS: Cell<Option<u32>> = const { Cell::new(None) };
}

fn sample_rejection_rounds(configured: u32) -> u32 {
    #[cfg(test)]
    if let Some(rounds) = FORCED_SAMPLE_REJECTION_ROUNDS.with(Cell::get) {
        return rounds.min(configured);
    }
    configured
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

fn commit_result(feedback: &CommitResult, submission: Option<u64>) -> Result<(), BackendError> {
    match feedback {
        CommitResult::Feedback(feedback) => {
            assert_not_in_metal_callback();
            if let Some(error) = feedback.0.error() {
                eprintln!(
                    "Metal execution failed: submission={submission:?} domain={} code={} description={}",
                    error.domain(),
                    error.code(),
                    error.localizedDescription(),
                );
                submission.map_or(Err(BackendError::ExecutionFailed), |submission| {
                    Err(BackendError::DeviceExecutionFailed {
                        submission,
                        code: error.code() as i64,
                    })
                })
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
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    resident_buffers: Vec<MetalBufferRef>,
    scratch: Arc<Mutex<ReusableScratch>>,
    retain_oversized_scratch: bool,
    timestamps: GpuTimestamps,
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

    fn make_resident(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        resources: &CommandResources,
    ) -> Result<Retained<ProtocolObject<dyn MTLResidencySet>>, BackendError> {
        let objects = self
            .reusable
            .as_mut()
            .ok_or(BackendError::ExecutionFailed)?;
        let current = resources
            .buffers
            .iter()
            .filter(|buffer| !buffer.pool_resident)
            .map(|buffer| buffer.raw.clone())
            .collect::<Vec<_>>();
        let unchanged = current.len() == objects.resident_buffers.len()
            && current
                .iter()
                .zip(objects.resident_buffers.iter())
                .all(|(next, old)| std::ptr::eq(&raw const **next, &raw const **old));
        if !unchanged {
            objects.residency.removeAllAllocations();
            for buffer in &current {
                let buffer: &ProtocolObject<dyn MTLBuffer> = buffer;
                let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
                objects.residency.addAllocation(allocation);
            }
            objects.residency.commit();
            objects.resident_buffers = current;
        }
        command_buffer.useResidencySet(&objects.residency);
        Ok(objects.residency.clone())
    }

    fn scratch(&self) -> Result<Arc<Mutex<ReusableScratch>>, BackendError> {
        self.reusable
            .as_ref()
            .map(|objects| Arc::clone(&objects.scratch))
            .ok_or(BackendError::ExecutionFailed)
    }

    fn timestamps(&self) -> Result<GpuTimestamps, BackendError> {
        self.reusable
            .as_ref()
            .map(|objects| objects.timestamps.clone())
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

#[derive(Clone)]
struct GpuTimestamps {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    frequency: u64,
    entry_size: usize,
    count: usize,
}

// SAFETY: The heap is encoded by the checkout owner, then resolved after GPU completion and before
// the owning slot returns to the pool. The completion retains only the resulting owned bytes.
unsafe impl Send for GpuTimestamps {}

// SAFETY: Shared access only retains the heap; resolution is serialized by the completion.
unsafe impl Sync for GpuTimestamps {}

impl GpuTimestamps {
    fn resolve(self) -> Option<ResolvedTimestamps> {
        // SAFETY: The range covers the heap, GPU completion has been observed, and the tracker
        // resolves the data before returning the heap's slot to the reusable pool.
        let data = unsafe { self.heap.resolveCounterRange(NSRange::new(0, self.count)) }?;
        Some(ResolvedTimestamps {
            bytes: data.to_vec(),
            frequency: self.frequency,
            entry_size: self.entry_size,
        })
    }
}

struct ResolvedTimestamps {
    bytes: Vec<u8>,
    frequency: u64,
    entry_size: usize,
}

impl ResolvedTimestamps {
    fn elapsed(&self) -> Option<Duration> {
        self.elapsed_pairs(&[(0, 1)])?.pop()
    }

    fn elapsed_pairs(&self, pairs: &[(usize, usize)]) -> Option<Vec<Duration>> {
        pairs
            .iter()
            .map(|&(start, end)| {
                let read = |index: usize| {
                    let offset = self.entry_size.checked_mul(index)?;
                    Some(u64::from_ne_bytes(
                        self.bytes
                            .get(offset..offset.checked_add(8)?)?
                            .try_into()
                            .ok()?,
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
    error_flags: Vec<(usize, usize, usize, &'static str)>,
}

pub(super) struct PreparedMetalGraph {
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: HashSet<BufferId>,
    encoding: Option<MetalEncodingPlan>,
    variant_rechecks: Vec<usize>,
    memory_plan: Option<forja_core::GraphMemoryPlan>,
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

fn dedupe_pipeline_calls(plans: &mut [Option<StaticEncodingPlan>]) {
    let mut pipeline = None::<Retained<ProtocolObject<dyn MTLComputePipelineState>>>;
    for plan in plans {
        let Some(plan) = plan else {
            pipeline = None;
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
            PlanCall::ArgumentTable
            | PlanCall::Bind { .. }
            | PlanCall::Dispatch { .. }
            | PlanCall::IndirectDispatch { .. }
            | PlanCall::Barrier
            | PlanCall::InternalBoundary => true,
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
    timestamps: Mutex<Option<GpuTimestamps>>,
    resolved_timestamps: OnceLock<Option<ResolvedTimestamps>>,
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
                timestamps: Mutex::new(timestamps),
                resolved_timestamps: OnceLock::new(),
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
                let mut result = commit_result(feedback, state.event_value);
                state.resolving = true;
                drop(state);
                self.resolve_timestamps();
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
        for &(flag, offset, dispatch, operation) in &self.resources.error_flags {
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
                eprintln!(
                    "Metal error flag: submission={:?} dispatch={dispatch} operation={operation} flag={has_error} value={index}",
                    self.event_value(),
                );
                return Err(BackendError::DeviceErrorFlag {
                    submission: self.event_value().ok_or(BackendError::ExecutionFailed)?,
                    dispatch: u32::try_from(dispatch).unwrap_or(u32::MAX),
                    operation,
                    flag: has_error,
                    value: index,
                });
            }
        }
        Ok(())
    }

    fn sample_fallbacks(&self) -> u64 {
        self.resources
            .error_flags
            .iter()
            .fold(0_u64, |count, &(flag, offset, _, _)| {
                // SAFETY: Profiles are read only after GPU completion, and every error flag owns two
                // aligned u32 words in a retained shared buffer.
                let value = unsafe {
                    self.resources.buffers[flag]
                        .raw
                        .contents()
                        .cast::<u32>()
                        .as_ptr()
                        .add(offset / size_of::<u32>() + 1)
                        .read()
                };
                if value == u32::MAX {
                    count
                } else {
                    count.saturating_add(u64::from(value))
                }
            })
    }

    fn resolve_timestamps(&self) {
        self.resolved_timestamps.get_or_init(|| {
            self.timestamps
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .and_then(GpuTimestamps::resolve)
        });
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.resolved_timestamps.get()?.as_ref()?.elapsed()
    }

    fn dispatch_times(&self, metadata: &[DispatchProfile]) -> Option<Vec<DispatchProfile>> {
        let timestamps = self.resolved_timestamps.get()?.as_ref()?;
        metadata
            .iter()
            .enumerate()
            .map(|(index, metadata)| {
                let origin = 2;
                let start = 2 + index * 2;
                let end = start + 1;
                let previous = if index == 0 { start } else { start - 1 };
                let [gpu_start, gpu_end, gap_before, gpu_time] = timestamps
                    .elapsed_pairs(&[
                        (origin, start),
                        (origin, end),
                        (previous, start),
                        (start, end),
                    ])?
                    .try_into()
                    .ok()?;
                Some(DispatchProfile {
                    gpu_start,
                    gpu_end,
                    gap_before,
                    gpu_time,
                    ..metadata.clone()
                })
            })
            .collect()
    }

    fn internal_dispatch_times(
        &self,
        metadata: &[InternalDispatchProfile],
    ) -> Option<Vec<(usize, KernelProfile)>> {
        let timestamps = self.resolved_timestamps.get()?.as_ref()?;
        metadata
            .iter()
            .map(|metadata| {
                let [gpu_start, gpu_end, gpu_time] = timestamps
                    .elapsed_pairs(&[
                        (2, metadata.start),
                        (2, metadata.end),
                        (metadata.start, metadata.end),
                    ])?
                    .try_into()
                    .ok()?;
                Some((
                    metadata.dispatch,
                    KernelProfile {
                        kernel: metadata.kernel,
                        gpu_start,
                        gpu_end,
                        gpu_time,
                    },
                ))
            })
            .collect()
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
    progress: Condvar,
    progress_lock: Mutex<()>,
    slots: AtomicUsize,
    #[cfg(test)]
    backpressure_waits: AtomicUsize,
}

// The host permits two overlapping replays per graph; two extra slots keep unrelated work moving.
/// Caps retained command resources while permitting CPU/GPU overlap.
const MAX_IN_FLIGHT_SUBMISSIONS: usize = 4;
const MAX_RETAINED_SCRATCH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

impl ReusableSubmissionObjects {
    fn trim_scratch(&mut self) {
        let limit = if self.retain_oversized_scratch {
            usize::MAX
        } else {
            MAX_RETAINED_SCRATCH_BUFFER_BYTES
        };
        let trimmed = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .trim_above(limit);
        if trimmed {
            self.residency.removeAllAllocations();
            self.residency.commit();
            self.resident_buffers.clear();
        }
    }
}

impl InFlightTracker {
    pub(super) const fn new() -> Self {
        Self {
            completions: Mutex::new(Vec::new()),
            done: Mutex::new(Vec::new()),
            pool: Mutex::new(Vec::new()),
            progress: Condvar::new(),
            progress_lock: Mutex::new(()),
            slots: AtomicUsize::new(0),
            #[cfg(test)]
            backpressure_waits: AtomicUsize::new(0),
        }
    }

    fn checkout(
        self: &Arc<Self>,
        backend: &MetalBackend,
        argument_capacity: usize,
    ) -> Result<SubmissionObjects, BackendError> {
        assert_not_in_metal_callback();
        let mut reusable = loop {
            self.drain_done();
            if let Some(objects) = self
                .pool
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .pop()
            {
                break objects;
            }
            if let Ok(slot) = self
                .slots
                .try_update(Ordering::AcqRel, Ordering::Acquire, |slots| {
                    (slots < MAX_IN_FLIGHT_SUBMISSIONS).then_some(slots + 1)
                })
            {
                match Self::new_slot(backend, argument_capacity, slot == 0) {
                    Ok(objects) => break objects,
                    Err(error) => {
                        self.slots.fetch_sub(1, Ordering::AcqRel);
                        self.notify_progress();
                        return Err(error);
                    }
                }
            }
            let pending = self
                .completions
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .first()
                .map(|entry| Arc::clone(&entry.completion));
            #[cfg(test)]
            self.backpressure_waits.fetch_add(1, Ordering::Relaxed);
            if let Some(pending) = pending {
                pending.wait(backend.gpu_timeout)?;
            } else {
                let progress = self
                    .progress_lock
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?;
                if !self
                    .pool
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .is_empty()
                    || self.slots.load(Ordering::Acquire) < MAX_IN_FLIGHT_SUBMISSIONS
                    || !self
                        .completions
                        .lock()
                        .map_err(|_| BackendError::ExecutionFailed)?
                        .is_empty()
                {
                    continue;
                }
                let (_progress, wait) = self
                    .progress
                    .wait_timeout(progress, backend.gpu_timeout)
                    .map_err(|_| BackendError::ExecutionFailed)?;
                if wait.timed_out() {
                    return Err(BackendError::QuotaExceeded);
                }
            }
        };
        reset_allocator(&reusable.allocator);
        if reusable.argument_buffer.capacity != argument_capacity {
            let raw = match backend.new_buffer(argument_capacity) {
                Ok(raw) => raw,
                Err(error) => {
                    self.return_dirty(reusable);
                    return Err(error);
                }
            };
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

    fn new_slot(
        backend: &MetalBackend,
        argument_capacity: usize,
        retain_oversized_scratch: bool,
    ) -> Result<ReusableSubmissionObjects, BackendError> {
        use objc2_metal::MTL4ArgumentTableDescriptor;

        let device = &backend.device;
        let allocator = backend.allocate_device_object(|| {
            device
                .newCommandAllocator()
                .ok_or(BackendError::ExecutionFailed)
        })?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(24);
        let argument_table = backend.allocate_device_object(|| {
            device
                .newArgumentTableWithDescriptor_error(&descriptor)
                .map_err(|_| BackendError::ExecutionFailed)
        })?;
        let raw = backend.new_buffer(argument_capacity)?;
        let residency = backend.allocate_device_object(|| {
            let residency = device
                .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
                .map_err(|_| BackendError::ExecutionFailed)?;
            residency.commit();
            Ok(residency)
        })?;
        let timestamps = backend.make_timestamps(2)?;
        Ok(ReusableSubmissionObjects {
            allocator,
            argument_table,
            argument_buffer: ArgumentBuffer {
                raw,
                capacity: argument_capacity,
            },
            residency,
            resident_buffers: Vec::new(),
            scratch: Arc::new(Mutex::new(ReusableScratch::default())),
            retain_oversized_scratch,
            timestamps,
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
        self.notify_progress();
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
        if done.is_empty() {
            return;
        }
        for completed in &done {
            completed.completion.resolve_timestamps();
        }
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for mut completed in done {
            completed.objects.trim_scratch();
            pool.push(completed.objects);
        }
        drop(pool);
        self.notify_progress();
    }

    fn return_dirty(&self, mut objects: ReusableSubmissionObjects) {
        objects.trim_scratch();
        self.pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(objects);
        self.notify_progress();
    }

    fn notify_progress(&self) {
        let _progress = self
            .progress_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.progress.notify_all();
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

    #[cfg(test)]
    pub(super) fn slot_count(&self) -> usize {
        self.slots.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn backpressure_wait_count(&self) -> usize {
        self.backpressure_waits.load(Ordering::Relaxed)
    }
}

/// Completion state for a Metal command buffer.
pub struct MetalSubmission {
    completion: Arc<Completion>,
    timeout: Duration,
    profile: Option<Mutex<SubmissionProfile>>,
    dispatch_profiles: Vec<DispatchProfile>,
    internal_profiles: Vec<InternalDispatchProfile>,
}

/// Timestamp granularity used for per-dispatch GPU profiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileTimestampGranularity {
    /// Lower-overhead timestamps with implementation-defined placement.
    Relaxed,
    /// Timestamps placed precisely at dispatch boundaries.
    Precise,
}

impl Submission for MetalSubmission {
    fn completion_serial(&self) -> Option<u64> {
        self.completion.event_value()
    }

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
        profile.sample_fallbacks = self.completion.sample_fallbacks();
        let mut dispatches = self.completion.dispatch_times(&self.dispatch_profiles)?;
        for (dispatch, internal) in self
            .completion
            .internal_dispatch_times(&self.internal_profiles)?
        {
            dispatches.get_mut(dispatch)?.internal.push(internal);
        }
        profile.per_dispatch = dispatches;
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
        self.submit_commands_inner::<false>(
            commands,
            None,
            None,
            ProfileTimestampGranularity::Precise,
        )
    }

    pub(super) fn submit_commands_profiled(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
        self.submit_profiled_with_granularity(commands, ProfileTimestampGranularity::Precise)
    }

    /// Submits commands with per-dispatch timestamps at the requested granularity.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, encoding, or submission fails.
    pub fn submit_profiled_with_granularity(
        &self,
        commands: CommandList,
        granularity: ProfileTimestampGranularity,
    ) -> Result<MetalSubmission, BackendError> {
        self.submit_commands_inner::<true>(commands, None, None, granularity)
    }

    pub(super) fn prepare_metal_graph(
        &self,
        graph: &GraphTemplate,
    ) -> Result<PreparedMetalGraph, BackendError> {
        let fallback_count = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .epilogue_counts
            .fallbacks;
        self.prepare_graph_epilogues(graph)?;
        let encoding = (self.graph_replay == crate::storage::MetalGraphReplay::Tier2)
            .then(|| self.prepare_encoding_plan(graph))
            .transpose();
        let fallback_count_after = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .epilogue_counts
            .fallbacks;
        if fallback_count_after != fallback_count {
            return Err(BackendError::ExecutionFailed);
        }
        let encoding = encoding?;
        let _reservation = self.reserve_allocation(0)?;
        let residency = self
            .device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let tensors = graph.retained_tensors();
        let graph_arena = graph.memory_arena();
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let mut retained = HashSet::with_capacity(tensors.len());
        for tensor in &tensors {
            if Some(tensor.buffer()) != graph_arena && retained.insert(tensor.buffer()) {
                let buffer: &ProtocolObject<dyn MTLBuffer> = &buffers.get(tensor)?.raw;
                let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
                residency.addAllocation(allocation);
            }
        }
        residency.commit();
        self.check_allocation_limit()?;
        Ok(PreparedMetalGraph {
            residency,
            buffers: retained,
            encoding,
            variant_rechecks: graph.parameter_dependent_dispatch_indices().collect(),
            memory_plan: None,
        })
    }

    fn prepare_graph_epilogues(&self, graph: &GraphTemplate) -> Result<(), BackendError> {
        let mut requests = Vec::new();
        for (_, parameter_dependent, dispatch) in graph
            .backend_dispatch_candidates()
            .map_err(|_| BackendError::InvalidInput)?
        {
            if dispatch.prepared_epilogue().is_some() {
                requests.extend(epilogue_pipeline_requests(&dispatch, parameter_dependent)?);
            }
        }
        self.prepare_epilogue_requests(&requests)
    }

    fn prepare_dispatch_epilogues(&self, dispatches: &[Dispatch]) -> Result<(), BackendError> {
        let mut requests = Vec::new();
        for dispatch in dispatches {
            if dispatch.prepared_epilogue().is_some() {
                requests.extend(epilogue_pipeline_requests(dispatch, false)?);
            }
        }
        self.prepare_epilogue_requests(&requests)
    }

    fn prepare_epilogue_requests(
        &self,
        requests: &[EpiloguePipelineRequest],
    ) -> Result<(), BackendError> {
        if requests.is_empty() {
            return Ok(());
        }
        let required = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .missing_epilogue_pipelines(requests);
        self.program_compile_tokens
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .consume(required)
            .then_some(())
            .ok_or(BackendError::QuotaExceeded)?;
        self.pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .prepare_epilogue_pipelines(requests)
    }

    pub(super) fn prepare_metal_graph_memory(
        &self,
        graph: GraphTemplate,
        plan: forja_core::GraphMemoryPlan,
    ) -> Result<PreparedGraph, BackendError> {
        self.ensure_graph_arenas(plan.arena_bytes())?;
        let arena = self.acquire_graph_arena()?;
        let bound = graph
            .bind_memory(&plan, arena.tensor())
            .map_err(|_| BackendError::InvalidInput);
        let state = bound.and_then(|bound| self.prepare_metal_graph(&bound));
        self.release_graph_arena(&arena);
        let mut state = state?;
        state.memory_plan = Some(plan);
        Ok(PreparedGraph::with_backend_state(graph, state))
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
        let objects = self.in_flight.checkout(self, capacity)?;
        let _scratch = ScratchScope::enter(objects.scratch()?)?;
        let command_buffer = self.begin_command_buffer(objects.allocator()?)?;
        let mut arguments = ArgumentWriter::new(objects.argument_buffer()?);
        let graph_arena_base = graph
            .memory_arena()
            .map(|arena| {
                self.buffers
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .get_buffer(arena)
                    .map(|buffer| buffer.raw.gpuAddress())
            })
            .transpose()?;
        for (index, dispatch) in static_dispatches {
            let recorder = PlanRecorderScope::enter(arguments.raw.gpuAddress(), graph_arena_base);
            let encoded = self.encode_dispatches(
                &command_buffer,
                objects.argument_table()?,
                &DispatchEncoding {
                    dispatches: std::slice::from_ref(dispatch),
                    barriers: &[false],
                },
                None,
                ProfileTimestampGranularity::Precise,
                &mut arguments,
                None,
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
        dedupe_pipeline_calls(&mut plans);
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
        let mut arena = state
            .memory_plan
            .as_ref()
            .map(|_| self.acquire_graph_arena())
            .transpose()?;
        let bound;
        let template = if let (Some(plan), Some(arena)) = (&state.memory_plan, &arena) {
            bound = graph
                .template()
                .bind_memory(plan, arena.tensor())
                .map_err(|_| BackendError::InvalidInput)?;
            &bound
        } else {
            graph.template()
        };
        let commands = template
            .instantiate(&values)
            .map_err(|_| BackendError::InvalidInput)?;
        for &index in &state.variant_rechecks {
            let dispatch = commands
                .dispatches()
                .get(index)
                .ok_or(BackendError::InvalidInput)?;
            if let Some(&variant) = dispatch.backend_data::<crate::ValidatedVariant>() {
                crate::variants::revalidate_replay(
                    dispatch,
                    variant,
                    &[forja_core::DeviceCapability::Metal4],
                )
                .map_err(|_| BackendError::InvalidInput)?;
            }
        }
        let submission = if profile {
            self.submit_commands_inner::<true>(
                commands,
                Some(state),
                arena.as_ref(),
                ProfileTimestampGranularity::Precise,
            )
        } else {
            self.submit_commands_inner::<false>(
                commands,
                Some(state),
                arena.as_ref(),
                ProfileTimestampGranularity::Precise,
            )
        };
        match submission {
            Ok(submission) => {
                if let Some(arena) = &mut arena {
                    self.commit_graph_arena(arena, &submission.completion)?;
                }
                Ok(submission)
            }
            Err(error) => {
                if let Some(arena) = arena {
                    self.release_graph_arena(&arena);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn submit_commands_inner<const PROFILE: bool>(
        &self,
        commands: CommandList,
        graph: Option<&PreparedMetalGraph>,
        graph_arena: Option<&GraphArenaReservation>,
        timestamp_granularity: ProfileTimestampGranularity,
    ) -> Result<MetalSubmission, BackendError> {
        #[cfg(test)]
        let submit_started = Instant::now();
        self.in_flight.drain_done();
        let validation_started = PROFILE.then(Instant::now);
        let program_recording = commands.program_recording();
        let retained_tensors_validated = commands.retained_tensors_validated();
        let barriers = required_barriers(&commands);
        let dispatches = commands.into_dispatches();
        if graph.is_none() {
            self.prepare_dispatch_epilogues(&dispatches)?;
        }
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
        let mut objects = self.in_flight.checkout(self, argument_capacity)?;
        let _scratch = ScratchScope::enter(objects.scratch()?)?;
        let command_buffer = self.begin_command_buffer(objects.allocator()?)?;
        let timestamps = if PROFILE {
            let internal = dispatches
                .iter()
                .map(internal_dispatch_names)
                .try_fold(0_usize, |count, names| {
                    count.checked_add(names.len().saturating_sub(1))
                })
                .ok_or(BackendError::ExecutionFailed)?;
            let count = dispatches
                .len()
                .checked_mul(2)
                .and_then(|count| count.checked_add(2))
                .and_then(|count| count.checked_add(internal))
                .ok_or(BackendError::ExecutionFailed)?;
            self.make_timestamps(count)?
        } else {
            objects.timestamps()?
        };
        // SAFETY: `make_timestamps` always reserves at least two entries, so indices 0 and 1 are
        // in range.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 0);
        }
        let encoding_started = PROFILE.then(Instant::now);
        let mut temporary_profile = TemporaryProfileScope::enter(PROFILE);
        let mut argument_writer = ArgumentWriter::new(objects.argument_buffer()?);
        let graph_arena_binding = graph_arena
            .map(|arena| {
                self.buffers
                    .lock()
                    .map_err(|_| BackendError::ExecutionFailed)?
                    .get(arena.tensor())
                    .map(|buffer| BufferBinding::whole(buffer.raw.clone()))
            })
            .transpose()?;
        if let Some(encoding) = graph.and_then(|graph| graph.encoding.as_ref()) {
            argument_writer.copy_image(&encoding.arena)?;
        }
        let mut encoded = self.encode_dispatches(
            &command_buffer,
            objects.argument_table()?,
            &DispatchEncoding {
                dispatches: &dispatches,
                barriers: &barriers,
            },
            PROFILE.then_some(&timestamps),
            timestamp_granularity,
            &mut argument_writer,
            graph.and_then(|prepared| {
                prepared
                    .encoding
                    .as_ref()
                    .map(|encoding| encoding.dispatches.as_slice())
            }),
            graph_arena_binding.as_ref(),
        )?;
        argument_writer.validate_capacity(argument_capacity, dispatches.is_empty())?;
        let internal_profiles = if let Some(profile) = &mut profile {
            let epilogue = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .take_epilogue_counts();
            profile.metadata_buffers = temporary_profile.finish();
            profile.program_encoding = encoded.program_encoding;
            profile.program_compile_fallbacks = encoded.program_compile_fallbacks;
            profile.epilogue_library_compilations = epilogue.libraries;
            profile.epilogue_pipeline_specializations = epilogue.pipelines;
            profile.epilogue_cache_hits = epilogue.cache_hits;
            profile.epilogue_compile_fallbacks = epilogue.fallbacks;
            let encoding = encoding_started.map_or(Duration::ZERO, |started| started.elapsed());
            profile.encoding = encoding.saturating_sub(profile.metadata_buffers.time);
            std::mem::take(&mut encoded.internal_profiles)
        } else {
            Vec::new()
        };
        let resources =
            self.command_resources(&tensors, encoded, graph.map(|prepared| &prepared.buffers))?;
        objects.programs = prepared_programs;
        let residency_started = PROFILE.then(Instant::now);
        let residency = objects.make_resident(&command_buffer, &resources)?;
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
        let dispatch_profiles = dispatches
            .iter()
            .zip(&barriers)
            .map(|(dispatch, &barrier)| dispatch_profile(dispatch, barrier))
            .collect::<Result<Vec<_>, _>>()?;
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
            submission.dispatch_profiles = dispatch_profiles;
            submission.internal_profiles = internal_profiles;
        }
        #[cfg(test)]
        store_duration(&LAST_SUBMIT_NANOS, submit_started.elapsed());
        Ok(submission)
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        plan: &DispatchEncoding<'_>,
        timestamps: Option<&GpuTimestamps>,
        timestamp_granularity: ProfileTimestampGranularity,
        arguments: &mut ArgumentWriter,
        prepared: Option<&[Option<StaticEncodingPlan>]>,
        graph_arena: Option<&BufferBinding>,
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
        let mut internal_timestamp = plan
            .dispatches
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(2))
            .ok_or(BackendError::ExecutionFailed)?;
        let mut internal_profiles = Vec::new();
        for (index, (dispatch, &barrier)) in plan.dispatches.iter().zip(plan.barriers).enumerate() {
            if index > 0 {
                write_dispatch_timestamp(
                    &encoder,
                    timestamps,
                    timestamp_granularity,
                    1 + index * 2,
                );
            }
            if barrier {
                encode_dispatch_barrier(&encoder);
            }
            write_dispatch_timestamp(&encoder, timestamps, timestamp_granularity, 2 + index * 2);
            if timestamps.is_some() {
                internal_profiles.extend(internal_dispatch_profiles(
                    dispatch,
                    index,
                    internal_timestamp,
                ));
            }
            if let Some(plan) = prepared
                .and_then(|plans| plans.get(index))
                .and_then(Option::as_ref)
            {
                Self::encode_plan(
                    &encoder,
                    table,
                    plan,
                    timestamps,
                    &mut internal_timestamp,
                    &mut bindings,
                    arguments,
                    graph_arena,
                )?;
                continue;
            }
            self.encode_dispatch(
                &encoder,
                table,
                dispatch,
                index,
                timestamps.is_some(),
                timestamps,
                &mut internal_timestamp,
                &mut temporaries,
                &mut error_flags,
                &mut bindings,
                arguments,
                &mut program_encoding,
                &mut program_compile_fallbacks,
            )?;
        }
        if !plan.dispatches.is_empty() {
            write_dispatch_timestamp(
                &encoder,
                timestamps,
                timestamp_granularity,
                1 + plan.dispatches.len() * 2,
            );
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
            internal_profiles,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_plan(
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        plan: &StaticEncodingPlan,
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
        graph_arena: Option<&BufferBinding>,
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
                        PlannedRange::GraphArena { offset, len } => {
                            let arena = graph_arena.ok_or(BackendError::ExecutionFailed)?;
                            let address = arena
                                .address
                                .checked_add(
                                    u64::try_from(*offset)
                                        .map_err(|_| BackendError::ExecutionFailed)?,
                                )
                                .ok_or(BackendError::ExecutionFailed)?;
                            BoundRange {
                                base: arena.raw.gpuAddress(),
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
                PlanCall::IndirectDispatch {
                    address,
                    threads_per_threadgroup,
                } => {
                    // SAFETY: The address was captured from a live retained scratch buffer and
                    // names a three-u32 indirect dispatch record initialized by the sample kernel.
                    unsafe {
                        encoder.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                            *address,
                            *threads_per_threadgroup,
                        );
                    }
                }
                PlanCall::Barrier => encode_dispatch_barrier(encoder),
                PlanCall::InternalBoundary => {
                    write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_dispatch(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        dispatch_index: usize,
        profile: bool,
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        temporaries: &mut Vec<BufferBinding>,
        error_flags: &mut Vec<EncodedErrorFlag>,
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
            Op::Argmax { chunk_width } => {
                self.encode_argmax(encoder, table, dispatch, chunk_width, bindings, arguments)?
            }
            Op::TopK {
                k,
                normalize,
                chunk_width,
            } => self.encode_top_k(
                encoder,
                table,
                dispatch,
                (k, normalize, chunk_width),
                bindings,
                arguments,
            )?,
            Op::Sample {
                position,
                chunk_width,
                fast_top_k,
                fast_max_chunks,
                rejection_rounds,
            } => {
                let params = SampleKernelParams {
                    position,
                    chunk_width,
                    fast_top_k,
                    fast_max_chunks,
                    rejection_rounds,
                };
                let (buffers, flag) = match selected_variant(dispatch) {
                    Some(MetalVariant::SamplePortable) => {
                        self.encode_sample(encoder, table, dispatch, params, bindings, arguments)?
                    }
                    Some(MetalVariant::SampleParallel) => self.encode_sample_parallel(
                        encoder, table, dispatch, params, None, bindings, arguments,
                    )?,
                    _ => return Err(BackendError::InvalidInput),
                };
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "sample"));
                buffers
            }
            Op::SampleMasked {
                position,
                chunk_width,
                fast_top_k,
                fast_max_chunks,
                rejection_rounds,
            } => {
                let params = SampleKernelParams {
                    position,
                    chunk_width,
                    fast_top_k,
                    fast_max_chunks,
                    rejection_rounds,
                };
                let allow = dispatch.inputs().get(2).ok_or(BackendError::InvalidInput)?;
                let (buffers, flag) = self.encode_sample_parallel(
                    encoder,
                    table,
                    dispatch,
                    params,
                    Some(allow),
                    bindings,
                    arguments,
                )?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "sample-masked"));
                buffers
            }
            Op::Rope { theta } => {
                self.encode_rope(encoder, table, dispatch, theta, bindings, arguments)?
            }
            Op::Embed => {
                let (buffers, flag) =
                    self.encode_embed(encoder, table, dispatch, bindings, arguments)?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "embed"));
                buffers
            }
            Op::QuantEmbed { bits, group_size } => {
                let (buffers, flag) = self.encode_quant_embed(
                    encoder, table, dispatch, bits, group_size, bindings, arguments,
                )?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "quant-embed"));
                buffers
            }
            Op::Copy => self.encode_copy(encoder, table, dispatch, bindings, arguments)?,
            Op::PagedCopy { .. } => {
                let (buffers, flag) =
                    self.encode_paged_copy(encoder, table, dispatch, bindings, arguments)?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "paged-copy"));
                buffers
            }
            Op::Matmul => self.encode_matmul(encoder, table, dispatch, bindings, arguments)?,
            Op::GatherMatmul => {
                let (buffers, flag) =
                    self.encode_gather_matmul(encoder, table, dispatch, bindings, arguments)?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "gather-matmul"));
                buffers
            }
            Op::QuantMatmul { bits, group_size } => self.encode_quant_matmul(
                encoder, table, dispatch, bits, group_size, bindings, arguments,
            )?,
            Op::GatherQuantMatmul { bits, group_size } => {
                let (buffers, flag) = if dispatch.primitive_inputs().len() == 6 {
                    self.encode_gather_quant_weighted_sum(
                        encoder, table, dispatch, bits, group_size, bindings, arguments,
                    )?
                } else {
                    self.encode_gather_quant_matmul(
                        encoder,
                        table,
                        dispatch,
                        bits,
                        group_size,
                        timestamps,
                        internal_timestamp,
                        bindings,
                        arguments,
                    )?
                };
                error_flags.push(EncodedErrorFlag::new(
                    flag,
                    dispatch_index,
                    "gather-quant-matmul",
                ));
                buffers
            }
            Op::Program(_) => return Err(BackendError::InvalidInput),
            Op::Sdpa { .. } => {
                self.encode_sdpa_dispatch(encoder, table, dispatch, bindings, arguments)?
            }
            Op::PagedSdpa { .. } => {
                let (buffers, flag) =
                    self.encode_paged_sdpa(encoder, table, dispatch, bindings, arguments)?;
                error_flags.push(EncodedErrorFlag::new(flag, dispatch_index, "paged-sdpa"));
                buffers
            }
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

    #[allow(
        clippy::too_many_lines,
        reason = "paged-copy encoding keeps one argument-table layout together"
    )]
    fn encode_paged_copy(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let Op::PagedCopy {
            direction,
            start,
            page_tokens,
        } = dispatch.op()
        else {
            return Err(BackendError::InvalidInput);
        };
        let (first, table_tensor, generations, activity_tensor) = match dispatch.inputs() {
            [first, table_tensor, generations] => (first, table_tensor, generations, None),
            [first, table_tensor, generations, activity] => {
                (first, table_tensor, generations, Some(activity))
            }
            _ => return Err(BackendError::InvalidInput),
        };
        let (ordinary_tensor, pool_tensor) = match direction {
            PagedCopyDirection::IntoPages => (first, dispatch.output()),
            PagedCopyDirection::OutOfPages => (dispatch.output(), first),
        };
        let ordinary = self.encoder_tensor(ordinary_tensor)?;
        let pool = self.encoder_dense_tensor(pool_tensor)?;
        let page_table = self.encoder_dense_tensor(table_tensor)?;
        let generations = self.encoder_dense_tensor(generations)?;
        let has_activity = activity_tensor.is_some();
        let activity = activity_tensor
            .map(|tensor| self.encoder_dense_tensor(tensor))
            .transpose()?;
        let tokens = ordinary
            .layout
            .shape()
            .first()
            .copied()
            .filter(|&tokens| tokens != 0)
            .ok_or(BackendError::InvalidInput)?;
        let elements = u32::try_from(ordinary.layout.element_count() / u64::from(tokens))
            .map_err(|_| BackendError::InvalidInput)?;
        let pool_shape: [u32; 3] = pool
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let table_shape: [u32; 2] = page_table
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let mut params = Vec::with_capacity(7 * size_of::<u32>());
        for value in [
            start,
            tokens,
            elements,
            page_tokens,
            table_shape[0],
            pool_shape[0],
            u32::from(direction == PagedCopyDirection::OutOfPages),
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let params = arguments.write(&params)?;
        let ordinary_layout = Self::layout_buffer(&ordinary.layout, arguments)?;
        let error_flag = arguments.write(&error_state())?;
        let default_activity = activity
            .is_none()
            .then(|| arguments.write(&1_u32.to_ne_bytes()))
            .transpose()?;
        let activity = activity
            .as_ref()
            .map_or_else(|| default_activity.as_ref(), |tensor| Some(&tensor.buffer))
            .ok_or(BackendError::ExecutionFailed)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                "paged_copy",
                &[
                    (0, dtype_code(ordinary.layout.dtype())),
                    (1, dtype_code(pool.layout.dtype())),
                    (216, u32::from(has_activity)),
                ],
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&ordinary, &pool, &page_table, &generations]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &ordinary_layout);
        bindings.bind(table, 5, &params);
        bindings.bind(table, 6, &error_flag);
        bindings.bind(table, 7, activity);
        set_argument_table(encoder, table);
        let elements = usize::try_from(elements).map_err(|_| BackendError::InvalidInput)?;
        let threads = elements.min(pipeline.maxTotalThreadsPerThreadgroup());
        if threads == 0 {
            return Err(BackendError::InvalidInput);
        }
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: elements.div_ceil(threads),
                height: usize::try_from(tokens).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            [ordinary_layout, params, error_flag.clone()]
                .into_iter()
                .chain(default_activity)
                .collect(),
            error_flag,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_paged_copy_out(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        pool: &EncoderTensor,
        page_table: &EncoderTensor,
        generations: &EncoderTensor,
        ordinary: &EncoderTensor,
        tokens: u32,
        page_tokens: u32,
        logical_pages: u32,
        physical_pages: u32,
        error_flag: &BufferBinding,
        activity: &BufferBinding,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let elements = u32::try_from(ordinary.layout.element_count() / u64::from(tokens))
            .map_err(|_| BackendError::InvalidInput)?;
        let mut raw_params = Vec::with_capacity(7 * size_of::<u32>());
        for value in [
            0,
            tokens,
            elements,
            page_tokens,
            logical_pages,
            physical_pages,
            1,
        ] {
            raw_params.extend_from_slice(&value.to_ne_bytes());
        }
        let params = arguments.write(&raw_params)?;
        let ordinary_layout = Self::layout_buffer(&ordinary.layout, arguments)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                "paged_copy",
                &[
                    (0, dtype_code(ordinary.layout.dtype())),
                    (1, dtype_code(pool.layout.dtype())),
                    (216, 1),
                ],
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [ordinary, pool, page_table, generations]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &ordinary_layout);
        bindings.bind(table, 5, &params);
        bindings.bind(table, 6, error_flag);
        bindings.bind(table, 7, activity);
        set_argument_table(encoder, table);
        let elements = usize::try_from(elements).map_err(|_| BackendError::InvalidInput)?;
        let threads = elements.min(pipeline.maxTotalThreadsPerThreadgroup());
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: elements.div_ceil(threads),
                height: usize::try_from(tokens).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok(vec![ordinary_layout, params])
    }

    #[allow(clippy::too_many_lines)]
    fn encode_paged_sdpa(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let Op::PagedSdpa {
            scale,
            causal,
            q_start,
            key_length,
            page_tokens,
        } = dispatch.op()
        else {
            return Err(BackendError::InvalidInput);
        };
        let (query_tensor, key_tensor, value_tensor, table_tensor, generation_tensor, activity) =
            match dispatch.inputs() {
                [query, key, value, table, generations] => {
                    (query, key, value, table, generations, None)
                }
                [query, key, value, table, generations, activity] => {
                    (query, key, value, table, generations, Some(activity))
                }
                _ => return Err(BackendError::InvalidInput),
            };
        let query = self.encoder_tensor(query_tensor)?;
        let key = self.encoder_dense_tensor(key_tensor)?;
        let value = self.encoder_dense_tensor(value_tensor)?;
        let page_table = self.encoder_dense_tensor(table_tensor)?;
        let generations = self.encoder_dense_tensor(generation_tensor)?;
        let has_activity = activity.is_some();
        let activity = activity
            .map(|tensor| self.encoder_dense_tensor(tensor))
            .transpose()?;
        let output = self.encoder_tensor(dispatch.output())?;
        let q: [u32; 3] = query
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let k: [u32; 4] = key
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let v: [u32; 4] = value
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let t: [u32; 2] = page_table
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let gqa_blocks = match selected_variant(dispatch) {
            Some(MetalVariant::PagedSdpaVectorGqaTwoPass32) => Some(32),
            Some(MetalVariant::PagedSdpaVectorGqaTwoPass64) => Some(64),
            Some(MetalVariant::PagedSdpaVectorGqaTwoPass128) => Some(128),
            Some(MetalVariant::PagedSdpaVectorGqaTwoPass256) => Some(256),
            _ => None,
        };
        let mut params = Vec::with_capacity(128);
        params.extend_from_slice(&scale.to_ne_bytes());
        for value in [
            q[0],
            q[1],
            q[2],
            k[2],
            key_length,
            v[3],
            q[0] / k[2],
            q_start,
            u32::from(causal),
            page_tokens,
            t[0],
            k[0],
            gqa_blocks.unwrap_or(0),
            0,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        params.extend_from_slice(&0_u32.to_ne_bytes());
        for value in [
            query.layout.offset(),
            query.layout.strides()[0],
            query.layout.strides()[1],
            query.layout.strides()[2],
            output.layout.offset(),
            output.layout.strides()[0],
            output.layout.strides()[1],
            output.layout.strides()[2],
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let params = arguments.write(&params)?;
        let error_flag = arguments.write(&paged_attention_state())?;
        let default_activity = activity
            .is_none()
            .then(|| arguments.write(&1_u32.to_ne_bytes()))
            .transpose()?;
        let activity = activity
            .as_ref()
            .map_or_else(|| default_activity.as_ref(), |tensor| Some(&tensor.buffer))
            .ok_or(BackendError::ExecutionFailed)?;
        let kernel = match q[2] {
            64 => "paged_sdpa_64",
            128 => "paged_sdpa_128",
            _ => return Err(BackendError::UnsupportedOperation),
        };
        let constants = vec![
            (0, dtype_code(query.layout.dtype())),
            (1, dtype_code(key.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (3, dtype_code(value.layout.dtype())),
            (216, u32::from(has_activity)),
        ];
        let validation = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("paged_sdpa_validate", &[(216, u32::from(has_activity))])?;
        set_pipeline(encoder, &validation);
        for (index, tensor) in [&page_table, &generations].into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 2, &params);
        bindings.bind(table, 3, &error_flag);
        bindings.bind(table, 4, activity);
        set_argument_table(encoder, table);
        if gqa_blocks.is_none() {
            dispatch_threadgroups(
                encoder,
                MTLSize {
                    width: usize::try_from(key_length.div_ceil(page_tokens).div_ceil(64))
                        .map_err(|_| BackendError::InvalidInput)?,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            encode_dispatch_barrier(encoder);
        }
        if selected_variant(dispatch) == Some(MetalVariant::PagedSdpaDecomposed) {
            let key_scratch = self.scratch_tensor(key.layout.dtype(), &[key_length, k[2], k[3]])?;
            let value_scratch =
                self.scratch_tensor(value.layout.dtype(), &[key_length, v[2], v[3]])?;
            let mut temporaries = vec![params.clone(), error_flag.clone()];
            temporaries.extend(default_activity.clone());
            for (pool, ordinary, shape) in [(&key, &key_scratch, k), (&value, &value_scratch, v)] {
                temporaries.extend(self.encode_paged_copy_out(
                    encoder,
                    table,
                    pool,
                    &page_table,
                    &generations,
                    ordinary,
                    key_length,
                    page_tokens,
                    t[0],
                    shape[0],
                    &error_flag,
                    activity,
                    bindings,
                    arguments,
                )?);
            }
            encode_dispatch_barrier(encoder);
            let key_heads = EncoderTensor {
                buffer: key_scratch.buffer.clone(),
                layout: key_scratch
                    .layout
                    .permute(&[1, 0, 2])
                    .map_err(|_| BackendError::InvalidInput)?,
            };
            let value_heads = EncoderTensor {
                buffer: value_scratch.buffer.clone(),
                layout: value_scratch
                    .layout
                    .permute(&[1, 0, 2])
                    .map_err(|_| BackendError::InvalidInput)?,
            };
            let scores = self.scratch_tensor(DType::F32, &[q[0], q[1], key_length])?;
            temporaries.extend(self.encode_sdpa_query_key(
                encoder,
                table,
                &query,
                &key_heads,
                &scores,
                k[2],
                q[0] / k[2],
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
                &value_heads,
                &output,
                k[2],
                q[0] / k[2],
                q[1],
                v[3],
                bindings,
                arguments,
            )?);
            temporaries.extend([key_scratch.buffer, value_scratch.buffer, scores.buffer]);
            return Ok((temporaries, error_flag));
        }
        if let Some(kernel) = match selected_variant(dispatch) {
            Some(MetalVariant::PagedSdpaSteel) => Some(SdpaKernel::Steel),
            Some(MetalVariant::PagedSdpaSteelHalf) => Some(SdpaKernel::SteelHalf),
            _ => None,
        } {
            let (block_query, block_key) = steel_attention_tile(kernel, q[2])?;
            let mut steel_constants = constants.clone();
            steel_constants.extend([
                (214, u32::from(key_length % block_key == 0)),
                (215, u32::from(causal)),
            ]);
            let pipeline = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .get(paged_sdpa_kernel(dispatch, q[2])?, &steel_constants);
            match pipeline {
                Ok(pipeline) => {
                    set_pipeline(encoder, &pipeline);
                    for (index, tensor) in
                        [&query, &key, &value, &page_table, &generations, &output]
                            .into_iter()
                            .enumerate()
                    {
                        bindings.bind(table, index, &tensor.buffer);
                    }
                    bindings.bind(table, 6, &params);
                    bindings.bind(table, 7, &error_flag);
                    bindings.bind(table, 8, activity);
                    set_argument_table(encoder, table);
                    dispatch_threadgroups(
                        encoder,
                        MTLSize {
                            width: usize::try_from(q[1].div_ceil(block_query))
                                .map_err(|_| BackendError::InvalidInput)?,
                            height: usize::try_from(q[0])
                                .map_err(|_| BackendError::InvalidInput)?,
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: usize::try_from(block_query / 8)
                                .map_err(|_| BackendError::InvalidInput)?,
                            depth: 1,
                        },
                    );
                    let mut temporaries = vec![params, error_flag.clone()];
                    temporaries.extend(default_activity.clone());
                    return Ok((temporaries, error_flag));
                }
                Err(BackendError::UnsupportedOperation) => {}
                Err(error) => return Err(error),
            }
        }
        if let Some(gqa_blocks) = gqa_blocks {
            let (first_kernel, first_threads, kv_heads_per_group) =
                paged_gqa_first_pass(q[0], k[2])?;
            let partials = self.scratch_tensor(query.layout.dtype(), &[q[0], gqa_blocks, v[3]])?;
            let sums = self.scratch_tensor(DType::F32, &[q[0], gqa_blocks])?;
            let maxs = self.scratch_tensor(DType::F32, &[q[0], gqa_blocks])?;
            let first = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .get(first_kernel, &constants)?;
            set_pipeline(encoder, &first);
            for (index, tensor) in [
                &query,
                &key,
                &value,
                &page_table,
                &generations,
                &partials,
                &sums,
                &maxs,
            ]
            .into_iter()
            .enumerate()
            {
                bindings.bind(table, index, &tensor.buffer);
            }
            bindings.bind(table, 8, &params);
            bindings.bind(table, 9, &error_flag);
            bindings.bind(table, 10, activity);
            set_argument_table(encoder, table);
            dispatch_threadgroups(
                encoder,
                MTLSize {
                    width: usize::try_from(k[2] / kv_heads_per_group)
                        .map_err(|_| BackendError::InvalidInput)?,
                    height: 1,
                    depth: usize::try_from(gqa_blocks).map_err(|_| BackendError::InvalidInput)?,
                },
                MTLSize {
                    width: first_threads,
                    height: 1,
                    depth: 1,
                },
            );
            encode_dispatch_barrier(encoder);
            let second = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .get("paged_sdpa_2pass_2_128", &constants)?;
            set_pipeline(encoder, &second);
            for (index, tensor) in [&partials, &sums, &maxs, &output].into_iter().enumerate() {
                bindings.bind(table, index, &tensor.buffer);
            }
            bindings.bind(table, 4, &params);
            bindings.bind(table, 5, activity);
            set_argument_table(encoder, table);
            dispatch_threadgroups(
                encoder,
                MTLSize {
                    width: usize::try_from(q[0]).map_err(|_| BackendError::InvalidInput)?,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: second.maxTotalThreadsPerThreadgroup(),
                    height: 1,
                    depth: 1,
                },
            );
            let mut temporaries = vec![
                params,
                error_flag.clone(),
                partials.buffer,
                sums.buffer,
                maxs.buffer,
            ];
            temporaries.extend(default_activity.clone());
            return Ok((temporaries, error_flag));
        }
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < 1_024 {
            return Err(BackendError::UnsupportedOperation);
        }
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&query, &key, &value, &page_table, &generations, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 6, &params);
        bindings.bind(table, 7, &error_flag);
        bindings.bind(table, 8, activity);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(q[0]).map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(q[1]).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 1_024,
                height: 1,
                depth: 1,
            },
        );
        let mut temporaries = vec![params, error_flag.clone()];
        temporaries.extend(default_activity);
        Ok((temporaries, error_flag))
    }

    fn encode_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [left_tensor, right_tensor] = dispatch.primitive_inputs() else {
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
        if let Some(layouts) = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)? {
            temporaries.push(layouts);
        }
        let parameter_buffer = self.encode_matmul_kernel(
            encoder,
            table,
            Some(selected_variant(dispatch).ok_or(BackendError::InvalidInput)?),
            dispatch.prepared_epilogue().map(AsRef::as_ref),
            &left,
            &right,
            &output,
            bindings,
            arguments,
        )?;
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

    fn bind_epilogue_arguments(
        &self,
        dispatch: &Dispatch,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Option<BufferBinding>, BackendError> {
        if dispatch.prepared_epilogue().is_none() {
            return Ok(None);
        }
        let extras = &dispatch.inputs()[dispatch.primitive_inputs().len()..];
        if extras.len() > epilogue_codegen::EXTRA_CAPACITY {
            return Err(BackendError::InvalidInput);
        }
        let mut layouts = Vec::with_capacity(extras.len().saturating_mul(112));
        for (slot, extra) in extras.iter().enumerate() {
            let extra = self.encoder_tensor(extra)?;
            bindings.bind(table, 16 + slot, &extra.buffer);
            layouts.extend_from_slice(&encode_layout(&extra.layout)?);
        }
        if layouts.is_empty() {
            return Ok(None);
        }
        let layouts = arguments.write(&layouts)?;
        bindings.bind(table, 23, &layouts);
        Ok(Some(layouts))
    }

    fn encode_gather_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [input, weights, indices] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let weights = self.encoder_tensor(weights)?;
        let indices = self.encoder_tensor(indices)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let [rows, inner]: [u32; 2] = input
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let [experts, _, columns]: [u32; 3] = weights
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let [_, routes]: [u32; 2] = indices
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let mut params = Vec::with_capacity(136);
        for value in [
            input.layout.offset(),
            weights.layout.offset(),
            indices.layout.offset(),
            output.layout.offset(),
        ]
        .into_iter()
        .chain(input.layout.strides().iter().copied())
        .chain(weights.layout.strides().iter().copied())
        .chain(indices.layout.strides().iter().copied())
        .chain(output.layout.strides().iter().copied())
        {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        for value in [rows, routes, inner, experts, columns, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let params = arguments.write(&params)?;
        let mut error_state = [0_u8; 8];
        error_state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
        let error_flag = arguments.write(&error_state)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                "gather_gemv",
                &[
                    (0, dtype_code(input.layout.dtype())),
                    (1, dtype_code(weights.layout.dtype())),
                    (2, dtype_code(output.layout.dtype())),
                ],
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&input, &weights, &indices, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &params);
        bindings.bind(table, 5, &error_flag);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(16))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(
                    rows.checked_mul(routes).ok_or(BackendError::InvalidInput)?,
                )
                .map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((vec![params, error_flag.clone()], error_flag))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_quant_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bits: u8,
        group_size: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let [input, packed, scales, biases] = dispatch.primitive_inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let packed = self.encoder_tensor(packed)?;
        let scales = self.encoder_tensor(scales)?;
        let biases = self.encoder_tensor(biases)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let (params, [rows, inner, columns]) = Self::quant_matmul_params(
            [&input, &packed, &scales, &biases, &output],
            bits,
            group_size,
        )?;
        let params = arguments.write(&params)?;
        let kernel = quant_matmul_kernel(dispatch)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (1, dtype_code(scales.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (12, u32::from(bits)),
            (13, group_size),
            (15, quant_epilogue_specialization(columns, inner)),
        ];
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = if let Some(epilogue) = dispatch.prepared_epilogue() {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::Quant,
                epilogue,
                kernel,
                &constants,
            )?
        } else {
            pipelines.get(kernel, &constants)?
        };
        drop(pipelines);
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&input, &packed, &scales, &biases, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 5, &params);
        let mut temporaries = vec![params];
        if let Some(layouts) = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)? {
            temporaries.push(layouts);
        }
        set_argument_table(encoder, table);
        let (column_tile, row_tile, threads) = match kernel {
            "quantized_gemv" => (if bits == 8 { 16 } else { 8 }, 1, 64),
            "quantized_gemv_q8_fast" => (8, 1, 64),
            "quantized_gemm_tiled_f32"
            | "quantized_gemm_tiled_f16"
            | "quantized_gemm_tiled_bf16" => (32, 32, 128),
            _ => (8, 8, 256),
        };
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(column_tile))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(rows.div_ceil(row_tile))
                    .map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        Ok(temporaries)
    }

    fn quant_matmul_params(
        tensors: [&EncoderTensor; 5],
        bits: u8,
        group_size: u32,
    ) -> Result<(Vec<u8>, [u32; 3]), BackendError> {
        let [input, packed, scales, biases, output] = tensors;
        let [rows, inner]: [u32; 2] = input
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let [columns, packed_width]: [u32; 2] = packed
            .layout
            .shape()
            .try_into()
            .map_err(|_| BackendError::InvalidInput)?;
        let mut params = Vec::with_capacity(144);
        for value in [
            input.layout.offset(),
            packed.layout.offset(),
            scales.layout.offset(),
            biases.layout.offset(),
            output.layout.offset(),
            input.layout.strides()[0],
            input.layout.strides()[1],
            packed.layout.strides()[0],
            packed.layout.strides()[1],
            scales.layout.strides()[0],
            scales.layout.strides()[1],
            biases.layout.strides()[0],
            biases.layout.strides()[1],
            output.layout.strides()[0],
            output.layout.strides()[1],
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        for value in [
            rows,
            inner,
            columns,
            packed_width,
            u32::from(bits),
            group_size,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        Ok((params, [rows, inner, columns]))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_gather_quant_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bits: u8,
        group_size: u32,
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        if dispatch.primitive_inputs().len() == 8 {
            return self.encode_gather_quant_matmul_pair(
                encoder,
                table,
                dispatch,
                bits,
                group_size,
                timestamps,
                internal_timestamp,
                bindings,
                arguments,
            );
        }
        let [input, packed, scales, biases, indices] = dispatch.primitive_inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let packed = self.encoder_tensor(packed)?;
        let scales = self.encoder_tensor(scales)?;
        let biases = self.encoder_tensor(biases)?;
        let indices = self.encoder_tensor(indices)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let (params, [rows, routes, columns]) = gather_quant_params(
            [&input, &packed, &scales, &biases, &indices, &output],
            (bits, group_size),
            grouped_block_rows(dispatch),
        )?;
        if grouped_gather_dispatch(dispatch) {
            return self.encode_grouped_quant_matmul(
                encoder,
                table,
                dispatch,
                [&input, &packed, &scales, &biases, &indices, &output],
                &params,
                [rows, routes, columns],
                (bits, group_size),
                timestamps,
                internal_timestamp,
                bindings,
                arguments,
            );
        }
        let inner = input
            .layout
            .shape()
            .last()
            .copied()
            .ok_or(BackendError::InvalidInput)?;
        let words_per_thread = gather_epilogue_specialization(columns, inner);
        let params = arguments.write(&params)?;
        let mut error_state = [0_u8; 8];
        error_state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
        let error_flag = arguments.write(&error_state)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (1, dtype_code(scales.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (12, u32::from(bits)),
            (13, group_size),
            (14, words_per_thread),
        ];
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = if let Some(epilogue) = dispatch.prepared_epilogue() {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue,
                "gather_quantized_gemv",
                &constants,
            )?
        } else {
            pipelines.get("gather_quantized_gemv", &constants)?
        };
        drop(pipelines);
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&input, &packed, &scales, &biases, &indices, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 6, &params);
        bindings.bind(table, 7, &error_flag);
        let mut temporaries = vec![params, error_flag.clone()];
        if let Some(layouts) = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)? {
            temporaries.push(layouts);
        }
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(8))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(
                    rows.checked_mul(routes).ok_or(BackendError::InvalidInput)?,
                )
                .map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gather_quant_weighted_sum(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bits: u8,
        group_size: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [input, packed, scales, biases, indices, weights] = dispatch.primitive_inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let packed = self.encoder_tensor(packed)?;
        let scales = self.encoder_tensor(scales)?;
        let biases = self.encoder_tensor(biases)?;
        let indices = self.encoder_tensor(indices)?;
        let weights = self.encoder_tensor(weights)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let tensors = [
            &input, &packed, &scales, &biases, &indices, &weights, &output,
        ];
        let (params, [rows, routes, columns]) =
            gather_quant_combine_params(tensors, (bits, group_size))?;
        let inner = input.layout.shape()[2];
        let words_per_thread = u32::from(columns % 4 == 0 && inner % 512 == 0) + 1;
        let schedule = weighted_gather_schedule(selected_variant(dispatch), routes)?;
        let params = arguments.write(&params)?;
        let error_flag = grouped_error_flag(arguments)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (1, dtype_code(scales.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (12, u32::from(bits)),
            (13, group_size),
            (14, words_per_thread),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .lookup_epilogue(
                epilogue_codegen::FamilySourceTag::GatherQuant,
                dispatch
                    .prepared_epilogue()
                    .ok_or(BackendError::InvalidInput)?,
                schedule.implementation,
                &constants,
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in tensors.into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 7, &params);
        bindings.bind(table, 8, &error_flag);
        let layouts = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)?;
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(4))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(rows).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: usize::try_from(schedule.threadgroup_width)
                    .map_err(|_| BackendError::InvalidInput)?,
                height: 1,
                depth: 1,
            },
        );
        let mut temporaries = vec![params, error_flag.clone()];
        temporaries.extend(layouts);
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_gather_quant_matmul_pair(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bits: u8,
        group_size: u32,
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [
            input,
            first_packed,
            first_scales,
            first_biases,
            second_packed,
            second_scales,
            second_biases,
            indices,
        ] = dispatch.primitive_inputs()
        else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let first_packed = self.encoder_tensor(first_packed)?;
        let first_scales = self.encoder_tensor(first_scales)?;
        let first_biases = self.encoder_tensor(first_biases)?;
        let second_packed = self.encoder_tensor(second_packed)?;
        let second_scales = self.encoder_tensor(second_scales)?;
        let second_biases = self.encoder_tensor(second_biases)?;
        let indices = self.encoder_tensor(indices)?;
        let output = self.encoder_tensor(dispatch.output())?;
        let config = (bits, group_size);
        let (first_params, [rows, routes, columns]) = gather_quant_params(
            [
                &input,
                &first_packed,
                &first_scales,
                &first_biases,
                &indices,
                &output,
            ],
            config,
            grouped_block_rows(dispatch),
        )?;
        let (second_params, _) = gather_quant_params(
            [
                &input,
                &second_packed,
                &second_scales,
                &second_biases,
                &indices,
                &output,
            ],
            config,
            grouped_block_rows(dispatch),
        )?;
        if grouped_gather_dispatch(dispatch) {
            return self.encode_grouped_quant_pair(
                encoder,
                table,
                dispatch,
                [
                    &input,
                    &first_packed,
                    &first_scales,
                    &first_biases,
                    &second_packed,
                    &second_scales,
                    &second_biases,
                    &indices,
                    &output,
                ],
                &[first_params, second_params],
                [rows, routes, columns],
                config,
                timestamps,
                internal_timestamp,
                bindings,
                arguments,
            );
        }
        let inner = input
            .layout
            .shape()
            .last()
            .copied()
            .ok_or(BackendError::InvalidInput)?;
        let words_per_thread = gather_epilogue_specialization(columns, inner);
        let first_params = arguments.write(&first_params)?;
        let second_params = arguments.write(&second_params)?;
        let error_flag = grouped_error_flag(arguments)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (1, dtype_code(first_scales.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (12, u32::from(bits)),
            (13, group_size),
            (14, words_per_thread),
        ];
        let epilogue = dispatch
            .prepared_epilogue()
            .ok_or(BackendError::InvalidInput)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .lookup_epilogue(
                epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue,
                "gather_quantized_pair",
                &constants,
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [
            &input,
            &first_packed,
            &first_scales,
            &first_biases,
            &second_packed,
            &second_scales,
            &second_biases,
            &indices,
            &output,
        ]
        .into_iter()
        .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 9, &first_params);
        bindings.bind(table, 10, &second_params);
        bindings.bind(table, 11, &error_flag);
        let mut temporaries = vec![first_params, second_params, error_flag.clone()];
        if let Some(layouts) = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)? {
            temporaries.push(layouts);
        }
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(columns.div_ceil(8))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(
                    rows.checked_mul(routes).ok_or(BackendError::InvalidInput)?,
                )
                .map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_grouped_quant_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        tensors: [&EncoderTensor; 6],
        params: &[u8],
        shape: [u32; 3],
        config: (u8, u32),
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [input, packed, scales, biases, indices, output] = tensors;
        let params = arguments.write(params)?;
        let error_flag = grouped_error_flag(arguments)?;
        let experts = packed.layout.shape()[0];
        let block_rows = grouped_block_rows(dispatch);
        let grouped = GroupedRouteShape::new(shape[0], shape[1], experts, block_rows)?;
        let sorted = self.scratch_tensor(DType::U32, &[grouped.route_count])?;
        let offsets = self.scratch_tensor(
            DType::U32,
            &[experts.checked_add(1).ok_or(BackendError::InvalidInput)?],
        )?;
        let blocks = self.scratch_tensor(DType::U32, &[grouped.block_elements])?;
        self.encode_expert_buckets(
            encoder,
            table,
            indices,
            &sorted,
            &offsets,
            &blocks,
            output,
            &params,
            &error_flag,
            block_rows,
            timestamps,
            internal_timestamp,
            bindings,
        )?;
        encode_dispatch_barrier(encoder);
        let layouts = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)?;
        self.encode_grouped_quantized_gemm(
            encoder,
            table,
            [input, packed, scales, biases, &sorted, &blocks, output],
            &params,
            shape,
            grouped,
            block_rows,
            config,
            dispatch.prepared_epilogue().map(AsRef::as_ref),
            bindings,
        )?;
        let mut temporaries = vec![
            params,
            error_flag.clone(),
            sorted.buffer,
            offsets.buffer,
            blocks.buffer,
        ];
        temporaries.extend(layouts);
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_grouped_quant_pair(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        tensors: [&EncoderTensor; 9],
        params: &[Vec<u8>; 2],
        shape: [u32; 3],
        config: (u8, u32),
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [
            input,
            gate_packed,
            gate_scales,
            gate_biases,
            up_packed,
            up_scales,
            up_biases,
            indices,
            output,
        ] = tensors;
        let experts = gate_packed.layout.shape()[0];
        let block_rows = grouped_block_rows(dispatch);
        let grouped = GroupedRouteShape::new(shape[0], shape[1], experts, block_rows)?;
        let gate = self.scratch_tensor(DType::F32, &shape)?;
        let up = self.scratch_tensor(DType::F32, &shape)?;
        let sorted = self.scratch_tensor(DType::U32, &[grouped.route_count])?;
        let offsets = self.scratch_tensor(
            DType::U32,
            &[experts.checked_add(1).ok_or(BackendError::InvalidInput)?],
        )?;
        let blocks = self.scratch_tensor(DType::U32, &[grouped.block_elements])?;
        let (gate_qmm_params, _) = gather_quant_params(
            [input, gate_packed, gate_scales, gate_biases, indices, &gate],
            config,
            block_rows,
        )?;
        let (up_qmm_params, _) = gather_quant_params(
            [input, up_packed, up_scales, up_biases, indices, &up],
            config,
            block_rows,
        )?;
        let final_params = arguments.write(&params[0])?;
        let gate_qmm_params = arguments.write(&gate_qmm_params)?;
        let up_qmm_params = arguments.write(&up_qmm_params)?;
        let error_flag = grouped_error_flag(arguments)?;
        self.encode_expert_buckets(
            encoder,
            table,
            indices,
            &sorted,
            &offsets,
            &blocks,
            output,
            &final_params,
            &error_flag,
            block_rows,
            timestamps,
            internal_timestamp,
            bindings,
        )?;
        encode_dispatch_barrier(encoder);
        self.encode_grouped_quantized_gemm(
            encoder,
            table,
            [
                input,
                gate_packed,
                gate_scales,
                gate_biases,
                &sorted,
                &blocks,
                &gate,
            ],
            &gate_qmm_params,
            shape,
            grouped,
            block_rows,
            config,
            None,
            bindings,
        )?;
        write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
        self.encode_grouped_quantized_gemm(
            encoder,
            table,
            [
                input, up_packed, up_scales, up_biases, &sorted, &blocks, &up,
            ],
            &up_qmm_params,
            shape,
            grouped,
            block_rows,
            config,
            None,
            bindings,
        )?;
        write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
        encode_dispatch_barrier(encoder);
        let constants = [(2, dtype_code(output.layout.dtype()))];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .lookup_epilogue(
                epilogue_codegen::FamilySourceTag::GatherQuant,
                dispatch
                    .prepared_epilogue()
                    .ok_or(BackendError::InvalidInput)?,
                "gathered_pair_from_projections",
                &constants,
            )?;
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [&gate, &up, indices, output].into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &final_params);
        let layouts = self.bind_epilogue_arguments(dispatch, table, bindings, arguments)?;
        set_argument_table(encoder, table);
        let elements = grouped
            .route_count
            .checked_mul(shape[2])
            .ok_or(BackendError::InvalidInput)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(elements.div_ceil(256))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        let mut temporaries = vec![
            final_params,
            gate_qmm_params,
            up_qmm_params,
            error_flag.clone(),
            sorted.buffer,
            offsets.buffer,
            blocks.buffer,
            gate.buffer,
            up.buffer,
        ];
        temporaries.extend(layouts);
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_expert_buckets(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        indices: &EncoderTensor,
        sorted: &EncoderTensor,
        offsets: &EncoderTensor,
        blocks: &EncoderTensor,
        output: &EncoderTensor,
        params: &BufferBinding,
        error_flag: &BufferBinding,
        block_rows: u32,
        timestamps: Option<&GpuTimestamps>,
        internal_timestamp: &mut usize,
        bindings: &mut ArgumentBindings,
    ) -> Result<(), BackendError> {
        let count_pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                "count_expert_routes",
                &[(2, dtype_code(output.layout.dtype()))],
            )?;
        set_pipeline(encoder, &count_pipeline);
        bindings.bind(table, 0, &indices.buffer);
        bindings.bind(table, 1, &offsets.buffer);
        bindings.bind(table, 2, &output.buffer);
        bindings.bind(table, 3, params);
        bindings.bind(table, 4, error_flag);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
        encode_dispatch_barrier(encoder);
        let arrange_pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("arrange_expert_routes", &[(16, block_rows)])?;
        set_pipeline(encoder, &arrange_pipeline);
        bindings.bind(table, 0, &indices.buffer);
        bindings.bind(table, 1, &sorted.buffer);
        bindings.bind(table, 2, &offsets.buffer);
        bindings.bind(table, 3, &blocks.buffer);
        bindings.bind(table, 4, params);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_grouped_quantized_gemm(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        tensors: [&EncoderTensor; 7],
        params: &BufferBinding,
        shape: [u32; 3],
        grouped: GroupedRouteShape,
        block_rows: u32,
        config: (u8, u32),
        epilogue: Option<&PreparedEpilogue>,
        bindings: &mut ArgumentBindings,
    ) -> Result<(), BackendError> {
        let [input, packed, scales, biases, sorted, blocks, output] = tensors;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (1, dtype_code(scales.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
            (12, u32::from(config.0)),
            (13, config.1),
            (16, block_rows),
        ];
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let kernel = if block_rows == 8 {
            "grouped_quantized_gemm_small_m"
        } else {
            "grouped_quantized_gemm"
        };
        let pipeline = if let Some(epilogue) = epilogue {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue,
                kernel,
                &constants,
            )?
        } else {
            pipelines.get(kernel, &constants)?
        };
        drop(pipelines);
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in [input, packed, scales, biases, sorted, blocks, output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 7, params);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(shape[2].div_ceil(32))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(grouped.block_capacity)
                    .map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
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
        match self.prepare_sdpa_kernel(dispatch)? {
            kernel @ (SdpaKernel::Vector | SdpaKernel::VectorGqa) => self.encode_vector_sdpa(
                encoder, table, dispatch, scale, causal, q_start, kernel, bindings, arguments,
            ),
            kernel @ (SdpaKernel::Steel | SdpaKernel::SteelHalf) => self.encode_steel_sdpa(
                encoder, table, dispatch, scale, causal, q_start, kernel, bindings, arguments,
            ),
            SdpaKernel::Decomposed => self.encode_decomposed_sdpa(
                encoder, table, dispatch, scale, causal, q_start, bindings, arguments,
            ),
        }
    }

    fn prepare_sdpa_kernel(&self, dispatch: &Dispatch) -> Result<SdpaKernel, BackendError> {
        let kernel = select_sdpa(dispatch)?;
        let [query, key, value] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let [_, _, width] = shape3(query.layout())?;
        let (name, constants, fallback) = match kernel {
            kernel @ (SdpaKernel::Vector | SdpaKernel::VectorGqa) => {
                let fast_f32 = vector_f32_supported(dispatch);
                let prefix = if vector_two_pass_blocks(dispatch).is_some() {
                    if kernel == SdpaKernel::VectorGqa {
                        "mlx_sdpa_vector_2pass_1_gqa"
                    } else if fast_f32 {
                        "mlx_sdpa_vector_2pass_1_f32"
                    } else {
                        "mlx_sdpa_vector_2pass_1"
                    }
                } else if fast_f32 {
                    "mlx_sdpa_vector_f32"
                } else {
                    "mlx_sdpa_vector"
                };
                (
                    vector_kernel(prefix, width)?,
                    vec![
                        (0, dtype_code(query.layout().dtype())),
                        (1, dtype_code(key.layout().dtype())),
                        (2, dtype_code(dispatch.output().layout().dtype())),
                        (3, dtype_code(value.layout().dtype())),
                    ],
                    if kernel == SdpaKernel::VectorGqa {
                        SdpaKernel::Vector
                    } else {
                        SdpaKernel::Decomposed
                    },
                )
            }
            SdpaKernel::Steel | SdpaKernel::SteelHalf => (
                steel_attention_kernel(kernel, width)?,
                steel_attention_constants(dispatch, kernel)?.to_vec(),
                SdpaKernel::Decomposed,
            ),
            SdpaKernel::Decomposed => return Ok(kernel),
        };
        match self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(name, &constants)
        {
            Ok(_) => Ok(kernel),
            Err(BackendError::UnsupportedOperation) => Ok(fallback),
            Err(error) => Err(error),
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
        kernel: SdpaKernel,
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
        let [kv_heads, _, _] = shape3(&key.layout)?;
        let [_, _, value_width] = shape3(&value.layout)?;
        let fast_f32 = vector_f32_supported(dispatch);
        let heads_per_group = query_heads / kv_heads;
        let two_pass_blocks = vector_two_pass_blocks(dispatch);
        let blocks = two_pass_blocks.unwrap_or(1);
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
        if two_pass_blocks.is_none() {
            let pipeline = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .get(
                    vector_kernel(
                        if fast_f32 {
                            "mlx_sdpa_vector_f32"
                        } else {
                            "mlx_sdpa_vector"
                        },
                        width,
                    )?,
                    &constants,
                )?;
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
                    width: pipeline.maxTotalThreadsPerThreadgroup(),
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
            .get(
                vector_kernel(
                    if kernel == SdpaKernel::VectorGqa {
                        "mlx_sdpa_vector_2pass_1_gqa"
                    } else if fast_f32 {
                        "mlx_sdpa_vector_2pass_1_f32"
                    } else {
                        "mlx_sdpa_vector_2pass_1"
                    },
                    width,
                )?,
                &constants,
            )?;
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
            .get(
                vector_kernel(
                    if fast_f32 {
                        "mlx_sdpa_vector_2pass_2_f32"
                    } else {
                        "mlx_sdpa_vector_2pass_2"
                    },
                    width,
                )?,
                &constants,
            )?;
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
                width: second.maxTotalThreadsPerThreadgroup(),
                height: 1,
                depth: 1,
            },
        );
        temporaries.extend([intermediate.buffer, sums.buffer, maxs.buffer]);
        Ok(temporaries)
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
        kernel: SdpaKernel,
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
        let (block_query, _) = steel_attention_tile(kernel, width)?;
        let constants = steel_attention_constants(dispatch, kernel)?;
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(steel_attention_kernel(kernel, width)?, &constants)?;
        let params = Self::sdpa_params(
            &query, &key, &value, &output, scale, causal, q_start, 1, arguments,
        )?;
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
                height: usize::try_from(block_query / 8).map_err(|_| BackendError::InvalidInput)?,
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
            let matmul_variant = sdpa_matmul_variant(query_group.layout.shape()[1], true);
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
                Some(matmul_variant),
                None,
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
                Some(sdpa_matmul_variant(query_length, false)),
                None,
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
        selected: Option<MetalVariant>,
        epilogue: Option<&PreparedEpilogue>,
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
            batch,
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
        let launch = self.matmul_launch(shape, selected, epilogue)?;
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

    fn matmul_launch(
        &self,
        shape: MatmulShape,
        selected: Option<MetalVariant>,
        epilogue: Option<&PreparedEpilogue>,
    ) -> Result<MatmulLaunch, BackendError> {
        let (kernel, block_rows, block_columns, thread_count, constants) =
            if shape.rows == 1 && selected != Some(MetalVariant::MatmulPortable) {
                let transposed = match selected {
                    Some(MetalVariant::MatmulGemv) => false,
                    Some(MetalVariant::MatmulGemvTransposed) => true,
                    Some(_) => return Err(BackendError::InvalidInput),
                    None => DEFAULT_GEMV_TRANSPOSED,
                };
                let (kernel, block_columns, thread_count) = if transposed {
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
                let config = match selected {
                    Some(MetalVariant::MatmulPortable | MetalVariant::MatmulSteel64x32x32_2x2) => {
                        GEMM_64_32
                    }
                    Some(MetalVariant::MatmulSteel64x64x16_2x2) => GEMM_64_64_FLOAT,
                    Some(MetalVariant::MatmulSteel64x64x16_1x2) => GEMM_64_64_HALF,
                    Some(MetalVariant::MatmulSteel32x64x16_1x2) => GEMM_32_64,
                    Some(_) => return Err(BackendError::InvalidInput),
                    None => select_internal_gemm(
                        shape.left_dtype,
                        shape.batch,
                        shape.rows,
                        shape.columns,
                        shape.inner,
                        shape.left_column_major,
                        shape.right_column_major,
                    )
                    .ok_or(BackendError::InvalidInput)?,
                };
                let mut constants = vec![
                    (0, dtype_code(shape.left_dtype)),
                    (1, dtype_code(shape.right_dtype)),
                    (2, dtype_code(shape.output_dtype)),
                ];
                constants.extend(dense_epilogue_specialization(
                    config,
                    shape.rows,
                    shape.columns,
                    shape.inner,
                ));
                (
                    config.kernel,
                    config.block_rows,
                    config.block_columns,
                    config.thread_count,
                    constants,
                )
            };
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = if let Some(epilogue) = epilogue {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::Dense,
                epilogue,
                kernel,
                &constants,
            )?
        } else {
            pipelines.get(kernel, &constants)?
        };
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

        let vectorized = vectorized_copy(&input.layout, &output.layout);
        let kernel = if vectorized {
            "copy_contiguous_f32x4"
        } else if input.layout.is_contiguous() && output.layout.is_contiguous() {
            "copy_contiguous"
        } else {
            "copy_strided"
        };
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (2, dtype_code(output.layout.dtype())),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, if vectorized { &[] } else { &constants })?;
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
        let count = usize::try_from(output.layout.element_count() / if vectorized { 4 } else { 1 })
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
        let buffer = buffers.get(tensor)?;
        let layout = tensor.layout();
        let span = layout.byte_span();
        let view_offset = usize::try_from(span.start).map_err(|_| BackendError::InvalidInput)?;
        let view_len =
            usize::try_from(span.end - span.start).map_err(|_| BackendError::InvalidInput)?;
        if span.end > u64::try_from(buffer.len).map_err(|_| BackendError::InvalidInput)? {
            return Err(BackendError::InvalidInput);
        }
        let binding_offset = buffer
            .data_offset
            .checked_add(view_offset)
            .ok_or(BackendError::InvalidInput)?;
        let offset = u64::try_from(binding_offset).map_err(|_| BackendError::ExecutionFailed)?;
        let address = buffer
            .raw
            .gpuAddress()
            .checked_add(offset)
            .ok_or(BackendError::ExecutionFailed)?;
        let normalized = Layout::new(
            layout.dtype(),
            0,
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            u64::try_from(view_len).map_err(|_| BackendError::InvalidInput)?,
        )
        .map_err(|_| BackendError::InvalidInput)?;
        Ok(EncoderTensor {
            buffer: BufferBinding {
                raw: buffer.raw.clone(),
                offset: binding_offset,
                len: view_len,
                address,
            },
            layout: normalized,
        })
    }

    fn encoder_dense_tensor(&self, tensor: &Tensor) -> Result<EncoderTensor, BackendError> {
        if !tensor.layout().is_contiguous() {
            return Err(BackendError::InvalidInput);
        }
        let mut tensor = self.encoder_tensor(tensor)?;
        let span = tensor.layout.byte_span();
        let offset = usize::try_from(span.start).map_err(|_| BackendError::InvalidInput)?;
        let len = usize::try_from(span.end - span.start).map_err(|_| BackendError::InvalidInput)?;
        tensor.buffer.offset = tensor
            .buffer
            .offset
            .checked_add(offset)
            .ok_or(BackendError::InvalidInput)?;
        tensor.buffer.address = tensor
            .buffer
            .address
            .checked_add(u64::try_from(offset).map_err(|_| BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)?;
        tensor.buffer.len = len;
        Ok(tensor)
    }

    fn scratch_tensor(&self, dtype: DType, shape: &[u32]) -> Result<EncoderTensor, BackendError> {
        let elements = shape
            .iter()
            .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
        let byte_len = elements
            .and_then(|count| count.checked_mul(dtype.byte_size()))
            .ok_or(BackendError::AllocationFailed)?;
        let len = usize::try_from(byte_len).map_err(|_| BackendError::AllocationFailed)?;
        let buffer = SUBMISSION_SCRATCH.with(|current| {
            let scratch = current.borrow().clone();
            scratch.map_or_else(
                || self.new_buffer(len),
                |scratch| {
                    scratch
                        .lock()
                        .map_err(|_| BackendError::ExecutionFailed)?
                        .acquire(self, len)
                },
            )
        })?;
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
            bindings.bind_raw(table, index, buffers.get(tensor)?)?;
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
            bindings.bind_raw(table, index, buffers.get(tensor)?)?;
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

    #[allow(clippy::too_many_arguments)]
    fn encode_quant_embed(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bits: u8,
        group_size: u32,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [packed, scales, biases, ids] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let vocab = packed.layout().shape()[0];
        let width = output.layout().shape()[1];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(
                "quant_embed",
                &[
                    (1, dtype_code(scales.layout().dtype())),
                    (2, dtype_code(output.layout().dtype())),
                    (12, u32::from(bits)),
                    (13, group_size),
                ],
            )?;
        set_pipeline(encoder, &pipeline);
        let mut params = [0_u8; 8];
        params[..4].copy_from_slice(&vocab.to_ne_bytes());
        params[4..].copy_from_slice(&width.to_ne_bytes());
        let mut error_state = [0_u8; 8];
        error_state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
        let error_flag = arguments.write(&error_state)?;
        let temporaries = vec![
            Self::layout_buffer(packed.layout(), arguments)?,
            Self::layout_buffer(scales.layout(), arguments)?,
            Self::layout_buffer(biases.layout(), arguments)?,
            Self::layout_buffer(ids.layout(), arguments)?,
            Self::layout_buffer(output.layout(), arguments)?,
            arguments.write(&params)?,
            error_flag.clone(),
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [packed, scales, biases, ids, output]
            .into_iter()
            .enumerate()
        {
            bindings.bind_raw(table, index, buffers.get(tensor)?)?;
            bindings.bind(table, index + 5, &temporaries[index]);
        }
        bindings.bind(table, 10, &temporaries[5]);
        bindings.bind(table, 11, &temporaries[6]);
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
        let pipeline = self.row_pipeline("softmax_single", "softmax_looped", width, &constants)?;
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
        chunk_width: u32,
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
        let chunks = width.div_ceil(chunk_width);
        let rows = u32::try_from(output.layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let scratch = self.scratch_tensor(DType::U32, &[rows, chunks, 2])?;
        let (partials_pipeline, finalize_pipeline) =
            self.argmax_pipelines(input.layout(), width, chunk_width)?;
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
        bindings.bind_raw(table, 0, buffers.get(input)?)?;
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
        bindings.bind_raw(table, 1, buffers.get(output)?)?;
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

    fn argmax_pipelines(
        &self,
        input: &Layout,
        width: u32,
        chunk_width: u32,
    ) -> Result<(ComputePipeline, ComputePipeline), BackendError> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let partials = if vectorized_argmax(input, width, chunk_width) {
            pipelines.get(
                "argmax_partials_f32x4",
                &[(REDUCTION_CHUNK_WIDTH_CONSTANT, chunk_width)],
            )?
        } else {
            pipelines.get(
                "argmax_partials",
                &[
                    (0, dtype_code(input.dtype())),
                    (REDUCTION_CHUNK_WIDTH_CONSTANT, chunk_width),
                ],
            )?
        };
        Ok((partials, pipelines.get("argmax_finalize", &[])?))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_top_k_direct(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        input: &EncoderTensor,
        values: &EncoderTensor,
        indices: &EncoderTensor,
        width: u32,
        rows: u32,
        k: u32,
        normalize: bool,
        kernel: &str,
        requested_threads: usize,
        constants: &[(u32, u32)],
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, constants)?;
        let temporaries = vec![
            Self::layout_buffer(&input.layout, arguments)?,
            Self::layout_buffer(&values.layout, arguments)?,
            Self::layout_buffer(&indices.layout, arguments)?,
            arguments.write(&width.to_ne_bytes())?,
            arguments.write(&k.to_ne_bytes())?,
            arguments.write(&u32::from(normalize).to_ne_bytes())?,
        ];
        for (index, tensor) in [input, values, indices].into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        for (index, temporary) in temporaries.iter().enumerate() {
            bindings.bind(table, index + 3, temporary);
        }
        set_pipeline(encoder, &pipeline);
        set_argument_table(encoder, table);
        let threads = simd_thread_count(&pipeline, requested_threads)?;
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
        Ok(temporaries)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_top_k(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        config: (u32, bool, u32),
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<Vec<BufferBinding>, BackendError> {
        let (k, normalize, chunk_width) = config;
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let [values, indices] = dispatch.outputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let values = self.encoder_tensor(values)?;
        let indices = self.encoder_tensor(indices)?;
        let width = *input
            .layout
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let chunks = width.div_ceil(chunk_width);
        let rows = u32::try_from(values.layout.element_count() / u64::from(k))
            .map_err(|_| BackendError::ExecutionFailed)?;
        let constants = [
            (0, dtype_code(input.layout.dtype())),
            (2, dtype_code(values.layout.dtype())),
        ];
        if selected_variant(dispatch) == Some(MetalVariant::TopKPortable) {
            let requested_threads = usize::try_from(width)
                .map_err(|_| BackendError::ExecutionFailed)?
                .next_multiple_of(32)
                .min(256);
            return self.encode_top_k_direct(
                encoder,
                table,
                &input,
                &values,
                &indices,
                width,
                rows,
                k,
                normalize,
                "topk_exact_fallback",
                requested_threads,
                &constants,
                bindings,
                arguments,
            );
        }
        if selected_variant(dispatch) == Some(MetalVariant::TopKSingleK8) {
            let requested_threads = usize::try_from(width)
                .map_err(|_| BackendError::ExecutionFailed)?
                .next_multiple_of(32)
                .min(256);
            return self.encode_top_k_direct(
                encoder,
                table,
                &input,
                &values,
                &indices,
                width,
                rows,
                k,
                normalize,
                "topk_single",
                requested_threads,
                &constants,
                bindings,
                arguments,
            );
        }
        let scratch = self.scratch_tensor(DType::U32, &[rows, chunks, k, 2])?;
        let partial_constants = [
            (0, dtype_code(input.layout.dtype())),
            (2, dtype_code(values.layout.dtype())),
            (REDUCTION_CHUNK_WIDTH_CONSTANT, chunk_width),
        ];
        let (partials_pipeline, finalize_pipeline) = {
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            (
                pipelines.get("topk_partials", &partial_constants)?,
                pipelines.get("topk_finalize", &constants)?,
            )
        };
        let temporaries = vec![
            scratch.buffer.clone(),
            Self::layout_buffer(&input.layout, arguments)?,
            Self::layout_buffer(&values.layout, arguments)?,
            Self::layout_buffer(&indices.layout, arguments)?,
            arguments.write(&width.to_ne_bytes())?,
            arguments.write(&chunks.to_ne_bytes())?,
            arguments.write(&k.to_ne_bytes())?,
            arguments.write(&u32::from(normalize).to_ne_bytes())?,
        ];
        set_pipeline(encoder, &partials_pipeline);
        bindings.bind(table, 0, &input.buffer);
        bindings.bind(table, 1, &scratch.buffer);
        bindings.bind(table, 2, &temporaries[1]);
        bindings.bind(table, 3, &temporaries[4]);
        bindings.bind(table, 4, &temporaries[5]);
        bindings.bind(table, 5, &temporaries[6]);
        set_argument_table(encoder, table);
        let partial_groups = usize::try_from(u64::from(rows) * u64::from(chunks))
            .map_err(|_| BackendError::ExecutionFailed)?;
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: partial_groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &finalize_pipeline);
        for (index, tensor) in [&input, &scratch, &values, &indices]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        for (index, temporary) in temporaries[1..].iter().enumerate() {
            bindings.bind(table, index + 4, temporary);
        }
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        Ok(temporaries)
    }

    fn write_sample_arguments<A: ArgumentSink>(
        dispatch: &Dispatch,
        params: SampleKernelParams,
        arguments: &mut A,
    ) -> Result<Vec<A::Written>, BackendError> {
        let [logits, sampling] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let width = *logits
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let chunks = width.div_ceil(params.chunk_width);
        let capacity = 8_usize
            .checked_add(
                usize::try_from(params.rejection_rounds).map_err(|_| BackendError::InvalidInput)?,
            )
            .ok_or(BackendError::InvalidInput)?;
        let mut written = Vec::with_capacity(capacity);
        written.push(arguments.write_argument(&[0_u8; 8])?);
        for layout in [
            logits.layout(),
            sampling.layout(),
            dispatch.output().layout(),
        ] {
            written.push(arguments.write_argument(&encode_layout(layout)?)?);
        }
        for value in [width, params.position, chunks] {
            written.push(arguments.write_argument(&value.to_ne_bytes())?);
        }
        written.push(
            arguments
                .write_argument(&sample_rejection_rounds(params.rejection_rounds).to_ne_bytes())?,
        );
        for round in 0..params.rejection_rounds {
            written.push(arguments.write_argument(&round.to_ne_bytes())?);
        }
        Ok(written)
    }

    #[allow(clippy::too_many_lines)]
    fn encode_sample(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        params: SampleKernelParams,
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
        let chunks = width.div_ceil(params.chunk_width);
        let state = self.scratch_tensor(DType::U32, &[rows, 7])?;
        let rejection_dispatch = self.scratch_tensor(DType::U32, &[3])?;
        let rejection_partials = self.scratch_tensor(DType::U32, &[rows, chunks, 4])?;
        let topk_words = params
            .fast_top_k
            .checked_mul(2)
            .ok_or(BackendError::InvalidInput)?;
        let topk_partials = self.scratch_tensor(DType::U32, &[rows, chunks, topk_words])?;
        let partials = self.scratch_tensor(DType::U32, &[rows, chunks, 2])?;
        let constants = params.function_constants(dtype_code(logits.layout().dtype()));
        let (
            prepare_pipeline,
            rejection_proposal_partials_pipeline,
            rejection_proposal_finalize_pipeline,
            rejection_threshold_partials_pipeline,
            rejection_threshold_finalize_pipeline,
            topk_partials_pipeline,
            topk_finalize_pipeline,
            exact_fallback_pipeline,
            partials_pipeline,
            reduce_finalize_pipeline,
        ) = {
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            (
                pipelines.get("sample_prepare", &constants)?,
                pipelines.get("sample_rejection_proposal_partials", &constants)?,
                pipelines.get("sample_rejection_proposal_finalize", &constants)?,
                pipelines.get("sample_rejection_threshold_partials", &constants)?,
                pipelines.get("sample_rejection_threshold_finalize", &constants)?,
                pipelines.get("sample_topk_partials", &constants)?,
                pipelines.get("sample_topk_finalize", &constants)?,
                pipelines.get("sample_exact_fallback", &constants)?,
                pipelines.get("sample_partials", &constants)?,
                pipelines.get("sample_reduce_finalize", &constants)?,
            )
        };
        let mut temporaries = Self::write_sample_arguments(dispatch, params, arguments)?;
        let error_flag = temporaries[SAMPLE_ERROR].clone();
        let (logits_buffer, sampling_buffer, output_buffer) = {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            (
                BufferBinding::whole(buffers.get(logits)?.raw.clone()),
                BufferBinding::whole(buffers.get(sampling)?.raw.clone()),
                BufferBinding::whole(buffers.get(output)?.raw.clone()),
            )
        };
        let row_groups = MTLSize {
            width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        };
        let chunk_groups = MTLSize {
            width: usize::try_from(rows)
                .map_err(|_| BackendError::ExecutionFailed)?
                .checked_mul(usize::try_from(chunks).map_err(|_| BackendError::ExecutionFailed)?)
                .ok_or(BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: simd_thread_count(&rejection_proposal_partials_pipeline, 256)?,
            height: 1,
            depth: 1,
        };
        set_pipeline(encoder, &prepare_pipeline);
        bindings.bind(table, 0, &sampling_buffer);
        bindings.bind(table, 1, &state.buffer);
        bindings.bind(table, 2, &temporaries[SAMPLE_PARAMS_LAYOUT]);
        bindings.bind(table, 3, &temporaries[SAMPLE_WIDTH]);
        bindings.bind(table, 4, &temporaries[SAMPLE_ERROR]);
        bindings.bind(table, 5, &temporaries[SAMPLE_MAX_ROUNDS]);
        bindings.bind(table, 6, &rejection_dispatch.buffer);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, row_groups, threads);

        let common = SampleCommonBindings {
            logits: &logits_buffer,
            sampling: &sampling_buffer,
            state: &state.buffer,
            logits_layout: &temporaries[SAMPLE_LOGITS_LAYOUT],
            sampling_layout: &temporaries[SAMPLE_PARAMS_LAYOUT],
            width: &temporaries[SAMPLE_WIDTH],
        };
        for round in 0..sample_rejection_rounds(params.rejection_rounds) {
            let round_buffer = &temporaries[SAMPLE_ROUND_ARGUMENTS
                + usize::try_from(round).map_err(|_| BackendError::ExecutionFailed)?];
            encode_dispatch_barrier(encoder);
            set_pipeline(encoder, &rejection_proposal_partials_pipeline);
            common.bind(table, bindings, &rejection_partials.buffer);
            bindings.bind(table, 7, &temporaries[SAMPLE_POSITION]);
            bindings.bind(table, 8, round_buffer);
            bindings.bind(table, 9, &temporaries[SAMPLE_CHUNKS]);
            set_argument_table(encoder, table);
            if round == 0 {
                dispatch_threadgroups(encoder, chunk_groups, threads);
            } else {
                dispatch_threadgroups_indirect(encoder, &rejection_dispatch.buffer, threads);
            }

            encode_dispatch_barrier(encoder);
            set_pipeline(encoder, &rejection_proposal_finalize_pipeline);
            bindings.bind(table, 0, &rejection_partials.buffer);
            bindings.bind(table, 1, &state.buffer);
            bindings.bind(table, 2, &temporaries[SAMPLE_CHUNKS]);
            set_argument_table(encoder, table);
            dispatch_threadgroups(encoder, row_groups, threads);

            encode_dispatch_barrier(encoder);
            set_pipeline(encoder, &rejection_threshold_partials_pipeline);
            common.bind(table, bindings, &rejection_partials.buffer);
            bindings.bind(table, 7, &temporaries[SAMPLE_CHUNKS]);
            set_argument_table(encoder, table);
            if round == 0 {
                dispatch_threadgroups(encoder, chunk_groups, threads);
            } else {
                dispatch_threadgroups_indirect(encoder, &rejection_dispatch.buffer, threads);
            }

            encode_dispatch_barrier(encoder);
            set_pipeline(encoder, &rejection_threshold_finalize_pipeline);
            bindings.bind(table, 0, &logits_buffer);
            bindings.bind(table, 1, &sampling_buffer);
            bindings.bind(table, 2, &rejection_partials.buffer);
            bindings.bind(table, 3, &state.buffer);
            bindings.bind(table, 4, &output_buffer);
            bindings.bind(table, 5, &temporaries[SAMPLE_LOGITS_LAYOUT]);
            bindings.bind(table, 6, &temporaries[SAMPLE_PARAMS_LAYOUT]);
            bindings.bind(table, 7, &temporaries[SAMPLE_OUTPUT_LAYOUT]);
            bindings.bind(table, 8, &temporaries[SAMPLE_WIDTH]);
            bindings.bind(table, 9, round_buffer);
            bindings.bind(table, 10, &temporaries[SAMPLE_MAX_ROUNDS]);
            bindings.bind(table, 11, &temporaries[SAMPLE_CHUNKS]);
            bindings.bind(table, 12, &rejection_dispatch.buffer);
            set_argument_table(encoder, table);
            dispatch_threadgroups(encoder, row_groups, threads);
        }
        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &topk_partials_pipeline);
        bindings.bind(table, 0, &logits_buffer);
        bindings.bind(table, 1, &topk_partials.buffer);
        bindings.bind(table, 2, &state.buffer);
        bindings.bind(table, 3, &temporaries[SAMPLE_LOGITS_LAYOUT]);
        bindings.bind(table, 4, &temporaries[SAMPLE_WIDTH]);
        bindings.bind(table, 5, &temporaries[SAMPLE_CHUNKS]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, chunk_groups, threads);

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &topk_finalize_pipeline);
        bindings.bind(table, 0, &sampling_buffer);
        bindings.bind(table, 1, &topk_partials.buffer);
        bindings.bind(table, 2, &state.buffer);
        bindings.bind(table, 3, &output_buffer);
        bindings.bind(table, 4, &temporaries[SAMPLE_PARAMS_LAYOUT]);
        bindings.bind(table, 5, &temporaries[SAMPLE_OUTPUT_LAYOUT]);
        bindings.bind(table, 6, &temporaries[SAMPLE_WIDTH]);
        bindings.bind(table, 7, &temporaries[SAMPLE_POSITION]);
        bindings.bind(table, 8, &temporaries[SAMPLE_MAX_ROUNDS]);
        bindings.bind(table, 9, &temporaries[SAMPLE_ERROR]);
        bindings.bind(table, 10, &temporaries[SAMPLE_CHUNKS]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, row_groups, threads);

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &exact_fallback_pipeline);
        bindings.bind(table, 0, &logits_buffer);
        bindings.bind(table, 1, &sampling_buffer);
        bindings.bind(table, 2, &state.buffer);
        bindings.bind(table, 3, &output_buffer);
        bindings.bind(table, 4, &temporaries[SAMPLE_LOGITS_LAYOUT]);
        bindings.bind(table, 5, &temporaries[SAMPLE_PARAMS_LAYOUT]);
        bindings.bind(table, 6, &temporaries[SAMPLE_OUTPUT_LAYOUT]);
        bindings.bind(table, 7, &temporaries[SAMPLE_WIDTH]);
        bindings.bind(table, 8, &temporaries[SAMPLE_POSITION]);
        bindings.bind(table, 9, &temporaries[SAMPLE_MAX_ROUNDS]);
        bindings.bind(table, 10, &temporaries[SAMPLE_ERROR]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, row_groups, threads);

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &partials_pipeline);
        common.bind(table, bindings, &partials.buffer);
        bindings.bind(table, 7, &temporaries[SAMPLE_POSITION]);
        bindings.bind(table, 8, &temporaries[SAMPLE_CHUNKS]);
        bindings.bind(table, 9, &temporaries[SAMPLE_MAX_ROUNDS]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, chunk_groups, threads);

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &reduce_finalize_pipeline);
        bindings.bind(table, 0, &sampling_buffer);
        bindings.bind(table, 1, &partials.buffer);
        bindings.bind(table, 2, &state.buffer);
        bindings.bind(table, 3, &output_buffer);
        bindings.bind(table, 4, &temporaries[SAMPLE_PARAMS_LAYOUT]);
        bindings.bind(table, 5, &temporaries[SAMPLE_OUTPUT_LAYOUT]);
        bindings.bind(table, 6, &temporaries[SAMPLE_WIDTH]);
        bindings.bind(table, 7, &temporaries[SAMPLE_CHUNKS]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, row_groups, threads);

        temporaries.extend([
            logits_buffer,
            sampling_buffer,
            output_buffer,
            state.buffer,
            rejection_dispatch.buffer,
            rejection_partials.buffer,
            topk_partials.buffer,
            partials.buffer,
        ]);
        Ok((temporaries, error_flag))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn encode_sample_parallel(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        params: SampleKernelParams,
        allow: Option<&Tensor>,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<EncodedEmbed, BackendError> {
        let [logits, sampling, ..] = dispatch.inputs() else {
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
        let chunks = width.div_ceil(params.chunk_width);
        let partial_words = params
            .fast_top_k
            .checked_add(3)
            .and_then(|values| values.checked_mul(2))
            .ok_or(BackendError::InvalidInput)?;
        let partials = self.scratch_tensor(DType::U32, &[rows, chunks, partial_words])?;
        let (partials_pipeline, finalize_pipeline) = {
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            let mut constants = params
                .function_constants(dtype_code(logits.layout().dtype()))
                .to_vec();
            constants.push((SAMPLE_HAS_MASK_CONSTANT, u32::from(allow.is_some())));
            (
                pipelines.get("sample_masked_partials", &constants)?,
                pipelines.get("sample_masked_finalize", &constants)?,
            )
        };
        let mut temporaries = vec![arguments.write(&[0_u8; 8])?];
        for layout in [
            logits.layout(),
            sampling.layout(),
            allow.map_or(&partials.layout, Tensor::layout),
            output.layout(),
        ] {
            temporaries.push(Self::layout_buffer(layout, arguments)?);
        }
        temporaries.push(arguments.write(&width.to_ne_bytes())?);
        temporaries.push(arguments.write(&params.position.to_ne_bytes())?);
        temporaries.push(arguments.write(&chunks.to_ne_bytes())?);
        let error_flag = temporaries[0].clone();
        let (logits_buffer, sampling_buffer, allow_buffer, output_buffer) = {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            let allow_buffer = allow.map_or_else(
                || Ok(partials.buffer.clone()),
                |mask| {
                    buffers
                        .get(mask)
                        .map(|buffer| BufferBinding::whole(buffer.raw.clone()))
                },
            )?;
            (
                BufferBinding::whole(buffers.get(logits)?.raw.clone()),
                BufferBinding::whole(buffers.get(sampling)?.raw.clone()),
                allow_buffer,
                BufferBinding::whole(buffers.get(output)?.raw.clone()),
            )
        };
        let row_groups = MTLSize {
            width: usize::try_from(rows).map_err(|_| BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        };
        let chunk_groups = MTLSize {
            width: usize::try_from(rows)
                .map_err(|_| BackendError::ExecutionFailed)?
                .checked_mul(usize::try_from(chunks).map_err(|_| BackendError::ExecutionFailed)?)
                .ok_or(BackendError::ExecutionFailed)?,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: simd_thread_count(&partials_pipeline, 256)?,
            height: 1,
            depth: 1,
        };
        set_pipeline(encoder, &partials_pipeline);
        for (index, buffer) in [
            &logits_buffer,
            &sampling_buffer,
            &allow_buffer,
            &partials.buffer,
        ]
        .into_iter()
        .enumerate()
        {
            bindings.bind(table, index, buffer);
        }
        for index in 0..3 {
            bindings.bind(table, index + 4, &temporaries[index + 1]);
        }
        bindings.bind(table, 7, &temporaries[5]);
        bindings.bind(table, 8, &temporaries[6]);
        bindings.bind(table, 9, &temporaries[7]);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, chunk_groups, threads);

        encode_dispatch_barrier(encoder);
        set_pipeline(encoder, &finalize_pipeline);
        for (index, buffer) in [
            &logits_buffer,
            &sampling_buffer,
            &allow_buffer,
            &partials.buffer,
            &output_buffer,
        ]
        .into_iter()
        .enumerate()
        {
            bindings.bind(table, index, buffer);
        }
        for index in 0..4 {
            bindings.bind(table, index + 5, &temporaries[index + 1]);
        }
        bindings.bind(table, 9, &temporaries[5]);
        bindings.bind(table, 10, &temporaries[6]);
        bindings.bind(table, 11, &temporaries[7]);
        bindings.bind(table, 12, &error_flag);
        set_argument_table(encoder, table);
        dispatch_threadgroups(encoder, row_groups, threads);
        temporaries.extend([
            logits_buffer,
            sampling_buffer,
            allow_buffer,
            output_buffer,
            partials.buffer,
            error_flag.clone(),
        ]);
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
        let pipeline =
            self.row_pipeline("rms_norm_single", "rms_norm_looped", width, &constants)?;
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
            bindings.bind_raw(table, index, buffers.get(tensor)?)?;
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        drop(buffers);
        set_argument_table(encoder, table);
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, output.layout(), width)?;
        dispatch_threadgroups(encoder, threadgroups, threads);
        Ok(temporaries)
    }

    fn row_pipeline(
        &self,
        single: &str,
        looped: &str,
        width: u32,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = pipelines.get(single, constants)?;
        let rounded_width = usize::try_from(width)
            .map_err(|_| BackendError::ExecutionFailed)?
            .next_multiple_of(32);
        if rounded_width <= pipeline.maxTotalThreadsPerThreadgroup() {
            Ok(pipeline)
        } else {
            pipelines.get(looped, constants)
        }
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
                .bind_raw(table, index, buffers.get(tensor)?)?;
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

    #[allow(clippy::too_many_lines)]
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
            Op::PagedCopy { .. } => {
                arguments.write(7 * size_of::<u32>())?;
                arguments.write(112)?;
                arguments.write(2 * size_of::<u32>())?;
                if dispatch.inputs().len() == 3 {
                    arguments.write(size_of::<u32>())?;
                }
            }
            Op::RmsNorm { .. } => {
                for len in [112, 112, 112, 8] {
                    arguments.write(len)?;
                }
            }
            Op::Argmax { .. } => {
                Self::size_softmax_arguments(arguments)?;
                arguments.write(size_of::<u32>())?;
            }
            Op::TopK { .. } => {
                Self::size_top_k_arguments(dispatch, arguments)?;
            }
            Op::Sample {
                position,
                chunk_width,
                fast_top_k,
                fast_max_chunks,
                rejection_rounds,
            } => {
                if selected_variant(dispatch) == Some(MetalVariant::SampleParallel) {
                    Self::size_parallel_sample_arguments(arguments)?;
                } else {
                    Self::write_sample_arguments(
                        dispatch,
                        SampleKernelParams {
                            position,
                            chunk_width,
                            fast_top_k,
                            fast_max_chunks,
                            rejection_rounds,
                        },
                        arguments,
                    )?;
                }
            }
            Op::SampleMasked { .. } => {
                Self::size_parallel_sample_arguments(arguments)?;
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
            Op::QuantEmbed { .. } => {
                for len in [8, 112, 112, 112, 112, 112, 8] {
                    arguments.write(len)?;
                }
            }
            Op::Matmul => Self::size_matmul_arguments(dispatch, arguments)?,
            Op::GatherMatmul => {
                arguments.write(136)?;
                arguments.write(8)?;
            }
            Op::QuantMatmul { .. } => {
                arguments.write(144)?;
                Self::size_epilogue_arguments(dispatch, arguments)?;
            }
            Op::GatherQuantMatmul { .. } => {
                Self::size_gather_quant_arguments(dispatch, arguments)?;
            }
            Op::Sdpa { .. } => self.size_sdpa_arguments(dispatch, arguments)?,
            Op::PagedSdpa { .. } => {
                arguments.write(128)?;
                arguments.write(4 * size_of::<u32>())?;
                if dispatch.inputs().len() == 5 {
                    arguments.write(size_of::<u32>())?;
                }
                if selected_variant(dispatch) == Some(MetalVariant::PagedSdpaDecomposed) {
                    for _ in 0..2 {
                        arguments.write(7 * size_of::<u32>())?;
                        arguments.write(112)?;
                    }
                    let kv_heads = dispatch
                        .inputs()
                        .get(1)
                        .and_then(|key| key.layout().shape().get(2))
                        .copied()
                        .ok_or(BackendError::InvalidInput)?;
                    for _ in 0..kv_heads {
                        arguments.write(96)?;
                    }
                    arguments.write(24)?;
                    Self::size_softmax_arguments(arguments)?;
                    for _ in 0..kv_heads {
                        arguments.write(96)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn size_top_k_arguments(
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        let direct = matches!(
            selected_variant(dispatch),
            Some(MetalVariant::TopKPortable | MetalVariant::TopKSingleK8)
        );
        let lengths: &[usize] = if direct {
            &[112, 112, 112, 4, 4, 4]
        } else {
            &[112, 112, 112, 4, 4, 4, 4]
        };
        for &len in lengths {
            arguments.write(len)?;
        }
        Ok(())
    }

    fn size_parallel_sample_arguments(arguments: &mut ArgumentSizer) -> Result<(), BackendError> {
        for len in [8, 112, 112, 112, 112, 4, 4, 4] {
            arguments.write(len)?;
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
        let [left, right] = dispatch.primitive_inputs() else {
            return Err(BackendError::InvalidInput);
        };
        Self::size_prepared_matmul_input(left.layout(), arguments)?;
        Self::size_prepared_matmul_input(right.layout(), arguments)?;
        Self::size_epilogue_arguments(dispatch, arguments)?;
        arguments.write(96)?;
        if classify(dispatch.output().layout())
            .kernel_strides()
            .is_none_or(|(column_major, _, _)| column_major != 0)
        {
            Self::size_copy_arguments(arguments)?;
        }
        Ok(())
    }

    fn size_epilogue_arguments(
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        let extra_count = dispatch
            .inputs()
            .len()
            .saturating_sub(dispatch.primitive_inputs().len());
        if extra_count > 0 {
            arguments.write(extra_count.saturating_mul(112))?;
        }
        Ok(())
    }

    fn size_gather_quant_arguments(
        dispatch: &Dispatch,
        arguments: &mut ArgumentSizer,
    ) -> Result<(), BackendError> {
        if dispatch.primitive_inputs().len() == 6 {
            arguments.write(232)?;
        } else {
            arguments.write(212)?;
        }
        if dispatch.primitive_inputs().len() == 8 {
            arguments.write(212)?;
            if grouped_gather_dispatch(dispatch) {
                arguments.write(212)?;
            }
        }
        arguments.write(8)?;
        Self::size_epilogue_arguments(dispatch, arguments)
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
        if self.prepare_sdpa_kernel(dispatch)? != SdpaKernel::Decomposed {
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
        let heap = self.allocate_device_object(|| {
            self.device
                .newCounterHeapWithDescriptor_error(&descriptor)
                .map_err(|_| BackendError::ExecutionFailed)
        })?;
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
            internal_profiles: _,
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
                    .get(&flag.binding.raw.gpuAddress())
                    .copied()
                    .map(|index| (index, flag.binding.offset, flag.dispatch, flag.operation))
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
            dispatch_profiles: Vec::new(),
            internal_profiles: Vec::new(),
        })
    }
}

fn shape3(layout: &Layout) -> Result<[u32; 3], BackendError> {
    layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)
}

fn sdpa_matmul_variant(rows: u32, right_transposed: bool) -> MetalVariant {
    if rows == 1 {
        if right_transposed {
            MetalVariant::MatmulGemvTransposed
        } else {
            MetalVariant::MatmulGemv
        }
    } else {
        MetalVariant::MatmulPortable
    }
}

fn supported_dispatch(dispatch: &Dispatch) -> bool {
    if selected_variant(dispatch) == Some(MetalVariant::GatherQuantMatmulWeightedBroad)
        && dispatch.primitive_inputs().len() != 6
    {
        return false;
    }
    if dispatch.prepared_epilogue().is_some()
        && !matches!(
            selected_variant(dispatch),
            Some(
                MetalVariant::MatmulGemv
                    | MetalVariant::MatmulGemvTransposed
                    | MetalVariant::MatmulPortable
                    | MetalVariant::MatmulSteel64x64x16_2x2
                    | MetalVariant::MatmulSteel64x64x16_1x2
                    | MetalVariant::MatmulSteel64x32x32_2x2
                    | MetalVariant::MatmulSteel32x64x16_1x2
                    | MetalVariant::QuantMatmulPortable
                    | MetalVariant::QuantMatmulGemv
                    | MetalVariant::QuantMatmulQ8FastGemv
                    | MetalVariant::QuantMatmulSmallM
                    | MetalVariant::QuantMatmulTiled
                    | MetalVariant::GatherQuantMatmulPortable
                    | MetalVariant::GatherQuantMatmulRouteGemv
                    | MetalVariant::GatherQuantMatmulGroupedSmallM
                    | MetalVariant::GatherQuantMatmulGroupedPairSmallM
                    | MetalVariant::GatherQuantMatmulGrouped
                    | MetalVariant::GatherQuantMatmulWeightedBroad
            )
        )
    {
        return false;
    }
    match dispatch.op() {
        Op::Program(_) => dispatch.bound_program().is_some(),
        Op::Copy
        | Op::PagedCopy { .. }
        | Op::RmsNorm { .. }
        | Op::Argmax { .. }
        | Op::TopK { .. }
        | Op::Rope { .. }
        | Op::Embed
        | Op::QuantEmbed { .. }
        | Op::Matmul
        | Op::GatherMatmul
        | Op::QuantMatmul { .. }
        | Op::GatherQuantMatmul { .. }
        | Op::Sdpa { .. }
        | Op::PagedSdpa { .. }
        | Op::Sample { .. }
        | Op::SampleMasked { .. } => true,
    }
}

fn vectorized_copy(input: &Layout, output: &Layout) -> bool {
    input.is_contiguous()
        && output.is_contiguous()
        && input.dtype() == DType::F32
        && output.dtype() == DType::F32
        && input.offset().is_multiple_of(4)
        && output.offset().is_multiple_of(4)
        && output.element_count().is_multiple_of(4)
}

fn vectorized_argmax(input: &Layout, width: u32, chunk_width: u32) -> bool {
    input.dtype() == DType::F32
        && input.is_contiguous()
        && input.offset().is_multiple_of(4)
        && width.is_multiple_of(4)
        && chunk_width.is_multiple_of(4)
}

fn dispatch_profile(dispatch: &Dispatch, barrier: bool) -> Result<DispatchProfile, BackendError> {
    Ok(DispatchProfile {
        op: dispatch.op(),
        kernel: dispatch_kernel(dispatch)?,
        inputs: dispatch.inputs().iter().map(ProfileTensor::from).collect(),
        outputs: dispatch.outputs().iter().map(ProfileTensor::from).collect(),
        barrier,
        gpu_start: Duration::ZERO,
        gpu_end: Duration::ZERO,
        gap_before: Duration::ZERO,
        gpu_time: Duration::ZERO,
        internal: Vec::new(),
    })
}

fn internal_dispatch_names(dispatch: &Dispatch) -> &'static [&'static str] {
    if !matches!(dispatch.op(), Op::GatherQuantMatmul { .. }) || !grouped_gather_dispatch(dispatch)
    {
        return &[];
    }
    match dispatch.primitive_inputs().len() {
        8 => &[
            "count_expert_routes",
            "arrange_expert_routes",
            "grouped_quantized_gemm_gate",
            "grouped_quantized_gemm_up",
            "gathered_pair_from_projections",
        ],
        6 => &[],
        _ => &[
            "count_expert_routes",
            "arrange_expert_routes",
            "grouped_quantized_gemm",
        ],
    }
}

fn internal_dispatch_profiles(
    dispatch: &Dispatch,
    dispatch_index: usize,
    first_internal: usize,
) -> Vec<InternalDispatchProfile> {
    let names = internal_dispatch_names(dispatch);
    let outer_start = 2 + dispatch_index * 2;
    let outer_end = outer_start + 1;
    names
        .iter()
        .enumerate()
        .map(|(index, &kernel)| InternalDispatchProfile {
            dispatch: dispatch_index,
            kernel,
            start: if index == 0 {
                outer_start
            } else {
                first_internal + index - 1
            },
            end: if index + 1 == names.len() {
                outer_end
            } else {
                first_internal + index
            },
        })
        .collect()
}

pub(super) fn dispatch_kernel(dispatch: &Dispatch) -> Result<&'static str, BackendError> {
    Ok(match dispatch.op() {
        Op::Program(_) => match dispatch
            .bound_program()
            .ok_or(BackendError::InvalidInput)?
            .program()
            .program()
            .kind
        {
            ProgramKind::Map => "forja_map",
            ProgramKind::Row => "forja_row",
        },
        Op::Copy => {
            let [input] = dispatch.inputs() else {
                return Err(BackendError::InvalidInput);
            };
            if vectorized_copy(input.layout(), dispatch.output().layout()) {
                "copy_contiguous_f32x4"
            } else if input.layout().is_contiguous() && dispatch.output().layout().is_contiguous() {
                "copy_contiguous"
            } else {
                "copy_strided"
            }
        }
        Op::PagedCopy { .. } => "paged_copy",
        Op::RmsNorm { .. } => "rms_norm",
        Op::Argmax { chunk_width } => {
            let [input] = dispatch.inputs() else {
                return Err(BackendError::InvalidInput);
            };
            let width = *input
                .layout()
                .shape()
                .last()
                .ok_or(BackendError::InvalidInput)?;
            if vectorized_argmax(input.layout(), width, chunk_width) {
                "argmax_partials_f32x4+finalize"
            } else {
                "argmax_partials+finalize"
            }
        }
        Op::TopK { .. } => match selected_variant(dispatch) {
            Some(MetalVariant::TopKSingleK8) => "topk_single",
            Some(MetalVariant::TopKPortable) => "topk_exact_fallback",
            Some(MetalVariant::TopKPartials) => "topk_partials+finalize",
            _ => return Err(BackendError::InvalidInput),
        },
        Op::Sample { .. } => match selected_variant(dispatch) {
            Some(MetalVariant::SamplePortable) => "sample_rejection",
            Some(MetalVariant::SampleParallel) => "sample_parallel",
            _ => return Err(BackendError::InvalidInput),
        },
        Op::SampleMasked { .. } => "sample_masked",
        Op::Rope { .. } => "rope",
        Op::Embed => "embed",
        Op::QuantEmbed { .. } => "quant-embed",
        Op::Matmul => matmul_kernel(dispatch)?,
        Op::GatherMatmul => "gather_gemv",
        Op::QuantMatmul { .. } => quant_matmul_kernel(dispatch)?,
        Op::GatherQuantMatmul { .. } => {
            if dispatch.primitive_inputs().len() == 6 {
                let routes = dispatch
                    .primitive_inputs()
                    .first()
                    .and_then(|input| input.layout().shape().get(1))
                    .ok_or(BackendError::InvalidInput)?;
                weighted_gather_schedule(selected_variant(dispatch), *routes)?.implementation
            } else if dispatch.primitive_inputs().len() == 8 && grouped_gather_dispatch(dispatch) {
                if grouped_block_rows(dispatch) == 8 {
                    "gathered_pair_from_projections_small_m"
                } else {
                    "gathered_pair_from_projections"
                }
            } else if dispatch.primitive_inputs().len() == 8 {
                "gather_quantized_pair"
            } else if grouped_gather_dispatch(dispatch) {
                if grouped_block_rows(dispatch) == 8 {
                    "grouped_quantized_gemm_small_m"
                } else {
                    "grouped_quantized_gemm"
                }
            } else {
                "gather_quantized_gemv"
            }
        }
        Op::Sdpa { .. } => sdpa_kernel(dispatch)?,
        Op::PagedSdpa { .. } => {
            let width = dispatch
                .inputs()
                .first()
                .and_then(|query| query.layout().shape().get(2))
                .copied()
                .ok_or(BackendError::InvalidInput)?;
            paged_sdpa_kernel(dispatch, width)?
        }
    })
}

fn quant_matmul_kernel(dispatch: &Dispatch) -> Result<&'static str, BackendError> {
    let [input, _, _, _] = dispatch.primitive_inputs() else {
        return Err(BackendError::InvalidInput);
    };
    match selected_variant(dispatch) {
        Some(MetalVariant::QuantMatmulGemv) => Ok("quantized_gemv"),
        Some(MetalVariant::QuantMatmulQ8FastGemv) => Ok("quantized_gemv_q8_fast"),
        Some(MetalVariant::QuantMatmulPortable | MetalVariant::QuantMatmulSmallM) => {
            Ok("quantized_gemm_small_m")
        }
        Some(MetalVariant::QuantMatmulTiled) => match input.layout().dtype() {
            DType::F32 => Ok("quantized_gemm_tiled_f32"),
            DType::F16 => Ok("quantized_gemm_tiled_f16"),
            DType::BF16 => Ok("quantized_gemm_tiled_bf16"),
            DType::I32 | DType::U32 => Err(BackendError::InvalidInput),
        },
        _ => Err(BackendError::InvalidInput),
    }
}

fn matmul_kernel(dispatch: &Dispatch) -> Result<&'static str, BackendError> {
    match selected_variant(dispatch) {
        Some(MetalVariant::MatmulPortable | MetalVariant::MatmulSteel64x32x32_2x2) => {
            Ok("steel_gemm_64_32_32_2_2")
        }
        Some(MetalVariant::MatmulGemv) => Ok("gemv"),
        Some(MetalVariant::MatmulGemvTransposed) => Ok("gemv_transposed"),
        Some(MetalVariant::MatmulSteel64x64x16_2x2) => Ok("steel_gemm_64_64_16_2_2"),
        Some(MetalVariant::MatmulSteel64x64x16_1x2) => Ok("steel_gemm_64_64_16_1_2"),
        Some(MetalVariant::MatmulSteel32x64x16_1x2) => Ok("steel_gemm_32_64_16_1_2"),
        _ => Err(BackendError::InvalidInput),
    }
}

fn sdpa_kernel(dispatch: &Dispatch) -> Result<&'static str, BackendError> {
    let [query, _, _] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let width = shape3(query.layout())?[2];
    let fast_f32 = vector_f32_supported(dispatch);
    Ok(match select_sdpa(dispatch)? {
        SdpaKernel::VectorGqa => vector_kernel("mlx_sdpa_vector_2pass_gqa", width)?,
        SdpaKernel::Vector if vector_two_pass_blocks(dispatch).is_some() => vector_kernel(
            if fast_f32 {
                "mlx_sdpa_vector_2pass_f32"
            } else {
                "mlx_sdpa_vector_2pass"
            },
            width,
        )?,
        SdpaKernel::Vector => vector_kernel(
            if fast_f32 {
                "mlx_sdpa_vector_f32"
            } else {
                "mlx_sdpa_vector"
            },
            width,
        )?,
        kernel @ (SdpaKernel::Steel | SdpaKernel::SteelHalf) => {
            steel_attention_kernel(kernel, width)?
        }
        SdpaKernel::Decomposed => "sdpa_decomposed",
    })
}

fn paged_sdpa_kernel(dispatch: &Dispatch, width: u32) -> Result<&'static str, BackendError> {
    match (selected_variant(dispatch), width) {
        (Some(MetalVariant::PagedSdpaDecomposed), 64 | 128) => Ok("sdpa_decomposed"),
        (
            Some(
                MetalVariant::PagedSdpaVectorGqaTwoPass32
                | MetalVariant::PagedSdpaVectorGqaTwoPass64
                | MetalVariant::PagedSdpaVectorGqaTwoPass128
                | MetalVariant::PagedSdpaVectorGqaTwoPass256,
            ),
            128,
        ) => {
            let inputs = dispatch.inputs();
            let query_heads = inputs
                .first()
                .and_then(|query| query.layout().shape().first())
                .copied()
                .ok_or(BackendError::InvalidInput)?;
            let kv_heads = inputs
                .get(1)
                .and_then(|key| key.layout().shape().get(2))
                .copied()
                .ok_or(BackendError::InvalidInput)?;
            paged_gqa_first_pass(query_heads, kv_heads).map(|(kernel, _, _)| kernel)
        }
        (Some(MetalVariant::PagedSdpaSteel), 64) => Ok("paged_steel_attention_64"),
        (Some(MetalVariant::PagedSdpaSteel), 128) => Ok("paged_steel_attention_128"),
        (Some(MetalVariant::PagedSdpaSteelHalf), 64) => Ok("paged_steel_attention_half_64_64x32"),
        (Some(MetalVariant::PagedSdpaSteelHalf), 128) => Ok("paged_steel_attention_half_128_96x16"),
        (_, 64) => Ok("paged_sdpa_64"),
        (_, 128) => Ok("paged_sdpa_128"),
        _ => Err(BackendError::UnsupportedOperation),
    }
}

fn paged_gqa_first_pass(
    query_heads: u32,
    kv_heads: u32,
) -> Result<(&'static str, usize, u32), BackendError> {
    match query_heads
        .checked_div(kv_heads)
        .filter(|_| query_heads.is_multiple_of(kv_heads))
    {
        Some(1) if kv_heads.is_multiple_of(8) => Ok(("paged_sdpa_2pass_1_gqa1x8_128", 256, 8)),
        Some(1) => Ok(("paged_sdpa_2pass_1_gqa1_128", 32, 1)),
        Some(2) if kv_heads.is_multiple_of(8) => Ok(("paged_sdpa_2pass_1_gqa2x8_128", 512, 8)),
        Some(2) => Ok(("paged_sdpa_2pass_1_gqa2_128", 64, 1)),
        Some(8) => Ok(("paged_sdpa_2pass_1_gqa8_128", 256, 1)),
        _ => Err(BackendError::UnsupportedOperation),
    }
}

fn reusable_dispatch(dispatch: &Dispatch) -> bool {
    match dispatch.op() {
        Op::Argmax { .. }
        | Op::TopK { .. }
        | Op::Sample { .. }
        | Op::SampleMasked { .. }
        | Op::Embed
        | Op::QuantEmbed { .. }
        | Op::GatherMatmul
        | Op::GatherQuantMatmul { .. }
        | Op::Sdpa { .. }
        | Op::PagedCopy { .. }
        | Op::PagedSdpa { .. } => false,
        Op::Matmul => {
            dispatch
                .primitive_inputs()
                .iter()
                .all(|tensor| classify(tensor.layout()).kernel_strides().is_some())
                && classify(dispatch.output().layout())
                    .kernel_strides()
                    .is_some_and(|(column_major, _, _)| column_major == 0)
        }
        _ => true,
    }
}

#[cfg(test)]
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

fn vector_f32_supported(dispatch: &Dispatch) -> bool {
    dispatch
        .inputs()
        .iter()
        .chain(std::iter::once(dispatch.output()))
        .all(|tensor| {
            tensor.layout().dtype() == DType::F32
                && tensor.layout().strides().last().copied() == Some(1)
        })
}

fn vector_kernel(prefix: &str, width: u32) -> Result<&'static str, BackendError> {
    match (prefix, width) {
        ("mlx_sdpa_vector", 64) => Ok("mlx_sdpa_vector_64"),
        ("mlx_sdpa_vector", 128) => Ok("mlx_sdpa_vector_128"),
        ("mlx_sdpa_vector_2pass_1", 64) => Ok("mlx_sdpa_vector_2pass_1_64"),
        ("mlx_sdpa_vector_2pass_1", 128) => Ok("mlx_sdpa_vector_2pass_1_128"),
        ("mlx_sdpa_vector_2pass_1_gqa", 64) => Ok("mlx_sdpa_vector_2pass_1_gqa_64"),
        ("mlx_sdpa_vector_2pass_1_gqa", 128) => Ok("mlx_sdpa_vector_2pass_1_gqa_128"),
        ("mlx_sdpa_vector_2pass_2", 64) => Ok("mlx_sdpa_vector_2pass_2_64"),
        ("mlx_sdpa_vector_2pass_2", 128) => Ok("mlx_sdpa_vector_2pass_2_128"),
        ("mlx_sdpa_vector_2pass", 64) => Ok("mlx_sdpa_vector_2pass_64"),
        ("mlx_sdpa_vector_2pass", 128) => Ok("mlx_sdpa_vector_2pass_128"),
        ("mlx_sdpa_vector_2pass_gqa", 64) => Ok("mlx_sdpa_vector_2pass_gqa_64"),
        ("mlx_sdpa_vector_2pass_gqa", 128) => Ok("mlx_sdpa_vector_2pass_gqa_128"),
        ("mlx_sdpa_vector_f32", 64) => Ok("mlx_sdpa_vector_f32_64"),
        ("mlx_sdpa_vector_f32", 128) => Ok("mlx_sdpa_vector_f32_128"),
        ("mlx_sdpa_vector_2pass_f32", 64) => Ok("mlx_sdpa_vector_2pass_f32_64"),
        ("mlx_sdpa_vector_2pass_f32", 128) => Ok("mlx_sdpa_vector_2pass_f32_128"),
        ("mlx_sdpa_vector_2pass_1_f32", 64) => Ok("mlx_sdpa_vector_2pass_1_f32_64"),
        ("mlx_sdpa_vector_2pass_1_f32", 128) => Ok("mlx_sdpa_vector_2pass_1_f32_128"),
        ("mlx_sdpa_vector_2pass_2_f32", 64) => Ok("mlx_sdpa_vector_2pass_2_f32_64"),
        ("mlx_sdpa_vector_2pass_2_f32", 128) => Ok("mlx_sdpa_vector_2pass_2_f32_128"),
        _ => Err(BackendError::InvalidInput),
    }
}

fn steel_attention_kernel(kernel: SdpaKernel, width: u32) -> Result<&'static str, BackendError> {
    match (kernel, width) {
        (SdpaKernel::Steel, 64) => Ok("steel_attention_64"),
        (SdpaKernel::Steel, 128) => Ok("steel_attention_128"),
        (SdpaKernel::SteelHalf, 64) => Ok("steel_attention_half_64_64x32"),
        (SdpaKernel::SteelHalf, 128) => Ok("steel_attention_half_128_96x16"),
        _ => Err(BackendError::InvalidInput),
    }
}

fn steel_attention_tile(kernel: SdpaKernel, width: u32) -> Result<(u32, u32), BackendError> {
    match kernel {
        SdpaKernel::Steel => Ok((32, if width == 64 { 32 } else { 16 })),
        SdpaKernel::SteelHalf if width == 128 => Ok((96, 16)),
        SdpaKernel::SteelHalf => Ok((64, 32)),
        SdpaKernel::Vector | SdpaKernel::VectorGqa | SdpaKernel::Decomposed => {
            Err(BackendError::InvalidInput)
        }
    }
}

fn steel_attention_constants(
    dispatch: &Dispatch,
    kernel: SdpaKernel,
) -> Result<[(u32, u32); 7], BackendError> {
    let [query, key, value] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let [_, query_length, width] = shape3(query.layout())?;
    let [_, key_length, _] = shape3(key.layout())?;
    let (block_query, block_key) = steel_attention_tile(kernel, width)?;
    let Op::Sdpa { causal, .. } = dispatch.op() else {
        return Err(BackendError::InvalidInput);
    };
    Ok([
        (0, dtype_code(query.layout().dtype())),
        (1, dtype_code(key.layout().dtype())),
        (2, dtype_code(dispatch.output().layout().dtype())),
        (3, dtype_code(value.layout().dtype())),
        (210, u32::from(query_length % block_query == 0)),
        (211, u32::from(key_length % block_key == 0)),
        (212, u32::from(causal)),
    ])
}

fn select_sdpa(dispatch: &Dispatch) -> Result<SdpaKernel, BackendError> {
    #[cfg(test)]
    if let Some(kernel) = FORCED_SDPA_KERNEL.with(Cell::get) {
        return Ok(kernel);
    }
    match selected_variant(dispatch) {
        Some(MetalVariant::SdpaPortable | MetalVariant::SdpaDecomposed) => {
            Ok(SdpaKernel::Decomposed)
        }
        Some(
            MetalVariant::SdpaVectorSinglePass
            | MetalVariant::SdpaVectorTwoPass64
            | MetalVariant::SdpaVectorTwoPass128
            | MetalVariant::SdpaVectorTwoPass256
            | MetalVariant::SdpaVectorTwoPass512
            | MetalVariant::SdpaVectorTwoPass1024,
        ) => Ok(SdpaKernel::Vector),
        Some(
            MetalVariant::SdpaVectorGqaTwoPass32
            | MetalVariant::SdpaVectorGqaTwoPass64
            | MetalVariant::SdpaVectorGqaTwoPass128
            | MetalVariant::SdpaVectorGqaTwoPass256,
        ) => Ok(SdpaKernel::VectorGqa),
        Some(MetalVariant::SdpaSteel) => Ok(SdpaKernel::Steel),
        Some(MetalVariant::SdpaSteelHalf) => Ok(SdpaKernel::SteelHalf),
        _ => Err(BackendError::InvalidInput),
    }
}

fn pinned_variant(dispatch: &Dispatch) -> Option<MetalVariant> {
    dispatch
        .backend_data::<crate::ValidatedVariant>()
        .copied()
        .map(crate::ValidatedVariant::implementation)
}

fn selected_variant(dispatch: &Dispatch) -> Option<MetalVariant> {
    pinned_variant(dispatch).or_else(|| {
        crate::variants::portable_implementation(forja_core::OperationKind::of(dispatch.op()))
    })
}

fn vector_two_pass_blocks(dispatch: &Dispatch) -> Option<u32> {
    #[cfg(test)]
    if let Some(blocks) = FORCED_VECTOR_BLOCKS.with(Cell::get) {
        return Some(blocks);
    }
    match selected_variant(dispatch)? {
        MetalVariant::SdpaVectorTwoPass64 | MetalVariant::SdpaVectorGqaTwoPass64 => Some(64),
        MetalVariant::SdpaVectorTwoPass128 | MetalVariant::SdpaVectorGqaTwoPass128 => Some(128),
        MetalVariant::SdpaVectorTwoPass256 | MetalVariant::SdpaVectorGqaTwoPass256 => Some(256),
        MetalVariant::SdpaVectorTwoPass512 => Some(512),
        MetalVariant::SdpaVectorTwoPass1024 => Some(1024),
        MetalVariant::SdpaVectorGqaTwoPass32 => Some(32),
        _ => None,
    }
}

#[cfg(test)]
fn steel_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [query, _, value] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let [_, query_length, width] = shape3(query.layout())?;
    let [_, _, value_width] = shape3(value.layout())?;
    Ok(query_length > 1 && width == value_width && matches!(width, 64 | 128))
}

#[cfg(test)]
fn half_steel_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [_, key, value] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    Ok(steel_sdpa_supported(dispatch)?
        && matches!(key.layout().dtype(), DType::F16 | DType::BF16)
        && value.layout().dtype() == key.layout().dtype())
}

#[cfg(test)]
fn paged_steel_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [query, key, value, _, _] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let [_, query_length, width] = shape3(query.layout())?;
    let value_width = value
        .layout()
        .shape()
        .get(3)
        .copied()
        .ok_or(BackendError::InvalidInput)?;
    Ok(query_length > 1
        && width == value_width
        && matches!(width, 64 | 128)
        && key.layout().shape().get(3) == Some(&width))
}

#[cfg(test)]
fn paged_half_steel_sdpa_supported(dispatch: &Dispatch) -> Result<bool, BackendError> {
    let [_, key, value, _, _] = dispatch.inputs() else {
        return Err(BackendError::InvalidInput);
    };
    Ok(paged_steel_sdpa_supported(dispatch)?
        && matches!(key.layout().dtype(), DType::F16 | DType::BF16)
        && value.layout().dtype() == key.layout().dtype())
}

#[cfg(test)]
pub(super) fn variant_supported(
    dispatch: &Dispatch,
    variant: MetalVariant,
) -> Result<bool, BackendError> {
    let rows = || {
        dispatch
            .inputs()
            .first()
            .and_then(|tensor| tensor.layout().shape().first().copied())
            .ok_or(BackendError::InvalidInput)
    };
    Ok(match variant {
        MetalVariant::MatmulPortable
        | MetalVariant::QuantMatmulPortable
        | MetalVariant::GatherQuantMatmulPortable
        | MetalVariant::SdpaPortable
        | MetalVariant::TopKPortable
        | MetalVariant::GatherQuantMatmulRouteGemv
        | MetalVariant::SdpaDecomposed
        | MetalVariant::TopKPartials => true,
        MetalVariant::GatherQuantMatmulWeightedBroad => dispatch.primitive_inputs().len() == 6,
        MetalVariant::MatmulGemv => matmul_rows(dispatch)? == 1,
        MetalVariant::MatmulGemvTransposed => matmul_kernel(dispatch)? == "gemv_transposed",
        MetalVariant::MatmulSteel64x64x16_2x2
        | MetalVariant::MatmulSteel64x64x16_1x2
        | MetalVariant::MatmulSteel64x32x32_2x2
        | MetalVariant::MatmulSteel32x64x16_1x2 => matmul_rows(dispatch)? > 1,
        MetalVariant::QuantMatmulGemv => (1..=8).contains(&rows()?),
        MetalVariant::QuantMatmulQ8FastGemv => {
            quant_matmul_kernel(dispatch)? == "quantized_gemv_q8_fast"
        }
        MetalVariant::QuantMatmulSmallM => rows()? > 1,
        MetalVariant::QuantMatmulTiled => {
            quant_matmul_kernel(dispatch)?.starts_with("quantized_gemm_tiled_")
        }
        MetalVariant::GatherQuantMatmulGrouped
        | MetalVariant::GatherQuantMatmulGroupedSmallM
        | MetalVariant::GatherQuantMatmulGroupedPairSmallM => grouped_gather_dispatch(dispatch),
        MetalVariant::SdpaVectorSinglePass
        | MetalVariant::SdpaVectorTwoPass64
        | MetalVariant::SdpaVectorTwoPass128
        | MetalVariant::SdpaVectorTwoPass256
        | MetalVariant::SdpaVectorTwoPass512
        | MetalVariant::SdpaVectorTwoPass1024
        | MetalVariant::SdpaVectorGqaTwoPass32
        | MetalVariant::SdpaVectorGqaTwoPass64
        | MetalVariant::SdpaVectorGqaTwoPass128
        | MetalVariant::SdpaVectorGqaTwoPass256 => vector_sdpa_supported(dispatch)?,
        MetalVariant::SdpaSteel => steel_sdpa_supported(dispatch)?,
        MetalVariant::SdpaSteelHalf => half_steel_sdpa_supported(dispatch)?,
        MetalVariant::PagedSdpaVector | MetalVariant::PagedSdpaDecomposed => {
            matches!(dispatch.op(), Op::PagedSdpa { .. })
        }
        MetalVariant::PagedSdpaVectorGqaTwoPass32
        | MetalVariant::PagedSdpaVectorGqaTwoPass64
        | MetalVariant::PagedSdpaVectorGqaTwoPass128
        | MetalVariant::PagedSdpaVectorGqaTwoPass256 => {
            matches!(dispatch.op(), Op::PagedSdpa { .. })
        }
        MetalVariant::PagedSdpaSteel => paged_steel_sdpa_supported(dispatch)?,
        MetalVariant::PagedSdpaSteelHalf => paged_half_steel_sdpa_supported(dispatch)?,
        MetalVariant::TopKSingleK8 => dispatch_kernel(dispatch)? == "topk_single",
        MetalVariant::SamplePortable => dispatch_kernel(dispatch)? == "sample_rejection",
        MetalVariant::SampleParallel => dispatch_kernel(dispatch)? == "sample_parallel",
    })
}

#[cfg(test)]
fn matmul_rows(dispatch: &Dispatch) -> Result<u32, BackendError> {
    let shape = dispatch
        .inputs()
        .first()
        .ok_or(BackendError::InvalidInput)?
        .layout()
        .shape();
    shape
        .len()
        .checked_sub(2)
        .and_then(|index| shape.get(index))
        .copied()
        .ok_or(BackendError::InvalidInput)
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

fn dispatch_threadgroups_indirect(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    arguments: &BufferBinding,
    threads_per_threadgroup: MTLSize,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    debug_assert!(arguments.len >= 3 * size_of::<u32>());
    record_plan_call(PlanCall::IndirectDispatch {
        address: arguments.address,
        threads_per_threadgroup,
    });
    // SAFETY: `arguments` retains a live allocation containing the aligned three-u32 indirect
    // dispatch record initialized by the preceding sample kernels.
    unsafe {
        encoder.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
            arguments.address,
            threads_per_threadgroup,
        );
    }
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
    granularity: ProfileTimestampGranularity,
    index: usize,
) {
    use objc2_metal::MTL4ComputeCommandEncoder;

    let Some(timestamps) = timestamps else {
        return;
    };
    // SAFETY: Profiled heaps reserve `2 + 2 * dispatches + internal` entries. Outer indices stay
    // below `2 + 2 * dispatches`; the checked internal writer uses that value as its lower bound.
    unsafe {
        encoder.writeTimestampWithGranularity_intoHeap_atIndex(
            match granularity {
                ProfileTimestampGranularity::Relaxed => MTL4TimestampGranularity::Relaxed,
                ProfileTimestampGranularity::Precise => MTL4TimestampGranularity::Precise,
            },
            &timestamps.heap,
            index,
        );
    }
}

fn write_internal_timestamp(
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    timestamps: Option<&GpuTimestamps>,
    next_timestamp: &mut usize,
) -> Result<(), BackendError> {
    record_plan_call(PlanCall::InternalBoundary);
    if let Some(timestamps) = timestamps {
        if *next_timestamp >= timestamps.count {
            return Err(BackendError::ExecutionFailed);
        }
        write_dispatch_timestamp(
            encoder,
            Some(timestamps),
            ProfileTimestampGranularity::Precise,
            *next_timestamp,
        );
        *next_timestamp = next_timestamp
            .checked_add(1)
            .ok_or(BackendError::ExecutionFailed)?;
    }
    Ok(())
}

fn row_thread_count(max_threads: usize, width: u32) -> Result<usize, BackendError> {
    let rounded_width = usize::try_from(width)
        .map_err(|_| BackendError::ExecutionFailed)?
        .next_multiple_of(32);
    Ok(if rounded_width <= max_threads {
        rounded_width
    } else {
        max_threads.min(256)
    })
}

fn row_dispatch_geometry(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output: &Layout,
    width: u32,
) -> Result<(MTLSize, MTLSize), BackendError> {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let thread_count = row_thread_count(max_threads, width)?;
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

fn dense_epilogue_specialization(
    config: GemmConfig,
    rows: u32,
    columns: u32,
    inner: u32,
) -> [(u32, u32); 3] {
    [
        (200, u32::from(rows.is_multiple_of(config.block_rows))),
        (201, u32::from(columns.is_multiple_of(config.block_columns))),
        (202, u32::from(inner.is_multiple_of(config.block_inner))),
    ]
}

fn dense_epilogue_specializations(config: GemmConfig) -> [[(u32, u32); 3]; 8] {
    let mut values = [[(0, 0); 3]; 8];
    let mut index = 0;
    for rows in [config.block_rows - 1, config.block_rows] {
        for columns in [config.block_columns - 1, config.block_columns] {
            for inner in [config.block_inner - 1, config.block_inner] {
                values[index] = dense_epilogue_specialization(config, rows, columns, inner);
                index += 1;
            }
        }
    }
    values
}

fn quant_epilogue_specialization(columns: u32, inner: u32) -> u32 {
    u32::from(columns.is_multiple_of(8) && inner.is_multiple_of(512)) + 1
}

fn gather_epilogue_specialization(columns: u32, inner: u32) -> u32 {
    u32::from(columns.is_multiple_of(8) && inner.is_multiple_of(512)) + 1
}

struct EpiloguePipelineRequest {
    family: epilogue_codegen::FamilySourceTag,
    epilogue: Arc<PreparedEpilogue>,
    implementation: &'static str,
    constants: Vec<(u32, u32)>,
}

fn epilogue_pipeline_requests(
    dispatch: &Dispatch,
    parameter_dependent: bool,
) -> Result<Vec<EpiloguePipelineRequest>, BackendError> {
    let epilogue = Arc::clone(
        dispatch
            .prepared_epilogue()
            .ok_or(BackendError::InvalidInput)?,
    );
    match dispatch.op() {
        Op::Matmul => dense_epilogue_requests(dispatch, epilogue, parameter_dependent),
        Op::QuantMatmul { bits, group_size } => {
            let [input, _, scales, _] = dispatch.primitive_inputs() else {
                return Err(BackendError::InvalidInput);
            };
            let mut constants = vec![
                (0, dtype_code(input.layout().dtype())),
                (1, dtype_code(scales.layout().dtype())),
                (2, dtype_code(dispatch.output().layout().dtype())),
                (12, u32::from(bits)),
                (13, group_size),
            ];
            let implementation = quant_matmul_kernel(dispatch)?;
            if parameter_dependent {
                Ok([
                    quant_epilogue_specialization(7, 511),
                    quant_epilogue_specialization(8, 512),
                ]
                .into_iter()
                .map(|words| {
                    let mut constants = constants.clone();
                    constants.push((15, words));
                    EpiloguePipelineRequest {
                        family: epilogue_codegen::FamilySourceTag::Quant,
                        epilogue: Arc::clone(&epilogue),
                        implementation,
                        constants,
                    }
                })
                .collect())
            } else {
                let inner = *input
                    .layout()
                    .shape()
                    .last()
                    .ok_or(BackendError::InvalidInput)?;
                let columns = *dispatch
                    .output()
                    .layout()
                    .shape()
                    .last()
                    .ok_or(BackendError::InvalidInput)?;
                constants.push((15, quant_epilogue_specialization(columns, inner)));
                Ok(vec![EpiloguePipelineRequest {
                    family: epilogue_codegen::FamilySourceTag::Quant,
                    epilogue,
                    implementation,
                    constants,
                }])
            }
        }
        Op::GatherQuantMatmul { bits, group_size } => {
            gather_epilogue_requests(dispatch, epilogue, bits, group_size, parameter_dependent)
        }
        _ => Err(BackendError::UnsupportedOperation),
    }
}

fn dense_epilogue_requests(
    dispatch: &Dispatch,
    epilogue: Arc<PreparedEpilogue>,
    parameter_dependent: bool,
) -> Result<Vec<EpiloguePipelineRequest>, BackendError> {
    let [left, right] = dispatch.primitive_inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let implementation = matmul_kernel(dispatch)?;
    let constants = vec![
        (0, dtype_code(left.layout().dtype())),
        (1, dtype_code(right.layout().dtype())),
        (2, dtype_code(dispatch.output().layout().dtype())),
    ];
    if matches!(
        selected_variant(dispatch),
        Some(MetalVariant::MatmulGemv | MetalVariant::MatmulGemvTransposed)
    ) {
        return Ok(vec![EpiloguePipelineRequest {
            family: epilogue_codegen::FamilySourceTag::Dense,
            epilogue,
            implementation,
            constants,
        }]);
    }
    let config = match selected_variant(dispatch) {
        Some(MetalVariant::MatmulPortable | MetalVariant::MatmulSteel64x32x32_2x2) => GEMM_64_32,
        Some(MetalVariant::MatmulSteel64x64x16_2x2) => GEMM_64_64_FLOAT,
        Some(MetalVariant::MatmulSteel64x64x16_1x2) => GEMM_64_64_HALF,
        Some(MetalVariant::MatmulSteel32x64x16_1x2) => GEMM_32_64,
        _ => return Err(BackendError::UnsupportedOperation),
    };
    if parameter_dependent {
        return Ok(dense_epilogue_specializations(config)
            .into_iter()
            .map(|specialization| {
                let mut constants = constants.clone();
                constants.extend(specialization);
                EpiloguePipelineRequest {
                    family: epilogue_codegen::FamilySourceTag::Dense,
                    epilogue: Arc::clone(&epilogue),
                    implementation,
                    constants,
                }
            })
            .collect());
    }
    let left_shape = left.layout().shape();
    let right_shape = right.layout().shape();
    let mut constants = constants;
    constants.extend(dense_epilogue_specialization(
        config,
        left_shape[left_shape.len() - 2],
        right_shape[right_shape.len() - 1],
        left_shape[left_shape.len() - 1],
    ));
    Ok(vec![EpiloguePipelineRequest {
        family: epilogue_codegen::FamilySourceTag::Dense,
        epilogue,
        implementation,
        constants,
    }])
}

fn gather_epilogue_requests(
    dispatch: &Dispatch,
    epilogue: Arc<PreparedEpilogue>,
    bits: u8,
    group_size: u32,
    parameter_dependent: bool,
) -> Result<Vec<EpiloguePipelineRequest>, BackendError> {
    let inputs = dispatch.primitive_inputs();
    let input = inputs.first().ok_or(BackendError::InvalidInput)?;
    let scales = inputs.get(2).ok_or(BackendError::InvalidInput)?;
    let mut constants = vec![
        (0, dtype_code(input.layout().dtype())),
        (1, dtype_code(scales.layout().dtype())),
        (2, dtype_code(dispatch.output().layout().dtype())),
        (12, u32::from(bits)),
        (13, group_size),
    ];
    if inputs.len() == 6 {
        let inner = *input
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let columns = *dispatch
            .output()
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let routes = *input
            .layout()
            .shape()
            .get(1)
            .ok_or(BackendError::InvalidInput)?;
        constants.push((14, u32::from(columns % 4 == 0 && inner % 512 == 0) + 1));
        return Ok(vec![EpiloguePipelineRequest {
            family: epilogue_codegen::FamilySourceTag::GatherQuant,
            epilogue,
            implementation: weighted_gather_schedule(selected_variant(dispatch), routes)?
                .implementation,
            constants,
        }]);
    }
    if inputs.len() == 8 {
        return paired_gather_epilogue_requests(
            dispatch,
            epilogue,
            &constants,
            parameter_dependent,
        );
    }
    if grouped_gather_dispatch(dispatch) {
        let block_rows = grouped_block_rows(dispatch);
        constants.push((16, block_rows));
        return Ok(vec![EpiloguePipelineRequest {
            family: epilogue_codegen::FamilySourceTag::GatherQuant,
            epilogue,
            implementation: if block_rows == 8 {
                "grouped_quantized_gemm_small_m"
            } else {
                "grouped_quantized_gemm"
            },
            constants,
        }]);
    }
    if !matches!(
        selected_variant(dispatch),
        Some(MetalVariant::GatherQuantMatmulPortable | MetalVariant::GatherQuantMatmulRouteGemv)
    ) {
        return Err(BackendError::UnsupportedOperation);
    }
    if parameter_dependent {
        return Ok([
            gather_epilogue_specialization(7, 511),
            gather_epilogue_specialization(8, 512),
        ]
        .into_iter()
        .map(|words| {
            let mut constants = constants.clone();
            constants.push((14, words));
            EpiloguePipelineRequest {
                family: epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue: Arc::clone(&epilogue),
                implementation: "gather_quantized_gemv",
                constants,
            }
        })
        .collect());
    }
    let inner = *input
        .layout()
        .shape()
        .last()
        .ok_or(BackendError::InvalidInput)?;
    let columns = *dispatch
        .output()
        .layout()
        .shape()
        .last()
        .ok_or(BackendError::InvalidInput)?;
    constants.push((14, gather_epilogue_specialization(columns, inner)));
    Ok(vec![EpiloguePipelineRequest {
        family: epilogue_codegen::FamilySourceTag::GatherQuant,
        epilogue,
        implementation: "gather_quantized_gemv",
        constants,
    }])
}

fn paired_gather_epilogue_requests(
    dispatch: &Dispatch,
    epilogue: Arc<PreparedEpilogue>,
    constants: &[(u32, u32)],
    parameter_dependent: bool,
) -> Result<Vec<EpiloguePipelineRequest>, BackendError> {
    if grouped_gather_dispatch(dispatch) {
        return Ok(vec![EpiloguePipelineRequest {
            family: epilogue_codegen::FamilySourceTag::GatherQuant,
            epilogue,
            implementation: "gathered_pair_from_projections",
            constants: vec![(2, dtype_code(dispatch.output().layout().dtype()))],
        }]);
    }
    if !matches!(
        selected_variant(dispatch),
        Some(MetalVariant::GatherQuantMatmulPortable | MetalVariant::GatherQuantMatmulRouteGemv)
    ) {
        return Err(BackendError::UnsupportedOperation);
    }
    let specializations = if parameter_dependent {
        vec![
            gather_epilogue_specialization(7, 511),
            gather_epilogue_specialization(8, 512),
        ]
    } else {
        let input = dispatch
            .primitive_inputs()
            .first()
            .ok_or(BackendError::InvalidInput)?;
        let inner = *input
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        let columns = *dispatch
            .output()
            .layout()
            .shape()
            .last()
            .ok_or(BackendError::InvalidInput)?;
        vec![gather_epilogue_specialization(columns, inner)]
    };
    Ok(specializations
        .into_iter()
        .map(|words| {
            let mut constants = constants.to_vec();
            constants.push((14, words));
            EpiloguePipelineRequest {
                family: epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue: Arc::clone(&epilogue),
                implementation: "gather_quantized_pair",
                constants,
            }
        })
        .collect())
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
    max_threadgroup_memory_length: usize,
    fast_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    nan_preserving_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    unavailable_pipelines: HashSet<PipelineKey>,
    epilogue_libraries:
        HashMap<epilogue_codegen::SourceKey, Retained<ProtocolObject<dyn MTLLibrary>>>,
    epilogue_pipelines: HashMap<
        epilogue_codegen::PipelineKey,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    >,
    epilogue_counts: EpilogueCompileCounts,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EpilogueCompileCounts {
    libraries: u64,
    pipelines: u64,
    cache_hits: u64,
    fallbacks: u64,
}

impl PipelineCache {
    #[cfg(test)]
    pub(super) fn set_threadgroup_memory_limit(&mut self, bytes: usize) {
        self.max_threadgroup_memory_length = bytes;
    }

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
            max_threadgroup_memory_length: device.maxThreadgroupMemoryLength(),
            fast_library,
            nan_preserving_library,
            pipelines: HashMap::new(),
            unavailable_pipelines: HashSet::new(),
            epilogue_libraries: HashMap::new(),
            epilogue_pipelines: HashMap::new(),
            epilogue_counts: EpilogueCompileCounts::default(),
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
        if self.unavailable_pipelines.contains(&key) {
            return Err(BackendError::UnsupportedOperation);
        }
        let values = function_constants(constants)?;
        let library = if nan_preserving_kernel(name) {
            &self.nan_preserving_library
        } else {
            &self.fast_library
        };
        let function_name = NSString::from_str(name);
        let function = library
            .newFunctionWithName_constantValues_error(&function_name, &values)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        if (name.starts_with("steel_attention_") || name.starts_with("paged_steel_attention_"))
            && pipeline.staticThreadgroupMemoryLength() > self.max_threadgroup_memory_length
        {
            self.unavailable_pipelines.insert(key);
            return Err(BackendError::UnsupportedOperation);
        }
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn prepare_epilogue(
        &mut self,
        family: epilogue_codegen::FamilySourceTag,
        epilogue: &PreparedEpilogue,
        name: &str,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let source_key = epilogue_codegen::SourceKey::new(family, epilogue);
        let pipeline_key =
            epilogue_codegen::PipelineKey::new(source_key.clone(), name, constants.to_vec());
        if let Some(pipeline) = self.epilogue_pipelines.get(&pipeline_key) {
            self.epilogue_counts.cache_hits = self.epilogue_counts.cache_hits.saturating_add(1);
            return Ok(pipeline.clone());
        }
        if self.epilogue_libraries.contains_key(&source_key) {
            self.epilogue_counts.cache_hits = self.epilogue_counts.cache_hits.saturating_add(1);
        } else {
            let generated = epilogue_codegen::generate_source(epilogue)
                .map_err(|_| program_compile_failed(source_key.hash))?;
            let source = NSString::from_str(&generated);
            let options = compile_options(MTLMathMode::Safe);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, Some(&options))
                .map_err(|_| program_compile_failed(source_key.hash))?;
            self.epilogue_libraries.insert(source_key.clone(), library);
            self.epilogue_counts.libraries = self.epilogue_counts.libraries.saturating_add(1);
        }
        let library = self
            .epilogue_libraries
            .get(&source_key)
            .ok_or(BackendError::ExecutionFailed)?;
        let values = function_constants(&pipeline_key.constants)?;
        let function = library
            .newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
            .map_err(|_| program_compile_failed(source_key.hash))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| program_compile_failed(source_key.hash))?;
        self.epilogue_pipelines
            .insert(pipeline_key, pipeline.clone());
        self.epilogue_counts.pipelines = self.epilogue_counts.pipelines.saturating_add(1);
        Ok(pipeline)
    }

    fn lookup_epilogue(
        &mut self,
        family: epilogue_codegen::FamilySourceTag,
        epilogue: &PreparedEpilogue,
        name: &str,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let result = self.lookup_prepared_epilogue(family, epilogue, name, constants);
        debug_assert!(result.is_ok(), "epilogue pipeline was not prepared");
        result
    }

    fn lookup_prepared_epilogue(
        &mut self,
        family: epilogue_codegen::FamilySourceTag,
        epilogue: &PreparedEpilogue,
        name: &str,
        constants: &[(u32, u32)],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let source = epilogue_codegen::SourceKey::new(family, epilogue);
        let key = epilogue_codegen::PipelineKey::new(source, name, constants.to_vec());
        if let Some(pipeline) = self.epilogue_pipelines.get(&key) {
            self.epilogue_counts.cache_hits = self.epilogue_counts.cache_hits.saturating_add(1);
            return Ok(pipeline.clone());
        }
        self.epilogue_counts.fallbacks = self.epilogue_counts.fallbacks.saturating_add(1);
        Err(BackendError::ExecutionFailed)
    }

    fn missing_epilogue_pipelines(&self, requests: &[EpiloguePipelineRequest]) -> usize {
        requests
            .iter()
            .map(|request| {
                let source = epilogue_codegen::SourceKey::new(request.family, &request.epilogue);
                epilogue_codegen::PipelineKey::new(
                    source,
                    request.implementation,
                    request.constants.clone(),
                )
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|key| !self.epilogue_pipelines.contains_key(key))
            .count()
    }

    fn prepare_epilogue_pipelines(
        &mut self,
        requests: &[EpiloguePipelineRequest],
    ) -> Result<(), BackendError> {
        for request in requests {
            self.prepare_epilogue(
                request.family,
                &request.epilogue,
                request.implementation,
                &request.constants,
            )?;
        }
        Ok(())
    }

    fn take_epilogue_counts(&mut self) -> EpilogueCompileCounts {
        std::mem::take(&mut self.epilogue_counts)
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

fn function_constants(
    constants: &[(u32, u32)],
) -> Result<Retained<MTLFunctionConstantValues>, BackendError> {
    let values = MTLFunctionConstantValues::new();
    for &(index, value) in constants {
        // SAFETY: `value` is live for the call, its type matches `MTLDataType::UInt`, and the
        // caller supplies indices declared by the selected kernel.
        unsafe {
            values.setConstantValue_type_atIndex(
                NonNull::from(&value).cast::<c_void>(),
                MTLDataType::UInt,
                usize::try_from(index).map_err(|_| BackendError::ExecutionFailed)?,
            );
        }
    }
    Ok(values)
}

fn nan_preserving_kernel(name: &str) -> bool {
    matches!(
        name,
        "softmax_single"
            | "softmax_looped"
            | "argmax_partials"
            | "topk_single"
            | "topk_exact_fallback"
            | "topk_partials"
            | "topk_finalize"
            | "sample_rejection_proposal_partials"
            | "sample_rejection_threshold_partials"
            | "sample_rejection_threshold_finalize"
            | "sample_topk_partials"
            | "sample_topk_finalize"
            | "sample_exact_fallback"
            | "sample_radix_histogram"
            | "sample_index_histogram"
            | "sample_index_locate"
            | "sample_compact"
            | "sample_partials"
            | "sample_reduce_finalize"
            | "sample_candidates_finalize"
            | "sample_masked_partials"
            | "sample_masked_finalize"
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
        sync::Barrier,
        thread,
        time::{Duration, Instant},
    };

    use forja_core::{
        Affine, Backend, BackendDispatchData, CommandList, DType, GraphLimits, Op, ParamSpace,
        RuleArm, Slice, Submission, SymbolicLayout, TemplateTensor, Tensor, VariantRule, ViewOp,
        correction_sampling_seed,
        program::{
            BinOp, BoundProgram, EpilogueSignature, Inst, KernelSignature, Program, ProgramKind,
            RedOp, UnOp, ValidatedProgram, bind_program, prepare_epilogue, prepare_program,
        },
    };
    use forja_cpu::CpuBackend;
    use forja_testing::{
        DifferentialComparison, TensorSpec, assert_backends_agree,
        assert_backends_agree_at_tolerance, assert_fused_decomposed, assert_outputs_agree,
    };

    use super::*;

    #[test]
    fn metal_compiles_paged_half_steel_attention() {
        let backend = MetalBackend::new().unwrap();
        backend
            .pipelines
            .lock()
            .unwrap()
            .get(
                "paged_steel_attention_half_128_96x16",
                &[(0, 0), (1, 1), (2, 0), (3, 1)],
            )
            .unwrap();
    }

    fn argmax_op() -> Op {
        Op::Argmax { chunk_width: 2_048 }
    }

    fn sample_op(position: u32) -> Op {
        Op::Sample {
            position,
            chunk_width: 2_048,
            fast_top_k: 20,
            fast_max_chunks: 102,
            rejection_rounds: 2,
        }
    }

    fn sample_masked_op(position: u32) -> Op {
        Op::SampleMasked {
            position,
            chunk_width: 2_048,
            fast_top_k: 20,
            fast_max_chunks: 102,
            rejection_rounds: 2,
        }
    }

    fn symbolic_prefix(base: &Tensor, space: ParamSpace, len: Affine) -> TemplateTensor {
        let layout = SymbolicLayout::new(base.layout().clone(), space)
            .slice(0, 0.into(), len, 1)
            .unwrap();
        TemplateTensor::symbolic(base.clone(), layout).unwrap()
    }

    fn dynamic_epilogue_variant_graph(backend: &MetalBackend) -> GraphTemplate {
        let space = ParamSpace::new(std::iter::once(1..=7).collect()).unwrap();
        let rows = Affine::parameter(0, 0, 1);
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let right = TemplateTensor::from(backend.alloc(DType::F32, &[33, 7]).unwrap());
        let output = backend.alloc(DType::F32, &[7, 7]).unwrap();
        let left = symbolic_prefix(&left, space.clone(), rows);
        let output = symbolic_prefix(&output, space.clone(), rows);
        let epilogue = prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Const(2.0),
                    Inst::Binary(BinOp::Mul, 0, 1),
                ],
                outputs: vec![(0, 2)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, 2, vec![], DType::F32),
        )
        .unwrap();
        let rule = VariantRule::new(
            &space,
            0,
            vec![
                RuleArm::new(1, 1, "matmul.gemv"),
                RuleArm::new(2, 7, "matmul.steel-64x64x16-2x2"),
            ],
        )
        .unwrap();
        let variants = backend
            .validate_variant_rule(
                &space,
                &rule,
                Op::Matmul.into(),
                &[&left, &right],
                &[&output],
            )
            .unwrap();
        let mut graph = GraphTemplate::new(space, GraphLimits::default());
        graph
            .dispatch_epilogue_variant_rule(
                Op::Matmul,
                &[&left, &right],
                &output,
                &epilogue,
                &[],
                0,
                rule.arms()
                    .iter()
                    .zip(variants)
                    .map(|(arm, variant)| (arm.hi(), BackendDispatchData::new(variant)))
                    .collect(),
            )
            .unwrap();
        graph
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

    fn allowed_sample_indices(
        logits: &[f32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Vec<u32> {
        let mut allowed = (0..logits.len()).collect::<Vec<_>>();
        allowed.sort_unstable_by(|&left, &right| {
            logits[right]
                .total_cmp(&logits[left])
                .then_with(|| right.cmp(&left))
        });
        let requested = usize::try_from(top_k).unwrap();
        let cap = if requested == 0 {
            allowed.len()
        } else {
            requested
        };
        allowed.truncate(cap.min(allowed.len()));
        if top_p < 1.0 {
            let maximum = logits[allowed[0]];
            let weights = allowed
                .iter()
                .map(|&index| ((logits[index] - maximum) / temperature).exp())
                .collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            let mut cumulative = 0.0;
            let keep = weights
                .iter()
                .position(|weight| {
                    cumulative += weight / total;
                    cumulative >= top_p
                })
                .map_or(weights.len(), |index| index + 1);
            allowed.truncate(keep.max(1));
        }
        allowed
            .into_iter()
            .map(|index| u32::try_from(index).unwrap())
            .collect()
    }

    struct SampleRoundsOverride(Option<u32>);

    impl SampleRoundsOverride {
        fn set(rounds: u32) -> Self {
            let previous =
                FORCED_SAMPLE_REJECTION_ROUNDS.with(|forced| forced.replace(Some(rounds)));
            Self(previous)
        }
    }

    impl Drop for SampleRoundsOverride {
        fn drop(&mut self) {
            FORCED_SAMPLE_REJECTION_ROUNDS.with(|forced| forced.set(self.0));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sample<B: Backend>(
        backend: &B,
        logits: &[u8],
        width: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        position: u32,
    ) -> Result<u32, BackendError> {
        run_typed_sample(
            backend,
            DType::F32,
            logits,
            width,
            temperature,
            top_k,
            top_p,
            seed,
            position,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_metal_sample(
        backend: &MetalBackend,
        dtype: DType,
        logits: &[u8],
        width: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        position: u32,
        variant: &str,
    ) -> Result<u32, BackendError> {
        run_typed_sample(
            backend,
            dtype,
            logits,
            width,
            temperature,
            top_k,
            top_p,
            seed,
            position,
            |commands| {
                let dispatch = commands.last_dispatch().ok_or(BackendError::InvalidInput)?;
                let selected = backend
                    .validate_variant(dispatch, variant)
                    .map_err(|_| BackendError::InvalidInput)?;
                if commands.set_last_backend_data(BackendDispatchData::new(selected)) {
                    Ok(())
                } else {
                    Err(BackendError::InvalidInput)
                }
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_typed_sample<B: Backend>(
        backend: &B,
        dtype: DType,
        logits: &[u8],
        width: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        position: u32,
        select: impl FnOnce(&mut CommandList) -> Result<(), BackendError>,
    ) -> Result<u32, BackendError> {
        let input = backend.alloc(dtype, &[1, width])?;
        backend.write(&input, logits)?;
        let params = backend.alloc(DType::U32, &[5])?;
        backend.write(&params, &sample_params(temperature, top_k, top_p, seed))?;
        let output = backend.alloc(DType::U32, &[1])?;
        let mut commands = CommandList::new();
        commands
            .dispatch(sample_op(position), &[&input, &params], &output)
            .map_err(|_| BackendError::InvalidInput)?;
        select(&mut commands)?;
        backend.submit(commands)?.wait()?;
        let sampled = backend
            .read(&output)?
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| BackendError::ExecutionFailed)?;
        for tensor in [&input, &params, &output] {
            backend.release(tensor)?;
        }
        Ok(sampled)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_masked_sample<B: Backend>(
        backend: &B,
        logits: &[u8],
        width: u32,
        allow: &[u32],
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
        let mask = backend.alloc(DType::U32, &[width.div_ceil(32)])?;
        backend.write(
            &mask,
            &allow
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let output = backend.alloc(DType::U32, &[1])?;
        let mut commands = CommandList::new();
        commands
            .dispatch(
                sample_masked_op(position),
                &[&input, &params, &mask],
                &output,
            )
            .map_err(|_| BackendError::InvalidInput)?;
        backend.submit(commands)?.wait()?;
        let sampled = backend
            .read(&output)?
            .as_slice()
            .try_into()
            .map(u32::from_le_bytes)
            .map_err(|_| BackendError::ExecutionFailed)?;
        for tensor in [&input, &params, &mask, &output] {
            backend.release(tensor)?;
        }
        Ok(sampled)
    }

    fn run_speculate_then_correct<B: Backend>(
        backend: &B,
        logits: &[u8],
        allow: &[u32],
        seed: u64,
        position: u32,
    ) -> Result<(u32, bool), BackendError> {
        let provisional = run_sample(backend, logits, 8, 1.0, 0, 1.0, seed, position)?;
        if allow[usize::try_from(provisional / 32).map_err(|_| BackendError::InvalidInput)?]
            & (1 << (provisional % 32))
            != 0
        {
            Ok((provisional, false))
        } else {
            run_masked_sample(
                backend,
                logits,
                8,
                allow,
                1.0,
                0,
                1.0,
                correction_sampling_seed(seed),
                position,
            )
            .map(|token| (token, true))
        }
    }

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
    }

    #[test]
    fn plan_recorder_is_scoped_and_preserves_call_order() {
        record_plan_call(PlanCall::ArgumentTable);
        let recorder = PlanRecorderScope::enter(0, None);
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
        assert_eq!(PlanRecorderScope::enter(0, None).finish().len(), 0);
    }

    #[test]
    fn replay_plans_preserve_argument_table_activation() {
        let plan = || StaticEncodingPlan {
            calls: vec![PlanCall::ArgumentTable],
        };
        let mut plans = vec![Some(plan()), Some(plan())];

        dedupe_pipeline_calls(&mut plans);

        assert!(plans.iter().all(|plan| matches!(
            plan.as_ref().map(|plan| plan.calls.as_slice()),
            Some([PlanCall::ArgumentTable])
        )));
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
    fn row_thread_count_respects_pipeline_limits() {
        assert_eq!(row_thread_count(512, 33).unwrap(), 64);
        assert_eq!(row_thread_count(512, 4_097).unwrap(), 256);
        assert_eq!(row_thread_count(768, 700).unwrap(), 704);
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
            assert_eq!(expected_remainder.len(), 0);
            assert_eq!(actual_remainder.len(), 0);
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
            commands.dispatch(Op::Copy, &[&left], &output).unwrap();
            let submission = backend.submit(commands).unwrap();
            if sequence % 2 == 0 {
                submission.wait().unwrap();
            } else {
                drop(submission);
            }
            let actual = backend.read(&output).unwrap();
            let expected = vec![1.0_f32; len]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
            backend.release(&left).unwrap();
            backend.release(&right).unwrap();
            backend.release(&output).unwrap();
        }
    }

    fn repeated_copy_commands(tensors: &[Tensor; 3], dispatches: usize) -> CommandList {
        let mut commands = CommandList::new();
        for _ in 0..dispatches {
            commands
                .dispatch(Op::Copy, &[&tensors[0]], &tensors[2])
                .unwrap();
        }
        commands
    }

    fn submission_workloads(backend: &MetalBackend, count: usize) -> Vec<[Tensor; 3]> {
        (0..count)
            .map(|_| {
                [
                    backend.alloc(DType::F32, &[1024, 4097]).unwrap(),
                    backend.alloc(DType::F32, &[1024, 4097]).unwrap(),
                    backend.alloc(DType::F32, &[1024, 4097]).unwrap(),
                ]
            })
            .collect()
    }

    fn run_submission_wave(
        backend: &Arc<MetalBackend>,
        workloads: &[[Tensor; 3]],
        dispatch_counts: &[usize],
    ) -> Vec<(MetalSubmission, Duration, Duration)> {
        assert_eq!(workloads.len(), dispatch_counts.len());
        let submitted = Arc::new(Barrier::new(workloads.len()));
        thread::scope(|scope| {
            let workers = workloads
                .iter()
                .zip(dispatch_counts)
                .map(|(tensors, &dispatches)| {
                    let backend = Arc::clone(backend);
                    let submitted = Arc::clone(&submitted);
                    scope.spawn(move || {
                        let commands = repeated_copy_commands(tensors, dispatches);
                        let started = Instant::now();
                        let submission = backend.submit(commands).unwrap();
                        submitted.wait();
                        submission.wait().unwrap();
                        let wall = started.elapsed();
                        let gpu = submission.gpu_time().unwrap();
                        (submission, gpu, wall)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect()
        })
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
        let objects = backend.in_flight.checkout(&backend, capacity).unwrap();
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
            internal_profiles: Vec::new(),
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
            internal_profiles: Vec::new(),
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

    fn submit_with_planned_arguments(backend: &MetalBackend, commands: CommandList) {
        let capacity = backend
            .argument_capacity(&commands.clone().into_dispatches())
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        backend.in_flight.drain_done();
        assert!(
            backend
                .in_flight
                .pool
                .lock()
                .unwrap()
                .iter()
                .any(|objects| objects.argument_buffer.capacity == capacity)
        );
    }

    #[test]
    fn paged_variants_encode_with_exact_planned_arguments() {
        let backend = MetalBackend::new().unwrap();
        let table = backend.alloc(DType::U32, &[1, 2]).unwrap();
        let generations = backend.alloc(DType::U32, &[1]).unwrap();
        let activity = backend.alloc(DType::U32, &[1]).unwrap();
        backend
            .write(
                &table,
                &[0_u32, 1]
                    .into_iter()
                    .flat_map(u32::to_ne_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        backend.write(&generations, &1_u32.to_ne_bytes()).unwrap();
        backend.write(&activity, &1_u32.to_ne_bytes()).unwrap();

        let ordinary = backend.alloc(DType::F32, &[1, 2]).unwrap();
        let pool = backend.alloc(DType::F32, &[1, 4, 2]).unwrap();
        let copy_output = backend.alloc(DType::F32, &[1, 2]).unwrap();
        for direction in [
            PagedCopyDirection::IntoPages,
            PagedCopyDirection::OutOfPages,
        ] {
            for explicit_activity in [false, true] {
                let (first, output) = match direction {
                    PagedCopyDirection::IntoPages => (&ordinary, &pool),
                    PagedCopyDirection::OutOfPages => (&pool, &copy_output),
                };
                let mut inputs = vec![first, &table, &generations];
                if explicit_activity {
                    inputs.push(&activity);
                }
                let mut commands = CommandList::new();
                commands
                    .dispatch(
                        Op::PagedCopy {
                            direction,
                            start: 0,
                            page_tokens: 4,
                        },
                        &inputs,
                        output,
                    )
                    .unwrap();
                submit_with_planned_arguments(&backend, commands);
            }
        }

        for (variant, dtype, query_length) in [
            ("paged-sdpa.vector", DType::F32, 1),
            ("paged-sdpa.vector-gqa-two-pass-32", DType::F16, 1),
            ("paged-sdpa.vector-gqa-two-pass", DType::F16, 1),
            ("paged-sdpa.vector-gqa-two-pass-128", DType::F16, 1),
            ("paged-sdpa.vector-gqa-two-pass-256", DType::F16, 1),
            ("paged-sdpa.decomposed", DType::F32, 7),
            ("paged-sdpa.steel", DType::F32, 7),
            ("paged-sdpa.steel-half", DType::F16, 7),
        ] {
            let query = backend.alloc(DType::F32, &[16, query_length, 128]).unwrap();
            let key = backend.alloc(dtype, &[1, 64, 8, 128]).unwrap();
            let value = backend.alloc(dtype, &[1, 64, 8, 128]).unwrap();
            let output = backend.alloc(DType::F32, &[16, query_length, 128]).unwrap();
            for explicit_activity in [false, true] {
                let mut inputs = vec![&query, &key, &value, &table, &generations];
                if explicit_activity {
                    inputs.push(&activity);
                }
                let mut commands = CommandList::new();
                commands
                    .dispatch(
                        Op::PagedSdpa {
                            scale: 128.0_f32.sqrt().recip(),
                            causal: true,
                            q_start: 33 - query_length,
                            key_length: 33,
                            page_tokens: 64,
                        },
                        &inputs,
                        &output,
                    )
                    .unwrap();
                let selection = backend
                    .validate_variant(commands.last_dispatch().unwrap(), variant)
                    .unwrap();
                assert!(commands.set_last_backend_data(BackendDispatchData::new(selection)));
                submit_with_planned_arguments(&backend, commands);
            }
        }
    }

    #[test]
    fn concurrent_submissions_share_one_backend() {
        let backend = Arc::new(MetalBackend::new().unwrap());
        let workers = (0..MAX_IN_FLIGHT_SUBMISSIONS)
            .map(|worker| {
                let backend = Arc::clone(&backend);
                thread::spawn(move || run_varying_submissions(&backend, 400, worker * 400))
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
    fn submission_checkout_waits_for_a_reserved_slot() {
        let backend = Arc::new(MetalBackend::new().unwrap());
        let mut held = (0..MAX_IN_FLIGHT_SUBMISSIONS)
            .map(|_| backend.in_flight.checkout(&backend, 4096).unwrap())
            .collect::<Vec<_>>();
        let waiting_backend = Arc::clone(&backend);
        let waiter =
            thread::spawn(move || waiting_backend.in_flight.checkout(&waiting_backend, 4096));

        let started = Instant::now();
        while backend.in_flight.backpressure_wait_count() == 0 {
            assert!(started.elapsed() < backend.gpu_timeout);
            thread::yield_now();
        }
        drop(held.pop());

        let returned = waiter.join().unwrap().unwrap();
        drop(returned);
    }

    #[test]
    fn concurrent_submissions_apply_backpressure_without_allocation_growth() {
        const WORKERS: usize = MAX_IN_FLIGHT_SUBMISSIONS * 2;
        const DISPATCHES: usize = 4096;

        let backend = Arc::new(MetalBackend::new().unwrap());
        let workloads = submission_workloads(&backend, WORKERS);
        backend
            .submit(repeated_copy_commands(&workloads[0], 1))
            .unwrap()
            .wait()
            .unwrap();
        let outstanding = workloads[..MAX_IN_FLIGHT_SUBMISSIONS]
            .iter()
            .map(|tensors| {
                backend
                    .submit(repeated_copy_commands(tensors, DISPATCHES))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(backend.in_flight.len(), MAX_IN_FLIGHT_SUBMISSIONS);
        let allocated = backend.debug_state().allocation_requests;

        let completed = run_submission_wave(
            &backend,
            &workloads[MAX_IN_FLIGHT_SUBMISSIONS..],
            &[DISPATCHES; MAX_IN_FLIGHT_SUBMISSIONS],
        );
        for submission in &outstanding {
            submission.wait().unwrap();
        }
        drop(completed);
        drop(outstanding);
        backend.in_flight.drain_done();

        let state = backend.debug_state();
        assert!(state.submission_waits >= MAX_IN_FLIGHT_SUBMISSIONS);
        assert_eq!(state.submission_slots, MAX_IN_FLIGHT_SUBMISSIONS);
        assert_eq!(state.pooled_submissions, MAX_IN_FLIGHT_SUBMISSIONS);
        assert_eq!(state.allocation_requests, allocated);
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
    fn metal_submissions_expose_shared_event_serials() {
        let backend = MetalBackend::new().unwrap();
        let first = backend.submit(CommandList::new()).unwrap();
        let second = backend.submit(CommandList::new()).unwrap();

        assert_eq!(
            second.completion_serial(),
            first
                .completion_serial()
                .and_then(|value| value.checked_add(1))
        );
        first.wait().unwrap();
        second.wait().unwrap();
    }

    #[test]
    fn metal_submission_reports_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let shape = [1024, 4097];
        let input = backend.alloc(DType::F32, &shape).unwrap();
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();

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
    fn metal_profile_timestamp_granularity_preserves_submission_timing() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[1]).unwrap();
        let output = backend.alloc(DType::F32, &[1]).unwrap();
        for granularity in [
            ProfileTimestampGranularity::Relaxed,
            ProfileTimestampGranularity::Precise,
        ] {
            let mut commands = CommandList::new();
            commands.dispatch(Op::Copy, &[&input], &output).unwrap();
            let submission = backend
                .submit_profiled_with_granularity(commands, granularity)
                .unwrap();
            submission.wait().unwrap();
            assert!(
                submission
                    .gpu_time()
                    .is_some_and(|time| time > Duration::ZERO)
            );
        }
    }

    #[test]
    fn concurrent_submission_gpu_times_survive_slot_reuse() {
        let backend = Arc::new(MetalBackend::new().unwrap());
        let workloads = submission_workloads(&backend, MAX_IN_FLIGHT_SUBMISSIONS);
        let pooled = (0..MAX_IN_FLIGHT_SUBMISSIONS)
            .map(|_| backend.in_flight.checkout(&backend, 4096).unwrap())
            .collect::<Vec<_>>();
        drop(pooled);

        let counts = [1, 16, 256, 4096];
        let original = run_submission_wave(&backend, &workloads, &counts);
        let shortest = original[0].1;
        let longest = original[3].1;
        assert!(shortest > Duration::ZERO);
        assert!(longest >= shortest.saturating_mul(4));
        for (_, gpu, wall) in &original {
            assert!(*gpu <= *wall, "GPU {gpu:?}, wall {wall:?}");
        }

        for _ in 0..4 {
            drop(run_submission_wave(
                &backend,
                &workloads,
                &[4096, 256, 16, 1],
            ));
        }
        for (submission, expected, _) in original {
            assert_eq!(submission.gpu_time(), Some(expected));
        }
    }

    #[test]
    fn profiled_submission_reports_phases_and_dispatches() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&left], &output).unwrap();
        let submission = backend.submit_profiled(commands).unwrap();
        submission.wait().unwrap();
        let profile = submission.profile().unwrap();
        assert_eq!(profile.dispatches, 1);
        assert_eq!(profile.per_dispatch.len(), 1);
        let dispatch = &profile.per_dispatch[0];
        assert_eq!(dispatch.op, Op::Copy);
        assert_eq!(dispatch.kernel, "copy_contiguous");
        assert_eq!(dispatch.inputs[0].shape, [7, 33]);
        assert_eq!(dispatch.outputs[0].dtype, DType::F32);
        assert!(!dispatch.barrier);
        assert!(profile.gpu_time > Duration::ZERO);
        assert!(dispatch.gpu_time > Duration::ZERO);
        assert_eq!(dispatch.gap_before, dispatch.gpu_start);
        assert_eq!(dispatch.gpu_end, dispatch.gpu_start + dispatch.gpu_time);
        assert_eq!(profile.metadata_buffers.count, 1);
    }

    #[test]
    fn prepared_graph_owns_one_persistent_residency_set() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        graph
            .dispatch(Op::Copy, &[&left.into()], &output.into())
            .unwrap();

        let graph = backend.prepare_graph(graph).unwrap();
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        assert_eq!(state.residency.allocationCount(), 2);
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
    fn metal_graph_prepares_every_epilogue_variant_specialization() {
        let backend = MetalBackend::new().unwrap();
        let graph = backend
            .prepare_graph(dynamic_epilogue_variant_graph(&backend))
            .unwrap();
        {
            let pipelines = backend.pipelines.lock().unwrap();
            assert_eq!(pipelines.epilogue_libraries.len(), 1);
            assert_eq!(pipelines.epilogue_pipelines.len(), 9);
            assert_eq!(pipelines.epilogue_counts.libraries, 1);
            assert_eq!(pipelines.epilogue_counts.pipelines, 9);
        }

        backend.replay(&graph, vec![1]).unwrap().wait().unwrap();
        backend.replay(&graph, vec![7]).unwrap().wait().unwrap();
        let pipelines = backend.pipelines.lock().unwrap();
        assert_eq!(pipelines.epilogue_pipelines.len(), 9);
        assert_eq!(pipelines.epilogue_counts.cache_hits, 10);
        assert_eq!(pipelines.epilogue_counts.fallbacks, 0);
    }

    #[test]
    fn metal_graph_refuses_epilogue_specializations_beyond_compile_quota() {
        let budget = crate::ProgramCompileBudget::new(8, Duration::from_hours(24)).unwrap();
        let backend = MetalBackend::with_program_compile_budget(budget).unwrap();
        assert!(matches!(
            backend.prepare_graph(dynamic_epilogue_variant_graph(&backend)),
            Err(BackendError::QuotaExceeded)
        ));
        let pipelines = backend.pipelines.lock().unwrap();
        assert_eq!(pipelines.epilogue_libraries.len(), 0);
        assert_eq!(pipelines.epilogue_pipelines.len(), 0);
    }

    #[test]
    fn metal_encode_time_epilogue_miss_never_compiles() {
        let backend = MetalBackend::new().unwrap();
        let epilogue = prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![Inst::Input(0)],
                outputs: vec![(0, 0)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, 2, vec![], DType::F32),
        )
        .unwrap();
        let mut pipelines = backend.pipelines.lock().unwrap();
        assert!(matches!(
            pipelines.lookup_prepared_epilogue(
                epilogue_codegen::FamilySourceTag::Dense,
                &epilogue,
                "gemv",
                &[(0, 0), (1, 0), (2, 0)],
            ),
            Err(BackendError::ExecutionFailed)
        ));
        assert_eq!(pipelines.epilogue_counts.fallbacks, 1);
        assert_eq!(pipelines.epilogue_libraries.len(), 0);
        assert_eq!(pipelines.epilogue_pipelines.len(), 0);

        let assertion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::Dense,
                &epilogue,
                "gemv",
                &[(0, 0), (1, 0), (2, 0)],
            )
        }));
        assert!(match assertion {
            Err(_) => cfg!(debug_assertions),
            Ok(Err(BackendError::ExecutionFailed)) => !cfg!(debug_assertions),
            _ => false,
        });
        assert_eq!(pipelines.epilogue_counts.fallbacks, 2);
        assert_eq!(pipelines.epilogue_libraries.len(), 0);
        assert_eq!(pipelines.epilogue_pipelines.len(), 0);
    }

    fn push_grouped_epilogue_requests(
        requests: &mut Vec<EpiloguePipelineRequest>,
        epilogue: &Arc<PreparedEpilogue>,
        constants: &[(u32, u32)],
    ) {
        for (implementation, block_rows) in [
            ("grouped_quantized_gemm", 16),
            ("grouped_quantized_gemm_small_m", 8),
        ] {
            let mut constants = constants.to_vec();
            constants.push((16, block_rows));
            requests.push(EpiloguePipelineRequest {
                family: epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue: Arc::clone(epilogue),
                implementation,
                constants,
            });
        }
    }

    #[test]
    fn metal_eager_epilogue_specializations_cover_every_boundary_arm() {
        let backend = MetalBackend::new().unwrap();
        let epilogue = prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![Inst::Input(0)],
                outputs: vec![(0, 0)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, 2, vec![], DType::F32),
        )
        .unwrap();
        let mut requests = Vec::new();
        let dense_constants = vec![(0, 0), (1, 0), (2, 0)];
        for implementation in ["gemv", "gemv_transposed"] {
            requests.push(EpiloguePipelineRequest {
                family: epilogue_codegen::FamilySourceTag::Dense,
                epilogue: Arc::clone(&epilogue),
                implementation,
                constants: dense_constants.clone(),
            });
        }
        for config in [GEMM_64_64_FLOAT, GEMM_64_64_HALF, GEMM_64_32, GEMM_32_64] {
            for specialization in dense_epilogue_specializations(config) {
                let mut constants = dense_constants.clone();
                constants.extend(specialization);
                requests.push(EpiloguePipelineRequest {
                    family: epilogue_codegen::FamilySourceTag::Dense,
                    epilogue: Arc::clone(&epilogue),
                    implementation: config.kernel,
                    constants,
                });
            }
        }
        let quant_constants = vec![(0, 0), (1, 1), (2, 0), (12, 8), (13, 64)];
        for implementation in [
            "quantized_gemv",
            "quantized_gemv_q8_fast",
            "quantized_gemm_small_m",
            "quantized_gemm_tiled_f32",
            "quantized_gemm_tiled_f16",
            "quantized_gemm_tiled_bf16",
        ] {
            for words in [
                quant_epilogue_specialization(7, 511),
                quant_epilogue_specialization(8, 512),
            ] {
                let mut constants = quant_constants.clone();
                constants.push((15, words));
                requests.push(EpiloguePipelineRequest {
                    family: epilogue_codegen::FamilySourceTag::Quant,
                    epilogue: Arc::clone(&epilogue),
                    implementation,
                    constants,
                });
            }
        }
        let gather_constants = vec![(0, 0), (1, 1), (2, 0), (12, 8), (13, 64)];
        for words in [
            gather_epilogue_specialization(7, 511),
            gather_epilogue_specialization(8, 512),
        ] {
            let mut constants = gather_constants.clone();
            constants.push((14, words));
            requests.push(EpiloguePipelineRequest {
                family: epilogue_codegen::FamilySourceTag::GatherQuant,
                epilogue: Arc::clone(&epilogue),
                implementation: "gather_quantized_gemv",
                constants,
            });
        }
        push_grouped_epilogue_requests(&mut requests, &epilogue, &gather_constants);

        let mut pipelines = backend.pipelines.lock().unwrap();
        pipelines.prepare_epilogue_pipelines(&requests).unwrap();
        let preparation_hits = pipelines.epilogue_counts.cache_hits;
        for request in &requests {
            pipelines
                .lookup_epilogue(
                    request.family,
                    &request.epilogue,
                    request.implementation,
                    &request.constants,
                )
                .unwrap();
        }
        assert_eq!(pipelines.epilogue_counts.fallbacks, 0);
        assert_eq!(
            pipelines.epilogue_counts.cache_hits,
            preparation_hits + u64::try_from(requests.len()).unwrap()
        );
    }

    #[test]
    fn tier_two_preencodes_a_pinned_static_dispatch() {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let right = backend.alloc(DType::F32, &[33, 7]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 7]).unwrap();
        let variant = backend
            .validate_variant_for_op(
                Op::Matmul,
                &[&left, &right],
                &[&output],
                "matmul.steel-64x64x16-2x2",
            )
            .unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&left, &right], &output)
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(variant)));
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        graph
            .record_validated(commands.into_dispatches().remove(0))
            .unwrap();

        let graph = backend.prepare_graph(graph).unwrap();
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        assert!(state.encoding.as_ref().unwrap().dispatches[0].is_some());
        assert_eq!(state.variant_rechecks.len(), 0);
        crate::variants::reset_replay_validation_count();
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        assert_eq!(crate::variants::replay_validation_count(), 0);
    }

    #[test]
    fn replay_refuses_a_resolved_variant_that_fails_concrete_constraints() {
        let backend = MetalBackend::new().unwrap();
        let valid_left = backend.alloc(DType::F32, &[1, 33]).unwrap();
        let right = backend.alloc(DType::F32, &[33, 7]).unwrap();
        let valid_output = backend.alloc(DType::F32, &[1, 7]).unwrap();
        let variant = backend
            .validate_variant_for_op(
                Op::Matmul,
                &[&valid_left, &right],
                &[&valid_output],
                "matmul.gemv",
            )
            .unwrap()
            .with_all_replay_constraints();
        let space = ParamSpace::new(std::iter::once(1..=2).collect()).unwrap();
        let left = backend.alloc(DType::F32, &[2, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[2, 7]).unwrap();
        let left = symbolic_prefix(&left, space.clone(), Affine::parameter(0, 0, 1));
        let output = symbolic_prefix(&output, space.clone(), Affine::parameter(0, 0, 1));
        let right = TemplateTensor::from(right);
        let mut template = GraphTemplate::new(space, GraphLimits::default());
        template
            .dispatch_variant_rule_many(
                Op::Matmul,
                &[&left, &right],
                &[&output],
                0,
                vec![(2, BackendDispatchData::new(variant))],
            )
            .unwrap();

        let graph = backend.prepare_graph(template).unwrap();
        assert_eq!(
            graph
                .backend_state::<PreparedMetalGraph>()
                .unwrap()
                .variant_rechecks,
            [0]
        );
        crate::variants::reset_replay_validation_count();
        backend.replay(&graph, vec![1]).unwrap().wait().unwrap();
        assert_eq!(crate::variants::replay_validation_count(), 1);
        assert!(matches!(
            backend.replay(&graph, vec![2]),
            Err(BackendError::InvalidInput)
        ));
        assert_eq!(crate::variants::replay_validation_count(), 2);
    }

    #[test]
    fn tier_two_replays_argmax_with_dynamic_scratch() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .dispatch(
                argmax_op(),
                &[&input.clone().into()],
                &output.clone().into(),
            )
            .unwrap();

        let graph = backend.prepare_graph(template).unwrap();
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        assert!(state.encoding.as_ref().unwrap().dispatches[0].is_none());
        let first = backend.replay(&graph, Vec::new()).unwrap();
        let second = backend.replay(&graph, Vec::new()).unwrap();
        second.wait().unwrap();
        first.wait().unwrap();
        let allocations = backend.debug_state().allocation_requests;
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        assert_eq!(backend.debug_state().allocation_requests, allocations);
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
        let add = Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Add, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
        .validate()
        .unwrap();
        let add = prepare_program(
            &backend,
            add,
            KernelSignature::new(1, vec![DType::F32, DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
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
            .dispatch_kernel(
                &add,
                &[&staged.into(), &one.into()],
                &[&feedback.clone().into()],
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
    fn replayed_decode_reuses_every_device_allocation_after_warmup() {
        let backend = MetalBackend::new().unwrap();
        let hidden = backend.alloc(DType::F32, &[1, 1024]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 1024]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .dispatch(Op::Copy, &[&hidden.into()], &output.into())
            .unwrap();
        let graph = backend.prepare_graph(template).unwrap();
        backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        let warm = backend.debug_state().allocation_requests;

        for _ in 0..128 {
            backend.replay(&graph, Vec::new()).unwrap().wait().unwrap();
        }

        assert_eq!(backend.debug_state().allocation_requests, warm);
    }

    fn exercise_graph_lifetime(backend: &MetalBackend, hidden: &Tensor, output: &Tensor) {
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .dispatch(Op::Copy, &[&hidden.clone().into()], &output.clone().into())
            .unwrap();
        let graph = backend.prepare_graph(template).unwrap();
        let mut submissions = VecDeque::new();
        for _ in 0..16 {
            submissions.push_back(backend.replay(&graph, Vec::new()).unwrap());
            if submissions.len() == MAX_IN_FLIGHT_SUBMISSIONS {
                submissions.pop_front().unwrap().wait().unwrap();
            }
        }
        for submission in submissions {
            submission.wait().unwrap();
        }
    }

    #[test]
    fn graph_replay_allocation_reaches_a_bounded_steady_state() {
        const FIXED_BACKEND_BYTES: u64 = 1024 * 1024;
        const WARMUP_ITERATIONS: usize = 16;
        const CHECK_ITERATIONS: usize = 64;

        let backend = MetalBackend::new().unwrap();
        let hidden = backend.alloc(DType::F32, &[1, 1024]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 1024]).unwrap();
        for _ in 0..WARMUP_ITERATIONS {
            exercise_graph_lifetime(&backend, &hidden, &output);
        }
        let settled = backend.memory_usage().device_bytes.unwrap();

        for _ in 0..CHECK_ITERATIONS {
            exercise_graph_lifetime(&backend, &hidden, &output);
            let memory = backend.memory_usage();
            let device = memory.device_bytes.unwrap();
            assert!(
                device <= memory.charged_bytes + FIXED_BACKEND_BYTES,
                "device allocation {device} exceeds {} accounted bytes plus fixed overhead",
                memory.charged_bytes
            );
            assert_eq!(device, settled);
        }
        assert_eq!(
            backend.debug_state().submission_slots,
            MAX_IN_FLIGHT_SUBMISSIONS
        );
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
    fn static_plan_pipelines_are_deduplicated_within_dynamic_boundaries() {
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
        assert!(!has_pipeline(1) && has_table(1));
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

        assert!(matches!(
            backend.replay(&graph, Vec::new()).unwrap().wait(),
            Err(BackendError::DeviceErrorFlag {
                dispatch: 0,
                operation: "embed",
                flag: 1,
                value: 99,
                ..
            })
        ));
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
        for result in [failed.wait(), same.wait(), shared.wait()] {
            assert!(matches!(
                result,
                Err(BackendError::DeviceErrorFlag {
                    dispatch: 0,
                    operation: "embed",
                    flag: 1,
                    value: 99,
                    ..
                })
            ));
        }
    }

    #[test]
    fn dropped_submissions_retain_released_buffers() {
        let backend = MetalBackend::new().unwrap();
        for _ in 0..100 {
            let input = backend.alloc(DType::F32, &[4097]).unwrap();
            let output = backend.alloc(DType::F32, &[4097]).unwrap();
            let mut commands = CommandList::new();
            commands.dispatch(Op::Copy, &[&input], &output).unwrap();
            let submission = backend.submit(commands).unwrap();
            drop(submission);
            backend.release(&input).unwrap();
            backend.release(&output).unwrap();
        }
        drop(backend);
    }

    #[test]
    fn timed_out_work_never_exposes_an_active_output() {
        let backend = MetalBackend::with_gpu_timeout(Duration::from_micros(1)).unwrap();
        let shape = [1024, 4097];
        let input = backend.alloc(DType::F32, &shape).unwrap();
        let output = backend.alloc(DType::F32, &shape).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();

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
    fn metal_copy_matches_cpu_for_casts_shapes_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let floats = [DType::F32, DType::F16, DType::BF16];
        for input_dtype in floats {
            for output_dtype in floats {
                for width in [1, 7, 33, 4096, 4097] {
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

        let mut objects = backend.in_flight.checkout(&backend, 4096).unwrap();
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
                    internal_profiles: Vec::new(),
                },
                None,
            )
            .unwrap();
        let residency = objects.make_resident(&command_buffer, &resources).unwrap();
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

    #[test]
    fn reusable_scratch_grows_after_release() {
        let backend = MetalBackend::new().unwrap();
        let mut scratch = ReusableScratch::default();
        let small = scratch.acquire(&backend, 33).unwrap();
        let small_len = small.length();
        drop(small);
        scratch.next = 0;
        let large = scratch.acquire(&backend, 4097).unwrap();
        assert!(small_len >= 33);
        assert!(large.length() >= 4097);
        assert_eq!(scratch.buffers.len(), 1);
    }

    #[test]
    fn reusable_scratch_trims_oversized_buffers() {
        let backend = MetalBackend::new().unwrap();
        let mut scratch = ReusableScratch::default();
        let small = scratch.acquire(&backend, 33).unwrap();
        let limit = small.length();
        drop(small);
        scratch.acquire(&backend, 4097).unwrap();
        assert!(scratch.trim_above(limit));
        assert_eq!(scratch.buffers.len(), 1);
        assert!(scratch.buffers[0].raw.length() >= 33);
        assert_eq!(scratch.next, 0);
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
    fn metal_dense_gemv_epilogues_match_decomposed_bias_and_residual() {
        for (variant, transposed, broadcast) in [
            ("matmul.gemv", false, true),
            ("matmul.gemv-transposed", true, false),
        ] {
            assert_dense_epilogue(variant, transposed, broadcast, 1, 33, 17);
        }
    }

    #[test]
    fn metal_steel_epilogues_cover_every_tile_and_edges() {
        for variant in [
            "matmul.steel-64x64x16-2x2",
            "matmul.steel-64x64x16-1x2",
            "matmul.steel-64x32x32-2x2",
            "matmul.steel-32x64x16-1x2",
        ] {
            assert_dense_epilogue(variant, false, true, 33, 33, 33);
        }
    }

    #[test]
    fn metal_epilogue_families_check_both_edge_columns_at_required_widths() {
        let backend = MetalBackend::new().unwrap();
        for columns in [1, 7, 33, 4097, 151_936] {
            assert_dense_epilogue("matmul.steel-64x32x32-2x2", false, true, 2, 64, columns);
            assert_quant_epilogue(
                &backend,
                "quant-matmul.small-m",
                DType::F32,
                4,
                2,
                64,
                columns,
                true,
            );
            assert_gather_quant_epilogue(
                &backend,
                "gather-quant-matmul.route-gemv",
                DType::F32,
                4,
                [2, 2, columns],
                true,
            );
        }
    }

    #[test]
    fn metal_quant_gemv_epilogues_cover_formats_dtypes_and_odd_columns() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in [4, 8] {
                assert_quant_epilogue(&backend, "quant-matmul.gemv", dtype, bits, 1, 256, 33, true);
            }
        }
        assert_quant_epilogue(
            &backend,
            "quant-matmul.gemv",
            DType::F32,
            4,
            1,
            256,
            4097,
            true,
        );
    }

    #[test]
    fn metal_quant_q8_fast_epilogues_cover_every_dtype() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for rows in [1, 2, 4, 8] {
                assert_quant_epilogue(
                    &backend,
                    "quant-matmul.q8-fast-gemv",
                    dtype,
                    8,
                    rows,
                    256,
                    33,
                    true,
                );
            }
        }
    }

    #[test]
    fn metal_quant_small_m_epilogues_cover_formats_and_noncontiguous_extras() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in [4, 8] {
                assert_quant_epilogue(
                    &backend,
                    "quant-matmul.small-m",
                    dtype,
                    bits,
                    7,
                    256,
                    33,
                    false,
                );
            }
        }
    }

    #[test]
    fn metal_quant_tiled_epilogues_cover_formats_dtypes_and_edge_tiles() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in [4, 8] {
                assert_quant_epilogue(
                    &backend,
                    "quant-matmul.tiled",
                    dtype,
                    bits,
                    33,
                    128,
                    33,
                    false,
                );
            }
        }
        assert_quant_epilogue(
            &backend,
            "quant-matmul.tiled",
            DType::BF16,
            8,
            16,
            1024,
            128,
            true,
        );
    }

    #[test]
    fn metal_gather_quant_route_epilogues_cover_formats_and_route_broadcasts() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in [4, 8] {
                assert_gather_quant_epilogue(
                    &backend,
                    "gather-quant-matmul.route-gemv",
                    dtype,
                    bits,
                    [7, 3, 33],
                    true,
                );
            }
        }
    }

    #[test]
    fn metal_gather_quant_grouped_epilogues_preserve_unsorted_coordinates() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in [4, 8] {
                assert_gather_quant_epilogue(
                    &backend,
                    "gather-quant-matmul.grouped",
                    dtype,
                    bits,
                    [8, 8, 33],
                    false,
                );
            }
        }
    }

    #[test]
    fn metal_paired_gather_epilogues_execute_route_and_grouped_variants() {
        let backend = MetalBackend::new().unwrap();
        for (variant, [rows, routes, columns]) in [
            ("gather-quant-matmul.route-gemv", [1, 3, 33]),
            ("gather-quant-matmul.grouped", [16, 8, 33]),
        ] {
            let inner = 64;
            let experts = 8;
            let input = backend.alloc(DType::F32, &[rows, inner]).unwrap();
            let packed = backend
                .alloc(DType::U32, &[experts, columns, inner / 8])
                .unwrap();
            let scales = backend
                .alloc(DType::F16, &[experts, columns, inner / 64])
                .unwrap();
            let biases = backend
                .alloc(DType::F16, &[experts, columns, inner / 64])
                .unwrap();
            let indices = backend.alloc(DType::U32, &[rows, routes]).unwrap();
            let output = backend.alloc(DType::F32, &[rows, routes, columns]).unwrap();
            backend
                .write(&input, &float_pattern(DType::F32, rows * inner))
                .unwrap();
            backend
                .write(
                    &indices,
                    &(0..rows * routes)
                        .flat_map(|index| (index % experts).to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let op = Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            };
            let inputs = [
                &input, &packed, &scales, &biases, &packed, &scales, &biases, &indices,
            ];
            let epilogue = prepare_epilogue(
                silu_mul_program(),
                EpilogueSignature::new(2, 3, vec![], DType::F32),
            )
            .unwrap();
            let selection = backend
                .validate_variant_for_op(op, &inputs, &[&output], variant)
                .unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch_epilogue(op, &inputs, &output, &epilogue, &[])
                .unwrap();
            assert!(commands.set_last_backend_data(BackendDispatchData::new(selection)));
            let submission = if variant == "gather-quant-matmul.grouped" {
                backend.submit_profiled(commands.clone()).unwrap()
            } else {
                backend.submit(commands.clone()).unwrap()
            };
            submission.wait().unwrap();
            if variant == "gather-quant-matmul.grouped" {
                let profile = submission.profile().unwrap();
                let internal = &profile.per_dispatch[0].internal;
                assert_eq!(internal.len(), 5);
                assert_eq!(internal[0].kernel, "count_expert_routes");
                assert_eq!(internal[4].kernel, "gathered_pair_from_projections");
                assert!(
                    internal
                        .iter()
                        .all(|kernel| kernel.gpu_time > Duration::ZERO)
                );

                let mut template = GraphTemplate::new(
                    ParamSpace::new(Vec::new()).unwrap(),
                    GraphLimits::default(),
                );
                template
                    .record_validated(commands.into_dispatches().remove(0))
                    .unwrap();
                let graph = backend.prepare_graph(template).unwrap();
                let replay = backend.replay_graph(&graph, Vec::new(), true).unwrap();
                replay.wait().unwrap();
                let replay_profile = replay.profile().unwrap();
                assert_eq!(replay_profile.per_dispatch[0].internal.len(), 5);
                assert!(
                    replay_profile.per_dispatch[0]
                        .internal
                        .iter()
                        .all(|kernel| kernel.gpu_time > Duration::ZERO)
                );
            }
        }
    }

    #[test]
    fn metal_weighted_gather_epilogue_executes_optimized_and_broad_schedules() {
        let backend = MetalBackend::new().unwrap();
        for routes in [5, 33] {
            let rows = 1;
            let experts = 8;
            let inner = 64;
            let columns = 7;
            let input = backend.alloc(DType::F32, &[rows, routes, inner]).unwrap();
            let packed = backend
                .alloc(DType::U32, &[experts, columns, inner / 8])
                .unwrap();
            let scales = backend
                .alloc(DType::F16, &[experts, columns, inner / 64])
                .unwrap();
            let biases = backend
                .alloc(DType::F16, &[experts, columns, inner / 64])
                .unwrap();
            let indices = backend.alloc(DType::U32, &[rows, routes]).unwrap();
            let weights = backend.alloc(DType::F32, &[rows, routes]).unwrap();
            let residual = backend.alloc(DType::F32, &[rows, columns]).unwrap();
            let output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
            backend
                .write(
                    &indices,
                    &(0..routes)
                        .flat_map(|index| (index % experts).to_ne_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let epilogue = prepare_epilogue(
                add_program(),
                EpilogueSignature::new(1, 2, vec![DType::F32], DType::F32),
            )
            .unwrap();
            let mut commands = CommandList::new();
            let op = Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            };
            let inputs = [&input, &packed, &scales, &biases, &indices, &weights];
            let variant = if routes <= 8 {
                "gather-quant-matmul.route-gemv"
            } else {
                "gather-quant-matmul.weighted-broad"
            };
            let selection = backend
                .validate_variant_for_op(op, &inputs, &[&output], variant)
                .unwrap();
            commands
                .dispatch_epilogue(op, &inputs, &output, &epilogue, &[&residual])
                .unwrap();
            assert!(commands.set_last_backend_data(BackendDispatchData::new(selection)));
            backend.submit(commands).unwrap().wait().unwrap();
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn metal_broad_weighted_gather_zeros_invalid_route_and_reports_error() {
        let backend = MetalBackend::new().unwrap();
        let rows = 1;
        let routes = 33;
        let experts = 8;
        let inner = 64;
        let columns = 7;
        let invalid_route = 17;
        let invalid_expert = 99_u32;
        let input = backend.alloc(DType::F32, &[rows, routes, inner]).unwrap();
        let packed = backend
            .alloc(DType::U32, &[experts, columns, inner / 8])
            .unwrap();
        let scales = backend
            .alloc(DType::F16, &[experts, columns, inner / 64])
            .unwrap();
        let biases = backend
            .alloc(DType::F16, &[experts, columns, inner / 64])
            .unwrap();
        let indices = backend.alloc(DType::U32, &[rows, routes]).unwrap();
        let weights = backend.alloc(DType::F32, &[rows, routes]).unwrap();
        let residual = backend.alloc(DType::F32, &[rows, columns]).unwrap();
        let output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
        backend
            .write(
                &input,
                &(0..routes * inner)
                    .flat_map(|_| 1.0_f32.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        backend
            .write(
                &biases,
                &(0..experts * columns * (inner / 64))
                    .flat_map(|_| 0x3c00_u16.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut selected = vec![0_u32; usize::try_from(routes).unwrap()];
        selected[invalid_route] = invalid_expert;
        backend
            .write(
                &indices,
                &selected
                    .into_iter()
                    .flat_map(u32::to_ne_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut route_weights = vec![0.0_f32; usize::try_from(routes).unwrap()];
        route_weights[invalid_route] = 1.0;
        backend
            .write(
                &weights,
                &route_weights
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        backend
            .write(&output, &vec![0xa5; usize::try_from(columns * 4).unwrap()])
            .unwrap();
        let epilogue = prepare_epilogue(
            add_program(),
            EpilogueSignature::new(1, 2, vec![DType::F32], DType::F32),
        )
        .unwrap();
        let mut commands = CommandList::new();
        let op = Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        };
        let inputs = [&input, &packed, &scales, &biases, &indices, &weights];
        let selection = backend
            .validate_variant_for_op(
                op,
                &inputs,
                &[&output],
                "gather-quant-matmul.weighted-broad",
            )
            .unwrap();
        commands
            .dispatch_epilogue(op, &inputs, &output, &epilogue, &[&residual])
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(selection)));
        let submission = backend.submit(commands).unwrap();
        assert!(matches!(
            submission.wait(),
            Err(BackendError::DeviceErrorFlag {
                operation: "gather-quant-matmul",
                flag: 1,
                value,
                ..
            }) if value == invalid_expert
        ));
        drop(submission);
        assert_eq!(
            backend.read(&output).unwrap(),
            vec![0; usize::try_from(columns * 4).unwrap()]
        );
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn assert_gather_quant_epilogue(
        backend: &MetalBackend,
        variant: &str,
        dtype: DType,
        bits: u8,
        [rows, routes, columns]: [u32; 3],
        broadcast: bool,
    ) {
        let inner = 128;
        let experts = 4;
        let group_size = 64;
        let packed_width = inner * u32::from(bits) / 32;
        let groups = inner / group_size;
        let parameter_dtype = if dtype == DType::F32 {
            DType::F16
        } else {
            dtype
        };
        let input = backend.alloc(dtype, &[rows, inner]).unwrap();
        let packed = backend
            .alloc(DType::U32, &[experts, columns, packed_width])
            .unwrap();
        let scales = backend
            .alloc(parameter_dtype, &[experts, columns, groups])
            .unwrap();
        let biases = backend
            .alloc(parameter_dtype, &[experts, columns, groups])
            .unwrap();
        backend
            .write(&input, &float_pattern(dtype, rows * inner))
            .unwrap();
        let packed_bytes = (0..experts * columns * packed_width)
            .flat_map(|index| index.wrapping_mul(0x0101_0101).to_ne_bytes())
            .collect::<Vec<_>>();
        backend.write(&packed, &packed_bytes).unwrap();
        let parameter_count = experts * columns * groups;
        backend
            .write(&scales, &float_pattern(parameter_dtype, parameter_count))
            .unwrap();
        backend
            .write(&biases, &float_pattern(parameter_dtype, parameter_count))
            .unwrap();
        let index_storage = backend.alloc(DType::U32, &[routes, rows]).unwrap();
        let indices = backend
            .view(&index_storage, ViewOp::Permute(vec![1, 0]))
            .unwrap();
        let index_bytes = (0..rows * routes)
            .flat_map(|index| (index % experts).to_ne_bytes())
            .collect::<Vec<_>>();
        backend.write(&index_storage, &index_bytes).unwrap();
        let extra_storage = if broadcast {
            backend.alloc(dtype, &[rows, routes, 1]).unwrap()
        } else {
            backend.alloc(dtype, &[routes, rows, columns]).unwrap()
        };
        let extra = if broadcast {
            backend
                .view(
                    &extra_storage,
                    ViewOp::Broadcast(vec![rows, routes, columns]),
                )
                .unwrap()
        } else {
            backend
                .view(&extra_storage, ViewOp::Permute(vec![1, 0, 2]))
                .unwrap()
        };
        let extra_count = if broadcast {
            rows * routes
        } else {
            rows * routes * columns
        };
        backend
            .write(&extra_storage, &float_pattern(dtype, extra_count))
            .unwrap();
        let output_shape = [rows, routes, columns];
        let fused_output = backend.alloc(dtype, &output_shape).unwrap();
        let decomposed_output = backend.alloc(dtype, &output_shape).unwrap();
        let accumulator = backend.alloc(dtype, &output_shape).unwrap();
        let program = multiply_program();
        let epilogue = prepare_epilogue(
            program.clone(),
            EpilogueSignature::new(1, 3, vec![dtype], dtype),
        )
        .unwrap();
        let map = prepare_program(
            backend,
            program,
            KernelSignature::new(3, vec![dtype, dtype], vec![dtype], 0),
        )
        .unwrap();
        let op = Op::GatherQuantMatmul { bits, group_size };
        let primitive_inputs = [&input, &packed, &scales, &biases, &indices];
        let selection = backend
            .validate_variant_for_op(op, &primitive_inputs, &[&fused_output], variant)
            .unwrap();
        let mut fused = CommandList::new();
        fused
            .dispatch_epilogue(op, &primitive_inputs, &fused_output, &epilogue, &[&extra])
            .unwrap();
        assert!(fused.set_last_backend_data(BackendDispatchData::new(selection)));
        let mut decomposed = CommandList::new();
        decomposed
            .dispatch(op, &primitive_inputs, &accumulator)
            .unwrap();
        assert!(decomposed.set_last_backend_data(BackendDispatchData::new(selection)));
        decomposed
            .dispatch_kernel(&map, &[&accumulator, &extra], &[&decomposed_output])
            .unwrap();
        assert_fused_decomposed(
            backend,
            fused,
            &fused_output,
            decomposed,
            &decomposed_output,
            DifferentialComparison::Quantized(dtype),
        )
        .unwrap();
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn assert_quant_epilogue(
        backend: &MetalBackend,
        variant: &str,
        dtype: DType,
        bits: u8,
        rows: u32,
        inner: u32,
        columns: u32,
        broadcast: bool,
    ) {
        let group_size = 64;
        let packed_width = inner * u32::from(bits) / 32;
        let groups = inner / group_size;
        let parameter_dtype = if dtype == DType::F32 {
            DType::F16
        } else {
            dtype
        };
        let input = backend.alloc(dtype, &[rows, inner]).unwrap();
        let packed = backend.alloc(DType::U32, &[columns, packed_width]).unwrap();
        let scales = backend.alloc(parameter_dtype, &[columns, groups]).unwrap();
        let biases = backend.alloc(parameter_dtype, &[columns, groups]).unwrap();
        backend
            .write(&input, &float_pattern(dtype, rows * inner))
            .unwrap();
        let packed_bytes = (0..columns * packed_width)
            .flat_map(|index| index.wrapping_mul(0x0101_0101).to_ne_bytes())
            .collect::<Vec<_>>();
        backend.write(&packed, &packed_bytes).unwrap();
        backend
            .write(&scales, &float_pattern(parameter_dtype, columns * groups))
            .unwrap();
        backend
            .write(&biases, &float_pattern(parameter_dtype, columns * groups))
            .unwrap();
        let extra_storage = if broadcast {
            backend.alloc(dtype, &[columns]).unwrap()
        } else {
            backend.alloc(dtype, &[columns, rows]).unwrap()
        };
        let extra = if broadcast {
            backend
                .view(&extra_storage, ViewOp::Broadcast(vec![rows, columns]))
                .unwrap()
        } else {
            backend
                .view(&extra_storage, ViewOp::Permute(vec![1, 0]))
                .unwrap()
        };
        backend
            .write(
                &extra_storage,
                &float_pattern(dtype, if broadcast { columns } else { rows * columns }),
            )
            .unwrap();
        let fused_output = backend.alloc(dtype, &[rows, columns]).unwrap();
        let decomposed_output = backend.alloc(dtype, &[rows, columns]).unwrap();
        let accumulator = backend.alloc(dtype, &[rows, columns]).unwrap();
        let program = multiply_program();
        let epilogue = prepare_epilogue(
            program.clone(),
            EpilogueSignature::new(1, 2, vec![dtype], dtype),
        )
        .unwrap();
        let map = prepare_program(
            backend,
            program,
            KernelSignature::new(2, vec![dtype, dtype], vec![dtype], 0),
        )
        .unwrap();
        let op = Op::QuantMatmul { bits, group_size };
        let primitive_inputs = [&input, &packed, &scales, &biases];
        let selection = backend
            .validate_variant_for_op(op, &primitive_inputs, &[&fused_output], variant)
            .unwrap();
        let mut fused = CommandList::new();
        fused
            .dispatch_epilogue(op, &primitive_inputs, &fused_output, &epilogue, &[&extra])
            .unwrap();
        assert!(fused.set_last_backend_data(BackendDispatchData::new(selection)));
        let mut decomposed = CommandList::new();
        decomposed
            .dispatch(op, &primitive_inputs, &accumulator)
            .unwrap();
        assert!(decomposed.set_last_backend_data(BackendDispatchData::new(selection)));
        decomposed
            .dispatch_kernel(&map, &[&accumulator, &extra], &[&decomposed_output])
            .unwrap();
        assert_fused_decomposed(
            backend,
            fused,
            &fused_output,
            decomposed,
            &decomposed_output,
            DifferentialComparison::Quantized(dtype),
        )
        .unwrap();
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn assert_dense_epilogue(
        variant: &str,
        transposed: bool,
        broadcast: bool,
        rows: u32,
        inner: u32,
        columns: u32,
    ) {
        let backend = MetalBackend::new().unwrap();
        let left = backend.alloc(DType::F32, &[rows, inner]).unwrap();
        let right_shape = if transposed {
            [columns, inner]
        } else {
            [inner, columns]
        };
        let right_storage = backend.alloc(DType::F32, &right_shape).unwrap();
        let right = if transposed {
            backend
                .view(&right_storage, ViewOp::Permute(vec![1, 0]))
                .unwrap()
        } else {
            right_storage.clone()
        };
        let extra_shape = if broadcast {
            vec![columns]
        } else {
            vec![rows, columns]
        };
        let extra_storage = backend.alloc(DType::F32, &extra_shape).unwrap();
        let extra = if broadcast {
            backend
                .view(&extra_storage, ViewOp::Broadcast(vec![rows, columns]))
                .unwrap()
        } else {
            extra_storage.clone()
        };
        let left_count = usize::try_from(rows * inner).unwrap();
        let right_count = usize::try_from(inner * columns).unwrap();
        let output_count = usize::try_from(rows * columns).unwrap();
        backend.write(&left, &f32_pattern(left_count, 19)).unwrap();
        backend
            .write(&right_storage, &f32_pattern(right_count, 23))
            .unwrap();
        let extra_count = if broadcast {
            usize::try_from(columns).unwrap()
        } else {
            output_count
        };
        backend
            .write(&extra_storage, &f32_pattern(extra_count, 11))
            .unwrap();
        let fused_output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
        let decomposed_output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
        let accumulator = backend.alloc(DType::F32, &[rows, columns]).unwrap();
        let program = add_program();
        let epilogue = prepare_epilogue(
            program.clone(),
            EpilogueSignature::new(1, 2, vec![DType::F32], DType::F32),
        )
        .unwrap();
        let map = prepare_program(
            &backend,
            program,
            KernelSignature::new(2, vec![DType::F32, DType::F32], vec![DType::F32], 0),
        )
        .unwrap();
        let selection = backend
            .validate_variant_for_op(Op::Matmul, &[&left, &right], &[&fused_output], variant)
            .unwrap();

        let mut fused = CommandList::new();
        fused
            .dispatch_epilogue(
                Op::Matmul,
                &[&left, &right],
                &fused_output,
                &epilogue,
                &[&extra],
            )
            .unwrap();
        assert!(fused.set_last_backend_data(BackendDispatchData::new(selection)));
        let mut template =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        template
            .record_validated(fused.last_dispatch().unwrap().clone())
            .unwrap();
        let graph = backend.prepare_graph(template).unwrap();
        {
            let pipelines = backend.pipelines.lock().unwrap();
            assert_eq!(pipelines.epilogue_libraries.len(), 1);
            assert_eq!(pipelines.epilogue_pipelines.len(), 1);
        }
        drop(graph);
        let fused_again = fused.clone();
        let mut decomposed = CommandList::new();
        decomposed
            .dispatch(Op::Matmul, &[&left, &right], &accumulator)
            .unwrap();
        assert!(decomposed.set_last_backend_data(BackendDispatchData::new(selection)));
        decomposed
            .dispatch_kernel(&map, &[&accumulator, &extra], &[&decomposed_output])
            .unwrap();
        assert_fused_decomposed(
            &backend,
            fused,
            &fused_output,
            decomposed,
            &decomposed_output,
            DifferentialComparison::Exact,
        )
        .unwrap();
        let profiled = backend.submit_profiled(fused_again).unwrap();
        profiled.wait().unwrap();
        let profile = profiled.profile().unwrap();
        assert_eq!(profile.epilogue_library_compilations, 1);
        assert_eq!(profile.epilogue_pipeline_specializations, 1);
        assert_eq!(profile.epilogue_cache_hits, 5);
        assert_eq!(profile.epilogue_compile_fallbacks, 0);
        let pipelines = backend.pipelines.lock().unwrap();
        assert_eq!(pipelines.epilogue_libraries.len(), 1);
        assert_eq!(pipelines.epilogue_pipelines.len(), 1);
        assert_eq!(pipelines.epilogue_counts, EpilogueCompileCounts::default());
    }

    fn f32_pattern(count: usize, modulus: u16) -> Vec<u8> {
        (0..count)
            .flat_map(|index| {
                let value = f32::from(u16::try_from(index % usize::from(modulus)).unwrap())
                    / f32::from(modulus)
                    - 0.5;
                value.to_ne_bytes()
            })
            .collect()
    }

    fn float_pattern(dtype: DType, count: u32) -> Vec<u8> {
        const F16_VALUES: [u16; 7] = [0xb600, 0xb400, 0xb000, 0, 0x3000, 0x3400, 0x3600];
        (0..count)
            .flat_map(|index| {
                let slot = usize::try_from(index % 7).unwrap();
                let value = (f32::from(u16::try_from(slot).unwrap()) - 3.0) / 8.0;
                match dtype {
                    DType::F32 => value.to_ne_bytes().to_vec(),
                    DType::F16 => F16_VALUES[slot].to_ne_bytes().to_vec(),
                    DType::BF16 => u16::try_from(value.to_bits() >> 16)
                        .unwrap()
                        .to_ne_bytes()
                        .to_vec(),
                    DType::I32 | DType::U32 => unreachable!(),
                }
            })
            .collect()
    }

    fn add_program() -> ValidatedProgram {
        Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Add, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
        .validate()
        .unwrap()
    }

    fn multiply_program() -> ValidatedProgram {
        Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Mul, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
        .validate()
        .unwrap()
    }

    fn silu_mul_program() -> ValidatedProgram {
        Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Unary(UnOp::Neg, 0),
                Inst::Unary(UnOp::Exp, 2),
                Inst::Const(1.0),
                Inst::Binary(BinOp::Add, 4, 3),
                Inst::Binary(BinOp::Div, 0, 5),
                Inst::Binary(BinOp::Mul, 6, 1),
            ],
            outputs: vec![(0, 7)],
        }
        .validate()
        .unwrap()
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

    struct SdpaOverride(Option<SdpaKernel>, Option<u32>);

    impl SdpaOverride {
        fn set(kernel: SdpaKernel) -> Self {
            let previous = FORCED_SDPA_KERNEL.with(|forced| forced.replace(Some(kernel)));
            let blocks = FORCED_VECTOR_BLOCKS.with(|forced| forced.replace(None));
            Self(previous, blocks)
        }

        fn vector_two_pass(blocks: u32) -> Self {
            let previous =
                FORCED_SDPA_KERNEL.with(|forced| forced.replace(Some(SdpaKernel::Vector)));
            let blocks = FORCED_VECTOR_BLOCKS.with(|forced| forced.replace(Some(blocks)));
            Self(previous, blocks)
        }

        fn vector_gqa_two_pass(blocks: u32) -> Self {
            let previous =
                FORCED_SDPA_KERNEL.with(|forced| forced.replace(Some(SdpaKernel::VectorGqa)));
            let blocks = FORCED_VECTOR_BLOCKS.with(|forced| forced.replace(Some(blocks)));
            Self(previous, blocks)
        }
    }

    impl Drop for SdpaOverride {
        fn drop(&mut self) {
            FORCED_SDPA_KERNEL.with(|forced| forced.set(self.0));
            FORCED_VECTOR_BLOCKS.with(|forced| forced.set(self.1));
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

    fn assert_mixed_sdpa(
        kernel: SdpaKernel,
        blocks: Option<u32>,
        query_length: u32,
        key_length: u32,
        cache_dtype: DType,
    ) {
        let _override =
            blocks.map_or_else(|| SdpaOverride::set(kernel), SdpaOverride::vector_two_pass);
        assert_backends_agree_at_tolerance(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: key_length - query_length,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[16, query_length, 128]),
                TensorSpec::contiguous(cache_dtype, &[8, key_length, 128]),
                TensorSpec::contiguous(cache_dtype, &[8, key_length, 128]),
            ],
            &TensorSpec::contiguous(DType::F32, &[16, query_length, 128]),
            cache_dtype,
        )
        .unwrap();
    }

    fn assert_half_steel_profile_case(
        backend: &MetalBackend,
        cache_dtype: DType,
        query_length: u32,
        key_length: u32,
        query_heads: u32,
        kv_heads: u32,
        width: u32,
    ) {
        let cache_slice = [
            Slice::new(0, kv_heads, 1).unwrap(),
            Slice::new(3, key_length, 1).unwrap(),
            Slice::new(0, width, 1).unwrap(),
        ];
        let _override = SdpaOverride::set(SdpaKernel::SteelHalf);
        assert_backends_agree_at_tolerance(
            &CpuBackend::new(),
            backend,
            Op::Sdpa {
                scale: f32::from(u16::try_from(width).unwrap()).sqrt().recip(),
                causal: true,
                q_start: key_length - query_length,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[query_heads, query_length, width]),
                TensorSpec::sliced(
                    cache_dtype,
                    &[kv_heads, key_length + 3, width],
                    &cache_slice,
                ),
                TensorSpec::sliced(
                    cache_dtype,
                    &[kv_heads, key_length + 3, width],
                    &cache_slice,
                ),
            ],
            &TensorSpec::contiguous(DType::F32, &[query_heads, query_length, width]),
            cache_dtype,
        )
        .unwrap();
    }

    fn assert_gqa_profile_case(
        backend: &MetalBackend,
        cache_dtype: DType,
        key_length: u32,
        blocks: u32,
    ) {
        let cache_slice = [
            Slice::new(0, 4, 1).unwrap(),
            Slice::new(3, key_length, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        let _override = SdpaOverride::vector_gqa_two_pass(blocks);
        assert_backends_agree_at_tolerance(
            &CpuBackend::new(),
            backend,
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: key_length - 1,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[32, 1, 128]),
                TensorSpec::sliced(cache_dtype, &[4, key_length + 3, 128], &cache_slice),
                TensorSpec::sliced(cache_dtype, &[4, key_length + 3, 128], &cache_slice),
            ],
            &TensorSpec::contiguous(DType::F32, &[32, 1, 128]),
            cache_dtype,
        )
        .unwrap();
    }

    fn patterned_bf16(count: usize, modulus: u16) -> Vec<u8> {
        (0..count)
            .flat_map(|index| {
                let value = f32::from(u16::try_from(index % usize::from(modulus)).unwrap())
                    / f32::from(modulus)
                    - 0.5;
                u16::try_from(value.to_bits() >> 16).unwrap().to_le_bytes()
            })
            .collect()
    }

    fn run_sdpa_kernel(
        backend: &MetalBackend,
        kernel: SdpaKernel,
        op: Op,
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Vec<u8> {
        let mut commands = CommandList::new();
        commands.dispatch(op, inputs, output).unwrap();
        let submission = {
            let _override = SdpaOverride::set(kernel);
            backend.submit(commands).unwrap()
        };
        submission.wait().unwrap();
        backend.read(output).unwrap()
    }

    #[test]
    fn sdpa_matmul_variants_are_explicit() {
        assert_eq!(
            sdpa_matmul_variant(1, true),
            MetalVariant::MatmulGemvTransposed
        );
        assert_eq!(sdpa_matmul_variant(1, false), MetalVariant::MatmulGemv);
        assert_eq!(sdpa_matmul_variant(2, true), MetalVariant::MatmulPortable);
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
    fn decomposed_attention_refuses_scratch_over_the_guest_quota() {
        let backend = MetalBackend::new().unwrap();
        let query = backend.alloc(DType::F32, &[16, 7, 128]).unwrap();
        let key = backend.alloc(DType::F32, &[8, 4097, 128]).unwrap();
        let value = backend.alloc(DType::F32, &[8, 4097, 128]).unwrap();
        let output = backend.alloc(DType::F32, &[16, 7, 128]).unwrap();
        let resident = backend.memory_usage().charged_bytes;
        backend.set_memory_limit(resident.checked_add(1 << 20).unwrap());
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 128.0_f32.sqrt().recip(),
                    causal: true,
                    q_start: 4090,
                },
                &[&query, &key, &value],
                &output,
            )
            .unwrap();
        let _override = SdpaOverride::set(SdpaKernel::Decomposed);

        assert!(matches!(
            backend.submit(commands),
            Err(BackendError::QuotaExceeded)
        ));
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
    fn metal_attention_reads_strided_grouped_cache_views() {
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
    fn metal_sdpa_offsets_narrowed_operands() {
        let query_slice = [
            Slice::new(0, 16, 1).unwrap(),
            Slice::new(2, 7, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        let cache_slice = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(3, 7, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        let output_slice = [
            Slice::new(0, 16, 1).unwrap(),
            Slice::new(1, 7, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        assert_sdpa_with(
            SdpaKernel::Decomposed,
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: 0,
            },
            TensorSpec::sliced(DType::F32, &[16, 11, 128], &query_slice),
            TensorSpec::sliced(DType::F32, &[8, 13, 128], &cache_slice),
            TensorSpec::sliced(DType::F32, &[8, 13, 128], &cache_slice),
            &TensorSpec::sliced(DType::F32, &[16, 9, 128], &output_slice),
        );
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
    fn metal_vector_attention_falls_back_for_permuted_f32_inputs() {
        let _override = SdpaOverride::set(SdpaKernel::Vector);
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: 36,
            },
            &[
                TensorSpec::permuted(DType::F32, &[128, 1, 16], &[2, 1, 0]),
                TensorSpec::permuted(DType::F32, &[128, 37, 8], &[2, 1, 0]),
                TensorSpec::permuted(DType::F32, &[128, 37, 8], &[2, 1, 0]),
            ],
            &TensorSpec::contiguous(DType::F32, &[16, 1, 128]),
        )
        .unwrap();
    }

    #[test]
    fn metal_vector_attention_matches_two_pass_decode() {
        let _override = SdpaOverride::vector_two_pass(64);
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
    fn metal_gqa_vector_attention_matches_profile_lengths() {
        let _override = SdpaOverride::vector_gqa_two_pass(128);
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for key_length in [1, 7, 33, 4_097, 16_384] {
            let cache_slice = [
                Slice::new(0, 4, 1).unwrap(),
                Slice::new(3, key_length, 1).unwrap(),
                Slice::new(0, 128, 1).unwrap(),
            ];
            assert_backends_agree_at_tolerance(
                &reference,
                &candidate,
                Op::Sdpa {
                    scale: 128.0_f32.sqrt().recip(),
                    causal: true,
                    q_start: key_length - 1,
                },
                &[
                    TensorSpec::contiguous(DType::F32, &[32, 1, 128]),
                    TensorSpec::sliced(DType::F16, &[4, key_length + 3, 128], &cache_slice),
                    TensorSpec::sliced(DType::F16, &[4, key_length + 3, 128], &cache_slice),
                ],
                &TensorSpec::contiguous(DType::F32, &[32, 1, 128]),
                DType::F16,
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_gqa_vector_attention_accepts_each_cache_dtype() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            assert_gqa_profile_case(&backend, dtype, 1024, 128);
        }
    }

    #[test]
    fn metal_attention_paths_accept_reduced_precision_cache() {
        for dtype in [DType::F16, DType::BF16] {
            assert_mixed_sdpa(SdpaKernel::Decomposed, None, 1, 1, dtype);
            assert_mixed_sdpa(SdpaKernel::Decomposed, None, 7, 33, dtype);
            assert_mixed_sdpa(SdpaKernel::Steel, None, 7, 33, dtype);
            assert_mixed_sdpa(SdpaKernel::Vector, None, 1, 33, dtype);
            assert_mixed_sdpa(SdpaKernel::Vector, Some(64), 1, 1024, dtype);
        }
    }

    #[test]
    fn metal_half_steel_attention_matches_cpu_at_profile_tolerance() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F16, DType::BF16] {
            for (query_length, key_length) in [(7, 7), (33, 33), (7, 4_097), (2, 16_384)] {
                assert_half_steel_profile_case(
                    &backend,
                    dtype,
                    query_length,
                    key_length,
                    32,
                    4,
                    128,
                );
            }
        }
    }

    #[test]
    fn metal_half_steel_width_64_matches_cpu_at_profile_tolerance() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F16, DType::BF16] {
            for key_length in [1, 7, 33, 4_097] {
                let query_length = key_length.min(2);
                assert_half_steel_profile_case(
                    &backend,
                    dtype,
                    query_length,
                    key_length,
                    16,
                    8,
                    64,
                );
            }
        }
    }

    #[test]
    fn metal_half_steel_masks_unaligned_long_key_boundary() {
        let backend = MetalBackend::new().unwrap();
        for dtype in [DType::F16, DType::BF16] {
            assert_half_steel_profile_case(&backend, dtype, 1, 32_759, 32, 4, 128);
        }
    }

    #[test]
    fn metal_gqa_vector_attention_reuses_grown_scratch() {
        let backend = MetalBackend::new().unwrap();
        assert_gqa_profile_case(&backend, DType::F16, 600, 32);
        assert_gqa_profile_case(&backend, DType::F16, 9_000, 256);
    }

    #[test]
    fn metal_steel_attention_falls_back_over_mocked_threadgroup_limit() {
        let backend = MetalBackend::new().unwrap();
        backend.set_pipeline_threadgroup_memory_limit_for_test(1);
        assert_half_steel_profile_case(&backend, DType::F16, 2, 7, 32, 4, 128);
    }

    #[test]
    fn metal_gqa_vector_attention_falls_back_over_mocked_threadgroup_limit() {
        let backend = MetalBackend::new().unwrap();
        backend.set_pipeline_threadgroup_memory_limit_for_test(1);
        assert_gqa_profile_case(&backend, DType::F16, 33, 128);
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
    fn metal_steel_attention_matches_decomposed_at_long_offsets() {
        const WIDTH: u32 = 128;
        const QUERY_HEADS: u32 = 16;
        const KV_HEADS: u32 = 8;
        const MAX_KEY_LENGTH: u32 = 32_767;

        let backend = MetalBackend::new().unwrap();
        let key_cache = backend
            .alloc(DType::BF16, &[KV_HEADS, MAX_KEY_LENGTH, WIDTH])
            .unwrap();
        let value_cache = backend
            .alloc(DType::BF16, &[KV_HEADS, MAX_KEY_LENGTH, WIDTH])
            .unwrap();
        let cache_elements = usize::try_from(KV_HEADS * MAX_KEY_LENGTH * WIDTH).unwrap();
        backend
            .write(&key_cache, &patterned_bf16(cache_elements, 31))
            .unwrap();
        backend
            .write(&value_cache, &patterned_bf16(cache_elements, 37))
            .unwrap();

        for (offset, chunk) in [(0, 1), (511, 7), (4_097, 33), (32_255, 512)] {
            let key_length = offset + chunk;
            let cache_slice = ViewOp::Slice(vec![
                Slice::new(0, KV_HEADS, 1).unwrap(),
                Slice::new(0, key_length, 1).unwrap(),
                Slice::new(0, WIDTH, 1).unwrap(),
            ]);
            let key = backend.view(&key_cache, cache_slice.clone()).unwrap();
            let value = backend.view(&value_cache, cache_slice).unwrap();
            let query = backend
                .alloc(DType::BF16, &[QUERY_HEADS, chunk, WIDTH])
                .unwrap();
            let query_elements = usize::try_from(QUERY_HEADS * chunk * WIDTH).unwrap();
            backend
                .write(&query, &patterned_bf16(query_elements, 29))
                .unwrap();
            let steel = backend
                .alloc(DType::BF16, &[QUERY_HEADS, chunk, WIDTH])
                .unwrap();
            let decomposed = backend
                .alloc(DType::BF16, &[QUERY_HEADS, chunk, WIDTH])
                .unwrap();
            let op = Op::Sdpa {
                scale: 128.0_f32.sqrt().recip(),
                causal: true,
                q_start: offset,
            };

            let steel_bytes = run_sdpa_kernel(
                &backend,
                SdpaKernel::Steel,
                op,
                &[&query, &key, &value],
                &steel,
            );
            let decomposed_bytes = run_sdpa_kernel(
                &backend,
                SdpaKernel::Decomposed,
                op,
                &[&query, &key, &value],
                &decomposed,
            );
            assert_outputs_agree(DType::BF16, &decomposed_bytes, &steel_bytes).unwrap();
            backend.release(&query).unwrap();
            backend.release(&steel).unwrap();
            backend.release(&decomposed).unwrap();
        }
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
    fn metal_gemm_matches_transformer_projection_shapes() {
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
    fn metal_gemv_matches_transformer_decode_shapes() {
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
    fn metal_matmul_offsets_narrowed_operands() {
        assert_matmul(
            TensorSpec::sliced(
                DType::F32,
                &[9, 33],
                &[Slice::new(1, 7, 1).unwrap(), Slice::new(0, 33, 1).unwrap()],
            ),
            TensorSpec::sliced_permuted(
                DType::F32,
                &[19, 33],
                &[Slice::new(2, 17, 1).unwrap(), Slice::new(0, 33, 1).unwrap()],
                &[1, 0],
            ),
            &TensorSpec::sliced(
                DType::F32,
                &[9, 17],
                &[Slice::new(1, 7, 1).unwrap(), Slice::new(0, 17, 1).unwrap()],
            ),
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
    fn metal_argmax_matches_cpu_for_dtypes_rows_and_views() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 2047, 2048, 2049, 4097, 151_936] {
                assert_backends_agree(
                    &reference,
                    &candidate,
                    argmax_op(),
                    &[TensorSpec::contiguous(dtype, &[2, width])],
                    &TensorSpec::contiguous(DType::U32, &[2]),
                )
                .unwrap();
            }
            assert_backends_agree(
                &reference,
                &candidate,
                argmax_op(),
                &[TensorSpec::permuted(dtype, &[33, 7], &[1, 0])],
                &TensorSpec::contiguous(DType::U32, &[7]),
            )
            .unwrap();
        }
        assert_backends_agree(
            &reference,
            &candidate,
            Op::Argmax { chunk_width: 33 },
            &[TensorSpec::contiguous(DType::F32, &[2, 4_097])],
            &TensorSpec::contiguous(DType::U32, &[2]),
        )
        .unwrap();
    }

    #[test]
    fn metal_argmax_reports_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[1, 151_936]).unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mut timings = Vec::with_capacity(20);
        for iteration in 0..23 {
            let mut commands = CommandList::new();
            commands.dispatch(argmax_op(), &[&input], &output).unwrap();
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
                argmax_op(),
                &[TensorSpec::initialized(dtype, &[1, 7], bytes)],
                &TensorSpec::contiguous(DType::U32, &[1]),
            )
            .unwrap();
        }
    }

    #[test]
    fn metal_greedy_sample_matches_argmax_edges_and_large_vocabulary_width() {
        let params = sample_params(0.0, 0, 1.0, 7);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for width in [1, 7, 33, 4097, 151_936] {
                assert_backends_agree(
                    &CpuBackend::new(),
                    &MetalBackend::new().unwrap(),
                    sample_op(33),
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
            sample_op(33),
            &[
                TensorSpec::initialized(DType::F32, &[1, 7], edges),
                TensorSpec::initialized(DType::U32, &[5], params),
            ],
            &TensorSpec::contiguous(DType::U32, &[1]),
        )
        .unwrap();
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Sample {
                position: 33,
                chunk_width: 1_024,
                fast_top_k: 10,
                fast_max_chunks: 100,
                rejection_rounds: 1,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[1, 4_097]),
                TensorSpec::initialized(DType::U32, &[5], sample_params(0.0, 0, 1.0, 7)),
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
    fn metal_forced_sampling_fallback_preserves_distribution() {
        let _rounds = SampleRoundsOverride::set(0);
        let backend = MetalBackend::new().unwrap();
        let logits = [2.0_f32, 1.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let trials = 4_096_u32;
        let mut counts = [0_u32; 3];
        for position in 0..trials {
            let token = run_sample(&backend, &logits, 3, 1.0, 2, 1.0, 95, position).unwrap();
            counts[usize::try_from(token).unwrap()] += 1;
        }
        let expected = [0.731_058_6_f32, 0.268_941_4];
        let statistic = counts[..2]
            .iter()
            .zip(expected)
            .map(|(&count, probability)| {
                let expected_count = probability * 4096.0;
                let difference = f32::from(u16::try_from(count).unwrap()) - expected_count;
                difference * difference / expected_count
            })
            .sum::<f32>();
        assert!(statistic < 16.3, "chi-squared statistic was {statistic}");
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn metal_and_cpu_sample_from_same_large_tied_allowed_sets() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for width in [4_097_u32, 151_936] {
            let values = (0..usize::try_from(width).unwrap())
                .map(|index| [3.0_f32, 1.0, 3.0, 0.0, 2.0][index % 5])
                .collect::<Vec<_>>();
            let logits = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            for (top_k, top_p) in [(513, 1.0), (2_049, 1.0), (0, 0.9)] {
                let allowed = allowed_sample_indices(&values, 0.7, top_k, top_p);
                for backend_token in [
                    run_sample(
                        &reference,
                        &logits,
                        width,
                        0.7,
                        top_k,
                        top_p,
                        0xfeed_beef_dead_cafe,
                        57,
                    )
                    .unwrap(),
                    run_sample(
                        &candidate,
                        &logits,
                        width,
                        0.7,
                        top_k,
                        top_p,
                        0xfeed_beef_dead_cafe,
                        57,
                    )
                    .unwrap(),
                ] {
                    assert!(allowed.contains(&backend_token));
                }
            }
        }
    }

    #[test]
    fn metal_sample_is_deterministic_and_reports_gpu_phases() {
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
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        for (name, temperature, top_k, top_p) in [
            ("greedy", 0.0, 0, 1.0),
            ("temperature", 0.7, 0, 1.0),
            ("Qwen defaults", 0.6, 20, 0.95),
            ("top-k 20", 0.7, 20, 1.0),
            ("top-k 5000", 0.7, 5000, 1.0),
            ("top-k 1024", 0.7, 1024, 1.0),
            ("top-p 0.9", 0.7, 0, 0.9),
        ] {
            let params = backend.alloc(DType::U32, &[5]).unwrap();
            backend
                .write(&params, &sample_params(temperature, top_k, top_p, 91))
                .unwrap();
            let mut timings = Vec::with_capacity(20);
            for iteration in 0..23 {
                let mut commands = CommandList::new();
                commands
                    .dispatch(sample_op(511), &[&input, &params], &output)
                    .unwrap();
                let submission = backend.submit_profiled(commands).unwrap();
                submission.wait().unwrap();
                if iteration >= 3 {
                    timings.push(submission.profile().unwrap().per_dispatch[0].gpu_time);
                }
            }
            timings.sort_unstable();
            let median = timings[timings.len() / 2];
            eprintln!("sample width 151936 {name}: median GPU {median:?}");
            assert!(median > Duration::ZERO);
        }
    }

    #[test]
    fn metal_sample_refuses_invalid_live_parameters() {
        let backend = MetalBackend::new().unwrap();
        let logits = vec![0_u8; 7 * 4];
        assert!(matches!(
            run_sample(&backend, &logits, 7, f32::NAN, 0, 1.0, 0, 0),
            Err(BackendError::DeviceErrorFlag {
                dispatch: 0,
                operation: "sample",
                flag: 2,
                value: 0,
                ..
            })
        ));
    }

    #[test]
    fn metal_masked_sample_matches_cpu_for_conditioned_sets() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let values = (0_u16..33)
            .map(|index| f32::from(index) / 7.0)
            .collect::<Vec<_>>();
        let logits = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        for allow in [[0x8000_0011, 1], [u32::MAX, 1]] {
            let allow_bytes = allow
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            for (temperature, top_k, top_p) in
                [(0.0, 0, 1.0), (0.7, 3, 1.0), (0.7, 0, 0.8), (0.7, 3, 0.9)]
            {
                for position in [0, 1, 33, 4097] {
                    assert_backends_agree(
                        &reference,
                        &candidate,
                        sample_masked_op(position),
                        &[
                            TensorSpec::initialized(DType::F32, &[1, 33], logits.clone()),
                            TensorSpec::initialized(
                                DType::U32,
                                &[5],
                                sample_params(temperature, top_k, top_p, 0xfeed_beef),
                            ),
                            TensorSpec::initialized(DType::U32, &[2], allow_bytes.clone()),
                        ],
                        &TensorSpec::contiguous(DType::U32, &[1]),
                    )
                    .unwrap();
                }
            }
        }

        let mut allow = vec![u32::MAX; 129];
        allow[128] = 1;
        assert_backends_agree(
            &reference,
            &candidate,
            Op::SampleMasked {
                position: 33,
                chunk_width: 1_024,
                fast_top_k: 10,
                fast_max_chunks: 100,
                rejection_rounds: 1,
            },
            &[
                TensorSpec::contiguous(DType::F32, &[1, 4_097]),
                TensorSpec::initialized(DType::U32, &[5], sample_params(0.7, 10, 0.9, 0xfeed_beef)),
                TensorSpec::initialized(
                    DType::U32,
                    &[129],
                    allow.into_iter().flat_map(u32::to_le_bytes).collect(),
                ),
            ],
            &TensorSpec::contiguous(DType::U32, &[1]),
        )
        .unwrap();
    }

    #[test]
    fn metal_speculate_then_correct_matches_cpu() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let logits = [0.3_f32, -0.2, 1.1, 0.7, -0.5, 0.2, 0.9, -0.8]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let allow = [(1 << 1) | (1 << 4) | (1 << 6)];
        let seed = 0xfeed_beef_dead_cafe;
        let mut kept = false;
        let mut corrected = false;
        for position in 0..128 {
            let expected =
                run_speculate_then_correct(&reference, &logits, &allow, seed, position).unwrap();
            let actual =
                run_speculate_then_correct(&candidate, &logits, &allow, seed, position).unwrap();
            assert_eq!(actual, expected);
            kept |= !actual.1;
            corrected |= actual.1;
        }
        assert!(kept && corrected);
    }

    #[test]
    fn metal_masked_sample_handles_qwen_vocabulary_and_refuses_bad_masks() {
        let backend = MetalBackend::new().unwrap();
        let logits = (0_u32..151_936)
            .flat_map(|index| f32::from(u16::try_from(index % 97).unwrap()).to_le_bytes())
            .collect::<Vec<_>>();
        let mut allow = vec![0_u32; usize::try_from(151_936_u32.div_ceil(32)).unwrap()];
        allow[0] |= 1 << 17;
        allow[151_935 / 32] |= 1 << (151_935 % 32);
        assert_eq!(
            run_masked_sample(&backend, &logits, 151_936, &allow, 0.0, 0, 1.0, 7, 33,).unwrap(),
            151_935
        );

        let short_logits = vec![0_u8; 33 * 4];
        for bad in [[0_u32, 0], [1, 2]] {
            assert!(matches!(
                run_masked_sample(&backend, &short_logits, 33, &bad, 0.0, 0, 1.0, 7, 0,),
                Err(BackendError::DeviceErrorFlag {
                    operation: "sample-masked",
                    flag: 2,
                    ..
                })
            ));
        }
    }

    #[test]
    fn metal_parallel_sample_matches_cpu_at_model_and_edge_widths() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        for width in [1_u32, 7, 33, 4097, 151_936] {
            let (dtype, logits) = if width == 1 {
                (DType::F16, vec![0_u8; 2])
            } else {
                (
                    DType::F32,
                    (0..width)
                        .flat_map(|index| {
                            (f32::from(u16::try_from(index % 31).unwrap()) / 8.0
                                - f32::from(u16::try_from(index % 7).unwrap()))
                            .to_le_bytes()
                        })
                        .collect(),
                )
            };
            for (temperature, top_k, top_p) in [
                (0.0, 0, 1.0),
                (0.7, 0, 1.0),
                (0.7, width.min(7), 1.0),
                (0.7, 0, 0.9),
            ] {
                let expected = run_typed_sample(
                    &reference,
                    dtype,
                    &logits,
                    width,
                    temperature,
                    top_k,
                    top_p,
                    0xfeed_beef_dead_cafe,
                    33,
                    |_| Ok(()),
                )
                .unwrap();
                let actual = run_metal_sample(
                    &candidate,
                    dtype,
                    &logits,
                    width,
                    temperature,
                    top_k,
                    top_p,
                    0xfeed_beef_dead_cafe,
                    33,
                    "sample.parallel",
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "parallel sample failed for width={width}, top_k={top_k}, top_p={top_p}: {error:?}"
                    )
                });
                assert_eq!(
                    actual, expected,
                    "width={width}, top_k={top_k}, top_p={top_p}"
                );
            }
        }
    }

    #[test]
    fn metal_sampling_variants_match_cpu_for_non_finite_logits() {
        let reference = CpuBackend::new();
        let candidate = MetalBackend::new().unwrap();
        let mut values = (0_u16..33)
            .map(|index| f32::from(index) / 8.0 - 2.0)
            .collect::<Vec<_>>();
        values[3] = f32::NEG_INFINITY;
        values[7] = f32::INFINITY;
        values[11] = f32::NAN;
        values[17] = f32::from_bits(0xffc0_0001);
        let logits = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let allow = [0x000a_088a, 1];
        for (temperature, top_k, top_p) in [(0.0, 0, 1.0), (0.6, 20, 0.95), (0.7, 0, 1.0)] {
            let expected = run_sample(
                &reference,
                &logits,
                33,
                temperature,
                top_k,
                top_p,
                0xfeed_beef_dead_cafe,
                33,
            )
            .unwrap();
            for variant in ["sample.portable", "sample.parallel"] {
                let actual = run_metal_sample(
                    &candidate,
                    DType::F32,
                    &logits,
                    33,
                    temperature,
                    top_k,
                    top_p,
                    0xfeed_beef_dead_cafe,
                    33,
                    variant,
                )
                .unwrap();
                assert_eq!(actual, expected, "variant={variant}");
            }

            let expected = run_masked_sample(
                &reference,
                &logits,
                33,
                &allow,
                temperature,
                top_k,
                top_p,
                0xfeed_beef_dead_cafe,
                33,
            )
            .unwrap();
            let actual = run_masked_sample(
                &candidate,
                &logits,
                33,
                &allow,
                temperature,
                top_k,
                top_p,
                0xfeed_beef_dead_cafe,
                33,
            )
            .unwrap();
            assert_eq!(actual, expected, "masked sample");
        }
    }

    #[test]
    fn metal_parallel_and_masked_sample_track_ordinary_sample_gpu_time() {
        let backend = MetalBackend::new().unwrap();
        let width = 151_936_u32;
        let logits = backend.alloc(DType::F32, &[1, width]).unwrap();
        let values = (0..width)
            .flat_map(|index| (f32::from(u16::try_from(index % 97).unwrap()) / 32.0).to_le_bytes())
            .collect::<Vec<_>>();
        backend.write(&logits, &values).unwrap();
        let params = backend.alloc(DType::U32, &[5]).unwrap();
        backend
            .write(&params, &sample_params(0.6, 20, 0.95, 91))
            .unwrap();
        let output = backend.alloc(DType::U32, &[1]).unwrap();
        let mask_words = usize::try_from(width.div_ceil(32)).unwrap();
        let dense = backend.alloc(DType::U32, &[width.div_ceil(32)]).unwrap();
        backend
            .write(&dense, &vec![0xff_u8; mask_words * 4])
            .unwrap();
        let sparse = backend.alloc(DType::U32, &[width.div_ceil(32)]).unwrap();
        let mut sparse_words = vec![0_u32; mask_words];
        for token in (0..width).step_by(997) {
            sparse_words[usize::try_from(token / 32).unwrap()] |= 1 << (token % 32);
        }
        backend
            .write(
                &sparse,
                &sparse_words
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let measure = |op: Op, inputs: &[&Tensor], variant: Option<&str>| {
            let mut timings = Vec::with_capacity(20);
            for iteration in 0..23 {
                let mut commands = CommandList::new();
                commands.dispatch(op, inputs, &output).unwrap();
                if let Some(name) = variant {
                    let dispatch = commands.last_dispatch().unwrap();
                    let selected = backend.validate_variant(dispatch, name).unwrap();
                    assert!(commands.set_last_backend_data(BackendDispatchData::new(selected)));
                }
                let submission = backend.submit_profiled(commands).unwrap();
                submission.wait().unwrap();
                if iteration >= 3 {
                    timings.push(submission.profile().unwrap().per_dispatch[0].gpu_time);
                }
            }
            timings.sort_unstable();
            timings[timings.len() / 2]
        };
        let ordinary = measure(sample_op(511), &[&logits, &params], None);
        let parallel = measure(sample_op(511), &[&logits, &params], Some("sample.parallel"));
        eprintln!("sample width {width} parallel: {parallel:?}, ordinary {ordinary:?}");
        assert!(parallel.as_secs_f64() <= ordinary.as_secs_f64() * 1.5);
        for (name, mask) in [("sparse", &sparse), ("dense", &dense)] {
            let masked = measure(sample_masked_op(511), &[&logits, &params, mask], None);
            eprintln!("sample width {width} masked {name}: {masked:?}, ordinary {ordinary:?}");
            assert!(masked.as_secs_f64() <= ordinary.as_secs_f64() * 1.5);
        }
    }

    #[test]
    fn metal_rope_matches_cpu_for_grouped_attention_shapes_and_position() {
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

            assert!(matches!(
                backend.submit(commands).unwrap().wait(),
                Err(BackendError::DeviceErrorFlag {
                    dispatch: 0,
                    operation: "embed",
                    flag: 1,
                    value,
                    ..
                }) if value == expected
            ));
            let _completion_error = backend.read(&output);
            assert_eq!(backend.read(&output).unwrap(), vec![0_u8; bytes.len() * 7]);
        }
    }
}
