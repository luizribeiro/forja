use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::{self, NonNull},
    slice,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
use crate::encoding::last_submission_timing;
use crate::encoding::{
    Completion, InFlightTracker, MetalProgram, MetalProgramHandle, MetalSubmission, PipelineCache,
};
use block2::RcBlock;
use forja_core::{
    AllocationRegistry, Backend, BackendError, BufferId, CommandList, DType, GraphTemplate, Layout,
    MappedRegion, PreparedGraph, ReadonlyImport, Tensor, ViewOp,
    program::{KernelSignature, ProgramHash, ValidatedProgram},
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSPageSize;
use objc2_metal::{
    MTL4CommandQueue, MTLAllocation, MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent,
    MTLSharedEventListener,
};

pub(super) struct MetalBuffer {
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) len: usize,
    pub(super) data_offset: usize,
    pub(super) pooled: bool,
    pub(super) pool_resident: bool,
    pending: Vec<PendingCompletion>,
}

struct PendingCompletion {
    completion: Weak<Completion>,
    range: std::ops::Range<u64>,
    writes: bool,
}

// SAFETY: Buffer access and pending-submission tracking are serialized by the backend registry
// mutex, and Metal buffers support use from multiple host threads.
unsafe impl Send for MetalBuffer {}

impl MetalBuffer {
    fn clear(&mut self) {
        // SAFETY: `raw` is a live shared-storage buffer of `len` bytes and the pool has exclusive
        // CPU access before the allocation is exposed to a caller.
        unsafe {
            ptr::write_bytes(self.raw.contents().cast::<u8>().as_ptr(), 0, self.len);
        }
    }

    fn write(&mut self, range: std::ops::Range<usize>, source: &[u8]) {
        // SAFETY: `raw` is a live shared-storage buffer of `len` bytes, the registry grants
        // exclusive CPU access, and the validated range has exactly `source.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                source.as_ptr(),
                self.raw
                    .contents()
                    .cast::<u8>()
                    .as_ptr()
                    .add(self.data_offset + range.start),
                source.len(),
            );
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `raw` is retained for the returned borrow and Metal guarantees `contents()`
        // points to all `len` bytes of a shared-storage buffer.
        unsafe {
            slice::from_raw_parts(
                self.raw
                    .contents()
                    .cast::<u8>()
                    .as_ptr()
                    .add(self.data_offset),
                self.len,
            )
        }
    }

    pub(super) fn wait_pending(&mut self, timeout: Duration) -> Result<(), BackendError> {
        let mut result = Ok(());
        for pending in std::mem::take(&mut self.pending) {
            let Some(completion) = pending.completion.upgrade() else {
                continue;
            };
            if let Err(error) = completion.wait(timeout)
                && (pending.writes || error == BackendError::Timeout)
            {
                result = Err(error);
                self.pending.push(PendingCompletion {
                    completion: Arc::downgrade(&completion),
                    range: pending.range,
                    writes: pending.writes,
                });
            }
        }
        result
    }

    fn wait_pending_range(
        &mut self,
        range: &std::ops::Range<u64>,
        timeout: Duration,
    ) -> Result<(), BackendError> {
        let mut result = Ok(());
        self.pending.retain(|pending| {
            let Some(completion) = pending.completion.upgrade() else {
                return false;
            };
            if pending.range.start >= range.end || range.start >= pending.range.end {
                return true;
            }
            match completion.wait(timeout) {
                Err(error) if pending.writes => {
                    result = Err(error);
                    true
                }
                Ok(()) | Err(_) => false,
            }
        });
        result
    }

    pub(super) fn track(&mut self, tensor: &Tensor, completion: &Arc<Completion>, writes: bool) {
        self.pending.push(PendingCompletion {
            completion: Arc::downgrade(completion),
            range: tensor.layout().byte_span(),
            writes,
        });
    }

    pub(super) fn dependencies(&mut self, range: &std::ops::Range<u64>) -> Vec<Arc<Completion>> {
        let mut dependencies = Vec::new();
        self.pending.retain(|pending| {
            let Some(completion) = pending.completion.upgrade() else {
                return false;
            };
            if completion.result() == Some(Ok(())) {
                return false;
            }
            if pending.writes && pending.range.start < range.end && range.start < pending.range.end
            {
                dependencies.push(completion);
            }
            true
        });
        dependencies
    }

    pub(super) fn pending_event_value(&mut self) -> Option<u64> {
        let mut maximum = None;
        self.pending.retain(|pending| {
            let Some(completion) = pending.completion.upgrade() else {
                return false;
            };
            maximum = maximum.max(completion.event_value());
            true
        });
        maximum
    }
}

struct BufferPool {
    free: HashMap<usize, Vec<MetalBuffer>>,
    bytes: u64,
    capacity: u64,
    residency: PoolResidency,
}

struct PoolResidency {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: Host mutations of the residency set are serialized by the pool mutex, and Metal
// residency sets support queue use while committed allocations remain immutable.
unsafe impl Send for PoolResidency {}

impl BufferPool {
    fn take(&mut self, len: usize) -> Result<Option<MetalBuffer>, BackendError> {
        let Some(buffer) = self.free.get_mut(&len).and_then(Vec::pop) else {
            return Ok(None);
        };
        self.bytes = self
            .bytes
            .checked_sub(u64::try_from(buffer.len).map_err(|_| BackendError::ExecutionFailed)?)
            .ok_or(BackendError::ExecutionFailed)?;
        Ok(Some(buffer))
    }

    fn put(&mut self, mut buffer: MetalBuffer) -> Result<(), BackendError> {
        let buffer_len = u64::try_from(buffer.len).map_err(|_| BackendError::AllocationFailed)?;
        let mut evicted = Vec::new();
        while self
            .bytes
            .checked_add(buffer_len)
            .is_none_or(|bytes| bytes > self.capacity)
        {
            let Some(len) = self
                .free
                .iter()
                .find_map(|(&len, buffers)| (!buffers.is_empty()).then_some(len))
            else {
                break;
            };
            let Some(mut allocation) = self.free.get_mut(&len).and_then(Vec::pop) else {
                break;
            };
            self.bytes = self
                .bytes
                .checked_sub(
                    u64::try_from(allocation.len).map_err(|_| BackendError::ExecutionFailed)?,
                )
                .ok_or(BackendError::ExecutionFailed)?;
            let raw: &ProtocolObject<dyn MTLAllocation> = allocation.raw.as_ref();
            self.residency.raw.removeAllocation(raw);
            allocation.pool_resident = false;
            evicted.push(allocation);
        }
        if buffer_len > self.capacity {
            if buffer.pool_resident {
                let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.raw.as_ref();
                self.residency.raw.removeAllocation(allocation);
                buffer.pool_resident = false;
            }
            if !evicted.is_empty() || !buffer.pool_resident {
                self.residency.raw.commit();
            }
            return Ok(());
        }
        if !buffer.pool_resident {
            let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.raw.as_ref();
            self.residency.raw.addAllocation(allocation);
            buffer.pool_resident = true;
        }
        self.residency.raw.commit();
        drop(evicted);
        self.bytes = self
            .bytes
            .checked_add(buffer_len)
            .ok_or(BackendError::AllocationFailed)?;
        self.free.entry(buffer.len).or_default().push(buffer);
        Ok(())
    }
}

const DEFAULT_POOL_CAPACITY: u64 = 1 << 30;

#[cfg(test)]
static LIVE_BACKENDS: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static LIVE_LISTENER_QUEUES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BackendDebugState {
    pub(super) live_backends: usize,
    pub(super) live_listener_queues: usize,
    pub(super) in_flight_submissions: usize,
    pub(super) pooled_submissions: usize,
    pub(super) pooled_bytes: u64,
    pub(super) residency_allocations: usize,
    pub(super) residency_bytes: u64,
    pub(super) last_submit: Duration,
    pub(super) last_wait: Duration,
}

/// Limits runtime scalar-program compilations for one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProgramCompileBudget {
    burst: usize,
    refill_interval: Duration,
}

impl ProgramCompileBudget {
    /// Creates a budget with `burst` initial tokens and one replacement token per interval.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] when either value is zero.
    pub fn new(burst: usize, refill_interval: Duration) -> Result<Self, BackendError> {
        if burst == 0 || refill_interval.is_zero() {
            return Err(BackendError::InvalidInput);
        }
        Ok(Self {
            burst,
            refill_interval,
        })
    }
}

impl Default for ProgramCompileBudget {
    fn default() -> Self {
        Self {
            burst: 64,
            refill_interval: Duration::from_millis(250),
        }
    }
}

pub(super) struct ProgramCompileTokens {
    configuration: ProgramCompileBudget,
    available: usize,
    last_refill: Instant,
}

impl ProgramCompileTokens {
    fn new(configuration: ProgramCompileBudget) -> Self {
        Self {
            configuration,
            available: configuration.burst,
            last_refill: Instant::now(),
        }
    }

    pub(super) fn consume(&mut self, count: usize) -> bool {
        self.refill(Instant::now());
        if count > self.available {
            return false;
        }
        self.available -= count;
        true
    }

    fn refill(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_refill);
        let intervals = elapsed.as_nanos() / self.configuration.refill_interval.as_nanos();
        let added = usize::try_from(intervals).unwrap_or(usize::MAX);
        if added == 0 {
            return;
        }
        self.available = self
            .configuration
            .burst
            .min(self.available.saturating_add(added));
        self.last_refill = now;
    }
}

/// A Metal 4 backend using shared unified-memory buffers.
///
/// Each backend owns an isolated buffer pool and clears reused storage before allocation, so a
/// tensor cannot expose contents from an earlier allocation or another host store. Guest quotas
/// and allocation counts cover live logical tensors; released buffers instead count toward this
/// backend's separately capped pool using their page-rounded Metal allocation sizes.
/// A submission may compile at most four cache-missing scalar programs, and cache misses also
/// consume this backend's configurable cumulative compile budget.
pub struct MetalBackend {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pub(super) buffers: Mutex<AllocationRegistry<MetalBuffer>>,
    pool: Mutex<BufferPool>,
    pub(super) pipelines: Mutex<PipelineCache>,
    pub(super) prepared_programs:
        Mutex<HashMap<(ProgramHash, KernelSignature), Weak<MetalProgram>>>,
    pub(super) program_compile_tokens: Mutex<ProgramCompileTokens>,
    pub(super) in_flight: Arc<InFlightTracker>,
    pub(super) shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(super) next_event_value: Mutex<u64>,
    pub(super) event_listener: Retained<MTLSharedEventListener>,
    pub(super) gpu_timeout: Duration,
    pub(super) graph_replay: MetalGraphReplay,
}

/// Reusable graph execution strategy selected by the host operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalGraphReplay {
    /// Instantiate and encode every dispatch through the prevalidated path.
    Tier1,
    /// Replay prepared static dispatch plans and encode dynamic work normally.
    Tier2,
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        self.in_flight.drain(self.gpu_timeout);
        #[cfg(test)]
        {
            LIVE_LISTENER_QUEUES.fetch_sub(1, Ordering::Relaxed);
            LIVE_BACKENDS.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl MetalBackend {
    /// Returns the selected Metal device name.
    #[must_use]
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Creates a backend on the system default Metal 4 device.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn new() -> Result<Self, BackendError> {
        Self::with_configuration(
            Duration::from_secs(10),
            DEFAULT_POOL_CAPACITY,
            ProgramCompileBudget::default(),
            MetalGraphReplay::Tier2,
        )
    }

    /// Creates a backend with an explicit reusable graph execution strategy.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_graph_replay(graph_replay: MetalGraphReplay) -> Result<Self, BackendError> {
        Self::with_configuration(
            Duration::from_secs(10),
            DEFAULT_POOL_CAPACITY,
            ProgramCompileBudget::default(),
            graph_replay,
        )
    }

    /// Returns the configured reusable graph execution strategy.
    #[must_use]
    pub const fn graph_replay(&self) -> MetalGraphReplay {
        self.graph_replay
    }

    /// Creates a backend with a cumulative runtime scalar-program compile budget.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_program_compile_budget(budget: ProgramCompileBudget) -> Result<Self, BackendError> {
        Self::with_configuration(
            Duration::from_secs(10),
            DEFAULT_POOL_CAPACITY,
            budget,
            MetalGraphReplay::Tier2,
        )
    }

    /// Creates a backend with the default GPU timeout and a free-buffer pool byte cap.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::AllocationFailed`] when `pool_capacity` does not fit the host, or
    /// [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_pool_capacity(pool_capacity: u64) -> Result<Self, BackendError> {
        Self::with_configuration(
            Duration::from_secs(10),
            pool_capacity,
            ProgramCompileBudget::default(),
            MetalGraphReplay::Tier2,
        )
    }

    /// Creates a backend with a deadline for each wait on submitted GPU work.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_gpu_timeout(gpu_timeout: Duration) -> Result<Self, BackendError> {
        Self::with_configuration(
            gpu_timeout,
            DEFAULT_POOL_CAPACITY,
            ProgramCompileBudget::default(),
            MetalGraphReplay::Tier2,
        )
    }

    /// Creates a backend with a GPU wait deadline and a free-buffer pool byte cap.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::AllocationFailed`] when `pool_capacity` does not fit the host, or
    /// [`BackendError::ExecutionFailed`] when Metal 4 is unavailable.
    pub fn with_gpu_timeout_and_pool_capacity(
        gpu_timeout: Duration,
        pool_capacity: u64,
    ) -> Result<Self, BackendError> {
        Self::with_configuration(
            gpu_timeout,
            pool_capacity,
            ProgramCompileBudget::default(),
            MetalGraphReplay::Tier2,
        )
    }

    fn with_configuration(
        gpu_timeout: Duration,
        pool_capacity: u64,
        program_compile_budget: ProgramCompileBudget,
        graph_replay: MetalGraphReplay,
    ) -> Result<Self, BackendError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(BackendError::ExecutionFailed)?;
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(BackendError::ExecutionFailed);
        }
        let queue = device
            .newMTL4CommandQueue()
            .ok_or(BackendError::ExecutionFailed)?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|_| BackendError::ExecutionFailed)?;
        residency.commit();
        queue.addResidencySet(&residency);
        let source = concat!(
            include_str!("elementwise.metal"),
            "\n",
            include_str!("kernels.metal"),
            "\n",
            include_str!("steel_gemm_loader.metal"),
            "\n",
            include_str!("steel_gemm_mma.metal"),
            "\n",
            include_str!("matmul.metal"),
            "\n",
            include_str!("quant_matmul.metal"),
            "\n",
            include_str!("gather_matmul.metal"),
            "\n",
            include_str!("sdpa.metal"),
            "\n",
            include_str!("sdpa_vector.metal")
        );
        let pipelines = PipelineCache::new(&device, source)?;
        let shared_event = device
            .newSharedEvent()
            .ok_or(BackendError::ExecutionFailed)?;
        let event_listener = MTLSharedEventListener::new();
        let backend = Self {
            device,
            queue,
            buffers: Mutex::new(AllocationRegistry::new()),
            pool: Mutex::new(BufferPool {
                free: HashMap::new(),
                bytes: 0,
                capacity: pool_capacity,
                residency: PoolResidency { raw: residency },
            }),
            pipelines: Mutex::new(pipelines),
            prepared_programs: Mutex::new(HashMap::new()),
            program_compile_tokens: Mutex::new(ProgramCompileTokens::new(program_compile_budget)),
            in_flight: Arc::new(InFlightTracker::new()),
            shared_event,
            next_event_value: Mutex::new(1),
            event_listener,
            gpu_timeout,
            graph_replay,
        };
        #[cfg(test)]
        {
            LIVE_BACKENDS.fetch_add(1, Ordering::Relaxed);
            LIVE_LISTENER_QUEUES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(backend)
    }

    #[cfg(test)]
    pub(super) fn debug_state(&self) -> BackendDebugState {
        self.in_flight.drain_done();
        let (last_submit, last_wait) = last_submission_timing();
        let pool = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        BackendDebugState {
            live_backends: LIVE_BACKENDS.load(Ordering::Relaxed),
            live_listener_queues: LIVE_LISTENER_QUEUES.load(Ordering::Relaxed),
            in_flight_submissions: self.in_flight.len(),
            pooled_submissions: self.in_flight.pooled_len(),
            pooled_bytes: pool.bytes,
            residency_allocations: pool.residency.raw.allocationCount(),
            residency_bytes: pool.residency.raw.allocatedSize(),
            last_submit,
            last_wait,
        }
    }

    #[cfg(test)]
    pub(super) fn debug_live_counts() -> (usize, usize) {
        (
            LIVE_BACKENDS.load(Ordering::Relaxed),
            LIVE_LISTENER_QUEUES.load(Ordering::Relaxed),
        )
    }

    pub(super) fn validate(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(tensor)
            .map(|_| ())
    }
}

impl Backend for MetalBackend {
    type Submission = MetalSubmission;
    type ProgramHandle = MetalProgramHandle;

    fn prepare_program(
        &self,
        program: &ValidatedProgram,
        signature: &KernelSignature,
    ) -> Result<Self::ProgramHandle, BackendError> {
        self.prepare_scalar_program(program, signature)
    }

    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        let byte_len = element_count(shape)?
            .checked_mul(dtype.byte_size())
            .ok_or(BackendError::AllocationFailed)?;
        let len = usize::try_from(byte_len)
            .map_err(|_| BackendError::AllocationFailed)?
            .max(1)
            .checked_next_multiple_of(4096)
            .ok_or(BackendError::AllocationFailed)?;
        let mut pool = self
            .pool
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = if let Some(mut buffer) = pool.take(len)? {
            buffer.clear();
            buffer
        } else {
            let raw = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or(BackendError::AllocationFailed)?;
            MetalBuffer {
                raw,
                len,
                data_offset: 0,
                pooled: true,
                pool_resident: false,
                pending: Vec::new(),
            }
        };
        drop(pool);
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let layout = Layout::contiguous(dtype, 0, shape.to_vec(), byte_len)
            .map_err(|_| BackendError::InvalidInput)?;
        let id = buffers.insert(buffer, byte_len)?;
        buffers.tensor(id, layout)
    }

    fn import_readonly(&self, bytes: MappedRegion) -> Result<ReadonlyImport, BackendError> {
        let byte_len = u64::try_from(bytes.len()).map_err(|_| BackendError::AllocationFailed)?;
        let copied = !bytes.offset().is_multiple_of(GPU_ADDRESS_ALIGNMENT);
        let buffer = readonly_buffer(&self.device, bytes)?;
        let buffer = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .insert_read_only(buffer, byte_len)?;
        Ok(if copied {
            ReadonlyImport::Copied(buffer)
        } else {
            ReadonlyImport::Mapped(buffer)
        })
    }

    fn read_buffer_range(
        &self,
        buffer: BufferId,
        range: std::ops::Range<u64>,
    ) -> Result<Vec<u8>, BackendError> {
        let start = usize::try_from(range.start).map_err(|_| BackendError::InvalidInput)?;
        let end = usize::try_from(range.end).map_err(|_| BackendError::InvalidInput)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_buffer_mut(buffer)?;
        buffer.wait_pending(self.gpu_timeout)?;
        buffer
            .bytes()
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or(BackendError::InvalidInput)
    }

    fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
        self.buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .tensor(buffer, layout)
    }

    fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let layout = match op {
            ViewOp::Slice(spec) => tensor.layout().slice(&spec),
            ViewOp::Reshape(shape) => tensor.layout().reshape(&shape),
            ViewOp::Permute(axes) => tensor.layout().permute(&axes),
            ViewOp::Broadcast(shape) => tensor.layout().broadcast(&shape),
        }
        .map_err(|_| BackendError::InvalidInput)?;
        buffers.view(tensor, layout)
    }

    fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        if !tensor.is_writable()
            || !tensor.layout().is_contiguous()
            || bytes.len() != logical_byte_len(tensor.layout())?
        {
            return Err(BackendError::InvalidInput);
        }
        let range = tensor.layout().byte_span();
        let range = usize::try_from(range.start).map_err(|_| BackendError::InvalidInput)?
            ..usize::try_from(range.end).map_err(|_| BackendError::InvalidInput)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_mut(tensor)?;
        buffer.wait_pending(self.gpu_timeout)?;
        buffer.write(range, bytes);
        Ok(())
    }

    fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_mut(tensor)?;
        buffer.wait_pending(self.gpu_timeout)?;
        gather(buffer.bytes(), tensor.layout())
    }

    fn read_replay_output(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let buffer = buffers.get_mut(tensor)?;
        buffer.wait_pending_range(&tensor.layout().byte_span(), self.gpu_timeout)?;
        gather(buffer.bytes(), tensor.layout())
    }

    fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        buffers.get_mut(tensor)?.wait_pending(self.gpu_timeout)?;
        let buffer = buffers.remove(tensor)?;
        drop(buffers);
        if buffer.pooled {
            self.pool
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?
                .put(buffer)?;
        }
        Ok(())
    }

    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        self.submit_commands(commands)
    }

    fn submit_profiled(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        self.submit_commands_profiled(commands)
    }

    fn prepare_graph(&self, graph: GraphTemplate) -> Result<PreparedGraph, BackendError> {
        let state = self.prepare_metal_graph(&graph)?;
        Ok(PreparedGraph::with_backend_state(graph, state))
    }

    fn replay(
        &self,
        graph: &PreparedGraph,
        values: Vec<u32>,
    ) -> Result<Self::Submission, BackendError> {
        self.replay_graph(graph, values, false)
    }

    fn replay_profiled(
        &self,
        graph: &PreparedGraph,
        values: Vec<u32>,
    ) -> Result<Self::Submission, BackendError> {
        self.replay_graph(graph, values, true)
    }

    fn supports_replay_overlap(&self) -> bool {
        true
    }
}

type BufferDeallocator = RcBlock<dyn Fn(NonNull<c_void>, usize)>;
const GPU_ADDRESS_ALIGNMENT: usize = 16;

fn readonly_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    region: MappedRegion,
) -> Result<MetalBuffer, BackendError> {
    let len = region.len();
    let data_offset = region.offset();
    if !data_offset.is_multiple_of(GPU_ADDRESS_ALIGNMENT) {
        let pointer = NonNull::new(region.bytes().as_ptr().cast_mut().cast::<c_void>())
            .ok_or(BackendError::InvalidInput)?;
        // SAFETY: The pointer covers `len` initialized bytes for the duration of this synchronous
        // call, which copies them into a new shared-storage Metal allocation.
        let raw = unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(BackendError::AllocationFailed)?;
        return Ok(MetalBuffer {
            raw,
            len,
            data_offset: 0,
            pooled: false,
            pool_resident: false,
            pending: Vec::new(),
        });
    }
    let rounded_len = region
        .mapped_len()
        .checked_add(NSPageSize().saturating_sub(1))
        .map(|bytes| bytes / NSPageSize() * NSPageSize())
        .ok_or(BackendError::AllocationFailed)?;
    let pointer = NonNull::new(region.as_ptr().cast_mut().cast::<c_void>())
        .ok_or(BackendError::InvalidInput)?;
    let deallocator: BufferDeallocator = RcBlock::new(move |_pointer, _length| {
        let _ = region.len();
    });
    // SAFETY: The full-file mapping starts at a page-aligned address and remains alive in the
    // sendable deallocator block until Metal releases the buffer. The OS mapping covers the final
    // partial page, and validated tensor layouts expose only the file's actual byte length.
    let raw = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            pointer,
            rounded_len,
            MTLResourceOptions::StorageModeShared,
            Some(&deallocator),
        )
    }
    .ok_or(BackendError::AllocationFailed)?;
    Ok(MetalBuffer {
        raw,
        len,
        data_offset,
        pooled: false,
        pool_resident: false,
        pending: Vec::new(),
    })
}

fn element_count(shape: &[u32]) -> Result<u64, BackendError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1_u64, |count, &extent| {
        count
            .checked_mul(u64::from(extent))
            .ok_or(BackendError::AllocationFailed)
    })
}

fn logical_byte_len(layout: &Layout) -> Result<usize, BackendError> {
    let bytes = layout
        .element_count()
        .checked_mul(layout.dtype().byte_size())
        .ok_or(BackendError::InvalidInput)?;
    usize::try_from(bytes).map_err(|_| BackendError::InvalidInput)
}

fn element_offsets(layout: &Layout) -> impl Iterator<Item = u64> + '_ {
    (0..layout.element_count()).map(|mut linear| {
        let mut offset = layout.offset();
        for (&extent, &stride) in layout.shape().iter().zip(layout.strides()).rev() {
            offset += (linear % u64::from(extent)) * stride;
            linear /= u64::from(extent);
        }
        offset
    })
}

fn gather(source: &[u8], layout: &Layout) -> Result<Vec<u8>, BackendError> {
    let width =
        usize::try_from(layout.dtype().byte_size()).map_err(|_| BackendError::InvalidInput)?;
    let mut result = Vec::with_capacity(logical_byte_len(layout)?);
    for offset in element_offsets(layout) {
        let start = usize::try_from(
            offset
                .checked_mul(layout.dtype().byte_size())
                .ok_or(BackendError::InvalidInput)?,
        )
        .map_err(|_| BackendError::InvalidInput)?;
        result.extend_from_slice(
            source
                .get(start..start + width)
                .ok_or(BackendError::InvalidInput)?,
        );
    }
    Ok(result)
}

#[cfg(test)]
mod mapped_tests {
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    use forja_core::{Op, OpError, Submission};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn mapped_weights_are_read_only_gpu_inputs() {
        let expected = [1.0_f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "forja-metal-mapping-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut stored = vec![0xff];
        stored.extend_from_slice(&expected);
        fs::write(&path, stored).unwrap();
        let region = MappedRegion::map(&fs::File::open(&path).unwrap())
            .unwrap()
            .split_at(1)
            .unwrap();
        let backend = MetalBackend::new().unwrap();
        let imported = backend.import_readonly(region).unwrap();
        assert!(matches!(imported, ReadonlyImport::Copied(_)));
        let buffer = imported.buffer();
        fs::remove_file(path).unwrap();
        let weight = backend
            .tensor(
                buffer,
                Layout::contiguous(DType::F32, 0, vec![4], 16).unwrap(),
            )
            .unwrap();
        let output = backend.alloc(DType::F32, &[4]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&weight], &output).unwrap();
        backend.submit(commands).unwrap().wait().unwrap();

        assert_eq!(backend.read(&output).unwrap(), expected);
        assert_eq!(
            backend.write(&weight, &[0; 16]),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&output], &weight),
            Err(OpError::ReadOnlyOutput)
        );
    }
}

#[cfg(test)]
mod tests {
    use forja_core::{Op, Submission};

    use super::*;

    #[test]
    fn backend_debug_state_tracks_owned_resources() {
        assert_eq!(MetalBackend::debug_live_counts(), (0, 0));
        let backend = MetalBackend::new().unwrap();
        assert_eq!(
            backend.debug_state(),
            BackendDebugState {
                live_backends: 1,
                live_listener_queues: 1,
                in_flight_submissions: 0,
                pooled_submissions: 0,
                pooled_bytes: 0,
                residency_allocations: 0,
                residency_bytes: 0,
                last_submit: Duration::ZERO,
                last_wait: Duration::ZERO,
            }
        );
        drop(backend);
        assert_eq!(MetalBackend::debug_live_counts(), (0, 0));
    }

    #[test]
    fn program_compile_tokens_refill_to_the_configured_burst() {
        let configuration = ProgramCompileBudget::new(2, Duration::from_hours(24)).unwrap();
        let mut tokens = ProgramCompileTokens::new(configuration);
        let started = tokens.last_refill;
        assert!(tokens.consume(2));
        assert!(!tokens.consume(1));
        tokens.refill(started + Duration::from_hours(24));
        assert!(tokens.consume(1));
        assert!(!tokens.consume(1));
        tokens.refill(started + Duration::from_hours(72));
        assert!(tokens.consume(2));
    }

    #[test]
    fn metal_storage_round_trips_bytes() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let bytes = (0_u8..=u8::MAX)
            .cycle()
            .take(7 * 33 * 4)
            .collect::<Vec<_>>();
        backend.write(&tensor, &bytes).unwrap();
        assert_eq!(backend.read(&tensor).unwrap(), bytes);
    }

    #[test]
    fn metal_storage_reads_a_permuted_view() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[2, 3]).unwrap();
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&tensor, &values).unwrap();
        let view = backend.view(&tensor, ViewOp::Permute(vec![1, 0])).unwrap();
        let expected = [1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(backend.read(&view).unwrap(), expected);
    }

    #[test]
    fn metal_storage_rejects_use_after_release() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::F32, &[7]).unwrap();
        let view = backend.view(&tensor, ViewOp::Reshape(vec![1, 7])).unwrap();
        backend.release(&tensor).unwrap();
        assert_eq!(backend.read(&view), Err(BackendError::InvalidInput));
    }

    #[test]
    fn pooled_storage_is_reused_zeroed_and_resident() {
        let backend = MetalBackend::new().unwrap();
        let tensor = backend.alloc(DType::U32, &[7]).unwrap();
        let address = backend
            .buffers
            .lock()
            .unwrap()
            .get(&tensor)
            .unwrap()
            .raw
            .gpuAddress();
        backend.write(&tensor, &[0xa5; 28]).unwrap();
        backend.release(&tensor).unwrap();

        let reused = backend.alloc(DType::U32, &[5]).unwrap();
        let reused_address = backend
            .buffers
            .lock()
            .unwrap()
            .get(&reused)
            .unwrap()
            .raw
            .gpuAddress();
        assert_eq!(reused_address, address);
        assert_eq!(backend.read(&reused).unwrap(), [0; 20]);
        assert_eq!(
            backend.pool.lock().unwrap().residency.raw.allocationCount(),
            1
        );

        let source = backend.alloc(DType::U32, &[5]).unwrap();
        let expected = [7_u32; 5]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&source, &expected).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&source], &reused).unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        assert_eq!(backend.read(&reused).unwrap(), expected);
    }

    #[test]
    fn pooled_storage_is_zeroed_within_and_across_backends() {
        let first = MetalBackend::new().unwrap();
        let second = MetalBackend::new().unwrap();
        let tensor = first.alloc(DType::U32, &[33]).unwrap();
        first.write(&tensor, &[0xa5; 33 * 4]).unwrap();
        first.release(&tensor).unwrap();

        let isolated = second.alloc(DType::U32, &[33]).unwrap();
        assert_eq!(second.read(&isolated).unwrap(), [0; 33 * 4]);
        second.write(&isolated, &[0x5a; 33 * 4]).unwrap();
        second.release(&isolated).unwrap();
        let reused = second.alloc(DType::U32, &[33]).unwrap();
        assert_eq!(second.read(&reused).unwrap(), [0; 33 * 4]);
    }

    #[test]
    fn pooled_bytes_are_capped_and_evictions_leave_residency() {
        let backend = MetalBackend::with_pool_capacity(4096).unwrap();
        let first = backend.alloc(DType::U32, &[1]).unwrap();
        let retained = backend
            .buffers
            .lock()
            .unwrap()
            .get(&first)
            .unwrap()
            .raw
            .clone();
        backend.release(&first).unwrap();

        for size in 2..=1000 {
            let tensor = backend.alloc(DType::U32, &[size]).unwrap();
            backend.release(&tensor).unwrap();
            assert!(backend.pool.lock().unwrap().bytes <= 4096);
        }

        let oversized = backend.alloc(DType::U32, &[1025]).unwrap();
        backend.release(&oversized).unwrap();
        let pool = backend.pool.lock().unwrap();
        let allocation: &ProtocolObject<dyn MTLAllocation> = retained.as_ref();
        assert_eq!(pool.bytes, 0);
        assert!(!pool.residency.raw.containsAllocation(allocation));
    }

    #[test]
    fn release_pools_storage_after_submission_completion() {
        let backend = MetalBackend::new().unwrap();
        let source = backend.alloc(DType::F32, &[4097]).unwrap();
        let output = backend.alloc(DType::F32, &[4097]).unwrap();
        let address = backend
            .buffers
            .lock()
            .unwrap()
            .get(&output)
            .unwrap()
            .raw
            .gpuAddress();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&source], &output).unwrap();
        let submission = backend.submit(commands).unwrap();
        drop(submission);

        backend.release(&output).unwrap();
        let replacement = backend.alloc(DType::F32, &[4097]).unwrap();
        let replacement_address = backend
            .buffers
            .lock()
            .unwrap()
            .get(&replacement)
            .unwrap()
            .raw
            .gpuAddress();
        assert_eq!(replacement_address, address);
    }
}
