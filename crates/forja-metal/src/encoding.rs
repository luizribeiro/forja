use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, Instant},
};

use block2::RcBlock;
use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, Layout, Op, Slice, Submission, Tensor,
    required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue,
    MTL4CommitFeedback, MTL4CommitOptions, MTL4CounterHeap, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLEvent, MTLFunctionConstantValues, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent, MTLSharedEventListener, MTLSize,
};

use crate::{
    matmul::{classify, select_gemm},
    storage::MetalBackend,
};

type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;
type EncodedEmbed = (Vec<MetalBufferRef>, u64);

#[derive(Clone)]
struct EncoderTensor {
    buffer: MetalBufferRef,
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

struct EncodedDispatches {
    temporaries: Vec<MetalBufferRef>,
    error_flags: Vec<u64>,
    bindings: ArgumentBindings,
}

#[derive(Default)]
struct ArgumentBindings {
    addresses: HashSet<u64>,
}

impl ArgumentBindings {
    fn bind(
        &mut self,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
    ) {
        let address = buffer.gpuAddress();
        // SAFETY: Each caller uses an index within its argument-table descriptor and registers
        // the bound buffer in the command resource owner before submission.
        unsafe {
            table.setAddress_atIndex(address, index);
        }
        self.addresses.insert(address);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PipelineKey {
    name: String,
    constants: Vec<(u32, u32)>,
}

struct CompletionState {
    feedback: Option<CommitResult>,
    event_signaled: bool,
    committed: Option<Instant>,
    feedback_elapsed: Option<Duration>,
    event_elapsed: Option<Duration>,
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
    pub(super) _raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: The residency set is committed before submission and remains immutable while shared
// across completion and waiting threads.
unsafe impl Send for InFlightResidency {}

// SAFETY: The wrapper exposes no operations on the retained residency set.
unsafe impl Sync for InFlightResidency {}

struct ReusableSubmissionObjects {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
}

// SAFETY: Ownership moves between the encoding thread and the completion tracker, and the
// allocator and argument table are not reused until GPU completion.
unsafe impl Send for ReusableSubmissionObjects {}

struct SubmissionObjects {
    reusable: Option<ReusableSubmissionObjects>,
    owner: Weak<InFlightTracker>,
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
}

// SAFETY: Counter heaps remain immutable on the CPU while the retained submission is in flight,
// and timestamp resolution happens only after the shared event establishes GPU completion.
unsafe impl Send for GpuTimestamps {}

// SAFETY: Shared access only resolves completed timestamp data and does not mutate the heap.
unsafe impl Sync for GpuTimestamps {}

impl GpuTimestamps {
    fn elapsed(&self) -> Option<Duration> {
        // SAFETY: The heap has two entries and callers only resolve timestamps after waiting for
        // the submission's shared-event signal.
        let data = unsafe { self.heap.resolveCounterRange(NSRange::new(0, 2)) }?;
        let bytes = data.to_vec();
        let end_offset = self.entry_size;
        let start = u64::from_ne_bytes(bytes.get(..8)?.try_into().ok()?);
        let end = u64::from_ne_bytes(
            bytes
                .get(end_offset..end_offset.checked_add(8)?)?
                .try_into()
                .ok()?,
        );
        let ticks = end.checked_sub(start)?;
        let seconds = ticks / self.frequency;
        let nanos = u128::from(ticks % self.frequency).checked_mul(1_000_000_000)?
            / u128::from(self.frequency);
        Some(Duration::new(seconds, u32::try_from(nanos).ok()?))
    }
}

struct CommandResources {
    buffers: Vec<InFlightBuffer>,
    error_flags: Vec<usize>,
}

pub(super) struct Completion {
    state: Mutex<CompletionState>,
    ready: Condvar,
    event: InFlightEvent,
    resources: CommandResources,
    _residency: InFlightResidency,
    timestamps: Option<GpuTimestamps>,
    tracker: Weak<InFlightTracker>,
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
                    committed: None,
                    feedback_elapsed: None,
                    event_elapsed: None,
                }),
                ready: Condvar::new(),
                event,
                resources,
                _residency: residency,
                timestamps,
                tracker,
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
            if state.event_signaled
                && let Some(feedback) = &state.feedback
            {
                let result = commit_result(feedback);
                drop(state);
                if let Some(tracker) = self.tracker.upgrade() {
                    tracker.drain_done();
                }
                result?;
                return self.check_error_flags();
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(BackendError::Timeout);
            }
            let (next, wait) = self
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && (!state.event_signaled || state.feedback.is_none()) {
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

    fn mark_committed(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .committed = Some(Instant::now());
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
        for &flag in &self.resources.error_flags {
            // SAFETY: The queue event has signaled GPU completion, and each indexed retained
            // shared buffer contains two aligned u32 values initialized by the host.
            let (has_error, index) = unsafe {
                let words = self.resources.buffers[flag]
                    .raw
                    .contents()
                    .cast::<u32>()
                    .as_ptr();
                (words.read(), words.add(1).read())
            };
            if has_error != 0 {
                return Err(BackendError::IndexOutOfRange { index });
            }
        }
        Ok(())
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.timestamps.as_ref()?.elapsed()
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
    ) -> Result<SubmissionObjects, BackendError> {
        use objc2_metal::MTL4ArgumentTableDescriptor;

        self.drain_done();
        let reusable = self
            .pool
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .pop();
        let reusable = if let Some(objects) = reusable {
            reset_allocator(&objects.allocator);
            objects
        } else {
            let allocator = device
                .newCommandAllocator()
                .ok_or(BackendError::ExecutionFailed)?;
            let descriptor = MTL4ArgumentTableDescriptor::new();
            descriptor.setMaxBufferBindCount(8);
            let argument_table = device
                .newArgumentTableWithDescriptor_error(&descriptor)
                .map_err(|_| BackendError::ExecutionFailed)?;
            ReusableSubmissionObjects {
                allocator,
                argument_table,
            }
        };
        Ok(SubmissionObjects {
            reusable: Some(reusable),
            owner: Arc::downgrade(self),
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

    fn drain_done(&self) {
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
    fn len(&self) -> usize {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    #[cfg(test)]
    fn pooled_len(&self) -> usize {
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
}

impl Submission for MetalSubmission {
    fn wait(&self) -> Result<(), BackendError> {
        self.completion.wait(self.timeout)
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<(), BackendError> {
        self.completion.wait(timeout)
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.completion.gpu_time()
    }
}

impl MetalSubmission {
    /// Returns feedback and shared-event callback latency from queue commit after a successful wait.
    #[must_use]
    pub fn completion_timing(&self) -> Option<(Duration, Duration)> {
        self.completion.completion_timing()
    }
}

impl MetalBackend {
    pub(super) fn submit_commands(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
        self.in_flight.drain_done();
        let barriers = required_barriers(&commands);
        let dispatches = commands.into_dispatches();
        if dispatches.iter().any(|dispatch| {
            !matches!(
                dispatch.op(),
                Op::Copy
                    | Op::Add
                    | Op::SiluMul
                    | Op::RmsNorm { .. }
                    | Op::Softmax
                    | Op::Rope { .. }
                    | Op::Embed
                    | Op::Matmul
                    | Op::Sdpa { .. }
            )
        }) {
            return Err(BackendError::InvalidInput);
        }
        let tensors = dispatches
            .iter()
            .flat_map(|dispatch| {
                dispatch
                    .inputs()
                    .iter()
                    .chain(std::iter::once(dispatch.output()))
            })
            .cloned()
            .collect::<Vec<_>>();
        for tensor in &tensors {
            self.validate(tensor)?;
        }
        let mut objects = self.in_flight.checkout(&self.device)?;
        let command_buffer = self.begin_command_buffer(objects.allocator()?)?;
        let timestamps = self.make_timestamps()?;
        // SAFETY: The timestamp heap has two entries and is retained until completion.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 0);
        }
        let encoded = self.encode_dispatches(
            &command_buffer,
            objects.argument_table()?,
            &dispatches,
            &barriers,
        )?;
        let resources = self.command_resources(&tensors, encoded)?;
        let residency = self.make_resident(&command_buffer, &resources)?;
        // SAFETY: The timestamp heap has two entries and is retained until completion.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 1);
        }
        command_buffer.endCommandBuffer();
        self.commit(
            &command_buffer,
            &tensors,
            resources,
            residency,
            timestamps,
            &mut objects,
        )
    }

    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatches: &[Dispatch],
        barriers: &[bool],
    ) -> Result<EncodedDispatches, BackendError> {
        use objc2_metal::{MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages};

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(BackendError::ExecutionFailed)?;
        let mut temporaries = Vec::with_capacity(dispatches.len().saturating_mul(3));
        let mut error_flags = Vec::new();
        let mut bindings = ArgumentBindings::default();
        for (dispatch, &barrier) in dispatches.iter().zip(barriers) {
            if barrier {
                encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    MTLStages::Dispatch,
                    MTLStages::Dispatch,
                    MTL4VisibilityOptions::Device,
                );
            }
            if let Op::RmsNorm { eps } = dispatch.op() {
                temporaries.extend(self.encode_rms_norm(
                    &encoder,
                    table,
                    dispatch,
                    eps,
                    &mut bindings,
                )?);
                continue;
            }
            if dispatch.op() == Op::Softmax {
                temporaries.extend(self.encode_softmax(
                    &encoder,
                    table,
                    dispatch,
                    &mut bindings,
                )?);
                continue;
            }
            if let Op::Rope { theta } = dispatch.op() {
                temporaries.extend(self.encode_rope(
                    &encoder,
                    table,
                    dispatch,
                    theta,
                    &mut bindings,
                )?);
                continue;
            }
            if dispatch.op() == Op::Embed {
                let (buffers, flag) =
                    self.encode_embed(&encoder, table, dispatch, &mut bindings)?;
                temporaries.extend(buffers);
                error_flags.push(flag);
                continue;
            }
            if dispatch.op() == Op::Copy {
                temporaries.extend(self.encode_copy(&encoder, table, dispatch, &mut bindings)?);
                continue;
            }
            if dispatch.op() == Op::Matmul {
                temporaries.extend(self.encode_matmul(&encoder, table, dispatch, &mut bindings)?);
                continue;
            }
            if matches!(dispatch.op(), Op::Sdpa { .. }) {
                temporaries.extend(self.encode_sdpa_dispatch(
                    &encoder,
                    table,
                    dispatch,
                    &mut bindings,
                )?);
                continue;
            }
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
            temporaries.extend(self.encode_elementwise(
                &encoder,
                table,
                dispatch,
                kernel,
                &mut bindings,
            )?);
        }
        encoder.endEncoding();
        Ok(EncodedDispatches {
            temporaries,
            error_flags,
            bindings,
        })
    }

    fn encode_copy(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let output = self.encoder_tensor(dispatch.output())?;
        self.encode_copy_tensors(encoder, table, &input, &output, bindings)
    }

    fn encode_matmul(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        use objc2_metal::{MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages};

        let [left_tensor, right_tensor] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output_tensor = dispatch.output();
        let dtype = output_tensor.layout().dtype();
        let mut temporaries = Vec::new();
        let left =
            self.prepare_matmul_input(encoder, table, left_tensor, bindings, &mut temporaries)?;
        let right =
            self.prepare_matmul_input(encoder, table, right_tensor, bindings, &mut temporaries)?;
        if left.copied || right.copied {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
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
                    },
                    true,
                )
            };
        let parameter_buffer =
            self.encode_matmul_kernel(encoder, table, &left, &right, &output, bindings)?;
        temporaries.push(parameter_buffer);
        if copy_output {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
            temporaries.extend(self.encode_copy_tensors(
                encoder,
                table,
                &output.tensor,
                &final_output,
                bindings,
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        let Op::Sdpa {
            scale,
            causal,
            q_start,
        } = dispatch.op()
        else {
            return Err(BackendError::InvalidInput);
        };
        match select_sdpa(dispatch)? {
            SdpaKernel::Vector => {
                self.encode_vector_sdpa(encoder, table, dispatch, scale, causal, q_start, bindings)
            }
            SdpaKernel::Steel => {
                self.encode_steel_sdpa(encoder, table, dispatch, scale, causal, q_start, bindings)
            }
            SdpaKernel::Decomposed => self
                .encode_decomposed_sdpa(encoder, table, dispatch, scale, causal, q_start, bindings),
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        let params = self.sdpa_params(
            &query, &key, &value, &output, scale, causal, q_start, blocks,
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
            encoder.setComputePipelineState(&pipeline);
            for (index, tensor) in [&query, &key, &value, &output].into_iter().enumerate() {
                bindings.bind(table, index, &tensor.buffer);
            }
            bindings.bind(table, 4, &params);
            encoder.setArgumentTable(Some(table));
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        encoder.setComputePipelineState(&first);
        for (index, tensor) in [&query, &key, &value, &intermediate, &sums, &maxs]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 6, &params);
        encoder.setArgumentTable(Some(table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        encoder.setComputePipelineState(&second);
        for (index, tensor) in [&intermediate, &sums, &maxs, &output]
            .into_iter()
            .enumerate()
        {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &params);
        encoder.setArgumentTable(Some(table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        let params = self.sdpa_params(&query, &key, &value, &output, scale, causal, q_start, 1)?;
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
        encoder.setComputePipelineState(&pipeline);
        for (index, tensor) in [&query, &key, &value, &output].into_iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 4, &params);
        encoder.setArgumentTable(Some(table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        &self,
        query: &EncoderTensor,
        key: &EncoderTensor,
        value: &EncoderTensor,
        output: &EncoderTensor,
        scale: f32,
        causal: bool,
        q_start: u32,
        blocks: u32,
    ) -> Result<MetalBufferRef, BackendError> {
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
        self.temporary_buffer(&bytes)
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
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
        )?);
        encode_dispatch_barrier(encoder);
        temporaries.push(
            self.encode_sdpa_scale_mask(encoder, table, &scores, scale, causal, q_start, bindings)?,
        );
        encode_dispatch_barrier(encoder);
        temporaries
            .extend(self.encode_softmax_tensors(encoder, table, &scores, &scores, bindings)?);
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
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
            )?;
            let right = self.prepare_matmul_encoder_input(
                encoder,
                table,
                key_head,
                bindings,
                &mut temporaries,
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
                },
                bindings,
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
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
            )?;
            let right = self.prepare_matmul_encoder_input(
                encoder,
                table,
                value_head,
                bindings,
                &mut temporaries,
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
            )?);
            if copy_output {
                encode_dispatch_barrier(encoder);
                temporaries.extend(self.encode_copy_tensors(
                    encoder,
                    table,
                    &prepared_output.tensor,
                    &output_group,
                    bindings,
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
    ) -> Result<MetalBufferRef, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        let params = self.temporary_buffer(&params)?;
        encoder.setComputePipelineState(&pipeline);
        bindings.bind(table, 0, &scores.buffer);
        bindings.bind(table, 1, &params);
        encoder.setArgumentTable(Some(table));
        let count = usize::try_from(scores.layout.element_count())
            .map_err(|_| BackendError::InvalidInput)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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

    fn encode_matmul_kernel(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        left: &PreparedMatmulInput,
        right: &PreparedMatmulInput,
        output: &PreparedMatmulOutput,
        bindings: &mut ArgumentBindings,
    ) -> Result<MetalBufferRef, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        let parameter_buffer = self.temporary_buffer(&params)?;
        let launch = self.matmul_launch(MatmulShape {
            left_dtype: left.tensor.layout.dtype(),
            right_dtype: right.tensor.layout.dtype(),
            output_dtype: dtype,
            batch,
            rows,
            columns,
            inner,
            left_column_major: left.column_major != 0,
            right_column_major: right.column_major != 0,
        })?;
        encoder.setComputePipelineState(&launch.pipeline);
        bindings.bind(table, 0, &left.tensor.buffer);
        bindings.bind(table, 1, &right.tensor.buffer);
        bindings.bind(table, 2, &output.tensor.buffer);
        bindings.bind(table, 3, &parameter_buffer);
        encoder.setArgumentTable(Some(table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        temporaries: &mut Vec<MetalBufferRef>,
    ) -> Result<PreparedMatmulInput, BackendError> {
        let source = self.encoder_tensor(tensor)?;
        self.prepare_matmul_encoder_input(encoder, table, source, bindings, temporaries)
    }

    fn prepare_matmul_encoder_input(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        source: EncoderTensor,
        bindings: &mut ArgumentBindings,
        temporaries: &mut Vec<MetalBufferRef>,
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
        temporaries.extend(self.encode_copy_tensors(encoder, table, &source, &scratch, bindings)?);
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
            (
                "gemv",
                1,
                4,
                32,
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
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        encoder.setComputePipelineState(&pipeline);
        let layouts = vec![
            self.layout_buffer(&input.layout)?,
            self.layout_buffer(&output.layout)?,
        ];
        bindings.bind(table, 0, &input.buffer);
        bindings.bind(table, 1, &output.buffer);
        bindings.bind(table, 2, &layouts[0]);
        bindings.bind(table, 3, &layouts[1]);
        encoder.setArgumentTable(Some(table));
        let count = usize::try_from(output.layout.element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
            buffer: buffers.get(tensor)?.raw.clone(),
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
        Ok(EncoderTensor { buffer, layout })
    }

    #[allow(clippy::cast_precision_loss)]
    fn encode_rope(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        theta: f32,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::MTL4ComputeCommandEncoder;

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
        encoder.setComputePipelineState(&pipeline);
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
            self.layout_buffer(input.layout())?,
            self.layout_buffer(positions.layout())?,
            self.layout_buffer(output.layout())?,
            self.temporary_buffer(&params)?,
            self.temporary_buffer(&frequencies)?,
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [input, positions, output].into_iter().enumerate() {
            bindings.bind(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        bindings.bind(table, 7, &temporaries[4]);
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let pair_count = output
            .layout()
            .element_count()
            .checked_div(2)
            .ok_or(BackendError::InvalidInput)?;
        let thread_count = usize::try_from(pair_count).map_err(|_| BackendError::InvalidInput)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
    ) -> Result<EncodedEmbed, BackendError> {
        use objc2_metal::MTL4ComputeCommandEncoder;

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
        encoder.setComputePipelineState(&pipeline);
        let mut params = [0_u8; 8];
        params[..4].copy_from_slice(&vocab.to_ne_bytes());
        params[4..].copy_from_slice(&width.to_ne_bytes());
        let mut error_state = [0_u8; 8];
        error_state[4..].copy_from_slice(&u32::MAX.to_ne_bytes());
        let error_flag = self.temporary_buffer(&error_state)?;
        let error_address = error_flag.gpuAddress();
        let temporaries = vec![
            self.layout_buffer(embeddings.layout())?,
            self.layout_buffer(ids.layout())?,
            self.layout_buffer(output.layout())?,
            self.temporary_buffer(&params)?,
            error_flag,
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [embeddings, ids, output].into_iter().enumerate() {
            bindings.bind(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        bindings.bind(table, 7, &temporaries[4]);
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let thread_count = usize::try_from(output.layout().element_count())
            .map_err(|_| BackendError::InvalidInput)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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
        Ok((temporaries, error_address))
    }

    fn encode_softmax(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        let [input] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let input = self.encoder_tensor(input)?;
        let output = self.encoder_tensor(dispatch.output())?;
        self.encode_softmax_tensors(encoder, table, &input, &output, bindings)
    }

    fn encode_softmax_tensors(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        input: &EncoderTensor,
        output: &EncoderTensor,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<MetalBufferRef>, BackendError> {
        use objc2_metal::MTL4ComputeCommandEncoder;

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
        encoder.setComputePipelineState(&pipeline);
        let temporaries = vec![
            self.layout_buffer(&input.layout)?,
            self.layout_buffer(&output.layout)?,
            self.temporary_buffer(&width.to_ne_bytes())?,
        ];
        bindings.bind(table, 0, &input.buffer);
        bindings.bind(table, 1, &output.buffer);
        bindings.bind(table, 2, &temporaries[0]);
        bindings.bind(table, 3, &temporaries[1]);
        bindings.bind(table, 4, &temporaries[2]);
        encoder.setArgumentTable(Some(table));
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, &output.layout, width)?;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
        Ok(temporaries)
    }

    fn encode_rms_norm(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        eps: f32,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::MTL4ComputeCommandEncoder;

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
        encoder.setComputePipelineState(&pipeline);
        let mut temporaries = vec![
            self.layout_buffer(input.layout())?,
            self.layout_buffer(weight.layout())?,
            self.layout_buffer(output.layout())?,
        ];
        let mut params = [0_u8; 8];
        params[..4].copy_from_slice(&eps.to_ne_bytes());
        params[4..].copy_from_slice(&width.to_ne_bytes());
        temporaries.push(self.temporary_buffer(&params)?);
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in [input, weight, output].into_iter().enumerate() {
            bindings.bind(table, index, &buffers.get(tensor)?.raw);
            bindings.bind(table, index + 3, &temporaries[index]);
        }
        bindings.bind(table, 6, &temporaries[3]);
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, output.layout(), width)?;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
        Ok(temporaries)
    }

    fn encode_elementwise(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>,
        dispatch: &Dispatch,
        kernel: &str,
        bindings: &mut ArgumentBindings,
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, BackendError> {
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

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
        encoder.setComputePipelineState(&pipeline);
        let layouts = operands
            .iter()
            .map(|tensor| self.layout_buffer(tensor.layout()))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for (index, tensor) in operands.iter().enumerate() {
            let buffer = buffers.get(tensor)?;
            bindings.bind(table, index, &buffer.raw);
            bindings.bind(table, index + operands.len(), &layouts[index]);
        }
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let thread_count = usize::try_from(dispatch.output().layout().element_count())
            .map_err(|_| BackendError::ExecutionFailed)?;
        let group_width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
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

    fn layout_buffer(
        &self,
        layout: &Layout,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, BackendError> {
        let bytes = encode_layout(layout)?;
        self.temporary_buffer(&bytes)
    }

    fn temporary_buffer(
        &self,
        bytes: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, BackendError> {
        use objc2_metal::MTLResourceOptions;

        let buffer = self
            .device
            .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModeShared)
            .ok_or(BackendError::AllocationFailed)?;
        // SAFETY: `buffer` is a live shared allocation of exactly `bytes.len()` bytes and the
        // source and destination do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().cast::<u8>().as_ptr(),
                bytes.len(),
            );
        }
        Ok(buffer)
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

    fn make_timestamps(&self) -> Result<GpuTimestamps, BackendError> {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: Two command-buffer timestamps are written at indices zero and one.
        unsafe {
            descriptor.setCount(2);
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
        })
    }

    fn command_resources(
        &self,
        tensors: &[Tensor],
        encoded: EncodedDispatches,
    ) -> Result<CommandResources, BackendError> {
        let EncodedDispatches {
            temporaries,
            error_flags,
            bindings,
        } = encoded;
        let mut indices = HashMap::<u64, usize>::new();
        let mut owned = Vec::<InFlightBuffer>::new();
        let mut add = |raw: MetalBufferRef| {
            let address = raw.gpuAddress();
            *indices.entry(address).or_insert_with(|| {
                let index = owned.len();
                owned.push(InFlightBuffer { raw });
                index
            })
        };
        {
            let buffers = self
                .buffers
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            for tensor in tensors {
                add(buffers.get(tensor)?.raw.clone());
            }
        }
        for temporary in temporaries {
            add(temporary);
        }
        let error_flags = error_flags
            .iter()
            .map(|address| {
                indices
                    .get(address)
                    .copied()
                    .ok_or(BackendError::ExecutionFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let owned_addresses = indices.keys().copied().collect::<HashSet<_>>();
        debug_assert_eq!(bindings.addresses, owned_addresses);
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
        tensors: &[Tensor],
        event: InFlightEvent,
        resources: CommandResources,
        residency: InFlightResidency,
        timestamps: GpuTimestamps,
    ) -> Result<Arc<Completion>, BackendError> {
        let mut seen = HashSet::<BufferId>::new();
        let unique = tensors
            .iter()
            .filter(|tensor| seen.insert(tensor.buffer()))
            .collect::<Vec<_>>();
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        for tensor in &unique {
            let buffer = buffers.get_mut(tensor)?;
            buffer.wait_pending(self.gpu_timeout)?;
        }
        let completion = Completion::new(
            resources,
            event,
            residency,
            Some(timestamps),
            Arc::downgrade(&self.in_flight),
        );
        for tensor in unique {
            buffers.get_mut(tensor)?.track(&completion);
        }
        Ok(completion)
    }

    fn commit(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        tensors: &[Tensor],
        resources: CommandResources,
        residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
        timestamps: GpuTimestamps,
        objects: &mut SubmissionObjects,
    ) -> Result<MetalSubmission, BackendError> {
        let completion = self.retain_tensors(
            tensors,
            InFlightEvent {
                raw: self.shared_event.clone(),
            },
            resources,
            InFlightResidency { _raw: residency },
            timestamps,
        )?;
        let registration = self
            .in_flight
            .track(&completion, &self.event_listener, objects)?;
        let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = command_buffer;
        let mut command_buffers = [NonNull::from(command_buffer_ref)];
        let Ok(mut next_event_value) = self.next_event_value.lock() else {
            self.in_flight.cancel(&completion);
            return Err(BackendError::ExecutionFailed);
        };
        let event_value = *next_event_value;
        let Some(following_event_value) = event_value.checked_add(1) else {
            drop(next_event_value);
            self.in_flight.cancel(&completion);
            return Err(BackendError::ExecutionFailed);
        };
        completion.mark_committed();
        // SAFETY: The pointer names one live command buffer and the count matches the array.
        unsafe {
            self.queue.commit_count_options(
                NonNull::from(&mut command_buffers[0]),
                command_buffers.len(),
                &completion.commit.options,
            );
        }
        let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &self.shared_event;
        let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
        self.queue.signalEvent_value(event, event_value);
        *next_event_value = following_event_value;
        drop(next_event_value);
        registration.register(event_value);
        Ok(MetalSubmission {
            completion,
            timeout: self.gpu_timeout,
        })
    }
}

fn shape3(layout: &Layout) -> Result<[u32; 3], BackendError> {
    layout
        .shape()
        .try_into()
        .map_err(|_| BackendError::InvalidInput)
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

fn encode_dispatch_barrier(encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>) {
    use objc2_metal::{MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages};

    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
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

const fn dtype_code(dtype: DType) -> u32 {
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
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: HashMap<PipelineKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl PipelineCache {
    pub(super) fn new(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        source: &str,
    ) -> Result<Self, BackendError> {
        let source = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|_| BackendError::ExecutionFailed)?;
        Ok(Self {
            device: device.clone(),
            library,
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
        let name = NSString::from_str(name);
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&name, &values)
            .map_err(|_| BackendError::ExecutionFailed)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.pipelines.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use forja_core::{Backend, CommandList, DType, Op, Slice, Submission, ViewOp};
    use forja_cpu::CpuBackend;
    use forja_testing::{TensorSpec, assert_backends_agree, assert_outputs_agree};

    use super::*;

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
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

        let mut objects = backend.in_flight.checkout(&backend.device).unwrap();
        let command_buffer = backend
            .begin_command_buffer(objects.allocator().unwrap())
            .unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        let table = objects.argument_table().unwrap();
        let source_buffer = backend.encoder_tensor(&source).unwrap();
        let output_buffer = backend.encoder_tensor(&output).unwrap();
        let scratch = backend.scratch_tensor(DType::F32, &[33]).unwrap();
        let mut bindings = ArgumentBindings::default();
        let mut temporaries = vec![scratch.buffer.clone()];
        temporaries.extend(
            backend
                .encode_copy_tensors(&encoder, table, &source_buffer, &scratch, &mut bindings)
                .unwrap(),
        );
        encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        temporaries.extend(
            backend
                .encode_copy_tensors(&encoder, table, &scratch, &output_buffer, &mut bindings)
                .unwrap(),
        );
        encoder.endEncoding();
        let tensors = [source.clone(), output.clone()];
        let resources = backend
            .command_resources(
                &tensors,
                EncodedDispatches {
                    temporaries,
                    error_flags: Vec::new(),
                    bindings,
                },
            )
            .unwrap();
        let residency = backend.make_resident(&command_buffer, &resources).unwrap();
        let timestamps = backend.make_timestamps().unwrap();
        // SAFETY: The heap has two entries and remains live through completion.
        unsafe {
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 0);
            command_buffer.writeTimestampIntoHeap_atIndex(&timestamps.heap, 1);
        }
        command_buffer.endCommandBuffer();
        backend
            .commit(
                &command_buffer,
                &tensors,
                resources,
                residency,
                timestamps,
                &mut objects,
            )
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(backend.read(&output).unwrap(), bytes);
    }

    fn assert_matmul(a: TensorSpec, b: TensorSpec, output: &TensorSpec) {
        assert_backends_agree(
            &CpuBackend::new(),
            &MetalBackend::new().unwrap(),
            Op::Matmul,
            &[a, b],
            output,
        )
        .unwrap();
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
        for m in [7, 128, 512] {
            for (k, n) in [
                (1024, 2048),
                (1024, 1024),
                (2048, 1024),
                (1024, 3072),
                (3072, 1024),
            ] {
                assert_matmul(
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
    fn metal_softmax_matches_cpu_for_dtypes_rows_and_masking() {
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
