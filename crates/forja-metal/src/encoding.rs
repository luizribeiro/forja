use std::{
    collections::{HashMap, HashSet},
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Condvar, Mutex, Weak},
    time::{Duration, Instant},
};

use block2::RcBlock;
use forja_core::{
    BackendError, BufferId, CommandList, DType, Dispatch, Layout, Op, Submission, Tensor,
    required_barriers,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandBuffer, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTLAllocation, MTLBuffer, MTLComputePipelineState, MTLDataType, MTLDevice, MTLEvent,
    MTLFunctionConstantValues, MTLLibrary, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent, MTLSharedEventListener, MTLSize,
};

use crate::{
    matmul::{classify, select_gemm},
    storage::MetalBackend,
};

type MetalBufferRef = Retained<ProtocolObject<dyn MTLBuffer>>;
type EncodedEmbed = (Vec<MetalBufferRef>, u64);

struct EncoderTensor {
    buffer: MetalBufferRef,
    layout: Layout,
}

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
    result: Option<Result<(), BackendError>>,
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
    ) -> Arc<Self> {
        Arc::new_cyclic(|completion: &Weak<Self>| {
            let callback_completion = completion.clone();
            let handler: FeedbackHandler = RcBlock::new(
                move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    // SAFETY: Metal supplies a live, non-null feedback object for this call.
                    let feedback = unsafe { feedback.as_ref() };
                    let result = if feedback.error().is_some() {
                        Err(BackendError::ExecutionFailed)
                    } else {
                        Ok(())
                    };
                    if let Some(completion) = callback_completion.upgrade() {
                        completion.finish(result);
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
                state: Mutex::new(CompletionState { result: None }),
                ready: Condvar::new(),
                event,
                resources,
                _residency: residency,
                commit: CommitRetention {
                    _handler: handler,
                    options,
                },
            }
        })
    }

    pub(super) fn finish(&self, result: Result<(), BackendError>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.result = Some(result);
        self.ready.notify_all();
    }

    pub(super) fn wait(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        if !self
            .event
            .raw
            .waitUntilSignaledValue_timeoutMS(1, timeout_millis(timeout))
        {
            return Err(BackendError::ExecutionFailed);
        }
        self.wait_for_feedback(timeout.saturating_sub(started.elapsed()))?;
        self.check_error_flags()
    }

    fn wait_unbounded(&self) {
        let _signaled = self.event.raw.waitUntilSignaledValue_timeoutMS(1, u64::MAX);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn wait_for_feedback(&self, timeout: Duration) -> Result<(), BackendError> {
        let started = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(result) = state.result {
                return result;
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(BackendError::ExecutionFailed);
            }
            let (next, wait) = self
                .ready
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            if wait.timed_out() && state.result.is_none() {
                return Err(BackendError::ExecutionFailed);
            }
        }
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
}

type NotificationHandler = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;

struct InFlightCompletion {
    completion: Arc<Completion>,
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
}

impl InFlightTracker {
    pub(super) const fn new() -> Self {
        Self {
            completions: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn track(
        self: &Arc<Self>,
        completion: &Arc<Completion>,
        listener: &Retained<MTLSharedEventListener>,
    ) -> Result<(), BackendError> {
        let pending = Arc::clone(completion);
        let owner = Arc::clone(self);
        let notification: NotificationHandler = RcBlock::new(move |_event, _value| {
            let active_completion = Arc::clone(&pending);
            let active_owner = Arc::clone(&owner);
            active_completion.wait_unbounded();
            active_owner.remove(&active_completion);
        });
        self.completions
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .push(InFlightCompletion {
                completion: Arc::clone(completion),
                _listener: listener.clone(),
                _notification: notification.clone(),
            });
        // SAFETY: The tracker stores the live heap block and listener before registration. The
        // block owns `Arc`s to both the completion and tracker, and clones them before removal, so
        // its captures remain valid even if the backend is dropped before Metal invokes it.
        unsafe {
            completion.event.raw.notifyListener_atValue_block(
                listener,
                1,
                RcBlock::as_ptr(&notification),
            );
        }
        Ok(())
    }

    fn remove(&self, completion: &Arc<Completion>) {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|candidate| !Arc::ptr_eq(&candidate.completion, completion));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.completions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

fn timeout_millis(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX)
}

/// Completion state for a Metal command buffer.
pub struct MetalSubmission {
    completion: Arc<Completion>,
    timeout: Duration,
}

impl Submission for MetalSubmission {
    fn wait(self) -> Result<(), BackendError> {
        self.completion.wait(self.timeout)
    }
}

impl MetalBackend {
    pub(super) fn submit_commands(
        &self,
        commands: CommandList,
    ) -> Result<MetalSubmission, BackendError> {
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
        let command_buffer = self.begin_command_buffer()?;
        let encoded = self.encode_dispatches(&command_buffer, &dispatches, &barriers)?;
        let resources = self.command_resources(&tensors, encoded)?;
        let residency = self.make_resident(&command_buffer, &resources)?;
        command_buffer.endCommandBuffer();
        self.commit(&command_buffer, &tensors, resources, residency)
    }

    fn encode_dispatches(
        &self,
        command_buffer: &ProtocolObject<dyn MTL4CommandBuffer>,
        dispatches: &[Dispatch],
        barriers: &[bool],
    ) -> Result<EncodedDispatches, BackendError> {
        use objc2_metal::{
            MTL4ArgumentTableDescriptor, MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages,
        };

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(BackendError::ExecutionFailed)?;
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(8);
        let table = self
            .device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|_| BackendError::ExecutionFailed)?;
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
                    &table,
                    dispatch,
                    eps,
                    &mut bindings,
                )?);
                continue;
            }
            if dispatch.op() == Op::Softmax {
                temporaries.extend(self.encode_softmax(
                    &encoder,
                    &table,
                    dispatch,
                    &mut bindings,
                )?);
                continue;
            }
            if let Op::Rope { theta } = dispatch.op() {
                temporaries.extend(self.encode_rope(
                    &encoder,
                    &table,
                    dispatch,
                    theta,
                    &mut bindings,
                )?);
                continue;
            }
            if dispatch.op() == Op::Embed {
                let (buffers, flag) =
                    self.encode_embed(&encoder, &table, dispatch, &mut bindings)?;
                temporaries.extend(buffers);
                error_flags.push(flag);
                continue;
            }
            if dispatch.op() == Op::Copy {
                temporaries.extend(self.encode_copy(&encoder, &table, dispatch, &mut bindings)?);
                continue;
            }
            if dispatch.op() == Op::Matmul {
                temporaries.extend(self.encode_matmul(
                    &encoder,
                    &table,
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
                &table,
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
        use objc2_metal::{MTL4ComputeCommandEncoder, MTLSize};

        let [left, right] = dispatch.inputs() else {
            return Err(BackendError::InvalidInput);
        };
        let output = dispatch.output();
        let dtype = output.layout().dtype();
        if left.layout().dtype() != dtype || right.layout().dtype() != dtype {
            return Err(BackendError::InvalidInput);
        }
        let (a_column_major, lda, batch_stride_a) = classify(left.layout())
            .kernel_strides()
            .ok_or(BackendError::InvalidInput)?;
        let (b_column_major, ldb, batch_stride_b) = classify(right.layout())
            .kernel_strides()
            .ok_or(BackendError::InvalidInput)?;
        let (output_column_major, ldd, batch_stride_d) = classify(output.layout())
            .kernel_strides()
            .ok_or(BackendError::InvalidInput)?;
        if output_column_major != 0 {
            return Err(BackendError::InvalidInput);
        }
        let shape = left.layout().shape();
        let rank = shape.len();
        let rows = shape[rank - 2];
        let inner = shape[rank - 1];
        let columns = right.layout().shape()[rank - 1];
        let batch = if rank == 3 { shape[0] } else { 1 };
        let mut params = Vec::with_capacity(96);
        for value in [
            left.layout().offset(),
            right.layout().offset(),
            output.layout().offset(),
            lda,
            ldb,
            ldd,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d,
        ] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        for value in [rows, columns, inner, a_column_major, b_column_major, 0] {
            params.extend_from_slice(&value.to_ne_bytes());
        }
        let parameter_buffer = self.temporary_buffer(&params)?;
        let config = select_gemm(
            dtype,
            batch,
            rows,
            columns,
            inner,
            a_column_major != 0,
            b_column_major != 0,
        )
        .ok_or(BackendError::InvalidInput)?;
        let constants = [
            (0, dtype_code(dtype)),
            (1, dtype_code(dtype)),
            (2, dtype_code(dtype)),
            (200, u32::from(rows.is_multiple_of(config.block_rows))),
            (201, u32::from(columns.is_multiple_of(config.block_columns))),
            (202, u32::from(inner.is_multiple_of(config.block_inner))),
        ];
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(config.kernel, &constants)?;
        encoder.setComputePipelineState(&pipeline);
        let left = self.encoder_tensor(left)?;
        let right = self.encoder_tensor(right)?;
        let output = self.encoder_tensor(output)?;
        bindings.bind(table, 0, &left.buffer);
        bindings.bind(table, 1, &right.buffer);
        bindings.bind(table, 2, &output.buffer);
        bindings.bind(table, 3, &parameter_buffer);
        encoder.setArgumentTable(Some(table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: usize::try_from(columns.div_ceil(config.block_columns))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(rows.div_ceil(config.block_rows))
                    .map_err(|_| BackendError::InvalidInput)?,
                depth: usize::try_from(batch).map_err(|_| BackendError::InvalidInput)?,
            },
            MTLSize {
                width: config.thread_count,
                height: 1,
                depth: 1,
            },
        );
        Ok(vec![parameter_buffer])
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

    #[allow(dead_code)]
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
        use objc2_metal::MTL4ComputeCommandEncoder;

        let [input] = dispatch.inputs() else {
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
            (2, dtype_code(output.layout().dtype())),
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
            self.layout_buffer(input.layout())?,
            self.layout_buffer(output.layout())?,
            self.temporary_buffer(&width.to_ne_bytes())?,
        ];
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        bindings.bind(table, 0, &buffers.get(input)?.raw);
        bindings.bind(table, 1, &buffers.get(output)?.raw);
        bindings.bind(table, 2, &temporaries[0]);
        bindings.bind(table, 3, &temporaries[1]);
        bindings.bind(table, 4, &temporaries[2]);
        drop(buffers);
        encoder.setArgumentTable(Some(table));
        let (threadgroups, threads) = row_dispatch_geometry(&pipeline, output.layout(), width)?;
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
    ) -> Result<Retained<ProtocolObject<dyn MTL4CommandBuffer>>, BackendError> {
        let allocator = self
            .device
            .newCommandAllocator()
            .ok_or(BackendError::ExecutionFailed)?;
        let command_buffer = self
            .device
            .newCommandBuffer()
            .ok_or(BackendError::ExecutionFailed)?;
        command_buffer.beginCommandBufferWithAllocator(&allocator);
        Ok(command_buffer)
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
        let completion = Completion::new(resources, event, residency);
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
    ) -> Result<MetalSubmission, BackendError> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or(BackendError::ExecutionFailed)?;
        let completion = self.retain_tensors(
            tensors,
            InFlightEvent { raw: event.clone() },
            resources,
            InFlightResidency { _raw: residency },
        )?;
        self.in_flight.track(&completion, &self.event_listener)?;
        let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = command_buffer;
        let mut command_buffers = [NonNull::from(command_buffer_ref)];
        // SAFETY: The pointer names one live command buffer and the count matches the array.
        unsafe {
            self.queue.commit_count_options(
                NonNull::from(&mut command_buffers[0]),
                command_buffers.len(),
                &completion.commit.options,
            );
        }
        let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &event;
        let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
        self.queue.signalEvent_value(event, 1);
        Ok(MetalSubmission {
            completion,
            timeout: self.gpu_timeout,
        })
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
    use std::time::{Duration, Instant};

    use forja_core::{Backend, CommandList, DType, Op, Slice, Submission};
    use forja_cpu::CpuBackend;
    use forja_testing::{TensorSpec, assert_backends_agree};
    use objc2_metal::{MTL4CommandQueue, MTLEvent};

    use super::*;

    #[test]
    fn metal_compile_error_is_reported() {
        let backend = MetalBackend::new().unwrap();
        assert!(PipelineCache::new(&backend.device, "kernel void broken(").is_err());
    }

    #[test]
    fn shared_event_listener_reaps_many_command_buffers() {
        let backend = MetalBackend::new().unwrap();
        let queue = backend.device.newMTL4CommandQueue().unwrap();
        let descriptor = MTLResidencySetDescriptor::new();
        let residency = backend
            .device
            .newResidencySetWithDescriptor_error(&descriptor)
            .unwrap();
        residency.commit();

        for _ in 0..1000 {
            let command_buffer = backend.begin_command_buffer().unwrap();
            command_buffer.endCommandBuffer();
            let event = backend.device.newSharedEvent().unwrap();
            let completion = Completion::new(
                CommandResources {
                    buffers: Vec::new(),
                    error_flags: Vec::new(),
                },
                InFlightEvent { raw: event.clone() },
                InFlightResidency {
                    _raw: residency.clone(),
                },
            );
            completion.finish(Ok(()));
            backend
                .in_flight
                .track(&completion, &backend.event_listener)
                .unwrap();
            let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = &command_buffer;
            let mut command_buffers = [NonNull::from(command_buffer_ref)];
            // SAFETY: The pointer names one live command buffer and the count matches the array.
            unsafe {
                queue.commit_count(
                    NonNull::from(&mut command_buffers[0]),
                    command_buffers.len(),
                );
            }
            let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &event;
            let event: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();
            queue.signalEvent_value(event, 1);
        }

        let started = Instant::now();
        while backend.in_flight.len() != 0 {
            assert!(started.elapsed() < backend.gpu_timeout);
            std::thread::yield_now();
        }
    }

    #[test]
    fn metal_empty_command_list_completes() {
        let backend = MetalBackend::new().unwrap();
        backend.submit(CommandList::new()).unwrap().wait().unwrap();
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
            Err(BackendError::ExecutionFailed)
        );
        match backend.read(&output) {
            Ok(bytes) => assert_eq!(bytes.len(), 1024 * 4097 * 4),
            Err(error) => assert_eq!(error, BackendError::ExecutionFailed),
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
        use objc2_metal::{
            MTL4ArgumentTableDescriptor, MTL4CommandEncoder, MTL4VisibilityOptions, MTLStages,
        };

        let backend = MetalBackend::new().unwrap();
        let source = backend.alloc(DType::F32, &[33]).unwrap();
        let output = backend.alloc(DType::F32, &[33]).unwrap();
        let bytes = (0_u16..33)
            .map(f32::from)
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        backend.write(&source, &bytes).unwrap();

        let command_buffer = backend.begin_command_buffer().unwrap();
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(4);
        let table = backend
            .device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .unwrap();
        let source_buffer = backend.encoder_tensor(&source).unwrap();
        let output_buffer = backend.encoder_tensor(&output).unwrap();
        let scratch = backend.scratch_tensor(DType::F32, &[33]).unwrap();
        let mut bindings = ArgumentBindings::default();
        let mut temporaries = vec![scratch.buffer.clone()];
        temporaries.extend(
            backend
                .encode_copy_tensors(&encoder, &table, &source_buffer, &scratch, &mut bindings)
                .unwrap(),
        );
        encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        temporaries.extend(
            backend
                .encode_copy_tensors(&encoder, &table, &scratch, &output_buffer, &mut bindings)
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
        command_buffer.endCommandBuffer();
        backend
            .commit(&command_buffer, &tensors, resources, residency)
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

    #[test]
    fn metal_gemm_matches_cpu_at_tile_edges_and_in_batches() {
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
