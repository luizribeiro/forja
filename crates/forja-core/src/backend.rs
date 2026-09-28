use std::{
    collections::HashMap,
    error::Error,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use crate::{
    BufferId, CommandList, DType, Layout, MappedRegion, Op, Slice, Tensor,
    program::{KernelSignature, ValidatedProgram},
};

static NEXT_BACKEND: AtomicU64 = AtomicU64::new(1);

/// Live allocations owned by one backend instance.
#[derive(Debug)]
pub struct AllocationRegistry<S> {
    backend: u64,
    next_allocation: u64,
    allocations: HashMap<u64, Allocation<S>>,
}

#[derive(Debug)]
struct Allocation<S> {
    storage: S,
    byte_len: u64,
    writable: bool,
}

impl<S> AllocationRegistry<S> {
    /// Creates an empty registry with a unique backend identity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: NEXT_BACKEND.fetch_add(1, Ordering::Relaxed),
            next_allocation: 0,
            allocations: HashMap::new(),
        }
    }

    /// Registers storage and returns its allocation identity.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::AllocationFailed`] if allocation identities are exhausted.
    pub fn insert(&mut self, storage: S, byte_len: u64) -> Result<BufferId, BackendError> {
        self.insert_with_access(storage, byte_len, true)
    }

    /// Registers read-only storage and returns its allocation identity.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::AllocationFailed`] if allocation identities are exhausted.
    pub fn insert_read_only(
        &mut self,
        storage: S,
        byte_len: u64,
    ) -> Result<BufferId, BackendError> {
        self.insert_with_access(storage, byte_len, false)
    }

    fn insert_with_access(
        &mut self,
        storage: S,
        byte_len: u64,
        writable: bool,
    ) -> Result<BufferId, BackendError> {
        let allocation = self.next_allocation;
        self.next_allocation = allocation
            .checked_add(1)
            .ok_or(BackendError::AllocationFailed)?;
        self.allocations.insert(
            allocation,
            Allocation {
                storage,
                byte_len,
                writable,
            },
        );
        Ok(BufferId::new(self.backend, allocation, byte_len))
    }

    /// Creates a tensor for a registered allocation and validated layout.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] for a foreign, released, or length-mismatched
    /// allocation.
    pub fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
        let allocation = self.validate_buffer(buffer, layout.buffer_len())?;
        Tensor::from_allocation(buffer, layout, allocation.writable)
            .map_err(|_| BackendError::InvalidInput)
    }

    /// Creates a tensor view after validating its source allocation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] for a foreign, released, or length-mismatched
    /// tensor or layout.
    pub fn view(&self, tensor: &Tensor, layout: Layout) -> Result<Tensor, BackendError> {
        let allocation = self.validate(tensor)?;
        Tensor::from_allocation(tensor.buffer(), layout, allocation.writable)
            .map_err(|_| BackendError::InvalidInput)
    }

    /// Returns storage after validating that the tensor names this live allocation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] for a foreign, released, or length-mismatched tensor.
    pub fn get(&self, tensor: &Tensor) -> Result<&S, BackendError> {
        let allocation = self.validate(tensor)?;
        Ok(&allocation.storage)
    }

    /// Returns mutable storage after validating that the tensor names this live allocation.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] for a foreign, released, or length-mismatched tensor.
    pub fn get_mut(&mut self, tensor: &Tensor) -> Result<&mut S, BackendError> {
        self.validate(tensor)?;
        self.allocations
            .get_mut(&tensor.buffer().allocation())
            .map(|allocation| &mut allocation.storage)
            .ok_or(BackendError::InvalidInput)
    }

    /// Removes and returns storage after validating its tensor identity.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidInput`] for a foreign, released, or length-mismatched tensor.
    pub fn remove(&mut self, tensor: &Tensor) -> Result<S, BackendError> {
        self.validate(tensor)?;
        self.allocations
            .remove(&tensor.buffer().allocation())
            .map(|allocation| allocation.storage)
            .ok_or(BackendError::InvalidInput)
    }

    fn validate(&self, tensor: &Tensor) -> Result<&Allocation<S>, BackendError> {
        let allocation = self.validate_buffer(tensor.buffer(), tensor.layout().buffer_len())?;
        if allocation.writable != tensor.is_writable() {
            return Err(BackendError::InvalidInput);
        }
        Ok(allocation)
    }

    fn validate_buffer(
        &self,
        id: BufferId,
        layout_buffer_len: u64,
    ) -> Result<&Allocation<S>, BackendError> {
        if id.backend() != self.backend {
            return Err(BackendError::InvalidInput);
        }
        let allocation = self
            .allocations
            .get(&id.allocation())
            .ok_or(BackendError::InvalidInput)?;
        if allocation.byte_len != id.byte_len() || allocation.byte_len != layout_buffer_len {
            return Err(BackendError::InvalidInput);
        }
        Ok(allocation)
    }
}

impl<S> Default for AllocationRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A metadata-only transformation applied to a tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ViewOp {
    /// Selects elements independently along every axis.
    Slice(Vec<Slice>),
    /// Changes the shape of a contiguous tensor.
    Reshape(Vec<u32>),
    /// Reorders every axis.
    Permute(Vec<u8>),
    /// Expands size-one axes using zero strides.
    Broadcast(Vec<u32>),
}

/// A backend operation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendError {
    /// A configured memory quota would be exceeded.
    QuotaExceeded,
    /// Host or device memory could not be allocated.
    AllocationFailed,
    /// Validated work failed while executing.
    ExecutionFailed,
    /// Submitted work did not complete before its configured deadline.
    Timeout,
    /// An argument does not belong to this backend or violates an API rule.
    InvalidInput,
    /// The backend does not implement the requested operation.
    UnsupportedOperation,
    /// A data-dependent read used an index outside its source extent.
    IndexOutOfRange {
        /// The first invalid index value.
        index: u32,
    },
}

/// Count and wall time for one class of profiling event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProfileCount {
    /// Number of events.
    pub count: u64,
    /// Total host wall time spent in the events.
    pub time: Duration,
}

/// Device time attributed to one recorded dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispatchProfile {
    /// Operation recorded by the command list.
    pub op: Op,
    /// Elapsed device time surrounding that operation.
    pub gpu_time: Duration,
}

/// Host phases and device timings from one profiled submission.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubmissionProfile {
    /// Scalar-program dispatch recording on the submitting thread.
    pub program_recording: ProfileCount,
    /// Scalar-program dispatch encoding on the submitting thread.
    pub program_encoding: ProfileCount,
    /// Resident program compilations skipped because no compile token was available.
    pub program_compile_fallbacks: u64,
    /// Command validation and hazard analysis.
    pub validation: Duration,
    /// Temporary Metal buffers created while encoding.
    pub metadata_buffers: ProfileCount,
    /// Residency-set construction.
    pub residency: Duration,
    /// Command encoding excluding temporary-buffer creation.
    pub encoding: Duration,
    /// Queue commit and in-flight resource registration.
    pub commit: Duration,
    /// Host time waiting for completion.
    pub wait: Duration,
    /// Device time for the complete submission.
    pub gpu_time: Duration,
    /// Number of encoded dispatches.
    pub dispatches: u64,
    /// Number of inserted dependency barriers.
    pub barriers: u64,
    /// Device timings for individual dispatches.
    pub per_dispatch: Vec<DispatchProfile>,
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "backend error: {self:?}")
    }
}

impl Error for BackendError {}

/// Completion state returned by a backend submission.
pub trait Submission {
    /// Waits for execution and returns the command list result.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] if execution failed.
    fn wait(&self) -> Result<(), BackendError>;

    /// Waits for execution up to a caller-provided deadline.
    ///
    /// Backends without asynchronous device work may ignore the deadline.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Timeout`] when work remains incomplete at the deadline, or an
    /// execution error when completed work failed.
    fn wait_timeout(&self, timeout: Duration) -> Result<(), BackendError> {
        let _ = timeout;
        self.wait()
    }

    /// Returns elapsed device execution time after a successful wait.
    ///
    /// GPU backends return device timestamps. Backends without device timestamps may return the
    /// wall time spent executing the submission or `None`.
    fn gpu_time(&self) -> Option<Duration>;

    /// Returns profiling detail after a successful wait, when requested at submission.
    fn profile(&self) -> Option<SubmissionProfile> {
        None
    }
}

/// Storage and execution implemented by every trusted backend.
pub trait Backend {
    /// The completion handle produced by this backend.
    type Submission: Submission;
    /// Backend state retained by a prepared scalar program.
    type ProgramHandle: Send + Sync + 'static;

    /// Prepares a validated scalar program for repeated dispatch.
    ///
    /// # Errors
    ///
    /// Returns a quota, compilation, or unsupported-operation error.
    fn prepare_program(
        &self,
        program: &ValidatedProgram,
        signature: &KernelSignature,
    ) -> Result<Self::ProgramHandle, BackendError>;

    /// Allocates a contiguous tensor initialized to zero.
    ///
    /// # Errors
    ///
    /// Returns a quota, allocation, or invalid-input error.
    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError>;
    /// Imports one read-only file mapping as a backend allocation.
    ///
    /// # Errors
    ///
    /// Returns an allocation or invalid-input error when the mapping cannot be imported.
    fn import_readonly(&self, bytes: MappedRegion) -> Result<BufferId, BackendError> {
        let _ = bytes;
        Err(BackendError::InvalidInput)
    }
    /// Creates a validated tensor for an imported allocation.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a foreign allocation or mismatched layout.
    fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
        let _ = (buffer, layout);
        Err(BackendError::InvalidInput)
    }
    /// Applies a validated metadata-only view operation.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a foreign tensor or invalid view.
    fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError>;
    /// Writes contiguous logical tensor bytes.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a foreign, non-contiguous, or size-mismatched tensor.
    fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError>;
    /// Gathers a tensor view into contiguous logical bytes.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a foreign tensor.
    fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError>;
    /// Releases the allocation underlying a tensor and all of its views.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a foreign or already released tensor.
    fn release(&self, tensor: &Tensor) -> Result<(), BackendError>;
    /// Submits a validated command list.
    ///
    /// # Errors
    ///
    /// Returns invalid input if any tensor belongs to another backend.
    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError>;

    /// Submits work with detailed timing enabled.
    ///
    /// Backends without detailed instrumentation use the normal submission path.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Backend::submit`].
    fn submit_profiled(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        self.submit(commands)
    }
}
