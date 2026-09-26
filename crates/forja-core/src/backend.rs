use std::{error::Error, fmt};

use crate::{CommandList, DType, Slice, Tensor};

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
    /// An argument does not belong to this backend or violates an API rule.
    InvalidInput,
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
    fn wait(self) -> Result<(), BackendError>;
}

/// Storage and execution implemented by every trusted backend.
pub trait Backend {
    /// The completion handle produced by this backend.
    type Submission: Submission;

    /// Allocates a contiguous tensor initialized to zero.
    ///
    /// # Errors
    ///
    /// Returns a quota, allocation, or invalid-input error.
    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError>;
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
    /// Submits a validated command list.
    ///
    /// # Errors
    ///
    /// Returns invalid input if any tensor belongs to another backend.
    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError>;
}
