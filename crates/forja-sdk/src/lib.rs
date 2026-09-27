//! Tensors and neural-network helpers for Forja inference engines.
//!
//! Operations return [`Result`] so host validation and execution failures can
//! be propagated with `?`. Engine code stays synchronous even though the host
//! reads asynchronously.

#![forbid(unsafe_code)]

mod element;
mod sys;
mod tensor;

use std::{error, fmt};

pub use element::Element;
pub use half::{bf16, f16};
pub use tensor::{Slice, Tensor};

/// A result returned by the guest tensor API.
pub type Result<T> = std::result::Result<T, Error>;

/// A host or tensor-validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(String);

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}
