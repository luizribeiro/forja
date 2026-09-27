//! Tensors and neural-network helpers for Forja inference engines.

#![forbid(unsafe_code)]

mod element;
mod sys;

use std::{error, fmt};

pub use element::Element;
pub use half::{bf16, f16};

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
