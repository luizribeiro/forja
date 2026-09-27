//! Tensors and neural-network helpers for Forja inference engines.
//!
//! Operations return [`Result`] so host validation and execution failures can
//! be propagated with `?`. Engine code stays synchronous even though the host
//! reads and submissions are asynchronous.
//!
//! The `native` feature uses the CPU backend for tests and development.
//! `native-metal` makes Metal selectable with [`set_native_device`] on macOS.
//! Production engines must ship as components because native mode bypasses the
//! WebAssembly sandbox.

#![forbid(unsafe_code)]

mod element;
mod graph;
mod sys;
mod tensor;

use std::{error, fmt};

pub use element::Element;
pub use graph::eval;
pub use half::{bf16, f16};
pub use tensor::{Slice, Tensor};

/// An in-process backend selected for the current thread.
#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDevice {
    /// The reference CPU backend.
    Cpu,
    /// The Metal backend.
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    Metal,
}

/// Selects the in-process backend used for tensors subsequently created on this thread.
///
/// Select the device before creating tensors. Tensors from different devices cannot
/// participate in one operation.
#[cfg(feature = "native")]
pub fn set_native_device(device: NativeDevice) {
    sys::set_native_device(device);
}

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
