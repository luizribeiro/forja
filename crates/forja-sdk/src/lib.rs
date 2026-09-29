//! Tensors and neural-network helpers for Forja inference engines.
//!
//! Operations return [`Result`] so host validation and execution failures can
//! be propagated with `?`. Engine code stays synchronous even though the host
//! reads and submissions are asynchronous.
//!
//! Operations are lazy: they allocate outputs and record work in the current
//! thread's graph. [`eval`] submits explicitly, while [`Tensor::to_vec`] submits
//! automatically before reading.
//!
//! ```no_run
//! use forja_sdk::{
//!     Engine, EngineInfo, Load, Result, StepInput, StepOutput, Weights, export_engine,
//!     nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
//! };
//!
//! const HIDDEN: u32 = 1024;
//! const VOCAB: u32 = 32_000;
//!
//! #[derive(Clone, Copy)]
//! struct Config;
//!
//! #[derive(Load)]
//! #[load(config = Config)]
//! struct Block {
//!     #[load(prefix, config = RmsNormConfig::new(HIDDEN, 1.0e-6))]
//!     norm: RmsNorm<f32>,
//!     #[load(prefix, config = LinearConfig::new(HIDDEN, HIDDEN))]
//!     projection: Linear<f32>,
//! }
//!
//! #[derive(Load)]
//! #[load(config = Config)]
//! struct Llama {
//!     #[load(prefix, config = EmbeddingConfig::new(VOCAB, HIDDEN))]
//!     embed: Embedding<f32>,
//!     #[load(prefix, count = 2)]
//!     blocks: Vec<Block>,
//!     #[load(prefix, config = RmsNormConfig::new(HIDDEN, 1.0e-6))]
//!     norm: RmsNorm<f32>,
//!     #[load(prefix, config = LinearConfig::new(HIDDEN, VOCAB))]
//!     lm_head: Linear<f32>,
//! }
//!
//! #[export_engine]
//! impl Engine for Llama {
//!     fn describe() -> EngineInfo {
//!         EngineInfo { vocab: VOCAB, max_context: 4096, tap_layers: vec![] }
//!     }
//!
//!     fn load(weights: &Weights<'_>) -> Result<Self> {
//!         <Self as Load<Config>>::load(weights, &Config)
//!     }
//!
//!     fn step(&mut self, input: StepInput) -> Result<StepOutput> {
//!         let last = input.tokens.shape().first().copied()
//!             .ok_or_else(|| forja_sdk::Error::loading("tokens must have rank one"))?
//!             .checked_sub(1)
//!             .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?;
//!         let mut hidden = self.embed.forward(&input.tokens)?;
//!         for block in &self.blocks {
//!             let projected = block.projection.forward(&block.norm.forward(&hidden)?)?;
//!             hidden = (&hidden + &projected)?;
//!         }
//!         let logits = self.lm_head.forward(&self.norm.forward(&hidden)?)?
//!             .narrow(0, last, 1)?
//!             .reshape(&[VOCAB])?;
//!         Ok(StepOutput { logits, taps: vec![] })
//!     }
//! }
//! ```
//!
//! Custom kernels use a restricted Rust expression subset. [`kernel::Elem`]
//! defines elementwise work, [`kernel::Row`] permits reductions over the last
//! axis, and helpers are inlined while the program is built.
//! Mutual recursion between helpers is not detected and overflows the build.
//!
//! ```no_run
//! use forja_sdk::{kernel, kernel::Row};
//!
//! #[kernel(helper)]
//! fn square(x: f32) -> f32 {
//!     x * x
//! }
//!
//! #[kernel(row)]
//! fn rms_norm(x: Row, weight: Row, epsilon: f32) -> Row {
//!     x * (square(x).row_mean() + epsilon).rsqrt() * weight
//! }
//! ```
//!
//! The `native` feature uses the CPU backend for tests and development.
//! `native-metal` makes Metal selectable with [`set_native_device`] on macOS.
//! Production engines must ship as components because native mode bypasses the
//! WebAssembly sandbox.

#![forbid(unsafe_code)]

mod element;
mod engine;
mod graph;
pub mod kernel;
mod load;
pub mod nn;
pub mod program;
mod sys;
mod tensor;

use std::{error, fmt};

pub use element::{Element, FloatElement};
pub use engine::{DecodeInput, DecodeOutput, Engine, EngineInfo, StepInput, StepOutput};
pub use forja_sdk_macros::{Load, export_engine, kernel};
pub use graph::{Dim, Graph, Param, Pos, capture, eval};
pub use half::{bf16, f16};
pub use load::{Load, Weights};
pub use sys::DType;
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
    graph::clear_kernel_cache();
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

    #[doc(hidden)]
    pub fn loading(message: impl Into<String>) -> Self {
        Self::new(message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}

#[cfg(target_family = "wasm")]
#[doc(hidden)]
pub mod __private {
    pub use crate::sys::guest::compute;
    pub use wit_bindgen;
}
