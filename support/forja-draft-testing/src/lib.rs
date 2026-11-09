//! Dev-only deterministic sources, CPU targets and exact draft-stream assertions.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod draft;
mod target;
pub use draft::{RequestOutput, TestDrafter, Trace, assert_draft_exact};
pub use target::Target;
