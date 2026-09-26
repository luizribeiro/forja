//! Build-time access to WebAssembly components used by host-side tests.
//! Guest artifacts are kept outside the native workspace build graph.

use std::path::Path;

/// Returns the path to the guest that exports a greeting function.
#[must_use]
pub fn hello() -> &'static Path {
    Path::new(env!("HELLO_COMPONENT"))
}
