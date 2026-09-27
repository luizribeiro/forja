//! Trusted host integration for running Forja guest components.

#![allow(clippy::manual_async_fn)]

/// Host bindings for the guest-facing compute interface.
#[allow(missing_docs)]
pub mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "host",
        imports: { default: async | trappable },
        require_store_data_send: true,
        with: {
            "l9o:gpu/compute.tensor": crate::TensorEntry,
        },
    });
}

/// Host-owned state behind a guest tensor resource.
#[derive(Debug)]
pub struct TensorEntry;
