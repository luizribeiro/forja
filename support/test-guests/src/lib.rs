//! Build-time access to WebAssembly components used by host-side tests.
//! Guest artifacts are kept outside the native workspace build graph.

use std::path::Path;

/// Returns the path to the guest that exports a greeting function.
#[must_use]
pub fn hello() -> &'static Path {
    Path::new(env!("HELLO_COMPONENT"))
}

/// Returns the path to the guest that submits normalization and activation operations.
#[must_use]
pub fn rmsnorm_smoke() -> &'static Path {
    Path::new(env!("RMSNORM_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that exercises the tensor interface.
#[must_use]
pub fn tensor_smoke() -> &'static Path {
    Path::new(env!("TENSOR_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that probes tensor boundary failures.
#[must_use]
pub fn tensor_abuse() -> &'static Path {
    Path::new(env!("TENSOR_ABUSE_COMPONENT"))
}

/// Returns the path to the guest that consumes host-granted weights.
#[must_use]
pub fn weights_smoke() -> &'static Path {
    Path::new(env!("WEIGHTS_SMOKE_COMPONENT"))
}

#[cfg(test)]
mod tests {
    use wasmtime::component::{Component, Linker, ResourceTable};
    use wasmtime::{Engine, Store};
    use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

    wasmtime::component::bindgen!({
        path: "../guests/hello/wit",
        world: "hello",
    });

    struct State {
        table: ResourceTable,
        wasi: WasiCtx,
    }

    impl WasiView for State {
        fn ctx(&mut self) -> WasiCtxView<'_> {
            WasiCtxView {
                ctx: &mut self.wasi,
                table: &mut self.table,
            }
        }
    }

    #[test]
    fn greets_name() -> wasmtime::Result<()> {
        let engine = Engine::default();
        let component = Component::from_file(&engine, super::hello())?;
        let mut linker = Linker::new(&engine);
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
        let mut store = Store::new(
            &engine,
            State {
                table: ResourceTable::new(),
                wasi: WasiCtxBuilder::new().build(),
            },
        );
        let hello = Hello::instantiate(&mut store, &component, &linker)?;

        let greeting = hello.call_greet(&mut store, "Ada")?;

        assert_eq!(greeting, "Hello, Ada!");
        Ok(())
    }
}
