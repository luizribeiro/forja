//! Build-time access to WebAssembly components used by host-side tests.
//! Guest artifacts are kept outside the native workspace build graph.

use std::path::Path;

/// Returns the path to the zero-import chat-format test component.
#[must_use]
pub fn chat_format_smoke() -> &'static Path {
    Path::new(env!("CHAT_FORMAT_SMOKE_COMPONENT"))
}

/// Returns the path to the Qwen3-Coder chat-format component.
#[must_use]
pub fn qwen3_coder_chat_format() -> &'static Path {
    Path::new(env!("QWEN3_CODER_CHAT_FORMAT"))
}

/// Returns the path to the custom chat-format component.
#[must_use]
pub fn custom_chat_format() -> &'static Path {
    Path::new(env!("CUSTOM_CHAT_FORMAT"))
}

/// Returns the path to the chat format used by continuous-serving tests.
#[must_use]
pub fn continuous_chat_format() -> &'static Path {
    Path::new(env!("CONTINUOUS_CHAT_FORMAT"))
}

/// Returns the path to the minimal engine component.
#[must_use]
pub fn engine_smoke() -> &'static Path {
    Path::new(env!("ENGINE_SMOKE_COMPONENT"))
}

/// Returns the path to the SDK-exported engine component.
#[must_use]
pub fn engine_sdk_smoke() -> &'static Path {
    Path::new(env!("ENGINE_SDK_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that exports a greeting function.
#[must_use]
pub fn hello() -> &'static Path {
    Path::new(env!("HELLO_COMPONENT"))
}

/// Returns the path to a component importing the optional Metal variant interface.
#[must_use]
pub fn metal_variant_smoke() -> &'static Path {
    Path::new(env!("METAL_VARIANT_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that submits normalization and activation operations.
#[must_use]
pub fn rmsnorm_smoke() -> &'static Path {
    Path::new(env!("RMSNORM_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that runs a fused scalar program.
#[must_use]
pub fn program_smoke() -> &'static Path {
    Path::new(env!("PROGRAM_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that exposes chunked-prefill replay state.
#[must_use]
pub fn prefill_replay_smoke() -> &'static Path {
    Path::new(env!("PREFILL_REPLAY_SMOKE_COMPONENT"))
}

/// Returns the path to the guest that runs a Qwen-shaped SDK attention block.
#[must_use]
pub fn sdk_smoke() -> &'static Path {
    Path::new(env!("SDK_SMOKE_COMPONENT"))
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

/// Returns the path to the SDK toy MLP engine.
#[must_use]
pub fn toy_mlp() -> &'static Path {
    Path::new(env!("TOY_MLP_COMPONENT"))
}

/// Returns the path to the Qwen3-0.6B engine component.
#[must_use]
pub fn qwen3() -> &'static Path {
    Path::new(env!("QWEN3_COMPONENT"))
}

/// Returns the bf16 `OLMoE-1B-7B-0924` engine component.
#[must_use]
pub fn olmoe() -> &'static Path {
    Path::new(env!("OLMOE_COMPONENT"))
}

/// Returns the Qwen3-Coder-30B-A3B-Instruct MLX 4-bit engine component.
#[must_use]
pub fn qwen3_coder() -> &'static Path {
    Path::new(env!("QWEN3_CODER_COMPONENT"))
}

/// Returns the path to the bf16 Qwen3-0.6B engine component.
#[must_use]
pub fn qwen3_bf16() -> &'static Path {
    Path::new(env!("QWEN3_BF16_COMPONENT"))
}

/// Returns a Qwen3 component carrying synthetic tuning defaults for CLI tests.
#[must_use]
pub fn qwen3_tuning_profile() -> &'static Path {
    Path::new(env!("QWEN3_TUNING_PROFILE_COMPONENT"))
}

/// Returns the path to the guest that consumes host-granted weights.
#[must_use]
pub fn weights_smoke() -> &'static Path {
    Path::new(env!("WEIGHTS_SMOKE_COMPONENT"))
}

/// Returns the optimized `SmolLM2-135M-Instruct` component.
#[must_use]
pub fn smollm2() -> &'static Path {
    Path::new(env!("SMOLLM2_COMPONENT"))
}

/// Returns the reference `SmolLM2-135M-Instruct` component.
#[must_use]
pub fn smollm2_reference() -> &'static Path {
    Path::new(env!("SMOLLM2_REFERENCE_COMPONENT"))
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
