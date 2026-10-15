use std::{
    path::Path,
    sync::OnceLock,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value};
use wasmtime::component::{Component, Func, Linker, WasmStr};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder};

#[allow(missing_docs)]
mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "chat-format-component",
    });
}

const EPOCH_TICK: Duration = Duration::from_millis(10);
static CHAT_FORMAT_ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
static EPOCH_TICKER: OnceLock<()> = OnceLock::new();

/// Resource limits applied to each chat-format component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatFormatLimits {
    /// WebAssembly instructions available to one call.
    pub fuel: u64,
    /// Linear-memory byte limit and hard bound on any guest-forced host copy.
    pub memory_bytes: usize,
    /// Maximum combined byte size of the three JSON inputs.
    pub input_bytes: usize,
    /// Maximum returned string size.
    pub output_bytes: usize,
    /// Wall-clock deadline enforced by epoch interruption.
    pub timeout: Duration,
}

impl Default for ChatFormatLimits {
    fn default() -> Self {
        Self {
            fuel: 10_000_000,
            memory_bytes: 4 * 1024 * 1024,
            input_bytes: 1024 * 1024,
            output_bytes: 1024 * 1024,
            timeout: Duration::from_secs(1),
        }
    }
}

/// Static model serving defaults declared by a chat format.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatInfo {
    /// Token identifiers that finish generation.
    pub stop_token_ids: Vec<u32>,
    /// Sampling defaults.
    pub sampling: SamplingDefaults,
}

/// Optional sampling defaults declared by a chat format.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingDefaults {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability.
    pub top_p: Option<f32>,
    /// Highest-scoring token count.
    pub top_k: Option<u32>,
    /// Repetition penalty requested by the publisher.
    pub repetition_penalty: Option<f32>,
    /// Suggested output-token limit.
    pub max_new_tokens: Option<u32>,
}

struct State {
    limits: StoreLimits,
}

/// Sandboxed runner for a zero-import chat-format component.
pub struct ChatFormat {
    store: Store<State>,
    instance: bindings::ChatFormatComponent,
    render: Func,
    limits: ChatFormatLimits,
}

impl ChatFormat {
    /// Loads a chat-format component without linking any imports.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid components, imports, or resource configuration.
    pub fn load(path: &Path, limits: ChatFormatLimits) -> wasmtime::Result<Self> {
        let engine = shared_engine()?;
        let component = Component::from_file(engine, path)?;
        let linker = Linker::new(engine);
        let mut store = Store::new(
            engine,
            State {
                limits: StoreLimitsBuilder::new()
                    .memory_size(limits.memory_bytes)
                    .build(),
            },
        );
        store.limiter(|state| &mut state.limits);
        store.set_fuel(limits.fuel)?;
        store.set_epoch_deadline(epoch_ticks(limits.timeout));
        let instance = linker.instantiate(&mut store, &component)?;
        let format_index = component
            .get_export_index(None, "l9o:gpu/chat-format@0.1.0")
            .ok_or_else(|| wasmtime::Error::msg("chat format interface export is missing"))?;
        let render_index = component
            .get_export_index(Some(&format_index), "render")
            .ok_or_else(|| wasmtime::Error::msg("chat format render export is missing"))?;
        let render = instance
            .get_func(&mut store, render_index)
            .ok_or_else(|| wasmtime::Error::msg("chat format render function is missing"))?;
        let instance = bindings::ChatFormatComponent::new(&mut store, &instance)?;
        Ok(Self {
            store,
            instance,
            render,
            limits,
        })
    }

    /// Reads model defaults from the component.
    ///
    /// # Errors
    ///
    /// Returns an error when execution traps or exceeds a resource limit.
    pub fn info(&mut self) -> wasmtime::Result<ChatInfo> {
        self.prepare_call()?;
        let info = self
            .instance
            .l9o_gpu_chat_format()
            .call_info(&mut self.store)?;
        Ok(ChatInfo {
            stop_token_ids: info.stop_token_ids,
            sampling: SamplingDefaults {
                temperature: info.sampling.temperature,
                top_p: info.sampling.top_p,
                top_k: info.sampling.top_k,
                repetition_penalty: info.sampling.repetition_penalty,
                max_new_tokens: info.sampling.max_new_tokens,
            },
        })
    }

    /// Renders one prompt from JSON inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when execution traps or an input or output exceeds its byte limit.
    pub fn render(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        options_json: &str,
    ) -> wasmtime::Result<Result<String, String>> {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| wasmtime::Error::msg("system time is before the Unix epoch"))?
            .as_secs();
        let seconds = i64::try_from(seconds)
            .map_err(|_| wasmtime::Error::msg("system time exceeds the chat clock range"))?;
        self.render_at(messages_json, tools_json, options_json, seconds, 0)
    }

    /// Renders one prompt with an explicit clock value for deterministic replay.
    ///
    /// # Errors
    ///
    /// Returns an error when execution traps or an input or output exceeds its byte limit.
    pub fn render_at(
        &mut self,
        messages_json: &str,
        tools_json: &str,
        options_json: &str,
        now_unix_seconds: i64,
        now_utc_offset_seconds: i32,
    ) -> wasmtime::Result<Result<String, String>> {
        self.check_input_size(messages_json, tools_json, options_json)?;
        let options_json =
            match clock_options(options_json, now_unix_seconds, now_utc_offset_seconds) {
                Ok(options) => options,
                Err(error) => return Ok(Err(error)),
            };
        self.check_input_size(messages_json, tools_json, &options_json)?;
        self.prepare_call()?;
        let render = self
            .render
            .typed::<(&str, &str, &str), (Result<WasmStr, WasmStr>,)>(&self.store)?;
        let (result,) = render.call(&mut self.store, (messages_json, tools_json, &options_json))?;
        match result {
            Ok(value) => self.lift_output(&value).map(Ok),
            Err(value) => self.lift_output(&value).map(Err),
        }
    }

    fn lift_output(&self, value: &WasmStr) -> wasmtime::Result<String> {
        let value = value.to_str(&self.store)?;
        if value.len() > self.limits.output_bytes {
            return Err(wasmtime::Error::msg(
                "chat format output exceeds its byte limit",
            ));
        }
        Ok(value.into_owned())
    }

    fn check_input_size(&self, messages: &str, tools: &str, options: &str) -> wasmtime::Result<()> {
        let bytes = messages
            .len()
            .checked_add(tools.len())
            .and_then(|bytes| bytes.checked_add(options.len()));
        if bytes.is_none_or(|bytes| bytes > self.limits.input_bytes) {
            return Err(wasmtime::Error::msg(
                "chat format input exceeds its byte limit",
            ));
        }
        Ok(())
    }

    fn prepare_call(&mut self) -> wasmtime::Result<()> {
        self.store.set_fuel(self.limits.fuel)?;
        self.store
            .set_epoch_deadline(epoch_ticks(self.limits.timeout));
        Ok(())
    }
}

fn epoch_ticks(timeout: Duration) -> u64 {
    let ticks = timeout.as_millis().div_ceil(EPOCH_TICK.as_millis());
    u64::try_from(ticks).unwrap_or(u64::MAX).max(1)
}

fn shared_engine() -> wasmtime::Result<&'static Engine> {
    let engine = CHAT_FORMAT_ENGINE.get_or_init(|| {
        let mut config = Config::new();
        config.consume_fuel(true);
        config.epoch_interruption(true);
        Engine::new(&config).map_err(|error| error.to_string())
    });
    let engine = engine
        .as_ref()
        .map_err(|error| wasmtime::Error::msg(error.clone()))?;
    EPOCH_TICKER.get_or_init(|| {
        let engine = engine.clone();
        drop(thread::spawn(move || {
            loop {
                thread::sleep(EPOCH_TICK);
                engine.increment_epoch();
            }
        }));
    });
    Ok(engine)
}

fn clock_options(options: &str, seconds: i64, offset: i32) -> Result<String, String> {
    let mut options: Map<String, Value> =
        serde_json::from_str(options).map_err(|error| error.to_string())?;
    options.insert("now_unix_seconds".to_owned(), seconds.into());
    options.insert("now_utc_offset_seconds".to_owned(), offset.into());
    serde_json::to_string(&options).map_err(|error| error.to_string())
}
