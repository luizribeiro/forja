use std::{
    net::SocketAddr,
    num::{NonZeroU32, NonZeroU64},
    path::PathBuf,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{Backend, ByteSize, Choice, Duration, Limits};

/// Configuration used by the serving host.
#[derive(Clone, Debug, Default, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct HostConfig {
    /// Listener and connection lifecycle settings.
    pub server: Server,
    /// One loaded model.
    pub model: ServedModel,
    /// Cross-request serving behavior.
    pub serving: Serving,
    /// Backend execution strategy.
    pub backend: Backend,
    /// Host resource limits for the engine instance.
    pub limits: Limits,
    /// Per-request admission ceilings.
    pub request: RequestCeiling,
    /// CPU worker pools used by serving.
    pub pools: Pools,
    /// Structured logging policy.
    pub logging: Logging,
    /// Metrics and health listener.
    pub metrics: Metrics,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HostConfigInput {
    server: Server,
    model: ServedModel,
    serving: Serving,
    backend: Backend,
    limits: Limits,
    request: RequestCeiling,
    pools: Pools,
    logging: Logging,
    metrics: Metrics,
}

impl Default for HostConfigInput {
    fn default() -> Self {
        let value = HostConfig::default();
        Self {
            server: value.server,
            model: value.model,
            serving: value.serving,
            backend: value.backend,
            limits: value.limits,
            request: value.request,
            pools: value.pools,
            logging: value.logging,
            metrics: value.metrics,
        }
    }
}

impl<'de> Deserialize<'de> for HostConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = HostConfigInput::deserialize(deserializer)?;
        let config = Self {
            server: value.server,
            model: value.model,
            serving: value.serving,
            backend: value.backend,
            limits: value.limits,
            request: value.request,
            pools: value.pools,
            logging: value.logging,
            metrics: value.metrics,
        };
        config.validate().map_err(serde::de::Error::custom)?;
        Ok(config)
    }
}

/// Cross-request serving behavior.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Serving {
    /// Reuse matching prefixes retained by a paged KV cache.
    #[serde(rename = "prefix-reuse")]
    pub prefix_reuse: bool,
    /// Maximum simultaneously active sequences.
    pub max_active_sequences: Choice<NonZeroU32>,
    /// Maximum sequences in one batch.
    pub max_batch_size: Choice<NonZeroU32>,
    /// Maximum prompt chunks admitted together.
    pub max_prefill_chunks: Choice<NonZeroU32>,
    /// Maximum aggregate prompt tokens admitted together.
    pub prompt_token_budget: Choice<NonZeroU64>,
    /// Maximum reusable prefix records.
    pub prefix_cache_entries: Choice<NonZeroU32>,
    /// Maximum tokens retained by reusable prefixes.
    pub prefix_cache_tokens: Choice<NonZeroU64>,
    /// Maximum pages retained by reusable prefixes.
    pub prefix_cache_pages: Choice<NonZeroU32>,
    /// Bytes reserved for the page pool.
    pub page_pool_bytes: Choice<PositiveByteSize>,
}

impl Default for Serving {
    fn default() -> Self {
        Self {
            prefix_reuse: true,
            max_active_sequences: Choice::Auto,
            max_batch_size: Choice::Auto,
            max_prefill_chunks: Choice::Auto,
            prompt_token_budget: Choice::Auto,
            prefix_cache_entries: Choice::Auto,
            prefix_cache_tokens: Choice::Auto,
            prefix_cache_pages: Choice::Auto,
            page_pool_bytes: Choice::Auto,
        }
    }
}

/// Listener and connection lifecycle settings.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    /// Listener address.
    pub listen: SocketAddr,
    /// Maximum queued requests.
    pub queue_length: u32,
    /// End-to-end request deadline.
    pub request_deadline: Duration,
    /// Maximum time to read request headers.
    pub header_read_timeout: Duration,
    /// Maximum time to read a request body.
    pub body_read_timeout: Duration,
    /// Maximum time an SSE write may remain stalled.
    pub sse_write_stall_timeout: Duration,
    /// Maximum time to drain the current model before cancelling work.
    pub model_drain_deadline: Duration,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([127, 0, 0, 1], 8080)),
            queue_length: 16,
            request_deadline: Duration::from_secs(300),
            header_read_timeout: Duration::from_secs(10),
            body_read_timeout: Duration::from_secs(30),
            sse_write_stall_timeout: Duration::from_secs(30),
            model_drain_deadline: Duration::from_secs(30),
        }
    }
}

/// A nonzero byte quantity used for explicitly sized pools.
#[derive(Clone, Copy, Debug, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(transparent)]
pub struct PositiveByteSize(ByteSize);

impl PositiveByteSize {
    /// Returns the validated byte count.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

impl<'de> Deserialize<'de> for PositiveByteSize {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = ByteSize::deserialize(deserializer)?;
        if value.get() == 0 {
            return Err(serde::de::Error::custom("byte quantity must be positive"));
        }
        Ok(Self(value))
    }
}

/// Files that define the loaded model.
#[derive(Clone, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServedModel {
    /// Engine component path.
    pub engine: PathBuf,
    /// Engine profile path.
    pub profile: PathBuf,
    /// Model directory path.
    pub model: PathBuf,
    /// Chat-format component path.
    pub chat_format: PathBuf,
}

/// Per-request admission ceilings.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RequestCeiling {
    /// Maximum request body size.
    pub max_body_bytes: ByteSize,
    /// Maximum messages in one request.
    pub max_messages: u32,
    /// Maximum aggregate request text size.
    pub max_text_bytes: ByteSize,
    /// Optional maximum prompt tokens; absent uses the loaded engine maximum.
    pub max_prompt_tokens: Option<u32>,
    /// Maximum generated tokens.
    pub max_output_tokens: u32,
    /// Output-token limit used when a request omits one.
    pub default_output_tokens: u32,
    /// Maximum text bytes in one message.
    pub max_message_bytes: ByteSize,
    /// Maximum bytes emitted by chat rendering.
    pub max_rendered_bytes: ByteSize,
    /// Maximum stop strings.
    pub max_stops: u32,
    /// Maximum bytes in one stop string.
    pub max_stop_bytes: ByteSize,
    /// Maximum declared tools.
    pub max_tools: u32,
    /// Maximum aggregate serialized tool bytes.
    pub max_tool_bytes: ByteSize,
    /// Maximum JSON Schema nesting depth.
    pub max_schema_depth: u32,
    /// Maximum incremental chat-parser events.
    pub max_parser_events: u32,
    /// Maximum bytes in a generated grammar.
    pub max_grammar_bytes: ByteSize,
    /// Maximum lexer states in a compiled grammar.
    pub max_grammar_states: u32,
    /// Maximum time spent compiling one grammar.
    pub max_grammar_compile_time: Duration,
}

impl Default for RequestCeiling {
    fn default() -> Self {
        Self {
            max_body_bytes: ByteSize::new(1024 * 1024),
            max_messages: 256,
            max_text_bytes: ByteSize::new(1024 * 1024),
            max_prompt_tokens: None,
            max_output_tokens: 8_192,
            default_output_tokens: 128,
            max_message_bytes: ByteSize::new(1024 * 1024),
            max_rendered_bytes: ByteSize::new(2 * 1024 * 1024),
            max_stops: 4,
            max_stop_bytes: ByteSize::new(64 * 1024),
            max_tools: 128,
            max_tool_bytes: ByteSize::new(256 * 1024),
            max_schema_depth: 64,
            max_parser_events: 8_192,
            max_grammar_bytes: ByteSize::new(256 * 1024),
            max_grammar_states: 100_000,
            max_grammar_compile_time: Duration::from_millis(500),
        }
    }
}

/// CPU worker-pool configuration.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pools {
    /// Number of grammar compilation workers.
    pub grammar_workers: Choice<NonZeroU32>,
    /// Maximum queued grammar jobs.
    pub grammar_queue_length: u32,
}

impl Default for Pools {
    fn default() -> Self {
        Self {
            grammar_workers: Choice::Auto,
            grammar_queue_length: 1,
        }
    }
}

/// Structured log output configuration.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Logging {
    /// Log filter level.
    pub level: String,
    /// Output encoding.
    pub format: LogFormat,
    /// Include tracing span lifecycle events.
    #[serde(rename = "include-spans")]
    pub include_spans: bool,
}

impl Default for Logging {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            format: LogFormat::Json,
            include_spans: false,
        }
    }
}

/// Structured log output encoding.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LogFormat {
    /// One JSON object per event.
    Json,
    /// Human-readable output for interactive use.
    Pretty,
}

/// Metrics and health endpoint configuration.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Metrics {
    /// Listener address, or absent to disable operational endpoints.
    pub listen: Option<SocketAddr>,
    /// Prometheus metrics path.
    pub path: String,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            listen: Some(SocketAddr::from(([127, 0, 0, 1], 9091))),
            path: "/metrics".to_owned(),
        }
    }
}

impl HostConfig {
    fn validate(&self) -> Result<(), String> {
        if self.request.default_output_tokens == 0
            || self.request.default_output_tokens > self.request.max_output_tokens
        {
            return Err(
                "request.default_output_tokens must be within max_output_tokens".to_owned(),
            );
        }
        if let (Choice::Fixed(batch), Choice::Fixed(active)) = (
            self.serving.max_batch_size,
            self.serving.max_active_sequences,
        ) && batch > active
        {
            return Err("serving.max_batch_size must not exceed max_active_sequences".to_owned());
        }
        checked_product(
            u64::from(self.request.max_messages),
            self.request.max_message_bytes.get(),
            "request message byte ceiling",
        )?;
        checked_product(
            u64::from(self.request.max_stops),
            self.request.max_stop_bytes.get(),
            "request stop byte ceiling",
        )?;
        if let (Choice::Fixed(active), Choice::Fixed(tokens)) = (
            self.serving.max_active_sequences,
            self.serving.prompt_token_budget,
        ) {
            checked_product(
                u64::from(active.get()),
                tokens.get(),
                "prompt reservation ceiling",
            )?;
        }
        if let (Choice::Fixed(entries), Choice::Fixed(tokens)) = (
            self.serving.prefix_cache_entries,
            self.serving.prefix_cache_tokens,
        ) {
            checked_product(
                u64::from(entries.get()),
                tokens.get(),
                "prefix cache ceiling",
            )?;
        }
        if let (Choice::Fixed(live), Choice::Fixed(pool), Choice::Fixed(pages)) = (
            self.limits.live_bytes,
            self.backend.metal.pool_capacity,
            self.serving.page_pool_bytes,
        ) {
            let reserved = pool
                .get()
                .checked_add(pages.get())
                .ok_or_else(|| "configured memory pool sum overflows u64".to_owned())?;
            if reserved > live.get() {
                return Err("configured memory pools exceed limits.live_bytes".to_owned());
            }
        }
        if !self.metrics.path.starts_with('/')
            || matches!(self.metrics.path.as_str(), "/health/live" | "/health/ready")
        {
            return Err("metrics.path must be an absolute, non-health path".to_owned());
        }
        if self.logging.level.trim().is_empty() {
            return Err("logging.level must not be empty".to_owned());
        }
        Ok(())
    }
}

fn checked_product(left: u64, right: u64, name: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflows u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_root_is_strict_and_has_loopback_defaults() {
        let config: HostConfig = toml::from_str(
            r#"
[server]
queue_length = 8

[model]
engine = "/srv/engine.wasm"
profile = "/srv/profile.toml"
model = "/models/qwen"
chat_format = "/srv/chat-format.wasm"

[serving]
prefix-reuse = false

[backend.metal]
graph_replay = "tier1"

[request]
max_output_tokens = 4096
"#,
        )
        .unwrap();
        assert_eq!(
            config.server.listen,
            SocketAddr::from(([127, 0, 0, 1], 8080))
        );
        assert_eq!(config.request.max_output_tokens, 4_096);
        assert!(!config.serving.prefix_reuse);
        assert!(toml::from_str::<HostConfig>("api_key = 'secret'").is_err());
        assert!(toml::from_str::<HostConfig>("[run]\nmax_tokens = 1").is_err());
    }

    #[test]
    fn serving_defaults_bound_requests_and_connection_lifetimes() {
        let config = HostConfig::default();

        assert_eq!(config.server.queue_length, 16);
        assert!(config.serving.prefix_reuse);
        assert_eq!(config.server.request_deadline, Duration::from_secs(300));
        assert_eq!(config.server.header_read_timeout, Duration::from_secs(10));
        assert_eq!(config.server.body_read_timeout, Duration::from_secs(30));
        assert_eq!(
            config.server.sse_write_stall_timeout,
            Duration::from_secs(30)
        );
        assert_eq!(config.request.max_body_bytes, ByteSize::new(1024 * 1024));
        assert_eq!(config.request.max_messages, 256);
        assert_eq!(config.request.max_text_bytes, ByteSize::new(1024 * 1024));
        assert_eq!(config.request.max_prompt_tokens, None);
        assert_eq!(config.request.max_output_tokens, 8_192);
        assert_eq!(config.request.default_output_tokens, 128);
        assert_eq!(config.request.max_message_bytes, ByteSize::new(1024 * 1024));
        assert_eq!(
            config.request.max_rendered_bytes,
            ByteSize::new(2 * 1024 * 1024)
        );
        assert_eq!(config.request.max_stops, 4);
        assert_eq!(config.request.max_stop_bytes, ByteSize::new(64 * 1024));
        assert_eq!(config.request.max_tools, 128);
        assert_eq!(config.request.max_tool_bytes, ByteSize::new(256 * 1024));
        assert_eq!(config.request.max_schema_depth, 64);
        assert_eq!(config.request.max_parser_events, 8_192);
        assert_eq!(config.request.max_grammar_bytes, ByteSize::new(256 * 1024));
        assert_eq!(config.request.max_grammar_states, 100_000);
        assert_eq!(
            config.request.max_grammar_compile_time,
            Duration::from_millis(500)
        );
        assert_eq!(config.metrics.listen, Some(([127, 0, 0, 1], 9091).into()));
        assert_eq!(config.metrics.path, "/metrics");
        assert_eq!(config.logging.format, LogFormat::Json);
        assert_eq!(config.serving.max_active_sequences, Choice::Auto);
        assert_eq!(config.serving.page_pool_bytes, Choice::Auto);
    }

    #[test]
    fn rejects_inconsistent_or_overflowing_host_ceilings() {
        for source in [
            "[request]\ndefault_output_tokens = 129\nmax_output_tokens = 128",
            "[serving]\nmax_batch_size = 8\nmax_active_sequences = 4",
            "[request]\nmax_messages = 4294967295\nmax_message_bytes = 18446744073709551615",
            "[limits]\nlive_bytes = 100\n[backend.metal]\npool_capacity = 60\n[serving]\npage_pool_bytes = 41",
            "[metrics]\npath = 'health'",
        ] {
            assert!(toml::from_str::<HostConfig>(source).is_err(), "{source}");
        }
        assert!(toml::from_str::<HostConfig>("[serving]\nmax_batch_size = 0").is_err());
        assert!(toml::from_str::<HostConfig>("[serving]\npage_pool_bytes = 0").is_err());
    }

    #[test]
    fn preserves_every_preexisting_default_value() {
        let config = HostConfig::default();
        assert_eq!(config.server.listen, ([127, 0, 0, 1], 8080).into());
        assert_eq!(config.server.queue_length, 16);
        assert_eq!(config.server.request_deadline, Duration::from_secs(300));
        assert_eq!(config.server.header_read_timeout, Duration::from_secs(10));
        assert_eq!(config.server.body_read_timeout, Duration::from_secs(30));
        assert_eq!(
            config.server.sse_write_stall_timeout,
            Duration::from_secs(30)
        );
        assert_eq!(config.model, ServedModel::default());
        assert!(config.serving.prefix_reuse);
        assert_eq!(config.backend, Backend::default());
        assert_eq!(config.limits, Limits::default());
        assert_eq!(config.request.max_body_bytes, ByteSize::new(1024 * 1024));
        assert_eq!(config.request.max_messages, 256);
        assert_eq!(config.request.max_text_bytes, ByteSize::new(1024 * 1024));
        assert_eq!(config.request.max_prompt_tokens, None);
        assert_eq!(config.request.max_output_tokens, 8_192);
        assert_eq!(config.request.max_stops, 4);
        assert_eq!(config.request.max_tools, 128);
        assert_eq!(config.request.max_tool_bytes, ByteSize::new(256 * 1024));
        assert_eq!(config.request.max_schema_depth, 64);
    }
}
