use std::{net::SocketAddr, num::NonZeroU32, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Backend, ByteSize, Choice, Duration, Limits};

/// Configuration used by the serving host.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
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

/// Cross-request serving behavior.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Serving {
    /// Reuse the immediately preceding request's matching live KV prefix.
    #[serde(rename = "prefix-reuse")]
    pub prefix_reuse: bool,
}

impl Default for Serving {
    fn default() -> Self {
        Self { prefix_reuse: true }
    }
}

/// Listener and connection lifecycle settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

/// Files that define the loaded model.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
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
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
    /// Maximum stop strings.
    pub max_stops: u32,
    /// Maximum declared tools.
    pub max_tools: u32,
    /// Maximum aggregate serialized tool bytes.
    pub max_tool_bytes: ByteSize,
    /// Maximum JSON Schema nesting depth.
    pub max_schema_depth: u32,
}

impl Default for RequestCeiling {
    fn default() -> Self {
        Self {
            max_body_bytes: ByteSize::new(1024 * 1024),
            max_messages: 256,
            max_text_bytes: ByteSize::new(1024 * 1024),
            max_prompt_tokens: None,
            max_output_tokens: 8_192,
            max_stops: 4,
            max_tools: 128,
            max_tool_bytes: ByteSize::new(256 * 1024),
            max_schema_depth: 64,
        }
    }
}

/// CPU worker-pool configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LogFormat {
    /// One JSON object per event.
    Json,
    /// Human-readable output for interactive use.
    Pretty,
}

/// Metrics and health endpoint configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
        assert_eq!(config.request.max_stops, 4);
        assert_eq!(config.request.max_tools, 128);
        assert_eq!(config.request.max_tool_bytes, ByteSize::new(256 * 1024));
        assert_eq!(config.request.max_schema_depth, 64);
        assert_eq!(config.metrics.listen, Some(([127, 0, 0, 1], 9091).into()));
        assert_eq!(config.metrics.path, "/metrics");
        assert_eq!(config.logging.format, LogFormat::Json);
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
