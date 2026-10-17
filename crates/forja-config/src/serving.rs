use std::{net::SocketAddr, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Backend, ByteSize, Duration, Limits};

/// Configuration used by the serving host.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct HostConfig {
    /// Listener and connection lifecycle settings.
    pub server: Server,
    /// One loaded model.
    pub model: ServedModel,
    /// Backend execution strategy.
    pub backend: Backend,
    /// Host resource limits for the engine instance.
    pub limits: Limits,
    /// Per-request admission ceilings.
    pub request: RequestCeiling,
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
        assert!(toml::from_str::<HostConfig>("api_key = 'secret'").is_err());
        assert!(toml::from_str::<HostConfig>("[run]\nmax_tokens = 1").is_err());
    }

    #[test]
    fn serving_defaults_bound_requests_and_connection_lifetimes() {
        let config = HostConfig::default();

        assert_eq!(config.server.queue_length, 16);
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
    }
}
