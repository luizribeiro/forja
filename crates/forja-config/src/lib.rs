//! Typed configuration values and layered configuration assembly.

#![forbid(unsafe_code)]

mod files;
mod layer;
mod set;
mod units;

pub use files::{dev_layers, file_layer, user_config_path};
pub use layer::{ConfigError, KeyPath, Layer, Layered, Origin, Schema, layer};
pub use set::set_layer;
pub use units::{ByteSize, Duration, Unbounded};

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration used by development commands.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DevConfig {
    /// Developer filesystem locations.
    pub paths: Paths,
    /// Host resource limits.
    pub limits: Limits,
}

/// Developer filesystem locations.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Paths {
    /// Root used to resolve model names.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<PathBuf>,
    /// Directory for transient benchmark results.
    pub scratch: PathBuf,
}

impl Default for Paths {
    fn default() -> Self {
        Self {
            models: None,
            scratch: PathBuf::from("target/forja-bench"),
        }
    }
}

/// Resource limits applied by the host to one guest instance.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Limits {
    /// Maximum live device allocation bytes.
    pub live_bytes: ByteSize,
    /// Maximum tensor rank.
    pub tensor_rank: u32,
    /// Maximum elements in one tensor.
    pub tensor_elements: u64,
    /// Maximum number of live tensor handles.
    pub live_tensor_handles: u32,
    /// Maximum number of live prepared kernels.
    pub live_kernels: u32,
    /// Maximum number of live command graphs.
    pub live_graphs: u32,
    /// Maximum bytes returned by one tensor read.
    pub read_bytes: ByteSize,
    /// Maximum linear memory bytes available to the guest.
    pub guest_memory_bytes: ByteSize,
    /// Maximum table elements available to the guest.
    pub table_elements: u32,
    /// Maximum component instances available to the guest.
    pub instances: u32,
    /// Maximum dispatches recorded in one command list.
    pub dispatches_per_list: Unbounded<u32>,
    /// Maximum work items in one dispatch.
    pub work_per_dispatch: Unbounded<u64>,
    /// Maximum CPU time for one guest export call.
    pub guest_call_timeout: Duration,
    /// Maximum time to wait for one GPU submission.
    pub submission_timeout: Duration,
    /// Cumulative GPU time available to one guest call.
    pub gpu_time_budget: Unbounded<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            live_bytes: ByteSize::new(8 * 1024 * 1024 * 1024),
            tensor_rank: 4,
            tensor_elements: 1_000_000_000,
            live_tensor_handles: 20_000,
            live_kernels: 64,
            live_graphs: 16,
            read_bytes: ByteSize::new(1024 * 1024 * 1024),
            guest_memory_bytes: ByteSize::new(4 * 1024 * 1024 * 1024),
            table_elements: 10_000,
            instances: 10_000,
            dispatches_per_list: Unbounded::Limited(4_096),
            work_per_dispatch: Unbounded::Unlimited,
            guest_call_timeout: Duration::from_secs(300),
            submission_timeout: Duration::from_secs(60),
            gpu_time_budget: Unbounded::Limited(Duration::from_secs(3_600)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time;

    use super::*;

    #[test]
    fn limit_defaults_match_the_cli_host_limits() {
        let limits = Limits::default();
        assert_eq!(limits.live_bytes.get(), 8 * 1024 * 1024 * 1024);
        assert_eq!(limits.tensor_rank, 4);
        assert_eq!(limits.tensor_elements, 1_000_000_000);
        assert_eq!(limits.live_tensor_handles, 20_000);
        assert_eq!(limits.live_kernels, 64);
        assert_eq!(limits.live_graphs, 16);
        assert_eq!(limits.read_bytes.get(), 1024 * 1024 * 1024);
        assert_eq!(limits.guest_memory_bytes.get(), 4 * 1024 * 1024 * 1024);
        assert_eq!(limits.table_elements, 10_000);
        assert_eq!(limits.instances, 10_000);
        assert_eq!(limits.dispatches_per_list, Unbounded::Limited(4_096));
        assert_eq!(limits.work_per_dispatch, Unbounded::Unlimited);
        assert_eq!(
            limits.guest_call_timeout.get(),
            time::Duration::from_secs(300)
        );
        assert_eq!(
            limits.submission_timeout.get(),
            time::Duration::from_secs(60)
        );
        assert_eq!(
            limits.gpu_time_budget,
            Unbounded::Limited(Duration::from_secs(3_600))
        );
    }

    #[test]
    fn deserializes_partial_limits_over_defaults() {
        let limits: Limits = toml::from_str(
            r#"live_bytes = "48GiB"
dispatches_per_list = "unlimited"
submission_timeout = "250ms"
"#,
        )
        .unwrap();
        assert_eq!(limits.live_bytes.get(), 48 * 1024 * 1024 * 1024);
        assert_eq!(limits.tensor_rank, 4);
        assert_eq!(limits.dispatches_per_list, Unbounded::Unlimited);
        assert_eq!(limits.submission_timeout, Duration::from_millis(250));
    }

    #[test]
    fn enum_leaves_serialize_as_scalars() {
        let value = toml::Value::try_from(DevConfig::default()).unwrap();
        let limits = value.get("limits").unwrap();
        for key in [
            "dispatches_per_list",
            "work_per_dispatch",
            "gpu_time_budget",
        ] {
            assert!(!limits.get(key).unwrap().is_table(), "{key}");
        }
    }

    #[test]
    fn every_section_rejects_unknown_keys() {
        let toml::Value::Table(defaults) = toml::Value::try_from(DevConfig::default()).unwrap()
        else {
            panic!("configuration root must be a table");
        };
        for section in defaults.keys() {
            let source = format!("[{section}]\nunknown = true\n");
            let table = toml::from_str(&source).unwrap();
            let error =
                layer::<DevConfig>(vec![Layer::new(Origin::Flag("test"), table)]).unwrap_err();
            assert!(error.to_string().contains("unknown field `unknown`"));
        }
    }
}
