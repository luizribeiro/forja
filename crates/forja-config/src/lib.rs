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

use std::{collections::BTreeMap, num::NonZeroU32, path::PathBuf};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// Configuration used by development commands.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DevConfig {
    /// Developer filesystem locations.
    pub paths: Paths,
    /// Host resource limits.
    pub limits: Limits,
    /// Benchmark workload and measurement settings.
    pub bench: Bench,
    /// Text-generation settings.
    pub run: Run,
    /// Golden-fixture verification settings.
    pub verify: Verify,
    /// Compute backend settings.
    pub backend: Backend,
}

/// Compute backend settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Backend {
    /// Backend implementation used by run and verify.
    pub kind: BackendKind,
    /// Metal-specific settings.
    pub metal: Metal,
}

impl Default for Backend {
    fn default() -> Self {
        Self {
            kind: BackendKind::Metal,
            metal: Metal::default(),
        }
    }
}

/// Compute backend implementation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// Metal GPU backend.
    Metal,
    /// Portable CPU interpreter.
    Cpu,
}

/// Metal backend settings.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Metal {
    /// Command graph replay strategy.
    pub graph_replay: Choice<GraphReplay>,
}

impl Default for Metal {
    fn default() -> Self {
        Self {
            graph_replay: Choice::Auto,
        }
    }
}

impl Metal {
    /// Resolves Metal settings, using the documented fallback until device tables are available.
    #[must_use]
    pub const fn resolve(&self) -> ResolvedMetal {
        match self.graph_replay {
            Choice::Auto => ResolvedMetal {
                graph_replay: GraphReplay::Tier2,
                auto: Some(AutoNote { fallback: true }),
            },
            Choice::Fixed(graph_replay) => ResolvedMetal {
                graph_replay,
                auto: None,
            },
        }
    }
}

/// A value selected automatically or fixed explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Choice<T> {
    /// Resolve the value for the active environment.
    Auto,
    /// Use one explicit value.
    Fixed(T),
}

impl<T: Serialize> Serialize for Choice<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Fixed(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for Choice<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Value<T> {
            Fixed(T),
            Auto(String),
        }

        match Value::deserialize(deserializer)? {
            Value::Fixed(value) => Ok(Self::Fixed(value)),
            Value::Auto(value) if value == "auto" => Ok(Self::Auto),
            Value::Auto(value) => Err(de::Error::custom(format!("unknown choice {value:?}"))),
        }
    }
}

/// Metal command graph replay strategy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum GraphReplay {
    /// Replay command structure while re-encoding resource bindings.
    Tier1,
    /// Replay commands with reusable indirect command buffers.
    Tier2,
}

impl GraphReplay {
    /// Returns the configuration spelling of this strategy.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tier1 => "tier1",
            Self::Tier2 => "tier2",
        }
    }
}

impl Serialize for GraphReplay {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

/// Concrete Metal settings ready for backend construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedMetal {
    /// Resolved command graph replay strategy.
    pub graph_replay: GraphReplay,
    /// Automatic-resolution provenance when the input was `auto`.
    pub auto: Option<AutoNote>,
}

/// Provenance for an automatically selected value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoNote {
    /// Whether resolution used the compiled fallback rather than a device-table row.
    pub fallback: bool,
}

/// Text-generation settings.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Run {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature, where zero selects greedily.
    pub temperature: f32,
    /// Number of greatest logits retained, where zero disables top-k.
    pub top_k: u32,
    /// Cumulative probability retained after top-k.
    pub top_p: f32,
    /// Reproducible sampling seed, or automatic entropy when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
        }
    }
}

/// Golden-fixture verification settings.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Verify {
    /// Fixture names to verify, or every available fixture when empty.
    pub prompts: Vec<PromptName>,
    /// Directory containing golden fixtures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixtures: Option<PathBuf>,
}

/// A non-empty golden-fixture name.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PromptName(String);

impl PromptName {
    /// Returns the fixture name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for PromptName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        if name.is_empty() {
            Err(de::Error::custom("prompt name cannot be empty"))
        } else {
            Ok(Self(name))
        }
    }
}

/// Benchmark workload and measurement settings.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Bench {
    /// Number of prompt-processing tokens.
    pub pp: NonZeroU32,
    /// Number of token-generation tokens.
    pub tg: NonZeroU32,
    /// Number of measured repetitions.
    pub reps: NonZeroU32,
    /// Number of unmeasured warmup repetitions.
    pub warmups: u32,
    /// Prefill length used before timed decode.
    pub decode_prefill: NonZeroU32,
    /// Decode context lengths to measure.
    pub contexts: ContextList,
    /// Token-selection strategies to measure.
    pub selection: SelectionList,
    /// Whether to collect per-operation timings.
    pub breakdown: bool,
    /// Configuration axes parsed for later benchmark expansion.
    pub vary: BTreeMap<KeyPath, Vec<toml::Value>>,
}

impl Default for Bench {
    fn default() -> Self {
        Self {
            pp: nonzero(512),
            tg: nonzero(128),
            reps: nonzero(30),
            warmups: 3,
            decode_prefill: nonzero(8),
            contexts: ContextList(vec![9, 512, 2_048, 4_000]),
            selection: SelectionList(vec![Selection::GpuSequential, Selection::GpuPipelined]),
            breakdown: false,
            vary: BTreeMap::new(),
        }
    }
}

const fn nonzero(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(value) => value,
        None => NonZeroU32::MIN,
    }
}

/// A non-empty list of benchmark token-selection strategies.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SelectionList(Vec<Selection>);

impl SelectionList {
    /// Returns the validated selections.
    #[must_use]
    pub fn as_slice(&self) -> &[Selection] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for SelectionList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let selection = Vec::<Selection>::deserialize(deserializer)?;
        if selection.is_empty() {
            return Err(de::Error::custom("bench selection must not be empty"));
        }
        Ok(Self(selection))
    }
}

/// A non-empty, strictly increasing list of benchmark contexts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ContextList(Vec<u32>);

impl ContextList {
    /// Returns the validated contexts.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ContextList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let contexts = Vec::<u32>::deserialize(deserializer)?;
        if contexts.is_empty() {
            return Err(de::Error::custom("bench contexts must not be empty"));
        }
        if contexts.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(de::Error::custom(
                "bench contexts must be strictly increasing",
            ));
        }
        Ok(Self(contexts))
    }
}

/// Token-selection strategy measured by a benchmark.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Selection {
    /// Read logits and select the token on the host.
    HostArgmax,
    /// Select tokens on the GPU and wait after each decode step.
    GpuSequential,
    /// Select tokens on the GPU while overlapping queued steps.
    GpuPipelined,
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
    use std::{fs, path::Path, time};

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
    fn parses_bench_selection_and_vary_axes() {
        let config: DevConfig = toml::from_str(
            r#"[bench]
contexts = [1, 7, 33, 4097]
selection = ["host-argmax"]
[bench.vary]
"backend.metal.graph_replay" = ["tier1", "tier2"]
"#,
        )
        .unwrap();
        assert_eq!(config.bench.contexts.as_slice(), [1, 7, 33, 4097]);
        assert_eq!(config.bench.selection.as_slice(), [Selection::HostArgmax]);
        assert_eq!(
            config
                .bench
                .vary
                .get(&KeyPath::new("backend.metal.graph_replay"))
                .unwrap(),
            &[
                toml::Value::String("tier1".to_owned()),
                toml::Value::String("tier2".to_owned())
            ]
        );
    }

    #[test]
    fn rejects_invalid_bench_lists() {
        for source in [
            "[bench]\ncontexts = []\n",
            "[bench]\ncontexts = [9, 7]\n",
            "[bench]\ncontexts = [9, 9]\n",
            "[bench]\nselection = []\n",
        ] {
            assert!(toml::from_str::<DevConfig>(source).is_err(), "{source}");
        }
    }

    #[test]
    fn parses_run_and_verify_sections() {
        let config: DevConfig = toml::from_str(
            r#"[run]
max_tokens = 0
temperature = 0.7
top_k = 40
top_p = 0.9
seed = 42
[verify]
prompts = ["short", "code"]
fixtures = "golden/qwen3-0.6b"
"#,
        )
        .unwrap();
        assert_eq!(config.run.max_tokens, 0);
        assert_eq!(config.run.temperature.to_bits(), 0.7_f32.to_bits());
        assert_eq!(config.run.top_k, 40);
        assert_eq!(config.run.top_p.to_bits(), 0.9_f32.to_bits());
        assert_eq!(config.run.seed, Some(42));
        assert_eq!(
            config
                .verify
                .prompts
                .iter()
                .map(PromptName::as_str)
                .collect::<Vec<_>>(),
            ["short", "code"]
        );
        assert_eq!(
            config.verify.fixtures,
            Some(PathBuf::from("golden/qwen3-0.6b"))
        );
    }

    #[test]
    fn choices_are_strings_and_auto_falls_back_to_tier2() {
        let config: DevConfig = toml::from_str(
            "[backend]\nkind = \"cpu\"\n[backend.metal]\ngraph_replay = \"tier1\"\n",
        )
        .unwrap();
        assert_eq!(config.backend.kind, BackendKind::Cpu);
        assert_eq!(
            config.backend.metal.resolve(),
            ResolvedMetal {
                graph_replay: GraphReplay::Tier1,
                auto: None
            }
        );

        let defaults = DevConfig::default();
        assert_eq!(
            defaults.backend.metal.resolve(),
            ResolvedMetal {
                graph_replay: GraphReplay::Tier2,
                auto: Some(AutoNote { fallback: true })
            }
        );
        let value = toml::Value::try_from(defaults).unwrap();
        assert_eq!(
            value["backend"]["metal"]["graph_replay"].as_str(),
            Some("auto")
        );
    }

    #[test]
    fn committed_default_suite_matches_bench_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/suites/default.toml");
        let config: DevConfig = toml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(config.bench, Bench::default());
    }

    #[test]
    fn enum_leaves_serialize_as_scalars() {
        let value = toml::Value::try_from(DevConfig::default()).unwrap();
        assert!(!value["backend"]["kind"].is_table());
        assert!(!value["bench"]["selection"][0].is_table());
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
