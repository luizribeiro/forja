use std::{collections::BTreeMap, fs, path::PathBuf};

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use forja_config::{
    BackendKind, BenchSampling, ConfigError, DevConfig, GraphReplay, KeyPath, Layer, Layered,
    Limits, Origin, Selection, dev_layers, file_layer, layer, set_layer,
};

use crate::benchmark_record::{Recorded, SCHEMA_VERSION};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Backend {
    Metal,
    Cpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Precision {
    F32,
    Bf16,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Verify {
    pub(crate) engine: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) fixtures: PathBuf,
    pub(crate) backend: Backend,
    pub(crate) precision: Precision,
    pub(crate) prompts: Vec<String>,
    pub(crate) graph_replay: GraphReplay,
    pub(crate) limits: Limits,
}

#[derive(Args)]
struct VerifyArgs {
    /// WebAssembly engine component.
    #[arg(long)]
    engine: PathBuf,
    /// Directory containing model weights.
    #[arg(long)]
    model_dir: PathBuf,
    /// Directory containing golden fixtures.
    #[arg(long)]
    fixtures: Option<PathBuf>,
    /// Compute backend.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// Engine precision.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    precision: Precision,
    /// Comma-separated fixture names. Repeating the option appends names.
    #[arg(long, value_delimiter = ',', value_parser = parse_prompt_name)]
    prompts: Vec<String>,
    /// Set a Metal backend option.
    #[arg(long = "backend-option", hide = true, value_parser = parse_backend_option)]
    graph_replay: Option<GraphReplay>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Run {
    pub(crate) engine: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) prompt: String,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) seed: Option<u64>,
    pub(crate) backend: Backend,
    pub(crate) graph_replay: GraphReplay,
    pub(crate) limits: Limits,
}

#[derive(Args)]
struct RunArgs {
    /// WebAssembly engine component.
    #[arg(long)]
    engine: PathBuf,
    /// Directory containing model weights and tokenizer files.
    #[arg(long)]
    model_dir: PathBuf,
    /// Text to continue.
    #[arg(long)]
    prompt: String,
    /// Maximum number of tokens to generate.
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Logit temperature, where zero selects greedily.
    #[arg(long)]
    temperature: Option<f32>,
    /// Number of greatest logits retained, where zero disables top-k.
    #[arg(long)]
    top_k: Option<u32>,
    /// Cumulative probability retained after top-k.
    #[arg(long)]
    top_p: Option<f32>,
    /// Reproducible sampling seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Compute backend.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// Set a Metal backend option.
    #[arg(long = "backend-option", hide = true, value_parser = parse_backend_option)]
    graph_replay: Option<GraphReplay>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Bench {
    pub(crate) engines: Vec<PathBuf>,
    pub(crate) model_dir: PathBuf,
    pub(crate) pp: usize,
    pub(crate) tg: usize,
    pub(crate) reps: usize,
    pub(crate) warmups: usize,
    pub(crate) decode_prefill: usize,
    pub(crate) contexts: Vec<usize>,
    pub(crate) selection: Vec<Selection>,
    pub(crate) json: Option<PathBuf>,
    pub(crate) breakdown: bool,
    pub(crate) graph_replay: GraphReplay,
    pub(crate) limits: Limits,
    pub(crate) config: Box<DevConfig>,
    pub(crate) origins: BTreeMap<KeyPath, Origin>,
    pub(crate) axes: BTreeMap<KeyPath, Vec<toml::Value>>,
    pub(crate) strategy_axes: Vec<KeyPath>,
    pub(crate) points: Vec<BenchPoint>,
    pub(crate) rerun: Option<Recorded>,
    pub(crate) allow_diff: Vec<KeyPath>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct BenchPoint {
    pub(crate) pp: usize,
    pub(crate) tg: usize,
    pub(crate) contexts: Vec<usize>,
    pub(crate) selection: Vec<Selection>,
    pub(crate) graph_replay: GraphReplay,
    pub(crate) config: Box<DevConfig>,
    pub(crate) origins: BTreeMap<KeyPath, Origin>,
    pub(crate) values: BTreeMap<KeyPath, toml::Value>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Profile {
    pub(crate) engine: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) context: usize,
    pub(crate) warmups: usize,
    pub(crate) sampling: BenchSampling,
    pub(crate) json: bool,
    pub(crate) scratch: PathBuf,
    pub(crate) graph_replay: GraphReplay,
    pub(crate) limits: Limits,
}

#[derive(Args)]
struct ProfileArgs {
    /// WebAssembly engine component.
    #[arg(long)]
    engine: PathBuf,
    /// Directory containing model weights.
    #[arg(long)]
    model_dir: PathBuf,
    /// Decode context length to profile.
    #[arg(long, value_parser = parse_profile_context)]
    context: u32,
    /// Write JSON under the configured scratch directory instead of Markdown.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct BenchArgs {
    /// WebAssembly engine component. Repeat to compare engines.
    #[arg(long = "engine", required = true)]
    engines: Vec<PathBuf>,
    /// Directory containing model weights.
    #[arg(long)]
    model_dir: PathBuf,
    /// Number of prompt-processing tokens.
    #[arg(long, value_parser = parse_positive)]
    pp: Option<u32>,
    /// Number of token-generation tokens.
    #[arg(long, value_parser = parse_positive)]
    tg: Option<u32>,
    /// Number of measured repetitions.
    #[arg(long, value_parser = parse_positive)]
    reps: Option<u32>,
    /// Path for the JSON report.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Print a per-operation timing breakdown.
    #[arg(long)]
    breakdown: bool,
    /// Select tokens on the host instead of comparing selection modes.
    #[arg(long)]
    host_argmax: bool,
    /// Set a Metal backend option.
    #[arg(long = "backend-option", hide = true, value_parser = parse_backend_option)]
    graph_replay: Option<GraphReplay>,
    /// Vary one configuration key over a comma list or TOML array.
    #[arg(long = "vary", value_parser = parse_vary)]
    vary: Vec<VaryArg>,
    /// Re-run a schema-v2 benchmark record.
    #[arg(long)]
    rerun: Option<PathBuf>,
    /// Permit one performance-key difference while re-running.
    #[arg(long = "allow-diff", value_parser = parse_key_path)]
    allow_diff: Vec<KeyPath>,
}

#[derive(Clone, Debug)]
struct VaryArg {
    key: KeyPath,
    values: Vec<toml::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxisClass {
    Strategy,
    Tuning,
    Workload,
}

#[derive(Debug, PartialEq)]
pub(crate) enum Command {
    Bench(Bench),
    Config(ConfigShow),
    Profile(Profile),
    Run(Run),
    Verify(Verify),
}

#[derive(Args)]
struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommand,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Print the effective development configuration.
    Show(ConfigShowArgs),
}

#[derive(Args)]
struct ConfigShowArgs {
    /// Annotate each value with its winning source.
    #[arg(long)]
    origin: bool,
    /// Print only one key or section.
    #[arg(long)]
    key: Option<String>,
    /// Print JSON instead of TOML.
    #[arg(long)]
    json: bool,
    /// Include values that still have their compiled defaults.
    #[arg(long)]
    defaults: bool,
}

#[derive(Debug, PartialEq)]
pub(crate) struct ConfigShow {
    pub(crate) layered: Layered<DevConfig>,
    pub(crate) origin: bool,
    pub(crate) key: Option<String>,
    pub(crate) json: bool,
    pub(crate) defaults: bool,
}

#[derive(Parser)]
#[command(name = "forja", about = "Run and inspect Forja inference engines")]
struct Cli {
    /// Add a configuration file. Files are layered in order before any --set values.
    #[arg(short = 'c', long, global = true)]
    config: Vec<PathBuf>,
    /// Set one key after all configuration files. Values are layered in order.
    #[arg(short = 's', long = "set", global = true)]
    set: Vec<String>,
    /// Skip the user configuration file.
    #[arg(long, global = true)]
    isolated: bool,
    #[command(subcommand)]
    command: ParsedCommand,
}

#[derive(Subcommand)]
enum ParsedCommand {
    /// Benchmark one or more engines.
    Bench(BenchArgs),
    /// Inspect configuration.
    Config(ConfigArgs),
    /// Profile one decode token.
    Profile(ProfileArgs),
    /// Generate a completion.
    Run(RunArgs),
    /// Compare engine outputs with golden fixtures.
    Verify(VerifyArgs),
}

pub(crate) fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Command, clap::Error> {
    parse_with(arguments, dev_layers)
}

fn parse_with(
    arguments: impl IntoIterator<Item = String>,
    base_layers: impl FnOnce(bool) -> Result<Vec<Layer>, ConfigError>,
) -> Result<Command, clap::Error> {
    let arguments = std::iter::once("forja".to_owned())
        .chain(arguments)
        .collect::<Vec<_>>();
    let mut matches = Cli::command().try_get_matches_from(arguments)?;
    let overrides = ordered_overrides(&matches);
    let cli = Cli::from_arg_matches_mut(&mut matches)?;
    let rerun = cli.command.rerun_record()?;
    let mut layers =
        base_layers(cli.isolated || rerun.is_some()).map_err(|error| config_error(&error))?;
    if let Some((path, record)) = &rerun {
        let table = toml::from_str(&record.config).map_err(|error| {
            Cli::command().error(
                clap::error::ErrorKind::ValueValidation,
                format!("{}: record config is invalid: {error}", path.display()),
            )
        })?;
        layers.push(Layer::new(Origin::File(path.clone()), table));
    }
    let mut set_layers = Vec::new();
    let mut set_keys = BTreeMap::new();
    let mut set_index = 0;
    for override_ in overrides {
        match override_ {
            Override::File(path) => {
                layers.push(file_layer::<DevConfig>(&path).map_err(|error| config_error(&error))?);
            }
            Override::Set(expression) => {
                set_index += 1;
                let (layer, key) = set_layer::<DevConfig>(set_index, &expression)
                    .map_err(|error| config_error(&error))?;
                set_keys.insert(key, set_index);
                set_layers.push(layer);
            }
        }
    }
    for (layer, key) in cli.command.sugar_layers()? {
        if let Some(index) = set_keys.get(&key) {
            return Err(Cli::command().error(
                clap::error::ErrorKind::ArgumentConflict,
                format!(
                    "configuration key {key} is set by both {} and --set #{index}",
                    layer.origin
                ),
            ));
        }
        layers.push(layer);
    }
    layers.extend(set_layers);
    let layered = layer::<DevConfig>(layers).map_err(|error| config_error(&error))?;
    let limits = layered.config.limits.clone();
    Ok(match cli.command {
        ParsedCommand::Bench(options) => Command::Bench(options.with_config(
            &layered,
            limits,
            rerun.map(|(_, record)| record),
        )?),
        ParsedCommand::Config(options) => match options.command {
            ConfigCommand::Show(options) => Command::Config(ConfigShow {
                layered,
                origin: options.origin,
                key: options.key,
                json: options.json,
                defaults: options.defaults,
            }),
        },
        ParsedCommand::Profile(options) => {
            Command::Profile(options.with_config(&layered.config, limits)?)
        }
        ParsedCommand::Run(options) => Command::Run(options.with_config(&layered.config, limits)?),
        ParsedCommand::Verify(options) => {
            Command::Verify(options.with_config(&layered.config, limits)?)
        }
    })
}

impl ParsedCommand {
    fn rerun_record(&self) -> Result<Option<(PathBuf, Recorded)>, clap::Error> {
        let Self::Bench(options) = self else {
            return Ok(None);
        };
        options.rerun.as_ref().map(load_record).transpose()
    }
}

fn load_record(path: &PathBuf) -> Result<(PathBuf, Recorded), clap::Error> {
    let bytes = fs::read(path).map_err(|error| {
        Cli::command().error(
            clap::error::ErrorKind::Io,
            format!("cannot read {}: {error}", path.display()),
        )
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        Cli::command().error(
            clap::error::ErrorKind::ValueValidation,
            format!("{}: invalid benchmark record: {error}", path.display()),
        )
    })?;
    let schema_version = value["schema_version"].as_u64().unwrap_or_default();
    if schema_version != u64::from(SCHEMA_VERSION) {
        return Err(Cli::command().error(
            clap::error::ErrorKind::ValueValidation,
            format!(
                "{}: rerun requires schema version {SCHEMA_VERSION}, found {}",
                path.display(),
                schema_version
            ),
        ));
    }
    let record: Recorded = serde_json::from_value(value).map_err(|error| {
        Cli::command().error(
            clap::error::ErrorKind::ValueValidation,
            format!("{}: invalid benchmark record: {error}", path.display()),
        )
    })?;
    Ok((path.clone(), record))
}

enum Override {
    File(PathBuf),
    Set(String),
}

fn ordered_overrides(matches: &clap::ArgMatches) -> Vec<Override> {
    let mut overrides = Vec::new();
    if let Some(paths) = matches.get_many::<PathBuf>("config") {
        overrides.extend(paths.cloned().map(Override::File));
    }
    if let Some(expressions) = matches.get_many::<String>("set") {
        overrides.extend(expressions.cloned().map(Override::Set));
    }
    overrides
}

fn config_error(error: &ConfigError) -> clap::Error {
    Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
}

fn parse_positive(value: &str) -> Result<u32, String> {
    value
        .parse()
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| "value must be a positive integer".to_owned())
}

fn parse_profile_context(value: &str) -> Result<u32, String> {
    value
        .parse()
        .ok()
        .filter(|&count| count > 1)
        .ok_or_else(|| "value must be an integer greater than one".to_owned())
}

fn parse_key_path(value: &str) -> Result<KeyPath, String> {
    if value.is_empty() {
        Err("configuration key must not be empty".to_owned())
    } else {
        Ok(KeyPath::new(value))
    }
}

struct Sugar {
    flag: &'static str,
    key: &'static str,
}

const SUGAR: &[Sugar] = &[
    Sugar {
        flag: "--pp",
        key: "bench.pp",
    },
    Sugar {
        flag: "--tg",
        key: "bench.tg",
    },
    Sugar {
        flag: "--reps",
        key: "bench.reps",
    },
    Sugar {
        flag: "--breakdown",
        key: "bench.breakdown",
    },
    Sugar {
        flag: "--host-argmax",
        key: "bench.selection",
    },
    Sugar {
        flag: "--max-tokens",
        key: "run.max_tokens",
    },
    Sugar {
        flag: "--temperature",
        key: "run.temperature",
    },
    Sugar {
        flag: "--top-k",
        key: "run.top_k",
    },
    Sugar {
        flag: "--top-p",
        key: "run.top_p",
    },
    Sugar {
        flag: "--seed",
        key: "run.seed",
    },
    Sugar {
        flag: "--prompts",
        key: "verify.prompts",
    },
    Sugar {
        flag: "--fixtures",
        key: "verify.fixtures",
    },
    Sugar {
        flag: "--backend",
        key: "backend.kind",
    },
    Sugar {
        flag: "--backend-option",
        key: "backend.metal.graph_replay",
    },
];

impl ParsedCommand {
    fn sugar_layers(&self) -> Result<Vec<(Layer, KeyPath)>, clap::Error> {
        let mut values = Vec::new();
        match self {
            Self::Bench(options) => {
                values.extend(options.pp.map(|value| ("--pp", value.to_string())));
                values.extend(options.tg.map(|value| ("--tg", value.to_string())));
                values.extend(options.reps.map(|value| ("--reps", value.to_string())));
                values.extend(
                    options
                        .breakdown
                        .then(|| ("--breakdown", "true".to_owned())),
                );
                values.extend(
                    options
                        .host_argmax
                        .then(|| ("--host-argmax", "[\"host-argmax\"]".to_owned())),
                );
                values.extend(graph_replay_sugar(options.graph_replay));
                for vary in &options.vary {
                    let mut vary_table = toml::Table::new();
                    vary_table.insert(
                        vary.key.as_str().to_owned(),
                        toml::Value::Array(vary.values.clone()),
                    );
                    let mut bench = toml::Table::new();
                    bench.insert("vary".to_owned(), toml::Value::Table(vary_table));
                    let mut table = toml::Table::new();
                    table.insert("bench".to_owned(), toml::Value::Table(bench));
                    values.push(("--vary", toml::Value::Table(table).to_string()));
                }
            }
            Self::Run(options) => {
                values.extend(
                    options
                        .max_tokens
                        .map(|value| ("--max-tokens", value.to_string())),
                );
                values.extend(
                    options
                        .temperature
                        .map(|value| ("--temperature", value.to_string())),
                );
                values.extend(options.top_k.map(|value| ("--top-k", value.to_string())));
                values.extend(options.top_p.map(|value| ("--top-p", value.to_string())));
                values.extend(options.seed.map(|value| ("--seed", value.to_string())));
                values.extend(backend_sugar(options.backend));
                values.extend(graph_replay_sugar(options.graph_replay));
            }
            Self::Verify(options) => {
                if !options.prompts.is_empty() {
                    let prompts = options
                        .prompts
                        .iter()
                        .cloned()
                        .map(toml::Value::String)
                        .collect();
                    values.push(("--prompts", toml::Value::Array(prompts).to_string()));
                }
                if let Some(path) = &options.fixtures {
                    let Some(path) = path.to_str() else {
                        return Err(Cli::command().error(
                            clap::error::ErrorKind::InvalidUtf8,
                            "--fixtures must be valid UTF-8",
                        ));
                    };
                    values.push((
                        "--fixtures",
                        toml::Value::String(path.to_owned()).to_string(),
                    ));
                }
                values.extend(backend_sugar(options.backend));
                values.extend(graph_replay_sugar(options.graph_replay));
            }
            Self::Config(_) | Self::Profile(_) => {}
        }
        values
            .into_iter()
            .map(|(flag, value)| sugar_layer(flag, &value))
            .collect()
    }
}

fn backend_sugar(backend: Option<Backend>) -> Option<(&'static str, String)> {
    backend.map(|backend| {
        let value = match backend {
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        };
        (
            "--backend",
            toml::Value::String(value.to_owned()).to_string(),
        )
    })
}

fn graph_replay_sugar(value: Option<GraphReplay>) -> Option<(&'static str, String)> {
    value.map(|value| {
        (
            "--backend-option",
            toml::Value::String(value.as_str().to_owned()).to_string(),
        )
    })
}

fn sugar_layer(flag: &'static str, value: &str) -> Result<(Layer, KeyPath), clap::Error> {
    if flag == "--vary" {
        let table = value
            .parse::<toml::Value>()
            .ok()
            .and_then(|value| value.as_table().cloned())
            .ok_or_else(|| {
                Cli::command().error(
                    clap::error::ErrorKind::InvalidValue,
                    "cannot construct --vary layer",
                )
            })?;
        return Ok((
            Layer::new(Origin::Flag("--vary"), table),
            KeyPath::new("bench.vary"),
        ));
    }
    let Some(sugar) = SUGAR.iter().find(|sugar| sugar.flag == flag) else {
        return Err(Cli::command().error(
            clap::error::ErrorKind::InvalidValue,
            format!("unknown sugar flag {flag}"),
        ));
    };
    let (mut layer, key) = set_layer::<DevConfig>(0, &format!("{}={value}", sugar.key))
        .map_err(|error| config_error(&error))?;
    layer.origin = Origin::Flag(flag);
    Ok((layer, key))
}

impl BenchArgs {
    fn with_config(
        self,
        layered: &Layered<DevConfig>,
        limits: Limits,
        rerun: Option<Recorded>,
    ) -> Result<Bench, clap::Error> {
        if rerun.is_none() && !self.allow_diff.is_empty() {
            return Err(Cli::command().error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "--allow-diff requires --rerun",
            ));
        }
        let config = &layered.config;
        let count = |value: u32| {
            usize::try_from(value).map_err(|error| {
                Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
            })
        };
        let points = expand_points(layered).map_err(|error| {
            Cli::command().error(clap::error::ErrorKind::ValueValidation, error)
        })?;
        let strategy_axes = config
            .bench
            .vary
            .keys()
            .filter(|key| axis_class(key.as_str()) == Ok(AxisClass::Strategy))
            .cloned()
            .collect();
        Ok(Bench {
            engines: self.engines,
            model_dir: self.model_dir,
            pp: count(config.bench.pp.get())?,
            tg: count(config.bench.tg.get())?,
            reps: count(config.bench.reps.get())?,
            warmups: count(config.bench.warmups)?,
            decode_prefill: count(config.bench.decode_prefill.get())?,
            contexts: config
                .bench
                .contexts
                .as_slice()
                .iter()
                .copied()
                .map(count)
                .collect::<Result<_, _>>()?,
            selection: config.bench.selection.as_slice().to_vec(),
            json: self.json,
            breakdown: config.bench.breakdown,
            graph_replay: config.backend.metal.resolve().graph_replay,
            limits,
            config: Box::new(config.clone()),
            origins: layered.origins.clone(),
            axes: config.bench.vary.clone(),
            strategy_axes,
            points,
            rerun,
            allow_diff: self.allow_diff,
        })
    }
}

fn expand_points(layered: &Layered<DevConfig>) -> Result<Vec<BenchPoint>, String> {
    for (key, values) in &layered.config.bench.vary {
        axis_class(key.as_str())?;
        if values.is_empty() {
            return Err(format!("vary axis {key} must not be empty"));
        }
    }
    let mut combinations = vec![BTreeMap::new()];
    for (key, values) in &layered.config.bench.vary {
        combinations = combinations
            .into_iter()
            .flat_map(|point| {
                values.iter().cloned().map(move |value| {
                    let mut next = point.clone();
                    next.insert(key.clone(), value);
                    next
                })
            })
            .collect();
    }
    combinations
        .into_iter()
        .map(|values| point(layered, values))
        .collect()
}

fn point(
    base: &Layered<DevConfig>,
    values: BTreeMap<KeyPath, toml::Value>,
) -> Result<BenchPoint, String> {
    let base_table = toml::Value::try_from(&base.config)
        .map_err(|error| format!("cannot serialize benchmark config: {error}"))?
        .as_table()
        .cloned()
        .ok_or_else(|| "benchmark config is not a table".to_owned())?;
    let mut layers = vec![Layer::new(Origin::Default, base_table)];
    for (key, value) in &values {
        let table = toml::from_str(&format!("{key} = {value}"))
            .map_err(|error| format!("vary axis {key}: {error}"))?;
        layers.push(Layer::new(Origin::Vary, table));
    }
    let resolved = layer::<DevConfig>(layers).map_err(|error| error.to_string())?;
    validate_bench_sampling(&resolved.config)?;
    let mut origins = base.origins.clone();
    origins.extend(values.keys().cloned().map(|key| (key, Origin::Vary)));
    let count =
        |value: u32| usize::try_from(value).map_err(|error| format!("benchmark count: {error}"));
    Ok(BenchPoint {
        pp: count(resolved.config.bench.pp.get())?,
        tg: count(resolved.config.bench.tg.get())?,
        contexts: resolved
            .config
            .bench
            .contexts
            .as_slice()
            .iter()
            .copied()
            .map(count)
            .collect::<Result<_, _>>()?,
        selection: resolved.config.bench.selection.as_slice().to_vec(),
        graph_replay: resolved.config.backend.metal.resolve().graph_replay,
        config: Box::new(resolved.config),
        origins,
        values,
    })
}

fn parse_vary(expression: &str) -> Result<VaryArg, String> {
    let (key, rhs) = expression
        .split_once('=')
        .ok_or_else(|| "--vary expects KEY=RHS".to_owned())?;
    let key = KeyPath::new(key.trim());
    axis_class(key.as_str())?;
    if key.as_str() == "engine.tunings" {
        return Err("tunings arrive with engine profiles".to_owned());
    }
    let rhs = rhs.trim();
    let values = if rhs.contains(['[', ']', '"', '\'', '{', '}']) {
        let table: toml::Table = toml::from_str(&format!("values = {rhs}"))
            .map_err(|error| format!("invalid --vary array: {error}"))?;
        table
            .get("values")
            .and_then(toml::Value::as_array)
            .cloned()
            .ok_or_else(|| "--vary RHS with TOML punctuation must be an array".to_owned())?
    } else {
        rhs.split(',')
            .map(str::trim)
            .map(parse_bare_vary_value)
            .collect::<Result<_, _>>()?
    };
    if values.is_empty() {
        return Err("--vary axis must not be empty".to_owned());
    }
    Ok(VaryArg { key, values })
}

fn parse_bare_vary_value(value: &str) -> Result<toml::Value, String> {
    if value.is_empty() {
        return Err("--vary bare scalar must not be empty".to_owned());
    }
    let source = format!("value = {value}");
    let parsed = toml::from_str::<toml::Table>(&source)
        .ok()
        .and_then(|mut table| table.remove("value"));
    if let Some(parsed) = parsed {
        return Ok(parsed);
    }
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || "_-./~:".contains(character))
    {
        Ok(toml::Value::String(value.to_owned()))
    } else {
        Err(format!("invalid --vary scalar {value:?}"))
    }
}

fn axis_class(key: &str) -> Result<AxisClass, String> {
    if key.starts_with("backend.metal.") {
        Ok(AxisClass::Strategy)
    } else if key == "engine.tunings" {
        Ok(AxisClass::Tuning)
    } else if matches!(
        key,
        "bench.selection" | "bench.contexts" | "bench.pp" | "bench.tg"
    ) || key.starts_with("bench.sampling.")
    {
        Ok(AxisClass::Workload)
    } else if ["limits", "paths", "run", "verify"]
        .iter()
        .any(|prefix| key == *prefix || key.starts_with(&format!("{prefix}.")))
        || key == "backend.kind"
    {
        Err(format!("configuration key {key} cannot be varied"))
    } else {
        Err(format!("configuration key {key} is not a benchmark axis"))
    }
}

fn validate_bench_sampling(config: &DevConfig) -> Result<(), String> {
    let sampling = config.bench.sampling;
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return Err("bench.sampling.temperature must be finite and nonnegative".to_owned());
    }
    if !(sampling.top_p.is_finite() && 0.0 < sampling.top_p && sampling.top_p <= 1.0) {
        return Err("bench.sampling.top_p must be finite and in (0, 1]".to_owned());
    }
    Ok(())
}

impl ProfileArgs {
    fn with_config(self, config: &DevConfig, limits: Limits) -> Result<Profile, clap::Error> {
        validate_bench_sampling(config).map_err(|error| {
            Cli::command().error(clap::error::ErrorKind::ValueValidation, error)
        })?;
        Ok(Profile {
            engine: self.engine,
            model_dir: self.model_dir,
            context: usize::try_from(self.context).map_err(|error| {
                Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
            })?,
            warmups: usize::try_from(config.bench.warmups).map_err(|error| {
                Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
            })?,
            sampling: config.bench.sampling,
            json: self.json,
            scratch: config.paths.scratch.clone(),
            graph_replay: config.backend.metal.resolve().graph_replay,
            limits,
        })
    }
}

impl RunArgs {
    fn with_config(self, config: &DevConfig, limits: Limits) -> Result<Run, clap::Error> {
        let max_tokens = usize::try_from(config.run.max_tokens).map_err(|error| {
            Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
        })?;
        if !config.run.temperature.is_finite() || config.run.temperature < 0.0 {
            return Err(Cli::command().error(
                clap::error::ErrorKind::ValueValidation,
                "run.temperature must be finite and nonnegative",
            ));
        }
        if !(config.run.top_p.is_finite() && 0.0 < config.run.top_p && config.run.top_p <= 1.0) {
            return Err(Cli::command().error(
                clap::error::ErrorKind::ValueValidation,
                "run.top_p must be finite and in (0, 1]",
            ));
        }
        Ok(Run {
            engine: self.engine,
            model_dir: self.model_dir,
            prompt: self.prompt,
            max_tokens,
            temperature: config.run.temperature,
            top_k: config.run.top_k,
            top_p: config.run.top_p,
            seed: config.run.seed,
            backend: config.backend.kind.into(),
            graph_replay: config.backend.metal.resolve().graph_replay,
            limits,
        })
    }
}

fn parse_backend_option(value: &str) -> Result<GraphReplay, String> {
    match value {
        "graph-replay=tier1" => Ok(GraphReplay::Tier1),
        "graph-replay=tier2" => Ok(GraphReplay::Tier2),
        _ => Err(format!("unknown Metal backend option {value:?}")),
    }
}

impl From<BackendKind> for Backend {
    fn from(value: BackendKind) -> Self {
        match value {
            BackendKind::Metal => Self::Metal,
            BackendKind::Cpu => Self::Cpu,
        }
    }
}

fn parse_prompt_name(value: &str) -> Result<String, String> {
    if value.is_empty() {
        Err("prompt name cannot be empty".to_owned())
    } else {
        Ok(value.to_owned())
    }
}

impl VerifyArgs {
    fn with_config(self, config: &DevConfig, limits: Limits) -> Result<Verify, clap::Error> {
        let fixtures = config.verify.fixtures.clone().ok_or_else(|| {
            Cli::command().error(
                clap::error::ErrorKind::MissingRequiredArgument,
                "verify requires --fixtures or verify.fixtures in configuration",
            )
        })?;
        Ok(Verify {
            engine: self.engine,
            model_dir: self.model_dir,
            fixtures,
            backend: config.backend.kind.into(),
            precision: self.precision,
            prompts: config
                .verify
                .prompts
                .iter()
                .map(|name| name.as_str().to_owned())
                .collect(),
            graph_replay: config.backend.metal.resolve().graph_replay,
            limits,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use forja_testing::temporary_directory;

    use super::*;

    fn parse_with_file(arguments: &[&str], source: &str) -> Command {
        let layer = Layer::new(
            Origin::File("test.toml".into()),
            toml::from_str(source).unwrap(),
        );
        parse_with(
            arguments.iter().map(|argument| (*argument).to_owned()),
            |_| Ok(vec![layer]),
        )
        .unwrap()
    }

    #[test]
    fn parses_benchmark_options() {
        let command = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--pp",
                "33",
                "--tg",
                "7",
                "--reps",
                "2",
                "--json",
                "/result.json",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.engines, [PathBuf::from("/engine.wasm")]);
        assert_eq!(options.model_dir, PathBuf::from("/model"));
        assert_eq!((options.pp, options.tg, options.reps), (33, 7, 2));
        assert_eq!(options.json, Some(PathBuf::from("/result.json")));
    }

    #[test]
    fn parses_profile_options_with_bench_configuration() {
        let command = parse(
            [
                "--isolated",
                "--set",
                "bench.warmups=7",
                "profile",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--context",
                "512",
                "--json",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Profile(options) = command else {
            panic!("expected profile command");
        };
        assert_eq!(options.engine, PathBuf::from("/engine.wasm"));
        assert_eq!(options.context, 512);
        assert_eq!(options.warmups, 7);
        assert!(options.json);
        assert_eq!(options.scratch, PathBuf::from("target/forja-bench"));
    }

    #[test]
    fn parses_benchmark_breakdown_flag() {
        let command = parse(
            [
                "bench",
                "--engine",
                "/f32.wasm",
                "--engine",
                "/bf16.wasm",
                "--model-dir",
                "/model",
                "--breakdown",
                "--host-argmax",
                "--backend-option",
                "graph-replay=tier1",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert!(options.breakdown);
        assert_eq!(options.selection, [Selection::HostArgmax]);
        assert_eq!(
            options.engines,
            [PathBuf::from("/f32.wasm"), PathBuf::from("/bf16.wasm")]
        );
        assert_eq!(options.graph_replay, GraphReplay::Tier1);
    }

    #[test]
    fn rejects_zero_benchmark_counts() {
        assert!(
            parse(
                [
                    "bench",
                    "--engine",
                    "/engine.wasm",
                    "--model-dir",
                    "/model",
                    "--reps",
                    "0",
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_sugar_and_set_for_the_same_key() {
        let error = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--reps",
                "5",
                "--set",
                "bench.reps=7",
            ]
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("bench.reps is set by both --reps and --set #1")
        );
    }

    #[test]
    fn parses_benchmark_defaults() {
        let command = parse(
            ["bench", "--engine", "/engine.wasm", "--model-dir", "/model"].map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.pp, 512);
        assert_eq!(options.tg, 128);
        assert_eq!(options.reps, 30);
        assert_eq!(options.graph_replay, GraphReplay::Tier2);
    }

    #[test]
    fn expands_vary_axes_as_a_cartesian_product() {
        let command = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--vary",
                "backend.metal.graph_replay=tier1,tier2",
                "--vary",
                "bench.tg=7,33",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.points.len(), 4);
        assert_eq!(
            options
                .points
                .iter()
                .map(|point| (point.graph_replay, point.tg))
                .collect::<Vec<_>>(),
            [
                (GraphReplay::Tier1, 7),
                (GraphReplay::Tier1, 33),
                (GraphReplay::Tier2, 7),
                (GraphReplay::Tier2, 33),
            ]
        );
        assert!(
            options
                .points
                .iter()
                .all(|point| point.origins.values().any(|origin| *origin == Origin::Vary))
        );
    }

    #[test]
    fn parses_toml_array_axes_and_replaces_repeated_keys() {
        let command = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--vary",
                "bench.selection=[[\"host-argmax\"],[\"gpu-pipelined\"]]",
                "--vary",
                "bench.selection=[[\"gpu-sequential\"]]",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.points.len(), 1);
        assert_eq!(options.points[0].selection, [Selection::GpuSequential]);
    }

    #[test]
    fn varies_benchmark_sampling_parameters() {
        let command = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--vary",
                "bench.sampling.temperature=0.0,0.7",
                "--vary",
                "bench.sampling.top_p=1.0,0.9",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.points.len(), 4);
        assert_eq!(
            options
                .points
                .iter()
                .map(|point| {
                    (
                        point.config.bench.sampling.temperature.to_bits(),
                        point.config.bench.sampling.top_p.to_bits(),
                    )
                })
                .collect::<Vec<_>>(),
            [
                (0.0_f32.to_bits(), 1.0_f32.to_bits()),
                (0.0_f32.to_bits(), 0.9_f32.to_bits()),
                (0.7_f32.to_bits(), 1.0_f32.to_bits()),
                (0.7_f32.to_bits(), 0.9_f32.to_bits()),
            ]
        );
    }

    #[test]
    fn rejects_invalid_benchmark_sampling_parameters() {
        for setting in [
            "bench.sampling.temperature=-1.0",
            "bench.sampling.temperature=nan",
            "bench.sampling.top_p=0.0",
            "bench.sampling.top_p=1.1",
            "bench.sampling.top_p=nan",
        ] {
            assert!(
                parse(
                    [
                        "bench",
                        "--engine",
                        "/engine.wasm",
                        "--model-dir",
                        "/model",
                        "--set",
                        setting,
                    ]
                    .map(str::to_owned),
                )
                .is_err(),
                "accepted {setting}"
            );
        }
    }

    #[test]
    fn classifies_axes_and_refuses_non_axes() {
        assert_eq!(
            axis_class("backend.metal.graph_replay"),
            Ok(AxisClass::Strategy)
        );
        assert_eq!(axis_class("engine.tunings"), Ok(AxisClass::Tuning));
        assert_eq!(axis_class("bench.contexts"), Ok(AxisClass::Workload));
        assert_eq!(
            axis_class("bench.sampling.temperature"),
            Ok(AxisClass::Workload)
        );
        for key in [
            "limits.live_bytes",
            "paths.models",
            "run.max_tokens",
            "verify.prompts",
            "backend.kind",
            "bench.reps",
        ] {
            assert!(axis_class(key).is_err(), "accepted {key}");
        }
    }

    #[test]
    fn rejects_non_array_rhs_with_toml_punctuation() {
        assert!(parse_vary("bench.tg=\"7\",\"33\"").is_err());
        assert!(parse_vary("bench.tg=7,").is_err());
        assert!(parse_vary("limits.tensor_rank=7,33").is_err());
    }

    #[test]
    fn explains_that_tunings_require_engine_profiles() {
        let error = parse_vary("engine.tunings=fast,small").unwrap_err();
        assert_eq!(error, "tunings arrive with engine profiles");
    }

    #[test]
    fn file_values_survive_without_bench_sugar_flags() {
        let command = parse_with_file(
            &["bench", "--engine", "/engine.wasm", "--model-dir", "/model"],
            r#"[bench]
pp = 33
tg = 7
reps = 2
breakdown = true
selection = ["host-argmax"]
"#,
        );
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.pp, 33);
        assert_eq!(options.tg, 7);
        assert_eq!(options.reps, 2);
        assert!(options.breakdown);
        assert_eq!(options.selection, [Selection::HostArgmax]);
    }

    #[test]
    fn rejects_invalid_benchmark_options() {
        for arguments in [
            vec!["bench", "--model-dir", "/model"],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--pp",
                "many",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--tg",
                "0",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--precision",
                "bf16",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--no-replay",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--backend-option",
                "graph-replay=tier3",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--json",
                "/one",
                "--json",
                "/two",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--breakdown",
                "--breakdown",
            ],
            vec![
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--wat",
                "value",
            ],
        ] {
            assert!(parse(arguments.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn parses_run_options() {
        let command = parse(
            [
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--max-tokens",
                "7",
                "--temperature",
                "0.7",
                "--top-k",
                "40",
                "--top-p",
                "0.9",
                "--seed",
                "42",
                "--backend",
                "cpu",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            command,
            Command::Run(Run {
                engine: PathBuf::from("/engine.wasm"),
                model_dir: PathBuf::from("/model"),
                prompt: "Hello".to_owned(),
                max_tokens: 7,
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                seed: Some(42),
                backend: Backend::Cpu,
                graph_replay: GraphReplay::Tier2,
                limits: Limits::default(),
            })
        );
    }

    #[test]
    fn parses_run_defaults() {
        let command = parse(
            [
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(options.max_tokens, 128);
        assert_eq!(options.temperature.to_bits(), 0.0_f32.to_bits());
        assert_eq!(options.top_k, 0);
        assert_eq!(options.top_p.to_bits(), 1.0_f32.to_bits());
        assert_eq!(options.seed, None);
        assert_eq!(options.backend, Backend::Metal);
    }

    #[test]
    fn rejects_invalid_sampling_values() {
        for setting in [
            "run.temperature=-1.0",
            "run.temperature=nan",
            "run.top_p=0.0",
            "run.top_p=1.1",
            "run.top_p=nan",
        ] {
            assert!(
                parse(
                    [
                        "run",
                        "--engine",
                        "/engine.wasm",
                        "--model-dir",
                        "/model",
                        "--prompt",
                        "Hello",
                        "--set",
                        setting,
                    ]
                    .map(str::to_owned),
                )
                .is_err(),
                "accepted {setting}"
            );
        }
    }

    #[test]
    fn file_value_survives_without_backend_sugar_flag() {
        let command = parse_with_file(
            &[
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
            ],
            "[backend]\nkind = \"cpu\"\n",
        );
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(options.backend, Backend::Cpu);
    }

    #[test]
    fn reads_run_settings_from_config_without_flag_defaults() {
        let command = parse_with_file(
            &[
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
            ],
            "[run]\nmax_tokens = 7\n",
        );
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(options.max_tokens, 7);
    }

    #[test]
    fn rejects_invalid_run_options() {
        for arguments in [
            vec!["run", "--model-dir", "/model", "--prompt", "Hello"],
            vec!["run", "--engine", "/engine.wasm", "--prompt", "Hello"],
            vec!["run", "--engine", "/engine.wasm", "--model-dir", "/model"],
            vec![
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--backend",
                "neural",
            ],
            vec![
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--max-tokens",
                "many",
            ],
            vec![
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--model-dir",
                "/other",
                "--prompt",
                "Hello",
            ],
            vec![
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--wat",
                "value",
            ],
        ] {
            assert!(parse(arguments.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn parses_verify_options() {
        let command = parse(
            [
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--backend",
                "cpu",
                "--precision",
                "bf16",
                "--prompts",
                "one,two",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            command,
            Command::Verify(Verify {
                engine: PathBuf::from("/engine.wasm"),
                model_dir: PathBuf::from("/model"),
                fixtures: PathBuf::from("/fixtures"),
                backend: Backend::Cpu,
                precision: Precision::Bf16,
                prompts: vec!["one".to_owned(), "two".to_owned()],
                graph_replay: GraphReplay::Tier2,
                limits: Limits::default(),
            })
        );
    }

    #[test]
    fn appends_repeated_verify_prompts() {
        let command = parse(
            [
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--prompts",
                "one,two",
                "--prompts",
                "three",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            command,
            Command::Verify(Verify {
                engine: PathBuf::from("/engine.wasm"),
                model_dir: PathBuf::from("/model"),
                fixtures: PathBuf::from("/fixtures"),
                backend: Backend::Metal,
                precision: Precision::F32,
                prompts: vec!["one".to_owned(), "two".to_owned(), "three".to_owned()],
                graph_replay: GraphReplay::Tier2,
                limits: Limits::default(),
            })
        );
    }

    #[test]
    fn parses_verify_defaults() {
        let command = parse(
            [
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Verify(options) = command else {
            panic!("expected verify command");
        };
        assert_eq!(options.backend, Backend::Metal);
        assert_eq!(options.precision, Precision::F32);
        assert!(options.prompts.is_empty());
    }

    #[test]
    fn file_values_survive_without_verify_sugar_flags() {
        let command = parse_with_file(
            &[
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
            ],
            r#"[verify]
prompts = ["short", "code"]
fixtures = "/fixtures"
"#,
        );
        let Command::Verify(options) = command else {
            panic!("expected verify command");
        };
        assert_eq!(options.prompts, ["short", "code"]);
        assert_eq!(options.fixtures, PathBuf::from("/fixtures"));
    }

    #[test]
    fn rejects_missing_and_unknown_options() {
        assert!(parse(["verify"].map(str::to_owned)).is_err());
        assert!(
            parse(["verify", "--model-dir", "/model", "--wat", "value"].map(str::to_owned))
                .is_err()
        );
    }

    #[test]
    fn rejects_an_option_without_a_value() {
        let error = parse(["verify", "--model-dir"].map(str::to_owned)).unwrap_err();
        assert!(error.to_string().contains("--model-dir"));
    }

    #[test]
    fn rejects_empty_prompt_names() {
        assert!(
            parse(
                [
                    "verify",
                    "--engine",
                    "/engine.wasm",
                    "--model-dir",
                    "/model",
                    "--fixtures",
                    "/fixtures",
                    "--prompts",
                    "one,,two",
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_invalid_verify_options() {
        for arguments in [
            vec![
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--backend",
                "neural",
            ],
            vec![
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--backend",
                "cpu",
                "--backend",
                "metal",
            ],
            vec![
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--precision",
                "bf16",
                "--precision",
                "f32",
            ],
            vec![
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--fixtures",
                "/fixtures",
                "--precision",
                "int8",
            ],
            vec![
                "verify",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--model-dir",
                "/other",
                "--fixtures",
                "/fixtures",
            ],
        ] {
            assert!(parse(arguments.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn layers_config_files_before_set_values() {
        let root = temporary_directory("cli-layers").unwrap();
        let user_path = root.join("user.toml");
        let first_path = root.join("first.toml");
        let second_path = root.join("second.toml");
        fs::write(&user_path, "[limits]\nlive_graphs = 33\ntensor_rank = 1\n").unwrap();
        fs::write(&first_path, "[limits]\ntensor_rank = 7\n").unwrap();
        fs::write(&second_path, "[limits]\ntensor_rank = 22\n").unwrap();
        let mut user = file_layer::<DevConfig>(&user_path).unwrap();
        user.origin = forja_config::Origin::UserFile(user_path);
        let command = parse_with(
            [
                "--config".to_owned(),
                first_path.display().to_string(),
                "run".to_owned(),
                "--engine".to_owned(),
                "/engine.wasm".to_owned(),
                "--model-dir".to_owned(),
                "/model".to_owned(),
                "--prompt".to_owned(),
                "hello".to_owned(),
                "--set".to_owned(),
                "limits.tensor_rank=11".to_owned(),
                "-c".to_owned(),
                second_path.display().to_string(),
                "-s".to_owned(),
                "limits.live_kernels=4097".to_owned(),
            ],
            |_| Ok(vec![user]),
        )
        .unwrap();
        let Command::Run(options) = command else {
            panic!("expected run command");
        };
        assert_eq!(options.limits.live_graphs, 33);
        assert_eq!(options.limits.tensor_rank, 11);
        assert_eq!(options.limits.live_kernels, 4097);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_numbered_set_errors_exactly() {
        let error = parse_with(
            [
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "hello",
                "--set",
                "limits.tensor_ranks=5",
            ]
            .map(str::to_owned),
            |_| Ok(Vec::new()),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string().lines().next().unwrap(),
            "error: --set #1 limits.tensor_ranks=5: unknown field `tensor_ranks`, expected one of `live_bytes`, `tensor_rank`, `tensor_elements`, `live_tensor_handles`, `live_kernels`, `live_graphs`, `read_bytes`, `guest_memory_bytes`, `table_elements`, `instances`, `dispatches_per_list`, `work_per_dispatch`, `guest_call_timeout`, `submission_timeout`, `gpu_time_budget`"
        );
    }

    #[test]
    fn passes_isolated_to_the_base_layer_builder() {
        parse_with(
            [
                "--isolated",
                "run",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--prompt",
                "hello",
            ]
            .map(str::to_owned),
            |isolated| {
                assert!(isolated);
                Ok(Vec::new())
            },
        )
        .unwrap();
    }

    #[test]
    fn rerun_implies_isolated_and_layers_the_recorded_config() {
        let root = temporary_directory("cli-rerun").unwrap();
        let path = root.join("record.json");
        fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 2,
                "provenance": {"commit": "recorded"},
                "inputs": [],
                "config": "[bench]\ntg = 7\n",
                "axes": {},
                "results": [],
            }))
            .unwrap(),
        )
        .unwrap();
        let command = parse_with(
            [
                "bench".to_owned(),
                "--engine".to_owned(),
                "/engine.wasm".to_owned(),
                "--model-dir".to_owned(),
                "/model".to_owned(),
                "--rerun".to_owned(),
                path.display().to_string(),
                "--allow-diff".to_owned(),
                "bench.reps".to_owned(),
            ],
            |isolated| {
                assert!(isolated);
                Ok(Vec::new())
            },
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert_eq!(options.tg, 7);
        assert!(options.rerun.is_some());
        assert_eq!(options.allow_diff, [KeyPath::new("bench.reps")]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn allow_diff_requires_rerun() {
        let error = parse(
            [
                "bench",
                "--engine",
                "/engine.wasm",
                "--model-dir",
                "/model",
                "--allow-diff",
                "bench.reps",
            ]
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.to_string().contains("--allow-diff requires --rerun"));
    }

    #[test]
    fn prints_root_and_command_help() {
        let root = parse(["--help"].map(str::to_owned)).unwrap_err();
        assert_eq!(root.kind(), clap::error::ErrorKind::DisplayHelp);
        let root = root.to_string();
        assert!(root.contains("bench"));
        assert!(root.contains("run"));
        assert!(root.contains("verify"));

        for command in ["bench", "run", "verify"] {
            let help = parse([command, "--help"].map(str::to_owned)).unwrap_err();
            assert_eq!(help.kind(), clap::error::ErrorKind::DisplayHelp);
            assert!(help.to_string().contains("--model-dir"));
        }
    }
}
