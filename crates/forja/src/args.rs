use std::{collections::BTreeMap, path::PathBuf};

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand, ValueEnum};
use forja_config::{
    BackendKind, ConfigError, DevConfig, GraphReplay, KeyPath, Layer, Layered, Limits, Origin,
    Selection, dev_layers, file_layer, layer, set_layer,
};

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

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Run {
    pub(crate) engine: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) prompt: String,
    pub(crate) max_tokens: usize,
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
    /// Compute backend.
    #[arg(long, value_enum)]
    backend: Option<Backend>,
    /// Set a Metal backend option.
    #[arg(long = "backend-option", hide = true, value_parser = parse_backend_option)]
    graph_replay: Option<GraphReplay>,
}

#[derive(Debug, Eq, PartialEq)]
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
}

#[derive(Debug, PartialEq)]
pub(crate) enum Command {
    Bench(Bench),
    Config(ConfigShow),
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
    let mut layers = base_layers(cli.isolated).map_err(|error| config_error(&error))?;
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
        ParsedCommand::Bench(options) => {
            Command::Bench(options.with_config(&layered.config, limits)?)
        }
        ParsedCommand::Config(options) => match options.command {
            ConfigCommand::Show(options) => Command::Config(ConfigShow {
                layered,
                origin: options.origin,
                key: options.key,
                json: options.json,
                defaults: options.defaults,
            }),
        },
        ParsedCommand::Run(options) => Command::Run(options.with_config(&layered.config, limits)?),
        ParsedCommand::Verify(options) => {
            Command::Verify(options.with_config(&layered.config, limits)?)
        }
    })
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
            }
            Self::Run(options) => {
                values.extend(
                    options
                        .max_tokens
                        .map(|value| ("--max-tokens", value.to_string())),
                );
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
            Self::Config(_) => {}
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
            toml::Value::String(graph_replay_name(value).to_owned()).to_string(),
        )
    })
}

fn sugar_layer(flag: &'static str, value: &str) -> Result<(Layer, KeyPath), clap::Error> {
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
    fn with_config(self, config: &DevConfig, limits: Limits) -> Result<Bench, clap::Error> {
        let count = |value: u32| {
            usize::try_from(value).map_err(|error| {
                Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
            })
        };
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
        })
    }
}

impl RunArgs {
    fn with_config(self, config: &DevConfig, limits: Limits) -> Result<Run, clap::Error> {
        let max_tokens = usize::try_from(config.run.max_tokens).map_err(|error| {
            Cli::command().error(clap::error::ErrorKind::ValueValidation, error.to_string())
        })?;
        Ok(Run {
            engine: self.engine,
            model_dir: self.model_dir,
            prompt: self.prompt,
            max_tokens,
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

const fn graph_replay_name(value: GraphReplay) -> &'static str {
    match value {
        GraphReplay::Tier1 => "tier1",
        GraphReplay::Tier2 => "tier2",
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
        assert_eq!(
            command,
            Command::Bench(Bench {
                engines: vec![PathBuf::from("/engine.wasm")],
                model_dir: PathBuf::from("/model"),
                pp: 33,
                tg: 7,
                reps: 2,
                warmups: 3,
                decode_prefill: 8,
                contexts: vec![9, 512, 2_048, 4_000],
                selection: vec![Selection::GpuSequential, Selection::GpuPipelined],
                json: Some(PathBuf::from("/result.json")),
                breakdown: false,
                graph_replay: GraphReplay::Tier2,
                limits: Limits::default(),
            })
        );
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
        assert_eq!(options.backend, Backend::Metal);
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
