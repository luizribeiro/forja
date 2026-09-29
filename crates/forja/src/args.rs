use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphReplay {
    Tier1,
    Tier2,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Verify {
    pub(crate) engine: PathBuf,
    pub(crate) model_dir: PathBuf,
    pub(crate) fixtures: PathBuf,
    pub(crate) backend: Backend,
    pub(crate) precision: Precision,
    pub(crate) prompts: Vec<String>,
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
    fixtures: PathBuf,
    /// Compute backend.
    #[arg(long, value_enum, default_value_t = Backend::Metal)]
    backend: Backend,
    /// Engine precision.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    precision: Precision,
    /// Comma-separated fixture names. Repeating the option appends names.
    #[arg(long, value_delimiter = ',', value_parser = parse_prompt_name)]
    prompts: Vec<String>,
}

#[derive(Args, Debug, Eq, PartialEq)]
pub(crate) struct Run {
    /// WebAssembly engine component.
    #[arg(long)]
    pub(crate) engine: PathBuf,
    /// Directory containing model weights and tokenizer files.
    #[arg(long)]
    pub(crate) model_dir: PathBuf,
    /// Text to continue.
    #[arg(long)]
    pub(crate) prompt: String,
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 128)]
    pub(crate) max_tokens: usize,
    /// Compute backend.
    #[arg(long, value_enum, default_value_t = Backend::Metal)]
    pub(crate) backend: Backend,
}

#[derive(Args, Debug, Eq, PartialEq)]
pub(crate) struct Bench {
    /// WebAssembly engine component. Repeat to compare engines.
    #[arg(long = "engine", required = true)]
    pub(crate) engines: Vec<PathBuf>,
    /// Directory containing model weights.
    #[arg(long)]
    pub(crate) model_dir: PathBuf,
    /// Number of prompt-processing tokens.
    #[arg(long, default_value_t = 512, value_parser = parse_positive)]
    pub(crate) pp: usize,
    /// Number of token-generation tokens.
    #[arg(long, default_value_t = 128, value_parser = parse_positive)]
    pub(crate) tg: usize,
    /// Number of measured repetitions.
    #[arg(long, default_value_t = 30, value_parser = parse_positive)]
    pub(crate) reps: usize,
    /// Path for the JSON report.
    #[arg(long)]
    pub(crate) json: Option<PathBuf>,
    /// Print a per-operation timing breakdown.
    #[arg(long)]
    pub(crate) breakdown: bool,
    /// Select tokens on the host instead of comparing selection modes.
    #[arg(long)]
    pub(crate) host_argmax: bool,
    /// Set a Metal backend option.
    #[arg(
        long = "backend-option",
        default_value = "graph-replay=tier2",
        value_parser = parse_backend_option
    )]
    pub(crate) graph_replay: GraphReplay,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Command {
    Bench(Bench),
    Run(Run),
    Verify(Verify),
}

#[derive(Parser)]
#[command(name = "forja", about = "Run and inspect Forja inference engines")]
struct Cli {
    #[command(subcommand)]
    command: ParsedCommand,
}

#[derive(Subcommand)]
enum ParsedCommand {
    /// Benchmark one or more engines.
    Bench(Bench),
    /// Generate a completion.
    Run(Run),
    /// Compare engine outputs with golden fixtures.
    Verify(VerifyArgs),
}

pub(crate) fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Command, clap::Error> {
    let cli = Cli::try_parse_from(std::iter::once("forja".to_owned()).chain(arguments))?;
    Ok(match cli.command {
        ParsedCommand::Bench(options) => Command::Bench(options),
        ParsedCommand::Run(options) => Command::Run(options),
        ParsedCommand::Verify(options) => Command::Verify(options.into()),
    })
}

fn parse_positive(value: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| "value must be a positive integer".to_owned())
}

fn parse_backend_option(value: &str) -> Result<GraphReplay, String> {
    match value {
        "graph-replay=tier1" => Ok(GraphReplay::Tier1),
        "graph-replay=tier2" => Ok(GraphReplay::Tier2),
        _ => Err(format!("unknown Metal backend option {value:?}")),
    }
}

fn parse_prompt_name(value: &str) -> Result<String, String> {
    if value.is_empty() {
        Err("prompt name cannot be empty".to_owned())
    } else {
        Ok(value.to_owned())
    }
}

impl From<VerifyArgs> for Verify {
    fn from(options: VerifyArgs) -> Self {
        Self {
            engine: options.engine,
            model_dir: options.model_dir,
            fixtures: options.fixtures,
            backend: options.backend,
            precision: options.precision,
            prompts: options.prompts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                json: Some(PathBuf::from("/result.json")),
                breakdown: false,
                host_argmax: false,
                graph_replay: GraphReplay::Tier2,
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
        assert!(options.host_argmax);
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
