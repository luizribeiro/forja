use std::path::PathBuf;

use clap::{Args, FromArgMatches, ValueEnum};

pub(crate) const USAGE: &str = "usage:
  forja run --model-dir PATH --prompt TEXT [--max-tokens N] [--backend metal|cpu]
  forja bench --model-dir PATH [--precision f32|bf16] [--pp N] [--tg N] [--reps N] [--json PATH] [--profile] [--host-argmax] [--no-replay] [--backend-option graph-replay=tier1|tier2]
  forja verify --model-dir PATH --fixtures PATH [--backend metal|cpu] [--precision f32|bf16] [--prompts NAME,...]";

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
    pub(crate) model_dir: PathBuf,
    pub(crate) fixtures: PathBuf,
    pub(crate) backend: Backend,
    pub(crate) precision: Precision,
    pub(crate) prompts: Vec<String>,
}

#[derive(Args, Debug, Eq, PartialEq)]
pub(crate) struct Run {
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
    pub(crate) profile: bool,
    /// Select tokens on the host instead of comparing selection modes.
    #[arg(long)]
    pub(crate) host_argmax: bool,
    /// Use an engine that records its graph lazily.
    #[arg(long)]
    pub(crate) no_replay: bool,
    /// Benchmark one precision instead of both.
    #[arg(long, value_enum)]
    pub(crate) precision: Option<Precision>,
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

pub(crate) fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut arguments = arguments.into_iter();
    match arguments.next().as_deref() {
        Some("bench") => parse_bench(arguments).map(Command::Bench),
        Some("run") => parse_run(arguments).map(Command::Run),
        Some("verify") => parse_verify(arguments).map(Command::Verify),
        Some(command) => Err(format!("unknown command {command:?}")),
        None => Err("a command is required".to_owned()),
    }
}

fn parse_bench(arguments: impl Iterator<Item = String>) -> Result<Bench, String> {
    let command = Bench::augment_args(clap::Command::new("bench"));
    let matches = command
        .try_get_matches_from(std::iter::once("bench".to_owned()).chain(arguments))
        .map_err(|error| error.to_string())?;
    Bench::from_arg_matches(&matches).map_err(|error| error.to_string())
}

fn parse_positive(value: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| "value must be a positive integer".to_owned())
}

fn parse_run(mut arguments: impl Iterator<Item = String>) -> Result<Run, String> {
    let command = Run::augment_args(clap::Command::new("run"));
    let matches = command
        .try_get_matches_from(std::iter::once("run".to_owned()).chain(arguments.by_ref()))
        .map_err(|error| error.to_string())?;
    Run::from_arg_matches(&matches).map_err(|error| error.to_string())
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "metal" => Ok(Backend::Metal),
        "cpu" => Ok(Backend::Cpu),
        _ => Err(format!("unknown backend {value:?}")),
    }
}

fn parse_precision(value: &str) -> Result<Precision, String> {
    match value {
        "f32" => Ok(Precision::F32),
        "bf16" => Ok(Precision::Bf16),
        _ => Err(format!("unknown precision {value:?}")),
    }
}

fn parse_backend_option(value: &str) -> Result<GraphReplay, String> {
    match value {
        "graph-replay=tier1" => Ok(GraphReplay::Tier1),
        "graph-replay=tier2" => Ok(GraphReplay::Tier2),
        _ => Err(format!("unknown Metal backend option {value:?}")),
    }
}

fn parse_verify(mut arguments: impl Iterator<Item = String>) -> Result<Verify, String> {
    let mut model_dir = None;
    let mut fixtures = None;
    let mut backend = Backend::Metal;
    let mut precision = Precision::F32;
    let mut prompts = Vec::new();
    while let Some(option) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{option} requires a value"))?;
        match option.as_str() {
            "--model-dir" if model_dir.is_none() => model_dir = Some(PathBuf::from(value)),
            "--fixtures" if fixtures.is_none() => fixtures = Some(PathBuf::from(value)),
            "--backend" => {
                backend = parse_backend(&value)?;
            }
            "--precision" => {
                precision = parse_precision(&value)?;
            }
            "--prompts" => {
                let names = value.split(',').collect::<Vec<_>>();
                if names.iter().any(|name| name.is_empty()) {
                    return Err("--prompts cannot contain an empty name".to_owned());
                }
                prompts.extend(names.into_iter().map(str::to_owned));
            }
            _ if option.starts_with("--") => {
                return Err(format!("unknown or repeated option {option:?}"));
            }
            _ => return Err(format!("unexpected argument {option:?}")),
        }
    }
    Ok(Verify {
        model_dir: model_dir.ok_or("--model-dir is required")?,
        fixtures: fixtures.ok_or("--fixtures is required")?,
        backend,
        precision,
        prompts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_benchmark_options() {
        let command = parse(
            [
                "bench",
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
                model_dir: PathBuf::from("/model"),
                pp: 33,
                tg: 7,
                reps: 2,
                json: Some(PathBuf::from("/result.json")),
                profile: false,
                host_argmax: false,
                no_replay: false,
                precision: None,
                graph_replay: GraphReplay::Tier2,
            })
        );
    }

    #[test]
    fn parses_benchmark_profile_flag() {
        let command = parse(
            [
                "bench",
                "--model-dir",
                "/model",
                "--profile",
                "--host-argmax",
                "--no-replay",
                "--precision",
                "bf16",
                "--backend-option",
                "graph-replay=tier1",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let Command::Bench(options) = command else {
            panic!("expected bench command");
        };
        assert!(options.profile);
        assert!(options.host_argmax);
        assert!(options.no_replay);
        assert_eq!(options.precision, Some(Precision::Bf16));
        assert_eq!(options.graph_replay, GraphReplay::Tier1);
    }

    #[test]
    fn rejects_zero_benchmark_counts() {
        assert!(
            parse(["bench", "--model-dir", "/model", "--reps", "0"].map(str::to_owned)).is_err()
        );
    }

    #[test]
    fn parses_benchmark_defaults() {
        let command = parse(["bench", "--model-dir", "/model"].map(str::to_owned)).unwrap();
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
            vec!["bench"],
            vec!["bench", "--model-dir", "/model", "--pp", "many"],
            vec!["bench", "--model-dir", "/model", "--tg", "0"],
            vec!["bench", "--model-dir", "/model", "--precision", "int8"],
            vec![
                "bench",
                "--model-dir",
                "/model",
                "--backend-option",
                "graph-replay=tier3",
            ],
            vec![
                "bench",
                "--model-dir",
                "/model",
                "--json",
                "/one",
                "--json",
                "/two",
            ],
            vec!["bench", "--model-dir", "/model", "--profile", "--profile"],
            vec!["bench", "--model-dir", "/model", "--wat", "value"],
        ] {
            assert!(parse(arguments.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn parses_run_options() {
        let command = parse(
            [
                "run",
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
                model_dir: PathBuf::from("/model"),
                prompt: "Hello".to_owned(),
                max_tokens: 7,
                backend: Backend::Cpu,
            })
        );
    }

    #[test]
    fn parses_run_defaults() {
        let command =
            parse(["run", "--model-dir", "/model", "--prompt", "Hello"].map(str::to_owned))
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
            vec!["run", "--prompt", "Hello"],
            vec!["run", "--model-dir", "/model"],
            vec![
                "run",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--backend",
                "neural",
            ],
            vec![
                "run",
                "--model-dir",
                "/model",
                "--prompt",
                "Hello",
                "--max-tokens",
                "many",
            ],
            vec![
                "run",
                "--model-dir",
                "/model",
                "--model-dir",
                "/other",
                "--prompt",
                "Hello",
            ],
            vec![
                "run",
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
                model_dir: PathBuf::from("/model"),
                fixtures: PathBuf::from("/fixtures"),
                backend: Backend::Cpu,
                precision: Precision::Bf16,
                prompts: vec!["one".to_owned(), "two".to_owned()],
            })
        );
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
        assert_eq!(error, "--model-dir requires a value");
    }

    #[test]
    fn rejects_empty_prompt_names() {
        assert!(
            parse(
                [
                    "verify",
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
}
