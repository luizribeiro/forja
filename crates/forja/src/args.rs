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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Bench {
    pub(crate) model_dir: PathBuf,
    pub(crate) pp: usize,
    pub(crate) tg: usize,
    pub(crate) reps: usize,
    pub(crate) json: Option<PathBuf>,
    pub(crate) profile: bool,
    pub(crate) host_argmax: bool,
    pub(crate) no_replay: bool,
    pub(crate) precision: Option<Precision>,
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

fn parse_bench(mut arguments: impl Iterator<Item = String>) -> Result<Bench, String> {
    let mut model_dir = None;
    let mut pp = None;
    let mut tg = None;
    let mut reps = None;
    let mut json = None;
    let mut profile = false;
    let mut host_argmax = false;
    let mut no_replay = false;
    let mut precision = None;
    let mut graph_replay = None;
    while let Some(option) = arguments.next() {
        if option == "--profile" && !profile {
            profile = true;
            continue;
        }
        if option == "--no-replay" && !no_replay {
            no_replay = true;
            continue;
        }
        if option == "--host-argmax" && !host_argmax {
            host_argmax = true;
            continue;
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("{option} requires a value"))?;
        match option.as_str() {
            "--model-dir" if model_dir.is_none() => model_dir = Some(PathBuf::from(value)),
            "--pp" if pp.is_none() => pp = Some(parse_count(&value, "prompt tokens")?),
            "--tg" if tg.is_none() => tg = Some(parse_count(&value, "generated tokens")?),
            "--reps" if reps.is_none() => reps = Some(parse_count(&value, "repetitions")?),
            "--json" if json.is_none() => json = Some(PathBuf::from(value)),
            "--precision" if precision.is_none() => precision = Some(parse_precision(&value)?),
            "--backend-option" if graph_replay.is_none() => {
                graph_replay = Some(parse_backend_option(&value)?);
            }
            _ if option.starts_with("--") => {
                return Err(format!("unknown or repeated option {option:?}"));
            }
            _ => return Err(format!("unexpected argument {option:?}")),
        }
    }
    Ok(Bench {
        model_dir: model_dir.ok_or("--model-dir is required")?,
        pp: pp.unwrap_or(512),
        tg: tg.unwrap_or(128),
        reps: reps.unwrap_or(30),
        json,
        profile,
        host_argmax,
        no_replay,
        precision,
        graph_replay: graph_replay.unwrap_or(GraphReplay::Tier2),
    })
}

fn parse_count(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|&count| count > 0)
        .ok_or_else(|| format!("{name} must be a positive integer"))
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
