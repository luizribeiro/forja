use std::{error::Error, fs, path::PathBuf, process::Command};

use clap::{Args, ValueEnum};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Args, Debug, PartialEq, Serialize)]
pub(crate) struct Options {
    /// Shared HTTP workload type.
    #[arg(long, value_enum, default_value_t = Mode::Conversation)]
    mode: Mode,
    /// Suite with the model and workload inputs.
    #[arg(long)]
    suite: PathBuf,
    /// URL of the server owned by this benchmark.
    #[arg(long)]
    url: String,
    /// JSON argv array for starting the server.
    #[arg(long)]
    server: PathBuf,
    /// Frozen artifact identities and effective server settings.
    #[arg(long)]
    identity: PathBuf,
    /// New JSON benchmark record path.
    #[arg(long)]
    json: PathBuf,
    /// Number of independent whole workload repetitions.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    reps: u32,
    /// Number of bounded initialization rounds per fresh workload.
    #[arg(long, default_value_t = 1)]
    warmups: u32,
    /// Retain individual requests and measurement windows.
    #[arg(long)]
    raw_samples: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Mode {
    Conversation,
    Serving,
}

const SOURCES: &[(&str, &str)] = &[
    (
        "http_serving.py",
        include_str!("../../../support/bench-serving/http_serving.py"),
    ),
    (
        "http_entry.py",
        include_str!("../../../support/bench-serving/http_entry.py"),
    ),
    (
        "conversation_identity.py",
        include_str!("../../../support/bench-serving/conversation_identity.py"),
    ),
    (
        "conversation_session.py",
        include_str!("../../../support/bench-serving/conversation_session.py"),
    ),
    (
        "http_client.py",
        include_str!("../../../support/bench-serving/http_client.py"),
    ),
    (
        "multi_turn.py",
        include_str!("../../../support/bench-serving/multi_turn.py"),
    ),
    (
        "serving_bench.py",
        include_str!("../../../support/bench-serving/serving_bench.py"),
    ),
    (
        "provenance.py",
        include_str!("../../../support/bench-serving/provenance.py"),
    ),
    (
        "priming.py",
        include_str!("../../../support/bench-serving/priming.py"),
    ),
    (
        "suite.py",
        include_str!("../../../support/bench-serving/suite.py"),
    ),
    (
        "bench_stats.py",
        include_str!("../../../support/bench_stats.py"),
    ),
];

pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    if options.json.exists() {
        return Err("refusing to overwrite a benchmark record".into());
    }
    let directory = options.json.with_extension("http-client");
    fs::create_dir_all(directory.parent().ok_or("benchmark path has no parent")?)?;
    fs::create_dir(&directory)?;
    let directory = directory.canonicalize()?;
    let mut sources = serde_json::Map::new();
    for (name, source) in SOURCES {
        fs::write(directory.join(name), source)?;
        sources.insert(
            (*name).to_owned(),
            json!(format!("{:x}", Sha256::digest(source.as_bytes()))),
        );
    }
    let mut document = serde_json::to_value(options)?;
    let binary = std::env::current_exe()?;
    document["provenance"] = json!({"commit": crate::provenance::commit(),
        "dirty": crate::provenance::dirty(), "build_environment": crate::provenance::BUILD_ENVIRONMENT,
        "binary_sha256": golden_fixtures::sha256_file(&binary)?, "client_sources": sources});
    let input = directory.join("options.json");
    fs::write(&input, serde_json::to_vec_pretty(&document)?)?;
    let python = std::env::var_os("FORJA_BENCH_PYTHON").unwrap_or_else(|| "python3".into());
    let mut command = Command::new(python);
    command.arg(directory.join("http_entry.py")).arg(input);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        Err(command.exec().into())
    }
    #[cfg(not(unix))]
    {
        if command.status()?.success() {
            Ok(())
        } else {
            Err("HTTP benchmark failed".into())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn http_mode_has_its_own_inputs_without_an_engine_argument() {
        let command = crate::args::parse(
            [
                "bench",
                "http",
                "--warmups",
                "3",
                "--suite",
                "suite.toml",
                "--url",
                "http://127.0.0.1:1",
                "--server",
                "command.json",
                "--identity",
                "identity.json",
                "--json",
                "record.json",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let crate::args::Command::BenchHttp(options) = command else {
            panic!("expected HTTP benchmark command");
        };
        assert_eq!(options.warmups, 3);
        assert!(
            crate::args::parse(
                [
                    "bench",
                    "--engine",
                    "engine.wasm",
                    "http",
                    "--suite",
                    "suite.toml",
                ]
                .map(str::to_owned)
            )
            .is_err()
        );
    }
}
