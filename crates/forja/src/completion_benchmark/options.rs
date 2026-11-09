use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Default, PartialEq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Mode {
    #[default]
    Plain,
    Draft,
}

#[derive(Clone, Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Engine component with an embedded profile.
    #[arg(long)]
    pub engine: PathBuf,
    /// Directory containing the full model and tokenizer.
    #[arg(long)]
    pub model: PathBuf,
    /// Model-compatible chat-format component.
    #[arg(long)]
    pub chat_format: PathBuf,
    /// JSON workload definitions and prompt provenance.
    #[arg(long)]
    pub suite: PathBuf,
    /// Workload group; long contexts must be selected separately.
    #[arg(long, default_value = "regular", value_parser = ["regular", "long"])]
    pub group: String,
    /// Row names; omitted selects every row in the chosen group.
    #[arg(long, value_delimiter = ',')]
    pub row: Vec<String>,
    /// Decode normally or enable the embedded profile's prompt-lookup policy.
    #[arg(long, value_enum, default_value_t = Mode::Plain)]
    pub mode: Mode,
    /// Plain record with identical workload, sampling and configuration.
    #[arg(long, required_if_eq("mode", "draft"))]
    pub plain_record: Option<PathBuf>,
    /// Reuse a separately prepared prompt prefix before each measured request.
    #[arg(long)]
    pub cached: bool,
    #[command(flatten)]
    pub measurement: Measurement,
    /// Independent whole request-group repetitions.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    pub reps: u32,
    /// Warm-up groups with the row concurrency and enough output for lookup.
    #[arg(long, default_value_t = 1)]
    pub warmups: u32,
    /// New record destination; existing files are refused.
    #[arg(long)]
    pub json: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, clap::Args)]
pub(crate) struct Measurement {
    /// Retain the loaded runner for immutable sample requests on stdin.
    #[arg(long, conflicts_with = "cached")]
    pub worker: bool,
    /// Retain individual samples (always enabled for completion records).
    #[arg(long)]
    pub raw_samples: bool,
    /// Explicitly retain fixed repetitions without sequential stopping.
    #[arg(long)]
    pub no_early_stop: bool,
}

#[cfg(test)]
mod tests {
    fn parse(arguments: Vec<String>) -> Result<super::Options, clap::Error> {
        let crate::args::Command::BenchCompletion(options, _) = crate::args::parse(arguments)?
        else {
            panic!("expected completion benchmark");
        };
        Ok(options)
    }

    #[test]
    fn plain_is_the_default() {
        let arguments = [
            "bench",
            "completion",
            "--engine",
            "engine.wasm",
            "--model",
            "model",
            "--chat-format",
            "chat.wasm",
            "--suite",
            "workload.json",
            "--json",
            "output.json",
        ]
        .map(str::to_owned)
        .to_vec();
        let options = parse(arguments.clone()).unwrap();
        assert_eq!(options.mode, super::Mode::Plain);
        assert!(!options.cached);
        assert_eq!(options.group, "regular");
        assert_eq!(options.reps, 5);
        let mut worker = arguments.clone();
        worker.extend(["--worker", "--raw-samples", "--no-early-stop"].map(str::to_owned));
        assert!(parse(worker.clone()).unwrap().measurement.worker);
        worker.push("--cached".to_owned());
        assert!(parse(worker).is_err());
        let mut draft = arguments;
        draft.extend(["--mode", "draft"].map(str::to_owned));
        assert!(parse(draft.clone()).is_err());
        draft.extend(["--plain-record", "plain.json"].map(str::to_owned));
        assert!(parse(draft).is_ok());
    }
}
