use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Default, PartialEq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Mode {
    #[default]
    Plain,
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
    /// Reuse a separately prepared prompt prefix before each measured request.
    #[arg(long)]
    pub cached: bool,
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

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    #[derive(clap::Parser)]
    struct Parser {
        #[command(flatten)]
        options: super::Options,
    }

    #[test]
    fn plain_is_the_default() {
        let arguments = [
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
        let options = Parser::try_parse_from(arguments.clone()).unwrap().options;
        assert_eq!(options.mode, super::Mode::Plain);
        assert!(!options.cached);
        assert_eq!(options.group, "regular");
        assert_eq!(options.reps, 5);
        let mut draft = arguments;
        draft.extend(["--mode", "draft"].map(str::to_owned));
        assert!(Parser::try_parse_from(draft).is_err());
    }
}
