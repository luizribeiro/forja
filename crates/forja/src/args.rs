use std::path::PathBuf;

pub(crate) const USAGE: &str = "usage: forja verify --model-dir PATH --fixtures PATH [--backend metal|cpu] [--prompts NAME,...]";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    Metal,
    Cpu,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Verify {
    pub(crate) model_dir: PathBuf,
    pub(crate) fixtures: PathBuf,
    pub(crate) backend: Backend,
    pub(crate) prompts: Vec<String>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Command {
    Verify(Verify),
}

pub(crate) fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut arguments = arguments.into_iter();
    match arguments.next().as_deref() {
        Some("verify") => parse_verify(arguments).map(Command::Verify),
        Some(command) => Err(format!("unknown command {command:?}")),
        None => Err("a command is required".to_owned()),
    }
}

fn parse_verify(mut arguments: impl Iterator<Item = String>) -> Result<Verify, String> {
    let mut model_dir = None;
    let mut fixtures = None;
    let mut backend = Backend::Metal;
    let mut prompts = Vec::new();
    while let Some(option) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{option} requires a value"))?;
        match option.as_str() {
            "--model-dir" if model_dir.is_none() => model_dir = Some(PathBuf::from(value)),
            "--fixtures" if fixtures.is_none() => fixtures = Some(PathBuf::from(value)),
            "--backend" => {
                backend = match value.as_str() {
                    "metal" => Backend::Metal,
                    "cpu" => Backend::Cpu,
                    _ => return Err(format!("unknown backend {value:?}")),
                };
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
        prompts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
