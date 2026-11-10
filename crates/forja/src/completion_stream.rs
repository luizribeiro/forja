mod provenance;
#[cfg(test)]
pub(crate) mod tests;

use std::{error::Error, fs, path::PathBuf};

use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Comparison {
    #[default]
    Candidate,
    BaselineVsCandidate,
}

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Plain-mode completion benchmark record.
    #[arg(long)]
    pub(crate) plain: PathBuf,
    /// Draft-mode completion benchmark record to check.
    #[arg(long)]
    pub(crate) draft: PathBuf,
    /// Require candidate identity or explicitly compare a labelled baseline.
    #[arg(long, value_enum, default_value_t = Comparison::Candidate)]
    pub(crate) comparison: Comparison,
}

pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    assert_matches(
        &serde_json::from_slice(&fs::read(&options.plain)?)?,
        &serde_json::from_slice(&fs::read(&options.draft)?)?,
        options.comparison,
    )?;
    println!(
        "token streams match; comparison={}",
        serde_json::to_string(&options.comparison)?
    );
    Ok(())
}

pub(crate) fn assert_matches(
    plain: &Value,
    draft: &Value,
    comparison: Comparison,
) -> Result<(), String> {
    for (record, mode) in [(plain, "plain"), (draft, "draft")] {
        if record["kind"] != "completion-benchmark"
            || record["schema_version"] != 1
            || record["mode"] != mode
            || record["stream_config"]
                .as_object()
                .is_none_or(serde_json::Map::is_empty)
        {
            return Err(format!(
                "expected a complete {mode} completion benchmark record"
            ));
        }
    }
    provenance::compare(plain, draft, comparison)?;
    if provenance::stream_config(plain, comparison) != provenance::stream_config(draft, comparison)
    {
        return Err("plain/draft stream configuration differs".to_owned());
    }
    let rows = |record: &Value| {
        record["results"]
            .as_array()
            .cloned()
            .filter(|rows| !rows.is_empty())
            .ok_or_else(|| "missing stream rows".to_owned())
    };
    let (plain, draft) = (rows(plain)?, rows(draft)?);
    if plain.len() != draft.len() {
        return Err("plain/draft row coverage differs".to_owned());
    }
    for (plain, draft) in plain.iter().zip(&draft) {
        if !plain["identity"].is_object() || plain["identity"] != draft["identity"] {
            return Err("plain/draft seed, prompt or workload differs".to_owned());
        }
        if hashes(plain)? != hashes(draft)? {
            return Err(format!(
                "token-stream hash mismatch for {}",
                draft["identity"]
            ));
        }
    }
    Ok(())
}

fn hashes(row: &Value) -> Result<&[Value], String> {
    let identity = &row["identity"];
    if identity["seed"].as_u64().is_none() || !valid_hash(&identity["prompt_sha256"]) {
        return Err("stream identity requires a seed and prompt hash".to_owned());
    }
    let count = identity["concurrency"]
        .as_u64()
        .filter(|count| *count > 0)
        .and_then(|count| {
            identity["reps"]
                .as_u64()
                .filter(|reps| *reps > 0)
                .and_then(|reps| count.checked_mul(reps))
        })
        .ok_or_else(|| "invalid stream sample coverage".to_owned())?;
    row["token_stream_hashes"]
        .as_array()
        .filter(|hashes| {
            u64::try_from(hashes.len()).ok() == Some(count) && hashes.iter().all(valid_hash)
        })
        .map(Vec::as_slice)
        .ok_or_else(|| "missing or malformed token-stream hashes".to_owned())
}

fn valid_hash(value: &Value) -> bool {
    value
        .as_str()
        .and_then(|hash| hash.strip_prefix("sha256:"))
        .is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
}
