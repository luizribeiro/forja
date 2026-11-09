use std::{error::Error, fs, path::PathBuf};

use serde_json::Value;

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Plain-mode completion benchmark record.
    #[arg(long)]
    pub(crate) plain: PathBuf,
    /// Draft-mode completion benchmark record to check.
    #[arg(long)]
    pub(crate) draft: PathBuf,
}

pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    assert_matches(
        &serde_json::from_slice(&fs::read(&options.plain)?)?,
        &serde_json::from_slice(&fs::read(&options.draft)?)?,
    )?;
    println!("token streams match");
    Ok(())
}

pub(crate) fn assert_matches(plain: &Value, draft: &Value) -> Result<(), String> {
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
    if plain["stream_config"] != draft["stream_config"] {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_mismatches_and_missing_exactness_evidence() {
        let plain = serde_json::json!({"kind":"completion-benchmark", "schema_version":1,
            "mode":"plain", "stream_config":{"engine":"component", "sampling":"greedy"},
            "results":[{"identity":{"seed":42,"prompt_sha256":format!("sha256:{}", "b".repeat(64)),"concurrency":1,"reps":1},
                "token_stream_hashes":[format!("sha256:{}", "a".repeat(64))]}]});
        let mut draft = plain.clone();
        draft["mode"] = "draft".into();
        assert_matches(&plain, &draft).unwrap();
        for field in ["seed", "prompt_sha256", "reps", "concurrency"] {
            let mut incomplete = plain.clone();
            incomplete["results"][0]["identity"][field] = Value::Null;
            let mut other = incomplete.clone();
            other["mode"] = "draft".into();
            assert!(assert_matches(&incomplete, &other).is_err());
        }
        let mut uncovered = draft.clone();
        uncovered["results"][0]["identity"]["concurrency"] = 4.into();
        let mut baseline = uncovered.clone();
        baseline["mode"] = "plain".into();
        assert!(assert_matches(&baseline, &uncovered).is_err());
        for pointer in [
            "/mode",
            "/stream_config/engine",
            "/stream_config/sampling",
            "/results/0/identity/seed",
            "/results/0/identity/prompt_sha256",
            "/results/0/identity/reps",
            "/results/0/identity/concurrency",
            "/results/0/token_stream_hashes/0",
        ] {
            let mut changed = draft.clone();
            *changed.pointer_mut(pointer).unwrap() = Value::Null;
            assert!(assert_matches(&plain, &changed).is_err(), "{pointer}");
        }
    }
}
