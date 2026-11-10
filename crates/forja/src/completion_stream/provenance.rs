use super::Comparison;
use serde_json::Value;

pub(super) fn compare(plain: &Value, draft: &Value, comparison: Comparison) -> Result<(), String> {
    for record in [plain, draft] {
        let provenance = &record["provenance"];
        if provenance["dirty"] != false || provenance["clean_source"] != true {
            return Err("stream checking refuses dirty or unverified source records".to_owned());
        }
        let binary = provenance["binary_sha256"]
            .as_str()
            .ok_or("missing binary hash")?;
        if binary.len() != 64 || !binary.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("invalid binary hash".to_owned());
        }
        for field in ["commit", "device"] {
            if provenance[field].as_str().is_none_or(str::is_empty) {
                return Err(format!("missing provenance {field}"));
            }
        }
    }
    for field in [
        "device",
        "machine",
        "architecture",
        "os",
        "os_version",
        "os_build",
        "build_environment",
        "gpu_cores",
        "memory_bytes",
    ] {
        if plain["provenance"].get(field).is_none()
            || plain["provenance"][field] != draft["provenance"][field]
        {
            return Err(format!("stream provenance differs or is missing: {field}"));
        }
    }
    if comparison == Comparison::Candidate {
        for field in ["binary_sha256", "commit"] {
            if plain["provenance"][field] != draft["provenance"][field] {
                return Err(format!(
                    "candidate stream checking requires the same {field}"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn stream_config(record: &Value, comparison: Comparison) -> Value {
    let mut config = record["stream_config"].clone();
    if comparison == Comparison::BaselineVsCandidate
        && let Some(inputs) = config["inputs"].as_object_mut()
    {
        inputs.remove("engine_sha256");
        inputs.remove("profile_hash");
    }
    config
}
