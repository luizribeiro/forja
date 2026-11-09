use std::error::Error;

use sha2::{Digest, Sha256};

pub(super) fn hash(value: &(impl serde::Serialize + ?Sized)) -> Result<String, Box<dyn Error>> {
    Ok(format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(value)?)
    ))
}

pub(super) fn decorate(document: &mut serde_json::Value) -> Result<(), Box<dyn Error>> {
    use serde_json::json;
    let identities = document["results"]
        .as_array()
        .ok_or("missing completion results")?
        .iter()
        .map(|row| &row["identity"])
        .collect::<Vec<_>>();
    document["config_hash"] = json!(hash(
        &json!({"config":document["stream_config"], "rows":identities})
    )?);
    document["perf_hash"] = json!(hash(
        &json!({"config_hash":document["config_hash"], "mode":document["mode"],
        "execution":document["measurement_mode"]["execution"],
        "binary_sha256":document["provenance"]["binary_sha256"], "clean_source":document["provenance"]["clean_source"], "device":document["provenance"]["device"]})
    )?);
    for row in document["results"]
        .as_array_mut()
        .ok_or("missing completion results")?
    {
        row["token_stream_hash"] = json!(hash(&row["token_stream_hashes"])?);
        row["output_digest"] = row["token_stream_hash"].clone();
        for field in [
            "wall_seconds",
            "decode_wall_seconds",
            "gpu_seconds",
            "plain_gpu_seconds",
            "verify_gpu_seconds",
        ] {
            let samples = row["samples"]
                .as_array()
                .ok_or("missing completion samples")?;
            let values = samples
                .iter()
                .map(|sample| sample[field].as_f64().ok_or("invalid timing sample"))
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty()
                || values.iter().any(|value| {
                    !value.is_finite()
                        || *value < 0.0
                        || (field == "decode_wall_seconds" && *value <= 0.0)
                })
            {
                return Err("empty or invalid completion timing samples".into());
            }
            let stats = crate::benchmark_stats::stats(values.into_iter());
            row[field] = json!({"median":stats.median, "ci95":[stats.low,stats.high]});
        }
    }
    Ok(())
}

pub(super) fn prompt_hash(prompt: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(prompt.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn record_keys_bind_workload_identity_mode_and_execution() {
        let mut plain = json!({"mode":"plain", "stream_config":{"seed":42},
            "measurement_mode":{"execution":"scheduled-request-groups"},
            "results":[{"identity":{"prompt_sha256":"prompt"},"token_stream_hashes":["tokens"],
                "samples":[{"wall_seconds":5.0,"decode_wall_seconds":4.0,"gpu_seconds":1.0,
                    "plain_gpu_seconds":0.9,"verify_gpu_seconds":0.1}]}]});
        decorate(&mut plain).unwrap();
        for field in ["binary_sha256", "clean_source", "device"] {
            let mut changed = plain.clone();
            changed["provenance"][field] = json!("different");
            decorate(&mut changed).unwrap();
            assert_ne!(plain["perf_hash"], changed["perf_hash"], "{field}");
            assert_eq!(plain["config_hash"], changed["config_hash"]);
        }
        for span in [0.0, -1.0] {
            let mut invalid = plain.clone();
            invalid["results"][0]["samples"][0]["decode_wall_seconds"] = json!(span);
            assert!(decorate(&mut invalid).is_err());
        }
        let mut draft = plain.clone();
        draft["mode"] = json!("draft");
        decorate(&mut draft).unwrap();
        assert_eq!(plain["config_hash"], draft["config_hash"]);
        assert_ne!(plain["perf_hash"], draft["perf_hash"]);
        assert_eq!(
            plain["results"][0]["token_stream_hash"],
            draft["results"][0]["token_stream_hash"]
        );
        draft["results"][0]["identity"]["prompt_sha256"] = json!("changed");
        decorate(&mut draft).unwrap();
        assert_ne!(plain["config_hash"], draft["config_hash"]);
        assert_eq!(
            plain["results"][0]["wall_seconds"],
            json!({"median":5.0,"ci95":[5.0,5.0]})
        );
        assert_eq!(
            prompt_hash(""),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
