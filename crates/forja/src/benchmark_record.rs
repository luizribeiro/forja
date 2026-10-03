use std::collections::BTreeMap;

use forja_config::{Choice, DevConfig, GraphReplay, KeyPath, Numerics, Origin, Profile};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::args::{Bench, BenchPoint};

pub(crate) const SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct Input {
    pub(crate) engine_sha256: String,
    #[serde(default, skip_serializing)]
    pub(crate) engine_variant: String,
    #[serde(default, skip_serializing)]
    pub(crate) engine_build_profile: String,
    #[serde(default)]
    pub(crate) profile_name: String,
    #[serde(default)]
    pub(crate) profile_family: String,
    #[serde(default)]
    pub(crate) numerics: Option<Numerics>,
    #[serde(default)]
    pub(crate) profile_hash: Option<String>,
    pub(crate) weights_sha256: String,
    #[serde(default)]
    pub(crate) model_revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct RecordedProvenance {
    pub(crate) commit: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct Recorded {
    pub(crate) schema_version: u32,
    pub(crate) provenance: RecordedProvenance,
    pub(crate) inputs: Vec<Input>,
    #[serde(default)]
    pub(crate) config: String,
    #[serde(default)]
    pub(crate) axes: BTreeMap<KeyPath, Vec<toml::Value>>,
    #[serde(default)]
    pub(crate) results: Vec<serde_json::Value>,
}

pub(crate) struct Snapshot {
    pub(crate) config: String,
    pub(crate) origins: BTreeMap<String, String>,
    pub(crate) auto_notes: Vec<String>,
    canonical: toml::Value,
}

#[derive(Serialize)]
struct ConfigKey<'a> {
    schema_version: u32,
    config: &'a toml::Value,
    inputs: &'a [Input],
}

pub(crate) type PerfKey = BTreeMap<String, serde_json::Value>;

pub(crate) fn snapshot(options: &Bench, device: &str, macos: &str) -> Result<Snapshot, String> {
    let mut config = (*options.config).clone();
    let automatic = matches!(config.backend.metal.graph_replay, Choice::Auto);
    config.backend.metal.graph_replay = Choice::Fixed(options.graph_replay);
    let mut table = toml::Value::try_from(config)
        .map_err(|error| format!("cannot serialize benchmark config: {error}"))?
        .as_table()
        .cloned()
        .ok_or_else(|| "benchmark config is not a table".to_owned())?;
    table.remove("paths");
    if let Some(toml::Value::Table(verify)) = table.get_mut("verify") {
        verify.remove("fixtures");
    }
    let canonical = toml::Value::Table(table);
    let config = toml::to_string(&canonical)
        .map_err(|error| format!("cannot render benchmark config: {error}"))?;
    let mut origins: BTreeMap<String, String> = options
        .origins
        .iter()
        .filter(|(key, _)| !path_bearing(key))
        .map(|(key, origin)| (key.as_str().to_owned(), record_origin(origin)))
        .collect();
    if automatic {
        origins.insert("backend.metal.graph_replay".to_owned(), "auto".to_owned());
    }
    let auto_notes = automatic
        .then(|| {
            format!(
                "backend.metal.graph_replay: auto -> {} ({device}/macOS {macos} @{})",
                options.graph_replay.as_str(),
                crate::provenance::commit()
            )
        })
        .into_iter()
        .collect();
    Ok(Snapshot {
        config,
        origins,
        auto_notes,
        canonical,
    })
}

pub(crate) fn config_hash(snapshot: &Snapshot, inputs: &[Input]) -> Result<String, String> {
    hash(&ConfigKey {
        schema_version: SCHEMA_VERSION,
        config: &snapshot.canonical,
        inputs,
    })
}

pub(crate) fn perf_key(
    point: &BenchPoint,
    input: &Input,
    profile: &Profile,
) -> Result<PerfKey, String> {
    let mut key = perf_key_from(&point.config, point.graph_replay, input)?;
    let selection = point.config.engine.resolve(profile)?;
    key.insert(
        "engine.tunings".to_owned(),
        serde_json::json!(selection.tunings),
    );
    key.insert(
        "engine.picks_hash".to_owned(),
        serde_json::json!(hash(&selection.variants)?),
    );
    Ok(key)
}

pub(crate) fn combined_perf_hash(hashes: &[String]) -> Result<String, String> {
    hash(hashes)
}

pub(crate) fn comparison_hash(key: &PerfKey) -> Result<String, String> {
    hash(key)
}

pub(crate) fn recorded_perf_keys(record: &Recorded) -> Result<Vec<PerfKey>, String> {
    let base = toml::from_str(&record.config)
        .map_err(|error| format!("record config is invalid: {error}"))?;
    record
        .results
        .iter()
        .map(|result| recorded_perf_key(record, &base, result))
        .collect()
}

fn recorded_perf_key(
    record: &Recorded,
    base: &toml::Table,
    result: &serde_json::Value,
) -> Result<PerfKey, String> {
    let mut layers = vec![forja_config::Layer::new(Origin::Vary, base.clone())];
    let point = result["point"]
        .as_object()
        .ok_or_else(|| "record result point is not an object".to_owned())?;
    for (key, value) in point {
        let value = toml::Value::try_from(value.clone())
            .map_err(|error| format!("record point {key} is invalid: {error}"))?;
        let table = toml::from_str(&format!("{key} = {value}"))
            .map_err(|error| format!("record point {key} is invalid: {error}"))?;
        layers.push(forja_config::Layer::new(Origin::Vary, table));
    }
    let config = forja_config::layer::<DevConfig>(layers)
        .map_err(|error| format!("record point is invalid: {error}"))?
        .config;
    let input = result["input"]
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| record.inputs.get(index))
        .ok_or_else(|| "record result input is invalid".to_owned())?;
    let replay = config.backend.metal.resolve().graph_replay;
    let mut key = perf_key_from(&config, replay, input)?;
    if record.schema_version >= 3 {
        for field in ["engine.tunings", "engine.picks_hash"] {
            let value = result[field].clone();
            if value.is_null() {
                return Err(format!("record result is missing {field}"));
            }
            key.insert(field.to_owned(), value);
        }
    }
    Ok(key)
}

fn perf_key_from(
    config: &DevConfig,
    graph_replay: GraphReplay,
    input: &Input,
) -> Result<PerfKey, String> {
    let mut key = BTreeMap::new();
    key.insert("backend.kind".to_owned(), serde_json::json!("metal"));
    key.insert(
        "backend.metal.graph_replay".to_owned(),
        serde_json::to_value(graph_replay)
            .map_err(|error| format!("cannot serialize graph replay: {error}"))?,
    );
    key.insert("bench.pp".to_owned(), serde_json::json!(config.bench.pp));
    key.insert("bench.tg".to_owned(), serde_json::json!(config.bench.tg));
    key.insert(
        "bench.reps".to_owned(),
        serde_json::json!(config.bench.reps),
    );
    key.insert(
        "bench.warmups".to_owned(),
        serde_json::json!(config.bench.warmups),
    );
    key.insert(
        "bench.decode_prefill".to_owned(),
        serde_json::json!(config.bench.decode_prefill),
    );
    key.insert(
        "bench.contexts".to_owned(),
        serde_json::json!(config.bench.contexts.as_slice()),
    );
    key.insert(
        "bench.selection".to_owned(),
        serde_json::json!(config.bench.selection.as_slice()),
    );
    key.insert(
        "bench.sampling".to_owned(),
        serde_json::to_value(config.bench.sampling)
            .map_err(|error| format!("cannot serialize benchmark sampling: {error}"))?,
    );
    key.insert(
        "bench.breakdown".to_owned(),
        serde_json::json!(config.bench.breakdown),
    );
    key.insert(
        "engine.sha256".to_owned(),
        serde_json::json!(input.engine_sha256),
    );
    key.insert(
        "weights.sha256".to_owned(),
        serde_json::json!(input.weights_sha256),
    );
    Ok(key)
}

pub(crate) fn output_digest(tokens: &[u32], logits: &[u8]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    digest.update(logits);
    format!("sha256:{:x}", digest.finalize())
}

fn hash(value: &(impl Serialize + ?Sized)) -> Result<String, String> {
    let bytes =
        serde_json::to_vec(value).map_err(|error| format!("cannot hash record: {error}"))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn path_bearing(key: &KeyPath) -> bool {
    key.as_str().starts_with("paths.") || key.as_str() == "verify.fixtures"
}

fn record_origin(origin: &Origin) -> String {
    match origin {
        Origin::File(_) => "config file".to_owned(),
        Origin::UserFile(_) => "user config".to_owned(),
        _ => origin.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> Profile {
        crate::resolution::read_embedded_profile(test_guests::qwen3()).unwrap()
    }

    fn options() -> Bench {
        let mut config = DevConfig::default();
        config.bench.tg = std::num::NonZeroU32::new(7).unwrap();
        config.bench.reps = std::num::NonZeroU32::new(3).unwrap();
        config.bench.warmups = 1;
        let contexts = config
            .bench
            .contexts
            .as_slice()
            .iter()
            .map(|&value| usize::try_from(value).unwrap())
            .collect();
        let selection = config.bench.selection.as_slice().to_vec();
        Bench {
            engines: vec!["engine.wasm".into()],
            model_dir: "model".into(),
            pp: 512,
            tg: 7,
            reps: 3,
            warmups: 1,
            decode_prefill: 8,
            contexts,
            selection,
            json: None,
            breakdown: false,
            graph_replay: GraphReplay::Tier2,
            limits: config.limits.clone(),
            config: Box::new(config),
            origins: BTreeMap::new(),
            axes: BTreeMap::new(),
            strategy_axes: Vec::new(),
            points: Vec::new(),
            rerun: None,
            allow_diff: Vec::new(),
        }
    }

    fn point(options: &Bench) -> BenchPoint {
        BenchPoint {
            pp: options.pp,
            tg: options.tg,
            contexts: options.contexts.clone(),
            selection: options.selection.clone(),
            graph_replay: options.graph_replay,
            config: options.config.clone(),
            origins: options.origins.clone(),
            values: BTreeMap::new(),
        }
    }

    #[test]
    fn config_and_performance_hashes_are_deterministic() {
        let options = options();
        let point = point(&options);
        let snapshot = snapshot(&options, "Apple M3 Ultra", "26.6").unwrap();
        let inputs = vec![Input {
            engine_sha256: "engine".to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        }];
        assert_eq!(
            config_hash(&snapshot, &inputs).unwrap(),
            "sha256:f65725041f3d0b6b5e8d34201039ab0d3662e56599e88c65e57c8838be69e9f4"
        );
        assert_eq!(
            comparison_hash(&perf_key(&point, &inputs[0], &profile()).unwrap()).unwrap(),
            "sha256:c03d73dae7a2ee964e660e751afb218b21bc15ae1309ef9391485c1d2b9d563f"
        );
    }

    #[test]
    fn snapshot_has_no_machine_paths_and_is_a_valid_layer() {
        let mut options = options();
        options.config.paths.models = Some("/private/models".into());
        options.config.verify.fixtures = Some("/private/fixtures".into());
        let snapshot = snapshot(&options, "Apple M3 Ultra", "26.6").unwrap();
        assert!(!snapshot.config.contains("/private"));
        let table = toml::from_str(&snapshot.config).unwrap();
        forja_config::layer::<DevConfig>(vec![forja_config::Layer::new(Origin::Vary, table)])
            .unwrap();
    }

    #[test]
    fn output_digest_has_a_stable_byte_order() {
        assert_eq!(
            output_digest(&[1, 0x1020_3040], &[5, 6]),
            "sha256:b8097b75ea7e84e2fb279fc7d7f62b6dc2f19af32be084035b0b8b2906580404"
        );
    }

    #[test]
    fn reconstructs_performance_keys_from_recorded_points() {
        let options = options();
        let point = point(&options);
        let input = Input {
            engine_sha256: "engine".to_owned(),
            engine_variant: "variant".to_owned(),
            engine_build_profile: "release".to_owned(),
            profile_name: String::new(),
            profile_family: String::new(),
            numerics: None,
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        };
        let expected = perf_key(&point, &input, &profile()).unwrap();
        let record = Recorded {
            schema_version: SCHEMA_VERSION,
            provenance: RecordedProvenance {
                commit: "recorded".to_owned(),
            },
            inputs: vec![input.clone()],
            config: snapshot(&options, "Apple M3 Ultra", "26.6").unwrap().config,
            axes: BTreeMap::new(),
            results: vec![serde_json::json!({
                "input": 0,
                "point": {},
                "engine.tunings": expected["engine.tunings"],
                "engine.picks_hash": expected["engine.picks_hash"],
            })],
        };
        assert_eq!(recorded_perf_keys(&record).unwrap(), [expected]);
    }

    #[test]
    fn v3_inputs_omit_legacy_filename_and_path_identity() {
        let input = Input {
            engine_sha256: "engine".to_owned(),
            engine_variant: "filename".to_owned(),
            engine_build_profile: "path-derived".to_owned(),
            profile_name: "qwen3-test".to_owned(),
            profile_family: "qwen3".to_owned(),
            numerics: Some(*profile().numerics()),
            profile_hash: Some("sha256:profile".to_owned()),
            weights_sha256: "sha256:weights".to_owned(),
            model_revision: Some("revision".to_owned()),
        };
        let value = serde_json::to_value(input).unwrap();
        assert!(value.get("engine_variant").is_none());
        assert!(value.get("engine_build_profile").is_none());
        assert_eq!(value["profile_name"], "qwen3-test");
    }
}
