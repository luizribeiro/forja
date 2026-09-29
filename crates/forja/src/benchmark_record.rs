use std::collections::BTreeMap;

use forja_config::{Choice, GraphReplay, KeyPath, Origin, Selection};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::args::{Bench, BenchPoint};

pub(crate) const SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Serialize)]
pub(crate) struct Input {
    pub(crate) engine_sha256: String,
    pub(crate) profile_hash: Option<String>,
    pub(crate) weights_sha256: String,
    pub(crate) model_revision: Option<String>,
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

#[derive(Serialize)]
struct PerfKey<'a> {
    backend: GraphReplay,
    pp: usize,
    tg: usize,
    reps: usize,
    warmups: usize,
    decode_prefill: usize,
    contexts: &'a [usize],
    selection: &'a [Selection],
    breakdown: bool,
    engine_sha256: &'a str,
    weights_sha256: &'a str,
}

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

pub(crate) fn perf_hash(
    options: &Bench,
    point: &BenchPoint,
    input: &Input,
) -> Result<String, String> {
    hash(&PerfKey {
        backend: point.graph_replay,
        pp: point.pp,
        tg: point.tg,
        reps: options.reps,
        warmups: options.warmups,
        decode_prefill: options.decode_prefill,
        contexts: &point.contexts,
        selection: &point.selection,
        breakdown: options.breakdown,
        engine_sha256: &input.engine_sha256,
        weights_sha256: &input.weights_sha256,
    })
}

pub(crate) fn combined_perf_hash(hashes: &[String]) -> Result<String, String> {
    hash(hashes)
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
    use forja_config::DevConfig;

    fn options() -> Bench {
        let config = DevConfig::default();
        Bench {
            engines: vec!["engine.wasm".into()],
            model_dir: "model".into(),
            pp: 512,
            tg: 7,
            reps: 3,
            warmups: 1,
            decode_prefill: 8,
            contexts: vec![9, 512, 4_000],
            selection: vec![Selection::GpuPipelined],
            json: None,
            breakdown: false,
            graph_replay: GraphReplay::Tier2,
            limits: config.limits.clone(),
            config: Box::new(config),
            origins: BTreeMap::new(),
            axes: BTreeMap::new(),
            strategy_axes: Vec::new(),
            points: Vec::new(),
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
            profile_hash: None,
            weights_sha256: "weights".to_owned(),
            model_revision: None,
        }];
        assert_eq!(
            config_hash(&snapshot, &inputs).unwrap(),
            "sha256:95d94399d6e91a69d57e5a7236df00cd71becc644ed58068e954222579f4fe96"
        );
        assert_eq!(
            perf_hash(&options, &point, &inputs[0]).unwrap(),
            "sha256:37cb92a99e577a083df83232c34cd1db744c181b3f232ed3fc1cc67c4eeccbc6"
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
}
