use std::{
    error::Error,
    fs,
    num::TryFromIntError,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    time::Duration,
};

use forja_config::{
    ComponentTarget, DeviceFamily, GraphReplay, Limits as ConfigLimits, Profile, ProfileBackend,
    Target, Unbounded, Variants, sharded_weights_sha256, single_weights_sha256,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{args::Backend, resolution::read_embedded_profile};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WeightFileIdentity {
    canonical_path: PathBuf,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    inode: u64,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WeightHashCache {
    index_sha256: Option<String>,
    files: Vec<WeightFileIdentity>,
    weights_sha256: String,
}

#[derive(Clone, Copy)]
struct TargetIdentity<'a> {
    component: &'a str,
    backend: &'a str,
    device_family: Option<&'a str>,
}

pub(crate) fn validate_engine(
    component: &Path,
    model_dir: &Path,
    backend: Backend,
    scratch: &Path,
) -> Result<(Profile, PathBuf), Box<dyn Error>> {
    let profile = read_embedded_profile(component)?;
    let weights = weights_path(model_dir)?;
    let (weights_hash, _) = strong_weights_sha256(&weights, scratch)?;
    let backend = match backend {
        Backend::Metal => "metal",
        Backend::Cpu => "cpu",
    };
    let device = if backend == "metal" {
        #[cfg(target_os = "macos")]
        {
            let name = forja_metal::device_name()?;
            Some(device_family(&name)?)
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    } else {
        None
    };
    validate_identity(
        &profile,
        &weights_hash,
        TargetIdentity {
            component: "wasm32-wasip2",
            backend,
            device_family: device,
        },
    )?;
    Ok((profile, weights))
}

fn validate_identity(
    profile: &Profile,
    weights_hash: &str,
    host: TargetIdentity<'_>,
) -> Result<(), String> {
    if profile.model().weights_sha256 != weights_hash {
        return Err(format!(
            "model weights hash mismatch: profile requires {}, found {weights_hash}",
            profile.model().weights_sha256
        ));
    }
    let required = target_identity(profile.target());
    if host.component != required.component {
        return Err(format!(
            "component target mismatch: profile requires {}, found {}",
            required.component, host.component
        ));
    }
    if host.backend != required.backend {
        return Err(format!(
            "backend mismatch: profile requires {}, found {}",
            required.backend, host.backend
        ));
    }
    if host.device_family != required.device_family {
        return Err(format!(
            "device family mismatch: profile requires {}, found {}",
            required.device_family.unwrap_or("unavailable"),
            host.device_family.unwrap_or("unavailable")
        ));
    }
    Ok(())
}

const fn target_identity(target: &Target) -> TargetIdentity<'static> {
    TargetIdentity {
        component: match target.component {
            ComponentTarget::Wasm32Wasip2 => "wasm32-wasip2",
        },
        backend: match target.backend {
            ProfileBackend::Metal => "metal",
            ProfileBackend::Cpu => "cpu",
        },
        device_family: Some(match target.device_family {
            DeviceFamily::AppleM3Ultra => "apple-m3-ultra",
        }),
    }
}

fn device_family(name: &str) -> Result<&'static str, String> {
    if name == "Apple M3 Ultra" {
        Ok("apple-m3-ultra")
    } else {
        Err(format!("unsupported Metal device family: {name}"))
    }
}

fn strong_weights_sha256(weights: &Path, scratch: &Path) -> Result<(String, bool), Box<dyn Error>> {
    let sources = forja_host::Safetensors::open_all(weights)?;
    let index_sha256 = if sources.iter().any(|source| source.shard_name().is_some()) {
        let length = weights.metadata()?.len();
        Some(single_weights_sha256(weights, length)?)
    } else {
        None
    };
    let before = sources
        .iter()
        .map(weight_file_identity)
        .collect::<Result<Vec<_>, _>>()?;
    let cache_path = weight_cache_path(scratch, weights);
    if let Ok(bytes) = fs::read(&cache_path)
        && let Ok(cache) = serde_json::from_slice::<WeightHashCache>(&bytes)
        && cache.index_sha256 == index_sha256
        && cache.files == before
    {
        // Scratch is trusted local user state; model directories and component profiles are not.
        return Ok((cache.weights_sha256, true));
    }
    let weights_sha256 = if sources.len() == 1 && sources[0].shard_name().is_none() {
        single_weights_sha256(sources[0].path(), sources[0].file_len())?
    } else {
        let shards = sources
            .iter()
            .map(|source| {
                Ok((
                    source
                        .shard_name()
                        .ok_or("indexed weight has no shard name")?,
                    source.path(),
                    source.file_len(),
                ))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        sharded_weights_sha256(&shards)?
    };
    let after = sources
        .iter()
        .map(weight_file_identity)
        .collect::<Result<Vec<_>, _>>()?;
    if after != before {
        return Err("model weights changed while hashing".into());
    }
    let cache = WeightHashCache {
        index_sha256,
        files: after,
        weights_sha256: weights_sha256.clone(),
    };
    let parent = cache_path.parent().ok_or("weights cache has no parent")?;
    fs::create_dir_all(parent)?;
    let mut bytes = serde_json::to_vec(&cache)?;
    bytes.push(b'\n');
    fs::write(cache_path, bytes)?;
    Ok((weights_sha256, false))
}

fn weight_file_identity(
    source: &forja_host::Safetensors,
) -> Result<WeightFileIdentity, Box<dyn Error>> {
    let path = fs::canonicalize(source.path())?;
    let metadata = path.symlink_metadata()?;
    if !metadata.is_file() || metadata.len() != source.file_len() {
        return Err(format!("weight source changed after validation: {}", path.display()).into());
    }
    Ok(WeightFileIdentity {
        canonical_path: path,
        size: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        inode: metadata.ino(),
    })
}

fn weight_cache_path(scratch: &Path, weights: &Path) -> PathBuf {
    let name = format!(
        "{:x}.json",
        Sha256::digest(weights.as_os_str().as_encoded_bytes())
    );
    scratch.join("weights-sha256").join(name)
}

pub(crate) fn engine_load_config(component: &Path) -> Result<forja_host::EngineLoadConfig, String> {
    let profile = read_embedded_profile(component)?;
    Ok(profile_load_config(&profile))
}

fn profile_load_config(profile: &Profile) -> forja_host::EngineLoadConfig {
    let mut variants = profile.variants().clone();
    for tuning in profile.default_tunings() {
        if let Some(overrides) = profile.tuning_variants().get(tuning) {
            overlay_variants(&mut variants, overrides);
        }
    }
    forja_host::EngineLoadConfig {
        replay: true,
        tunings: profile.default_tunings().to_vec(),
        fixed_variant_picks: variants
            .fixed
            .into_iter()
            .map(|(site, name)| forja_host::FixedVariantPick { site, name })
            .collect(),
        variant_rule_picks: variants
            .rules
            .into_iter()
            .map(|(site, rule)| forja_host::VariantRulePick {
                site,
                parameter: rule.parameter,
                arms: rule
                    .arms
                    .into_iter()
                    .map(|arm| forja_host::VariantPickArm {
                        lo: arm.lo,
                        hi: arm.hi,
                        name: arm.name,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn overlay_variants(base: &mut Variants, overrides: &Variants) {
    for (site, name) in &overrides.fixed {
        base.rules.remove(site);
        base.fixed.insert(site.clone(), name.clone());
    }
    for (site, rule) in &overrides.rules {
        base.fixed.remove(site);
        base.rules.insert(site.clone(), rule.clone());
    }
}

struct HostLimits(forja_host::Limits);

impl TryFrom<&ConfigLimits> for HostLimits {
    type Error = TryFromIntError;

    fn try_from(limits: &ConfigLimits) -> Result<Self, Self::Error> {
        let dispatches_per_list = match limits.dispatches_per_list {
            Unbounded::Unlimited => usize::MAX,
            Unbounded::Limited(value) => usize::try_from(value)?,
        };
        let work_per_dispatch = match limits.work_per_dispatch {
            Unbounded::Unlimited => u64::MAX,
            Unbounded::Limited(value) => value,
        };
        let gpu_time_budget = match limits.gpu_time_budget {
            Unbounded::Unlimited => Duration::MAX,
            Unbounded::Limited(value) => value.get(),
        };
        Ok(Self(
            forja_host::Limits::new(
                limits.live_bytes.get(),
                usize::try_from(limits.tensor_rank)?,
                limits.tensor_elements,
                usize::try_from(limits.live_tensor_handles)?,
                limits.read_bytes.get(),
            )
            .with_kernel_limit(usize::try_from(limits.live_kernels)?)
            .with_graph_limit(usize::try_from(limits.live_graphs)?)
            .with_store_limits(
                usize::try_from(limits.guest_memory_bytes.get())?,
                usize::try_from(limits.table_elements)?,
                usize::try_from(limits.instances)?,
            )
            .with_command_limits(dispatches_per_list, work_per_dispatch)
            .with_guest_call_timeout(limits.guest_call_timeout.get())
            .with_gpu_limits(limits.submission_timeout.get(), gpu_time_budget),
        ))
    }
}

/// Selects the last greatest value under the IEEE total order.
///
/// Equal values select the later index, and positive NaNs sort above finite values rather than
/// being ignored.
pub(crate) fn argmax(values: &[f32]) -> Result<u32, Box<dyn Error>> {
    let index = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or("cannot take argmax of empty logits")?;
    Ok(u32::try_from(index)?)
}

pub(crate) fn read_token(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "selected token must contain exactly four bytes")?;
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn limits(config: &ConfigLimits) -> Result<forja_host::Limits, TryFromIntError> {
    HostLimits::try_from(config).map(|limits| limits.0)
}

pub(crate) fn weights_path(model_dir: &Path) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(model_dir).map_err(|error| {
        format!(
            "cannot resolve model directory {}: {error}",
            model_dir.display()
        )
    })?;
    for name in ["model.safetensors", "model.safetensors.index.json"] {
        let candidate = model_dir.join(name);
        if !candidate.try_exists().map_err(|error| {
            format!(
                "cannot inspect model weights {}: {error}",
                candidate.display()
            )
        })? {
            continue;
        }
        let resolved = std::fs::canonicalize(&candidate).map_err(|error| {
            format!(
                "cannot resolve model weights {}: {error}",
                candidate.display()
            )
        })?;
        if !resolved.starts_with(&root) {
            return Err(format!(
                "model weights resolve outside the model directory: {}",
                candidate.display()
            ));
        }
        if !resolved.is_file() {
            return Err(format!(
                "model weights are not a file: {}",
                candidate.display()
            ));
        }
        return Ok(resolved);
    }
    Err(format!(
        "model directory has no model.safetensors or model.safetensors.index.json: {}",
        model_dir.display()
    ))
}

#[cfg(target_os = "macos")]
pub(crate) const fn metal_graph_replay(value: GraphReplay) -> forja_metal::MetalGraphReplay {
    match value {
        GraphReplay::Tier1 => forja_metal::MetalGraphReplay::Tier1,
        GraphReplay::Tier2 => forja_metal::MetalGraphReplay::Tier2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::SystemTime, time::UNIX_EPOCH};

    fn profile() -> Profile {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engines/qwen3/profiles/qwen3-0.6b.metal-apple-m3-ultra.bf16.toml");
        toml::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    fn host_target(
        backend: &'static str,
        device_family: Option<&'static str>,
    ) -> TargetIdentity<'static> {
        TargetIdentity {
            component: "wasm32-wasip2",
            backend,
            device_family,
        }
    }

    #[test]
    fn rejects_every_engine_identity_mismatch() {
        let profile = profile();
        let weights = &profile.model().weights_sha256;
        assert!(
            validate_identity(
                &profile,
                "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                host_target("metal", Some("apple-m3-ultra")),
            )
            .unwrap_err()
            .contains("weights hash mismatch")
        );
        assert!(
            validate_identity(
                &profile,
                weights,
                TargetIdentity {
                    component: "wasm32-unknown-unknown",
                    ..host_target("metal", Some("apple-m3-ultra"))
                },
            )
            .unwrap_err()
            .contains("component target mismatch")
        );
        assert!(
            validate_identity(
                &profile,
                weights,
                host_target("cpu", Some("apple-m3-ultra")),
            )
            .unwrap_err()
            .contains("backend mismatch")
        );
        assert!(
            validate_identity(&profile, weights, host_target("metal", Some("apple-m4")),)
                .unwrap_err()
                .contains("device family mismatch")
        );
    }

    #[test]
    fn identity_requirements_come_from_the_profile_target() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engines/qwen3/profiles/qwen3-0.6b.metal-apple-m3-ultra.bf16.toml");
        let source = fs::read_to_string(path)
            .unwrap()
            .replace("backend = \"metal\"", "backend = \"cpu\"");
        let profile: Profile = toml::from_str(&source).unwrap();
        let error = validate_identity(
            &profile,
            &profile.model().weights_sha256,
            host_target("metal", Some("apple-m3-ultra")),
        )
        .unwrap_err();
        assert!(error.contains("requires cpu, found metal"), "{error}");
    }

    #[test]
    fn metal_device_names_match_exactly() {
        assert_eq!(device_family("Apple M3 Ultra").unwrap(), "apple-m3-ultra");
        assert!(device_family("Apple M3 Ultra (80 cores)").is_err());
    }

    fn safetensors(data: &[u8]) -> Vec<u8> {
        let mut header = format!(
            r#"{{"weight":{{"dtype":"U32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            data.len() / 4,
            data.len()
        )
        .into_bytes();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        bytes
    }

    #[test]
    fn caches_weight_hashes_and_misses_after_metadata_changes() -> Result<(), Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-weight-cache-{nonce}"));
        let scratch = root.join("scratch");
        fs::create_dir(&root)?;
        let weights = root.join("model.safetensors");
        fs::write(&weights, safetensors(&[0; 4]))?;

        let (first, first_hit) = strong_weights_sha256(&weights, &scratch)?;
        let (second, second_hit) = strong_weights_sha256(&weights, &scratch)?;
        assert!(!first_hit);
        assert!(second_hit);
        assert_eq!(first, second);

        fs::write(&weights, safetensors(&[0; 8]))?;
        let (third, third_hit) = strong_weights_sha256(&weights, &scratch)?;
        assert!(!third_hit);
        assert_ne!(first, third);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn index_content_participates_in_the_weight_cache_key() -> Result<(), Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-index-cache-{nonce}"));
        let scratch = root.join("scratch");
        fs::create_dir(&root)?;
        fs::write(root.join("shard.safetensors"), safetensors(&[0; 4]))?;
        let index = root.join("model.safetensors.index.json");
        fs::write(&index, r#"{"weight_map":{"weight":"shard.safetensors"}}"#)?;

        assert!(!strong_weights_sha256(&index, &scratch)?.1);
        assert!(strong_weights_sha256(&index, &scratch)?.1);
        fs::write(
            &index,
            r#"{ "weight_map": { "weight": "shard.safetensors" } }"#,
        )?;
        assert!(!strong_weights_sha256(&index, &scratch)?.1);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn converts_profiles_to_deterministic_guest_load_config() {
        let config = profile_load_config(&profile());
        assert_eq!(
            config.tunings,
            ["residual-norm", "qk-norm-rope", "silu-mul", "final-norm"]
        );
        assert_eq!(
            config.fixed_variant_picks[0].site,
            "attention.prefill-small"
        );
        assert_eq!(config.variant_rule_picks[0].site, "attention.decode");
        assert_eq!(config.variant_rule_picks[0].arms[2].hi, 4095);
    }

    #[test]
    fn argmax_rejects_empty_values() {
        assert_eq!(
            argmax(&[]).unwrap_err().to_string(),
            "cannot take argmax of empty logits"
        );
    }

    #[test]
    fn builds_the_existing_host_limits_from_config() {
        let expected = forja_host::Limits::new(
            48 * 1024 * 1024 * 1024,
            4,
            1_000_000_000,
            20_000,
            1024 * 1024 * 1024,
        )
        .with_command_limits(4_096, u64::MAX)
        .with_guest_call_timeout(Duration::from_secs(300))
        .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));
        assert_eq!(limits(&ConfigLimits::default()).unwrap(), expected);
    }

    #[test]
    fn argmax_selects_the_last_tied_value() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0])?, 2);
        Ok(())
    }

    #[test]
    fn argmax_uses_total_order_for_nan() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0])?, 1);
        Ok(())
    }

    #[test]
    fn reads_one_selected_token() -> Result<(), Box<dyn Error>> {
        assert_eq!(read_token(&7_u32.to_le_bytes())?, 7);
        assert!(read_token(&[0; 8]).is_err());
        Ok(())
    }

    #[test]
    fn selects_single_or_indexed_model_weights() -> Result<(), Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-weights-path-{nonce}"));
        fs::create_dir(&root)?;
        let index = root.join("model.safetensors.index.json");
        fs::write(&index, b"index")?;
        assert_eq!(weights_path(&root)?, fs::canonicalize(index)?);
        let single = root.join("model.safetensors");
        fs::write(&single, b"single")?;
        assert_eq!(weights_path(&root)?, fs::canonicalize(single)?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    fn assert_outside_weight_symlink_is_rejected(name: &str) -> Result<(), Box<dyn Error>> {
        use std::os::unix::fs::symlink;

        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-weights-root-{name}-{nonce}"));
        let outside = std::env::temp_dir().join(format!("forja-weights-outside-{name}-{nonce}"));
        fs::create_dir(&root)?;
        fs::create_dir(&outside)?;
        let target = outside.join("weights");
        fs::write(&target, b"weights")?;
        symlink(target, root.join(name))?;
        assert!(weights_path(&root).unwrap_err().contains("outside"));
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_single_file_weights_symlinked_outside() -> Result<(), Box<dyn Error>> {
        assert_outside_weight_symlink_is_rejected("model.safetensors")
    }

    #[cfg(unix)]
    #[test]
    fn rejects_weight_indexes_symlinked_outside() -> Result<(), Box<dyn Error>> {
        assert_outside_weight_symlink_is_rejected("model.safetensors.index.json")
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn maps_resolved_graph_replay_to_metal() {
        assert_eq!(
            metal_graph_replay(GraphReplay::Tier1),
            forja_metal::MetalGraphReplay::Tier1
        );
        assert_eq!(
            metal_graph_replay(GraphReplay::Tier2),
            forja_metal::MetalGraphReplay::Tier2
        );
    }
}
