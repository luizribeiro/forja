use std::{error::Error, fs, path::PathBuf};

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Component whose production-resolved variant picks are captured.
    #[arg(long)]
    engine: PathBuf,
    /// Destination for the build snapshot.
    #[arg(long)]
    json: PathBuf,
}

pub(crate) fn run(
    options: &Options,
    config: &forja_config::DevConfig,
) -> Result<(), Box<dyn Error>> {
    let component = crate::resolution::engine(&options.engine);
    let profile = crate::resolution::read_embedded_profile(&component)?;
    let selection = config.engine.resolve(&profile)?;
    let mut non_pick_profile = serde_json::to_value(&profile)?;
    non_pick_profile
        .as_object_mut()
        .ok_or("profile must be an object")?
        .remove("variants");
    #[cfg(target_os = "macos")]
    let catalog: serde_json::Value =
        serde_json::from_str(&forja_metal::MetalBackend::new()?.variant_catalog_json())?;
    #[cfg(not(target_os = "macos"))]
    let catalog = serde_json::Value::Null;
    let snapshot = serde_json::json!({
        "commit": crate::provenance::commit(), "dirty": crate::provenance::dirty(),
        "binary_sha256": golden_fixtures::sha256_file(std::env::current_exe()?)?,
        "build_environment": crate::provenance::BUILD_ENVIRONMENT,
        "component_sha256": golden_fixtures::sha256_file(&component)?,
        "profile_name": component.file_stem().ok_or("component has no name")?.to_string_lossy(),
        "profile_hash": profile.sha256()?, "profile_family": profile.family(),
        "non_pick_profile": non_pick_profile, "variants": selection.variants,
        "tunings": selection.tunings, "catalog": catalog,
        "host_config": {"backend": config.backend, "limits": config.limits,
                        "replay": config.engine.replay},
    });
    fs::write(&options.json, serde_json::to_vec_pretty(&snapshot)?)?;
    Ok(())
}
