use std::{
    ffi::OsStr,
    fs, io,
    path::{Component, Path, PathBuf},
};

use forja_config::{ActivationFormat, Profile, embed_profile, read_profile};

pub(crate) fn engine(path: &Path) -> PathBuf {
    if path.extension() == Some(OsStr::new("wasm")) || path.components().count() != 1 {
        path.to_owned()
    } else {
        Path::new("target/engines").join(with_wasm_suffix(path.as_os_str()))
    }
}

pub(crate) fn model(
    requested: Option<&Path>,
    models: Option<&Path>,
    engines: &[PathBuf],
) -> Result<PathBuf, String> {
    if let Some(requested) = requested {
        if requested.try_exists().map_err(|error| error.to_string())? {
            return Ok(requested.to_owned());
        }
        if !is_safe_component(requested) {
            return Ok(requested.to_owned());
        }
        return models
            .map(|root| root.join(requested))
            .ok_or_else(|| "model names require paths.models or FORJA_MODELS".to_owned());
    }
    let first = engines
        .first()
        .ok_or_else(|| "at least one engine is required".to_owned())?;
    let profile = read_embedded_profile(first)?;
    let name = profile
        .model()
        .id
        .rsplit('/')
        .next()
        .filter(|name| is_safe_component(Path::new(name)))
        .ok_or_else(|| "profile model id has no final component".to_owned())?;
    for engine in &engines[1..] {
        if read_embedded_profile(engine)?.model().id != profile.model().id {
            return Err("engines in one command must use the same profile model".to_owned());
        }
    }
    models
        .map(|root| root.join(name))
        .ok_or_else(|| "profile model resolution requires paths.models or FORJA_MODELS".to_owned())
}

pub(crate) fn build(profile: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let profile_path = resolve_profile(profile)?;
    let source = fs::read_to_string(&profile_path)?;
    let profile: Profile = toml::from_str(&source)?;
    let name = profile_path
        .file_stem()
        .and_then(OsStr::to_str)
        .ok_or("profile name is not valid UTF-8")?;
    let manifest = fs::canonicalize(
        Path::new("engines")
            .join(profile.family())
            .join("Cargo.toml"),
    )?;
    if !manifest.is_file() {
        return Err(format!("engine family manifest is missing: {}", manifest.display()).into());
    }
    let build_dir = Path::new("target/engine-build").join(name);
    let args = match (profile.family(), profile.numerics().activations) {
        ("qwen3", ActivationFormat::Bf16) => vec!["--features", "bf16"],
        _ => Vec::new(),
    };
    forja_build::build_component(&manifest, &build_dir, &args)?;
    let crate_name = profile.family().replace('-', "_");
    let component = fs::read(
        build_dir
            .join("wasm32-wasip2/release")
            .join(crate_name)
            .with_extension("wasm"),
    )?;
    let component = embed_profile(&component, &profile)?;
    let output = Path::new("target/engines").join(with_wasm_suffix(OsStr::new(name)));
    fs::create_dir_all(output.parent().ok_or("engine output has no parent")?)?;
    fs::write(&output, component)?;
    Ok(output)
}

pub(crate) fn read_embedded_profile(path: &Path) -> Result<Profile, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read engine {}: {error}", path.display()))?;
    read_profile(&bytes).map_err(|error| format!("engine {}: {error}", path.display()))
}

fn is_safe_component(path: &Path) -> bool {
    matches!(
        path.components().collect::<Vec<_>>().as_slice(),
        [Component::Normal(_)]
    )
}

fn with_wasm_suffix(name: &OsStr) -> std::ffi::OsString {
    let mut output = name.to_owned();
    output.push(".wasm");
    output
}

fn resolve_profile(profile: &Path) -> io::Result<PathBuf> {
    resolve_profile_from(Path::new("."), profile)
}

fn resolve_profile_from(root: &Path, profile: &Path) -> io::Result<PathBuf> {
    if profile.try_exists()? {
        return Ok(profile.to_owned());
    }
    if !is_safe_component(profile) {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("profile does not exist: {}", profile.display()),
        ));
    }
    let mut file_name = profile.as_os_str().to_owned();
    file_name.push(".toml");
    let mut matches = Vec::new();
    for family in fs::read_dir(root.join("engines"))? {
        let candidate = family?.path().join("profiles").join(&file_name);
        if candidate.is_file() {
            matches.push(candidate);
        }
    }
    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("unknown engine profile {}", profile.display()),
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("engine profile name is not unique: {}", profile.display()),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_only_bare_engine_names() {
        assert_eq!(
            engine(Path::new("qwen")),
            Path::new("target/engines/qwen.wasm")
        );
        assert_eq!(
            engine(Path::new("qwen.bf16")),
            Path::new("target/engines/qwen.bf16.wasm")
        );
        assert_eq!(engine(Path::new("qwen.wasm")), Path::new("qwen.wasm"));
        assert_eq!(engine(Path::new("out/qwen")), Path::new("out/qwen"));
    }

    #[test]
    fn resolves_profile_names_without_recursive_discovery() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert!(
            resolve_profile_from(&root, Path::new("qwen3-0.6b.metal-apple-m3-ultra.bf16")).is_ok()
        );
        assert!(resolve_profile_from(&root, Path::new("profiles/missing")).is_err());
    }

    #[test]
    fn accepts_only_one_normal_path_component_as_a_name() {
        assert!(is_safe_component(Path::new("model")));
        for path in ["..", "a/..", "/abs", "a/b"] {
            assert!(!is_safe_component(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn repository_profiles_are_valid() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in [
            "qwen3-0.6b.metal-apple-m3-ultra.f32",
            "qwen3-0.6b.metal-apple-m3-ultra.bf16",
            "olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16",
            "qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4",
        ] {
            let path = resolve_profile_from(&root, Path::new(name)).unwrap();
            let source = fs::read_to_string(path).unwrap();
            toml::from_str::<Profile>(&source).unwrap();
        }
    }
}
