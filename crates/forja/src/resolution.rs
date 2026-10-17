use std::{
    ffi::OsStr,
    fs, io,
    path::{Component, Path, PathBuf},
};

use forja_chat_format::ToolFormat;
use forja_config::{Profile, embed_profile, read_profile};
use qwen3_coder_xml::Qwen3CoderXml;
use sha2::{Digest, Sha256};

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

pub(crate) fn build(
    profile: &Path,
    models: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
    let features = profile.build().features.join(",");
    let args = if features.is_empty() {
        Vec::new()
    } else {
        vec!["--features", features.as_str()]
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
    if let Some(chat) = profile.chat() {
        if let Some(format) = &chat.tool_format {
            build_chat_format(
                &profile,
                format,
                &chat.tool_format_options,
                models,
                &build_dir,
                &output,
            )?;
        } else if let Some(component) = &chat.component {
            build_custom_chat_format(component, &build_dir, &output)?;
        }
    }
    Ok(output)
}

fn build_chat_format(
    profile: &Profile,
    format: &str,
    format_options: &std::collections::BTreeMap<String, toml::Value>,
    models: Option<&Path>,
    build_dir: &Path,
    engine_output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = serde_json::to_string(format_options)?;
    validate_tool_format(format, &options)?;
    let model_name = profile
        .model()
        .id
        .rsplit('/')
        .next()
        .filter(|name| is_safe_component(Path::new(name)))
        .ok_or("profile model id has no safe final component")?;
    let model = models
        .ok_or("chat format build requires paths.models or FORJA_MODELS")?
        .join(model_name);
    let template = model.join("chat_template.jinja");
    let generation = model.join("generation_config.json");
    verify_chat_input(
        &template,
        profile
            .model()
            .chat_template_sha256
            .as_deref()
            .ok_or("profile has no chat template hash")?,
        "chat template",
    )?;
    verify_chat_input(
        &generation,
        profile
            .model()
            .generation_config_sha256
            .as_deref()
            .ok_or("profile has no generation config hash")?,
        "generation config",
    )?;
    let manifest = fs::canonicalize("crates/forja-chat-format-component/Cargo.toml")?;
    let target_dir = build_dir.join("chat-format");
    build_chat_component(
        &manifest,
        &target_dir,
        &["--no-default-features", "--features", format],
        &[
            ("FORJA_CHAT_TEMPLATE", template.as_os_str()),
            ("FORJA_GENERATION_CONFIG", generation.as_os_str()),
            ("FORJA_TOOL_FORMAT_OPTIONS", OsStr::new(&options)),
        ],
        "forja_chat_format_component",
        engine_output,
    )
}

fn verify_chat_input(
    path: &Path,
    expected: &str,
    description: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(path).map_err(|error| {
        format!(
            "{description} is missing or unreadable at {}: {error}",
            path.display()
        )
    })?;
    let actual = format!("sha256:{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(format!(
            "{description} hash mismatch: profile requires {expected}, found {actual}"
        )
        .into());
    }
    Ok(())
}

fn build_chat_component(
    manifest: &Path,
    target_dir: &Path,
    args: &[&str],
    environment: &[(&str, &OsStr)],
    crate_name: &str,
    engine_output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    forja_build::build_configured_core_module(manifest, target_dir, args, environment)?;
    let module = fs::read(
        target_dir
            .join("wasm32-unknown-unknown/release")
            .join(crate_name)
            .with_extension("wasm"),
    )?;
    let component = forja_build::encode_component(&module)?;
    fs::write(engine_output.with_extension("chat-format.wasm"), component)?;
    Ok(())
}

fn build_custom_chat_format(
    name: &str,
    build_dir: &Path,
    engine_output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = fs::canonicalize(Path::new("chat-formats").join(name).join("Cargo.toml"))?;
    let target_dir = build_dir.join("chat-format");
    build_chat_component(
        &manifest,
        &target_dir,
        &[],
        &[],
        &name.replace('-', "_"),
        engine_output,
    )
}

fn validate_tool_format(format: &str, options: &str) -> Result<(), String> {
    match format {
        "qwen3-coder-xml" => Qwen3CoderXml::options(options).map(|_| ()),
        format => Err(format!("unsupported tool format {format}")),
    }
}

pub(crate) fn read_embedded_profile(path: &Path) -> Result<Profile, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read engine {}: {error}", path.display()))?;
    read_profile(&bytes).map_err(|error| format!("engine {}: {error}", path.display()))
}

pub(crate) fn source_profile(hash: &str) -> Result<PathBuf, String> {
    let families =
        fs::read_dir("engines").map_err(|error| format!("cannot read engine families: {error}"))?;
    let mut matched = Vec::new();
    for family in families {
        let profiles = family
            .map_err(|error| format!("cannot read engine family: {error}"))?
            .path()
            .join("profiles");
        let Ok(entries) = fs::read_dir(profiles) else {
            continue;
        };
        for entry in entries {
            let path = entry
                .map_err(|error| format!("cannot read engine profile: {error}"))?
                .path();
            if path.extension().and_then(|value| value.to_str()) != Some("toml") {
                continue;
            }
            let source = fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let profile: Profile = toml::from_str(&source)
                .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
            if profile.sha256().map_err(|error| error.to_string())? == hash {
                matched.push(path);
            }
        }
    }
    match matched.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err("engine has no matching source profile".to_owned()),
        _ => Err("engine has multiple matching source profiles".to_owned()),
    }
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

    #[test]
    fn chat_profiles_name_their_tool_format() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = resolve_profile_from(
            &root,
            Path::new("qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4"),
        )
        .unwrap();
        let profile: Profile = toml::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            profile.chat().unwrap().tool_format.as_deref(),
            Some("qwen3-coder-xml")
        );
    }

    #[test]
    fn validates_options_with_the_selected_tool_format() {
        assert!(validate_tool_format("qwen3-coder-xml", "{}").is_ok());
        assert!(validate_tool_format("qwen3-coder-xml", r#"{"unknown":true}"#).is_err());
        assert!(validate_tool_format("missing", "{}").is_err());
    }

    #[test]
    fn rejects_missing_and_mismatched_chat_inputs() {
        let root = forja_testing::temporary_directory("chat-input-hashes").unwrap();
        let path = root.join("chat_template.jinja");
        let expected = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

        let missing = verify_chat_input(&path, expected, "chat template").unwrap_err();
        assert!(missing.to_string().contains("chat template is missing"));

        fs::write(&path, "changed").unwrap();
        let mismatch = verify_chat_input(&path, expected, "chat template").unwrap_err();
        assert!(mismatch.to_string().contains("chat template hash mismatch"));
        fs::remove_dir_all(root).unwrap();
    }
}
