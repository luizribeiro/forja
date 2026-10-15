//! Reproducible compilation support for WebAssembly components.

use std::{
    env,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

/// Builds one component workspace with reproducible compiler inputs.
///
/// # Errors
///
/// Returns an error when required environment state is unavailable, Cargo cannot be launched, or
/// the build fails.
pub fn build_component(manifest: &Path, target_dir: &Path, args: &[&str]) -> io::Result<()> {
    build_target(manifest, target_dir, "wasm32-wasip2", args)
}

/// Builds one zero-import core WebAssembly module with reproducible compiler inputs.
///
/// # Errors
///
/// Returns an error when required environment state is unavailable, Cargo cannot be launched, or
/// the build fails.
pub fn build_core_module(manifest: &Path, target_dir: &Path, args: &[&str]) -> io::Result<()> {
    build_target(manifest, target_dir, "wasm32-unknown-unknown", args)
}

fn build_target(manifest: &Path, target_dir: &Path, target: &str, args: &[&str]) -> io::Result<()> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let repository = manifest
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .ok_or_else(|| io::Error::other("component manifest has no repository root"))?;
    let repository = fs::canonicalize(repository)?;
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .ok_or_else(|| io::Error::other("CARGO_HOME and HOME are not set"))?;
    let mut command = Command::new(cargo);
    command.args([
        OsStr::new("build"),
        OsStr::new("--release"),
        OsStr::new("--target"),
        OsStr::new(target),
        OsStr::new("--manifest-path"),
        manifest.as_os_str(),
        OsStr::new("--target-dir"),
        target_dir.as_os_str(),
        OsStr::new("--locked"),
    ]);
    command.args(args);
    for (key, _) in env::vars_os() {
        if key.to_string_lossy().starts_with("CARGO_") || key == "RUSTFLAGS" {
            command.env_remove(key);
        }
    }
    command.env("CARGO_HOME", &cargo_home);
    command.env("RUSTFLAGS", "--cfg=forja_reproducible_guest_v1");
    command.env("RUSTC_WRAPPER", env::current_exe()?);
    command.env("FORJA_GUEST_RUSTC_WRAPPER", "1");
    command.env("FORJA_GUEST_REPOSITORY", repository);
    command.env("FORJA_GUEST_CARGO_HOME", cargo_home);
    command.env("FORJA_GUEST_TARGET_DIR", target_dir);

    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "component build failed with {status}"
        )))
    }
}

/// Encodes a wit-bindgen core module as a zero-import component.
///
/// # Errors
///
/// Returns an error when the module or its embedded component metadata is invalid.
pub fn encode_component(module: &[u8]) -> io::Result<Vec<u8>> {
    wit_component::ComponentEncoder::default()
        .module(module)
        .map_err(io::Error::other)?
        .validate(true)
        .encode()
        .map_err(io::Error::other)
}

/// Runs the current process as the reproducible rustc wrapper when requested.
///
/// Returns `false` when the process should continue with its normal entry point.
///
/// # Errors
///
/// Returns an error when wrapper state is incomplete, rustc cannot be launched, or compilation
/// fails.
pub fn run_rustc_wrapper() -> io::Result<bool> {
    if env::var_os("FORJA_GUEST_RUSTC_WRAPPER").is_none() {
        return Ok(false);
    }
    let mut args = env::args_os().skip(1);
    let rustc = args
        .next()
        .ok_or_else(|| io::Error::other("rustc wrapper received no compiler"))?;
    let mut args = args.collect::<Vec<_>>();
    if env::var_os("CARGO_PKG_NAME").is_some() {
        let repository = required_var("FORJA_GUEST_REPOSITORY")?;
        let cargo_home = required_var("FORJA_GUEST_CARGO_HOME")?;
        let target_dir = required_var("FORJA_GUEST_TARGET_DIR")?;
        let metadata = stable_metadata(&args, [&repository, &cargo_home, &target_dir]);
        let mut codegen = false;
        for arg in &mut args {
            if codegen && arg.to_string_lossy().starts_with("metadata=") {
                *arg = metadata.clone().into();
            }
            codegen = arg == "-C";
        }
        for (from, to) in [
            (&repository, "/workspace"),
            (&cargo_home, "/cargo"),
            (&target_dir, "/target"),
        ] {
            args.push(format!("--remap-path-prefix={}={to}", Path::new(from).display()).into());
        }
        if let (Ok(package), Some(manifest_dir)) = (
            env::var("CARGO_PKG_NAME"),
            env::var_os("CARGO_MANIFEST_DIR"),
        ) && is_engine_package(Path::new(&repository), Path::new(&manifest_dir), &package)
        {
            args.push(format!("--remap-path-prefix=src={package}/src").into());
        }
    }
    let status = Command::new(rustc).args(args).status()?;
    if status.success() {
        Ok(true)
    } else {
        Err(io::Error::other(format!("rustc failed with {status}")))
    }
}

fn stable_metadata(args: &[OsString], roots: [&OsString; 3]) -> String {
    let mut identity = format!(
        "{}:{}:{}",
        env::var("CARGO_PKG_NAME").unwrap_or_default(),
        env::var("CARGO_PKG_VERSION").unwrap_or_default(),
        env::var("CARGO_MANIFEST_PATH").unwrap_or_default()
    );
    let mut previous_codegen = false;
    let mut previous_extern = false;
    for arg in args {
        let value = arg.to_string_lossy();
        if !((previous_codegen
            && (value.starts_with("metadata=") || value.starts_with("extra-filename=")))
            || previous_extern)
        {
            identity.push('\0');
            identity.push_str(&value);
        } else if previous_extern {
            identity.push_str(value.split('=').next().unwrap_or_default());
        }
        previous_codegen = value == "-C";
        previous_extern = value == "--extern";
    }
    for (root, replacement) in roots.into_iter().zip(["/workspace", "/cargo", "/target"]) {
        identity = identity.replace(&*root.to_string_lossy(), replacement);
    }
    if let Ok(package) = env::var("CARGO_PKG_NAME") {
        identity = normalize_engine_identity(identity, &package);
    }
    let hash = identity
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("metadata={hash:016x}")
}

fn is_engine_package(repository: &Path, manifest_dir: &Path, package: &str) -> bool {
    manifest_dir
        .strip_prefix(repository)
        .ok()
        .and_then(|path| path.to_str())
        == Some(&format!("engines/{package}"))
}

fn normalize_engine_identity(mut identity: String, package: &str) -> String {
    let engine_root = format!("/workspace/engines/{package}");
    if identity.contains(&engine_root) {
        identity = identity.replace(
            &engine_root,
            &format!("/workspace/support/guests/{package}"),
        );
        identity = identity.replace("\0src/lib.rs\0", &format!("\0{package}/src/lib.rs\0"));
    }
    identity
}

fn required_var(name: &str) -> io::Result<OsString> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("{name} is not set")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_any_engine_package() {
        let repository = Path::new("/repo");
        assert!(is_engine_package(
            repository,
            Path::new("/repo/engines/synthetic"),
            "synthetic"
        ));
        assert_eq!(
            normalize_engine_identity(
                "synthetic:0:/workspace/engines/synthetic/Cargo.toml\0src/lib.rs\0".to_owned(),
                "synthetic"
            ),
            "synthetic:0:/workspace/support/guests/synthetic/Cargo.toml\0synthetic/src/lib.rs\0"
        );
    }
}
