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
        OsStr::new("wasm32-wasip2"),
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
        if let Ok(package) = env::var("CARGO_PKG_NAME")
            && matches!(package.as_str(), "qwen3" | "olmoe" | "qwen3-coder")
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
    identity = identity.replace(
        "/workspace/engines/qwen3",
        "/workspace/support/guests/qwen3",
    );
    identity = identity.replace(
        "/workspace/engines/olmoe",
        "/workspace/support/guests/olmoe",
    );
    identity = identity.replace(
        "/workspace/engines/qwen3-coder",
        "/workspace/support/guests/qwen3-coder",
    );
    if let Ok(package) = env::var("CARGO_PKG_NAME")
        && matches!(package.as_str(), "qwen3" | "olmoe" | "qwen3-coder")
    {
        identity = identity.replace("\0src/lib.rs\0", &format!("\0{package}/src/lib.rs\0"));
    }
    let hash = identity
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    format!("metadata={hash:016x}")
}

fn required_var(name: &str) -> io::Result<OsString> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("{name} is not set")))
}
