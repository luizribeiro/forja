//! Builds the isolated WebAssembly guest workspace for host-side tests.

use std::env;
use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let crate_dir = PathBuf::from(required_var("CARGO_MANIFEST_DIR")?);
    let repository = crate_dir.join("../..");
    let guest_manifest = repository.join("support/guests/Cargo.toml");
    let out_dir = PathBuf::from(required_var("OUT_DIR")?);
    let main_target_dir = main_target_dir(&repository, &out_dir)?;
    let guest_target_dir = main_target_dir.join("guest-build");
    let bf16_target_dir = main_target_dir.join("guest-build-bf16");

    build_guest_workspace(&guest_manifest, &guest_target_dir, &[])?;
    build_guest_workspace(
        &guest_manifest,
        &bf16_target_dir,
        &["-p", "qwen3", "--features", "bf16"],
    )?;

    let release_dir = guest_target_dir.join("wasm32-wasip2/release");
    emit_guest_path("HELLO_COMPONENT", &release_dir.join("hello.wasm"));
    emit_guest_path(
        "ENGINE_SMOKE_COMPONENT",
        &release_dir.join("engine_smoke.wasm"),
    );
    emit_guest_path(
        "ENGINE_SDK_SMOKE_COMPONENT",
        &release_dir.join("engine_sdk_smoke.wasm"),
    );
    emit_guest_path(
        "RMSNORM_SMOKE_COMPONENT",
        &release_dir.join("rmsnorm_smoke.wasm"),
    );
    emit_guest_path(
        "PROGRAM_SMOKE_COMPONENT",
        &release_dir.join("program_smoke.wasm"),
    );
    emit_guest_path("SDK_SMOKE_COMPONENT", &release_dir.join("sdk_smoke.wasm"));
    emit_guest_path(
        "TENSOR_SMOKE_COMPONENT",
        &release_dir.join("tensor_smoke.wasm"),
    );
    emit_guest_path(
        "TENSOR_ABUSE_COMPONENT",
        &release_dir.join("tensor_abuse.wasm"),
    );
    emit_guest_path("TOY_MLP_COMPONENT", &release_dir.join("toy_mlp.wasm"));
    emit_guest_path("QWEN3_COMPONENT", &release_dir.join("qwen3.wasm"));
    emit_guest_path(
        "QWEN3_BF16_COMPONENT",
        &bf16_target_dir.join("wasm32-wasip2/release/qwen3.wasm"),
    );
    emit_guest_path(
        "WEIGHTS_SMOKE_COMPONENT",
        &release_dir.join("weights_smoke.wasm"),
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("support/guests").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("crates/forja-sdk").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("crates/forja-sdk-macros").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("wit").display()
    );
    Ok(())
}

fn main_target_dir(repository: &Path, out_dir: &Path) -> io::Result<PathBuf> {
    if let Some(configured) = env::var_os("CARGO_TARGET_DIR") {
        let configured = PathBuf::from(configured);
        return Ok(if configured.is_absolute() {
            configured
        } else {
            repository.join(configured)
        });
    }

    let build_dir = out_dir
        .ancestors()
        .find(|path| path.file_name() == Some(OsStr::new("build")))
        .ok_or_else(|| {
            io::Error::other(format!(
                "cannot find Cargo profile directory from OUT_DIR {}",
                out_dir.display()
            ))
        })?;
    let profile_dir = build_dir.parent().ok_or_else(|| {
        io::Error::other(format!(
            "Cargo build directory has no profile parent: {}",
            build_dir.display()
        ))
    })?;
    profile_dir.parent().map(Path::to_path_buf).ok_or_else(|| {
        io::Error::other(format!(
            "Cargo profile directory has no target parent: {}",
            profile_dir.display()
        ))
    })
}

fn build_guest_workspace(manifest: &Path, target_dir: &Path, args: &[&str]) -> io::Result<()> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
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

    let status = command.status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "guest build failed with {status}"
        )));
    }
    Ok(())
}

fn required_var(name: &str) -> io::Result<std::ffi::OsString> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("{name} is not set")))
}

fn emit_guest_path(name: &str, path: &Path) {
    println!("cargo::rustc-env={name}={}", path.display());
}
