//! Builds the isolated WebAssembly guest workspace for host-side tests.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

#[allow(clippy::too_many_lines)]
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

    let [
        qwen3,
        qwen3_bf16,
        qwen3_residual_norm,
        qwen3_bf16_residual_norm,
        qwen3_qk_norm_rope,
        qwen3_silu_mul,
        qwen3_final_norm,
        qwen3_all_fusions,
        qwen3_bf16_all_fusions,
        qwen3_no_replay,
        qwen3_bf16_no_replay,
    ] = build_qwen_profiles(
        &guest_manifest,
        &guest_target_dir,
        &bf16_target_dir,
        &out_dir,
    )?;
    let release_dir = guest_target_dir.join("wasm32-wasip2/release");
    let [olmoe, olmoe_no_replay] =
        build_olmoe_profiles(&guest_manifest, &guest_target_dir, &out_dir)?;
    let [qwen3_coder, qwen3_coder_no_replay] =
        build_qwen3_coder_profiles(&guest_manifest, &guest_target_dir, &out_dir)?;
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
    emit_guest_path("QWEN3_COMPONENT", &qwen3);
    emit_guest_path("OLMOE_COMPONENT", &olmoe);
    emit_guest_path("OLMOE_NO_REPLAY_COMPONENT", &olmoe_no_replay);
    emit_qwen3_coder_paths(&qwen3_coder, &qwen3_coder_no_replay);
    emit_guest_path("QWEN3_BF16_COMPONENT", &qwen3_bf16);
    emit_guest_path("QWEN3_RESIDUAL_NORM_COMPONENT", &qwen3_residual_norm);
    emit_guest_path(
        "QWEN3_BF16_RESIDUAL_NORM_COMPONENT",
        &qwen3_bf16_residual_norm,
    );
    emit_guest_path("QWEN3_QK_NORM_ROPE_COMPONENT", &qwen3_qk_norm_rope);
    emit_guest_path("QWEN3_SILU_MUL_COMPONENT", &qwen3_silu_mul);
    emit_guest_path("QWEN3_FINAL_NORM_COMPONENT", &qwen3_final_norm);
    emit_guest_path("QWEN3_ALL_FUSIONS_COMPONENT", &qwen3_all_fusions);
    emit_guest_path("QWEN3_BF16_ALL_FUSIONS_COMPONENT", &qwen3_bf16_all_fusions);
    emit_guest_path("QWEN3_NO_REPLAY_COMPONENT", &qwen3_no_replay);
    emit_guest_path("QWEN3_BF16_NO_REPLAY_COMPONENT", &qwen3_bf16_no_replay);
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

fn emit_qwen3_coder_paths(default: &Path, no_replay: &Path) {
    emit_guest_path("QWEN3_CODER_COMPONENT", default);
    emit_guest_path("QWEN3_CODER_NO_REPLAY_COMPONENT", no_replay);
}

fn build_qwen3_coder_profiles(
    manifest: &Path,
    target_dir: &Path,
    out_dir: &Path,
) -> io::Result<[PathBuf; 2]> {
    let component = target_dir.join("wasm32-wasip2/release/qwen3_coder.wasm");
    let default = copy_component(&component, out_dir, "qwen3-coder.wasm")?;
    build_guest_workspace(
        manifest,
        target_dir,
        &["-p", "qwen3-coder", "--features", "no-replay"],
    )?;
    let no_replay = copy_component(&component, out_dir, "qwen3-coder-no-replay.wasm")?;
    Ok([default, no_replay])
}

fn build_olmoe_profiles(
    manifest: &Path,
    target_dir: &Path,
    out_dir: &Path,
) -> io::Result<[PathBuf; 2]> {
    let component = target_dir.join("wasm32-wasip2/release/olmoe.wasm");
    let default = copy_component(&component, out_dir, "olmoe.wasm")?;
    build_guest_workspace(
        manifest,
        target_dir,
        &["-p", "olmoe", "--features", "no-replay"],
    )?;
    let no_replay = copy_component(&component, out_dir, "olmoe-no-replay.wasm")?;
    Ok([default, no_replay])
}

fn build_qwen_profiles(
    manifest: &Path,
    target_dir: &Path,
    bf16_target_dir: &Path,
    out_dir: &Path,
) -> io::Result<[PathBuf; 11]> {
    let release_dir = target_dir.join("wasm32-wasip2/release");
    let bf16_release_dir = bf16_target_dir.join("wasm32-wasip2/release");
    let qwen3 = copy_component(&release_dir.join("qwen3.wasm"), out_dir, "qwen3.wasm")?;
    let qwen3_bf16 = copy_component(
        &bf16_release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-bf16.wasm",
    )?;
    let variants = [
        ("residual-norm-only", "qwen3-residual-norm.wasm"),
        ("qk-norm-rope-only", "qwen3-qk-norm-rope.wasm"),
        ("silu-mul-only", "qwen3-silu-mul.wasm"),
        ("final-norm-only", "qwen3-final-norm.wasm"),
        ("all-fusions", "qwen3-all-fusions.wasm"),
    ]
    .map(|(feature, name)| {
        build_qwen_profile(manifest, target_dir, feature)?;
        copy_component(&release_dir.join("qwen3.wasm"), out_dir, name)
    })
    .into_iter()
    .collect::<io::Result<Vec<_>>>()?;
    let [
        qwen3_residual_norm,
        qwen3_qk_norm_rope,
        qwen3_silu_mul,
        qwen3_final_norm,
        qwen3_all_fusions,
    ] = variants
        .try_into()
        .map_err(|_| io::Error::other("fusion profile count changed"))?;
    build_qwen_profile(manifest, bf16_target_dir, "bf16,residual-norm-only")?;
    let qwen3_bf16_residual_norm = copy_component(
        &bf16_release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-bf16-residual-norm.wasm",
    )?;
    build_qwen_profile(manifest, bf16_target_dir, "bf16,all-fusions")?;
    let qwen3_bf16_all_fusions = copy_component(
        &bf16_release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-bf16-all-fusions.wasm",
    )?;
    build_qwen_profile(manifest, target_dir, "all-fusions,no-replay")?;
    let qwen3_no_replay = copy_component(
        &release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-no-replay.wasm",
    )?;
    build_qwen_profile(manifest, bf16_target_dir, "bf16,all-fusions,no-replay")?;
    let qwen3_bf16_no_replay = copy_component(
        &bf16_release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-bf16-no-replay.wasm",
    )?;
    fs::copy(&qwen3, release_dir.join("qwen3.wasm"))?;
    fs::copy(&qwen3_bf16, bf16_release_dir.join("qwen3.wasm"))?;
    Ok([
        qwen3,
        qwen3_bf16,
        qwen3_residual_norm,
        qwen3_bf16_residual_norm,
        qwen3_qk_norm_rope,
        qwen3_silu_mul,
        qwen3_final_norm,
        qwen3_all_fusions,
        qwen3_bf16_all_fusions,
        qwen3_no_replay,
        qwen3_bf16_no_replay,
    ])
}

fn build_qwen_profile(manifest: &Path, target_dir: &Path, features: &str) -> io::Result<()> {
    build_guest_workspace(
        manifest,
        target_dir,
        &[
            "-p",
            "qwen3",
            "--no-default-features",
            "--features",
            features,
        ],
    )
}

fn copy_component(source: &Path, out_dir: &Path, name: &str) -> io::Result<PathBuf> {
    let destination = out_dir.join(name);
    fs::copy(source, &destination)?;
    Ok(destination)
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
