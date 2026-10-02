//! Builds the isolated WebAssembly guest workspace for host-side tests.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if forja_build::run_rustc_wrapper()? {
        return Ok(());
    }
    let crate_dir = PathBuf::from(required_var("CARGO_MANIFEST_DIR")?);
    let repository = fs::canonicalize(crate_dir.join("../.."))?;
    let guest_manifest = repository.join("support/guests/Cargo.toml");
    let olmoe_manifest = repository.join("engines/olmoe/Cargo.toml");
    let qwen3_manifest = repository.join("engines/qwen3/Cargo.toml");
    let qwen3_coder_manifest = repository.join("engines/qwen3-coder/Cargo.toml");
    let out_dir = PathBuf::from(required_var("OUT_DIR")?);
    let main_target_dir = main_target_dir(&repository, &out_dir)?;
    let guest_target_dir = main_target_dir.join("guest-build");
    let bf16_target_dir = main_target_dir.join("guest-build-bf16");

    forja_build::build_component(
        &guest_manifest,
        &guest_target_dir,
        &["--workspace", "--exclude", "metal-variant-smoke"],
    )?;
    forja_build::build_component(
        &guest_manifest,
        &guest_target_dir,
        &["-p", "metal-variant-smoke"],
    )?;
    forja_build::build_component(&qwen3_manifest, &guest_target_dir, &[])?;
    forja_build::build_component(&qwen3_manifest, &bf16_target_dir, &["--features", "bf16"])?;

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
        &qwen3_manifest,
        &guest_target_dir,
        &bf16_target_dir,
        &out_dir,
    )?;
    let release_dir = guest_target_dir.join("wasm32-wasip2/release");
    let [olmoe, olmoe_no_replay] =
        build_olmoe_profiles(&olmoe_manifest, &guest_target_dir, &out_dir)?;
    let [qwen3_coder, qwen3_coder_no_replay] =
        build_qwen3_coder_profiles(&qwen3_coder_manifest, &guest_target_dir, &out_dir)?;
    emit_guest_path("HELLO_COMPONENT", &release_dir.join("hello.wasm"));
    emit_guest_path(
        "METAL_VARIANT_SMOKE_COMPONENT",
        &release_dir.join("metal_variant_smoke.wasm"),
    );
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
        repository.join("engines/olmoe").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("engines/qwen3").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        repository.join("engines/qwen3-coder").display()
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
    forja_build::build_component(manifest, target_dir, &[])?;
    let profile = manifest
        .parent()
        .ok_or_else(|| io::Error::other("Qwen3-Coder manifest has no parent"))?
        .join("profiles/qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4.toml");
    let default = copy_profiled_component(&component, &profile, out_dir, "qwen3-coder.wasm")?;
    forja_build::build_component(manifest, target_dir, &["--features", "no-replay"])?;
    let no_replay = copy_component(&component, out_dir, "qwen3-coder-no-replay.wasm")?;
    Ok([default, no_replay])
}

fn build_olmoe_profiles(
    manifest: &Path,
    target_dir: &Path,
    out_dir: &Path,
) -> io::Result<[PathBuf; 2]> {
    let component = target_dir.join("wasm32-wasip2/release/olmoe.wasm");
    forja_build::build_component(manifest, target_dir, &[])?;
    let profile = manifest
        .parent()
        .ok_or_else(|| io::Error::other("OLMoE manifest has no parent"))?
        .join("profiles/olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16.toml");
    let default = copy_profiled_component(&component, &profile, out_dir, "olmoe.wasm")?;
    forja_build::build_component(manifest, target_dir, &["--features", "no-replay"])?;
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
    let profiles = manifest
        .parent()
        .ok_or_else(|| io::Error::other("Qwen3 manifest has no parent"))?
        .join("profiles");
    let qwen3 = copy_profiled_component(
        &release_dir.join("qwen3.wasm"),
        &profiles.join("qwen3-0.6b.metal-apple-m3-ultra.f32.toml"),
        out_dir,
        "qwen3.wasm",
    )?;
    let qwen3_bf16 = copy_profiled_component(
        &bf16_release_dir.join("qwen3.wasm"),
        &profiles.join("qwen3-0.6b.metal-apple-m3-ultra.bf16.toml"),
        out_dir,
        "qwen3-bf16.wasm",
    )?;
    let variants = [
        "qwen3-residual-norm.wasm",
        "qwen3-qk-norm-rope.wasm",
        "qwen3-silu-mul.wasm",
        "qwen3-final-norm.wasm",
        "qwen3-all-fusions.wasm",
    ]
    .map(|name| copy_component(&qwen3, out_dir, name))
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
    let qwen3_bf16_residual_norm =
        copy_component(&qwen3_bf16, out_dir, "qwen3-bf16-residual-norm.wasm")?;
    let qwen3_bf16_all_fusions =
        copy_component(&qwen3_bf16, out_dir, "qwen3-bf16-all-fusions.wasm")?;
    build_qwen_profile(manifest, target_dir, "no-replay")?;
    let qwen3_no_replay = copy_component(
        &release_dir.join("qwen3.wasm"),
        out_dir,
        "qwen3-no-replay.wasm",
    )?;
    build_qwen_profile(manifest, bf16_target_dir, "bf16,no-replay")?;
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
    forja_build::build_component(
        manifest,
        target_dir,
        &["--no-default-features", "--features", features],
    )
}

fn copy_component(source: &Path, out_dir: &Path, name: &str) -> io::Result<PathBuf> {
    let destination = out_dir.join(name);
    fs::copy(source, &destination)?;
    Ok(destination)
}

fn copy_profiled_component(
    source: &Path,
    profile: &Path,
    out_dir: &Path,
    name: &str,
) -> io::Result<PathBuf> {
    let profile = toml::from_str::<forja_config::Profile>(&fs::read_to_string(profile)?)
        .map_err(io::Error::other)?;
    let component =
        forja_config::embed_profile(&fs::read(source)?, &profile).map_err(io::Error::other)?;
    let destination = out_dir.join(name);
    fs::write(&destination, component)?;
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

fn required_var(name: &str) -> io::Result<std::ffi::OsString> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("{name} is not set")))
}

fn emit_guest_path(name: &str, path: &Path) {
    println!("cargo::rustc-env={name}={}", path.display());
}
