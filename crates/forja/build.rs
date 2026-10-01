//! Embeds the source revision without requiring runtime repository access.

use std::{env, process::Command};

fn main() {
    println!("cargo::rerun-if-env-changed=FORJA_BUILD_REV");
    track_git_path("HEAD");
    if let Some(reference) = git_output(&["symbolic-ref", "--quiet", "HEAD"]) {
        track_git_path(&reference);
    }
    println!("cargo::rustc-env=FORJA_BUILD_REV={}", revision());
}

fn track_git_path(path: &str) {
    if let Some(path) = git_output(&["rev-parse", "--git-path", path]) {
        println!("cargo::rerun-if-changed={path}");
    }
}

fn revision() -> String {
    env::var("FORJA_BUILD_REV")
        .ok()
        .filter(|revision| !revision.is_empty())
        .or_else(git_revision)
        .unwrap_or_else(|| "unknown".to_owned())
}

fn git_revision() -> Option<String> {
    git_output(&["describe", "--always", "--dirty"])
}

fn git_output(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
