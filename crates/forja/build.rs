//! Embeds the source revision without requiring runtime repository access.

use std::{env, process::Command};

fn main() {
    println!("cargo::rerun-if-env-changed=FORJA_BUILD_REV");
    println!("cargo::rerun-if-changed=../../.git/HEAD");
    println!("cargo::rustc-env=FORJA_BUILD_REV={}", revision());
}

fn revision() -> String {
    env::var("FORJA_BUILD_REV")
        .ok()
        .filter(|revision| !revision.is_empty())
        .or_else(git_revision)
        .unwrap_or_else(|| "unknown".to_owned())
}

fn git_revision() -> Option<String> {
    let output = Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
