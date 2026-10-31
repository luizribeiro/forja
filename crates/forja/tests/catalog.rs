//! Stable command-line catalog snapshots.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::{fmt::Write as _, process::Command};

use sha2::{Digest, Sha256};

#[path = "../../../engines/qwen3-coder/src/reduction_classes.rs"]
mod qwen3_coder_reduction_classes;

fn catalog(arguments: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_forja"))
        .args(["--isolated", "catalog"])
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(output.stderr.len(), 0);
    String::from_utf8(output.stdout).unwrap()
}

fn digest(value: &str) -> String {
    Sha256::digest(value)
        .iter()
        .fold(String::new(), |mut output, byte| {
            write!(output, "{byte:02x}").unwrap();
            output
        })
}

#[test]
fn metal_catalog_cli_snapshots() {
    let markdown = catalog(&[])
        .lines()
        .map(|line| {
            if line.starts_with("**Device:** ") {
                "**Device:** Test GPU"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    assert_eq!(
        digest(&markdown),
        "e20163b81e1361f792d8f958c4487e5ad794c36d14c620be4ba0cfd622a11518"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "3bf664d3dd47939ce4117c4828af2afdac8e9b393695a42ea289aa390513db7e"
    );
}

#[test]
fn metal_qwen3_coder_reduction_classes_match_the_registry() {
    let catalog: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    let variants = catalog["variants"].as_array().unwrap();
    let mut matched = 0;
    for variant in variants {
        let name = variant["name"].as_str().unwrap();
        if let Some(expected) = qwen3_coder_reduction_classes::serving_reduction_class(name) {
            assert_eq!(
                variant["reduction_class"].as_str(),
                Some(expected),
                "{name}"
            );
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        qwen3_coder_reduction_classes::SERVING_REDUCTION_CLASSES.len()
    );
}
