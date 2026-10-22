//! Stable command-line catalog snapshots.

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
        "93ab5070eae069c3bc094fc0c0665201b853df84770ed3b1e68711ff66de6428"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "deaa0a9c8b006a8a2afa17411054aaf043ec75caf1616768816bdb9a7c8b0dc7"
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
