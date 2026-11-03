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
        "cf34c5e17b48a02e84baf7fd3fe93905a723f89c05239cd1713ff05f1e8c58b3"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "cb9792b7260f088cf4711fd43dbd8ec987ef5fd69e2e3e78d7eab18f3ddc3d6e"
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

#[test]
fn metal_qwen38_decode_profiles_share_the_one_row_reduction_class() {
    let catalog: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    let variants = catalog["variants"].as_array().unwrap();
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../engines/qwen38/profiles");
    for suffix in ["q4", "q4-f32-kv", "q4-long"] {
        let path = root.join(format!(
            "qwen38-27b-4bit.metal-apple-m3-ultra.{suffix}.toml"
        ));
        let profile: forja_config::Profile =
            toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let fixed = &profile.variants().fixed;
        assert_eq!(fixed["quant"], "quant-matmul.tiled");
        for site in ["quant-decode", "quant-decode-batch"] {
            let variant = variants
                .iter()
                .find(|variant| variant["name"].as_str() == Some(fixed[site].as_str()))
                .unwrap();
            assert_eq!(variant["reduction_class"], "quant-matmul.row-gemv");
            assert_eq!(variant["batch_invariant"], true);
        }
    }
}
