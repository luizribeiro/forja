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
        "c116ec2dc933fcde38a5a1daf6a5a401a9aac975798677dd05a44fa01e24b628"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "674e4df8248a58626b6b12a8c859d8fa4b9be84079ef88e6321c8af8ac015492"
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
    for (suffix, decode_class) in [
        ("q4", "quant-matmul.vector8"),
        ("q4-f32-kv", "quant-matmul.row-gemv"),
        ("q4-long", "quant-matmul.vector8"),
    ] {
        let path = root.join(format!(
            "qwen38-27b-4bit.metal-apple-m3-ultra.{suffix}.toml"
        ));
        let profile: forja_config::Profile =
            toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let fixed = &profile.variants().fixed;
        let prefill_variant = if suffix == "q4-f32-kv" {
            "quant-matmul.tiled"
        } else {
            "quant-matmul.tiled-k64"
        };
        assert_eq!(fixed["quant"], prefill_variant);
        for site in ["quant-decode", "quant-decode-batch"] {
            let variant = variants
                .iter()
                .find(|variant| variant["name"].as_str() == Some(fixed[site].as_str()))
                .unwrap();
            assert_eq!(variant["reduction_class"], decode_class);
            assert_eq!(variant["batch_invariant"], true);
        }
    }
}
