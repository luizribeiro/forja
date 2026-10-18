//! Stable command-line catalog snapshots.

use std::{fmt::Write as _, process::Command};

use sha2::{Digest, Sha256};

fn catalog(arguments: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_forja"))
        .args(["--isolated", "catalog"])
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
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
        "b6ef9f668450bd8cb527390f088216ff55004ae9a81eec715635c5ee0cc77fd1"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "84e38d652ea796bc4c2f8106667906fb3c66993c92de0b3ccc8261da2a6901b9"
    );
}
