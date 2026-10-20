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
        "6e679b8cf574892edba3314b621aea0163254ec5319ace6225ebe2d77383b209"
    );

    let mut json: serde_json::Value = serde_json::from_str(&catalog(&["--json"])).unwrap();
    json["device"] = "Test GPU".into();
    let json = json.to_string() + "\n";
    assert_eq!(
        digest(&json),
        "664efd5b4932592cf3188a12fed43fe1d19cdf5458bb072284fb0f252c30d99b"
    );
}
