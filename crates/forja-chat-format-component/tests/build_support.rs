//! Build-input selection checks.

#[path = "../build_support.rs"]
mod build_support;

use std::path::Path;

#[test]
fn rejects_an_unconfigured_production_build() {
    let error = build_support::source(
        "FORJA_CHAT_TEMPLATE",
        None,
        Path::new("missing-fixture"),
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("test-fixtures feature"));
}

#[test]
fn allows_an_explicit_test_fixture() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../chat-formats/qwen3-coder-xml/tests/fixtures/chat_template.jinja");
    assert_eq!(
        build_support::source("FORJA_CHAT_TEMPLATE", None, &fixture, true).unwrap(),
        fixture
    );
}
