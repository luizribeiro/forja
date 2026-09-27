//! Compile-time coverage for model loading derives.

#[test]
fn load_derive_contracts() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/load-pass.rs");
    cases.compile_fail("tests/ui/load-vec-no-count.rs");
}
