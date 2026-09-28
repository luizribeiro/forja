//! Compile-time contracts for Rust-syntax kernels.

#[test]
fn kernel_syntax_contracts() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/kernel-pass-arithmetic.rs");
    cases.pass("tests/ui/kernel-pass-methods.rs");
    cases.pass("tests/ui/kernel-pass-signatures.rs");
    cases.pass("tests/ui/kernel-pass-typed-expressions.rs");
    cases.pass("tests/ui/kernel-pass-row-expressions.rs");
    cases.pass("tests/ui/kernel-pass-helpers.rs");
    cases.compile_fail("tests/ui/kernel-fail-*.rs");
}
