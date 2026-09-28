//! Compile-time contracts for Rust-syntax kernels.

#[test]
fn kernel_syntax_contracts() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/kernel-pass-arithmetic.rs");
    cases.pass("tests/ui/kernel-pass-methods.rs");
    cases.pass("tests/ui/kernel-pass-signatures.rs");
    cases.compile_fail("tests/ui/kernel-fail-closure.rs");
    cases.compile_fail("tests/ui/kernel-fail-loop.rs");
    cases.compile_fail("tests/ui/kernel-fail-match.rs");
    cases.compile_fail("tests/ui/kernel-fail-method-arity.rs");
}
