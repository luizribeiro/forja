//! Compile-time contracts for Rust-syntax kernels.

#[test]
fn kernel_syntax_contracts() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/ui/kernel-pass-arithmetic.rs");
    cases.pass("tests/ui/kernel-pass-methods.rs");
    cases.pass("tests/ui/kernel-pass-signatures.rs");
    cases.pass("tests/ui/kernel-pass-typed-expressions.rs");
    cases.compile_fail("tests/ui/kernel-fail-closure.rs");
    cases.compile_fail("tests/ui/kernel-fail-loop.rs");
    cases.compile_fail("tests/ui/kernel-fail-match.rs");
    cases.compile_fail("tests/ui/kernel-fail-method-arity.rs");
    cases.compile_fail("tests/ui/kernel-fail-float-nonfinite.rs");
    cases.compile_fail("tests/ui/kernel-fail-generic.rs");
    cases.compile_fail("tests/ui/kernel-fail-output-bool.rs");
    cases.compile_fail("tests/ui/kernel-fail-parameter-type.rs");
    cases.compile_fail("tests/ui/kernel-fail-unused-tensor.rs");
}
