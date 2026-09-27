#![cfg(feature = "native")]

//! CPU-backed native SDK coverage.

mod support;

#[test]
fn structured_ops_match_the_cpu_backend() {
    support::matmul_rope_embedding_match_cpu();
}
