//! Reduced-access capabilities over ordinary Metal allocations.
#[path = "../../../support/forja-testing/test-support/read_only_views.rs"]
mod common;

#[test]
fn lent_read_only_views_preserve_access_rights() {
    common::check(&forja_metal::MetalBackend::new().unwrap());
}
