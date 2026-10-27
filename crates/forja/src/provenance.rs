pub(crate) const REVISION: &str = env!("FORJA_BUILD_REV");
pub(crate) const BUILD_ENVIRONMENT: &str = env!("FORJA_BUILD_ENV");

pub(crate) fn commit() -> &'static str {
    REVISION.strip_suffix("-dirty").unwrap_or(REVISION)
}

pub(crate) fn dirty() -> bool {
    REVISION.ends_with("-dirty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_revision_has_a_commit_and_dirty_state() {
        assert_ne!(commit(), "");
        assert_eq!(dirty(), REVISION.ends_with("-dirty"));
        assert!(BUILD_ENVIRONMENT.contains("compiler="));
        assert!(BUILD_ENVIRONMENT.contains("OPT_LEVEL="));
    }
}
