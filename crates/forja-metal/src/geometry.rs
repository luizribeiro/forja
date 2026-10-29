//! Geometry shared by host validation and the Metal source prefix.

pub(crate) const DELTA_RULE_MAX_KEYS: u64 = 256;

pub(crate) fn shader_defines() -> String {
    format!("#define FORJA_DELTA_RULE_MAX_KEYS {DELTA_RULE_MAX_KEYS}\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_register_capacity_matches_the_uploaded_shader_bound() {
        assert_eq!(DELTA_RULE_MAX_KEYS.div_ceil(32), 8);
        assert!(crate::storage::shader_source().starts_with(&shader_defines()));
    }
}
