//! Geometry shared by host validation and the Metal source prefix.

pub(crate) const DELTA_RULE_MAX_KEYS: u64 = 256;
pub(crate) const STEEL_256_FLOAT: [u32; 2] = [16, 8];
pub(crate) const STEEL_256_HALF: [u32; 2] = [32, 16];

pub(crate) fn shader_defines() -> String {
    format!(
        concat!(
            "#define FORJA_DELTA_RULE_MAX_KEYS {}\n",
            "#define FORJA_STEEL_256_FLOAT_Q {}\n#define FORJA_STEEL_256_FLOAT_K {}\n",
            "#define FORJA_STEEL_256_HALF_Q {}\n#define FORJA_STEEL_256_HALF_K {}\n",
        ),
        DELTA_RULE_MAX_KEYS,
        STEEL_256_FLOAT[0],
        STEEL_256_FLOAT[1],
        STEEL_256_HALF[0],
        STEEL_256_HALF[1]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_register_capacity_matches_the_uploaded_shader_bound() {
        assert_eq!(DELTA_RULE_MAX_KEYS.div_ceil(32), 8);
        assert!(crate::storage::shader_source().starts_with(&shader_defines()));
    }

    #[test]
    fn wide_steel_tiles_respect_the_shared_memory_budget() {
        let bytes = |tile: [u32; 2], element_bytes: u32| {
            tile[0] * (256 + 16 / element_bytes) * element_bytes
                + (tile[1] + 16 / element_bytes) * 256 * element_bytes
        };
        assert_eq!(bytes(STEEL_256_FLOAT, 4), 28_928);
        assert_eq!(bytes(STEEL_256_HALF, 2), 29_184);
        for (tile, element_bytes) in [(STEEL_256_FLOAT, 4), (STEEL_256_HALF, 2)] {
            assert!(bytes(tile, element_bytes) + 4 * 4 <= 32 * 1024);
        }
    }
}
