use super::*;

fn bit_digest(values: &[f32]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
}

#[test]
fn metal_shipped_attention_matches_frozen_reference_past_flush_periods() {
    // Frozen from c3d2bf069aa2a46b42ab34203887cfa3a4408959 on Metal 4.
    for (width, rows, variant, digest) in [
        (64, 7, "sdpa.steel", 0xc549_7f95_d33d_1e08),
        (64, 7, "paged-sdpa.steel", 0xc549_7f95_d33d_1e08),
        (128, 7, "sdpa.steel", 0x0981_07bf_1591_a8b7),
        (128, 7, "paged-sdpa.steel", 0x0981_07bf_1591_a8b7),
        (256, 7, "sdpa.steel-256", 0x0949_0590_8829_1a35),
        (256, 7, "paged-sdpa.steel-256", 0x0949_0590_8829_1a35),
        (256, 1, "sdpa.vector-256-two-pass", 0xc12c_6410_a498_329e),
        (
            256,
            1,
            "paged-sdpa.vector-256-two-pass",
            0xab8d_2bb8_3e75_e6d3,
        ),
    ] {
        assert_eq!(
            bit_digest(&attend(width, 4097, rows, DType::F32, variant)),
            digest,
            "{width} {variant}"
        );
    }
}
