use forja_core::DispatchProfile;

/// Reports source-level private accumulator storage, without claiming compiler register allocation.
#[must_use]
pub fn private_accumulator_values_per_thread(
    dispatch: &DispatchProfile,
    kernel: &str,
) -> Option<u64> {
    let width = dispatch
        .outputs
        .first()
        .and_then(|tensor| tensor.shape.last())
        .copied()
        .map(u64::from);
    pressure(
        kernel,
        dispatch.variant,
        width,
        dispatch.inputs.get(1).map(|tensor| tensor.dtype),
    )
}

fn pressure(
    kernel: &str,
    variant: Option<&str>,
    width: Option<u64>,
    cache: Option<forja_core::DType>,
) -> Option<u64> {
    let bounded = variant.is_some_and(crate::variants::bounded_name);
    match kernel {
        "state_block_carry_mma" => Some(crate::geometry::DELTA_RULE_MAX_KEYS.div_ceil(8) * 2),
        "state_block_carry" => Some(crate::geometry::DELTA_RULE_MAX_KEYS),
        "state_delta_rule" => Some(crate::geometry::DELTA_RULE_MAX_KEYS.div_ceil(32)),
        "state_delta_rule_block" => Some(
            if variant.is_some_and(|name| {
                matches!(
                    name,
                    "delta-rule-block.wy64-mma-v16" | "delta-rule-block.wy64-mma-v32"
                )
            }) {
                crate::geometry::DELTA_RULE_MAX_KEYS.div_ceil(8) * 2
            } else {
                crate::geometry::DELTA_RULE_MAX_KEYS
            },
        ),
        name if name.starts_with("steel_attention_")
            || name.starts_with("paged_steel_attention_") =>
        {
            width.and_then(|width| (width / 4).checked_mul(if bounded { 2 } else { 1 }))
        }
        name if name.contains("independent_") => width.and_then(|width| {
            let extra = bounded && cache == Some(forja_core::DType::F32);
            (width / 32).checked_mul(if extra { 3 } else { 2 })
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_is_a_source_estimate_instead_of_an_observed_spill() {
        let source = include_str!("state_block_carry_mma.metal");
        assert!(source.contains("float2 state[(FORJA_DELTA_RULE_MAX_KEYS+7)/8]"));
        let row = pressure("state_block_carry_mma", None, None, None);
        assert_eq!(row, Some(64));
        for (variant, expected) in [("sdpa.steel-256", 64), ("sdpa.steel-bounded-256", 128)] {
            assert_eq!(
                pressure("steel_attention_256", Some(variant), Some(256), None),
                Some(expected)
            );
        }
        for (cache, expected) in [(forja_core::DType::F32, 24), (forja_core::DType::F16, 16)] {
            assert_eq!(
                pressure(
                    "paged_sdpa_2pass_1_independent_256",
                    Some("paged-sdpa.vector-bounded-256-two-pass"),
                    Some(256),
                    Some(cache)
                ),
                Some(expected)
            );
        }
        assert_eq!(pressure("copy_strided", None, Some(33), None), None);
    }
}
