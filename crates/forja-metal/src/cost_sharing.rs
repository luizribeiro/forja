use forja_core::KernelCost;

fn quotient(total: u64, divisor: u64) -> Option<u64> {
    if total.checked_rem(divisor)? != 0 {
        return None;
    }
    total.checked_div(divisor)
}

pub(crate) fn weights(parent: KernelCost, mut child: KernelCost) -> Option<KernelCost> {
    if child.weight_bytes == 0 {
        return Some(child);
    }
    let projections = quotient(parent.weight_bytes, child.weight_bytes)?;
    child.minimum_weight_bytes = quotient(parent.minimum_weight_bytes, projections)?;
    child.maximum_weight_bytes = quotient(parent.maximum_weight_bytes, projections)?;
    (child.minimum_weight_bytes <= child.maximum_weight_bytes
        && child.maximum_weight_bytes <= child.weight_bytes)
        .then_some(child)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_footprints_require_exact_division_at_every_stage() {
        let parent = KernelCost {
            weight_bytes: 120,
            minimum_weight_bytes: 40,
            maximum_weight_bytes: 80,
            ..KernelCost::default()
        };
        let child = KernelCost {
            weight_bytes: 60,
            ..KernelCost::default()
        };
        let shared = weights(parent, child).unwrap();
        assert_eq!(
            (shared.minimum_weight_bytes, shared.maximum_weight_bytes),
            (20, 40)
        );
        for parent in [
            KernelCost {
                weight_bytes: 121,
                ..parent
            },
            KernelCost {
                minimum_weight_bytes: 41,
                ..parent
            },
            KernelCost {
                maximum_weight_bytes: 81,
                ..parent
            },
            KernelCost {
                weight_bytes: 0,
                ..parent
            },
        ] {
            assert!(weights(parent, child).is_none());
        }
        assert_eq!(
            weights(parent, KernelCost::default()),
            Some(KernelCost::default())
        );
    }
}
