mod carry;
mod gram;
mod scan;
pub(crate) use scan::HeadInput;
mod solve;

const BLOCK: usize = forja_core::DeltaBlockScratch::TOKENS;

fn tree_sum(count: usize, value: impl Fn(usize) -> f32) -> f32 {
    let per_lane = count.div_ceil(32);
    let mut partial = [0.0_f32; 32];
    for (lane, sum) in partial.iter_mut().enumerate() {
        for item in 0..per_lane {
            let index = lane * per_lane + item;
            if index < count {
                *sum += value(index);
            }
        }
    }
    for offset in [16, 8, 4, 2, 1] {
        for lane in 0..offset {
            partial[lane] += partial[lane + offset];
        }
    }
    partial[0]
}

/// Returns padded WY scratch, public output and final state for one head/block.
/// Inputs are contiguous query, key, value, log-decay and beta rows; shape gives key/value widths.
///
/// # Errors
/// Refuses inconsistent widths, lengths or more than one canonical block.
#[cfg(feature = "test-oracles")]
pub fn block_reference(
    inputs: [&[f32]; 5],
    initial: &[f32],
    shape: [u32; 2],
) -> Result<[Vec<f32>; 3], forja_core::BackendError> {
    let bad = forja_core::BackendError::InvalidInput;
    let [keys, width] = shape.map(|value| usize::try_from(value).map_err(|_| bad));
    let (keys, width) = (keys?, width?);
    let [q, k, v, g, beta] = inputs;
    if !(1..=BLOCK).contains(&g.len())
        || keys == 0
        || width == 0
        || Some(q.len()) != g.len().checked_mul(keys)
        || k.len() != q.len()
        || Some(v.len()) != g.len().checked_mul(width)
        || beta.len() != g.len()
        || Some(initial.len()) != keys.checked_mul(width)
    {
        return Err(bad);
    }
    Ok(carry::Prepared::new(q, k, v, g, beta, keys, width).reference(initial))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) struct Fixture {
        pub(super) query: Vec<f32>,
        pub(super) key: Vec<f32>,
        pub(super) value: Vec<f32>,
        pub(super) beta: Vec<f32>,
    }
    pub(super) fn data(rows: usize, keys: usize, width: usize) -> Fixture {
        let mut rng = forja_testing::DeterministicValues::new(73);
        let query = (0..rows * keys)
            .map(|_| rng.next_f32() * 0.125)
            .collect::<Vec<_>>();
        let mut key = query.clone();
        for row in key.chunks_mut(keys) {
            let norm = (row.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for value in row {
                *value /= norm;
            }
        }
        Fixture {
            query,
            key,
            value: (0..rows * width).map(|_| rng.next_f32()).collect(),
            beta: vec![0.25; rows],
        }
    }

    #[cfg(feature = "test-oracles")]
    #[test]
    fn block_stage_reference_checks_geometry_and_known_recurrence() {
        let [scratch, output, state] = super::block_reference(
            [&[1.0, 2.0], &[0.5, 1.0], &[3.0], &[0.5_f32.ln()], &[0.25]],
            &[2.0, 4.0],
            [2, 1],
        )
        .unwrap();
        assert_eq!(scratch[0], 1.0);
        assert_eq!(output, [5.3125]);
        assert_eq!(state, [1.0625, 2.125]);
        assert!(super::block_reference([&[]; 5], &[], [0, 1]).is_err());
        assert!(super::block_reference([&[1.0]; 5], &[0.0], [2, 1]).is_err());
    }

    #[test]
    fn tree_reduction_masks_odd_tails_before_reading() {
        for count in [0, 1, 7, 33, 63, BLOCK, 65, 128] {
            let values = vec![1.0_f32; count];
            assert_eq!(
                tree_sum(count, |i| values[i]),
                f32::from(u16::try_from(count).unwrap())
            );
        }
        let values = [1e20, -1e20, 3.0, 4.0];
        assert_eq!(
            tree_sum(values.len(), |i| values[i]).to_bits(),
            0.0_f32.to_bits()
        );
    }
}
