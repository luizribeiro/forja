mod carry;
mod gram;
mod scan;
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
