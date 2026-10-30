use super::{BLOCK, tree_sum};

pub(super) struct Gram {
    pub(super) rows: usize,
    pub(super) lower: Vec<f32>,
    pub(super) query_key: Vec<f32>,
    pub(super) prefix_decay: Vec<f32>,
    pub(super) tail_decay: Vec<f32>,
}

impl Gram {
    pub(super) fn new(q: &[f32], k: &[f32], g: &[f32], beta: &[f32], keys: usize) -> Self {
        let rows = g.len();
        let mut result = Self {
            rows,
            lower: vec![0.0; BLOCK * BLOCK],
            query_key: vec![0.0; BLOCK * BLOCK],
            prefix_decay: vec![0.0; BLOCK],
            tail_decay: vec![0.0; BLOCK],
        };
        let mut prefix = 0.0_f32;
        for row in 0..rows {
            prefix += g[row];
            result.prefix_decay[row] = prefix.exp();
            let mut interval = 0.0_f32;
            for column in (0..=row).rev() {
                let decay = interval.exp();
                let dot =
                    |left: &[f32]| tree_sum(keys, |i| left[row * keys + i] * k[column * keys + i]);
                let index = row * BLOCK + column;
                result.lower[index] = if row == column {
                    1.0
                } else {
                    beta[row] * dot(k) * decay
                };
                result.query_key[index] = dot(q) * decay;
                if row + 1 == rows {
                    result.tail_decay[column] = decay;
                }
                interval += g[column];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_terms_keep_unit_diagonal_and_mask_future_rows() {
        let mut q = vec![f32::NAN; BLOCK];
        let mut k = q.clone();
        q[..2].copy_from_slice(&[1.0, 2.0]);
        k[..2].copy_from_slice(&[0.5, 0.25]);
        let terms = Gram::new(&q, &k, &[0.5_f32.ln(); 2], &[0.25, 0.5], 1);
        assert_eq!(terms.rows, 2);
        assert_eq!(terms.lower[0], 1.0);
        assert_eq!(terms.lower[BLOCK], 0.03125);
        assert_eq!(terms.lower[BLOCK + 1], 1.0);
        assert_eq!(terms.query_key[0], 0.5);
        assert_eq!(terms.query_key[BLOCK], 0.5);
        assert_eq!(terms.query_key[BLOCK + 1], 0.5);
        assert_eq!(&terms.prefix_decay[..2], &[0.5, 0.25]);
        assert_eq!(&terms.tail_decay[..2], &[0.5, 1.0]);
        for row in 0..BLOCK {
            for column in 0..BLOCK {
                if column > row || row >= 2 {
                    assert_eq!(terms.lower[row * BLOCK + column], 0.0);
                    assert_eq!(terms.query_key[row * BLOCK + column], 0.0);
                }
            }
        }
    }

    #[test]
    fn interval_decay_survives_prefix_underflow_without_nan() {
        for first in [-1000.0, f32::NEG_INFINITY] {
            let terms = Gram::new(&[1.0, 2.0], &[1.0, 2.0], &[first, -2.0], &[1.0; 2], 1);
            assert_eq!(terms.prefix_decay[0], 0.0);
            assert_eq!(terms.prefix_decay[1], 0.0);
            assert_eq!(terms.tail_decay[0], (-2.0_f32).exp());
            assert_eq!(terms.lower[BLOCK], 2.0 * (-2.0_f32).exp());
            assert_eq!(terms.lower[BLOCK + 1], 1.0);
            assert!(terms.query_key.iter().all(|x| x.is_finite()));
        }
    }
}
