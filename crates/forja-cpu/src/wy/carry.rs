use super::{BLOCK, gram::Gram, solve::solve, tree_sum};

pub(super) struct Prepared<'a> {
    q: &'a [f32],
    k: &'a [f32],
    gram: Gram,
    solved_values: Vec<f32>,
    solved_keys: Vec<f32>,
    keys: usize,
    width: usize,
}

impl<'a> Prepared<'a> {
    pub(super) fn new(
        q: &'a [f32],
        k: &'a [f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        keys: usize,
        width: usize,
    ) -> Self {
        let gram = Gram::new(q, k, g, beta, keys);
        let mut solved_values = vec![0.0; g.len() * width];
        let mut solved_keys = vec![0.0; g.len() * keys];
        for row in 0..g.len() {
            for column in 0..width {
                solved_values[row * width + column] = beta[row] * v[row * width + column];
            }
            for key in 0..keys {
                solved_keys[row * keys + key] =
                    beta[row] * gram.prefix_decay[row] * k[row * keys + key];
            }
        }
        Self {
            q,
            k,
            solved_values: solve(&gram, &solved_values, width),
            solved_keys: solve(&gram, &solved_keys, keys),
            gram,
            keys,
            width,
        }
    }

    pub(super) fn advance(&self, state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let rows = self.gram.rows;
        let mut delta = vec![0.0; rows * self.width];
        let mut output = delta.clone();
        for row in 0..rows {
            for column in 0..self.width {
                let index = row * self.width + column;
                let correction = tree_sum(self.keys, |key| {
                    self.solved_keys[row * self.keys + key] * state[key * self.width + column]
                });
                delta[index] = self.solved_values[index] - correction;
                output[index] = self.gram.prefix_decay[row]
                    * tree_sum(self.keys, |key| {
                        self.q[row * self.keys + key] * state[key * self.width + column]
                    });
            }
        }
        for row in 0..rows {
            for column in 0..self.width {
                output[row * self.width + column] += tree_sum(BLOCK, |previous| {
                    if previous <= row {
                        self.gram.query_key[row * BLOCK + previous]
                            * delta[previous * self.width + column]
                    } else {
                        0.0
                    }
                });
            }
        }
        let next = (0..state.len())
            .map(|index| {
                let key = index / self.width;
                let column = index % self.width;
                self.gram.prefix_decay[rows - 1] * state[index]
                    + tree_sum(BLOCK, |row| {
                        if row < rows {
                            (self.gram.tail_decay[row] * self.k[row * self.keys + key])
                                * delta[row * self.width + column]
                        } else {
                            0.0
                        }
                    })
            })
            .collect();
        (output, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recurrent(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        initial: &[f32],
        shape: [usize; 2],
    ) -> (Vec<f32>, Vec<f32>) {
        let [keys, width] = shape;
        let mut state = initial.to_vec();
        let mut output = vec![0.0; v.len()];
        for row in 0..g.len() {
            for column in 0..width {
                let mut prediction = 0.0;
                for key in 0..keys {
                    let index = key * width + column;
                    state[index] *= g[row].exp();
                    prediction += state[index] * k[row * keys + key];
                }
                let delta = beta[row] * (v[row * width + column] - prediction);
                for key in 0..keys {
                    let index = key * width + column;
                    state[index] += k[row * keys + key] * delta;
                    output[row * width + column] += q[row * keys + key] * state[index];
                }
            }
        }
        (output, state)
    }

    #[test]
    fn block_outputs_and_carry_match_recurrence_through_decay_underflow() {
        for rows in [1, 7, 33, 63, BLOCK] {
            for (keys, width) in [(7, 33), (128, 128)] {
                let data = super::super::tests::data(rows, keys, width);
                let (q, k, v, beta) = (&data.query, &data.key, &data.value, &data.beta);
                for (initial, underflow) in [(0.0, false), (0.125, false), (0.125, true)] {
                    let state = vec![initial; keys * width];
                    let g = (0..rows)
                        .map(|i| {
                            if underflow && i % 17 == 9 {
                                f32::NEG_INFINITY
                            } else {
                                -0.015
                            }
                        })
                        .collect::<Vec<_>>();
                    let expected = recurrent(q, k, v, &g, beta, &state, [keys, width]);
                    let actual = Prepared::new(q, k, v, &g, beta, keys, width).advance(&state);
                    for (expected, actual) in [(&expected.0, &actual.0), (&expected.1, &actual.1)] {
                        let error = forja_testing::normwise_relative_error(expected, actual);
                        assert!(
                            error <= forja_testing::F32_TOLERANCE,
                            "rows={rows} keys={keys} width={width} error={error}"
                        );
                    }
                }
            }
        }
    }
}
