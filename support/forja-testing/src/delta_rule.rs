//! Independent token-by-token state reference for block differentials.

/// Advances one head using float32 recurrence, returning all output rows and the final state.
/// Inputs use contiguous token/key or token/value rows; `shape` gives key and value widths.
#[must_use]
pub fn recurrent(
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

#[cfg(test)]
mod tests {
    #[test]
    fn recurrence_matches_an_explicit_decay_prediction_update() {
        let (output, state) = super::recurrent(
            &[1.0, 2.0],
            &[0.5, 1.0],
            &[3.0],
            &[0.5_f32.ln()],
            &[0.25],
            &[2.0, 4.0],
            [2, 1],
        );
        assert_eq!(output, [5.3125]);
        assert_eq!(state, [1.0625, 2.125]);
    }
}
