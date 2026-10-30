use super::{BLOCK, gram::Gram, tree_sum};

pub(super) fn solve(gram: &Gram, rhs: &[f32], width: usize) -> Vec<f32> {
    let mut result = vec![0.0; BLOCK * width];
    for row in 0..gram.rows {
        for column in 0..width {
            let prediction = tree_sum(BLOCK, |previous| {
                if previous < row {
                    gram.lower[row * BLOCK + previous] * result[previous * width + column]
                } else {
                    0.0
                }
            });
            result[row * width + column] = rhs[row * width + column] - prediction;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_lower_solve_recovers_odd_panels_without_reading_future_rhs() {
        for rows in [1, 7, 33, 63, BLOCK] {
            let keys = vec![0.125; rows];
            let gram = Gram::new(&keys, &keys, &vec![-0.01; rows], &vec![0.25; rows], 1);
            for width in [1, 7, 33] {
                let expected = (0..rows * width)
                    .map(|i| f32::from(u16::try_from(i % 17).unwrap()) * 0.0625 - 0.5)
                    .collect::<Vec<_>>();
                let mut rhs = vec![f32::NAN; BLOCK * width];
                for row in 0..rows {
                    for column in 0..width {
                        let sum = (0..row)
                            .map(|previous| {
                                gram.lower[row * BLOCK + previous]
                                    * expected[previous * width + column]
                            })
                            .sum::<f32>();
                        rhs[row * width + column] = expected[row * width + column] + sum;
                    }
                }
                let actual = solve(&gram, &rhs, width);
                let error =
                    forja_testing::normwise_relative_error(&expected, &actual[..rows * width]);
                assert!(
                    error <= forja_testing::F32_TOLERANCE,
                    "relative error {error}"
                );
                assert!(actual[rows * width..].iter().all(|x| x.to_bits() == 0));
            }
        }
    }
}
