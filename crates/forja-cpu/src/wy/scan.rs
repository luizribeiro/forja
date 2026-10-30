use super::{BLOCK, carry::Prepared};
use std::ops::Range;

pub(crate) struct HeadInput<'a> {
    pub(crate) query: &'a [f32],
    pub(crate) key: &'a [f32],
    pub(crate) value: &'a [f32],
    pub(crate) log_decay: &'a [f32],
    pub(crate) beta: &'a [f32],
    pub(crate) keys: usize,
    pub(crate) width: usize,
}

impl HeadInput<'_> {
    pub(crate) fn span(&self, rows: Range<usize>) -> Self {
        Self {
            query: &self.query[rows.start * self.keys..rows.end * self.keys],
            key: &self.key[rows.start * self.keys..rows.end * self.keys],
            value: &self.value[rows.start * self.width..rows.end * self.width],
            log_decay: &self.log_decay[rows.clone()],
            beta: &self.beta[rows],
            keys: self.keys,
            width: self.width,
        }
    }

    pub(crate) fn scan(&self, initial: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial.to_vec();
        let mut output = vec![0.0; self.value.len()];
        for start in (0..self.log_decay.len()).step_by(BLOCK) {
            let end = (start + BLOCK).min(self.log_decay.len());
            let block = self.span(start..end);
            let (values, next) = Prepared::new(
                block.query,
                block.key,
                block.value,
                block.log_decay,
                block.beta,
                block.keys,
                block.width,
            )
            .advance(&state);
            output[start * self.width..end * self.width].copy_from_slice(&values);
            state = next;
        }
        (output, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_blocks_are_bit_exact_across_submission_grouping() {
        for count in [1, 7, 33, 63, 64, 65, 4097] {
            let data = super::super::tests::data(count, 7, 7);
            let log_decay = (0..count)
                .map(|i| {
                    if i % 97 == 93 {
                        f32::NEG_INFINITY
                    } else {
                        -0.015
                    }
                })
                .collect::<Vec<_>>();
            let input = HeadInput {
                query: &data.query,
                key: &data.key,
                value: &data.value,
                beta: &data.beta,
                log_decay: &log_decay,
                keys: 7,
                width: 7,
            };
            for initial in [0.0, 0.125] {
                let state = vec![initial; 49];
                let (expected_output, expected_state) = input.scan(&state);
                for grouping in [64, 512, 2048] {
                    let mut state = state.clone();
                    let mut output = Vec::new();
                    for start in (0..count).step_by(grouping) {
                        let end = (start + grouping).min(count);
                        let (values, next) = input.span(start..end).scan(&state);
                        output.extend(values);
                        state = next;
                    }
                    for (expected, actual) in
                        [(&expected_output, &output), (&expected_state, &state)]
                    {
                        assert_eq!(
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}
