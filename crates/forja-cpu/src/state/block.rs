use crate::{execution_usize, wy::HeadInput};
use forja_core::{BackendError, Tensor};

pub(super) fn execute(
    values: &[Vec<f32>],
    inputs: &[Tensor],
) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
    let shape = inputs[5].layout().shape();
    let heads = execution_usize(shape[0])?;
    let keys = execution_usize(shape[1])?;
    let width = execution_usize(shape[2])?;
    let tokens = execution_usize(inputs[0].layout().shape()[0])?;
    let query_heads = execution_usize(inputs[0].layout().shape()[1])?;
    let gather = |operand: usize, head: usize, count: usize, columns: usize| {
        (0..tokens)
            .flat_map(|row| {
                let start = (row * count + head) * columns;
                values[operand][start..start + columns].iter().copied()
            })
            .collect::<Vec<_>>()
    };
    let mut output = vec![0.0; values[2].len()];
    let mut next = values[5].clone();
    for head in 0..heads {
        let query_head = head / (heads / query_heads);
        let query = gather(0, query_head, query_heads, keys);
        let key = gather(1, query_head, query_heads, keys);
        let value = gather(2, head, heads, width);
        let log_decay = gather(3, head, heads, 1);
        let beta = gather(4, head, heads, 1);
        let input = HeadInput {
            query: &query,
            key: &key,
            value: &value,
            log_decay: &log_decay,
            beta: &beta,
            keys,
            width,
        };
        let start = head * keys * width;
        let end = start + keys * width;
        let (rows, state) = input.scan(&values[5][start..end]);
        next[start..end].copy_from_slice(&state);
        for row in 0..tokens {
            let index = (row * heads + head) * width;
            output[index..index + width].copy_from_slice(&rows[row * width..(row + 1) * width]);
        }
    }
    Ok((output, next))
}
