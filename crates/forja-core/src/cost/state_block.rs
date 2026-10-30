use super::{KernelCost, product};
use crate::{Dispatch, Op, StateOp, Tensor};

const TOKENS: u64 = 64;

fn checked(values: &[u64]) -> Option<u64> {
    product(values)
}

fn add(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

fn triangular(rows: u64) -> Option<u64> {
    rows.checked_mul(rows.checked_add(1)?)?.checked_div(2)
}

fn interval_reads(rows: u64) -> Option<u64> {
    checked(&[rows, rows.saturating_sub(1), rows.checked_add(1)?])?.checked_div(6)
}

fn block_terms(tokens: u64, heads: u64) -> Option<[u64; 3]> {
    let full = tokens / TOKENS;
    let tail = tokens % TOKENS;
    let pairs = checked(&[full, triangular(TOKENS)?])?
        .checked_add(triangular(tail)?)?
        .checked_mul(heads)?;
    let interval = checked(&[full, interval_reads(TOKENS)?])?
        .checked_add(interval_reads(tail)?)?
        .checked_mul(heads)?;
    Some([pairs, interval, pairs])
}

pub(crate) fn stages(inputs: &[Tensor], output: &Tensor) -> Option<[KernelCost; 4]> {
    let shape = inputs[5].layout().shape();
    let [heads, keys, values] = [
        u64::from(shape[0]),
        u64::from(shape[1]),
        u64::from(shape[2]),
    ];
    let tokens = u64::from(inputs[0].layout().shape()[0]);
    let blocks = tokens.div_ceil(TOKENS);
    let panel = keys.checked_add(values)?;
    let rows = checked(&[tokens, heads])?;
    let [pairs, interval, prefix] = block_terms(tokens, heads)?;
    let strict = pairs.checked_sub(rows)?;
    let matrix = checked(&[blocks, heads, TOKENS, TOKENS])?;
    let state = checked(&[heads, keys, values, 4])?;
    let metadata = 28;
    let auth = |threads| checked(&[threads, metadata]);
    let value_bytes = inputs[2].layout().dtype().byte_size();
    let beta_bytes = inputs[4].layout().dtype().byte_size();

    let gram_read = add(&[
        checked(&[pairs.checked_add(strict)?, keys, 8])?,
        checked(&[interval.checked_add(prefix)?, 4])?,
        checked(&[strict, beta_bytes])?,
        auth(matrix)?,
    ])?;
    let gram_write = checked(&[matrix, 2, 4])?.checked_add(checked(&[rows, 2, 4])?)?;
    let gram = KernelCost {
        read_bytes: gram_read,
        write_bytes: gram_write,
        scratch_write_bytes: gram_write,
        flops: add(&[
            checked(&[pairs.checked_add(strict)?, keys, 2])?,
            checked(&[strict, 2])?,
            pairs,
        ])?,
        scalar_ops: add(&[matrix, interval, prefix, pairs])?,
        sfu_ops: pairs.checked_add(rows)?,
        ..KernelCost::default()
    };

    let solved_cells = checked(&[blocks, heads, TOKENS, panel])?;
    let solve_read_scratch = add(&[checked(&[strict, panel, 8])?, checked(&[rows, keys, 4])?])?;
    let solve_write = checked(&[solved_cells, 4])?;
    let solve = KernelCost {
        read_bytes: add(&[
            solve_read_scratch,
            checked(&[rows, panel, beta_bytes])?,
            checked(&[rows, values, value_bytes])?,
            checked(&[rows, keys, 4])?,
            auth(checked(&[blocks, heads, panel])?)?,
        ])?,
        write_bytes: solve_write,
        scratch_read_bytes: solve_read_scratch,
        scratch_write_bytes: solve_write,
        flops: add(&[
            checked(&[strict, panel, 2])?,
            checked(&[rows, values])?,
            checked(&[rows, keys, 2])?,
        ])?,
        scalar_ops: checked(&[solved_cells, 2])?,
        ..KernelCost::default()
    };

    let row_values = checked(&[rows, values])?;
    let key_values = checked(&[row_values, keys])?;
    let carry_read_scratch = add(&[
        checked(&[key_values, 3, 4])?,
        checked(&[row_values, 2, 4])?,
        checked(&[blocks, heads, values, 4])?,
    ])?;
    let carry_write_scratch = checked(&[row_values, 2, 4])?;
    let carry = KernelCost {
        read_bytes: add(&[
            carry_read_scratch,
            state,
            checked(&[key_values, 2, 4])?,
            auth(checked(&[heads, values])?)?,
        ])?,
        write_bytes: state.checked_add(carry_write_scratch)?,
        state_read_bytes: state,
        state_write_bytes: state,
        scratch_read_bytes: carry_read_scratch,
        scratch_write_bytes: carry_write_scratch,
        flops: add(&[
            checked(&[key_values, 6])?,
            checked(&[row_values, 3])?,
            checked(&[blocks, heads, keys, values, 2])?,
        ])?,
        scalar_ops: checked(&[row_values, 2])?,
        ..KernelCost::default()
    };

    let publish = publish_cost(
        row_values,
        pairs,
        values,
        metadata,
        output.layout().dtype().byte_size(),
    )?;
    Some([gram, solve, carry, publish])
}

fn publish_cost(
    rows: u64,
    pairs: u64,
    values: u64,
    metadata: u64,
    dtype_bytes: u64,
) -> Option<KernelCost> {
    let output_scratch = add(&[checked(&[pairs, values, 8])?, checked(&[rows, 4])?])?;
    Some(KernelCost {
        read_bytes: output_scratch.checked_add(checked(&[rows, metadata])?)?,
        write_bytes: checked(&[rows, dtype_bytes])?,
        scratch_read_bytes: output_scratch,
        flops: add(&[checked(&[pairs, values, 2])?, rows])?,
        scalar_ops: checked(&[rows, TOKENS])?,
        ..KernelCost::default()
    })
}

pub(crate) fn composite(stages: &[KernelCost; 4]) -> Option<KernelCost> {
    stages
        .iter()
        .try_fold(KernelCost::default(), |sum, stage| sum.checked_add(*stage))
}

pub(super) fn dispatch_stages(dispatch: &Dispatch) -> Option<[KernelCost; 4]> {
    if !matches!(dispatch.op(), Op::State(StateOp::DeltaRuleBlock(_))) {
        return None;
    }
    stages(dispatch.primitive_inputs(), dispatch.output())
}
