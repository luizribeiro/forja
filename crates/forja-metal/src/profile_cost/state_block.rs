use forja_core::{Dispatch, KernelCost};

fn mul(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(1_u64, |a, blocks| a.checked_mul(*blocks))
}
fn add(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |a, blocks| a.checked_add(*blocks))
}
fn triangle(n: u64) -> Option<u64> {
    n.checked_mul(n.checked_add(1)?)?.checked_div(2)
}
fn across(tokens: u64, heads: u64, f: impl Fn(u64) -> Option<u64>) -> Option<u64> {
    mul(&[tokens / 64, f(64)?])?
        .checked_add(f(tokens % 64)?)?
        .checked_mul(heads)
}

pub(super) fn stages(dispatch: &Dispatch, panel: u32) -> Option<[KernelCost; 4]> {
    let mut costs = KernelCost::delta_block_stages(dispatch)?;
    let inputs = dispatch.primitive_inputs();
    let shape = inputs[5].layout().shape();
    let [heads, keys, values] = [
        u64::from(shape[0]),
        u64::from(shape[1]),
        u64::from(shape[2]),
    ];
    let tokens = u64::from(inputs[0].layout().shape()[0]);
    let blocks = tokens.div_ceil(64);
    let rows = mul(&[tokens, heads])?;
    let block_heads = mul(&[blocks, heads])?;
    let pairs = across(tokens, heads, triangle)?;
    let strict = pairs.checked_sub(rows)?;
    let intervals = across(tokens, heads, |n| {
        mul(&[n, n.saturating_sub(1), n.checked_add(1)?])?.checked_div(6)
    })?;
    let value_panels = values.div_ceil(u64::from(panel));
    let output_panels = values.div_ceil(32);
    let rhs_panels = keys.checked_add(values)?.div_ceil(32);
    let value_bytes = inputs[2].layout().dtype().byte_size();
    let beta_bytes = inputs[4].layout().dtype().byte_size();
    let auth = |threads| mul(&[threads, 28]);
    let gram_operands = across(tokens, heads, |n| {
        if n <= 32 {
            mul(&[n, 3])
        } else {
            n.checked_mul(5)?.checked_sub(32)
        }
    })?;
    costs[0].read_bytes = add(&[
        mul(&[gram_operands, keys, 4])?,
        mul(&[intervals.checked_add(pairs)?, 4])?,
        mul(&[strict, beta_bytes])?,
        auth(mul(&[512, block_heads])?)?,
    ])?;
    let solve_scratch = add(&[
        mul(&[4096, 4, block_heads, rhs_panels])?,
        mul(&[rows, keys, 4])?,
    ])?;
    costs[1].scratch_read_bytes = solve_scratch;
    costs[1].read_bytes = add(&[
        solve_scratch,
        mul(&[rows, keys.checked_add(values)?, beta_bytes])?,
        mul(&[rows, values, value_bytes])?,
        mul(&[rows, keys, 4])?,
        auth(mul(&[128, block_heads, rhs_panels])?)?,
    ])?;
    let carry_scratch = add(&[
        mul(&[rows, keys, value_panels, 8])?,
        mul(&[rows, values, 8])?,
        mul(&[16, block_heads, value_panels, u64::from(panel)])?,
    ])?;
    costs[2].scratch_read_bytes = carry_scratch;
    costs[2].read_bytes = add(&[
        carry_scratch,
        costs[2].state_read_bytes,
        mul(&[rows, keys, value_panels, 8])?,
        auth(mul(&[4, heads, value_panels, u64::from(panel)])?)?,
    ])?;
    let diagonal = across(tokens, heads, |n| {
        mul(&[n / 8, 36])?.checked_add(triangle(n % 8)?)
    })?;
    let output_scratch = add(&[
        mul(&[pairs, output_panels, 4])?,
        mul(&[rows, values, 12])?,
        mul(&[diagonal, values, 8])?,
    ])?;
    costs[3].scratch_read_bytes = output_scratch;
    costs[3].read_bytes =
        output_scratch.checked_add(auth(mul(&[256, block_heads, output_panels])?)?)?;
    Some(costs)
}

pub(crate) fn composite(dispatch: &Dispatch) -> Option<KernelCost> {
    let panel = crate::encoding::delta_block_mma_panel(dispatch)?;
    stages(dispatch, panel)?
        .iter()
        .try_fold(KernelCost::default(), |sum, cost| sum.checked_add(*cost))
}
