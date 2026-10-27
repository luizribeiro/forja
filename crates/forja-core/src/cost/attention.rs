use crate::{Dispatch, KernelCost, Op, PagedCopyDirection};

use super::{bytes, product, tensor_bytes};

pub(super) fn count(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    let input = dispatch.primitive_inputs();
    if let Op::PagedCopy { direction, .. } = dispatch.op() {
        let (ordinary, pool) = match direction {
            PagedCopyDirection::IntoPages => (&input[0], dispatch.output()),
            PagedCopyDirection::OutOfPages => (dispatch.output(), &input[0]),
        };
        let elements = ordinary.layout().element_count();
        let pool_bytes = elements.checked_mul(pool.layout().dtype().byte_size())?;
        (cost.read_bytes, cost.write_bytes) = match direction {
            PagedCopyDirection::IntoPages => (tensor_bytes(ordinary)?, pool_bytes),
            PagedCopyDirection::OutOfPages => (pool_bytes, tensor_bytes(ordinary)?),
        };
        cost.read_bytes = cost.read_bytes.checked_add(bytes(&input[1..])?)?;
        return Some(cost);
    }
    let (causal, q_start, key_length, paged) = match dispatch.op() {
        Op::Sdpa {
            causal, q_start, ..
        } => (causal, q_start, input[1].layout().shape()[1], false),
        Op::PagedSdpa {
            causal,
            q_start,
            key_length,
            ..
        } => (causal, q_start, key_length, true),
        _ => return Some(cost),
    };
    let query = input[0].layout().shape();
    let keys = input[1].layout().shape();
    let values = input[2].layout().shape();
    let [heads, queries, width] = <[u32; 3]>::try_from(query).ok()?.map(u64::from);
    let value_width = u64::from(*values.last()?);
    let kv_heads = u64::from(keys[if paged { 2 } else { 0 }]);
    let key_length = u64::from(key_length);
    let pairs = if causal {
        queries
            .checked_mul(u64::from(q_start).checked_add(1)?)?
            .checked_add(
                queries
                    .checked_mul(queries.checked_sub(1)?)?
                    .checked_div(2)?,
            )?
    } else {
        queries.checked_mul(key_length)?
    };
    cost.flops = product(&[2, heads, pairs, width.checked_add(value_width)?])?;
    cost.scalar_ops = product(&[heads, pairs, 5])?;
    cost.kv_bytes = product(&[
        kv_heads,
        key_length,
        width.checked_add(value_width)?,
        input[1].layout().dtype().byte_size(),
    ])?;
    cost.read_bytes = tensor_bytes(&input[0])?.checked_add(cost.kv_bytes)?;
    if paged {
        cost.read_bytes = cost.read_bytes.checked_add(bytes(&input[3..])?)?;
    }
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommandList, DType, cost::tests::tensor};

    #[test]
    fn attention_counts_causal_pairs_and_kv_once() {
        let query = tensor(1, DType::F32, &[16, 7, 128]);
        let key = tensor(2, DType::F16, &[8, 33, 128]);
        let value = tensor(3, DType::F16, &[8, 33, 128]);
        let output = tensor(4, DType::F32, &[16, 7, 128]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 1.0,
                    causal: true,
                    q_start: 26,
                },
                &[&query, &key, &value],
                &output,
            )
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        assert_eq!(
            cost.flops,
            2 * 16 * (27 + 28 + 29 + 30 + 31 + 32 + 33) * 256
        );
        assert_eq!(cost.kv_bytes, 2 * 8 * 33 * 128 * 2);
    }

    #[test]
    fn paged_attention_ignores_unused_pool_capacity() {
        let query = tensor(1, DType::F32, &[16, 1, 128]);
        let key = tensor(2, DType::F16, &[99, 16, 8, 128]);
        let value = tensor(3, DType::F16, &[99, 16, 8, 128]);
        let table = tensor(4, DType::U32, &[3, 2]);
        let generations = tensor(5, DType::U32, &[99]);
        let output = tensor(6, DType::F32, &[16, 1, 128]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::PagedSdpa {
                    scale: 1.0,
                    causal: true,
                    q_start: 32,
                    key_length: 33,
                    page_tokens: 16,
                },
                &[&query, &key, &value, &table, &generations],
                &output,
            )
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        assert_eq!(cost.flops, 4 * 16 * 33 * 128);
        assert_eq!(cost.kv_bytes, 2 * 8 * 33 * 128 * 2);
    }
}
