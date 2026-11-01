use forja_core::{DType, Dispatch, Op};

pub(super) fn scalar_ops(dispatch: &Dispatch) -> Option<u64> {
    let Some(selected) = dispatch.backend_data::<crate::ValidatedVariant>() else {
        return Some(0);
    };
    if !selected.uses_bounded_attention() {
        return Some(0);
    }
    let input = dispatch.primitive_inputs();
    let query = input[0].layout().shape();
    let [heads, queries, width] = <[u32; 3]>::try_from(query).ok()?.map(u64::from);
    let (causal, start, keys) = match dispatch.op() {
        Op::Sdpa {
            causal, q_start, ..
        } => (
            causal,
            u64::from(q_start),
            u64::from(input[1].layout().shape()[1]),
        ),
        Op::PagedSdpa {
            causal,
            q_start,
            key_length,
            ..
        } => (causal, u64::from(q_start), u64::from(key_length)),
        _ => return None,
    };
    let model = crate::variants::profile_selection(dispatch).2;
    if let Some((_, tile)) = model
        .attention_tiles
        .iter()
        .find(|(d, _)| u64::from(*d) == width)
    {
        return steel(
            heads,
            queries,
            width,
            start,
            keys,
            causal,
            tile.map(u64::from),
        );
    }
    if input[1].layout().dtype() != DType::F32 {
        return Some(0);
    }
    vector(heads, width, keys, 32)
}

fn contributions(rows: u64) -> Option<u64> {
    rows.checked_add(rows / crate::geometry::ATTENTION_PARTIAL_PERIOD)?
        .checked_add(1)
}

fn vector(heads: u64, width: u64, keys: u64, blocks: u64) -> Option<u64> {
    let chunk = keys.div_ceil(blocks);
    let mut total = 0_u64;
    for block in 0..blocks {
        let start = block.checked_mul(chunk)?.min(keys);
        let rows = keys.checked_sub(start)?.min(chunk);
        total = total.checked_add(contributions(rows)?)?;
    }
    heads.checked_mul(width.checked_add(1)?)?.checked_mul(total)
}

fn steel(
    heads: u64,
    queries: u64,
    width: u64,
    start: u64,
    keys: u64,
    causal: bool,
    tile: [u64; 2],
) -> Option<u64> {
    let mut total = 0_u64;
    for block in 0..queries.div_ceil(tile[0]) {
        let end = block.checked_add(1)?.checked_mul(tile[0])?.min(queries);
        let limit = if causal {
            start.checked_add(end)?.min(keys)
        } else {
            keys
        };
        total = total.checked_add(contributions(limit.div_ceil(tile[1]))?)?;
    }
    total
        .checked_mul(heads)?
        .checked_mul(tile[0])?
        .checked_mul(width.checked_add(1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_partials_count_rescales_flushes_and_final_merges() {
        for width in [64, 128, 256] {
            for (keys, contributions) in [(1, 33), (64, 96), (65, 97), (4097, 4192), (32768, 33312)]
            {
                assert_eq!(
                    vector(24, width, keys, 32),
                    Some(24 * (width + 1) * contributions)
                );
            }
        }
        assert_eq!(
            steel(24, 1, 256, 0, 1, true, [16, 8]),
            Some(24 * 16 * 257 * 2)
        );
        assert_eq!(
            steel(24, 1, 256, 511, 512, true, [16, 8]),
            Some(24 * 16 * 257 * 66)
        );
        assert_eq!(
            steel(24, 17, 256, 496, 513, true, [16, 8]),
            Some(24 * 16 * 257 * 133)
        );
        assert_eq!(vector(u64::MAX, 256, 4097, 32), None);
    }
}
