use super::product;
use crate::{Dispatch, KernelCost, Op};

/// Algorithm tile geometry used to report work beyond logical arithmetic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CostModel {
    /// Matrix tile rows, columns and reduction width; zero means untiled.
    pub tile: [u32; 3],
    /// Whether runtime occupancy can select an untiled fallback.
    pub conditional_tiles: bool,
    /// Attention head widths with their query/key tile extents.
    pub attention_tiles: &'static [(u32, [u32; 2])],
}

impl CostModel {
    /// Reports the formula and tile geometry through the catalog.
    #[must_use]
    pub fn json(self) -> serde_json::Value {
        serde_json::json!({"logical": "validated-operation-v1", "tile": self.tile, "conditional_tiles": self.conditional_tiles, "attention_tiles": self.attention_tiles,
            "fma_flops": 2, "scalar_ops": "one per scalar IR operation; special functions separate from MMA",
            "traffic": "logical operands; routed weights report distinct-footprint bounds"})
    }

    /// Returns lower and upper padded FLOP counts, including routed occupancy uncertainty.
    #[must_use]
    pub fn padded_flops(self, dispatch: &Dispatch, cost: KernelCost) -> Option<[u64; 2]> {
        if !self.attention_tiles.is_empty() {
            return attention_padded(self.attention_tiles, dispatch);
        }
        let [tm, tn, tk] = self.tile.map(u64::from);
        if tm == 0 {
            return Some([cost.flops; 2]);
        }
        let input = dispatch.primitive_inputs();
        let output = dispatch.output().layout().shape();
        let inner = u64::from(*input[0].layout().shape().last()?);
        let columns = u64::from(*output.last()?);
        let rows = u64::from(input[0].layout().shape()[input[0].layout().shape().len() - 2]);
        let (minimum_rows, maximum_rows, batch) = match dispatch.op() {
            Op::Matmul => (
                round(rows, tm)?,
                round(rows, tm)?,
                if output.len() == 3 {
                    u64::from(output[0])
                } else {
                    1
                },
            ),
            Op::QuantMatmul { .. } => (round(rows, tm)?, round(rows, tm)?, 1),
            Op::GatherQuantMatmul { .. } => {
                let index = if input.len() == 8 { 7 } else { 4 };
                let routes = input[index].layout().element_count();
                let experts = u64::from(input[1].layout().shape()[0]).min(routes);
                let maximum_tiles = routes
                    .checked_add(experts.checked_mul(tm.checked_sub(1)?)?)?
                    .checked_div(tm)?;
                (
                    round(routes, tm)?,
                    maximum_tiles.checked_mul(tm)?,
                    if input.len() == 8 { 2 } else { 1 },
                )
            }
            _ => return Some([cost.flops; 2]),
        };
        let factor = product(&[2, batch, round(columns, tn)?, round(inner, tk)?])?;
        Some([
            if self.conditional_tiles {
                cost.flops
            } else {
                minimum_rows.checked_mul(factor)?
            },
            maximum_rows.checked_mul(factor)?,
        ])
    }
}

fn attention_padded(tiles: &[(u32, [u32; 2])], dispatch: &Dispatch) -> Option<[u64; 2]> {
    let input = dispatch.primitive_inputs();
    let query = input[0].layout().shape();
    let (causal, start, keys) = match dispatch.op() {
        Op::Sdpa {
            causal, q_start, ..
        } => (causal, q_start, input[1].layout().shape()[1]),
        Op::PagedSdpa {
            causal,
            q_start,
            key_length,
            ..
        } => (causal, q_start, key_length),
        _ => return None,
    };
    let tile = tiles
        .iter()
        .find(|(width, _)| *width == query[2])?
        .1
        .map(u64::from);
    if tile.contains(&0) {
        return None;
    }
    let mut first = 0_u64;
    let mut pairs = 0_u64;
    while first < u64::from(query[1]) {
        let end = first.checked_add(tile[0])?.min(u64::from(query[1]));
        let limit = if causal {
            u64::from(start).checked_add(end)?.min(u64::from(keys))
        } else {
            u64::from(keys)
        };
        pairs = pairs.checked_add(tile[0].checked_mul(round(limit, tile[1])?)?)?;
        first = end;
    }
    let width =
        u64::from(query[2]).checked_add(u64::from(*dispatch.output().layout().shape().last()?))?;
    Some([product(&[2, u64::from(query[0]), pairs, width])?; 2])
}

fn round(value: u64, tile: u64) -> Option<u64> {
    value
        .checked_add(tile.checked_sub(1)?)?
        .checked_div(tile)?
        .checked_mul(tile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommandList, DType, cost::tests::tensor};

    #[test]
    fn matrix_padding_is_separate_from_logical_work() {
        let input = tensor(1, DType::F32, &[7, 33]);
        let weights = tensor(2, DType::F32, &[33, 4097]);
        let output = tensor(3, DType::F32, &[7, 4097]);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&input, &weights], &output)
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        let cost = KernelCost::of(dispatch).unwrap();
        assert_eq!(cost.flops, 2 * 7 * 33 * 4097);
        assert_eq!(
            CostModel {
                tile: [32, 64, 16],
                ..CostModel::default()
            }
            .padded_flops(dispatch, cost),
            Some([2 * 32 * 4160 * 48; 2])
        );
        let conditional = CostModel {
            tile: [32, 64, 16],
            conditional_tiles: true,
            ..CostModel::default()
        };
        assert_eq!(
            conditional.padded_flops(dispatch, cost).unwrap()[0],
            cost.flops
        );
        assert_eq!(round(u64::MAX, 32), None);
    }
    #[test]
    fn attention_padding_follows_causal_key_tiles() {
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
        let dispatch = commands.last_dispatch().unwrap();
        let model = CostModel {
            attention_tiles: &[(128, [96, 16])],
            ..CostModel::default()
        };
        assert_eq!(
            model.padded_flops(dispatch, KernelCost::of(dispatch).unwrap()),
            Some([2 * 16 * 96 * 48 * 256; 2])
        );
    }
}
