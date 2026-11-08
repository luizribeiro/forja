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
        if !matches!(
            dispatch.op(),
            Op::Matmul | Op::QuantMatmul { .. } | Op::GatherQuantMatmul { .. }
        ) {
            return Some([cost.flops; 2]);
        }
        let input = dispatch.primitive_inputs();
        let output = dispatch.output().layout().shape();
        let inner = u64::from(*input.first()?.layout().shape().last()?);
        let columns = u64::from(*output.last()?);
        let input_shape = input.first()?.layout().shape();
        let rows = u64::from(*input_shape.get(input_shape.len().checked_sub(2)?)?);
        let (minimum_rows, maximum_rows, batch) = match dispatch.op() {
            Op::Matmul => (
                round(rows, tm)?,
                round(rows, tm)?,
                if output.len() == 3 {
                    u64::from(*output.first()?)
                } else {
                    1
                },
            ),
            Op::QuantMatmul { .. } => (round(rows, tm)?, round(rows, tm)?, 1),
            Op::GatherQuantMatmul { .. } => {
                let index = super::quantized_route_index(input.len())?;
                let routes = input.get(index)?.layout().element_count();
                let experts = u64::from(*input.get(1)?.layout().shape().first()?).min(routes);
                let maximum_tiles = routes
                    .checked_add(experts.checked_mul(tm.checked_sub(1)?)?)?
                    .checked_div(tm)?;
                (
                    round(routes, tm)?,
                    maximum_tiles.checked_mul(tm)?,
                    if input.len() == 8 { 2 } else { 1 },
                )
            }
            Op::State(_)
            | Op::Program(_)
            | Op::Copy
            | Op::PagedCopy { .. }
            | Op::RmsNorm { .. }
            | Op::Argmax { .. }
            | Op::TopK { .. }
            | Op::Sample { .. }
            | Op::SampleMasked { .. }
            | Op::Rope { .. }
            | Op::Embed
            | Op::QuantEmbed { .. }
            | Op::GatherMatmul
            | Op::Sdpa { .. }
            | Op::PagedSdpa { .. }
            | Op::PagedSdpaAppend { .. } => return Some([cost.flops; 2]),
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
    let [heads, queries, width]: [u32; 3] = input.first()?.layout().shape().try_into().ok()?;
    let (causal, start, keys) = match dispatch.op() {
        Op::Sdpa {
            causal, q_start, ..
        } => (causal, q_start, *input.get(1)?.layout().shape().get(1)?),
        Op::PagedSdpaAppend { q_start, .. } => (true, q_start, q_start.checked_add(1)?),
        Op::PagedSdpa {
            causal,
            q_start,
            key_length,
            ..
        } => (causal, q_start, key_length),
        Op::State(_)
        | Op::Program(_)
        | Op::Copy
        | Op::PagedCopy { .. }
        | Op::RmsNorm { .. }
        | Op::Argmax { .. }
        | Op::TopK { .. }
        | Op::Sample { .. }
        | Op::SampleMasked { .. }
        | Op::Rope { .. }
        | Op::Embed
        | Op::QuantEmbed { .. }
        | Op::Matmul
        | Op::GatherMatmul
        | Op::QuantMatmul { .. }
        | Op::GatherQuantMatmul { .. } => return None,
    };
    let tile = tiles
        .iter()
        .find(|(tile_width, _)| *tile_width == width)?
        .1
        .map(u64::from);
    if tile.contains(&0) {
        return None;
    }
    let queries = u64::from(queries);
    let trips = if causal {
        let full = queries / tile[0];
        let bias = u64::from(start)
            .checked_add(tile[1].checked_sub(1)?)?
            .checked_add(tile[0])?;
        let full_trips = floor_sum(full, tile[1], tile[0], bias)?;
        full_trips.checked_add(if queries.is_multiple_of(tile[0]) {
            0
        } else {
            u64::from(start).checked_add(queries)?.div_ceil(tile[1])
        })?
    } else {
        queries
            .div_ceil(tile[0])
            .checked_mul(u64::from(keys).div_ceil(tile[1]))?
    };
    let pairs = product(&[trips, tile[0], tile[1]])?;
    let width =
        u64::from(width).checked_add(u64::from(*dispatch.output().layout().shape().last()?))?;
    Some([product(&[2, u64::from(heads), pairs, width])?; 2])
}

fn triangular(value: u64) -> Option<u64> {
    let previous = value.checked_sub(1)?;
    if value.is_multiple_of(2) {
        (value / 2).checked_mul(previous)
    } else {
        value.checked_mul(previous / 2)
    }
}

fn floor_sum(mut count: u64, mut divisor: u64, mut slope: u64, mut bias: u64) -> Option<u64> {
    let mut sum = 0_u64;
    loop {
        if count == 0 {
            return Some(sum);
        }
        if slope >= divisor {
            sum = sum.checked_add(triangular(count)?.checked_mul(slope / divisor)?)?;
            slope %= divisor;
        }
        if bias >= divisor {
            sum = sum.checked_add(count.checked_mul(bias / divisor)?)?;
            bias %= divisor;
        }
        let top = slope.checked_mul(count)?.checked_add(bias)?;
        if top < divisor {
            return Some(sum);
        }
        count = top / divisor;
        bias = top % divisor;
        std::mem::swap(&mut divisor, &mut slope);
    }
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
    fn a_matrix_cost_model_preserves_one_dimensional_copy_cost() {
        let input = tensor(1, DType::F32, &[7]);
        let output = tensor(2, DType::F32, &[7]);
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        let cost = KernelCost::of(dispatch).unwrap();
        let model = CostModel {
            tile: [32, 64, 16],
            ..CostModel::default()
        };
        assert_eq!(model.padded_flops(dispatch, cost), Some([cost.flops; 2]));
    }
    #[test]
    fn floor_sum_matches_a_schedule_walk() {
        for count in [0_u64, 1, 7, 33, 4097] {
            for divisor in [1, 7, 16, 33] {
                for slope in [1, 7, 32, 96] {
                    for bias in [0, 1, 7, 33, 4097] {
                        let walked = (0..count)
                            .map(|index| (index * slope + bias) / divisor)
                            .sum();
                        assert_eq!(floor_sum(count, divisor, slope, bias), Some(walked));
                    }
                }
            }
        }
        assert_eq!(
            floor_sum(u64::from(u32::MAX), 1, 1, 0),
            triangular(u64::from(u32::MAX))
        );
        assert_eq!(floor_sum(u64::MAX, 1, 1, 0), None);
    }

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
    fn attention_padding_matches_query_and_key_tile_walks() {
        for queries in [1_u32, 7, 33, 4097] {
            for start in [0_u32, 7, 4097] {
                let keys = start + queries;
                let query = tensor(1, DType::F32, &[16, queries, 128]);
                let key = tensor(2, DType::F16, &[8, keys, 128]);
                let value = tensor(3, DType::F16, &[8, keys, 128]);
                let output = tensor(4, DType::F32, &[16, queries, 128]);
                for causal in [false, true] {
                    let mut commands = CommandList::new();
                    commands
                        .dispatch(
                            Op::Sdpa {
                                scale: 1.0,
                                causal,
                                q_start: start,
                            },
                            &[&query, &key, &value],
                            &output,
                        )
                        .unwrap();
                    let dispatch = commands.last_dispatch().unwrap();
                    let cost = KernelCost::of(dispatch).unwrap();
                    for tiles in [
                        &[(128, [16, 8])][..],
                        &[(128, [32, 32])],
                        &[(128, [96, 16])],
                    ] {
                        let model = CostModel {
                            attention_tiles: tiles,
                            ..CostModel::default()
                        };
                        let [q_tile, k_tile] = tiles[0].1;
                        let mut pairs = 0_u64;
                        for q_start in (0..queries).step_by(q_tile as usize) {
                            let limit = if causal {
                                start + (q_start + q_tile).min(queries)
                            } else {
                                keys
                            };
                            for _ in (0..limit).step_by(k_tile as usize) {
                                pairs += u64::from(q_tile) * u64::from(k_tile);
                            }
                        }
                        assert_eq!(
                            model.padded_flops(dispatch, cost),
                            Some([2 * 16 * pairs * 256; 2])
                        );
                    }
                }
            }
        }
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
