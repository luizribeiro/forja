use super::product;
use crate::{Dispatch, KernelCost, Op};

/// Algorithm tile geometry used to report work beyond logical arithmetic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CostModel {
    /// Matrix tile rows, columns and reduction width; zero means untiled.
    pub tile: [u32; 3],
}

impl CostModel {
    /// Reports the formula and tile geometry through the catalog.
    #[must_use]
    pub fn json(self) -> serde_json::Value {
        serde_json::json!({"logical": "validated-operation-v1", "tile": self.tile,
            "fma_flops": 2, "scalar_ops": "one per scalar IR operation; special functions separate from MMA",
            "traffic": "logical operands; routed weights report distinct-footprint bounds"})
    }

    /// Returns lower and upper padded FLOP counts, including routed occupancy uncertainty.
    #[must_use]
    pub fn padded_flops(self, dispatch: &Dispatch, cost: KernelCost) -> Option<[u64; 2]> {
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
            minimum_rows.checked_mul(factor)?,
            maximum_rows.checked_mul(factor)?,
        ])
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
            CostModel { tile: [32, 64, 16] }.padded_flops(dispatch, cost),
            Some([2 * 32 * 4160 * 48; 2])
        );
        assert_eq!(round(u64::MAX, 32), None);
    }
}
