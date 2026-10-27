use super::{bytes, tensor_bytes};
use crate::{Dispatch, KernelCost, Op, gather_matmul_flops, gather_quant_matmul_dispatch_flops};

/// Returns the index operand slot for a validated routed quantized operation.
#[must_use]
pub const fn quantized_route_index(input_count: usize) -> Option<usize> {
    match input_count {
        5 | 6 => Some(4),
        8 => Some(7),
        _ => None,
    }
}

fn per_expert_bytes(weights: u64, experts: u64) -> Option<u64> {
    (weights.checked_rem(experts)? == 0).then_some(weights.checked_div(experts)?)
}

pub(super) fn count(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    let inputs = dispatch.primitive_inputs();
    let (index, weights) = match dispatch.op() {
        Op::GatherMatmul => {
            cost.flops = gather_matmul_flops(
                inputs[0].layout().shape(),
                inputs[1].layout().shape(),
                inputs[2].layout().shape(),
            )?;
            (2, tensor_bytes(&inputs[1])?)
        }
        Op::GatherQuantMatmul { .. } => {
            let index = quantized_route_index(inputs.len())?;
            cost.flops = gather_quant_matmul_dispatch_flops(
                inputs[0].layout().shape(),
                inputs[1].layout().shape(),
                inputs[index].layout().shape(),
                inputs.len(),
            )?;
            (index, bytes(&inputs[1..index])?)
        }
        Op::Embed => {
            let width = u64::from(*inputs[0].layout().shape().last()?);
            cost.weight_bytes = inputs[1]
                .layout()
                .element_count()
                .checked_mul(width)?
                .checked_mul(inputs[0].layout().dtype().byte_size())?;
            cost.read_bytes = cost.weight_bytes.checked_add(tensor_bytes(&inputs[1])?)?;
            return Some(cost);
        }
        Op::QuantEmbed { .. } => {
            cost.weight_bytes = per_expert_bytes(
                bytes(&inputs[..3])?,
                u64::from(inputs[0].layout().shape()[0]),
            )?
            .checked_mul(inputs[3].layout().element_count())?;
            cost.read_bytes = cost.weight_bytes.checked_add(tensor_bytes(&inputs[3])?)?;
            return Some(cost);
        }
        _ => return Some(cost),
    };
    let experts = u64::from(inputs[1].layout().shape()[0]);
    let per_expert = per_expert_bytes(weights, experts)?;
    let routes = inputs[index].layout().element_count();
    cost.weight_bytes = per_expert.checked_mul(routes)?;
    cost.read_bytes = cost
        .read_bytes
        .checked_sub(weights)?
        .checked_add(cost.weight_bytes)?;
    let per_row = u64::from(*inputs[index].layout().shape().last()?);
    cost.minimum_weight_bytes = per_expert.checked_mul(per_row.min(experts))?;
    cost.maximum_weight_bytes = per_expert.checked_mul(experts.min(routes))?;
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommandList, DType, cost::tests::tensor};

    #[test]
    fn routed_index_slots_refuse_unknown_arities() {
        assert_eq!(quantized_route_index(5), Some(4));
        assert_eq!(quantized_route_index(6), Some(4));
        assert_eq!(quantized_route_index(8), Some(7));
        assert_eq!(quantized_route_index(7), None);
        assert_eq!(per_expert_bytes(7, 2), None);
        assert_eq!(per_expert_bytes(8, 0), None);
    }

    #[test]
    fn routed_quantized_cost_counts_assignments_and_reuse_bounds() {
        for rows in [1, 7, 33] {
            let input = tensor(1, DType::F32, &[rows, 1024]);
            let packed = tensor(2, DType::U32, &[16, 33, 128]);
            let scales = tensor(3, DType::BF16, &[16, 33, 16]);
            let biases = tensor(4, DType::BF16, &[16, 33, 16]);
            let indices = tensor(5, DType::U32, &[rows, 8]);
            let output = tensor(6, DType::F32, &[rows, 8, 33]);
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::GatherQuantMatmul {
                        bits: 4,
                        group_size: 64,
                    },
                    &[&input, &packed, &scales, &biases, &indices],
                    &output,
                )
                .unwrap();
            let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
            assert_eq!(cost.flops, 2 * u64::from(rows) * 8 * 1024 * 33);
            let expert_bytes = 33 * (128 * 4 + 16 * 4);
            assert_eq!(cost.weight_bytes, expert_bytes * u64::from(rows) * 8);
            assert_eq!(cost.minimum_weight_bytes, expert_bytes * 8);
            if rows == 1 {
                assert_eq!(cost.minimum_weight_bytes, cost.maximum_weight_bytes);
            }
            assert_eq!(
                cost.maximum_weight_bytes,
                expert_bytes * (u64::from(rows) * 8).min(16)
            );
        }
    }

    #[test]
    fn embeddings_read_only_selected_rows() {
        let table = tensor(1, DType::BF16, &[4097, 1024]);
        let ids = tensor(2, DType::U32, &[7]);
        let output = tensor(3, DType::BF16, &[7, 1024]);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Embed, &[&table, &ids], &output)
            .unwrap();
        assert_eq!(
            KernelCost::of(commands.last_dispatch().unwrap())
                .unwrap()
                .read_bytes,
            7 * (1024 * 2 + 4)
        );
    }
}
