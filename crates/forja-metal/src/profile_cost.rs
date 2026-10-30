use forja_core::{Dispatch, KernelCost, Op};
pub(crate) mod state_block;

fn product(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(1_u64, |product, value| product.checked_mul(*value))
}

pub(crate) fn internal(
    dispatch: &Dispatch,
    kernel: &str,
    blocks: Option<u32>,
) -> Option<KernelCost> {
    if matches!(
        dispatch.op(),
        Op::State(forja_core::StateOp::DeltaRuleBlock(_))
    ) {
        return block_internal(dispatch, kernel);
    }
    let mut cost = KernelCost::of(dispatch)?;
    let input = dispatch.primitive_inputs();
    if let Some(blocks) = blocks {
        let query = input[0].layout().shape();
        let width = u64::from(*dispatch.output().layout().shape().last()?);
        let partials = product(&[
            u64::from(query[0]),
            u64::from(blocks),
            width.checked_add(2)?,
            4,
        ])?;
        if matches!(kernel, "paged_sdpa_2pass_2_128" | "paged_sdpa_2pass_2_256") {
            return Some(KernelCost {
                read_bytes: partials,
                write_bytes: cost.write_bytes,
                scalar_ops: product(&[u64::from(query[0]), u64::from(blocks), width, 3])?,
                ..KernelCost::default()
            });
        }
        cost.write_bytes = partials;
        return Some(cost);
    }
    if !matches!(dispatch.op(), Op::GatherQuantMatmul { .. }) {
        return Some(cost);
    }
    let indices = &input[forja_core::quantized_route_index(input.len())?];
    let routes = indices.layout().element_count();
    let experts = u64::from(input[1].layout().shape()[0]);
    let index_bytes = routes.checked_mul(4)?;
    match kernel {
        "count_expert_routes" => Some(KernelCost {
            read_bytes: index_bytes.checked_mul(2)?,
            write_bytes: experts
                .checked_add(1)?
                .checked_mul(4)?
                .checked_add(index_bytes)?,
            ..KernelCost::default()
        }),
        "plan_expert_routes" => Some(KernelCost {
            read_bytes: experts.checked_mul(4)?,
            write_bytes: experts.checked_add(1)?.checked_mul(4)?.checked_add(
                routes
                    .div_ceil(u64::from(crate::encoding::grouped_block_rows(dispatch)))
                    .checked_add(experts)?
                    .checked_mul(12)?,
            )?,
            ..KernelCost::default()
        }),
        "arrange_expert_routes" => Some(KernelCost {
            read_bytes: product(&[index_bytes, experts, 2])?
                .checked_add(experts.checked_mul(4)?)?,
            write_bytes: index_bytes,
            scalar_ops: product(&[routes, experts, 2])?,
            ..KernelCost::default()
        }),
        "gathered_pair_from_projections" => Some(KernelCost {
            read_bytes: dispatch
                .output()
                .layout()
                .element_count()
                .checked_mul(8)?
                .checked_add(index_bytes)?,
            write_bytes: cost.write_bytes,
            scalar_ops: cost.scalar_ops.checked_sub(cost.flops)?,
            ..KernelCost::default()
        }),
        "grouped_quantized_gemm_gate" | "grouped_quantized_gemm_up" => {
            cost.flops = cost.flops.checked_div(2)?;
            cost.weight_bytes = cost.weight_bytes.checked_div(2)?;
            cost.minimum_weight_bytes = cost.minimum_weight_bytes.checked_div(2)?;
            cost.maximum_weight_bytes = cost.maximum_weight_bytes.checked_div(2)?;
            cost.read_bytes = input[0]
                .layout()
                .element_count()
                .checked_mul(input[0].layout().dtype().byte_size())?
                .checked_add(cost.weight_bytes)?
                .checked_add(index_bytes)?;
            cost.write_bytes = dispatch.output().layout().element_count().checked_mul(4)?;
            cost.scalar_ops = cost.flops;
            Some(cost)
        }
        _ => Some(cost),
    }
}

fn block_internal(dispatch: &Dispatch, kernel: &str) -> Option<KernelCost> {
    let stages = crate::encoding::delta_block_mma_panel(dispatch)
        .and_then(|panel| state_block::stages(dispatch, panel))
        .or_else(|| KernelCost::delta_block_stages(dispatch))?;
    let kernel = kernel
        .strip_suffix("_mma")
        .or_else(|| kernel.strip_suffix("_panel"))
        .unwrap_or(kernel);
    let index = [
        "state_block_gram",
        "state_block_solve",
        "state_block_carry",
        "state_block_output",
    ]
    .iter()
    .position(|name| *name == kernel)?;
    Some(stages[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType};
    #[test]
    fn routing_subkernels_have_independent_costs() {
        let backend = forja_cpu::CpuBackend::new();
        let shapes = [
            (DType::F32, vec![7, 1024]),
            (DType::U32, vec![16, 33, 128]),
            (DType::BF16, vec![16, 33, 16]),
            (DType::BF16, vec![16, 33, 16]),
            (DType::U32, vec![7, 8]),
        ];
        let inputs = shapes
            .iter()
            .map(|(dtype, shape)| backend.alloc(*dtype, shape).unwrap())
            .collect::<Vec<_>>();
        let output = backend.alloc(DType::F32, &[7, 8, 33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &inputs.iter().collect::<Vec<_>>(),
                &output,
            )
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        assert_eq!(
            internal(dispatch, "count_expert_routes", None)
                .unwrap()
                .read_bytes,
            7 * 8 * 4 * 2
        );
        assert_eq!(
            internal(dispatch, "arrange_expert_routes", None)
                .unwrap()
                .read_bytes,
            7 * 8 * 4 * 16 * 2 + 16 * 4
        );
        assert_eq!(
            internal(dispatch, "plan_expert_routes", None)
                .unwrap()
                .write_bytes,
            17 * 4 + (4 + 16) * 12
        );
    }
}
