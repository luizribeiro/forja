use forja_core::{Dispatch, KernelCost, Op, Tensor, byte_ranges_overlap};

#[derive(Default)]
pub(crate) struct RouteFacts {
    unique_per_row: Vec<Tensor>,
}

impl RouteFacts {
    pub(crate) fn refine(&mut self, dispatch: &Dispatch, cost: &mut KernelCost) -> Option<()> {
        let input = dispatch.primitive_inputs();
        let index = match dispatch.op() {
            Op::GatherMatmul => Some(2),
            Op::GatherQuantMatmul { .. } => Some(if input.len() == 8 { 7 } else { 4 }),
            _ => None,
        };
        if let Some(indices) = index.and_then(|index| input.get(index)) {
            let routes = indices.layout().element_count();
            if cost.weight_bytes.checked_rem(routes)? != 0 {
                return None;
            }
            let per_expert = cost.weight_bytes.checked_div(routes)?;
            cost.minimum_weight_bytes = per_expert;
            if self.unique_per_row.iter().any(|tensor| {
                tensor.buffer() == indices.buffer() && tensor.layout() == indices.layout()
            }) {
                let per_row = u64::from(*indices.layout().shape().last()?);
                let minimum = per_expert.checked_mul(per_row)?;
                if minimum > cost.maximum_weight_bytes {
                    return None;
                }
                cost.minimum_weight_bytes = minimum;
            }
        }
        self.unique_per_row.retain(|tensor| {
            !dispatch.outputs().iter().any(|output| {
                tensor.buffer() == output.buffer()
                    && byte_ranges_overlap(tensor.layout(), output.layout())
            })
        });
        if matches!(dispatch.op(), Op::TopK { .. }) {
            self.unique_per_row.push(dispatch.outputs().get(1)?.clone());
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType};

    #[test]
    fn top_k_proves_distinct_decode_weights_until_overwritten() {
        let backend = forja_cpu::CpuBackend::new();
        let logits = backend.alloc(DType::F32, &[1, 16]).unwrap();
        let values = backend.alloc(DType::F32, &[1, 8]).unwrap();
        let indices = backend.alloc(DType::U32, &[1, 8]).unwrap();
        let replacement = backend.alloc(DType::U32, &[1, 8]).unwrap();
        let input = backend.alloc(DType::F32, &[1, 1024]).unwrap();
        let packed = backend.alloc(DType::U32, &[16, 33, 128]).unwrap();
        let scales = backend.alloc(DType::BF16, &[16, 33, 16]).unwrap();
        let biases = backend.alloc(DType::BF16, &[16, 33, 16]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 8, 33]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::TopK {
                    k: 8,
                    normalize: true,
                    chunk_width: 2048,
                },
                &[&logits],
                &[&values, &indices],
            )
            .unwrap();
        let op = Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        };
        commands
            .dispatch(op, &[&input, &packed, &scales, &biases, &indices], &output)
            .unwrap();
        commands
            .dispatch(Op::Copy, &[&replacement], &indices)
            .unwrap();
        commands
            .dispatch(op, &[&input, &packed, &scales, &biases, &indices], &output)
            .unwrap();
        let mut facts = RouteFacts::default();
        let costs = commands
            .dispatches()
            .iter()
            .map(|dispatch| {
                let mut cost = KernelCost::of(dispatch).unwrap();
                facts.refine(dispatch, &mut cost).unwrap();
                cost
            })
            .collect::<Vec<_>>();
        assert_eq!(costs[1].minimum_weight_bytes, 152_064);
        assert_eq!(costs[1].maximum_weight_bytes, 152_064);
        assert_eq!(costs[1].weight_bytes, 152_064);
        assert_eq!(costs[3].minimum_weight_bytes, 19_008);
        assert_eq!(costs[3].maximum_weight_bytes, 152_064);
    }
}
