use forja_core::{Dispatch, KernelCost, Op};

pub(crate) fn of(dispatch: &Dispatch) -> Option<KernelCost> {
    let mut cost = super::of(dispatch)?;
    let model = crate::variants::selected_cost_model(dispatch);
    let [rows, columns, _] = model.tile.map(u64::from);
    if rows == 0 {
        cost.scalar_ops = cost.scalar_ops.checked_add(cost.flops)?;
        cost.flops = 0;
        return Some(cost);
    }
    let flops = model.padded_flops(dispatch, cost)?[1];
    let input_values = flops.checked_div(2)?.checked_div(columns)?;
    let weight_values = flops.checked_div(2)?.checked_div(rows)?;
    let input = dispatch.primitive_inputs();
    let input_bytes = input_values.checked_mul(input[0].layout().dtype().byte_size())?;
    let weight_bytes = match dispatch.op() {
        Op::QuantMatmul { bits, .. } | Op::GatherQuantMatmul { bits, .. } => {
            let words = weight_values.div_ceil(32 / u64::from(bits));
            let parameters = words
                .checked_mul(2)?
                .checked_mul(input[2].layout().dtype().byte_size())?;
            cost.scalar_ops = cost
                .scalar_ops
                .checked_sub(cost.flops)?
                .checked_add(weight_values.checked_mul(8)?)?;
            words.checked_mul(4)?.checked_add(parameters)?
        }
        Op::Matmul => weight_values.checked_mul(input[1].layout().dtype().byte_size())?,
        _ => return None,
    };
    cost.read_bytes = cost
        .read_bytes
        .checked_add(input_bytes)?
        .checked_add(weight_bytes)?;
    cost.flops = flops;
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{BackendDispatchData, CommandList, DType, DeviceCapability};
    use forja_cpu::CpuBackend;

    #[test]
    fn quantized_schedule_counts_decode_calls_per_tile_and_trip() {
        let backend = CpuBackend::new();
        for rows in [16, 33, 65] {
            for columns in [1, 7, 33] {
                let input = backend.alloc(DType::F32, &[rows, 1024]).unwrap();
                let packed = backend.alloc(DType::U32, &[columns, 128]).unwrap();
                let scales = backend.alloc(DType::BF16, &[columns, 16]).unwrap();
                let biases = backend.alloc(DType::BF16, &[columns, 16]).unwrap();
                let output = backend.alloc(DType::F32, &[rows, columns]).unwrap();
                let mut commands = CommandList::new();
                commands
                    .dispatch(
                        Op::QuantMatmul {
                            bits: 4,
                            group_size: 64,
                        },
                        &[&input, &packed, &scales, &biases],
                        &output,
                    )
                    .unwrap();
                let selected = crate::variants::resolve(
                    commands.last_dispatch().unwrap(),
                    "quant-matmul.tiled",
                    &[DeviceCapability::Metal4],
                )
                .unwrap();
                commands.set_last_backend_data(BackendDispatchData::new(selected));
                let dispatch = commands.last_dispatch().unwrap();
                let mut decoded = 0_u64;
                let mut input_loads = 0_u64;
                for _ in (0..rows).step_by(32) {
                    for _ in (0..columns).step_by(32) {
                        for _ in (0..1024).step_by(32) {
                            for _ in 0..32 {
                                for _ in 0..32 {
                                    decoded += 1;
                                    input_loads += 1;
                                }
                            }
                        }
                    }
                }
                let logical = super::super::of(dispatch).unwrap();
                let scheduled = of(dispatch).unwrap();
                assert_eq!(scheduled.flops, decoded * 32 * 2);
                assert_eq!(scheduled.scalar_ops, decoded * 8);
                assert_eq!(
                    scheduled.read_bytes,
                    logical.read_bytes + input_loads * 4 + decoded
                );
            }
        }
    }
}
