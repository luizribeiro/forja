use forja_core::{Dispatch, KernelCost, Op};

pub(super) fn stages(dispatch: &Dispatch) -> Option<[KernelCost; 2]> {
    let (Op::Sample {
        chunk_width,
        fast_top_k,
        ..
    }
    | Op::SampleMasked {
        chunk_width,
        fast_top_k,
        ..
    }) = dispatch.op()
    else {
        return None;
    };
    let logical = super::logical(dispatch)?;
    let width = *dispatch.inputs()[0].layout().shape().last()?;
    let rows = dispatch.output().layout().element_count();
    let chunks = rows.checked_mul(u64::from(width.div_ceil(chunk_width)))?;
    let partials = chunks
        .checked_mul(u64::from(fast_top_k).checked_add(3)?)?
        .checked_mul(8)?;
    Some([
        KernelCost {
            read_bytes: logical.read_bytes,
            write_bytes: partials,
            scratch_write_bytes: partials,
            scalar_ops: logical.scalar_ops,
            sfu_ops: logical.sfu_ops,
            ..KernelCost::default()
        },
        KernelCost {
            read_bytes: partials.checked_add(
                dispatch.inputs()[1]
                    .layout()
                    .element_count()
                    .checked_mul(4)?,
            )?,
            write_bytes: logical.write_bytes,
            scratch_read_bytes: partials,
            scalar_ops: chunks.checked_mul(3)?,
            ..KernelCost::default()
        },
    ])
}

pub(super) fn internal(dispatch: &Dispatch, kernel: &str) -> Option<KernelCost> {
    let costs = stages(dispatch)?;
    match kernel {
        "sample_masked_partials" => Some(costs[0]),
        "sample_masked_finalize" => Some(costs[1]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType};

    #[test]
    fn masked_sampling_stages_cover_odd_logits_and_partial_capacity() {
        let backend = forja_cpu::CpuBackend::new();
        for width in [1_u32, 33, 4097] {
            let logits = backend.alloc(DType::F32, &[7, width]).unwrap();
            let controls = backend.alloc(DType::U32, &[5]).unwrap();
            let mask = backend.alloc(DType::U32, &[width.div_ceil(32)]).unwrap();
            let output = backend.alloc(DType::U32, &[7]).unwrap();
            let mut commands = CommandList::new();
            commands
                .dispatch(
                    Op::SampleMasked {
                        position: 7,
                        chunk_width: 2048,
                        fast_top_k: 8,
                        fast_max_chunks: 32,
                        rejection_rounds: 4,
                    },
                    &[&logits, &controls, &mask],
                    &output,
                )
                .unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let [partials, finalize] = stages(dispatch).unwrap();
            let total = super::super::of(dispatch).unwrap();
            assert_eq!(partials.checked_add(finalize), Some(total));
            assert_eq!(
                partials.write_bytes,
                7 * u64::from(width.div_ceil(2048)) * 11 * 8
            );
            assert_eq!(partials.scratch_write_bytes, finalize.scratch_read_bytes);
            assert_eq!(
                total.sfu_ops,
                super::super::logical(dispatch).unwrap().sfu_ops
            );
            assert_eq!(finalize.write_bytes, 7 * 4);
            assert!(finalize.scalar_ops > 0);
        }
    }
}
