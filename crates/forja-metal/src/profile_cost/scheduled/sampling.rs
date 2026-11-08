use forja_core::{Dispatch, KernelCost, Op};

fn scan_passes(parallel: bool, rounds: u32) -> Option<u64> {
    if parallel {
        // Top-k and probability boundaries each walk all 64 key bits.
        Some(64 + 64 + 4)
    } else {
        // Proposal/threshold scans per rejection round, radix/tie scans and final draws.
        u64::from(rounds).checked_mul(2)?.checked_add(12)
    }
}

pub(super) fn of(dispatch: &Dispatch, mut cost: KernelCost) -> Option<KernelCost> {
    let (Op::Sample {
        chunk_width,
        fast_top_k,
        rejection_rounds,
        ..
    }
    | Op::SampleMasked {
        chunk_width,
        fast_top_k,
        rejection_rounds,
        ..
    }) = dispatch.op()
    else {
        return None;
    };
    let input = &dispatch.inputs()[0];
    let width = u64::from(*input.layout().shape().last()?);
    let rows = dispatch.output().layout().element_count();
    let chunks = width.div_ceil(u64::from(chunk_width));
    let parallel = crate::encoding::parallel_sampling(dispatch);
    let reads = width.checked_mul(scan_passes(parallel, rejection_rounds)?)?;
    let sfus = width.checked_mul(if parallel {
        80
    } else {
        u64::from(rejection_rounds)
            .checked_mul(8)?
            .checked_add(12)?
    })?;
    // The larger fallback network also bounds the smaller ranked-merge paths.
    let network = 512_u64.checked_mul(55)?;
    let local = chunks.checked_add(1)?.checked_mul(288)?.checked_mul(19)?;
    let ranked = chunks
        .checked_add(1)?
        .checked_mul(u64::from(fast_top_k))?
        .checked_mul(512)?;
    let comparisons = network.checked_add(local)?.checked_add(ranked)?;
    let partials = chunks.checked_mul(u64::from(fast_top_k).checked_add(3)?)?;
    let groups = if parallel {
        chunks.checked_add(1)?
    } else {
        u64::from(rejection_rounds)
            .checked_mul(chunks.checked_add(1)?.checked_mul(2)?)?
            .checked_add(chunks.checked_mul(2)?)?
            .checked_add(7)?
    };
    let config_values = groups.checked_mul(256)?.checked_mul(5)?;
    let values = reads
        .checked_add(partials.checked_mul(4)?)?
        .checked_add(8192)?
        .checked_add(config_values.checked_mul(2)?)?;
    let scalar = values
        .checked_mul(32)?
        .checked_add(comparisons.checked_mul(8)?)?;
    cost.scalar_ops = cost.scalar_ops.max(scalar.checked_mul(rows)?);
    cost.sfu_ops = cost.sfu_ops.max(sfus.checked_add(8192)?.checked_mul(rows)?);
    cost.scalar_ops = cost.scalar_ops.checked_add(cost.sfu_ops)?;
    cost.read_bytes = cost.read_bytes.checked_add(
        values
            .checked_mul(rows)?
            .checked_mul(input.layout().dtype().byte_size())?,
    )?;
    let scratch = partials.checked_mul(rows)?.checked_mul(8)?;
    cost.write_bytes = cost.write_bytes.checked_add(scratch)?;
    cost.scratch_write_bytes = cost.scratch_write_bytes.checked_add(scratch)?;
    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType};

    #[test]
    fn sampler_scan_counts_bound_an_independent_stage_walk() {
        for width in [1_u32, 7, 33, 4097, 151_936] {
            for rounds in [0_u32, 2, 7] {
                for parallel in [false, true] {
                    let mut visits = 0_u64;
                    let mut scan = || {
                        for _ in 0..width {
                            visits += 1;
                        }
                    };
                    if parallel {
                        for _ in 0..64 {
                            scan();
                        }
                        for _ in 0..64 {
                            scan();
                        }
                        for _ in 0..4 {
                            scan();
                        }
                    } else {
                        for _ in 0..rounds {
                            scan();
                            scan();
                        }
                        for _ in 0..3 {
                            scan();
                        }
                        for _ in 0..9 {
                            scan();
                        }
                    }
                    assert_eq!(
                        visits,
                        u64::from(width) * scan_passes(parallel, rounds).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn sampling_time_work_uses_source_passes_and_retains_the_logical_floor() {
        let backend = forja_cpu::CpuBackend::new();
        for masked in [false, true] {
            let logits = backend.alloc(DType::F32, &[1, 151_936]).unwrap();
            let config = backend.alloc(DType::U32, &[5]).unwrap();
            let mask = backend.alloc(DType::U32, &[151_936 / 32]).unwrap();
            let output = backend.alloc(DType::U32, &[1]).unwrap();
            let op = if masked {
                Op::SampleMasked {
                    position: 7,
                    chunk_width: 2048,
                    fast_top_k: 20,
                    fast_max_chunks: 102,
                    rejection_rounds: 2,
                }
            } else {
                Op::Sample {
                    position: 7,
                    chunk_width: 2048,
                    fast_top_k: 20,
                    fast_max_chunks: 102,
                    rejection_rounds: 2,
                }
            };
            let inputs = if masked {
                vec![&logits, &config, &mask]
            } else {
                vec![&logits, &config]
            };
            let mut commands = CommandList::new();
            commands.dispatch(op, &inputs, &output).unwrap();
            let dispatch = commands.last_dispatch().unwrap();
            let scheduled = super::super::of(dispatch).unwrap();
            assert!(dispatch.work().unwrap() >= 151_936 * 1024);
            assert!(scheduled.sfu_ops < dispatch.work().unwrap());
            assert!(scheduled.scalar_ops >= scheduled.sfu_ops);
            assert!(scheduled.scratch_write_bytes > 0);
        }
    }
}
