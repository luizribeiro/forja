use crate::encoding::AttentionSchedule;
use forja_core::{Dispatch, KernelCost, Op};

struct Shape {
    heads: u64,
    queries: u64,
    keys: u64,
    width: u64,
    values: u64,
    query_bytes: u64,
    kv_bytes: u64,
    output_bytes: u64,
    start: u64,
    causal: bool,
    paged: bool,
}
impl Shape {
    fn of(d: &Dispatch) -> Option<Self> {
        let input = d.primitive_inputs();
        let query = input.first()?.layout();
        let [heads, queries, width]: [u32; 3] = query.shape().try_into().ok()?;
        let (keys, key, value, start, causal, paged) = match d.op() {
            Op::Sdpa {
                q_start, causal, ..
            } => (
                u64::from(*input.get(1)?.layout().shape().get(1)?),
                input.get(1)?,
                input.get(2)?,
                q_start,
                causal,
                false,
            ),
            Op::PagedSdpa {
                key_length,
                q_start,
                causal,
                ..
            } => {
                let [_, key, value, ..] = d.paged_attention_inputs()?;
                (u64::from(key_length), key, value, q_start, causal, true)
            }
            Op::PagedSdpaAppend { q_start, .. } => {
                let [_, key, value, ..] = d.paged_attention_inputs()?;
                (
                    u64::from(q_start).checked_add(1)?,
                    key,
                    value,
                    q_start,
                    true,
                    true,
                )
            }
            _ => return None,
        };
        let values = u64::from(*value.layout().shape().last()?);
        Some(Self {
            heads: u64::from(heads),
            queries: u64::from(queries),
            keys,
            width: u64::from(width),
            values,
            query_bytes: query.dtype().byte_size(),
            kv_bytes: u64::from(width)
                .checked_mul(key.layout().dtype().byte_size())?
                .checked_add(values.checked_mul(value.layout().dtype().byte_size())?)?,
            output_bytes: d.output().layout().dtype().byte_size(),
            start: u64::from(start),
            causal,
            paged,
        })
    }
}

// The key limit advances by an integral number of key tiles for every full query tile.
fn visits(s: &Shape, bq: u64, bk: u64) -> Option<(u64, u64)> {
    if bq == 0 || bk == 0 {
        return None;
    }
    let maximum = s.keys.div_ceil(bk);
    let tiles = s.queries.div_ceil(bq);
    if !s.causal || !bq.is_multiple_of(bk) {
        return Some((tiles.checked_mul(maximum)?, maximum));
    }
    let full = s.queries / bq;
    let first = s.start.checked_add(bq)?.div_ceil(bk).min(maximum);
    let step = bq / bk;
    let linear = full.min(
        maximum
            .checked_sub(first)?
            .checked_div(step)?
            .checked_add(1)?,
    );
    let series = linear.checked_mul(first)?.checked_add(
        linear
            .checked_mul(linear.saturating_sub(1))?
            .checked_div(2)?
            .checked_mul(step)?,
    )?;
    let total = series.checked_add(full.checked_sub(linear)?.checked_mul(maximum)?)?;
    let last = s.start.checked_add(s.queries)?.min(s.keys).div_ceil(bk);
    Some((
        total.checked_add(if s.queries.is_multiple_of(bq) {
            0
        } else {
            last
        })?,
        last,
    ))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn costs(d: &Dispatch, schedule: AttentionSchedule) -> Option<(KernelCost, KernelCost)> {
    let s = Shape::of(d)?;
    let mut total = KernelCost::default();
    let mut critical;
    let output = s
        .heads
        .checked_mul(s.queries)?
        .checked_mul(s.values)?
        .checked_mul(s.output_bytes)?;
    match schedule {
        AttentionSchedule::Tiled { query, key } => {
            let (bq, bk) = (u64::from(query), u64::from(key));
            let (all, longest) = visits(&s, bq, bk)?;
            let tile_cost = |count: u64, rows: u64| -> Option<KernelCost> {
                let pairs = count.checked_mul(bq)?.checked_mul(bk)?;
                let kv = count.checked_mul(bk)?.checked_mul(s.kv_bytes)?;
                let loads = count
                    .checked_mul(bk)?
                    .checked_mul(s.width.checked_add(s.values)?)?
                    .checked_add(rows.checked_mul(s.width)?)?;
                let addresses = loads
                    .div_ceil(4)
                    .checked_mul(32)?
                    .checked_add(loads.checked_mul(4)?)?;
                Some(KernelCost {
                    read_bytes: kv
                        .checked_add(rows.checked_mul(s.width)?.checked_mul(s.query_bytes)?)?,
                    kv_bytes: kv,
                    flops: pairs
                        .checked_mul(2)?
                        .checked_mul(s.width.checked_add(s.values)?)?,
                    scalar_ops: addresses.checked_add(pairs.checked_mul(8)?)?.checked_add(
                        count
                            .checked_mul(bq)?
                            .checked_mul(s.values.checked_add(32)?)?
                            .checked_mul(3)?,
                    )?,
                    sfu_ops: pairs.checked_add(count.checked_mul(bq)?.checked_mul(32)?)?,
                    ..KernelCost::default()
                })
            };
            total = tile_cost(
                all.checked_mul(s.heads)?,
                s.heads
                    .checked_mul(s.queries.div_ceil(bq))?
                    .checked_mul(bq)?,
            )?;
            critical = tile_cost(longest, bq)?;
            critical.write_bytes = bq.checked_mul(s.values)?.checked_mul(s.output_bytes)?;
        }
        AttentionSchedule::Vector {
            blocks,
            heads_per_group,
            reuse,
        } => {
            let blocks = u64::from(blocks).max(1);
            let reuse = u64::from(reuse).max(1);
            let vector_cost = |pairs: u64, rows: u64, partials: u64| -> Option<KernelCost> {
                let kv = pairs.checked_mul(s.kv_bytes)?.div_ceil(reuse);
                let scratch =
                    partials.checked_mul(s.values.checked_mul(s.query_bytes)?.checked_add(8)?)?;
                Some(KernelCost {
                    read_bytes: kv
                        .checked_add(rows.checked_mul(s.width)?.checked_mul(s.query_bytes)?)?
                        .checked_add(scratch)?,
                    kv_bytes: kv,
                    write_bytes: scratch,
                    scratch_read_bytes: scratch,
                    scratch_write_bytes: scratch,
                    flops: pairs.checked_mul(
                        s.width
                            .checked_mul(2)?
                            .checked_add(s.values.checked_mul(3)?)?,
                    )?,
                    scalar_ops: pairs.checked_mul(32 * 8)?.checked_add(
                        partials
                            .checked_mul(s.values.checked_add(32)?)?
                            .checked_mul(4)?,
                    )?,
                    // Each SIMD lane executes both online-softmax exponentials.
                    sfu_ops: pairs
                        .checked_mul(32 * 2)?
                        .checked_add(partials.checked_mul(32 * 2)?)?,
                    ..KernelCost::default()
                })
            };
            let rows = s.heads.checked_mul(s.queries)?;
            total = vector_cost(
                rows.checked_mul(s.keys)?,
                rows.checked_mul(blocks)?,
                if blocks > 1 {
                    rows.checked_mul(blocks)?
                } else {
                    0
                },
            )?;
            let group_rows = u64::from(heads_per_group);
            critical = vector_cost(
                s.keys.div_ceil(blocks).checked_mul(group_rows)?,
                group_rows,
                blocks.checked_mul(group_rows)?,
            )?;
            critical.write_bytes = critical.write_bytes.checked_add(
                group_rows
                    .checked_mul(s.values)?
                    .checked_mul(s.output_bytes)?,
            )?;
        }
        AttentionSchedule::Decomposed => {
            let pairs = s.heads.checked_mul(s.queries)?.checked_mul(s.keys)?;
            let scores = pairs.checked_mul(4)?;
            total.flops = pairs
                .checked_mul(2)?
                .checked_mul(s.width.checked_add(s.values)?)?;
            total.sfu_ops = pairs;
            total.scalar_ops = pairs.checked_mul(8)?;
            total.scratch_read_bytes = scores.checked_mul(4)?;
            total.scratch_write_bytes = scores.checked_mul(3)?;
            total.read_bytes = pairs
                .checked_mul(s.kv_bytes)?
                .checked_add(total.scratch_read_bytes)?
                .checked_add(
                    s.heads
                        .checked_mul(s.queries)?
                        .checked_mul(s.width)?
                        .checked_mul(s.query_bytes)?,
                )?;
            total.write_bytes = total.scratch_write_bytes;
            if s.paged {
                let copy = super::super::logical(d)?.kv_bytes;
                total.read_bytes = total.read_bytes.checked_add(copy.checked_mul(2)?)?;
                total.write_bytes = total.write_bytes.checked_add(copy)?;
            }
            let rows = s.heads.checked_mul(s.queries)?.max(1);
            critical = KernelCost {
                read_bytes: total.read_bytes.div_ceil(rows),
                write_bytes: total.write_bytes.div_ceil(rows),
                scalar_ops: total.scalar_ops.checked_add(total.flops)?.div_ceil(rows),
                sfu_ops: s.keys,
                ..KernelCost::default()
            };
        }
    }
    total.write_bytes = total.write_bytes.checked_add(output)?;
    if s.paged {
        // Authentication is visited by every selected K/V tile or vector key visit.
        let metadata = total.kv_bytes.checked_div(s.kv_bytes)?.checked_mul(16)?;
        total.metadata_read_bytes = metadata;
        total.read_bytes = total.read_bytes.checked_add(metadata)?;
        let metadata = critical.kv_bytes.checked_div(s.kv_bytes)?.checked_mul(16)?;
        critical.metadata_read_bytes = metadata;
        critical.read_bytes = critical.read_bytes.checked_add(metadata)?;
    }
    if matches!(schedule, AttentionSchedule::Tiled { .. }) {
        for cost in [&mut total, &mut critical] {
            let metadata = cost.metadata_read_bytes.div_ceil(4).checked_mul(16)?;
            let stores = cost.write_bytes.div_ceil(s.output_bytes).checked_mul(16)?;
            cost.scalar_ops = cost.scalar_ops.checked_add(metadata)?.checked_add(stores)?;
        }
        Some((total, critical))
    } else {
        Some((
            super::addressing(d, total)?,
            super::addressing(d, critical)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{
        AllocationRegistry, BackendDispatchData, CommandList, DType, DeviceCapability, Layout,
        Tensor,
    };
    fn tensor(_id: u64, dtype: DType, shape: &[u32]) -> Tensor {
        let size = shape.iter().map(|&x| u64::from(x)).product::<u64>() * dtype.byte_size();
        let mut registry = AllocationRegistry::new();
        let buffer = registry.insert((), size).unwrap();
        registry
            .tensor(
                buffer,
                Layout::contiguous(dtype, 0, shape.to_vec(), size).unwrap(),
            )
            .unwrap()
    }
    fn dispatch(heads: u32, kv_heads: u32, queries: u32, keys: u32, variant: &str) -> Dispatch {
        dispatch_typed(heads, kv_heads, queries, keys, variant, DType::F16)
    }
    fn dispatch_typed(
        heads: u32,
        kv_heads: u32,
        queries: u32,
        keys: u32,
        variant: &str,
        dtype: DType,
    ) -> Dispatch {
        let input = [
            tensor(1, DType::F32, &[heads, queries, 128]),
            tensor(2, dtype, &[kv_heads, keys, 128]),
            tensor(3, dtype, &[kv_heads, keys, 128]),
        ];
        let output = tensor(4, DType::F32, &[heads, queries, 128]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 1.0,
                    causal: true,
                    q_start: keys - queries,
                },
                &input.iter().collect::<Vec<_>>(),
                &output,
            )
            .unwrap();
        let pick = crate::variants::resolve(
            commands.last_dispatch().unwrap(),
            variant,
            &[DeviceCapability::Metal4],
        )
        .unwrap();
        commands.set_last_backend_data(BackendDispatchData::new(pick));
        commands.into_dispatches().remove(0)
    }
    fn paged(heads: u32, kv_heads: u32, queries: u32, keys: u32, variant: &str) -> Dispatch {
        let pages = keys.div_ceil(128);
        let input = [
            tensor(1, DType::F32, &[heads, queries, 128]),
            tensor(2, DType::F16, &[pages, 128, kv_heads, 128]),
            tensor(3, DType::F16, &[pages, 128, kv_heads, 128]),
            tensor(4, DType::U32, &[pages, 2]),
            tensor(5, DType::U32, &[pages]),
        ];
        let output = tensor(6, DType::F32, &[heads, queries, 128]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::PagedSdpa {
                    scale: 1.0,
                    causal: true,
                    q_start: keys - queries,
                    key_length: keys,
                    page_tokens: 128,
                },
                &input.iter().collect::<Vec<_>>(),
                &output,
            )
            .unwrap();
        let pick = crate::variants::resolve(
            commands.last_dispatch().unwrap(),
            variant,
            &[DeviceCapability::Metal4],
        )
        .unwrap();
        commands.set_last_backend_data(BackendDispatchData::new(pick));
        commands.into_dispatches().remove(0)
    }
    #[test]
    fn tiled_address_work_depends_on_values_rather_than_storage_bytes() {
        let schedule = AttentionSchedule::Tiled { query: 32, key: 16 };
        let half = dispatch_typed(24, 8, 1, 262_144, "sdpa.steel-bounded", DType::F16);
        let float = dispatch_typed(24, 8, 1, 262_144, "sdpa.steel-bounded", DType::F32);
        let (half, hc) = costs(&half, schedule).unwrap();
        let (float, fc) = costs(&float, schedule).unwrap();
        assert_eq!(float.kv_bytes, half.kv_bytes * 2);
        assert_eq!(float.scalar_ops, half.scalar_ops);
        assert_eq!(fc.scalar_ops, hc.scalar_ops);
    }
    #[test]
    fn paged_gqa_critical_work_follows_the_encoded_head_pack() {
        for (ratio, expected) in [(1, 8), (2, 16), (8, 8)] {
            let d = paged(
                8 * ratio,
                8,
                1,
                262_144,
                "paged-sdpa.vector-gqa-two-pass-32",
            );
            let schedule = AttentionSchedule::selected(&d).unwrap();
            assert_eq!(
                schedule,
                AttentionSchedule::Vector {
                    blocks: 32,
                    heads_per_group: expected,
                    reuse: 1
                }
            );
            let (_, critical) = costs(&d, schedule).unwrap();
            assert!(critical.sfu_ops >= 262_144 / 32 * u64::from(expected) * 32 * 2);
        }
    }
    #[test]
    fn metal_pathological_attention_exceeds_the_submission_budget() {
        let backend = crate::MetalBackend::new().unwrap();
        let rates = crate::work_admission::calibration(&backend).unwrap();
        let limits =
            forja_core::GpuWorkLimits::for_timeout(std::time::Duration::from_secs(10)).unwrap();
        let predict = |d: &Dispatch| {
            let schedule = AttentionSchedule::selected(d).unwrap();
            let [heads, queries, _]: [u32; 3] = d.inputs()[0].layout().shape().try_into().unwrap();
            let (groups, threads, launches) = match schedule {
                AttentionSchedule::Tiled { query, .. } => (
                    u64::from(heads) * u64::from(queries.div_ceil(query)),
                    u64::from(query / 8 * 32),
                    1,
                ),
                AttentionSchedule::Vector {
                    blocks,
                    heads_per_group,
                    ..
                } => (
                    u64::from(heads / heads_per_group * blocks + heads),
                    u64::from(heads_per_group * 32),
                    2,
                ),
                AttentionSchedule::Decomposed => unreachable!(),
            };
            crate::work_admission::predict(
                &rates,
                d,
                launches,
                groups,
                groups * threads,
                Some(schedule),
            )
            .unwrap()
        };
        let d = paged(32, 8, 16384, 1_048_576, "paged-sdpa.steel-bounded");
        assert!(d.work().unwrap() < forja_core::DEFAULT_DISPATCH_WORK_LIMIT);
        assert!(predict(&d) > limits.submission());
    }
    #[test]
    fn fused_tiles_count_kv_reuse_without_a_materialized_score_matrix() {
        let d = dispatch(16, 8, 4096, 16384, "sdpa.steel-half");
        let (cost, critical) = costs(&d, AttentionSchedule::Tiled { query: 96, key: 16 }).unwrap();
        assert!(cost.flops > 0);
        assert_eq!(cost.scratch_read_bytes + cost.scratch_write_bytes, 0);
        assert!(cost.kv_bytes < 16 * 4096_u64 * 16384 * 128 * 4 / 32);
        assert!(critical.flops < cost.flops);
        let (fallback, _) = costs(&d, AttentionSchedule::Decomposed).unwrap();
        assert!(fallback.scratch_read_bytes > 0 && fallback.scratch_write_bytes > 0);
    }
    #[test]
    fn causal_tile_visits_match_an_independent_schedule_walk() {
        for queries in [1, 7, 33, 4097] {
            for start in [0, 7, 4097] {
                let d = dispatch(16, 8, queries, start + queries, "sdpa.steel-bounded");
                let s = Shape::of(&d).unwrap();
                for (bq, bk) in [(32, 16), (96, 16), (64, 32)] {
                    let mut sum = 0;
                    for begin in (0..queries).step_by(bq as usize) {
                        sum += u64::from((start + (begin + bq).min(queries)).div_ceil(bk));
                    }
                    assert_eq!(visits(&s, u64::from(bq), u64::from(bk)).unwrap().0, sum);
                }
            }
        }
    }
    #[test]
    fn vector_gqa_uses_source_head_packing_and_lane_replicated_softmax() {
        for (ratio, variant) in [
            (1, "sdpa.vector-two-pass"),
            (2, "sdpa.vector-two-pass"),
            (8, "sdpa.vector-gqa-two-pass-32"),
        ] {
            let d = dispatch(8 * ratio, 8, 1, 65536, variant);
            let schedule = AttentionSchedule::selected(&d).unwrap();
            let AttentionSchedule::Vector {
                blocks,
                heads_per_group,
                reuse,
            } = schedule
            else {
                unreachable!()
            };
            assert_eq!(heads_per_group, ratio);
            assert_eq!(reuse, if ratio == 8 { 2 } else { 1 });
            let (total, critical) = costs(&d, schedule).unwrap();
            assert!(total.flops > 0 && critical.flops > 0);
            let group_pairs = 65536_u64.div_ceil(u64::from(blocks)) * u64::from(ratio);
            assert!(critical.sfu_ops >= group_pairs * 32 * 2);
            assert!(critical.kv_bytes >= group_pairs * 128 * 4 / u64::from(reuse));
        }
    }
}
