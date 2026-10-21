//! Paged-copy and paged-attention CPU/Metal differentials.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use std::{any::Any, collections::HashSet, ops::Range};

use forja_core::{
    Backend, BackendDispatchData, BackendError, CommandList, DType, GraphLimits, GraphTemplate, Op,
    PagePool, PagedCopyDirection, PagedLayout, PagedPlane, ParamSpace, Slice, Submission, Tensor,
    ViewOp,
};
use half::{bf16, f16};

fn encode(values: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        DType::I32 | DType::U32 => unreachable!(),
    }
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert_eq!(remainder.len(), 0);
    words.iter().copied().map(f32::from_le_bytes).collect()
}

fn table_bytes(entries: &[(u32, u32)]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|&(page, generation)| [page, generation])
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn generation_bytes(generations: &[u32]) -> Vec<u8> {
    generations
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn run_copy<B: Backend>(
    backend: &B,
    cache_dtype: DType,
    strided: bool,
) -> Result<Vec<u8>, BackendError> {
    let ordinary_shape: &[u32] = if strided { &[7, 3, 3] } else { &[7, 3] };
    let source_storage = backend.alloc(DType::F32, ordinary_shape)?;
    let values = (0_u32..21)
        .map(|value| value as f32 / 8.0)
        .collect::<Vec<_>>();
    if strided {
        let mut backing = vec![0.0_f32; 7 * 3 * 3];
        for token in 0..7 {
            for column in 0..3 {
                backing[token * 9 + 3 + column] = values[token * 3 + column];
            }
        }
        backend.write(&source_storage, &encode(&backing, DType::F32))?;
    } else {
        backend.write(&source_storage, &encode(&values, DType::F32))?;
    }
    let strided_view = |tensor| {
        backend.view(
            tensor,
            ViewOp::Slice(vec![
                Slice::new(0, 7, 1).unwrap(),
                Slice::new(1, 1, 1).unwrap(),
                Slice::new(0, 3, 1).unwrap(),
            ]),
        )
    };
    let source = if strided {
        strided_view(&source_storage)?
    } else {
        source_storage
    };
    let output = backend.alloc(DType::F32, &[7, 3])?;
    let pool = backend.alloc(cache_dtype, &[3, 4, 3])?;
    let table = backend.alloc(DType::U32, &[2, 2])?;
    let generations = backend.alloc(DType::U32, &[3])?;
    backend.write(&table, &table_bytes(&[(2, 5), (0, 7)]))?;
    backend.write(&generations, &generation_bytes(&[7, 0, 5]))?;
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 1,
                page_tokens: 4,
            },
            &[&source, &table, &generations],
            &pool,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::OutOfPages,
                start: 1,
                page_tokens: 4,
            },
            &[&pool, &table, &generations],
            &output,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    backend.submit(commands)?.wait()?;
    backend.read(&output)
}

#[test]
fn metal_paged_copy_matches_cpu_across_page_edges() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    assert_eq!(
        run_copy(&metal, DType::F32, false).unwrap(),
        run_copy(&cpu, DType::F32, false).unwrap()
    );
}

#[test]
fn metal_paged_copy_converts_directly_into_reduced_pages() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    assert_eq!(
        run_copy(&metal, DType::F16, false).unwrap(),
        run_copy(&cpu, DType::F16, false).unwrap()
    );
}

#[test]
fn metal_paged_copy_reads_producer_views_directly() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    assert_eq!(
        run_copy(&metal, DType::F16, true).unwrap(),
        run_copy(&cpu, DType::F16, true).unwrap()
    );
}

#[test]
fn metal_graph_arena_offsets_paged_operands() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let source_a = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let source_b = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let transient_a = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let transient_b = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let pool = backend.alloc(DType::F32, &[1, 4, 2]).unwrap();
    let table = backend.alloc(DType::U32, &[1, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[1]).unwrap();
    let result = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let sink = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let expected = [3.0, 5.0, 7.0, 11.0];
    backend
        .write(&source_a, &encode(&[13.0, 17.0, 19.0, 23.0], DType::F32))
        .unwrap();
    backend
        .write(&source_b, &encode(&expected, DType::F32))
        .unwrap();
    backend.write(&table, &table_bytes(&[(0, 1)])).unwrap();
    backend
        .write(&generations, &generation_bytes(&[1]))
        .unwrap();

    let mut commands = CommandList::new();
    commands
        .dispatch(Op::Copy, &[&source_a], &transient_a)
        .unwrap();
    commands
        .dispatch(Op::Copy, &[&source_b], &transient_b)
        .unwrap();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 0,
                page_tokens: 4,
            },
            &[&transient_b, &table, &generations],
            &pool,
        )
        .unwrap();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::OutOfPages,
                start: 0,
                page_tokens: 4,
            },
            &[&pool, &table, &generations],
            &result,
        )
        .unwrap();
    commands.dispatch(Op::Copy, &[&transient_a], &sink).unwrap();
    let mut graph =
        GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
    for dispatch in commands.into_dispatches() {
        graph.record_validated(dispatch).unwrap();
    }
    let reachable = [
        source_a.buffer(),
        source_b.buffer(),
        pool.buffer(),
        table.buffer(),
        generations.buffer(),
        result.buffer(),
        sink.buffer(),
    ]
    .into_iter()
    .collect::<HashSet<_>>();
    let plan = graph.plan_memory(&reachable).unwrap();
    assert_ne!(plan.offset(transient_b.buffer()), Some(0));

    let prepared = backend.prepare_graph_memory(graph, plan).unwrap();
    backend
        .replay(&prepared, Vec::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        backend.read(&result).unwrap(),
        encode(&expected, DType::F32)
    );
}

#[test]
fn metal_paged_copy_reports_corrupt_entries_without_writing_pool() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let source = backend.alloc(DType::F32, &[1, 2]).unwrap();
    let pool = backend.alloc(DType::F32, &[2, 4, 2]).unwrap();
    let table = backend.alloc(DType::U32, &[1, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[2]).unwrap();
    backend
        .write(&source, &encode(&[3.0, 5.0], DType::F32))
        .unwrap();
    let canary = encode(&[7.0; 16], DType::F32);
    backend.write(&pool, &canary).unwrap();
    backend.write(&table, &table_bytes(&[(1, 99)])).unwrap();
    backend
        .write(&generations, &generation_bytes(&[1, 2]))
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 0,
                page_tokens: 4,
            },
            &[&source, &table, &generations],
            &pool,
        )
        .unwrap();
    let error = backend.submit(commands).unwrap().wait().unwrap_err();
    assert!(matches!(
        error,
        BackendError::DeviceErrorFlag {
            operation: "paged-copy",
            flag: 1,
            value: 0,
            ..
        }
    ));
    assert_eq!(backend.read(&pool).unwrap(), canary);
}

#[test]
fn metal_paged_copy_isolates_rejected_row_from_other_rows_in_same_dispatch() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let source = backend.alloc(DType::F32, &[3, 2]).unwrap();
    let pool = backend.alloc(DType::F32, &[3, 1, 2]).unwrap();
    let table = backend.alloc(DType::U32, &[3, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[3]).unwrap();
    backend
        .write(
            &source,
            &encode(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F32),
        )
        .unwrap();
    let canary = encode(&[9.0; 6], DType::F32);
    backend.write(&pool, &canary).unwrap();
    // No valid entry targets physical page 0, so a write there can only come from the rejected row.
    backend
        .write(&table, &table_bytes(&[(1, 10), (99, 11), (2, 12)]))
        .unwrap();
    backend
        .write(&generations, &generation_bytes(&[0, 10, 12]))
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 0,
                page_tokens: 1,
            },
            &[&source, &table, &generations],
            &pool,
        )
        .unwrap();
    let error = backend.submit(commands).unwrap().wait().unwrap_err();
    assert!(matches!(
        error,
        BackendError::DeviceErrorFlag {
            operation: "paged-copy",
            flag: 1,
            value: 1,
            ..
        }
    ));
    assert_eq!(
        decode_f32(&backend.read(&pool).unwrap()),
        vec![9.0, 9.0, 1.0, 2.0, 5.0, 6.0]
    );
}

#[test]
fn metal_paged_copy_rejects_wide_row_spanning_multiple_threadgroups() {
    const ELEMENTS: usize = 1536;
    let backend = forja_metal::MetalBackend::new().unwrap();
    let width = u32::try_from(ELEMENTS).unwrap();
    let source = backend.alloc(DType::F32, &[1, width]).unwrap();
    let pool = backend.alloc(DType::F32, &[1, 1, width]).unwrap();
    let table = backend.alloc(DType::U32, &[1, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[1]).unwrap();
    backend
        .write(&source, &encode(&vec![1.0_f32; ELEMENTS], DType::F32))
        .unwrap();
    let canary = encode(&vec![9.0_f32; ELEMENTS], DType::F32);
    backend.write(&pool, &canary).unwrap();
    backend.write(&table, &table_bytes(&[(0, 99)])).unwrap();
    backend
        .write(&generations, &generation_bytes(&[1]))
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 0,
                page_tokens: 1,
            },
            &[&source, &table, &generations],
            &pool,
        )
        .unwrap();
    let error = backend.submit(commands).unwrap().wait().unwrap_err();
    assert!(matches!(
        error,
        BackendError::DeviceErrorFlag {
            operation: "paged-copy",
            flag: 1,
            value: 0,
            ..
        }
    ));
    assert_eq!(backend.read(&pool).unwrap(), canary);
}

#[test]
fn metal_in_flight_paged_read_defers_page_reuse() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let allocator = PagePool::new(
        PagedLayout::new(
            4,
            vec![PagedPlane {
                dtype: DType::F32,
                elements_per_token: 2,
                alignment: 4,
                label: "cache".to_owned(),
            }],
        )
        .unwrap(),
        2,
        1_024,
    )
    .unwrap();
    let sequence = allocator.reserve(4).unwrap();
    let snapshot = sequence.table().unwrap();
    let old = snapshot.resolve(0).unwrap();
    let generation = snapshot.entries()[0].generation();
    let pages = backend.alloc(DType::F32, &[2, 4, 2]).unwrap();
    let table = backend.alloc(DType::U32, &[1, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[2]).unwrap();
    let output = backend.alloc(DType::F32, &[4, 2]).unwrap();
    let canary = encode(&[13.0; 16], DType::F32);
    backend.write(&pages, &canary).unwrap();
    backend
        .write(&table, &table_bytes(&[(old, generation)]))
        .unwrap();
    backend
        .write(&generations, &generation_bytes(snapshot.generations()))
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::OutOfPages,
                start: 0,
                page_tokens: 4,
            },
            &[&pages, &table, &generations],
            &output,
        )
        .unwrap();
    let submission = backend.submit(commands).unwrap();
    let serial = submission.completion_serial().unwrap();
    sequence.mark_submitted(serial).unwrap();
    drop(sequence);

    let replacement = allocator.reserve(4).unwrap();
    assert_ne!(replacement.table().unwrap().resolve(0).unwrap(), old);
    submission.wait().unwrap();
    assert_eq!(
        backend.read(&output).unwrap(),
        encode(&[13.0; 8], DType::F32)
    );
    allocator.observe_completion(serial).unwrap();
}

fn seeded_query_values(
    query_heads: u32,
    query_length: u32,
    width: u32,
    head_offset: u32,
) -> Vec<f32> {
    let element_offset = head_offset
        .checked_mul(query_length)
        .and_then(|offset| offset.checked_mul(width))
        .unwrap();
    (0..query_heads * query_length * width)
        .map(|index| {
            let source = element_offset.checked_add(index).unwrap();
            ((source * 17) % 89) as f32 / 88.0 - 0.5
        })
        .collect()
}

fn finish_sdpa<B: Backend>(
    backend: &B,
    output: &Tensor,
    submission: B::Submission,
    rejected_output: Option<&mut Vec<f32>>,
) -> Result<Vec<f32>, BackendError> {
    if let Err(error) = submission.wait() {
        drop(submission);
        if let Some(capture) = rejected_output {
            *capture = decode_f32(&backend.read(output)?);
        }
        return Err(error);
    }
    backend.read(output).map(|bytes| decode_f32(&bytes))
}

fn paged_entries(logical_pages: u32, contiguous: bool) -> (u32, Vec<(u32, u32)>) {
    let physical_pages = logical_pages + u32::from(!contiguous) + 1;
    let entries = (0..logical_pages)
        .map(|logical| {
            let page = if contiguous || logical + 2 < logical_pages {
                logical + 1
            } else {
                logical + 2
            };
            (page, 11 + page)
        })
        .collect();
    (physical_pages, entries)
}

#[allow(clippy::too_many_arguments)]
fn run_sdpa<B: Backend + 'static>(
    backend: &B,
    dtype: DType,
    key_length: u32,
    query_length: u32,
    causal: bool,
    contiguous: bool,
    query_heads: u32,
    kv_heads: u32,
    head_offsets: (u32, u32),
    corrupt_pages: &[usize],
    variant: Option<&str>,
    rejected_output: Option<&mut Vec<f32>>,
) -> Result<Vec<f32>, BackendError> {
    const PAGE_TOKENS: u32 = 64;
    const WIDTH: u32 = 128;

    let q_start = key_length
        .checked_sub(query_length)
        .ok_or(BackendError::InvalidInput)?;
    let logical_pages = key_length.div_ceil(PAGE_TOKENS);
    let (physical_pages, entries) = paged_entries(logical_pages, contiguous);
    let mut generations = (0..physical_pages)
        .map(|page| 11 + page)
        .collect::<Vec<_>>();
    for &logical_page in corrupt_pages {
        generations[usize::try_from(entries[logical_page].0).unwrap()] = 0;
    }
    let (query_head_offset, kv_head_offset) = head_offsets;
    let kv_element_offset = kv_head_offset.checked_mul(WIDTH).unwrap();
    let plane_elements = usize::try_from(kv_heads * WIDTH).unwrap();
    let pool_tokens = usize::try_from(physical_pages * PAGE_TOKENS).unwrap();
    let mut keys = vec![f32::NAN; pool_tokens * plane_elements];
    let mut values = vec![f32::NAN; pool_tokens * plane_elements];
    for token in 0..key_length {
        let logical_page = token / PAGE_TOKENS;
        let physical = entries[usize::try_from(logical_page).unwrap()].0;
        let physical_token = physical * PAGE_TOKENS + token % PAGE_TOKENS;
        let base = usize::try_from(physical_token).unwrap() * plane_elements;
        for element in 0..plane_elements {
            let seed = kv_element_offset
                .checked_add(u32::try_from(element).unwrap())
                .unwrap();
            keys[base + element] =
                ((token.wrapping_mul(13) + seed.wrapping_mul(7)) % 101) as f32 / 100.0 - 0.5;
            values[base + element] =
                ((token.wrapping_mul(5) + seed.wrapping_mul(11)) % 97) as f32 / 96.0 - 0.5;
        }
    }
    let query_values = seeded_query_values(query_heads, query_length, WIDTH, query_head_offset);
    let query_storage = backend.alloc(DType::F32, &[query_length, query_heads, WIDTH])?;
    let query = backend.view(&query_storage, ViewOp::Permute(vec![1, 0, 2]))?;
    let key = backend.alloc(dtype, &[physical_pages, PAGE_TOKENS, kv_heads, WIDTH])?;
    let value = backend.alloc(dtype, &[physical_pages, PAGE_TOKENS, kv_heads, WIDTH])?;
    let table = backend.alloc(DType::U32, &[logical_pages, 2])?;
    let generation_tensor = backend.alloc(DType::U32, &[physical_pages])?;
    let output_storage = backend.alloc(DType::F32, &[query_length, query_heads, WIDTH])?;
    let output = backend.view(&output_storage, ViewOp::Permute(vec![1, 0, 2]))?;
    let mut query_storage_values = vec![0.0_f32; query_values.len()];
    for head in 0..query_heads {
        for row in 0..query_length {
            for column in 0..WIDTH {
                let source = usize::try_from((head * query_length + row) * WIDTH + column).unwrap();
                let target = usize::try_from((row * query_heads + head) * WIDTH + column).unwrap();
                query_storage_values[target] = query_values[source];
            }
        }
    }
    backend.write(&query_storage, &encode(&query_storage_values, DType::F32))?;
    backend.write(&key, &encode(&keys, dtype))?;
    backend.write(&value, &encode(&values, dtype))?;
    backend.write(&table, &table_bytes(&entries))?;
    backend.write(&generation_tensor, &generation_bytes(&generations))?;
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedSdpa {
                scale: 1.0 / (WIDTH as f32).sqrt(),
                causal,
                q_start,
                key_length,
                page_tokens: PAGE_TOKENS,
                contiguous_entries: contiguous,
            },
            &[&query, &key, &value, &table, &generation_tensor],
            &output,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    if let Some(name) = variant {
        let metal = (backend as &dyn Any)
            .downcast_ref::<forja_metal::MetalBackend>()
            .unwrap();
        let selection = metal
            .validate_variant(commands.last_dispatch().unwrap(), name)
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(selection)));
    }
    finish_sdpa(backend, &output, backend.submit(commands)?, rejected_output)
}

fn check_paged_sdpa_real_shapes(dtype: DType, shapes: &[(u32, u32)], query_head_range: Range<u32>) {
    const QUERY_HEADS: u32 = 16;
    const KV_HEADS: u32 = 8;
    const WIDTH: u32 = 128;

    assert!(query_head_range.start < query_head_range.end);
    assert_eq!(query_head_range.start % 2, 0);
    assert_eq!(query_head_range.end % 2, 0);
    assert!(query_head_range.end <= QUERY_HEADS);
    let reference_query_heads = query_head_range.end - query_head_range.start;
    let reference_kv_heads = reference_query_heads / 2;
    let reference_kv_offset = query_head_range.start / 2;
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for &(length, query_length) in shapes {
        let reference = run_sdpa(
            &cpu,
            dtype,
            length,
            query_length,
            true,
            false,
            reference_query_heads,
            reference_kv_heads,
            (query_head_range.start, reference_kv_offset),
            &[],
            None,
            None,
        )
        .unwrap();
        let variant = match (dtype, query_length) {
            (_, 1) => "paged-sdpa.vector",
            (DType::F32, _) => "paged-sdpa.steel",
            _ => "paged-sdpa.steel-half",
        };
        let candidate = run_sdpa(
            &metal,
            dtype,
            length,
            query_length,
            true,
            false,
            QUERY_HEADS,
            KV_HEADS,
            (0, 0),
            &[],
            Some(variant),
            None,
        )
        .unwrap();
        let row_elements = usize::try_from(query_length * WIDTH).unwrap();
        let candidate_start = usize::try_from(query_head_range.start).unwrap() * row_elements;
        let candidate_end = usize::try_from(query_head_range.end).unwrap() * row_elements;
        let error = forja_testing::normwise_relative_error(
            &reference,
            &candidate[candidate_start..candidate_end],
        );
        assert!(
            error <= forja_testing::dtype_tolerance(dtype).unwrap(),
            "dtype={dtype:?} length={length} error={error}"
        );
    }
}

#[test]
fn metal_paged_sdpa_matches_cpu_at_short_real_shapes() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        check_paged_sdpa_real_shapes(dtype, &[(1, 1), (7, 2), (33, 7)], 0..16);
    }
}

macro_rules! real_shape_tests {
    ($(($name:ident, $dtype:expr, $length:expr, $query_length:expr, $heads:expr)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_paged_sdpa_real_shapes($dtype, &[($length, $query_length)], $heads);
            }
        )+
    };
}

#[rustfmt::skip]
real_shape_tests!(
    (metal_paged_sdpa_matches_cpu_at_4097_f32_heads_0_3, DType::F32, 4_097, 7, 0..4),
    (metal_paged_sdpa_matches_cpu_at_4097_f32_heads_4_7, DType::F32, 4_097, 7, 4..8),
    (metal_paged_sdpa_matches_cpu_at_4097_f32_heads_8_11, DType::F32, 4_097, 7, 8..12),
    (metal_paged_sdpa_matches_cpu_at_4097_f32_heads_12_15, DType::F32, 4_097, 7, 12..16),
    (metal_paged_sdpa_matches_cpu_at_4097_f16_heads_0_3, DType::F16, 4_097, 7, 0..4),
    (metal_paged_sdpa_matches_cpu_at_4097_f16_heads_4_7, DType::F16, 4_097, 7, 4..8),
    (metal_paged_sdpa_matches_cpu_at_4097_f16_heads_8_11, DType::F16, 4_097, 7, 8..12),
    (metal_paged_sdpa_matches_cpu_at_4097_f16_heads_12_15, DType::F16, 4_097, 7, 12..16),
    (metal_paged_sdpa_matches_cpu_at_4097_bf16_heads_0_3, DType::BF16, 4_097, 7, 0..4),
    (metal_paged_sdpa_matches_cpu_at_4097_bf16_heads_4_7, DType::BF16, 4_097, 7, 4..8),
    (metal_paged_sdpa_matches_cpu_at_4097_bf16_heads_8_11, DType::BF16, 4_097, 7, 8..12),
    (metal_paged_sdpa_matches_cpu_at_4097_bf16_heads_12_15, DType::BF16, 4_097, 7, 12..16),
    (metal_paged_sdpa_matches_cpu_at_16384_f16_heads_0_3, DType::F16, 16_384, 2, 0..4),
    (metal_paged_sdpa_matches_cpu_at_16384_f16_heads_4_7, DType::F16, 16_384, 2, 4..8),
    (metal_paged_sdpa_matches_cpu_at_16384_f16_heads_8_11, DType::F16, 16_384, 2, 8..12),
    (metal_paged_sdpa_matches_cpu_at_16384_f16_heads_12_15, DType::F16, 16_384, 2, 12..16),
    (metal_paged_sdpa_matches_cpu_at_16384_bf16_heads_0_3, DType::BF16, 16_384, 2, 0..4),
    (metal_paged_sdpa_matches_cpu_at_16384_bf16_heads_4_7, DType::BF16, 16_384, 2, 4..8),
    (metal_paged_sdpa_matches_cpu_at_16384_bf16_heads_8_11, DType::BF16, 16_384, 2, 8..12),
    (metal_paged_sdpa_matches_cpu_at_16384_bf16_heads_12_15, DType::BF16, 16_384, 2, 12..16),
);

fn check_paged_gqa_two_pass(length: u32, variant: &str) {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for contiguous in [false, true] {
        let reference = run_sdpa(
            &cpu,
            DType::F16,
            length,
            1,
            false,
            contiguous,
            32,
            4,
            (0, 0),
            &[],
            None,
            None,
        )
        .unwrap();
        let candidate = run_sdpa(
            &metal,
            DType::F16,
            length,
            1,
            false,
            contiguous,
            32,
            4,
            (0, 0),
            &[],
            Some(variant),
            None,
        )
        .unwrap();
        let error = forja_testing::normwise_relative_error(&reference, &candidate);
        assert!(
            error <= forja_testing::F16_TOLERANCE,
            "length={length} contiguous={contiguous} error={error}"
        );
    }
}

macro_rules! paged_gqa_two_pass_tests {
    ($(($name:ident, $length:expr, $variant:expr)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                check_paged_gqa_two_pass($length, $variant);
            }
        )+
    };
}

paged_gqa_two_pass_tests!(
    (
        metal_paged_gqa_two_pass_matches_cpu_at_512,
        512,
        "paged-sdpa.vector-gqa-two-pass-32"
    ),
    (
        metal_paged_gqa_two_pass_matches_cpu_at_513,
        513,
        "paged-sdpa.vector-gqa-two-pass-32"
    ),
    (
        metal_paged_gqa_two_pass_matches_cpu_at_2048,
        2_048,
        "paged-sdpa.vector-gqa-two-pass"
    ),
    (
        metal_paged_gqa_two_pass_matches_cpu_at_4000,
        4_000,
        "paged-sdpa.vector-gqa-two-pass-128"
    ),
    (
        metal_paged_gqa_two_pass_matches_cpu_at_4001,
        4_001,
        "paged-sdpa.vector-gqa-two-pass-128"
    ),
);

#[test]
fn metal_paged_gqa_two_pass_refuses_corrupt_page_without_reading_it() {
    let mut rejected_output = Vec::new();
    let error = run_sdpa(
        &forja_metal::MetalBackend::new().unwrap(),
        DType::F16,
        512,
        1,
        false,
        false,
        32,
        4,
        (0, 0),
        &[7],
        Some("paged-sdpa.vector-gqa-two-pass-32"),
        Some(&mut rejected_output),
    )
    .unwrap_err();
    assert!(matches!(error, BackendError::DeviceErrorFlag { .. }));
    assert_eq!(rejected_output.len(), 32 * 128);
    assert!(rejected_output.iter().all(|value| value.is_finite()));
}

#[test]
fn metal_paged_gqa_two_pass_keeps_barriers_uniform_across_rejected_pages() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for (length, variant) in [
        (512_u32, "paged-sdpa.vector-gqa-two-pass-32"),
        (513, "paged-sdpa.vector-gqa-two-pass-32"),
        (2_048, "paged-sdpa.vector-gqa-two-pass"),
        (4_000, "paged-sdpa.vector-gqa-two-pass-128"),
    ] {
        let pages = usize::try_from(length.div_ceil(64)).unwrap();
        let corrupt_pages = (1..pages).step_by(2).collect::<Vec<_>>();
        for _ in 0..50 {
            let mut rejected_output = Vec::new();
            let error = run_sdpa(
                &backend,
                DType::F16,
                length,
                1,
                false,
                false,
                32,
                4,
                (0, 0),
                &corrupt_pages,
                Some(variant),
                Some(&mut rejected_output),
            )
            .unwrap_err();
            assert!(matches!(error, BackendError::DeviceErrorFlag { .. }));
            assert_eq!(rejected_output.len(), 32 * 128);
            assert!(rejected_output.iter().all(|value| value.is_finite()));
        }
    }
}

#[test]
fn metal_paged_gqa_two_pass_replaces_stale_scratch_on_early_refusal() {
    let mut rejected_output = Vec::new();
    let error = run_sdpa(
        &forja_metal::MetalBackend::new().unwrap(),
        DType::F16,
        512,
        1,
        false,
        false,
        32,
        4,
        (0, 0),
        &[0],
        Some("paged-sdpa.vector-gqa-two-pass-32"),
        Some(&mut rejected_output),
    )
    .unwrap_err();
    assert!(matches!(error, BackendError::DeviceErrorFlag { .. }));
    assert_eq!(rejected_output, vec![0.0; 32 * 128]);
}

#[test]
fn metal_paged_sdpa_uses_contiguous_entry_path() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    let reference = run_sdpa(
        &cpu,
        DType::F16,
        129,
        1,
        false,
        true,
        16,
        8,
        (0, 0),
        &[],
        None,
        None,
    )
    .unwrap();
    let candidate = run_sdpa(
        &metal,
        DType::F16,
        129,
        1,
        false,
        true,
        16,
        8,
        (0, 0),
        &[],
        None,
        None,
    )
    .unwrap();
    assert!(
        forja_testing::normwise_relative_error(&reference, &candidate)
            <= forja_testing::F16_TOLERANCE
    );
}

#[test]
fn metal_paged_sdpa_falls_back_after_a_consecutive_prefix() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for (query_length, variant) in [(1, "paged-sdpa.vector"), (7, "paged-sdpa.steel-half")] {
        let reference = run_sdpa(
            &cpu,
            DType::F16,
            257,
            query_length,
            true,
            false,
            16,
            8,
            (0, 0),
            &[],
            None,
            None,
        )
        .unwrap();
        let candidate = run_sdpa(
            &metal,
            DType::F16,
            257,
            query_length,
            true,
            false,
            16,
            8,
            (0, 0),
            &[],
            Some(variant),
            None,
        )
        .unwrap();
        assert!(
            forja_testing::normwise_relative_error(&reference, &candidate)
                <= forja_testing::F16_TOLERANCE,
            "variant={variant}"
        );
    }
}

#[test]
fn metal_paged_sdpa_matches_causal_prefill() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    let reference = run_sdpa(
        &cpu,
        DType::F16,
        11,
        11,
        true,
        true,
        32,
        4,
        (0, 0),
        &[],
        None,
        None,
    )
    .unwrap();
    let candidate = run_sdpa(
        &metal,
        DType::F16,
        11,
        11,
        true,
        true,
        32,
        4,
        (0, 0),
        &[],
        Some("paged-sdpa.steel-half"),
        None,
    )
    .unwrap();
    assert!(
        forja_testing::normwise_relative_error(&reference, &candidate)
            <= forja_testing::F16_TOLERANCE
    );
}

#[test]
fn metal_paged_steel_refuses_corrupt_entries() {
    let error = run_sdpa(
        &forja_metal::MetalBackend::new().unwrap(),
        DType::F16,
        129,
        7,
        true,
        true,
        16,
        8,
        (0, 0),
        &[0],
        Some("paged-sdpa.steel-half"),
        None,
    )
    .unwrap_err();
    assert!(matches!(error, BackendError::DeviceErrorFlag { .. }));
}

#[test]
fn metal_paged_copy_then_sdpa_matches_cpu() {
    fn run<B: Backend>(backend: &B) -> Result<Vec<f32>, BackendError> {
        const TOKENS: u32 = 11;
        const PAGE_TOKENS: u32 = 64;
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 4;
        const WIDTH: u32 = 128;
        let elements = KV_HEADS * WIDTH;
        let ordinary_values = (0..TOKENS * elements)
            .map(|index| ((index * 7) % 101) as f32 / 100.0 - 0.5)
            .collect::<Vec<_>>();
        let query_values = (0..QUERY_HEADS * TOKENS * WIDTH)
            .map(|index| ((index * 17) % 89) as f32 / 88.0 - 0.5)
            .collect::<Vec<_>>();
        let ordinary = backend.alloc(DType::F16, &[TOKENS, elements])?;
        let key_rows = backend.alloc(DType::F16, &[1, PAGE_TOKENS, elements])?;
        let value_rows = backend.alloc(DType::F16, &[1, PAGE_TOKENS, elements])?;
        let key = backend.view(
            &key_rows,
            ViewOp::Reshape(vec![1, PAGE_TOKENS, KV_HEADS, WIDTH]),
        )?;
        let value = backend.view(
            &value_rows,
            ViewOp::Reshape(vec![1, PAGE_TOKENS, KV_HEADS, WIDTH]),
        )?;
        let table = backend.alloc(DType::U32, &[1, 2])?;
        let generations = backend.alloc(DType::U32, &[1])?;
        let query = backend.alloc(DType::F32, &[QUERY_HEADS, TOKENS, WIDTH])?;
        let output = backend.alloc(DType::F32, &[QUERY_HEADS, TOKENS, WIDTH])?;
        backend.write(&ordinary, &encode(&ordinary_values, DType::F16))?;
        backend.write(&table, &table_bytes(&[(0, 1)]))?;
        backend.write(&generations, &generation_bytes(&[1]))?;
        backend.write(&query, &encode(&query_values, DType::F32))?;
        let mut commands = CommandList::new();
        for pool in [&key_rows, &value_rows] {
            commands
                .dispatch(
                    Op::PagedCopy {
                        direction: PagedCopyDirection::IntoPages,
                        start: 0,
                        page_tokens: PAGE_TOKENS,
                    },
                    &[&ordinary, &table, &generations],
                    pool,
                )
                .map_err(|_| BackendError::InvalidInput)?;
        }
        commands
            .dispatch(
                Op::PagedSdpa {
                    scale: 1.0 / (WIDTH as f32).sqrt(),
                    causal: true,
                    q_start: 0,
                    key_length: TOKENS,
                    page_tokens: PAGE_TOKENS,
                    contiguous_entries: true,
                },
                &[&query, &key, &value, &table, &generations],
                &output,
            )
            .map_err(|_| BackendError::InvalidInput)?;
        backend.submit(commands)?.wait()?;
        backend.read(&output).map(|bytes| decode_f32(&bytes))
    }

    let reference = run(&forja_cpu::CpuBackend::new()).unwrap();
    let candidate = run(&forja_metal::MetalBackend::new().unwrap()).unwrap();
    assert!(
        forja_testing::normwise_relative_error(&reference, &candidate)
            <= forja_testing::F16_TOLERANCE
    );
}

#[allow(clippy::too_many_lines)]
fn benchmark_commands(
    backend: &forja_metal::MetalBackend,
    key_length: u32,
    query_length: u32,
    contiguous_variant: &str,
    paged_variant: &str,
) -> (
    impl FnMut() -> CommandList + use<>,
    impl FnMut() -> CommandList + use<>,
) {
    const PAGE_TOKENS: u32 = 64;
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 4;
    const WIDTH: u32 = 128;

    let pages = key_length.div_ceil(PAGE_TOKENS);
    let query = backend
        .alloc(DType::F32, &[QUERY_HEADS, query_length, WIDTH])
        .unwrap();
    let contiguous_key = backend
        .alloc(DType::F16, &[KV_HEADS, key_length, WIDTH])
        .unwrap();
    let contiguous_value = backend
        .alloc(DType::F16, &[KV_HEADS, key_length, WIDTH])
        .unwrap();
    let paged_key = backend
        .alloc(DType::F16, &[pages, PAGE_TOKENS, KV_HEADS, WIDTH])
        .unwrap();
    let paged_value = backend
        .alloc(DType::F16, &[pages, PAGE_TOKENS, KV_HEADS, WIDTH])
        .unwrap();
    let table = backend.alloc(DType::U32, &[pages, 2]).unwrap();
    let generations = backend.alloc(DType::U32, &[pages]).unwrap();
    let contiguous_output = backend
        .alloc(DType::F32, &[QUERY_HEADS, query_length, WIDTH])
        .unwrap();
    let paged_output = backend
        .alloc(DType::F32, &[QUERY_HEADS, query_length, WIDTH])
        .unwrap();
    let query_values = (0..QUERY_HEADS * query_length * WIDTH)
        .map(|index| ((index * 17) % 89) as f32 / 88.0 - 0.5)
        .collect::<Vec<_>>();
    let contiguous_values = (0..KV_HEADS * key_length * WIDTH)
        .map(|index| ((index * 7) % 101) as f32 / 100.0 - 0.5)
        .collect::<Vec<_>>();
    let padded_tokens = pages * PAGE_TOKENS;
    let mut paged_values =
        vec![0.0_f32; usize::try_from(padded_tokens * KV_HEADS * WIDTH).unwrap()];
    for head in 0..KV_HEADS {
        for token in 0..key_length {
            for column in 0..WIDTH {
                let contiguous_index =
                    usize::try_from((head * key_length + token) * WIDTH + column).unwrap();
                let paged_index =
                    usize::try_from((token * KV_HEADS + head) * WIDTH + column).unwrap();
                paged_values[paged_index] = contiguous_values[contiguous_index];
            }
        }
    }
    backend
        .write(&query, &encode(&query_values, DType::F32))
        .unwrap();
    backend
        .write(&contiguous_key, &encode(&contiguous_values, DType::F16))
        .unwrap();
    backend
        .write(&contiguous_value, &encode(&contiguous_values, DType::F16))
        .unwrap();
    backend
        .write(&paged_key, &encode(&paged_values, DType::F16))
        .unwrap();
    backend
        .write(&paged_value, &encode(&paged_values, DType::F16))
        .unwrap();
    let entries = (0..pages).map(|page| (page, 1)).collect::<Vec<_>>();
    backend.write(&table, &table_bytes(&entries)).unwrap();
    backend
        .write(
            &generations,
            &generation_bytes(&vec![1; usize::try_from(pages).unwrap()]),
        )
        .unwrap();
    let scale = 1.0 / (WIDTH as f32).sqrt();
    let contiguous_op = Op::Sdpa {
        scale,
        causal: true,
        q_start: key_length - query_length,
    };
    let contiguous_variant = backend
        .validate_variant_for_op(
            contiguous_op,
            &[&query, &contiguous_key, &contiguous_value],
            &[&contiguous_output],
            contiguous_variant,
        )
        .unwrap();
    let paged_op = Op::PagedSdpa {
        scale,
        causal: true,
        q_start: key_length - query_length,
        key_length,
        page_tokens: PAGE_TOKENS,
        contiguous_entries: true,
    };
    let paged_variant = backend
        .validate_variant_for_op(
            paged_op,
            &[&query, &paged_key, &paged_value, &table, &generations],
            &[&paged_output],
            paged_variant,
        )
        .unwrap();
    let paged_query = query.clone();
    let contiguous = move || {
        let mut commands = CommandList::new();
        commands
            .dispatch(
                contiguous_op,
                &[&query, &contiguous_key, &contiguous_value],
                &contiguous_output,
            )
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(contiguous_variant)));
        commands
    };
    let paged_commands = move || {
        let mut commands = CommandList::new();
        commands
            .dispatch(
                paged_op,
                &[&paged_query, &paged_key, &paged_value, &table, &generations],
                &paged_output,
            )
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(paged_variant)));
        commands
    };
    (contiguous, paged_commands)
}

#[test]
#[ignore = "manual paged-attention kernel checkpoint"]
fn metal_paged_sdpa_microbenchmark() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for length in [512, 4_096] {
        let (contiguous, paged) = benchmark_commands(
            &backend,
            length,
            length,
            "sdpa.steel-half",
            "paged-sdpa.steel-half",
        );
        let (contiguous_time, paged_time) =
            super::common::comparative_gpu_times(&backend, contiguous, paged);
        eprintln!(
            "paged-sdpa queries={length} steel_half_us={} paged_steel_half_us={} ratio={:.4}",
            contiguous_time.as_secs_f64() * 1.0e6,
            paged_time.as_secs_f64() * 1.0e6,
            paged_time.as_secs_f64() / contiguous_time.as_secs_f64(),
        );
        assert!(paged_time.as_secs_f64() / contiguous_time.as_secs_f64() <= 1.04);
    }
    let mut decode_regressions = Vec::new();
    for (length, blocks) in [
        (512, "-32"),
        (513, "-32"),
        (2_048, ""),
        (2_049, ""),
        (4_000, "-128"),
        (4_001, "-128"),
    ] {
        let (contiguous, paged) = benchmark_commands(
            &backend,
            length,
            1,
            &format!("sdpa.vector-gqa-two-pass{blocks}"),
            &format!("paged-sdpa.vector-gqa-two-pass{blocks}"),
        );
        let (contiguous_time, paged_time) =
            super::common::comparative_gpu_times(&backend, contiguous, paged);
        let ratio = paged_time.as_secs_f64() / contiguous_time.as_secs_f64();
        eprintln!(
            "paged-sdpa keys={length} two_pass_us={} paged_two_pass_us={} ratio={:.4}",
            contiguous_time.as_secs_f64() * 1.0e6,
            paged_time.as_secs_f64() * 1.0e6,
            ratio,
        );
        if ratio > 1.02 {
            decode_regressions.push((length, ratio));
        }
    }
    assert!(decode_regressions.is_empty(), "{decode_regressions:?}");
}
