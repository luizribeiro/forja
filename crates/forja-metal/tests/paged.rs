//! Paged-copy and paged-attention CPU/Metal differentials.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use forja_core::{
    Backend, BackendDispatchData, BackendError, CommandList, DType, Op, PagedCopyDirection,
    Submission,
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
    assert!(remainder.is_empty());
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

fn run_copy<B: Backend>(backend: &B) -> Result<Vec<u8>, BackendError> {
    let source = backend.alloc(DType::F32, &[7, 3])?;
    let pool = backend.alloc(DType::F32, &[3, 4, 3])?;
    let table = backend.alloc(DType::U32, &[2, 2])?;
    let generations = backend.alloc(DType::U32, &[3])?;
    let output = backend.alloc(DType::F32, &[7, 3])?;
    let values = (0_u32..21)
        .map(|value| value as f32 / 8.0)
        .collect::<Vec<_>>();
    backend.write(&source, &encode(&values, DType::F32))?;
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
    assert_eq!(run_copy(&metal).unwrap(), run_copy(&cpu).unwrap());
}

fn run_sdpa<B: Backend>(
    backend: &B,
    dtype: DType,
    key_length: u32,
    contiguous: bool,
    query_heads: u32,
    kv_heads: u32,
) -> Result<Vec<f32>, BackendError> {
    const PAGE_TOKENS: u32 = 64;
    const WIDTH: u32 = 128;

    let logical_pages = key_length.div_ceil(PAGE_TOKENS);
    let physical_pages = logical_pages + 1;
    let entries = (0..logical_pages)
        .map(|logical| {
            let page = if contiguous {
                logical + 1
            } else {
                logical_pages - logical
            };
            (page, 11 + page)
        })
        .collect::<Vec<_>>();
    let generations = (0..physical_pages)
        .map(|page| 11 + page)
        .collect::<Vec<_>>();
    let plane_elements = usize::try_from(kv_heads * WIDTH).unwrap();
    let pool_tokens = usize::try_from(physical_pages * PAGE_TOKENS).unwrap();
    let mut keys = vec![0.0_f32; pool_tokens * plane_elements];
    let mut values = vec![0.0_f32; pool_tokens * plane_elements];
    for token in 0..key_length {
        let logical_page = token / PAGE_TOKENS;
        let physical = entries[usize::try_from(logical_page).unwrap()].0;
        let physical_token = physical * PAGE_TOKENS + token % PAGE_TOKENS;
        let base = usize::try_from(physical_token).unwrap() * plane_elements;
        for element in 0..plane_elements {
            let seed = u32::try_from(element).unwrap();
            keys[base + element] =
                ((token.wrapping_mul(13) + seed.wrapping_mul(7)) % 101) as f32 / 100.0 - 0.5;
            values[base + element] =
                ((token.wrapping_mul(5) + seed.wrapping_mul(11)) % 97) as f32 / 96.0 - 0.5;
        }
    }
    let query_values = (0..query_heads * WIDTH)
        .map(|index| ((index * 17) % 89) as f32 / 88.0 - 0.5)
        .collect::<Vec<_>>();
    let query = backend.alloc(DType::F32, &[query_heads, 1, WIDTH])?;
    let key = backend.alloc(dtype, &[physical_pages, PAGE_TOKENS, kv_heads, WIDTH])?;
    let value = backend.alloc(dtype, &[physical_pages, PAGE_TOKENS, kv_heads, WIDTH])?;
    let table = backend.alloc(DType::U32, &[logical_pages, 2])?;
    let generation_tensor = backend.alloc(DType::U32, &[physical_pages])?;
    let output = backend.alloc(DType::F32, &[query_heads, 1, WIDTH])?;
    backend.write(&query, &encode(&query_values, DType::F32))?;
    backend.write(&key, &encode(&keys, dtype))?;
    backend.write(&value, &encode(&values, dtype))?;
    backend.write(&table, &table_bytes(&entries))?;
    backend.write(&generation_tensor, &generation_bytes(&generations))?;
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedSdpa {
                scale: 1.0 / (WIDTH as f32).sqrt(),
                causal: false,
                q_start: 0,
                key_length,
                page_tokens: PAGE_TOKENS,
                contiguous_entries: contiguous,
            },
            &[&query, &key, &value, &table, &generation_tensor],
            &output,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    backend.submit(commands)?.wait()?;
    backend.read(&output).map(|bytes| decode_f32(&bytes))
}

#[test]
fn metal_paged_sdpa_matches_cpu_at_real_shapes() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for length in [1, 7, 33, 4_097] {
            let reference = run_sdpa(&cpu, dtype, length, false, 16, 8).unwrap();
            let candidate = run_sdpa(&metal, dtype, length, false, 16, 8).unwrap();
            let error = forja_testing::normwise_relative_error(&reference, &candidate);
            assert!(
                error <= forja_testing::dtype_tolerance(dtype).unwrap(),
                "dtype={dtype:?} length={length} error={error}"
            );
        }
    }
}

#[test]
fn metal_paged_gqa_two_pass_matches_cpu() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    for length in [512, 4_000] {
        for contiguous in [false, true] {
            let reference = run_sdpa(&cpu, DType::F16, length, contiguous, 32, 4).unwrap();
            let candidate = run_sdpa(&metal, DType::F16, length, contiguous, 32, 4).unwrap();
            let error = forja_testing::normwise_relative_error(&reference, &candidate);
            assert!(
                error <= forja_testing::F16_TOLERANCE,
                "length={length} contiguous={contiguous} error={error}"
            );
        }
    }
}

#[test]
fn metal_paged_sdpa_uses_contiguous_entry_path() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    let reference = run_sdpa(&cpu, DType::F16, 129, true, 16, 8).unwrap();
    let candidate = run_sdpa(&metal, DType::F16, 129, true, 16, 8).unwrap();
    assert!(
        forja_testing::normwise_relative_error(&reference, &candidate)
            <= forja_testing::F16_TOLERANCE
    );
}

#[allow(clippy::too_many_lines)]
fn benchmark_commands(
    backend: &forja_metal::MetalBackend,
    key_length: u32,
) -> (
    impl FnMut() -> CommandList + use<>,
    impl FnMut() -> CommandList + use<>,
) {
    const PAGE_TOKENS: u32 = 64;
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 4;
    const WIDTH: u32 = 128;

    let pages = key_length.div_ceil(PAGE_TOKENS);
    let query = backend.alloc(DType::F32, &[QUERY_HEADS, 1, WIDTH]).unwrap();
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
    let contiguous_output = backend.alloc(DType::F32, &[QUERY_HEADS, 1, WIDTH]).unwrap();
    let paged_output = backend.alloc(DType::F32, &[QUERY_HEADS, 1, WIDTH]).unwrap();
    let query_values = (0..QUERY_HEADS * WIDTH)
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
        causal: false,
        q_start: 0,
    };
    let contiguous_variant = backend
        .validate_variant_for_op(
            contiguous_op,
            &[&query, &contiguous_key, &contiguous_value],
            &[&contiguous_output],
            "sdpa.vector-gqa-two-pass",
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
                Op::PagedSdpa {
                    scale,
                    causal: false,
                    q_start: 0,
                    key_length,
                    page_tokens: PAGE_TOKENS,
                    contiguous_entries: true,
                },
                &[&paged_query, &paged_key, &paged_value, &table, &generations],
                &paged_output,
            )
            .unwrap();
        commands
    };
    (contiguous, paged_commands)
}

#[test]
#[ignore = "manual paged-attention kernel checkpoint"]
fn metal_paged_sdpa_microbenchmark() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for length in [512, 4_000] {
        let (contiguous, paged) = benchmark_commands(&backend, length);
        let (contiguous_time, paged_time) =
            super::common::comparative_gpu_times(&backend, contiguous, paged);
        eprintln!(
            "paged-sdpa keys={length} gqa_two_pass_us={} paged_us={} ratio={:.4}",
            contiguous_time.as_secs_f64() * 1.0e6,
            paged_time.as_secs_f64() * 1.0e6,
            paged_time.as_secs_f64() / contiguous_time.as_secs_f64(),
        );
        assert!(
            paged_time.as_secs_f64() / contiguous_time.as_secs_f64() <= super::common::D5_RATIO
        );
    }
}
