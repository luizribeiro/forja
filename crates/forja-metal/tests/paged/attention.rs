#![allow(clippy::cast_possible_truncation)]

use super::{
    Backend, BackendDispatchData, CommandList, DType, Op, Submission, decode_f32, encode,
    generation_bytes, table_bytes,
};

fn pattern(token: u32, head: u32, column: u32, value: bool, group: u32) -> f32 {
    let seed = token * 13 + head * 7 + column * if value { 11 } else { 3 };
    (seed % 97) as f32 / 100.0 - if value { 0.125 } else { 0.375 }
        + group as f32 * if value { 0.5 } else { 0.125 }
}

fn query(head: u32, column: u32) -> f32 {
    ((head * 5 + column * 7) % 71) as f32 / 128.0 - 0.25
}

// Two distinct page populations detect long-address truncation while bounding the oracle.
fn reference(width: u32, length: u32) -> Vec<f32> {
    let split = length.div_ceil(64) / 2 * 64;
    let lengths = [split, length - split];
    let mut output = Vec::new();
    for head in 0..24 {
        let scores = (0..2)
            .flat_map(|group| {
                (0..64).map(move |token| {
                    let count = lengths[usize::try_from(group).unwrap()] / 64
                        + u32::from(token < lengths[usize::try_from(group).unwrap()] % 64);
                    let dot = (0..width)
                        .map(|column| {
                            f64::from(query(head, column))
                                * f64::from(pattern(token, head / 6, column, false, group))
                        })
                        .sum::<f64>()
                        / f64::from(width).sqrt();
                    (group, token, count, dot)
                })
            })
            .collect::<Vec<_>>();
        let max = scores
            .iter()
            .filter(|(_, _, count, _)| *count > 0)
            .map(|(_, _, _, score)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|(group, token, count, score)| {
                (*group, *token, f64::from(*count) * (*score - max).exp())
            })
            .collect::<Vec<_>>();
        let denominator = weights.iter().map(|(_, _, weight)| weight).sum::<f64>();
        for column in 0..width {
            output.push(
                (weights
                    .iter()
                    .map(|(group, token, weight)| {
                        weight * f64::from(pattern(*token, head / 6, column, true, *group))
                    })
                    .sum::<f64>()
                    / denominator) as f32,
            );
        }
    }
    output
}

#[allow(clippy::too_many_lines)]
fn attend(width: u32, length: u32, rows: u32, dtype: DType, variant: &str) -> Vec<f32> {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let uses_pages = variant.starts_with("paged-");
    let pages = length.div_ceil(64);
    let query_values = (0..24)
        .flat_map(|head| {
            (0..rows).flat_map(move |_| (0..width).map(move |column| query(head, column)))
        })
        .collect::<Vec<_>>();
    let query = backend.alloc(DType::F32, &[24, rows, width]).unwrap();
    backend
        .write(&query, &encode(&query_values, DType::F32))
        .unwrap();
    let mut pools = Vec::new();
    for value in [false, true] {
        let page_bytes = |group| {
            let page = (0..64)
                .flat_map(|token| {
                    (0..4).flat_map(move |head| {
                        (0..width).map(move |column| pattern(token, head, column, value, group))
                    })
                })
                .collect::<Vec<_>>();
            encode(&page, dtype)
        };
        let split = pages / 2;
        let groups = if uses_pages {
            [(1, pages - split), (0, split)]
        } else {
            [(0, split), (1, pages - split)]
        };
        let bytes = groups
            .into_iter()
            .flat_map(|(group, count)| page_bytes(group).repeat(usize::try_from(count).unwrap()))
            .collect::<Vec<_>>();
        let tensor = backend.alloc(dtype, &[pages, 64, 4, width]).unwrap();
        backend.write(&tensor, &bytes).unwrap();
        let tensor = if uses_pages {
            tensor
        } else {
            let tensor = backend
                .view(
                    &tensor,
                    forja_core::ViewOp::Reshape(vec![pages * 64, 4, width]),
                )
                .unwrap();
            let tensor = backend
                .view(&tensor, forja_core::ViewOp::Permute(vec![1, 0, 2]))
                .unwrap();
            backend
                .view(
                    &tensor,
                    forja_core::ViewOp::Slice(vec![
                        forja_core::Slice::new(0, 4, 1).unwrap(),
                        forja_core::Slice::new(0, length, 1).unwrap(),
                        forja_core::Slice::new(0, width, 1).unwrap(),
                    ]),
                )
                .unwrap()
        };
        pools.push(tensor);
    }
    let table = backend.alloc(DType::U32, &[pages, 2]).unwrap();
    let entries = (0..pages).rev().map(|page| (page, 1)).collect::<Vec<_>>();
    backend.write(&table, &table_bytes(&entries)).unwrap();
    let generations = backend.alloc(DType::U32, &[pages]).unwrap();
    backend
        .write(
            &generations,
            &generation_bytes(&vec![1; usize::try_from(pages).unwrap()]),
        )
        .unwrap();
    let activity = backend.alloc(DType::U32, &[1]).unwrap();
    backend.write(&activity, &1_u32.to_le_bytes()).unwrap();
    let output = backend.alloc(DType::F32, &[24, rows, width]).unwrap();
    let mut commands = CommandList::new();
    if uses_pages {
        commands
            .dispatch(
                Op::PagedSdpa {
                    scale: 1.0 / (width as f32).sqrt(),
                    causal: true,
                    q_start: length - rows,
                    key_length: length,
                    page_tokens: 64,
                },
                &[
                    &query,
                    &pools[0],
                    &pools[1],
                    &table,
                    &generations,
                    &activity,
                ],
                &output,
            )
            .unwrap();
    } else {
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 1.0 / (width as f32).sqrt(),
                    causal: true,
                    q_start: length - rows,
                },
                &[&query, &pools[0], &pools[1]],
                &output,
            )
            .unwrap();
    }
    let pick = backend
        .validate_variant(commands.last_dispatch().unwrap(), variant)
        .unwrap();
    assert!(commands.set_last_backend_data(BackendDispatchData::new(pick)));
    backend.submit(commands).unwrap().wait().unwrap();
    decode_f32(&backend.read(&output).unwrap())
}

#[test]
fn metal_steel_attention_reference_agrees_at_short_boundaries() {
    for width in [64, 128, 256] {
        for prefix in ["sdpa", "paged-sdpa"] {
            let suffix = if width == 256 { "-256" } else { "" };
            let actual = attend(width, 64, 7, DType::F32, &format!("{prefix}.steel{suffix}"));
            let stride = usize::try_from(width).unwrap();
            let last = actual
                .chunks_exact(7 * stride)
                .flat_map(|head| &head[6 * stride..])
                .copied()
                .collect::<Vec<_>>();
            assert!(
                forja_testing::normwise_relative_error(&reference(width, 64), &last)
                    <= forja_testing::dtype_tolerance(DType::F32).unwrap()
            );
        }
    }
}

#[path = "legacy.rs"]
mod legacy;

#[path = "bounded.rs"]
mod bounded;
