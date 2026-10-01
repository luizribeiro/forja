//! Differential coverage for affine-quantized matrix multiplication.

use forja_core::{DType, Op, Slice};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree, assert_backends_agree_many};

#[test]
fn decode_quantized_matmul_matches_cpu_for_formats_and_groups() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for bits in [4, 8] {
        for group_size in [32, 64, 128] {
            for parameter_dtype in [DType::F16, DType::BF16] {
                assert_case(&reference, &candidate, 1, bits, group_size, parameter_dtype);
            }
        }
    }
}

#[test]
fn small_row_quantized_matmul_matches_cpu_at_odd_sizes() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for rows in [7, 33] {
        for bits in [4, 8] {
            for parameter_dtype in [DType::F16, DType::BF16] {
                assert_case(&reference, &candidate, rows, bits, 64, parameter_dtype);
            }
        }
    }
}

#[test]
fn tiled_qwen_projection_shape_matches_cpu() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_projection_case(&reference, &candidate, 16, 1024, 3072, dtype);
    }
}

#[test]
fn qwen_router_shape_matches_cpu() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_decode_projection_case(&reference, &candidate, dtype);
    }
}

#[test]
fn fused_qwen_router_matches_cpu() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for rows in [1, 7] {
        let inputs = [
            TensorSpec::contiguous(DType::F32, &[rows, 2048]),
            TensorSpec::contiguous(DType::U32, &[128, 512]),
            TensorSpec::contiguous(DType::BF16, &[128, 32]),
            TensorSpec::contiguous(DType::BF16, &[128, 32]),
        ];
        let outputs = [
            TensorSpec::contiguous(DType::F32, &[rows, 128]),
            TensorSpec::contiguous(DType::F32, &[rows, 8]),
            TensorSpec::contiguous(DType::U32, &[rows, 8]),
        ];
        assert_backends_agree_many(
            &reference,
            &candidate,
            Op::QuantizedRouter {
                group_size: 64,
                k: 8,
                normalize: true,
            },
            &inputs,
            &outputs,
        )
        .unwrap();
    }
}

#[test]
fn fused_qwen_router_matches_cpu_for_partial_top_k_ties_and_nan() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    let inner = 256_usize;
    let columns = 128_usize;
    let groups = inner / 64;
    let input = TensorSpec::initialized(
        DType::F32,
        &[1, u32::try_from(inner).unwrap()],
        std::iter::repeat_n(1.0_f32, inner)
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    let packed = TensorSpec::initialized(
        DType::U32,
        &[
            u32::try_from(columns).unwrap(),
            u32::try_from(inner / 4).unwrap(),
        ],
        vec![0; columns * inner],
    );
    let scales = TensorSpec::initialized(
        DType::BF16,
        &[
            u32::try_from(columns).unwrap(),
            u32::try_from(groups).unwrap(),
        ],
        vec![0; columns * groups * 2],
    );
    for nan_expert in [None, Some(17_usize)] {
        let mut biases = vec![0_u8; columns * groups * 2];
        if let Some(expert) = nan_expert {
            for group in 0..groups {
                let offset = (expert * groups + group) * 2;
                biases[offset..offset + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
            }
        }
        let inputs = [
            input.clone(),
            packed.clone(),
            scales.clone(),
            TensorSpec::initialized(
                DType::BF16,
                &[
                    u32::try_from(columns).unwrap(),
                    u32::try_from(groups).unwrap(),
                ],
                biases,
            ),
        ];
        for k in [1, 3, 7, 8] {
            let outputs = [
                TensorSpec::contiguous(DType::F32, &[1, 128]),
                TensorSpec::contiguous(DType::F32, &[1, k]),
                TensorSpec::contiguous(DType::U32, &[1, k]),
            ];
            assert_backends_agree_many(
                &reference,
                &candidate,
                Op::QuantizedRouter {
                    group_size: 64,
                    k,
                    normalize: true,
                },
                &inputs,
                &outputs,
            )
            .unwrap();
        }
    }
}

#[test]
fn quantized_matmul_matches_cpu_for_strided_views() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    let rows = 7;
    let inner = 256;
    let columns = 33;
    let packed_width = 32;
    let groups = 4;
    let row_slice = Slice::new(0, rows, 1).unwrap();
    let inputs = [
        TensorSpec::sliced(
            DType::BF16,
            &[rows, inner + 1],
            &[row_slice, Slice::new(1, inner, 1).unwrap()],
        ),
        TensorSpec::sliced(
            DType::U32,
            &[columns, packed_width + 1],
            &[
                Slice::new(0, columns, 1).unwrap(),
                Slice::new(1, packed_width, 1).unwrap(),
            ],
        ),
        TensorSpec::sliced(
            DType::BF16,
            &[columns, groups + 1],
            &[
                Slice::new(0, columns, 1).unwrap(),
                Slice::new(1, groups, 1).unwrap(),
            ],
        ),
        TensorSpec::sliced(
            DType::BF16,
            &[columns, groups + 1],
            &[
                Slice::new(0, columns, 1).unwrap(),
                Slice::new(1, groups, 1).unwrap(),
            ],
        ),
    ];
    let output = TensorSpec::sliced(
        DType::BF16,
        &[rows, columns + 1],
        &[row_slice, Slice::new(1, columns, 1).unwrap()],
    );
    assert_backends_agree(
        &reference,
        &candidate,
        Op::QuantMatmul {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &output,
    )
    .unwrap();
}

fn assert_case(
    reference: &CpuBackend,
    candidate: &MetalBackend,
    rows: u32,
    bits: u8,
    group_size: u32,
    parameter_dtype: DType,
) {
    let inner = 256;
    let columns = 33;
    let packed_width = inner * u32::from(bits) / 32;
    let groups = inner / group_size;
    let inputs = [
        TensorSpec::contiguous(DType::BF16, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[columns, packed_width]),
        TensorSpec::contiguous(parameter_dtype, &[columns, groups]),
        TensorSpec::contiguous(parameter_dtype, &[columns, groups]),
    ];
    let output = TensorSpec::contiguous(DType::BF16, &[rows, columns]);
    assert_backends_agree(
        reference,
        candidate,
        Op::QuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}

fn assert_projection_case(
    reference: &CpuBackend,
    candidate: &MetalBackend,
    rows: u32,
    inner: u32,
    columns: u32,
    dtype: DType,
) {
    let bits = 4;
    let group_size = 64;
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[columns, inner / 8]),
        TensorSpec::contiguous(DType::BF16, &[columns, inner / group_size]),
        TensorSpec::contiguous(DType::BF16, &[columns, inner / group_size]),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, columns]);
    assert_backends_agree(
        reference,
        candidate,
        Op::QuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}

fn assert_decode_projection_case(reference: &CpuBackend, candidate: &MetalBackend, dtype: DType) {
    let inner = 2048;
    let columns = 128;
    let bits = 8;
    let group_size = 64;
    let inputs = [
        TensorSpec::contiguous(dtype, &[1, inner]),
        TensorSpec::contiguous(DType::U32, &[columns, inner / 4]),
        TensorSpec::contiguous(DType::BF16, &[columns, inner / group_size]),
        TensorSpec::contiguous(DType::BF16, &[columns, inner / group_size]),
    ];
    let output = TensorSpec::contiguous(dtype, &[1, columns]);
    assert_backends_agree(
        reference,
        candidate,
        Op::QuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}
