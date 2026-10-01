//! Differential coverage for affine-quantized matrix multiplication.

use forja_core::{DType, Op, Slice};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree};

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
