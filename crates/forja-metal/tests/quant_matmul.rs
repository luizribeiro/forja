//! Differential coverage for affine-quantized matrix multiplication.

use forja_core::{Backend, BackendDispatchData, CommandList, DType, Op, Slice, Submission, ViewOp};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{
    DeterministicValues, TensorSpec, assert_backends_agree, assert_outputs_agree,
    generated_tensor_bytes,
};

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
fn pinned_decode_batch_variants_match_cpu_for_distinct_rows() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    let mut values = DeterministicValues::new(0x510e_527f_ade6_82d1);
    for rows in [2, 4, 8] {
        assert_pinned_decode_batch(
            &reference,
            &candidate,
            rows,
            4,
            "quant-matmul.row-gemv",
            &mut values,
        );
        assert_pinned_decode_batch(
            &reference,
            &candidate,
            rows,
            8,
            "quant-matmul.q8-fast-gemv",
            &mut values,
        );
    }
}

fn assert_pinned_decode_batch(
    reference: &CpuBackend,
    candidate: &MetalBackend,
    rows: u32,
    bits: u8,
    name: &str,
    values: &mut DeterministicValues,
) {
    let inner = 256;
    let columns = 33;
    let group_size = 64;
    let activations = (0..rows * inner)
        .flat_map(|index| (f32::from(u16::try_from(index).unwrap()) / 257.0 - 4.0).to_le_bytes())
        .collect();
    let specs = [
        TensorSpec::initialized(DType::F32, &[rows, inner], activations),
        TensorSpec::contiguous(DType::U32, &[columns, inner * u32::from(bits) / 32]),
        TensorSpec::contiguous(DType::F16, &[columns, inner / group_size]),
        TensorSpec::contiguous(DType::F16, &[columns, inner / group_size]),
    ];
    let bytes = specs
        .iter()
        .map(|spec| generated_tensor_bytes(spec, values).unwrap())
        .collect::<Vec<_>>();
    let reference_inputs = allocate_inputs(reference, &specs, &bytes);
    let candidate_inputs = allocate_inputs(candidate, &specs, &bytes);
    let reference_output = reference.alloc(DType::F32, &[rows, columns]).unwrap();
    let candidate_output = candidate.alloc(DType::F32, &[rows, columns]).unwrap();
    let op = Op::QuantMatmul { bits, group_size };
    submit_quant(reference, op, &reference_inputs, &reference_output, None);
    let inputs = candidate_inputs.iter().collect::<Vec<_>>();
    let variant = candidate
        .validate_variant_for_op(op, &inputs, &[&candidate_output], name)
        .unwrap();
    submit_quant(
        candidate,
        op,
        &candidate_inputs,
        &candidate_output,
        Some(BackendDispatchData::new(variant)),
    );
    assert_outputs_agree(
        DType::F32,
        &reference.read(&reference_output).unwrap(),
        &candidate.read(&candidate_output).unwrap(),
    )
    .unwrap();
}

fn allocate_inputs<B: Backend>(
    backend: &B,
    specs: &[TensorSpec],
    bytes: &[Vec<u8>],
) -> Vec<forja_core::Tensor> {
    specs
        .iter()
        .zip(bytes)
        .map(|(spec, bytes)| {
            let tensor = backend
                .alloc(spec.dtype(), spec.allocation_shape())
                .unwrap();
            backend.write(&tensor, bytes).unwrap();
            tensor
        })
        .collect()
}

fn submit_quant<B: Backend>(
    backend: &B,
    op: Op,
    inputs: &[forja_core::Tensor],
    output: &forja_core::Tensor,
    variant: Option<BackendDispatchData>,
) {
    let mut commands = CommandList::new();
    let inputs = inputs.iter().collect::<Vec<_>>();
    commands.dispatch(op, &inputs, output).unwrap();
    if let Some(variant) = variant {
        assert!(commands.set_last_backend_data(variant));
    }
    backend.submit(commands).unwrap().wait().unwrap();
}

#[test]
fn tiled_1024_by_3072_projection_matches_cpu() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_projection_case(&reference, &candidate, 16, 1024, 3072, dtype);
    }
}

#[test]
fn quantized_2048_by_128_router_matches_cpu() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_decode_projection_case(&reference, &candidate, dtype);
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

#[test]
#[ignore = "manual decode batch kernel comparison"]
fn metal_decode_batch_quantized_matmul_microbenchmark() {
    let backend = MetalBackend::new().unwrap();
    for (site, inner, columns) in [
        ("qkv", 2_048, 5_120),
        ("o", 4_096, 2_048),
        ("lm-head", 2_048, 151_936),
    ] {
        for rows in [2, 4, 8] {
            let tensors = quant_benchmark_tensors(&backend, rows, inner, columns);
            let gemv = || quant_benchmark_commands(&backend, &tensors, rows, true);
            let small_m = || quant_benchmark_commands(&backend, &tensors, rows, false);
            let (gemv_time, small_m_time) =
                crate::common::comparative_gpu_times(&backend, gemv, small_m);
            eprintln!(
                "quant-batch site={site} rows={rows} row_gemv_us={} small_m_us={} ratio={:.4}",
                gemv_time.as_secs_f64() * 1.0e6,
                small_m_time.as_secs_f64() * 1.0e6,
                small_m_time.as_secs_f64() / gemv_time.as_secs_f64(),
            );
        }
    }
}

fn quant_benchmark_tensors(
    backend: &MetalBackend,
    rows: u32,
    inner: u32,
    columns: u32,
) -> [forja_core::Tensor; 5] {
    [
        backend.alloc(DType::F32, &[rows, inner]).unwrap(),
        backend.alloc(DType::U32, &[columns, inner / 8]).unwrap(),
        backend.alloc(DType::BF16, &[columns, inner / 64]).unwrap(),
        backend.alloc(DType::BF16, &[columns, inner / 64]).unwrap(),
        backend.alloc(DType::F32, &[rows, columns]).unwrap(),
    ]
}

fn quant_benchmark_commands(
    backend: &MetalBackend,
    tensors: &[forja_core::Tensor; 5],
    rows: u32,
    per_row: bool,
) -> CommandList {
    let op = Op::QuantMatmul {
        bits: 4,
        group_size: 64,
    };
    let mut commands = CommandList::new();
    let row_count = if per_row { rows } else { 1 };
    for row in 0..row_count {
        let (input, output) = if per_row {
            let input = backend
                .view(
                    &tensors[0],
                    ViewOp::Slice(vec![
                        Slice::new(row, 1, 1).unwrap(),
                        Slice::new(0, tensors[0].layout().shape()[1], 1).unwrap(),
                    ]),
                )
                .unwrap();
            let output = backend
                .view(
                    &tensors[4],
                    ViewOp::Slice(vec![
                        Slice::new(row, 1, 1).unwrap(),
                        Slice::new(0, tensors[4].layout().shape()[1], 1).unwrap(),
                    ]),
                )
                .unwrap();
            (input, output)
        } else {
            (tensors[0].clone(), tensors[4].clone())
        };
        let inputs = [&input, &tensors[1], &tensors[2], &tensors[3]];
        commands.dispatch(op, &inputs, &output).unwrap();
        let variant = backend
            .validate_variant_for_op(
                op,
                &inputs,
                &[&output],
                if per_row {
                    "quant-matmul.gemv"
                } else {
                    "quant-matmul.small-m"
                },
            )
            .unwrap();
        assert!(commands.set_last_backend_data(BackendDispatchData::new(variant)));
    }
    commands
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
