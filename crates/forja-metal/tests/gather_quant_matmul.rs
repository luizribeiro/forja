//! Differential coverage for gathered affine-quantized matrix multiplication.

use forja_core::{Backend, BackendError, CommandList, DType, Op, Slice, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree, assert_outputs_agree};

#[test]
fn qwen_expert_projection_shapes_match_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [1, 8, 128, 2048, 768], dtype);
        assert_case(&cpu, &metal, [1, 8, 128, 768, 2048], dtype);
    }
}

#[test]
fn qwen_combined_down_projection_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_combine_case(&cpu, &metal, dtype);
    }
}

#[test]
fn small_row_gathered_quantized_matmul_matches_cpu_with_duplicates() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, [7, 3, 5, 64, 33], DType::BF16);
    assert_case(&cpu, &metal, [33, 3, 5, 64, 33], DType::BF16);
}

#[test]
fn grouped_qwen_projection_shape_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [16, 8, 128, 1024, 128], dtype);
    }
}

#[test]
fn fused_gathered_gate_and_up_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let selected = [7_u32, 0, 7]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(DType::BF16, &[1, 64]),
        TensorSpec::contiguous(DType::U32, &[8, 33, 8]),
        TensorSpec::contiguous(DType::F16, &[8, 33, 1]),
        TensorSpec::contiguous(DType::F16, &[8, 33, 1]),
        TensorSpec::contiguous(DType::U32, &[8, 33, 8]),
        TensorSpec::contiguous(DType::F16, &[8, 33, 1]),
        TensorSpec::contiguous(DType::F16, &[8, 33, 1]),
        TensorSpec::initialized(DType::U32, &[1, 3], selected),
    ];
    let output = TensorSpec::contiguous(DType::BF16, &[1, 3, 33]);
    assert_backends_agree(
        &cpu,
        &metal,
        Op::GatherQuantSiluMul {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &output,
    )
    .unwrap();
}

#[test]
fn grouped_fused_gate_and_up_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_silu_case(&cpu, &metal, [16, 8, 128, 1024, 128], dtype);
    }
}

#[test]
fn qwen_fused_gate_and_up_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_silu_case(&cpu, &metal, [1, 8, 128, 2048, 768], dtype);
    }
}

#[test]
fn grouped_quantized_ops_match_cpu_for_strided_f32_inputs() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let rows = 16;
    let inner = 64;
    let row_slice = Slice::new(0, rows, 1).unwrap();
    let activations = [
        TensorSpec::sliced_broadcast(
            DType::F32,
            &[rows, 4],
            &[row_slice, Slice::new(0, 1, 1).unwrap()],
            &[rows, inner],
        ),
        TensorSpec::sliced(
            DType::F32,
            &[rows, inner * 2],
            &[row_slice, Slice::new(0, inner, 2).unwrap()],
        ),
    ];
    for activation in activations {
        assert_grouped_strided_case(&cpu, &metal, activation);
    }
}

fn assert_silu_case(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    [rows, routes, experts, inner, columns]: [u32; 5],
    dtype: DType,
) {
    let selected = (0..rows * routes)
        .map(|route| (route / 2) % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantSiluMul {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &output,
    )
    .unwrap();
}

fn assert_grouped_strided_case(cpu: &CpuBackend, metal: &MetalBackend, activation: TensorSpec) {
    let rows = 16;
    let routes = 4;
    let experts = 8;
    let inner = 64;
    let columns = 33;
    let selected = (0..rows * routes)
        .map(|route| route % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let packed = TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]);
    let scales = TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]);
    let biases = TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]);
    let indices = TensorSpec::initialized(DType::U32, &[rows, routes], selected);
    let output = TensorSpec::contiguous(DType::F32, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        },
        &[
            activation.clone(),
            packed.clone(),
            scales.clone(),
            biases.clone(),
            indices.clone(),
        ],
        &output,
    )
    .unwrap();
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantSiluMul {
            bits: 4,
            group_size: 64,
        },
        &[
            activation,
            packed.clone(),
            scales.clone(),
            biases.clone(),
            packed,
            scales,
            biases,
            indices,
        ],
        &output,
    )
    .unwrap();
}

#[test]
fn grouped_matmul_zeros_gpu_written_out_of_range_routes() {
    let metal = MetalBackend::new().unwrap();
    let input = metal.alloc(DType::F32, &[33, 64]).unwrap();
    let packed = metal.alloc(DType::U32, &[3, 2, 8]).unwrap();
    let scales = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let biases = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let source = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let mut selected = vec![0_u32; 66];
    selected[..2].copy_from_slice(&[99, 7]);
    metal
        .write(
            &source,
            &selected
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let indices = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let output = metal.alloc(DType::F32, &[33, 2, 2]).unwrap();
    metal.write(&output, &[0xa5; 33 * 2 * 2 * 4]).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    commands
        .dispatch(
            Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            },
            &[&input, &packed, &scales, &biases, &indices],
            &output,
        )
        .unwrap();
    let error = metal.submit(commands).unwrap().wait();
    assert_eq!(error, Err(BackendError::IndexOutOfRange { index: 7 }));
    assert_zero_output_after_error(&metal, &output, 33 * 2 * 2 * 4);
}

#[test]
fn grouped_silu_zeros_gpu_written_out_of_range_routes() {
    let metal = MetalBackend::new().unwrap();
    let input = metal.alloc(DType::F32, &[33, 64]).unwrap();
    let packed = metal.alloc(DType::U32, &[3, 2, 8]).unwrap();
    let scales = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let biases = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let source = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let mut selected = vec![0_u32; 66];
    selected[..2].copy_from_slice(&[99, 7]);
    metal
        .write(
            &source,
            &selected
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let indices = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let output = metal.alloc(DType::F32, &[33, 2, 2]).unwrap();
    metal.write(&output, &[0xa5; 33 * 2 * 2 * 4]).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    commands
        .dispatch(
            Op::GatherQuantSiluMul {
                bits: 4,
                group_size: 64,
            },
            &[
                &input, &packed, &scales, &biases, &packed, &scales, &biases, &indices,
            ],
            &output,
        )
        .unwrap();
    let error = metal.submit(commands).unwrap().wait();
    assert_eq!(error, Err(BackendError::IndexOutOfRange { index: 7 }));
    assert_zero_output_after_error(&metal, &output, 33 * 2 * 2 * 4);
}

#[test]
fn qwen_decode_matmul_zeros_gpu_written_out_of_range_routes() {
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let metal = MetalBackend::new().unwrap();
        let input = metal.alloc(dtype, &[8, 768]).unwrap();
        let packed = metal.alloc(DType::U32, &[128, 2048, 96]).unwrap();
        let scales = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let biases = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let indices = copied_invalid_indices(&metal, &[8, 1]);
        let output = metal.alloc(dtype, &[8, 1, 2048]).unwrap();
        let output_bytes = usize::try_from(8 * 2048 * dtype.byte_size()).unwrap();
        metal.write(&output, &vec![0xa5; output_bytes]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &[&input, &packed, &scales, &biases, &indices],
                &output,
            )
            .unwrap();
        assert_eq!(
            metal.submit(commands).unwrap().wait(),
            Err(BackendError::IndexOutOfRange { index: 129 })
        );
        assert_zero_output_after_error(&metal, &output, output_bytes);
    }
}

#[test]
fn qwen_decode_silu_zeros_gpu_written_out_of_range_routes() {
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let metal = MetalBackend::new().unwrap();
        let input = metal.alloc(dtype, &[1, 2048]).unwrap();
        let packed = metal.alloc(DType::U32, &[128, 768, 256]).unwrap();
        let scales = metal.alloc(DType::F16, &[128, 768, 32]).unwrap();
        let biases = metal.alloc(DType::F16, &[128, 768, 32]).unwrap();
        let indices = copied_invalid_indices(&metal, &[1, 8]);
        let output = metal.alloc(dtype, &[1, 8, 768]).unwrap();
        let output_bytes = usize::try_from(8 * 768 * dtype.byte_size()).unwrap();
        metal.write(&output, &vec![0xa5; output_bytes]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantSiluMul {
                    bits: 4,
                    group_size: 64,
                },
                &[
                    &input, &packed, &scales, &biases, &packed, &scales, &biases, &indices,
                ],
                &output,
            )
            .unwrap();
        assert_eq!(
            metal.submit(commands).unwrap().wait(),
            Err(BackendError::IndexOutOfRange { index: 129 })
        );
        assert_zero_output_after_error(&metal, &output, output_bytes);
    }
}

#[test]
fn qwen_combined_down_projection_zeros_gpu_written_out_of_range_routes() {
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let metal = MetalBackend::new().unwrap();
        let input = metal.alloc(dtype, &[1, 8, 768]).unwrap();
        let packed = metal.alloc(DType::U32, &[128, 2048, 96]).unwrap();
        let scales = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let biases = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let indices = copied_invalid_indices(&metal, &[1, 8]);
        let weights = metal.alloc(dtype, &[1, 8]).unwrap();
        let residual = metal.alloc(dtype, &[1, 2048]).unwrap();
        let output = metal.alloc(dtype, &[1, 2048]).unwrap();
        let output_bytes = usize::try_from(2048 * dtype.byte_size()).unwrap();
        metal.write(&output, &vec![0xa5; output_bytes]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantMatmulCombine {
                    bits: 4,
                    group_size: 64,
                },
                &[
                    &input, &packed, &scales, &biases, &indices, &weights, &residual,
                ],
                &output,
            )
            .unwrap();
        assert_eq!(
            metal.submit(commands).unwrap().wait(),
            Err(BackendError::IndexOutOfRange { index: 129 })
        );
        assert_zero_output_after_error(&metal, &output, output_bytes);
    }
}

#[test]
fn combined_projection_keeps_nonzero_valid_routes_in_a_mixed_row() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let (cpu_error, expected) = run_mixed_combine(&cpu);
    let (metal_error, actual) = run_mixed_combine(&metal);

    assert_eq!(cpu_error, Err(BackendError::IndexOutOfRange { index: 7 }));
    assert_eq!(metal_error, cpu_error);
    assert_outputs_agree(DType::F32, &expected, &actual).unwrap();
}

fn run_mixed_combine<B: Backend>(backend: &B) -> (Result<(), BackendError>, Vec<u8>) {
    let activations = (1_u16..=5)
        .flat_map(|value| std::iter::repeat_n(f32::from(value), 64))
        .collect::<Vec<_>>();
    let packed = (1_u32..=6)
        .flat_map(|value| std::iter::repeat_n(value * 0x1111_1111, 8))
        .collect::<Vec<_>>();
    let input = initialized(backend, DType::F32, &[1, 5, 64], &f32_bytes(&activations));
    let packed = initialized(backend, DType::U32, &[3, 2, 8], &u32_bytes(&packed));
    let scales = initialized(backend, DType::BF16, &[3, 2, 1], &bf16_bytes(&[0.5; 6]));
    let biases = initialized(backend, DType::BF16, &[3, 2, 1], &bf16_bytes(&[0.25; 6]));
    let indices = initialized(backend, DType::U32, &[1, 5], &u32_bytes(&[2, 99, 0, 7, 1]));
    let weights = initialized(
        backend,
        DType::F32,
        &[1, 5],
        &f32_bytes(&[0.5, 1.0, -0.25, 2.0, 0.75]),
    );
    let residual = initialized(backend, DType::F32, &[1, 2], &f32_bytes(&[1.5, -2.0]));
    let output = backend.alloc(DType::F32, &[1, 2]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::GatherQuantMatmulCombine {
                bits: 4,
                group_size: 64,
            },
            &[
                &input, &packed, &scales, &biases, &indices, &weights, &residual,
            ],
            &output,
        )
        .unwrap();
    let error = backend.submit(commands).unwrap().wait();
    let _ = backend.read(&output);
    (error, backend.read(&output).unwrap())
}

fn initialized<B: Backend>(
    backend: &B,
    dtype: DType,
    shape: &[u32],
    bytes: &[u8],
) -> forja_core::Tensor {
    let tensor = backend.alloc(dtype, shape).unwrap();
    backend.write(&tensor, bytes).unwrap();
    tensor
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| u16::try_from(value.to_bits() >> 16).unwrap().to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn copied_invalid_indices(metal: &MetalBackend, shape: &[u32]) -> forja_core::Tensor {
    let source = metal.alloc(DType::U32, shape).unwrap();
    let indices = metal.alloc(DType::U32, shape).unwrap();
    let mut selected = vec![0_u32; usize::try_from(shape.iter().product::<u32>()).unwrap()];
    selected[0] = 129;
    metal
        .write(
            &source,
            &selected
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    metal.submit(commands).unwrap().wait().unwrap();
    indices
}

fn assert_zero_output_after_error(metal: &MetalBackend, output: &forja_core::Tensor, len: usize) {
    let _completion_error = metal.read(output);
    assert_eq!(metal.read(output).unwrap(), vec![0; len]);
}

fn assert_case(cpu: &CpuBackend, metal: &MetalBackend, shape: [u32; 5], dtype: DType) {
    let [rows, routes, experts, inner, columns] = shape;
    let bits = 4;
    let group_size = 64;
    let packed_width = inner * u32::from(bits) / 32;
    let groups = inner / group_size;
    let selected = (0..rows * routes)
        .map(|slot| (slot / 2) % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, packed_width]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}

fn assert_combine_case(cpu: &CpuBackend, metal: &MetalBackend, dtype: DType) {
    let rows = 1;
    let routes = 8;
    let experts = 128;
    let inner = 768;
    let columns = 2048;
    let selected = [7_u32, 0, 7, 1, 1, 2, 2, 7]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, routes, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
        TensorSpec::contiguous(dtype, &[rows, routes]),
        TensorSpec::contiguous(dtype, &[rows, columns]),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmulCombine {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &output,
    )
    .unwrap();
}
