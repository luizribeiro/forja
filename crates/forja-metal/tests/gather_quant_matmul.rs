//! Differential coverage for gathered affine-quantized matrix multiplication.

use forja_core::{Backend, BackendError, CommandList, DType, Op, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree};

#[test]
fn qwen_expert_projection_shapes_match_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, 1, 8, 128, 2048, 768);
    assert_case(&cpu, &metal, 1, 8, 128, 768, 2048);
}

#[test]
fn small_row_gathered_quantized_matmul_matches_cpu_with_duplicates() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, 7, 3, 5, 64, 33);
    assert_case(&cpu, &metal, 33, 3, 5, 64, 33);
}

#[test]
fn grouped_qwen_projection_shape_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, 16, 8, 128, 1024, 128);
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
    let rows = 16;
    let routes = 8;
    let experts = 128;
    let inner = 1024;
    let columns = 128;
    let selected = (0..rows * routes)
        .map(|route| route % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(DType::BF16, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(DType::BF16, &[rows, routes, columns]);
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

fn assert_zero_output_after_error(metal: &MetalBackend, output: &forja_core::Tensor, len: usize) {
    let _completion_error = metal.read(output);
    assert_eq!(metal.read(output).unwrap(), vec![0; len]);
}

fn assert_case(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    rows: u32,
    routes: u32,
    experts: u32,
    inner: u32,
    columns: u32,
) {
    let bits = 4;
    let group_size = 64;
    let packed_width = inner * u32::from(bits) / 32;
    let groups = inner / group_size;
    let selected = (0..rows * routes)
        .map(|slot| (slot / 2) % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(DType::BF16, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, packed_width]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(DType::BF16, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}
