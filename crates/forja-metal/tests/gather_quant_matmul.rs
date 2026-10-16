//! Differential coverage for gathered affine-quantized matrix multiplication.

use forja_core::{Backend, BackendError, CommandList, DType, Op, Slice, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree};

#[test]
fn expert_128_projection_shapes_match_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [1, 8, 128, 2048, 768], dtype);
        assert_case(&cpu, &metal, [1, 8, 128, 768, 2048], dtype);
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
fn grouped_1024_by_128_projection_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [16, 8, 128, 1024, 128], dtype);
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
        &[activation, packed, scales, biases, indices],
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
    assert_error_and_zero_output(
        &metal,
        metal.submit(commands).unwrap(),
        &output,
        33 * 2 * 2 * 4,
        7,
        "gather-quant-matmul",
    );
}

#[test]
fn decode_matmul_zeros_gpu_written_out_of_range_routes() {
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
        assert_error_and_zero_output(
            &metal,
            metal.submit(commands).unwrap(),
            &output,
            output_bytes,
            129,
            "gather-quant-matmul",
        );
    }
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

fn assert_error_and_zero_output(
    metal: &MetalBackend,
    submission: impl Submission,
    output: &forja_core::Tensor,
    len: usize,
    index: u32,
    operation: &'static str,
) {
    assert!(matches!(
        submission.wait(),
        Err(BackendError::DeviceErrorFlag {
            operation: actual_operation,
            flag: 1,
            value,
            ..
        }) if actual_operation == operation && value == index
    ));
    drop(submission);
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
