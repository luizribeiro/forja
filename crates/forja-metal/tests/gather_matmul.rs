//! Differential coverage for gathered dense matrix multiplication.

use forja_core::{Backend, BackendError, CommandList, DType, Op, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree};

#[test]
fn olmoe_gate_and_up_projection_shape_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, DType::BF16, 1, 8, 8, 2048, 1024);
}

#[test]
fn olmoe_down_projection_shape_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, DType::BF16, 1, 8, 8, 1024, 2048);
}

#[test]
fn gathered_matmul_supports_float_dtypes_and_duplicate_experts() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        assert_case(&cpu, &metal, dtype, 7, 3, 5, 33, 17);
    }
}

#[test]
fn dense_gather_reports_gpu_written_out_of_range_expert() {
    let metal = MetalBackend::new().unwrap();
    let input = metal.alloc(DType::F32, &[1, 7]).unwrap();
    let weights = metal.alloc(DType::F32, &[3, 7, 2]).unwrap();
    let source = metal.alloc(DType::U32, &[1, 2]).unwrap();
    metal
        .write(
            &source,
            &[99_u32.to_le_bytes(), 7_u32.to_le_bytes()].concat(),
        )
        .unwrap();
    let indices = metal.alloc(DType::U32, &[1, 2]).unwrap();
    let output = metal.alloc(DType::F32, &[1, 2, 2]).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    commands
        .dispatch(Op::GatherMatmul, &[&input, &weights, &indices], &output)
        .unwrap();
    let error = metal.submit(commands).unwrap().wait();
    assert!(matches!(
        error,
        Err(BackendError::DeviceErrorFlag {
            dispatch: 1,
            operation: "gather-matmul",
            flag: 1,
            value: 7,
            ..
        })
    ));
}

#[allow(clippy::too_many_arguments)]
fn assert_case(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    dtype: DType,
    rows: u32,
    routes: u32,
    experts: u32,
    inner: u32,
    columns: u32,
) {
    let selected = (0..rows * routes)
        .map(|slot| {
            if (slot / 2).is_multiple_of(2) {
                0
            } else {
                experts - 1
            }
        })
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(dtype, &[experts, inner, columns]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, routes, columns]);
    assert_backends_agree(cpu, metal, Op::GatherMatmul, &inputs, &output).unwrap();
}
