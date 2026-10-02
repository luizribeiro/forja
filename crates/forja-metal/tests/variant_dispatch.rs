//! End-to-end checks for explicitly pinned Metal algorithms.

#![cfg(target_os = "macos")]

use forja_core::{Backend, BackendDispatchData, CommandList, DType, Op, Submission};
use forja_metal::MetalBackend;

#[test]
fn profiled_dispatch_reports_the_pinned_steel_kernel() {
    let backend = MetalBackend::new().unwrap();
    let left = backend.alloc(DType::F32, &[7, 33]).unwrap();
    let right = backend.alloc(DType::F32, &[33, 7]).unwrap();
    let output = backend.alloc(DType::F32, &[7, 7]).unwrap();
    let variant = backend
        .validate_variant_for_op(
            Op::Matmul,
            &[&left, &right],
            &[&output],
            "matmul.steel-64x64x16-2x2",
        )
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::Matmul, &[&left, &right], &output)
        .unwrap();
    assert!(commands.set_last_backend_data(BackendDispatchData::new(variant)));

    let submission = backend.submit_profiled(commands).unwrap();
    submission.wait().unwrap();
    let profile = submission.profile().unwrap();
    assert_eq!(profile.per_dispatch[0].kernel, "steel_gemm_64_64_16_2_2");
}
