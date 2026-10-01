//! Differential coverage for fused Q/K normalization, rotary embedding, and cache writes.

use forja_core::{DType, Op, Slice};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree_many};

#[test]
fn fused_qkv_cache_write_matches_cpu_at_a_nonzero_slot() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    let inputs = [
        TensorSpec::contiguous(DType::F32, &[1, 40, 128]),
        TensorSpec::contiguous(DType::F32, &[36, 128]),
        TensorSpec::initialized(DType::F32, &[1], 33.0_f32.to_le_bytes().to_vec()),
    ];
    let cache_slice = [
        Slice::new(0, 4, 1).unwrap(),
        Slice::new(33, 1, 1).unwrap(),
        Slice::new(0, 128, 1).unwrap(),
    ];
    let outputs = [
        TensorSpec::contiguous(DType::F32, &[32, 1, 128]),
        TensorSpec::sliced(DType::F32, &[4, 4096, 128], &cache_slice),
        TensorSpec::sliced(DType::F32, &[4, 4096, 128], &cache_slice),
    ];
    assert_backends_agree_many(
        &reference,
        &candidate,
        Op::QkvRopeCache {
            eps: 1.0e-6,
            theta: 1.0e6,
        },
        &inputs,
        &outputs,
    )
    .unwrap();
}
