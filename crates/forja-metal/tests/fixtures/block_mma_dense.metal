// Copyright © 2026 Luiz Ribeiro
// SPDX-License-Identifier: MIT

kernel void steel_gemm_64_32_32_2_2(
    device const uchar *a [[buffer(0)]],
    device const uchar *b [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant MatmulParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr bool transposed = TEST_TRANSPOSED;
    threadgroup float left[64 * 32];
    threadgroup float right[32 * 32];
    BlockMMA<float, 64, 32, 32, 2, 2, 32, 32, transposed> mma(simdgroup, lane);
    for (uint origin = 0; origin < params.k; origin += 32) {
        for (uint i = thread_index; i < 64 * 32; i += 128) {
            left[i] = load_float(a, ulong(i / 32) * params.lda + origin + i % 32, input0_dtype);
        }
        for (uint i = thread_index; i < 32 * 32; i += 128) {
            uint k = i / 32, n = i % 32;
            right[transposed ? n * 32 + k : i] =
                load_float(b, ulong(origin + k) * params.ldb + n, input1_dtype);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.multiply(left, right);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mma.store(output, 0, params.ldd, 0, 0, params.m, params.n, output_dtype);
}
