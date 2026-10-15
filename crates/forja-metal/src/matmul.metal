// Copyright © 2023-2024 Apple Inc.
// SPDX-License-Identifier: MIT

constant uint align_m [[function_constant(200)]];
constant uint align_n [[function_constant(201)]];
constant uint align_k [[function_constant(202)]];

struct MatmulParams {
    ulong a_offset;
    ulong b_offset;
    ulong d_offset;
    ulong lda;
    ulong ldb;
    ulong ldd;
    ulong batch_stride_a;
    ulong batch_stride_b;
    ulong batch_stride_d;
    uint m;
    uint n;
    uint k;
    uint a_column_major;
    uint b_column_major;
    uint batch;
};

template <int block_rows, int block_columns, int block_inner,
          int simdgroups_rows, int simdgroups_columns>
void steel_gemm(
    device const uchar *a,
    device const uchar *b,
    device uchar *d,
    constant MatmulParams &params,
    FORJA_EPILOGUE_PARAMETERS
    threadgroup float *a_tile,
    threadgroup float *b_tile,
    uint thread_index,
    uint thread_count,
    ushort simdgroup_index,
    ushort lane,
    uint3 tile) {
    uint row_origin = tile.y * block_rows;
    uint column_origin = tile.x * block_columns;
    ulong a_base = params.a_offset + ulong(tile.z) * params.batch_stride_a;
    ulong b_base = params.b_offset + ulong(tile.z) * params.batch_stride_b;
    ulong d_base = params.d_offset + ulong(tile.z) * params.batch_stride_d;
    ulong a_row_stride = params.a_column_major ? 1 : params.lda;
    ulong a_column_stride = params.a_column_major ? params.lda : 1;
    ulong b_row_stride = params.b_column_major ? 1 : params.ldb;
    ulong b_column_stride = params.b_column_major ? params.ldb : 1;
    BlockMMA<
        float,
        block_rows,
        block_columns,
        block_inner,
        simdgroups_rows,
        simdgroups_columns> mma(simdgroup_index, lane);

    for (uint inner_origin = 0; inner_origin < params.k; inner_origin += block_inner) {
        BlockLoader<block_rows, block_inner>::load(
            a, a_base, a_row_stride, a_column_stride, input0_dtype,
            row_origin, inner_origin, params.m, params.k,
            align_m != 0, align_k != 0,
            a_tile, thread_index, thread_count);
        BlockLoader<block_inner, block_columns>::load(
            b, b_base, b_row_stride, b_column_stride, input1_dtype,
            inner_origin, column_origin, params.k, params.n,
            align_k != 0, align_n != 0,
            b_tile, thread_index, thread_count);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.multiply(a_tile, b_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int row_fragment = 0; row_fragment < mma.row_fragments; ++row_fragment) {
        uint row = mma.simdgroup_row * mma.fragment_size + mma.fragment_row +
            row_fragment * mma.fragment_size * simdgroups_rows;
        for (int column_fragment = 0;
             column_fragment < mma.column_fragments;
             ++column_fragment) {
            uint column = mma.simdgroup_column * mma.fragment_size + mma.fragment_column +
                column_fragment * mma.fragment_size * simdgroups_columns;
            int fragment_index = row_fragment * mma.column_fragments + column_fragment;
            for (int element = 0; element < 2; ++element) {
                uint output_row = row_origin + row;
                uint output_column = column_origin + column + element;
                if (output_row < params.m && output_column < params.n) {
                    ulong index = d_base + ulong(output_row) * params.ldd + output_column;
                    float value = forja_epilogue(
                        float(mma.accumulators[fragment_index * 2 + element]), 0.0f,
                        tile.z, output_row, output_column, params.batch, params.m, params.n,
                        FORJA_EPILOGUE_VALUES);
                    store_float(d, index, output_dtype, value);
                }
            }
        }
    }
}
#define DEFINE_STEEL_GEMM(name, bm, bn, bk, wm, wn)                         \
kernel void name(                                                            \
    device const uchar *a [[buffer(0)]],                                     \
    device const uchar *b [[buffer(1)]],                                     \
    device uchar *d [[buffer(2)]],                                           \
    constant MatmulParams &params [[buffer(3)]],                             \
    FORJA_EPILOGUE_ARGUMENTS                                                 \
    uint thread_index [[thread_index_in_threadgroup]],                       \
    uint3 threadgroup_size [[threads_per_threadgroup]],                     \
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],               \
    ushort lane [[thread_index_in_simdgroup]],                               \
    uint3 tile [[threadgroup_position_in_grid]]) {                           \
    threadgroup float a_tile[bm * bk];                                       \
    threadgroup float b_tile[bk * bn];                                       \
    steel_gemm<bm, bn, bk, wm, wn>(                                         \
        a, b, d, params, FORJA_EPILOGUE_VALUES, a_tile, b_tile, thread_index, \
        threadgroup_size.x * threadgroup_size.y * threadgroup_size.z,       \
        simdgroup_index, lane, tile);                                        \
}

DEFINE_STEEL_GEMM(steel_gemm_64_64_16_2_2, 64, 64, 16, 2, 2)
DEFINE_STEEL_GEMM(steel_gemm_64_64_16_1_2, 64, 64, 16, 1, 2)
DEFINE_STEEL_GEMM(steel_gemm_64_32_32_2_2, 64, 32, 32, 2, 2)
DEFINE_STEEL_GEMM(steel_gemm_32_64_16_1_2, 32, 64, 16, 1, 2)

kernel void gemv(
    device const uchar *a [[buffer(0)]],
    device const uchar *b [[buffer(1)]],
    device uchar *d [[buffer(2)]],
    constant MatmulParams &params [[buffer(3)]],
    FORJA_EPILOGUE_ARGUMENTS
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tile [[threadgroup_position_in_grid]]) {
    ulong a_base = params.a_offset + ulong(tile.z) * params.batch_stride_a;
    ulong b_base = params.b_offset + ulong(tile.z) * params.batch_stride_b;
    float sums[4] = {0.0f};
    for (uint inner = lane; inner < params.k; inner += 32) {
        ulong a_index = params.a_column_major
            ? ulong(inner) * params.lda
            : ulong(inner);
        float a_value = load_float(a, a_base + a_index, input0_dtype);
        for (uint element = 0; element < 4; ++element) {
            uint column = tile.x * 4 + element;
            if (column < params.n) {
                ulong b_index = params.b_column_major
                    ? ulong(inner) + ulong(column) * params.ldb
                    : ulong(inner) * params.ldb + ulong(column);
                sums[element] += a_value * load_float(
                    b, b_base + b_index, input1_dtype);
            }
        }
    }
    for (uint element = 0; element < 4; ++element) {
        sums[element] = simd_sum(sums[element]);
    }
    if (lane == 0) {
        ulong d_base = params.d_offset + ulong(tile.z) * params.batch_stride_d;
        for (uint element = 0; element < 4; ++element) {
            uint column = tile.x * 4 + element;
            if (column < params.n) {
                float value = forja_epilogue(
                    sums[element], 0.0f,
                    tile.z, 0, column, params.batch, params.m, params.n,
                    FORJA_EPILOGUE_VALUES);
                store_float(d, d_base + ulong(column), output_dtype, value);
            }
        }
    }
}

kernel void gemv_transposed(
    device const uchar *a [[buffer(0)]],
    device const uchar *b [[buffer(1)]],
    device uchar *d [[buffer(2)]],
    constant MatmulParams &params [[buffer(3)]],
    FORJA_EPILOGUE_ARGUMENTS
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tile [[threadgroup_position_in_grid]]) {
    ulong a_base = params.a_offset + ulong(tile.z) * params.batch_stride_a;
    ulong b_base = params.b_offset + ulong(tile.z) * params.batch_stride_b;
    uint first_column = tile.x * 32 + uint(simdgroup) * 4;
    float sums[4] = {0.0f};
    for (uint inner = uint(lane) * 4; inner < params.k; inner += 128) {
        float inputs[4];
        for (uint element = 0; element < 4; ++element) {
            uint index = inner + element;
            inputs[element] = index < params.k
                ? load_float(a, a_base + index, input0_dtype)
                : 0.0f;
        }
        for (uint row = 0; row < 4; ++row) {
            uint column = first_column + row;
            if (column < params.n) {
                ulong weight = b_base + ulong(column) * params.ldb + inner;
                for (uint element = 0; element < 4; ++element) {
                    if (inner + element < params.k) {
                        sums[row] += inputs[element] * load_float(
                            b, weight + element, input1_dtype);
                    }
                }
            }
        }
    }
    for (uint row = 0; row < 4; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    if (lane == 0) {
        ulong d_base = params.d_offset + ulong(tile.z) * params.batch_stride_d;
        for (uint row = 0; row < 4; ++row) {
            uint column = first_column + row;
            if (column < params.n) {
                float value = forja_epilogue(
                    sums[row], 0.0f,
                    tile.z, 0, column, params.batch, params.m, params.n,
                    FORJA_EPILOGUE_VALUES);
                store_float(d, d_base + column, output_dtype, value);
            }
        }
    }
}
