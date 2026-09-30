// Copyright © 2026 Luiz Ribeiro
// SPDX-License-Identifier: MIT

struct GatherMatmulParams {
    ulong input_offset;
    ulong weight_offset;
    ulong indices_offset;
    ulong output_offset;
    ulong input_row_stride;
    ulong input_inner_stride;
    ulong weight_expert_stride;
    ulong weight_inner_stride;
    ulong weight_column_stride;
    ulong indices_row_stride;
    ulong indices_slot_stride;
    ulong output_row_stride;
    ulong output_slot_stride;
    ulong output_column_stride;
    uint rows;
    uint routes;
    uint inner;
    uint experts;
    uint columns;
    uint padding;
};

float4 gather_load_float4(device const uchar *buffer, ulong index, uint dtype) {
    if (dtype == 0) {
        return *reinterpret_cast<device const float4 *>(buffer + index * 4);
    }
    if (dtype == 1) {
        return float4(*reinterpret_cast<device const half4 *>(buffer + index * 2));
    }
    return float4(*reinterpret_cast<device const bfloat4 *>(buffer + index * 2));
}

kernel void gather_gemv(
    device const uchar *input [[buffer(0)]],
    device const uchar *weights [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant GatherMatmulParams &params [[buffer(4)]],
    device atomic_uint *error_flag [[buffer(5)]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint route = tile.y;
    uint row = route / params.routes;
    uint slot = route % params.routes;
    uint expert = indices[
        params.indices_offset + ulong(row) * params.indices_row_stride +
        ulong(slot) * params.indices_slot_stride];
    bool valid = expert < params.experts;
    if (!valid && lane == 0 && simdgroup == 0) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, expert, memory_order_relaxed);
    }
    uint first_column = tile.x * 16 + uint(simdgroup) * 4;
    float sums[4] = {0.0f};
    if (valid) {
        for (uint inner = uint(lane); inner < params.inner; inner += 32) {
            float activation = load_float(
                input,
                params.input_offset + ulong(row) * params.input_row_stride +
                    ulong(inner) * params.input_inner_stride,
                input0_dtype);
            ulong weight_index = params.weight_offset +
                ulong(expert) * params.weight_expert_stride +
                ulong(inner) * params.weight_inner_stride +
                ulong(first_column) * params.weight_column_stride;
            if (params.weight_column_stride == 1 && first_column + 3 < params.columns &&
                (weight_index & 3) == 0) {
                float4 values = gather_load_float4(weights, weight_index, input1_dtype);
                for (uint item = 0; item < 4; ++item) {
                    sums[item] += activation * values[item];
                }
            } else {
                for (uint item = 0; item < 4; ++item) {
                    uint column = first_column + item;
                    if (column < params.columns) {
                        float weight = load_float(
                            weights,
                            weight_index + ulong(item) * params.weight_column_stride,
                            input1_dtype);
                        sums[item] += activation * weight;
                    }
                }
            }
        }
    }
    for (uint item = 0; item < 4; ++item) {
        sums[item] = simd_sum(sums[item]);
    }
    if (lane == 0) {
        for (uint item = 0; item < 4; ++item) {
            uint column = first_column + item;
            if (column < params.columns) {
                ulong output_index = params.output_offset +
                    ulong(row) * params.output_row_stride +
                    ulong(slot) * params.output_slot_stride +
                    ulong(column) * params.output_column_stride;
                store_float(output, output_index, output_dtype, sums[item]);
            }
        }
    }
}
