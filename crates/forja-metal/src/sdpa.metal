// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

struct SdpaMaskParams {
    float scale;
    uint query_length;
    uint key_length;
    uint query_start;
    uint causal;
    uint score_count;
};

kernel void sdpa_scale_mask(
    device float *scores [[buffer(0)]],
    constant SdpaMaskParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.score_count) {
        return;
    }
    uint query = (index / params.key_length) % params.query_length;
    uint key = index % params.key_length;
    if (params.causal != 0 && key > params.query_start + query) {
        scores[index] = -INFINITY;
    } else {
        scores[index] *= params.scale;
    }
}

struct SdpaParams {
    ulong q_offset;
    ulong k_offset;
    ulong v_offset;
    ulong o_offset;
    ulong q_strides[3];
    ulong k_strides[3];
    ulong v_strides[3];
    ulong o_strides[3];
    float scale;
    uint query_heads;
    uint query_length;
    uint width;
    uint kv_heads;
    uint key_length;
    uint value_width;
    uint heads_per_group;
    uint query_start;
    uint causal;
    uint blocks;
};

float sdpa_dot(
    device const uchar *query,
    device const uchar *key,
    constant SdpaParams &params,
    uint head,
    uint query_row,
    uint key_row) {
    uint kv_head = head / params.heads_per_group;
    ulong q_base = params.q_offset + ulong(head) * params.q_strides[0]
        + ulong(query_row) * params.q_strides[1];
    ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
        + ulong(key_row) * params.k_strides[1];
    float sum = 0.0f;
    for (uint column = 0; column < params.width; ++column) {
        sum += load_float(query, q_base + ulong(column) * params.q_strides[2], input0_dtype)
            * load_float(key, k_base + ulong(column) * params.k_strides[2], input1_dtype);
    }
    return sum * params.scale;
}

float sdpa_reduce_max(
    float value,
    uint lane,
    uint width,
    threadgroup float *partial) {
    partial[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] = max(partial[lane], partial[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

float sdpa_reduce_sum(
    float value,
    uint lane,
    uint width,
    threadgroup float *partial) {
    partial[lane] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

kernel void steel_attention(
    device const uchar *query [[buffer(0)]],
    device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant SdpaParams &params [[buffer(4)]],
    uint3 position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float scores[1024];
    threadgroup float partial[256];
    uint head = position.x;
    uint first_query = position.y * 8;
    uint group_width = threads.x;
    uint kv_head = head / params.heads_per_group;
    for (uint tile_row = 0; tile_row < 8; ++tile_row) {
        uint query_row = first_query + tile_row;
        if (query_row >= params.query_length) {
            return;
        }
        float local_max = -INFINITY;
        for (uint key_row = lane; key_row < params.key_length; key_row += group_width) {
            float score = params.causal != 0 && key_row > params.query_start + query_row
                ? -INFINITY
                : sdpa_dot(query, key, params, head, query_row, key_row);
            scores[key_row] = score;
            local_max = max(local_max, score);
        }
        float maximum = sdpa_reduce_max(local_max, lane, group_width, partial);
        float local_sum = 0.0f;
        for (uint key_row = lane; key_row < params.key_length; key_row += group_width) {
            float weight = scores[key_row] == -INFINITY
                ? 0.0f
                : exp(scores[key_row] - maximum);
            scores[key_row] = weight;
            local_sum += weight;
        }
        float denominator = sdpa_reduce_sum(local_sum, lane, group_width, partial);
        for (uint column = lane; column < params.value_width; column += group_width) {
            float sum = 0.0f;
            for (uint key_row = 0; key_row < params.key_length; ++key_row) {
                ulong v_index = params.v_offset + ulong(kv_head) * params.v_strides[0]
                    + ulong(key_row) * params.v_strides[1]
                    + ulong(column) * params.v_strides[2];
                sum += scores[key_row] * load_float(value, v_index, input2_dtype);
            }
            ulong o_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(query_row) * params.o_strides[1]
                + ulong(column) * params.o_strides[2];
            store_float(output, o_index, output_dtype, sum / denominator);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
