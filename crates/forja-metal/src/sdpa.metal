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
