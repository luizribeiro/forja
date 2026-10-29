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
        scores[index] = negative_finite_sentinel();
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

constant uint align_K [[function_constant(211)]];
constant uint do_causal [[function_constant(212)]];

struct ContiguousSteelAddressing {
    device const uchar *key;
    device const uchar *value;
    constant SdpaParams &params;
    uint kv_head;

    static bool causal() { return do_causal != 0; }

    template <typename T, int BK, int BD, int LD, int THREADS>
    void load_key(uint first_key, uint key_length, threadgroup T *tile, uint thread_index) {
        ulong base = params.k_offset + ulong(kv_head) * params.k_strides[0]
            + ulong(first_key) * params.k_strides[1];
        device const uchar *source = key + base * steel_attention_dtype_size(input1_dtype);
        steel_attention_load_tile<T, BK, BD, 1, LD, THREADS>(source,
            uint(params.k_strides[1]), uint(params.k_strides[2]), input1_dtype,
            key_length - first_key, align_K != 0, base % 4 == 0, tile, thread_index);
    }

    template <typename T, int BK, int BD, int LD, int THREADS>
    void load_value(uint first_key, uint key_length, threadgroup T *tile, uint thread_index) {
        ulong base = params.v_offset + ulong(kv_head) * params.v_strides[0]
            + ulong(first_key) * params.v_strides[1];
        device const uchar *source = value + base * steel_attention_dtype_size(input2_dtype);
        steel_attention_load_tile<T, BK, BD, LD, 1, THREADS>(source,
            uint(params.v_strides[1]), uint(params.v_strides[2]), input2_dtype,
            key_length - first_key, align_K != 0, base % 4 == 0, tile, thread_index);
    }
};

#define DEFINE_STEEL_ATTENTION(name, tile_type, bq, bk, bd, wm, wn)       \
kernel void name(                                                         \
    device const uchar *query [[buffer(0)]],                              \
    device const uchar *key [[buffer(1)]],                                \
    device const uchar *value [[buffer(2)]],                              \
    device uchar *output [[buffer(3)]],                                   \
    constant SdpaParams &params [[buffer(4)]],                            \
    uint3 position [[threadgroup_position_in_grid]],                      \
    uint thread_index [[thread_index_in_threadgroup]],                    \
    uint3 threads [[threads_per_threadgroup]],                            \
    ushort simdgroup [[simdgroup_index_in_threadgroup]],                  \
    ushort lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup tile_type query_tile[bq * (bd + 16 / sizeof(tile_type))]; \
    threadgroup tile_type kv_tile[(bk + 16 / sizeof(tile_type)) * bd];    \
    ContiguousSteelAddressing addressing = {                              \
        key, value, params, position.y / params.heads_per_group};         \
    steel_attention_body<tile_type, bq, bk, bd>(                          \
        query, output, params, query_tile, kv_tile, addressing, position, \
        thread_index, simdgroup, lane);                                   \
}

DEFINE_STEEL_ATTENTION(steel_attention_64, float, 32, 32, 64, 4, 1)
DEFINE_STEEL_ATTENTION(steel_attention_128, float, 32, 16, 128, 4, 1)
DEFINE_STEEL_ATTENTION(steel_attention_half_64_64x32, half, 64, 32, 64, 8, 1)
// 96 * (128 + 8) * 2 + (16 + 8) * 128 * 2 = 32,256 bytes, 512 below 32 KiB.
DEFINE_STEEL_ATTENTION(steel_attention_half_128_96x16, half, 96, 16, 128, 12, 1)

// 16 * (256 + 4) * 4 + (8 + 4) * 256 * 4 = 28,928 bytes, below 32 KiB.
DEFINE_STEEL_ATTENTION(steel_attention_256, float, FORJA_STEEL_256_FLOAT_Q, FORJA_STEEL_256_FLOAT_K, 256, FORJA_STEEL_256_FLOAT_Q / 8, 1)
// 32 * (256 + 8) * 2 + (16 + 8) * 256 * 2 = 29,184 bytes, below 32 KiB.
DEFINE_STEEL_ATTENTION(steel_attention_half_256, half, FORJA_STEEL_256_HALF_Q, FORJA_STEEL_256_HALF_K, 256, FORJA_STEEL_256_HALF_Q / 8, 1)
