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

constant uint align_Q [[function_constant(210)]];
constant uint align_K [[function_constant(211)]];
constant uint do_causal [[function_constant(212)]];

struct SteelMax {
    static float apply(float left, float right) {
        return max(left, right);
    }
};

struct SteelSum {
    static float apply(float left, float right) {
        return left + right;
    }
};

struct SteelMultiply {
    static float apply(float left, float right) {
        return left * right;
    }
};

struct SteelExpSubtract {
    static float apply(float left, float right) {
        return fast::exp(left - right);
    }
};

struct SteelDivide {
    static float apply(float left, float right) {
        return left / right;
    }
};

template <int BQ, int BK, int BD, int WM, int WN>
void steel_attention_body(
    device const uchar *query [[buffer(0)]],
    device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant SdpaParams &params [[buffer(4)]],
    threadgroup float *query_tile,
    threadgroup float *kv_tile,
    uint3 position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short Q_PADDING = 4;
    constexpr short K_PADDING = 4;
    constexpr short V_PADDING = 4;
    constexpr short LDQ = BD + Q_PADDING;
    constexpr short LDK = BK + K_PADDING;
    constexpr short LDV = BD + V_PADDING;
    constexpr short FRAGMENT = 8;
    constexpr short TK = BK / FRAGMENT;
    constexpr short TD = BD / FRAGMENT;

    uint query_block = position.x;
    uint head = position.y;
    uint first_query = query_block * BQ;
    uint kv_head = head / params.heads_per_group;

    ulong query_base = params.q_offset + ulong(head) * params.q_strides[0];
    ulong key_base = params.k_offset + ulong(kv_head) * params.k_strides[0];
    ulong value_base = params.v_offset + ulong(kv_head) * params.v_strides[0];
    ulong output_base = params.o_offset + ulong(head) * params.o_strides[0]
        + ulong(first_query) * params.o_strides[1];
    uint thread_count = threads.x * threads.y * threads.z;

    BlockLoader<BQ, BD>::load_to(
        query, query_base, params.q_strides[1], params.q_strides[2], input0_dtype,
        first_query, 0, params.query_length, BD,
        align_Q != 0, true, query_tile, LDQ, 1, thread_index, thread_count);

    using Fragment = SteelMMAFragment<float>;
    SteelMMATile<float, 1, 1> query_mma;
    SteelMMATile<float, 1, TK> key_mma;
    SteelMMATile<float, 1, TK> scores;
    SteelMMATile<float, 1, 1> value_mma;
    SteelMMATile<float, 1, TD> result;
    result.clear();

    short2 coordinate = Fragment::coordinate(lane);
    short row = coordinate.y;
    short column = coordinate.x;
    short query_row = FRAGMENT * simdgroup + row;
    float maximum[1] = {negative_finite_sentinel()};
    float denominator[1] = {0.0f};

    uint key_blocks = (params.key_length + BK - 1) / BK;
    uint key_limit = key_blocks;
    uint first_causal_block = key_blocks;
    if (do_causal != 0) {
        uint query_limit = min(first_query + BQ, params.query_length);
        uint absolute_limit = params.query_start + query_limit;
        key_limit = min(key_blocks, (absolute_limit + BK - 1) / BK);
        first_causal_block = (params.query_start + first_query) / BK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_block = 0; key_block < key_limit; ++key_block) {
        uint first_key = key_block * BK;
        BlockLoader<BK, BD>::load_to(
            key, key_base, params.k_strides[1], params.k_strides[2], input1_dtype,
            first_key, 0, params.key_length, BD,
            align_K != 0, true, kv_tile, 1, LDK, thread_index, thread_count);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        scores.clear();
        for (short dimension = 0; dimension < TD; ++dimension) {
            simdgroup_barrier(mem_flags::mem_none);
            Fragment::load(
                query_mma.at(0, 0),
                &query_tile[query_row * LDQ + dimension * FRAGMENT + column],
                LDQ, 1);
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                Fragment::load(
                    key_mma.at(0, key_fragment),
                    &kv_tile[dimension * FRAGMENT * LDK
                        + row * LDK + key_fragment * FRAGMENT + column],
                    LDK, 1);
            }
            simdgroup_barrier(mem_flags::mem_none);
            steel_tile_multiply(scores, query_mma, key_mma, scores);
        }

        for (short element = 0; element < scores.element_count; ++element) {
            scores.elements()[element] *= params.scale;
        }
        if (align_K == 0 && first_key + BK > params.key_length) {
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_column = key_fragment * FRAGMENT + column;
                for (short element = 0; element < Fragment::elements; ++element) {
                    if (first_key + key_column + element >= params.key_length) {
                        scores.at(0, key_fragment)[element] = negative_finite_sentinel();
                    }
                }
            }
        }
        if (do_causal != 0 && key_block >= first_causal_block) {
            uint absolute_query = params.query_start + first_query + query_row;
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_column = key_fragment * FRAGMENT + column;
                for (short element = 0; element < Fragment::elements; ++element) {
                    if (first_key + key_column + element > absolute_query) {
                        scores.at(0, key_fragment)[element] = negative_finite_sentinel();
                    }
                }
            }
        }

        float next_maximum[1] = {maximum[0]};
        scores.template row_reduce<SteelMax>(next_maximum);
        scores.template row_apply<SteelExpSubtract>(next_maximum);
        float adjustment[1] = {fast::exp(maximum[0] - next_maximum[0])};
        maximum[0] = next_maximum[0];
        float block_sum[1] = {0.0f};
        scores.template row_reduce<SteelSum>(block_sum);
        denominator[0] = denominator[0] * adjustment[0] + block_sum[0];
        result.template row_apply<SteelMultiply>(adjustment);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        BlockLoader<BK, BD>::load_to(
            value, value_base, params.v_strides[1], params.v_strides[2], input2_dtype,
            first_key, 0, params.key_length, BD,
            align_K != 0, true, kv_tile, LDV, 1, thread_index, thread_count);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short dimension = 0; dimension < TD; ++dimension) {
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_row = key_fragment * FRAGMENT + row;
                Fragment::load(
                    value_mma.at(0, 0),
                    &kv_tile[key_row * LDV + dimension * FRAGMENT + column],
                    LDV, 1);
                Fragment::multiply(
                    result.at(0, dimension), scores.at(0, key_fragment),
                    value_mma.at(0, 0), result.at(0, dimension));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    result.template row_apply<SteelDivide>(denominator);
    uint output_row = first_query + query_row;
    if (output_row >= params.query_length) {
        return;
    }
    for (short dimension = 0; dimension < TD; ++dimension) {
        uint output_column = dimension * FRAGMENT + column;
        for (short element = 0; element < Fragment::elements; ++element) {
            ulong output_index = output_base
                + ulong(query_row) * params.o_strides[1]
                + ulong(output_column + element) * params.o_strides[2];
            store_float(
                output, output_index, output_dtype,
                result.at(0, dimension)[element]);
        }
    }
}

#define DEFINE_STEEL_ATTENTION(name, bq, bk, bd, wm, wn)                  \
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
    threadgroup float query_tile[bq * (bd + 4)];                         \
    threadgroup float kv_tile[(bk + 4) * bd];                            \
    steel_attention_body<bq, bk, bd, wm, wn>(                            \
        query, key, value, output, params, query_tile, kv_tile, position, \
        thread_index,                                                     \
        threads, simdgroup, lane);                                        \
}

DEFINE_STEEL_ATTENTION(steel_attention_64, 32, 32, 64, 4, 1)
DEFINE_STEEL_ATTENTION(steel_attention_128, 32, 16, 128, 4, 1)
