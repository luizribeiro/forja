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

struct SteelMax {
    static float apply(float left, float right) { return max(left, right); }
};

struct SteelSum {
    static float apply(float left, float right) { return left + right; }
};

struct SteelMultiply {
    static float apply(float left, float right) { return left * right; }
};

struct SteelExpSubtract {
    static float apply(float left, float right) { return fast::exp(left - right); }
};

struct SteelDivide {
    static float apply(float left, float right) { return left / right; }
};

inline uint steel_attention_dtype_size(uint dtype) {
    return dtype == 0 ? 4 : 2;
}

template <typename T, int ROWS, int COLUMNS, int DST_ROW, int DST_COLUMN, int THREADS>
void steel_attention_load_tile(
    device const uchar *source,
    ulong row_stride,
    ulong column_stride,
    uint dtype,
    uint row_extent,
    bool rows_aligned,
    threadgroup T *destination,
    uint thread_index) {
    for (uint linear = thread_index; linear < ROWS * COLUMNS; linear += THREADS) {
        uint row = linear / COLUMNS;
        uint column = linear % COLUMNS;
        bool valid = rows_aligned || row < row_extent;
        ulong source_offset = ulong(row) * row_stride + ulong(column) * column_stride;
        uint destination_offset = row * DST_ROW + column * DST_COLUMN;
        destination[destination_offset] = valid
            ? T(load_float(source, source_offset, dtype)) : T(0);
    }
}

template <typename TileT, int BQ, int BK, int BD, typename Params, typename Addressing>
void steel_attention_body(
    device const uchar *query,
    device uchar *output,
    constant Params &params,
    threadgroup TileT *query_tile,
    threadgroup TileT *kv_tile,
    thread Addressing &addressing,
    uint3 position,
    uint thread_index,
    ushort simdgroup,
    ushort lane) {
    constexpr short PADDING = 16 / sizeof(TileT);
    constexpr short LDQ = BD + PADDING;
    constexpr short LDK = BK + PADDING;
    constexpr short LDV = BD + PADDING;
    constexpr short FRAGMENT = 8;
    constexpr short TK = BK / FRAGMENT;
    constexpr short TD = BD / FRAGMENT;
    constexpr uint THREADS = BQ / FRAGMENT * 32;

    uint first_query = position.x * BQ;
    uint head = position.y;
    ulong query_base = params.q_offset + ulong(head) * params.q_strides[0]
        + ulong(first_query) * params.q_strides[1];
    device const uchar *query_pointer = query
        + query_base * steel_attention_dtype_size(input0_dtype);
    steel_attention_load_tile<TileT, BQ, BD, LDQ, 1, THREADS>(
        query_pointer, params.q_strides[1], params.q_strides[2], input0_dtype,
        params.query_length - first_query, false,
        query_tile, thread_index);

    using Fragment = SteelMMAFragment<float>;
    using TileFragment = SteelMMAFragment<TileT>;
    SteelMMATile<TileT, 1, 1> query_mma;
    SteelMMATile<TileT, 1, TK> key_mma;
    SteelMMATile<float, 1, TK> scores;
    SteelMMATile<TileT, 1, TK> probabilities;
    SteelMMATile<TileT, 1, 1> value_mma;
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
    if (Addressing::causal()) {
        uint query_limit = min(first_query + BQ, params.query_length);
        uint absolute_limit = params.query_start + query_limit;
        key_limit = min(key_blocks, (absolute_limit + BK - 1) / BK);
        first_causal_block = (params.query_start + first_query) / BK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_block = 0; key_block < key_limit; ++key_block) {
        uint first_key = key_block * BK;
        addressing.template load_key<TileT, BK, BD, LDK, THREADS>(
            first_key, params.key_length, kv_tile, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scores.clear();
        for (short dimension = 0; dimension < TD; ++dimension) {
            simdgroup_barrier(mem_flags::mem_none);
            TileFragment::load(query_mma.at(0, 0),
                &query_tile[query_row * LDQ + dimension * FRAGMENT + column], LDQ, 1);
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                TileFragment::load(key_mma.at(0, key_fragment),
                    &kv_tile[dimension * FRAGMENT * LDK + row * LDK
                        + key_fragment * FRAGMENT + column], LDK, 1);
            }
            simdgroup_barrier(mem_flags::mem_none);
            steel_tile_multiply(scores, query_mma, key_mma, scores);
        }
        for (short element = 0; element < scores.element_count; ++element) {
            scores.elements()[element] *= params.scale;
        }
        if (first_key + BK > params.key_length) {
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_column = key_fragment * FRAGMENT + column;
                for (short element = 0; element < Fragment::elements; ++element) {
                    if (first_key + key_column + element >= params.key_length) {
                        scores.at(0, key_fragment)[element] = negative_finite_sentinel();
                    }
                }
            }
        }
        if (Addressing::causal() && key_block >= first_causal_block) {
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
        if constexpr (!is_same_v<TileT, float>) {
            for (short element = 0; element < scores.element_count; ++element) {
                probabilities.elements()[element] = TileT(scores.elements()[element]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        addressing.template load_value<TileT, BK, BD, LDV, THREADS>(
            first_key, params.key_length, kv_tile, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short dimension = 0; dimension < TD; ++dimension) {
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_row = key_fragment * FRAGMENT + row;
                TileFragment::load(value_mma.at(0, 0),
                    &kv_tile[key_row * LDV + dimension * FRAGMENT + column], LDV, 1);
                if constexpr (is_same_v<TileT, float>) {
                    Fragment::multiply(result.at(0, dimension), scores.at(0, key_fragment),
                        value_mma.at(0, 0), result.at(0, dimension));
                } else {
                    Fragment::multiply(result.at(0, dimension), probabilities.at(0, key_fragment),
                        value_mma.at(0, 0), result.at(0, dimension));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    result.template row_apply<SteelDivide>(denominator);
    uint output_row = first_query + query_row;
    if (output_row >= params.query_length) {
        return;
    }
    ulong output_base = params.o_offset + ulong(head) * params.o_strides[0]
        + ulong(output_row) * params.o_strides[1];
    for (short dimension = 0; dimension < TD; ++dimension) {
        uint output_column = dimension * FRAGMENT + column;
        for (short element = 0; element < Fragment::elements; ++element) {
            ulong output_index = output_base
                + ulong(output_column + element) * params.o_strides[2];
            store_float(output, output_index, output_dtype,
                result.at(0, dimension)[element]);
        }
    }
}

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
            params.k_strides[1], params.k_strides[2], input1_dtype,
            key_length - first_key, align_K != 0, tile, thread_index);
    }

    template <typename T, int BK, int BD, int LD, int THREADS>
    void load_value(uint first_key, uint key_length, threadgroup T *tile, uint thread_index) {
        ulong base = params.v_offset + ulong(kv_head) * params.v_strides[0]
            + ulong(first_key) * params.v_strides[1];
        device const uchar *source = value + base * steel_attention_dtype_size(input2_dtype);
        steel_attention_load_tile<T, BK, BD, LD, 1, THREADS>(source,
            params.v_strides[1], params.v_strides[2], input2_dtype,
            key_length - first_key, align_K != 0, tile, thread_index);
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
