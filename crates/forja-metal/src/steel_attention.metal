// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

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
constant uint steel_vectorized_loaders [[function_constant(217)]];

inline uint steel_attention_dtype_size(uint dtype) {
    return dtype == 0 ? 4 : 2;
}
template <typename T, int ROWS, int COLUMNS, int DST_ROW, int DST_COLUMN, int THREADS>
void steel_attention_load_tile(
    device const uchar *source,
    uint row_stride,
    uint column_stride,
    uint dtype,
    uint row_extent,
    bool rows_aligned,
    bool vector_aligned,
    threadgroup T *destination,
    uint thread_index) {
    if (steel_vectorized_loaders == 0) {
        for (uint linear = thread_index; linear < ROWS * COLUMNS; linear += THREADS) {
            uint row = linear / COLUMNS;
            uint column = linear % COLUMNS;
            bool valid = rows_aligned || row < row_extent;
            uint source_offset = row * row_stride + column * column_stride;
            uint destination_offset = row * DST_ROW + column * DST_COLUMN;
            destination[destination_offset] = valid
                ? T(load_float(source, source_offset, dtype)) : T(0);
        }
        return;
    }
    constexpr uint VECTOR_WIDTH = 4;
    constexpr uint VECTORS_PER_ROW = COLUMNS / VECTOR_WIDTH;
    constexpr uint VECTOR_COUNT = ROWS * VECTORS_PER_ROW;
    constexpr uint READS_PER_THREAD = (VECTOR_COUNT + THREADS - 1) / THREADS;
    for (uint read = 0; read < READS_PER_THREAD; ++read) {
        uint linear = thread_index + read * THREADS;
        bool assigned = linear < VECTOR_COUNT;
        uint row = assigned ? linear / VECTORS_PER_ROW : 0;
        uint column = assigned ? (linear % VECTORS_PER_ROW) * VECTOR_WIDTH : 0;
        bool valid = assigned && (rows_aligned || row < row_extent);
        uint source_offset = row * row_stride + column * column_stride;
        uint destination_offset = row * DST_ROW + column * DST_COLUMN;
        if (valid && column_stride == 1 && vector_aligned) {
            float4 values = load_float4_contiguous(source, source_offset, dtype);
            for (uint element = 0; element < VECTOR_WIDTH; ++element) {
                destination[destination_offset + element * DST_COLUMN] = T(values[element]);
            }
        } else if (assigned) {
            for (uint element = 0; element < VECTOR_WIDTH; ++element) {
                destination[destination_offset + element * DST_COLUMN] = valid
                    ? T(load_float(source, source_offset + element * column_stride, dtype))
                    : T(0);
            }
        }
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
        query_pointer, uint(params.q_strides[1]), uint(params.q_strides[2]), input0_dtype,
        params.query_length - first_query, false, query_base % 4 == 0,
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
