// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
struct PagedCopyParams {
    uint start;
    uint tokens;
    uint elements;
    uint page_tokens;
    uint logical_pages;
    uint physical_pages;
    uint direction;
};

kernel void paged_copy(
    device uchar *ordinary [[buffer(0)]],
    device uchar *pool [[buffer(1)]],
    device const uint2 *table [[buffer(2)]],
    device const uint *generations [[buffer(3)]],
    constant TensorLayout &ordinary_layout [[buffer(4)]],
    constant PagedCopyParams &params [[buffer(5)]],
    device atomic_uint *error_flag [[buffer(6)]],
    device const uint *activity [[buffer(7)]],
    uint2 position [[thread_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup uint physical_page;
    threadgroup uint valid;
    uint element = position.x;
    uint token = position.y;
    uint logical_token = params.start + token;
    if (activity[0] == 0) {
        if (params.direction != 0 && element < params.elements) {
            uint index = token * params.elements + element;
            ulong ordinary_index = physical_index(ordinary_layout, index);
            store_float(ordinary, ordinary_index, input0_dtype, 0.0f);
        }
        return;
    }
    if (thread_index == 0) {
        uint logical_page = logical_token / params.page_tokens;
        valid = logical_page < params.logical_pages;
        uint2 entry = uint2(0);
        if (valid != 0) {
            entry = table[logical_page];
            valid = entry.x < params.physical_pages
                && generations[entry.x] == entry.y;
        }
        physical_page = valid != 0 ? entry.x : 0;
        if (valid == 0) {
            atomic_store_explicit(error_flag, 1, memory_order_relaxed);
            atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0 || element >= params.elements) {
        return;
    }
    ulong paged_index = (ulong(physical_page) * params.page_tokens
        + logical_token % params.page_tokens) * params.elements + element;
    uint index = token * params.elements + element;
    ulong ordinary_index = physical_index(ordinary_layout, index);
    float value = params.direction == 0
        ? load_float(ordinary, ordinary_index, input0_dtype)
        : load_float(pool, paged_index, input1_dtype);
    if (params.direction == 0) {
        store_float(pool, paged_index, input1_dtype, value);
    } else {
        store_float(ordinary, ordinary_index, input0_dtype, value);
    }
}

struct PagedSdpaParams {
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
    uint page_tokens;
    uint logical_pages;
    uint physical_pages;
    uint blocks;
    uint padding;
    ulong q_offset;
    ulong q_strides[3];
    ulong o_offset;
    ulong o_strides[3];
};

constant uint paged_align_key [[function_constant(214)]];
constant uint paged_do_causal [[function_constant(215)]];

kernel void paged_sdpa_validate(
    device const uint2 *table [[buffer(0)]],
    device const uint *generations [[buffer(1)]],
    constant PagedSdpaParams &params [[buffer(2)]],
    device atomic_uint *error_flag [[buffer(3)]],
    device const uint *activity [[buffer(4)]],
    uint logical_page [[thread_position_in_grid]]) {
    if (activity[0] == 0) {
        return;
    }
    uint pages = (params.key_length + params.page_tokens - 1) / params.page_tokens;
    if (logical_page >= pages) {
        return;
    }
    bool valid = logical_page < params.logical_pages;
    if (valid) {
        uint2 entry = table[logical_page];
        uint expected = table[0].x + logical_page;
        valid = entry.x < params.physical_pages
            && generations[entry.x] == entry.y;
        if (valid && entry.x != expected) {
            atomic_store_explicit(error_flag + 2, 0, memory_order_relaxed);
        }
    }
    if (!valid) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
    }
}

template <typename TileT, int BQ, int BK, int BD>
void paged_steel_attention_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device const uint2 *table,
    device const uint *generations,
    device uchar *output,
    constant PagedSdpaParams &params,
    device atomic_uint *error_flag,
    device const uint *activity,
    threadgroup TileT *query_tile,
    threadgroup TileT *kv_tile,
    threadgroup uint &physical_page,
    threadgroup uint &first_page,
    threadgroup uint &contiguous,
    threadgroup uint &valid,
    uint3 position,
    uint thread_index,
    uint3 threads,
    ushort simdgroup,
    ushort lane) {
    constexpr short PADDING = 16 / sizeof(TileT);
    constexpr short LDQ = BD + PADDING;
    constexpr short LDK = BK + PADDING;
    constexpr short LDV = BD + PADDING;
    constexpr short FRAGMENT = 8;
    constexpr short TK = BK / FRAGMENT;
    constexpr short TD = BD / FRAGMENT;
    uint query_block = position.x;
    uint head = position.y;
    uint first_query = query_block * BQ;
    uint kv_head = head / params.heads_per_group;
    uint thread_count = threads.x * threads.y * threads.z;
    ulong query_base = params.q_offset + ulong(head) * params.q_strides[0];
    ulong output_base = params.o_offset + ulong(head) * params.o_strides[0]
        + ulong(first_query) * params.o_strides[1];
    BlockLoader<BQ, BD>::load_to(
        query, query_base, params.q_strides[1], params.q_strides[2], input0_dtype, first_query, 0,
        params.query_length, BD, false, true, query_tile, LDQ, 1,
        thread_index, thread_count);
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
    if (paged_do_causal != 0) {
        uint query_limit = min(first_query + BQ, params.query_length);
        uint absolute_limit = params.query_start + query_limit;
        key_limit = min(key_blocks, (absolute_limit + BK - 1) / BK);
        first_causal_block = (params.query_start + first_query) / BK;
    }
    if (thread_index == 0) {
        valid = activity[0] != 0
            && atomic_load_explicit(error_flag, memory_order_relaxed) == 0;
        if (valid != 0) {
            first_page = table[0].x;
            physical_page = first_page;
            contiguous = atomic_load_explicit(error_flag + 2, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0) {
        uint rows = min(uint(BQ), params.query_length - first_query);
        for (uint linear = thread_index; linear < rows * BD; linear += thread_count) {
            uint output_row = linear / BD;
            uint output_column = linear % BD;
            ulong output_index = output_base + ulong(output_row) * params.o_strides[1]
                + ulong(output_column) * params.o_strides[2];
            store_float(output, output_index, output_dtype, 0.0f);
        }
        return;
    }
    uint previous_page = 0;
    uint row_stride = params.kv_heads * BD;
    ulong first_physical_token = ulong(first_page) * params.page_tokens;
    for (uint key_block = 0; key_block < key_limit; ++key_block) {
        uint first_key = key_block * BK;
        ulong page_base;
        if (contiguous != 0) {
            page_base = (first_physical_token + first_key) * row_stride
                + ulong(kv_head) * BD;
        } else {
            uint logical_page = first_key / params.page_tokens;
            if (logical_page != previous_page) {
                if (thread_index == 0) {
                    physical_page = table[logical_page].x;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                previous_page = logical_page;
            }
            uint page_row = first_key % params.page_tokens;
            page_base = (ulong(physical_page) * params.page_tokens + page_row)
                * row_stride + ulong(kv_head) * BD;
        }
        for (uint linear = thread_index; linear < BK * BD; linear += thread_count) {
            uint tile_row = linear / BD;
            uint tile_column = linear % BD;
            bool in_range = paged_align_key != 0
                || first_key + tile_row < params.key_length;
            ulong index = page_base + ulong(tile_row) * row_stride + tile_column;
            kv_tile[tile_column * LDK + tile_row] = in_range
                ? TileT(load_float(key, index, input1_dtype)) : TileT(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scores.clear();
        for (short dimension = 0; dimension < TD; ++dimension) {
            simdgroup_barrier(mem_flags::mem_none);
            TileFragment::load(
                query_mma.at(0, 0),
                &query_tile[query_row * LDQ + dimension * FRAGMENT + column], LDQ, 1);
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                TileFragment::load(
                    key_mma.at(0, key_fragment),
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
        if (paged_do_causal != 0 && key_block >= first_causal_block) {
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
        for (uint linear = thread_index; linear < BK * BD; linear += thread_count) {
            uint tile_row = linear / BD;
            uint tile_column = linear % BD;
            bool in_range = paged_align_key != 0
                || first_key + tile_row < params.key_length;
            ulong index = page_base + ulong(tile_row) * row_stride + tile_column;
            kv_tile[tile_row * LDV + tile_column] = in_range
                ? TileT(load_float(value, index, input2_dtype)) : TileT(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short dimension = 0; dimension < TD; ++dimension) {
            for (short key_fragment = 0; key_fragment < TK; ++key_fragment) {
                short key_row = key_fragment * FRAGMENT + row;
                TileFragment::load(
                    value_mma.at(0, 0),
                    &kv_tile[key_row * LDV + dimension * FRAGMENT + column], LDV, 1);
                if constexpr (is_same_v<TileT, float>) {
                    Fragment::multiply(result.at(0, dimension),
                        scores.at(0, key_fragment), value_mma.at(0, 0),
                        result.at(0, dimension));
                } else {
                    Fragment::multiply(result.at(0, dimension),
                        probabilities.at(0, key_fragment), value_mma.at(0, 0),
                        result.at(0, dimension));
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
    for (short dimension = 0; dimension < TD; ++dimension) {
        uint output_column = dimension * FRAGMENT + column;
        for (short element = 0; element < Fragment::elements; ++element) {
            ulong output_index = output_base + ulong(query_row) * params.o_strides[1]
                + ulong(output_column + element) * params.o_strides[2];
            store_float(output, output_index, output_dtype,
                result.at(0, dimension)[element]);
        }
    }
}

#define DEFINE_PAGED_STEEL_ATTENTION(name, tile_type, bq, bk, bd)          \
kernel void name(                                                           \
    device const uchar *query [[buffer(0)]],                                \
    device const uchar *key [[buffer(1)]],                                  \
    device const uchar *value [[buffer(2)]],                                \
    device const uint2 *table [[buffer(3)]],                                \
    device const uint *generations [[buffer(4)]],                           \
    device uchar *output [[buffer(5)]],                                     \
    constant PagedSdpaParams &params [[buffer(6)]],                         \
    device atomic_uint *error_flag [[buffer(7)]],                           \
    device const uint *activity [[buffer(8)]],                              \
    uint3 position [[threadgroup_position_in_grid]],                        \
    uint thread_index [[thread_index_in_threadgroup]],                      \
    uint3 threads [[threads_per_threadgroup]],                              \
    ushort simdgroup [[simdgroup_index_in_threadgroup]],                    \
    ushort lane [[thread_index_in_simdgroup]]) {                            \
    threadgroup tile_type query_tile[bq * (bd + 16 / sizeof(tile_type))];   \
    threadgroup tile_type kv_tile[(bk + 16 / sizeof(tile_type)) * bd];      \
    threadgroup uint physical_page;                                         \
    threadgroup uint first_page;                                            \
    threadgroup uint contiguous;                                            \
    threadgroup uint valid;                                                 \
    paged_steel_attention_body<tile_type, bq, bk, bd>(                      \
        query, key, value, table, generations, output, params, error_flag,  \
        activity,                                                           \
        query_tile, kv_tile, physical_page, first_page, contiguous, valid,  \
        position,                                                           \
        thread_index, threads, simdgroup, lane);                            \
}

DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_64, float, 32, 32, 64)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_128, float, 32, 16, 128)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_64_64x32, half, 64, 32, 64)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_128_96x16, half, 96, 16, 128)

template <uint D, uint G, uint HPT, uint KVPTG>
void paged_sdpa_2pass_1_gqa_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device const uint2 *table,
    device const uint *generations,
    device uchar *partials,
    device float *sums,
    device float *maxs,
    constant PagedSdpaParams &params,
    device atomic_uint *error_flag,
    device const uint *activity,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *partial_outputs,
    threadgroup float *partial_sums,
    threadgroup float *partial_maxs,
    threadgroup uint *physical_pages,
    threadgroup uint *page_valid,
    threadgroup uint &first_page,
    threadgroup uint &valid) {
    if (activity[0] == 0) {
        return;
    }
    constexpr uint elements_per_thread = D / 32;
    constexpr uint chunks_per_head = HPT;
    constexpr uint head_groups = G / HPT;
    uint packed_kv_head = simd_group / head_groups;
    uint kv_head = position.x * KVPTG + (KVPTG == 1 ? 0 : packed_kv_head);
    uint head_group = KVPTG == 1 ? simd_group : simd_group % head_groups;
    uint block = position.z;
    uint chunk = head_group / head_groups;
    uint first_head = (head_group % head_groups) * HPT;
    uint keys_per_block = (params.key_length + params.blocks - 1) / params.blocks;
    uint block_start = block * keys_per_block;
    uint block_end = min(params.key_length, block_start + keys_per_block);
    uint keys_per_chunk = (keys_per_block + chunks_per_head - 1) / chunks_per_head;
    uint chunk_start = block_start + chunk * keys_per_chunk;
    uint chunk_end = min(block_end, chunk_start + keys_per_chunk);

    float q[HPT][elements_per_thread];
    float result[HPT][elements_per_thread];
    float maximum[HPT];
    float denominator[HPT];
    for (uint local_head = 0; local_head < HPT; ++local_head) {
        uint head = kv_head * G + first_head + local_head;
        ulong q_base = params.q_offset + ulong(head) * params.q_strides[0];
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            q[local_head][element] = params.scale
                * load_float(query, q_base + column, input0_dtype);
            result[local_head][element] = 0.0f;
        }
        maximum[local_head] = negative_finite_sentinel();
        denominator[local_head] = 0.0f;
    }

    if (simd_group == 0 && simd_lane == 0) {
        valid = atomic_load_explicit(error_flag, memory_order_relaxed) == 0;
        if (valid != 0) {
            uint2 entry = table[0];
            valid = entry.x < params.physical_pages
                && generations[entry.x] == entry.y;
            first_page = valid != 0 ? entry.x : 0;
            if (valid == 0) {
                atomic_store_explicit(error_flag, 1, memory_order_relaxed);
                atomic_fetch_min_explicit(error_flag + 1, 0, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0) {
        uint head = kv_head * G + head_group;
        ulong partial_row = ulong(head) * params.blocks + block;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            store_float(partials, partial_row * D + column, input0_dtype, 0.0f);
        }
        if (simd_lane == 0) {
            sums[partial_row] = 1.0f;
            maxs[partial_row] = 0.0f;
        }
        return;
    }
    uint first_logical_page = block_start / params.page_tokens;
    uint end_logical_page = (block_end + params.page_tokens - 1) / params.page_tokens;
    ulong key_row_stride = ulong(params.kv_heads) * D;
    ulong value_row_stride = ulong(params.kv_heads) * params.value_width;
    constexpr uint pages_per_batch = 4;
    for (uint batch_start = first_logical_page;
         batch_start < end_logical_page; batch_start += pages_per_batch) {
        uint batch_pages = min(pages_per_batch, end_logical_page - batch_start);
        if (simd_group == 0 && simd_lane == 0) {
            for (uint index = 0; index < batch_pages; ++index) {
                uint logical_page = batch_start + index;
                bool accepted = logical_page < params.logical_pages;
                uint2 entry = uint2(0);
                if (accepted) {
                    entry = table[logical_page];
                    accepted = entry.x < params.physical_pages
                        && generations[entry.x] == entry.y;
                }
                page_valid[index] = accepted;
                physical_pages[index] = accepted ? entry.x : 0;
                if (!accepted) {
                    atomic_store_explicit(error_flag, 1, memory_order_relaxed);
                    atomic_fetch_min_explicit(
                        error_flag + 1, logical_page, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = 0; index < batch_pages; ++index) {
            if (page_valid[index] != 0) {
                uint logical_page = batch_start + index;
                uint page_start = logical_page * params.page_tokens;
                uint key_start = max(chunk_start, page_start);
                uint key_end = min(chunk_end, page_start + params.page_tokens);
                if (params.causal != 0) {
                    key_end = min(key_end, params.query_start + 1);
                }
                uint page_row = key_start - page_start;
                ulong physical_token = ulong(physical_pages[index])
                    * params.page_tokens + page_row;
                ulong k_base = physical_token * key_row_stride + ulong(kv_head) * D;
                ulong v_base = physical_token * value_row_stride
                    + ulong(kv_head) * params.value_width;
                for (uint key_row = key_start; key_row < key_end; ++key_row) {
                    float key_values[elements_per_thread];
                    float value_values[elements_per_thread];
                    for (uint element = 0; element < elements_per_thread; ++element) {
                        uint column = simd_lane * elements_per_thread + element;
                        key_values[element] = load_float(key, k_base + column, input1_dtype);
                        value_values[element] = load_float(value, v_base + column, input2_dtype);
                    }
                    for (uint local_head = 0; local_head < HPT; ++local_head) {
                        float score = 0.0f;
                        for (uint element = 0; element < elements_per_thread; ++element) {
                            score += q[local_head][element] * key_values[element];
                        }
                        score = simd_sum(score);
                        float next_maximum = max(maximum[local_head], score);
                        float previous_factor = fast::exp(maximum[local_head] - next_maximum);
                        float score_factor = fast::exp(score - next_maximum);
                        maximum[local_head] = next_maximum;
                        denominator[local_head] = denominator[local_head] * previous_factor
                            + score_factor;
                        for (uint element = 0; element < elements_per_thread; ++element) {
                            result[local_head][element] = result[local_head][element]
                                * previous_factor + score_factor * value_values[element];
                        }
                    }
                    k_base += key_row_stride;
                    v_base += value_row_stride;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if constexpr (KVPTG > 1) {
        uint head = kv_head * G + head_group;
        ulong partial_row = ulong(head) * params.blocks + block;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            store_float(partials, partial_row * D + column, input0_dtype,
                        result[0][element]);
        }
        if (simd_lane == 0) {
            sums[partial_row] = denominator[0];
            maxs[partial_row] = maximum[0];
        }
        return;
    }

    for (uint local_head = 0; local_head < HPT; ++local_head) {
        uint slot = (first_head + local_head) * HPT + chunk;
        float inverse = denominator[local_head] > 0.0f
            ? 1.0f / denominator[local_head] : 0.0f;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            partial_outputs[slot * D + column] = result[local_head][element] * inverse;
        }
        if (simd_lane == 0) {
            partial_sums[slot] = denominator[local_head];
            partial_maxs[slot] = maximum[local_head];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float group_maximum = negative_finite_sentinel();
    for (uint chunk_index = 0; chunk_index < HPT; ++chunk_index) {
        group_maximum = max(group_maximum, partial_maxs[head_group * HPT + chunk_index]);
    }
    float group_denominator = 0.0f;
    float group_result[elements_per_thread];
    for (uint element = 0; element < elements_per_thread; ++element) {
        group_result[element] = 0.0f;
    }
    for (uint chunk_index = 0; chunk_index < HPT; ++chunk_index) {
        uint slot = head_group * HPT + chunk_index;
        float weight = partial_sums[slot]
            * fast::exp(partial_maxs[slot] - group_maximum);
        group_denominator += weight;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            group_result[element] += weight * partial_outputs[slot * D + column];
        }
    }

    uint head = kv_head * G + head_group;
    ulong partial_row = ulong(head) * params.blocks + block;
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        store_float(partials, partial_row * D + column, input0_dtype,
                    group_result[element]);
    }
    if (simd_lane == 0) {
        sums[partial_row] = group_denominator;
        maxs[partial_row] = group_maximum;
    }
}

#define DEFINE_PAGED_SDPA_PASS1_GQA(NAME, D, G, HPT, KVPTG)                 \
kernel void NAME(                                                            \
    device const uchar *query [[buffer(0)]],                                 \
    device const uchar *key [[buffer(1)]],                                   \
    device const uchar *value [[buffer(2)]],                                 \
    device const uint2 *table [[buffer(3)]],                                 \
    device const uint *generations [[buffer(4)]],                            \
    device uchar *partials [[buffer(5)]],                                    \
    device float *sums [[buffer(6)]],                                        \
    device float *maxs [[buffer(7)]],                                        \
    constant PagedSdpaParams &params [[buffer(8)]],                          \
    device atomic_uint *error_flag [[buffer(9)]],                            \
    device const uint *activity [[buffer(10)]],                              \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float partial_outputs[G * HPT * D];                          \
    threadgroup float partial_sums[G * HPT];                                 \
    threadgroup float partial_maxs[G * HPT];                                 \
    threadgroup uint physical_pages[4];                                     \
    threadgroup uint page_valid[4];                                         \
    threadgroup uint first_page;                                             \
    threadgroup uint valid;                                                  \
    paged_sdpa_2pass_1_gqa_body<D, G, HPT, KVPTG>(                          \
        query, key, value, table, generations, partials, sums, maxs, params, \
        error_flag, activity, position, simd_group, simd_lane, partial_outputs,\
        partial_sums, partial_maxs, physical_pages, page_valid, first_page,  \
        valid);                                                              \
}

DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa1_128, 128, 1, 1, 1)
DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa1x8_128, 128, 1, 1, 8)
DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa2_128, 128, 2, 1, 1)
DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa2x8_128, 128, 2, 1, 8)
DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa8_128, 128, 8, 2, 1)

template <uint D>
void paged_sdpa_2pass_2_body(
    device const uchar *partials,
    device const float *sums,
    device const float *maxs,
    device uchar *output,
    constant PagedSdpaParams &params,
    device const uint *activity,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs) {
    constexpr uint groups = 32;
    constexpr uint elements_per_thread = D / groups;
    uint head = position.x;
    if (activity[0] == 0) {
        for (uint column = simd_lane; column < D; column += 32) {
            ulong output_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(column) * params.o_strides[2];
            store_float(output, output_index, output_dtype, 0.0f);
        }
        return;
    }
    ulong first_block = ulong(head) * params.blocks;
    float maximum = negative_finite_sentinel();
    for (uint group = 0; group < params.blocks / groups; ++group) {
        maximum = max(maximum, maxs[first_block + simd_lane + groups * group]);
    }
    maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint group = 0; group < params.blocks / groups; ++group) {
        uint block = simd_lane + groups * group;
        float factor = fast::exp(maxs[first_block + block] - maximum);
        denominator += factor * sums[first_block + block];
    }
    denominator = simd_sum(denominator);

    float result[elements_per_thread];
    for (uint element = 0; element < elements_per_thread; ++element) {
        result[element] = 0.0f;
    }
    for (uint group = 0; group < params.blocks / groups; ++group) {
        uint block = simd_group + groups * group;
        float factor = fast::exp(maxs[first_block + block] - maximum);
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            ulong index = (first_block + block) * D + column;
            result[element] += factor * load_float(partials, index, input0_dtype);
        }
    }
    for (uint element = 0; element < elements_per_thread; ++element) {
        outputs[simd_lane * groups + simd_group] = result[element];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(outputs[simd_group * groups + simd_lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            uint column = simd_group * elements_per_thread + element;
            ulong output_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(column) * params.o_strides[2];
            store_float(output, output_index, output_dtype, total / denominator);
        }
    }
}

#define DEFINE_PAGED_SDPA_PASS2(NAME, D)                                    \
kernel void NAME(                                                            \
    device const uchar *partials [[buffer(0)]],                              \
    device const float *sums [[buffer(1)]],                                  \
    device const float *maxs [[buffer(2)]],                                  \
    device uchar *output [[buffer(3)]],                                      \
    constant PagedSdpaParams &params [[buffer(4)]],                          \
    device const uint *activity [[buffer(5)]],                               \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    paged_sdpa_2pass_2_body<D>(partials, sums, maxs, output, params, activity,\
        position, simd_group, simd_lane, outputs);                           \
}

DEFINE_PAGED_SDPA_PASS2(paged_sdpa_2pass_2_128, 128)

template <uint D>
void paged_sdpa_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device const uint2 *table,
    device const uint *generations,
    device uchar *output,
    constant PagedSdpaParams &params,
    device atomic_uint *error_flag,
    device const uint *activity,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs,
    threadgroup float *max_scores,
    threadgroup float *sum_exp_scores,
    threadgroup uint &physical_page,
    threadgroup uint &first_page,
    threadgroup uint &contiguous,
    threadgroup uint &valid) {
    constexpr uint groups = 32;
    constexpr uint elements_per_thread = D / 32;
    uint head = position.x;
    uint query_row = position.y;
    uint kv_head = head / params.heads_per_group;
    float q[elements_per_thread];
    float result[elements_per_thread];
    ulong q_base = params.q_offset + ulong(head) * params.q_strides[0]
        + ulong(query_row) * params.q_strides[1];
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        q[element] = params.scale * load_float(query, q_base + column, input0_dtype);
        result[element] = 0.0f;
    }
    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    if (simd_group == 0 && simd_lane == 0) {
        valid = activity[0] != 0
            && atomic_load_explicit(error_flag, memory_order_relaxed) == 0;
        if (valid != 0) {
            first_page = table[0].x;
            physical_page = first_page;
            contiguous = atomic_load_explicit(error_flag + 2, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0) {
        if (simd_group == 0) {
            for (uint element = 0; element < elements_per_thread; ++element) {
                uint column = simd_lane * elements_per_thread + element;
                ulong out_index = params.o_offset + ulong(head) * params.o_strides[0]
                    + ulong(query_row) * params.o_strides[1]
                    + ulong(column) * params.o_strides[2];
                store_float(output, out_index, output_dtype, 0.0f);
            }
        }
        return;
    }
    uint pages = (params.key_length + params.page_tokens - 1) / params.page_tokens;
    for (uint logical_page = 0; logical_page < pages; ++logical_page) {
        uint page = first_page + logical_page;
        if (contiguous == 0) {
            if (simd_group == 0 && simd_lane == 0) {
                physical_page = table[logical_page].x;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            page = physical_page;
        }
        uint page_begin = logical_page * params.page_tokens;
        uint page_end = min(params.key_length, page_begin + params.page_tokens);
        for (uint key_row = page_begin + simd_group; key_row < page_end; key_row += groups) {
            if (params.causal != 0 && key_row > params.query_start + query_row) {
                continue;
            }
            uint page_row = key_row - page_begin;
            ulong k_base = ((ulong(page) * params.page_tokens + page_row)
                * params.kv_heads + kv_head) * D;
            float score = 0.0f;
            for (uint element = 0; element < elements_per_thread; ++element) {
                uint column = simd_lane * elements_per_thread + element;
                score += q[element] * load_float(
                    key, k_base + column, input1_dtype);
            }
            score = simd_sum(score);
            float next_maximum = max(maximum, score);
            float previous_factor = fast::exp(maximum - next_maximum);
            float score_factor = fast::exp(score - next_maximum);
            maximum = next_maximum;
            denominator = denominator * previous_factor + score_factor;
            ulong v_base = ((ulong(page) * params.page_tokens + page_row)
                * params.kv_heads + kv_head) * params.value_width;
            for (uint element = 0; element < elements_per_thread; ++element) {
                uint column = simd_lane * elements_per_thread + element;
                float v = load_float(value, v_base + column, input2_dtype);
                result[element] = result[element] * previous_factor + score_factor * v;
            }
        }
        if (contiguous == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (simd_lane == 0) {
        max_scores[simd_group] = maximum;
        sum_exp_scores[simd_group] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float source_maximum = max_scores[simd_lane];
    float global_maximum = simd_max(source_maximum);
    float factor = fast::exp(source_maximum - global_maximum);
    denominator = simd_sum(sum_exp_scores[simd_lane] * factor);
    for (uint element = 0; element < elements_per_thread; ++element) {
        outputs[simd_lane * 32 + simd_group] = result[element];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(outputs[simd_group * 32 + simd_lane] * factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            uint column = simd_group * elements_per_thread + element;
            ulong out_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(query_row) * params.o_strides[1]
                + ulong(column) * params.o_strides[2];
            store_float(output, out_index, output_dtype, total / denominator);
        }
    }
}

#define DEFINE_PAGED_SDPA(NAME, D)                                           \
kernel void NAME(                                                            \
    device const uchar *query [[buffer(0)]],                                 \
    device const uchar *key [[buffer(1)]],                                   \
    device const uchar *value [[buffer(2)]],                                 \
    device const uint2 *table [[buffer(3)]],                                 \
    device const uint *generations [[buffer(4)]],                            \
    device uchar *output [[buffer(5)]],                                      \
    constant PagedSdpaParams &params [[buffer(6)]],                          \
    device atomic_uint *error_flag [[buffer(7)]],                            \
    device const uint *activity [[buffer(8)]],                               \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    threadgroup float max_scores[32];                                        \
    threadgroup float sum_exp_scores[32];                                    \
    threadgroup uint physical_page;                                          \
    threadgroup uint first_page;                                             \
    threadgroup uint contiguous;                                             \
    threadgroup uint valid;                                                  \
    paged_sdpa_body<D>(query, key, value, table, generations, output, params,\
        error_flag, activity, position, simd_group, simd_lane, outputs, max_scores,\
        sum_exp_scores, physical_page, first_page,                            \
        contiguous, valid);                                                  \
}

DEFINE_PAGED_SDPA(paged_sdpa_64, 64)
DEFINE_PAGED_SDPA(paged_sdpa_128, 128)
