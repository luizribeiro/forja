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

constant uint paged_has_activity [[function_constant(216)]];

inline bool paged_lane_active(device const uint *activity) {
    return paged_has_activity == 0 || activity[0] != 0;
}

inline bool paged_pages_are_consecutive(uint first, uint logical, uint physical) {
    return ulong(physical) == ulong(first) + logical;
}

struct AuthenticatedPagedPage {
    uint physical;
    uint valid;
};

AuthenticatedPagedPage authenticate_paged_page(
    uint logical_page, uint logical_pages, uint physical_pages,
    device const uint2 *table, device const uint *generations,
    device atomic_uint *error_flag) {
    bool valid = logical_page < logical_pages;
    uint2 entry = uint2(0);
    if (valid) {
        entry = table[logical_page];
        valid = entry.x < physical_pages;
        if (valid) valid = generations[entry.x] == entry.y;
    }
    if (!valid) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
    }
    return {valid ? entry.x : 0, uint(valid)};
}

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
    if (!paged_lane_active(activity)) {
        if (params.direction != 0 && element < params.elements) {
            uint index = token * params.elements + element;
            ulong ordinary_index = physical_index(ordinary_layout, index);
            store_float(ordinary, ordinary_index, input0_dtype, 0.0f);
        }
        return;
    }
    if (thread_index == 0) {
        uint logical_page = logical_token / params.page_tokens;
        auto page = authenticate_paged_page(logical_page, params.logical_pages,
            params.physical_pages, table, generations, error_flag);
        valid = page.valid;
        physical_page = page.physical;
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
    if (!paged_lane_active(activity)) {
        return;
    }
    uint pages = (params.key_length + params.page_tokens - 1) / params.page_tokens;
    if (logical_page >= pages) {
        return;
    }
    auto page = authenticate_paged_page(logical_page, params.logical_pages,
        params.physical_pages, table, generations, error_flag);
    if (page.valid != 0 && !paged_pages_are_consecutive(table[0].x, logical_page, page.physical)) {
        atomic_store_explicit(error_flag + 2, 0, memory_order_relaxed);
    }
}


template <int BD>
struct PagedSteelAddressing {
    device const uchar *key;
    device const uchar *value;
    device const uint2 *table;
    constant PagedSdpaParams &params;
    threadgroup uint &physical_page;
    uint first_page;
    uint contiguous;
    uint kv_head;
    uint previous_page;
    uint thread_index;

    static bool causal() { return paged_do_causal != 0; }

    ulong block_base(uint first_key) {
        uint row_stride = params.kv_heads * BD;
        if (contiguous != 0) {
            return (ulong(first_page) * params.page_tokens + first_key) * row_stride
                + ulong(kv_head) * BD;
        }
        uint logical_page = first_key / params.page_tokens;
        if (logical_page != previous_page) {
            if (thread_index == 0) {
                physical_page = table[logical_page].x;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            previous_page = logical_page;
        }
        uint page_row = first_key % params.page_tokens;
        return (ulong(physical_page) * params.page_tokens + page_row) * row_stride
            + ulong(kv_head) * BD;
    }

    template <typename T, int BK, int WIDTH, int LD, int THREADS>
    void load_key(uint first_key, uint key_length, threadgroup T *tile, uint index) {
        ulong base = block_base(first_key);
        device const uchar *source = key + base * steel_attention_dtype_size(input1_dtype);
        steel_attention_load_tile<T, BK, WIDTH, 1, LD, THREADS>(source,
            params.kv_heads * BD, 1, input1_dtype, key_length - first_key,
            paged_align_key != 0, base % 4 == 0, tile, index);
    }

    template <typename T, int BK, int WIDTH, int LD, int THREADS>
    void load_value(uint first_key, uint key_length, threadgroup T *tile, uint index) {
        ulong base = block_base(first_key);
        device const uchar *source = value + base * steel_attention_dtype_size(input2_dtype);
        steel_attention_load_tile<T, BK, WIDTH, LD, 1, THREADS>(source,
            params.kv_heads * BD, 1, input2_dtype, key_length - first_key,
            paged_align_key != 0, base % 4 == 0, tile, index);
    }
};

template <typename TileT, int BQ, int BK, int BD>
void paged_steel_attention_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device const uint2 *table,
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
    ushort simdgroup,
    ushort lane) {
    uint first_query = position.x * BQ;
    if (thread_index == 0) {
        valid = paged_lane_active(activity)
            && atomic_load_explicit(error_flag, memory_order_relaxed) == 0;
        if (valid != 0) {
            first_page = table[0].x;
            physical_page = first_page;
            contiguous = atomic_load_explicit(error_flag + 2, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid == 0) {
        ulong output_base = params.o_offset + ulong(position.y) * params.o_strides[0]
            + ulong(first_query) * params.o_strides[1];
        uint rows = min(uint(BQ), params.query_length - first_query);
        constexpr uint THREADS = BQ / 8 * 32;
        for (uint linear = thread_index; linear < rows * BD; linear += THREADS) {
            uint output_row = linear / BD;
            uint output_column = linear % BD;
            ulong output_index = output_base + ulong(output_row) * params.o_strides[1]
                + ulong(output_column) * params.o_strides[2];
            store_float(output, output_index, output_dtype, 0.0f);
        }
        return;
    }
    PagedSteelAddressing<BD> addressing = {
        key, value, table, params, physical_page, first_page, contiguous,
        position.y / params.heads_per_group, 0, thread_index};
    steel_attention_body<TileT, BQ, BK, BD>(
        query, output, params, query_tile, kv_tile, addressing, position,
        thread_index, simdgroup, lane);
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
        query, key, value, table, output, params, error_flag,              \
        activity,                                                           \
        query_tile, kv_tile, physical_page, first_page, contiguous, valid,  \
        position, thread_index, simdgroup, lane);                           \
}

DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_64, float, 32, 32, 64)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_128, float, 32, 16, 128)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_64_64x32, half, 64, 32, 64)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_128_96x16, half, 96, 16, 128)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_128_32x16, half, 32, 16, 128)
DEFINE_PAGED_STEEL_ATTENTION(paged_steel_attention_half_128_64x16, half, 64, 16, 128)

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
    if (!paged_lane_active(activity)) {
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
            auto page = authenticate_paged_page(0, params.logical_pages,
                params.physical_pages, table, generations, error_flag);
            valid = page.valid;
            first_page = page.physical;
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
                auto page = authenticate_paged_page(logical_page, params.logical_pages,
                    params.physical_pages, table, generations, error_flag);
                page_valid[index] = page.valid;
                physical_pages[index] = page.physical;
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
DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa8_128, 128, 8, 4, 1)

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
    if (!paged_lane_active(activity)) {
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
        valid = paged_lane_active(activity)
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

kernel void paged_contiguity_probe(
    device const uint4 *cases [[buffer(0)]], device uint4 *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < 4) {
        uint4 value = cases[index];
        output[index] = uint4(uint(paged_pages_are_consecutive(value.x, value.y, value.z)), 0, 0, 0);
    }
}
