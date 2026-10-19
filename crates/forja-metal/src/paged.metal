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
    constant PagedCopyParams &params [[buffer(4)]],
    device atomic_uint *error_flag [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint count = params.tokens * params.elements;
    if (index >= count) {
        return;
    }
    uint token = index / params.elements;
    uint element = index % params.elements;
    uint logical_token = params.start + token;
    uint logical_page = logical_token / params.page_tokens;
    if (logical_page >= params.logical_pages) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
        return;
    }
    uint2 entry = table[logical_page];
    if (entry.x >= params.physical_pages || generations[entry.x] != entry.y) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
        return;
    }
    ulong paged_index = (ulong(entry.x) * params.page_tokens
        + logical_token % params.page_tokens) * params.elements + element;
    float value = params.direction == 0
        ? load_float(ordinary, index, input0_dtype)
        : load_float(pool, paged_index, input0_dtype);
    if (params.direction == 0) {
        store_float(pool, paged_index, input0_dtype, value);
    } else {
        store_float(ordinary, index, input0_dtype, value);
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
    uint contiguous_entries;
    uint blocks;
};

template <uint D, uint G, uint HPT>
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
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *partial_outputs,
    threadgroup float *partial_sums,
    threadgroup float *partial_maxs) {
    constexpr uint elements_per_thread = D / 32;
    constexpr uint chunks_per_head = HPT;
    constexpr uint head_groups = G / HPT;
    uint kv_head = position.x;
    uint block = position.z;
    uint chunk = simd_group / head_groups;
    uint first_head = (simd_group % head_groups) * HPT;
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
        ulong q_base = ulong(head) * D;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            q[local_head][element] = params.scale
                * load_float(query, q_base + column, input0_dtype);
            result[local_head][element] = 0.0f;
        }
        maximum[local_head] = negative_finite_sentinel();
        denominator[local_head] = 0.0f;
    }

    uint previous_logical_page = UINT_MAX;
    uint physical_page = 0;
    uint valid = 0;
    uint first_page = 0;
    if (simd_lane == 0) {
        // Validation rejects empty attention, so entry zero is always inside the table.
        first_page = table[0].x;
    }
    first_page = simd_broadcast(first_page, 0);
    for (uint key_row = chunk_start; key_row < chunk_end; ++key_row) {
        uint logical_page = key_row / params.page_tokens;
        if (logical_page != previous_logical_page) {
            uint2 entry = table[logical_page];
            uint expected = first_page + logical_page;
            uint lane_physical_page = params.contiguous_entries == 0 ? entry.x : expected;
            uint lane_valid = logical_page < params.logical_pages
                && entry.x < params.physical_pages
                && generations[entry.x] == entry.y
                && (params.contiguous_entries == 0 || entry.x == expected);
            physical_page = simd_broadcast(lane_physical_page, 0);
            valid = simd_broadcast(lane_valid, 0);
            if (simd_lane == 0 && valid == 0) {
                atomic_store_explicit(error_flag, 1, memory_order_relaxed);
                atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
            }
            previous_logical_page = logical_page;
        }
        if (valid == 0 || (params.causal != 0 && key_row > params.query_start)) {
            continue;
        }
        uint page_row = key_row % params.page_tokens;
        ulong k_base = ((ulong(physical_page) * params.page_tokens + page_row)
            * params.kv_heads + kv_head) * D;
        ulong v_base = ((ulong(physical_page) * params.page_tokens + page_row)
            * params.kv_heads + kv_head) * params.value_width;
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
            denominator[local_head] = denominator[local_head] * previous_factor + score_factor;
            for (uint element = 0; element < elements_per_thread; ++element) {
                result[local_head][element] = result[local_head][element] * previous_factor
                    + score_factor * value_values[element];
            }
        }
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
        group_maximum = max(group_maximum, partial_maxs[simd_group * HPT + chunk_index]);
    }
    float group_denominator = 0.0f;
    float group_result[elements_per_thread];
    for (uint element = 0; element < elements_per_thread; ++element) {
        group_result[element] = 0.0f;
    }
    for (uint chunk_index = 0; chunk_index < HPT; ++chunk_index) {
        uint slot = simd_group * HPT + chunk_index;
        float weight = partial_sums[slot]
            * fast::exp(partial_maxs[slot] - group_maximum);
        group_denominator += weight;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            group_result[element] += weight * partial_outputs[slot * D + column];
        }
    }

    uint head = kv_head * G + simd_group;
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

#define DEFINE_PAGED_SDPA_PASS1_GQA(NAME, D)                                \
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
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float partial_outputs[8 * 2 * D];                            \
    threadgroup float partial_sums[8 * 2];                                   \
    threadgroup float partial_maxs[8 * 2];                                   \
    paged_sdpa_2pass_1_gqa_body<D, 8, 2>(                                   \
        query, key, value, table, generations, partials, sums, maxs, params, \
        error_flag, position, simd_group, simd_lane, partial_outputs,        \
        partial_sums, partial_maxs);                                         \
}

DEFINE_PAGED_SDPA_PASS1_GQA(paged_sdpa_2pass_1_gqa_128, 128)

template <uint D>
void paged_sdpa_2pass_2_body(
    device const uchar *partials,
    device const float *sums,
    device const float *maxs,
    device uchar *output,
    constant PagedSdpaParams &params,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs) {
    constexpr uint groups = 32;
    constexpr uint elements_per_thread = D / groups;
    uint head = position.x;
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
            ulong output_index = ulong(head) * params.value_width + column;
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
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    paged_sdpa_2pass_2_body<D>(partials, sums, maxs, output, params,         \
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
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs,
    threadgroup float *max_scores,
    threadgroup float *sum_exp_scores,
    threadgroup uint &physical_page,
    threadgroup uint &first_page,
    threadgroup uint &valid) {
    constexpr uint groups = 32;
    constexpr uint elements_per_thread = D / 32;
    uint head = position.x;
    uint query_row = position.y;
    uint kv_head = head / params.heads_per_group;
    float q[elements_per_thread];
    float result[elements_per_thread];
    ulong q_base = (ulong(head) * params.query_length + query_row) * D;
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        q[element] = params.scale * load_float(query, q_base + column, input0_dtype);
        result[element] = 0.0f;
    }
    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    if (simd_group == 0 && simd_lane == 0) {
        // Validation rejects empty attention, so entry zero is always inside the table.
        first_page = table[0].x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pages = (params.key_length + params.page_tokens - 1) / params.page_tokens;
    for (uint logical_page = 0; logical_page < pages; ++logical_page) {
        if (simd_group == 0 && simd_lane == 0) {
            uint2 entry = table[logical_page];
            uint expected = first_page + logical_page;
            valid = logical_page < params.logical_pages
                && entry.x < params.physical_pages
                && generations[entry.x] == entry.y
                && (params.contiguous_entries == 0 || entry.x == expected);
            physical_page = params.contiguous_entries == 0 ? entry.x : expected;
            if (valid == 0) {
                atomic_store_explicit(error_flag, 1, memory_order_relaxed);
                atomic_fetch_min_explicit(error_flag + 1, logical_page, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint page_begin = logical_page * params.page_tokens;
        uint page_end = min(params.key_length, page_begin + params.page_tokens);
        if (valid != 0) {
            for (uint key_row = page_begin + simd_group; key_row < page_end; key_row += groups) {
                if (params.causal != 0 && key_row > params.query_start + query_row) {
                    continue;
                }
                uint page_row = key_row - page_begin;
                ulong k_base = ((ulong(physical_page) * params.page_tokens + page_row)
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
                ulong v_base = ((ulong(physical_page) * params.page_tokens + page_row)
                    * params.kv_heads + kv_head) * params.value_width;
                for (uint element = 0; element < elements_per_thread; ++element) {
                    uint column = simd_lane * elements_per_thread + element;
                    float v = load_float(value, v_base + column, input2_dtype);
                    result[element] = result[element] * previous_factor + score_factor * v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
            ulong out_index = (ulong(head) * params.query_length + query_row)
                * params.value_width + column;
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
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    threadgroup float max_scores[32];                                        \
    threadgroup float sum_exp_scores[32];                                    \
    threadgroup uint physical_page;                                          \
    threadgroup uint first_page;                                             \
    threadgroup uint valid;                                                  \
    paged_sdpa_body<D>(query, key, value, table, generations, output, params,\
        error_flag, position, simd_group, simd_lane, outputs, max_scores,     \
        sum_exp_scores, physical_page, first_page, valid);                   \
}

DEFINE_PAGED_SDPA(paged_sdpa_64, 64)
DEFINE_PAGED_SDPA(paged_sdpa_128, 128)
