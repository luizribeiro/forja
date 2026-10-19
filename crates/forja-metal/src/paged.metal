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
};

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
