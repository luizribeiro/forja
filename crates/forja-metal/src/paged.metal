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
