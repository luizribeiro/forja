// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

template <uint D>
void mlx_sdpa_vector_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device uchar *output,
    constant SdpaParams &params,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs,
    threadgroup float *max_scores,
    threadgroup float *sum_exp_scores) {
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint elements_per_thread = D / BD;
    float q[elements_per_thread];
    float result[elements_per_thread];
    uint head = position.x;
    uint query_row = position.y;
    uint kv_head = head / params.heads_per_group;
    ulong q_base = params.q_offset + ulong(head) * params.q_strides[0]
        + ulong(query_row) * params.q_strides[1];
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        q[element] = params.scale * load_float(
            query, q_base + ulong(column) * params.q_strides[2], input0_dtype);
        result[element] = 0.0f;
    }

    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    for (uint key_row = simd_group; key_row < params.key_length; key_row += BN) {
        if (params.causal != 0 && key_row > params.query_start + query_row) {
            continue;
        }
        ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
            + ulong(key_row) * params.k_strides[1];
        float score = 0.0f;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            score += q[element] * load_float(
                key, k_base + ulong(column) * params.k_strides[2], input1_dtype);
        }
        score = simd_sum(score);
        float next_maximum = max(maximum, score);
        float previous_factor = fast::exp(maximum - next_maximum);
        float score_factor = fast::exp(score - next_maximum);
        maximum = next_maximum;
        denominator = denominator * previous_factor + score_factor;
        ulong v_base = params.v_offset + ulong(kv_head) * params.v_strides[0]
            + ulong(key_row) * params.v_strides[1];
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            float v = load_float(
                value, v_base + ulong(column) * params.v_strides[2], input2_dtype);
            result[element] = result[element] * previous_factor + score_factor * v;
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
        outputs[simd_lane * BD + simd_group] = result[element];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            uint column = simd_group * elements_per_thread + element;
            ulong o_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(query_row) * params.o_strides[1]
                + ulong(column) * params.o_strides[2];
            store_float(output, o_index, output_dtype, total / denominator);
        }
    }
}

#define DEFINE_MLX_SDPA_VECTOR(NAME, D)                                      \
kernel void NAME(                                                            \
    device const uchar *query [[buffer(0)]],                                 \
    device const uchar *key [[buffer(1)]],                                   \
    device const uchar *value [[buffer(2)]],                                 \
    device uchar *output [[buffer(3)]],                                      \
    constant SdpaParams &params [[buffer(4)]],                               \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    threadgroup float max_scores[32];                                        \
    threadgroup float sum_exp_scores[32];                                    \
    mlx_sdpa_vector_body<D>(query, key, value, output, params, position,     \
                            simd_group, simd_lane, outputs, max_scores,       \
                            sum_exp_scores);                                 \
}

DEFINE_MLX_SDPA_VECTOR(mlx_sdpa_vector_64, 64)
DEFINE_MLX_SDPA_VECTOR(mlx_sdpa_vector_128, 128)

template <uint D>
void mlx_sdpa_vector_2pass_1_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device uchar *partials,
    device float *sums,
    device float *maxs,
    constant SdpaParams &params,
    uint3 position,
    uint simd_group,
    uint simd_lane) {
    constexpr uint BD = 32;
    constexpr uint elements_per_thread = D / BD;
    uint kv_head = position.x;
    uint block = position.z;
    uint query_row = simd_group / params.heads_per_group;
    uint group_head = simd_group % params.heads_per_group;
    uint head = kv_head * params.heads_per_group + group_head;
    float q[elements_per_thread];
    float result[elements_per_thread];
    ulong q_base = params.q_offset + ulong(head) * params.q_strides[0]
        + ulong(query_row) * params.q_strides[1];
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        q[element] = params.scale * load_float(
            query, q_base + ulong(column) * params.q_strides[2], input0_dtype);
        result[element] = 0.0f;
    }

    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    for (uint key_row = block; key_row < params.key_length; key_row += params.blocks) {
        if (params.causal != 0 && key_row > params.query_start + query_row) {
            continue;
        }
        ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
            + ulong(key_row) * params.k_strides[1];
        float score = 0.0f;
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            score += q[element] * load_float(
                key, k_base + ulong(column) * params.k_strides[2], input1_dtype);
        }
        score = simd_sum(score);
        float next_maximum = max(maximum, score);
        float previous_factor = fast::exp(maximum - next_maximum);
        float score_factor = fast::exp(score - next_maximum);
        maximum = next_maximum;
        denominator = denominator * previous_factor + score_factor;
        ulong v_base = params.v_offset + ulong(kv_head) * params.v_strides[0]
            + ulong(key_row) * params.v_strides[1];
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            float v = load_float(
                value, v_base + ulong(column) * params.v_strides[2], input2_dtype);
            result[element] = result[element] * previous_factor + score_factor * v;
        }
    }

    ulong partial_row = (ulong(head) * params.query_length + query_row) * params.blocks + block;
    if (simd_lane == 0) {
        sums[partial_row] = denominator;
        maxs[partial_row] = maximum;
    }
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        store_float(partials, partial_row * D + column, input0_dtype, result[element]);
    }
}

#define DEFINE_MLX_SDPA_VECTOR_PASS1(NAME, D)                               \
kernel void NAME(                                                            \
    device const uchar *query [[buffer(0)]],                                 \
    device const uchar *key [[buffer(1)]],                                   \
    device const uchar *value [[buffer(2)]],                                 \
    device uchar *partials [[buffer(3)]],                                    \
    device float *sums [[buffer(4)]],                                        \
    device float *maxs [[buffer(5)]],                                        \
    constant SdpaParams &params [[buffer(6)]],                               \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    mlx_sdpa_vector_2pass_1_body<D>(query, key, value, partials, sums, maxs, \
                                    params, position, simd_group, simd_lane); \
}

DEFINE_MLX_SDPA_VECTOR_PASS1(mlx_sdpa_vector_2pass_1_64, 64)
DEFINE_MLX_SDPA_VECTOR_PASS1(mlx_sdpa_vector_2pass_1_128, 128)

template <uint D>
void mlx_sdpa_vector_2pass_2_body(
    device const uchar *partials,
    device const float *sums,
    device const float *maxs,
    device uchar *output,
    constant SdpaParams &params,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *outputs) {
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint elements_per_thread = D / BD;
    uint head = position.x;
    uint query_row = position.y;
    ulong first_block = (ulong(head) * params.query_length + query_row) * params.blocks;
    float maximum = negative_finite_sentinel();
    for (uint group = 0; group < params.blocks / BN; ++group) {
        maximum = max(maximum, maxs[first_block + simd_lane + BN * group]);
    }
    maximum = simd_max(maximum);
    float denominator = 0.0f;
    for (uint group = 0; group < params.blocks / BN; ++group) {
        uint block = simd_lane + BN * group;
        float factor = fast::exp(maxs[first_block + block] - maximum);
        denominator += factor * sums[first_block + block];
    }
    denominator = simd_sum(denominator);

    float result[elements_per_thread];
    for (uint element = 0; element < elements_per_thread; ++element) {
        result[element] = 0.0f;
    }
    for (uint group = 0; group < params.blocks / BN; ++group) {
        uint block = simd_group + BN * group;
        float factor = fast::exp(maxs[first_block + block] - maximum);
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            ulong index = (first_block + block) * D + column;
            result[element] += factor * load_float(partials, index, input0_dtype);
        }
    }
    for (uint element = 0; element < elements_per_thread; ++element) {
        outputs[simd_lane * BD + simd_group] = result[element];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = simd_sum(outputs[simd_group * BD + simd_lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            uint column = simd_group * elements_per_thread + element;
            ulong o_index = params.o_offset + ulong(head) * params.o_strides[0]
                + ulong(query_row) * params.o_strides[1]
                + ulong(column) * params.o_strides[2];
            store_float(output, o_index, output_dtype, total / denominator);
        }
    }
}

#define DEFINE_MLX_SDPA_VECTOR_PASS2(NAME, D)                               \
kernel void NAME(                                                            \
    device const uchar *partials [[buffer(0)]],                              \
    device const float *sums [[buffer(1)]],                                  \
    device const float *maxs [[buffer(2)]],                                  \
    device uchar *output [[buffer(3)]],                                      \
    constant SdpaParams &params [[buffer(4)]],                               \
    uint3 position [[threadgroup_position_in_grid]],                         \
    uint simd_group [[simdgroup_index_in_threadgroup]],                      \
    uint simd_lane [[thread_index_in_simdgroup]]) {                          \
    threadgroup float outputs[32 * 32];                                      \
    mlx_sdpa_vector_2pass_2_body<D>(partials, sums, maxs, output, params,    \
                                    position, simd_group, simd_lane, outputs);\
}

DEFINE_MLX_SDPA_VECTOR_PASS2(mlx_sdpa_vector_2pass_2_64, 64)
DEFINE_MLX_SDPA_VECTOR_PASS2(mlx_sdpa_vector_2pass_2_128, 128)
