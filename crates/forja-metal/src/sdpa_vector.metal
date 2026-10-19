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

template <uint D, uint G, uint HPT>
void mlx_sdpa_vector_2pass_1_gqa_body(
    device const uchar *query,
    device const uchar *key,
    device const uchar *value,
    device uchar *partials,
    device float *sums,
    device float *maxs,
    constant SdpaParams &params,
    uint3 position,
    uint simd_group,
    uint simd_lane,
    threadgroup float *partial_outputs,
    threadgroup float *partial_sums,
    threadgroup float *partial_maxs) {
    constexpr uint BD = 32;
    constexpr uint elements_per_thread = D / BD;
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
        ulong q_base = params.q_offset + ulong(head) * params.q_strides[0];
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            q[local_head][element] = params.scale * load_float(
                query, q_base + ulong(column) * params.q_strides[2], input0_dtype);
            result[local_head][element] = 0.0f;
        }
        maximum[local_head] = negative_finite_sentinel();
        denominator[local_head] = 0.0f;
    }

    for (uint key_row = chunk_start; key_row < chunk_end; ++key_row) {
        if (params.causal != 0 && key_row > params.query_start) {
            continue;
        }
        float key_values[elements_per_thread];
        float value_values[elements_per_thread];
        ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
            + ulong(key_row) * params.k_strides[1];
        ulong v_base = params.v_offset + ulong(kv_head) * params.v_strides[0]
            + ulong(key_row) * params.v_strides[1];
        for (uint element = 0; element < elements_per_thread; ++element) {
            uint column = simd_lane * elements_per_thread + element;
            key_values[element] = load_float(
                key, k_base + ulong(column) * params.k_strides[2], input1_dtype);
            value_values[element] = load_float(
                value, v_base + ulong(column) * params.v_strides[2], input2_dtype);
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

#define DEFINE_MLX_SDPA_VECTOR_PASS1_GQA(NAME, D)                          \
kernel void NAME(                                                           \
    device const uchar *query [[buffer(0)]],                                \
    device const uchar *key [[buffer(1)]],                                  \
    device const uchar *value [[buffer(2)]],                                \
    device uchar *partials [[buffer(3)]],                                   \
    device float *sums [[buffer(4)]],                                       \
    device float *maxs [[buffer(5)]],                                       \
    constant SdpaParams &params [[buffer(6)]],                              \
    uint3 position [[threadgroup_position_in_grid]],                        \
    uint simd_group [[simdgroup_index_in_threadgroup]],                     \
    uint simd_lane [[thread_index_in_simdgroup]]) {                         \
    threadgroup float partial_outputs[8 * 2 * D];                           \
    threadgroup float partial_sums[8 * 2];                                  \
    threadgroup float partial_maxs[8 * 2];                                  \
    mlx_sdpa_vector_2pass_1_gqa_body<D, 8, 2>(                              \
        query, key, value, partials, sums, maxs, params, position,          \
        simd_group, simd_lane, partial_outputs, partial_sums, partial_maxs); \
}

DEFINE_MLX_SDPA_VECTOR_PASS1_GQA(mlx_sdpa_vector_2pass_1_gqa_64, 64)
DEFINE_MLX_SDPA_VECTOR_PASS1_GQA(mlx_sdpa_vector_2pass_1_gqa_128, 128)

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

template <uint D>
void mlx_sdpa_vector_f32_body(
    device const float *query,
    device const float *key,
    device const float *value,
    device float *output,
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
        + ulong(query_row) * params.q_strides[1] + simd_lane * elements_per_thread;
    for (uint element = 0; element < elements_per_thread; ++element) {
        q[element] = params.scale * query[q_base + element];
        result[element] = 0.0f;
    }

    ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
        + ulong(simd_group) * params.k_strides[1] + simd_lane * elements_per_thread;
    ulong v_base = params.v_offset + ulong(kv_head) * params.v_strides[0]
        + ulong(simd_group) * params.v_strides[1] + simd_lane * elements_per_thread;
    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    for (uint key_row = simd_group; key_row < params.key_length; key_row += BN) {
        if (params.causal == 0 || key_row <= params.query_start + query_row) {
            float score = 0.0f;
            for (uint element = 0; element < elements_per_thread; ++element) {
                score += q[element] * key[k_base + element];
            }
            score = simd_sum(score);
            float next_maximum = max(maximum, score);
            float previous_factor = fast::exp(maximum - next_maximum);
            float score_factor = fast::exp(score - next_maximum);
            maximum = next_maximum;
            denominator = denominator * previous_factor + score_factor;
            for (uint element = 0; element < elements_per_thread; ++element) {
                result[element] = result[element] * previous_factor
                    + score_factor * value[v_base + element];
            }
        }
        k_base += BN * params.k_strides[1];
        v_base += BN * params.v_strides[1];
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
                + ulong(query_row) * params.o_strides[1] + column;
            output[o_index] = total / denominator;
        }
    }
}

#define DEFINE_MLX_SDPA_VECTOR_F32(NAME, D)                                \
kernel void NAME(                                                          \
    device const float *query [[buffer(0)]],                               \
    device const float *key [[buffer(1)]],                                 \
    device const float *value [[buffer(2)]],                               \
    device float *output [[buffer(3)]],                                    \
    constant SdpaParams &params [[buffer(4)]],                             \
    uint3 position [[threadgroup_position_in_grid]],                       \
    uint simd_group [[simdgroup_index_in_threadgroup]],                    \
    uint simd_lane [[thread_index_in_simdgroup]]) {                        \
    threadgroup float outputs[32 * 32];                                    \
    threadgroup float max_scores[32];                                      \
    threadgroup float sum_exp_scores[32];                                  \
    mlx_sdpa_vector_f32_body<D>(query, key, value, output, params,         \
                                position, simd_group, simd_lane, outputs,  \
                                max_scores, sum_exp_scores);               \
}

DEFINE_MLX_SDPA_VECTOR_F32(mlx_sdpa_vector_f32_64, 64)
DEFINE_MLX_SDPA_VECTOR_F32(mlx_sdpa_vector_f32_128, 128)

template <uint D>
void mlx_sdpa_vector_2pass_1_f32_body(
    device const float *query,
    device const float *key,
    device const float *value,
    device float *partials,
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
        + ulong(query_row) * params.q_strides[1] + simd_lane * elements_per_thread;
    for (uint element = 0; element < elements_per_thread; ++element) {
        q[element] = params.scale * query[q_base + element];
        result[element] = 0.0f;
    }

    ulong k_base = params.k_offset + ulong(kv_head) * params.k_strides[0]
        + ulong(block) * params.k_strides[1] + simd_lane * elements_per_thread;
    ulong v_base = params.v_offset + ulong(kv_head) * params.v_strides[0]
        + ulong(block) * params.v_strides[1] + simd_lane * elements_per_thread;
    float maximum = negative_finite_sentinel();
    float denominator = 0.0f;
    for (uint key_row = block; key_row < params.key_length; key_row += params.blocks) {
        if (params.causal == 0 || key_row <= params.query_start + query_row) {
            float score = 0.0f;
            for (uint element = 0; element < elements_per_thread; ++element) {
                score += q[element] * key[k_base + element];
            }
            score = simd_sum(score);
            float next_maximum = max(maximum, score);
            float previous_factor = fast::exp(maximum - next_maximum);
            float score_factor = fast::exp(score - next_maximum);
            maximum = next_maximum;
            denominator = denominator * previous_factor + score_factor;
            for (uint element = 0; element < elements_per_thread; ++element) {
                result[element] = result[element] * previous_factor
                    + score_factor * value[v_base + element];
            }
        }
        k_base += ulong(params.blocks) * params.k_strides[1];
        v_base += ulong(params.blocks) * params.v_strides[1];
    }

    ulong partial_row = (ulong(head) * params.query_length + query_row) * params.blocks + block;
    if (simd_lane == 0) {
        sums[partial_row] = denominator;
        maxs[partial_row] = maximum;
    }
    for (uint element = 0; element < elements_per_thread; ++element) {
        uint column = simd_lane * elements_per_thread + element;
        partials[partial_row * D + column] = result[element];
    }
}

#define DEFINE_MLX_SDPA_VECTOR_PASS1_F32(NAME, D)                          \
kernel void NAME(                                                          \
    device const float *query [[buffer(0)]],                               \
    device const float *key [[buffer(1)]],                                 \
    device const float *value [[buffer(2)]],                               \
    device float *partials [[buffer(3)]],                                  \
    device float *sums [[buffer(4)]],                                      \
    device float *maxs [[buffer(5)]],                                      \
    constant SdpaParams &params [[buffer(6)]],                             \
    uint3 position [[threadgroup_position_in_grid]],                       \
    uint simd_group [[simdgroup_index_in_threadgroup]],                    \
    uint simd_lane [[thread_index_in_simdgroup]]) {                        \
    mlx_sdpa_vector_2pass_1_f32_body<D>(query, key, value, partials, sums, \
                                        maxs, params, position, simd_group,\
                                        simd_lane);                        \
}

DEFINE_MLX_SDPA_VECTOR_PASS1_F32(mlx_sdpa_vector_2pass_1_f32_64, 64)
DEFINE_MLX_SDPA_VECTOR_PASS1_F32(mlx_sdpa_vector_2pass_1_f32_128, 128)

template <uint D>
void mlx_sdpa_vector_2pass_2_f32_body(
    device const float *partials,
    device const float *sums,
    device const float *maxs,
    device float *output,
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
            result[element] += factor * partials[(first_block + block) * D + column];
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
                + ulong(query_row) * params.o_strides[1] + column;
            output[o_index] = total / denominator;
        }
    }
}

#define DEFINE_MLX_SDPA_VECTOR_PASS2_F32(NAME, D)                          \
kernel void NAME(                                                          \
    device const float *partials [[buffer(0)]],                            \
    device const float *sums [[buffer(1)]],                                \
    device const float *maxs [[buffer(2)]],                                \
    device float *output [[buffer(3)]],                                    \
    constant SdpaParams &params [[buffer(4)]],                             \
    uint3 position [[threadgroup_position_in_grid]],                       \
    uint simd_group [[simdgroup_index_in_threadgroup]],                    \
    uint simd_lane [[thread_index_in_simdgroup]]) {                        \
    threadgroup float outputs[32 * 32];                                    \
    mlx_sdpa_vector_2pass_2_f32_body<D>(partials, sums, maxs, output,      \
                                        params, position, simd_group,      \
                                        simd_lane, outputs);               \
}

DEFINE_MLX_SDPA_VECTOR_PASS2_F32(mlx_sdpa_vector_2pass_2_f32_64, 64)
DEFINE_MLX_SDPA_VECTOR_PASS2_F32(mlx_sdpa_vector_2pass_2_f32_128, 128)
