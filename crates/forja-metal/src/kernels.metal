// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

kernel void hold(device uchar *buffer [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        buffer[0] = buffer[0];
    }
}

kernel void copy_strided(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant TensorLayout &input_layout [[buffer(2)]],
    constant TensorLayout &output_layout [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < output_layout.element_count) {
        copy_value(input, output,
                   physical_index(input_layout, index),
                   physical_index(output_layout, index));
    }
}

kernel void copy_contiguous(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant TensorLayout &input_layout [[buffer(2)]],
    constant TensorLayout &output_layout [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < output_layout.element_count) {
        copy_value(input, output,
                   input_layout.offset + index,
                   output_layout.offset + index);
    }
}

kernel void add_strided(
    device const uchar *left [[buffer(0)]],
    device const uchar *right [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &left_layout [[buffer(3)]],
    constant TensorLayout &right_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index < output_layout.element_count) {
        float result =
            load_float(left, physical_index(left_layout, index), input0_dtype) +
            load_float(right, physical_index(right_layout, index), input1_dtype);
        store_float(output, physical_index(output_layout, index), output_dtype, result);
    }
}

kernel void add_contiguous(
    device const uchar *left [[buffer(0)]],
    device const uchar *right [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &left_layout [[buffer(3)]],
    constant TensorLayout &right_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index < output_layout.element_count) {
        float result =
            load_float(left, left_layout.offset + index, input0_dtype) +
            load_float(right, right_layout.offset + index, input1_dtype);
        store_float(output, output_layout.offset + index, output_dtype, result);
    }
}

struct RmsNormParams {
    float eps;
    uint width;
};

template <bool looped>
void rms_norm_body(
    device const uchar *input,
    device const uchar *weight,
    device uchar *output,
    constant TensorLayout &input_layout,
    constant TensorLayout &weight_layout,
    constant TensorLayout &output_layout,
    constant RmsNormParams &params,
    uint row,
    uint lane,
    uint group_width,
    uint simd_lane,
    uint simd_group,
    threadgroup float *partial,
    threadgroup float &inverse_rms) {
    float sum = 0.0f;
    uint first = lane;
    uint step = looped ? group_width : params.width;
    for (uint column = first; column < params.width; column += step) {
        float value = load_float(
            input, physical_index(input_layout, row * params.width + column), input0_dtype);
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (simd_group == 0) {
        partial[simd_lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        sum = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            inverse_rms = precise::rsqrt(sum / float(params.width) + params.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = lane; column < params.width; column += group_width) {
        uint linear = row * params.width + column;
        float value = load_float(input, physical_index(input_layout, linear), input0_dtype);
        float scale = load_float(weight, physical_index(weight_layout, column), input1_dtype);
        store_float(output, physical_index(output_layout, linear), output_dtype,
                    value * inverse_rms * scale);
    }
}

kernel void rms_norm_single(
    device const uchar *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &input_layout [[buffer(3)]],
    constant TensorLayout &weight_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    constant RmsNormParams &params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    threadgroup float inverse_rms;
    rms_norm_body<false>(input, weight, output, input_layout, weight_layout,
                         output_layout, params, row, lane, group_width,
                         simd_lane, simd_group, partial, inverse_rms);
}

kernel void rms_norm_looped(
    device const uchar *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &input_layout [[buffer(3)]],
    constant TensorLayout &weight_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    constant RmsNormParams &params [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    threadgroup float inverse_rms;
    rms_norm_body<true>(input, weight, output, input_layout, weight_layout,
                        output_layout, params, row, lane, group_width,
                        simd_lane, simd_group, partial, inverse_rms);
}

template <bool looped>
void softmax_body(
    device const uchar *input,
    device uchar *output,
    constant TensorLayout &input_layout,
    constant TensorLayout &output_layout,
    uint width,
    uint row,
    uint lane,
    uint group_width,
    uint simd_lane,
    uint simd_group,
    threadgroup float *partial,
    threadgroup float &row_value) {
    uint step = looped ? group_width : width;
    float maximum = -INFINITY;
    for (uint column = lane; column < width; column += step) {
        maximum = max(maximum, load_float(
            input, physical_index(input_layout, row * width + column), input0_dtype));
    }
    maximum = simd_max(maximum);
    if (simd_group == 0) {
        partial[simd_lane] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        partial[simd_group] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        maximum = simd_max(partial[simd_lane]);
        if (simd_lane == 0) {
            row_value = maximum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = row_value;
    float sum = 0.0f;
    if (maximum != -INFINITY) {
        for (uint column = lane; column < width; column += step) {
            float value = load_float(
                input, physical_index(input_layout, row * width + column), input0_dtype);
            sum += exp(value - maximum);
        }
    }
    sum = simd_sum(sum);
    if (simd_group == 0) {
        partial[simd_lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        partial[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        sum = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            row_value = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = lane; column < width; column += group_width) {
        uint linear = row * width + column;
        float value = maximum == -INFINITY ? 0.0f :
            exp(load_float(input, physical_index(input_layout, linear), input0_dtype) - maximum) /
                row_value;
        store_float(output, physical_index(output_layout, linear), output_dtype, value);
    }
}

#define DEFINE_SOFTMAX(NAME, LOOPED) \
kernel void NAME( \
    device const uchar *input [[buffer(0)]], \
    device uchar *output [[buffer(1)]], \
    constant TensorLayout &input_layout [[buffer(2)]], \
    constant TensorLayout &output_layout [[buffer(3)]], \
    constant uint &width [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_position_in_threadgroup]], \
    uint group_width [[threads_per_threadgroup]], \
    uint simd_lane [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float partial[32]; \
    threadgroup float row_value; \
    softmax_body<LOOPED>(input, output, input_layout, output_layout, width, row, lane, \
                         group_width, simd_lane, simd_group, partial, row_value); \
}

DEFINE_SOFTMAX(softmax_single, false)
DEFINE_SOFTMAX(softmax_looped, true)

uint total_order_key(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

ulong argmax_candidate(float value, uint index) {
    return (ulong(total_order_key(value)) << 32) | ulong(index);
}

ulong argmax_threadgroup_max(
    ulong candidate,
    uint simd_lane,
    uint simd_group,
    threadgroup uint *partial_keys,
    threadgroup uint *partial_indices) {
    uint key = uint(candidate >> 32);
    uint index = uint(candidate);
    uint maximum_key = simd_max(key);
    uint maximum_index = simd_max(key == maximum_key ? index : 0);
    if (simd_group == 0) {
        partial_keys[simd_lane] = 0;
        partial_indices[simd_lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) {
        partial_keys[simd_group] = maximum_key;
        partial_indices[simd_group] = maximum_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group != 0) {
        return 0;
    }
    maximum_key = simd_max(partial_keys[simd_lane]);
    maximum_index = simd_max(
        partial_keys[simd_lane] == maximum_key ? partial_indices[simd_lane] : 0);
    return (ulong(maximum_key) << 32) | ulong(maximum_index);
}

kernel void argmax_partials(
    device const uchar *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant TensorLayout &input_layout [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &chunks [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial_keys[32];
    threadgroup uint partial_indices[32];
    uint row = group / chunks;
    uint chunk = group % chunks;
    uint first = chunk * 2048u;
    uint end = first + min(2048u, width - first);
    ulong best = 0;
    for (uint column = first + lane; column < end; column += group_width) {
        float value = load_float(
            input, physical_index(input_layout, row * width + column), input0_dtype);
        best = max(best, argmax_candidate(value, column));
    }
    best = argmax_threadgroup_max(
        best, simd_lane, simd_group, partial_keys, partial_indices);
    if (lane == 0) {
        output[row * chunks + chunk] = best;
    }
}

kernel void argmax_finalize(
    device const ulong *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant TensorLayout &output_layout [[buffer(2)]],
    constant uint &chunks [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint partial_keys[32];
    threadgroup uint partial_indices[32];
    ulong best = 0;
    for (uint chunk = lane; chunk < chunks; chunk += group_width) {
        best = max(best, input[row * chunks + chunk]);
    }
    best = argmax_threadgroup_max(
        best, simd_lane, simd_group, partial_keys, partial_indices);
    if (lane == 0) {
        output[physical_index(output_layout, row)] = uint(best);
    }
}

ulong splitmix64(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

float gumbel_noise(ulong seed, uint position, uint index) {
    ulong counter = seed ^ (ulong(position) << 32) ^ ulong(index);
    uint mantissa = uint(splitmix64(counter) >> 40);
    float uniform = (float(mantissa) + 0.5f) * (1.0f / 16777216.0f);
    return -log(-log(uniform));
}

struct SamplingConfig {
    float temperature;
    uint top_k;
    float top_p;
    ulong seed;
};

SamplingConfig sampling_config(
    device const uint *sampling,
    constant TensorLayout &layout) {
    SamplingConfig config;
    config.temperature = as_type<float>(sampling[physical_index(layout, 0)]);
    config.top_k = sampling[physical_index(layout, 1)];
    config.top_p = as_type<float>(sampling[physical_index(layout, 2)]);
    config.seed = ulong(sampling[physical_index(layout, 3)]) |
                  (ulong(sampling[physical_index(layout, 4)]) << 32);
    return config;
}

bool invalid_sampling(SamplingConfig config) {
    return !isfinite(config.temperature) || config.temperature < 0.0f ||
           !isfinite(config.top_p) || config.top_p <= 0.0f || config.top_p > 1.0f;
}

uint sampling_candidate_count(SamplingConfig config, uint width) {
    if (config.top_p < 1.0f) {
        return config.top_k == 0
            ? min(width, 1024u)
            : min(width, min(config.top_k, 1024u));
    }
    return config.top_k == 0 ? width : min(width, config.top_k);
}

bool sampling_is_greedy(SamplingConfig config) {
    return config.temperature == 0.0f || config.top_k == 1;
}

bool sampling_needs_radix(SamplingConfig config, uint width) {
    return !sampling_is_greedy(config) && sampling_candidate_count(config, width) < width;
}

bool sampling_uses_candidates(SamplingConfig config, uint width) {
    uint count = sampling_candidate_count(config, width);
    return !sampling_is_greedy(config) &&
           (config.top_p < 1.0f || (count < width && count <= 1024u));
}

constant uint sample_prefix = 0;
constant uint sample_remaining = 1;
constant uint sample_cutoff = 2;
constant uint sample_selected = 3;
constant uint sample_state_words = 4;

void sample_histogram_add(
    uint bucket,
    bool valid,
    device atomic_uint *histogram,
    uint simd_lane) {
    uint value = valid ? bucket : 0xffffffffu;
    for (uint span = 2; span <= 32; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            uint other = simd_shuffle_xor(value, ushort(stride));
            bool ascending = (simd_lane & span) == 0;
            bool lower = (simd_lane & stride) == 0;
            value = ascending == lower ? min(value, other) : max(value, other);
        }
    }
    uint previous_lane = simd_lane == 0 ? 0 : simd_lane - 1;
    uint previous = simd_shuffle(value, ushort(previous_lane));
    simd_vote starts = simd_ballot(simd_lane == 0 || previous != value);
    ulong mask = ulong(simd_vote::vote_t(starts)) & ((1ul << (simd_lane + 1)) - 1ul);
    uint run_start = 63u - clz(mask);
    uint next_lane = min(simd_lane + 1, 31u);
    uint next = simd_shuffle(value, ushort(next_lane));
    if (value != 0xffffffffu && (simd_lane == 31 || next != value)) {
        atomic_fetch_add_explicit(
            histogram + value, simd_lane - run_start + 1, memory_order_relaxed);
    }
}

kernel void sample_prepare(
    device const uint *sampling [[buffer(0)]],
    device uint *histogram [[buffer(1)]],
    device atomic_uint *state [[buffer(2)]],
    constant TensorLayout &sampling_layout [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    device atomic_uint *error_flag [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config)) {
        if (lane == 0) {
            atomic_store_explicit(error_flag, 2, memory_order_relaxed);
        }
        return;
    }
    if (lane < sample_state_words) {
        uint values[4] = {0, sampling_candidate_count(config, width), 0, 0};
        atomic_store_explicit(
            state + row * sample_state_words + lane, values[lane], memory_order_relaxed);
    }
    if (!sampling_needs_radix(config, width)) {
        return;
    }
    for (uint bucket = lane; bucket < 2048; bucket += group_width) {
        histogram[row * 2048 + bucket] = 0;
    }
}

kernel void sample_radix_histogram(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device atomic_uint *histogram [[buffer(2)]],
    device atomic_uint *state [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &shift [[buffer(7)]],
    constant uint &digit_mask [[buffer(8)]],
    constant uint &chunks [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_needs_radix(config, width)) {
        return;
    }
    uint row = group / chunks;
    uint chunk = group % chunks;
    uint first = chunk * 2048u;
    uint end = first + min(2048u, width - first);
    uint prefix = atomic_load_explicit(
        state + row * sample_state_words + sample_prefix, memory_order_relaxed);
    uint high_mask = shift == 21 ? 0 : ~0u << (shift + (shift == 0 ? 10 : 11));
    for (uint offset = 0; offset < 2048; offset += group_width) {
        uint column = first + lane + offset;
        bool valid = column < end;
        uint key = valid
            ? total_order_key(load_float(
                  logits, physical_index(logits_layout, row * width + column), input0_dtype))
            : 0;
        uint bucket = (key >> shift) & digit_mask;
        bool selected = valid && (key & high_mask) == prefix;
        if (shift == 21) {
            sample_histogram_add(
                bucket, selected, histogram + row * 2048, simd_lane);
        } else if (selected) {
            atomic_fetch_add_explicit(
                histogram + row * 2048 + bucket, 1, memory_order_relaxed);
        }
    }
}

kernel void sample_radix_locate(
    device const uint *sampling [[buffer(0)]],
    device uint *histogram [[buffer(1)]],
    device atomic_uint *state [[buffer(2)]],
    constant TensorLayout &sampling_layout [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    constant uint &shift [[buffer(5)]],
    constant uint &digit_mask [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_needs_radix(config, width)) {
        return;
    }
    if (lane == 0) {
        uint remaining = atomic_load_explicit(
            state + row * sample_state_words + sample_remaining, memory_order_relaxed);
        uint prefix = atomic_load_explicit(
            state + row * sample_state_words + sample_prefix, memory_order_relaxed);
        for (int bucket = int(digit_mask); bucket >= 0; --bucket) {
            uint count = histogram[row * 2048 + uint(bucket)];
            if (remaining > count) {
                remaining -= count;
            } else {
                prefix |= uint(bucket) << shift;
                break;
            }
        }
        atomic_store_explicit(
            state + row * sample_state_words + sample_prefix, prefix, memory_order_relaxed);
        atomic_store_explicit(
            state + row * sample_state_words + sample_remaining,
            remaining,
            memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bucket = lane; bucket < 2048; bucket += group_width) {
        histogram[row * 2048 + bucket] = 0;
    }
}

kernel void sample_index_histogram(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device atomic_uint *histogram [[buffer(2)]],
    device atomic_uint *state [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &chunks [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_needs_radix(config, width)) {
        return;
    }
    uint row = group / chunks;
    uint chunk = group % chunks;
    uint first = chunk * 2048u;
    uint end = first + min(2048u, width - first);
    uint threshold = atomic_load_explicit(
        state + row * sample_state_words + sample_prefix, memory_order_relaxed);
    for (uint offset = 0; offset < 2048; offset += group_width) {
        uint column = first + lane + offset;
        bool valid = column < end;
        uint key = valid
            ? total_order_key(load_float(
                  logits, physical_index(logits_layout, row * width + column), input0_dtype))
            : 0;
        if (valid && key == threshold) {
            uint bucket = uint((ulong(column) * 1024ul) / ulong(width));
            atomic_fetch_add_explicit(
                histogram + row * 2048 + min(bucket, 1023u), 1, memory_order_relaxed);
        }
    }
}

kernel void sample_index_locate(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device const uint *histogram [[buffer(2)]],
    device atomic_uint *state [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_needs_radix(config, width) || lane != 0) {
        return;
    }
    uint remaining = atomic_load_explicit(
        state + row * sample_state_words + sample_remaining, memory_order_relaxed);
    uint threshold = atomic_load_explicit(
        state + row * sample_state_words + sample_prefix, memory_order_relaxed);
    uint selected_bucket = 0;
    for (int bucket = 1023; bucket >= 0; --bucket) {
        uint count = histogram[row * 2048 + uint(bucket)];
        if (remaining > count) {
            remaining -= count;
        } else {
            selected_bucket = uint(bucket);
            break;
        }
    }
    uint first = uint((ulong(selected_bucket) * ulong(width) + 1023ul) / 1024ul);
    uint end = uint((ulong(selected_bucket + 1) * ulong(width) + 1023ul) / 1024ul);
    uint cutoff = first;
    for (uint column = min(end, width); column-- > first;) {
        float value = load_float(
            logits, physical_index(logits_layout, row * width + column), input0_dtype);
        if (total_order_key(value) == threshold && --remaining == 0) {
            cutoff = column;
            break;
        }
    }
    atomic_store_explicit(
        state + row * sample_state_words + sample_cutoff, cutoff, memory_order_relaxed);
}

kernel void sample_compact(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device ulong *candidates [[buffer(2)]],
    device atomic_uint *state [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &chunks [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_uses_candidates(config, width)) {
        return;
    }
    uint row = group / chunks;
    uint chunk = group % chunks;
    uint first = chunk * 2048u;
    uint end = first + min(2048u, width - first);
    bool filter = sampling_needs_radix(config, width);
    uint threshold = atomic_load_explicit(
        state + row * sample_state_words + sample_prefix, memory_order_relaxed);
    uint cutoff = atomic_load_explicit(
        state + row * sample_state_words + sample_cutoff, memory_order_relaxed);
    for (uint column = first + lane; column < end; column += group_width) {
        float value = load_float(
            logits, physical_index(logits_layout, row * width + column), input0_dtype);
        uint key = total_order_key(value);
        if (!filter || key > threshold || (key == threshold && column >= cutoff)) {
            uint slot = atomic_fetch_add_explicit(
                state + row * sample_state_words + sample_selected, 1, memory_order_relaxed);
            if (slot < 1024) {
                candidates[row * 1024 + slot] = (ulong(key) << 32) | ulong(column);
            }
        }
    }
}

kernel void sample_partials(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device ulong *partials [[buffer(2)]],
    device atomic_uint *state [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &position [[buffer(7)]],
    constant uint &chunks [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (sampling_uses_candidates(config, width)) {
        return;
    }
    threadgroup uint partial_keys[32];
    threadgroup uint partial_indices[32];
    uint row = group / chunks;
    uint chunk = group % chunks;
    uint first = chunk * 2048u;
    uint end = first + min(2048u, width - first);
    bool greedy = sampling_is_greedy(config);
    bool filter = !greedy && sampling_needs_radix(config, width);
    uint threshold = atomic_load_explicit(
        state + row * sample_state_words + sample_prefix, memory_order_relaxed);
    uint cutoff = atomic_load_explicit(
        state + row * sample_state_words + sample_cutoff, memory_order_relaxed);
    ulong best = 0;
    if (!invalid_sampling(config)) {
        for (uint column = first + lane; column < end; column += group_width) {
            float value = load_float(
                logits, physical_index(logits_layout, row * width + column), input0_dtype);
            uint key = total_order_key(value);
            if (greedy) {
                best = max(best, (ulong(key) << 32) | ulong(column));
            } else if (!filter || key > threshold || (key == threshold && column >= cutoff)) {
                float score = value / config.temperature +
                              gumbel_noise(config.seed, position, column);
                best = max(best, argmax_candidate(score, column));
            }
        }
    }
    best = argmax_threadgroup_max(
        best, simd_lane, simd_group, partial_keys, partial_indices);
    if (lane == 0) {
        partials[row * chunks + chunk] = best;
    }
}

kernel void sample_reduce_finalize(
    device const uint *sampling [[buffer(0)]],
    device const ulong *partials [[buffer(1)]],
    device uint *output [[buffer(2)]],
    constant TensorLayout &sampling_layout [[buffer(3)]],
    constant TensorLayout &output_layout [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &chunks [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (sampling_uses_candidates(config, width)) {
        return;
    }
    threadgroup uint partial_keys[32];
    threadgroup uint partial_indices[32];
    ulong best = 0;
    if (!invalid_sampling(config)) {
        for (uint chunk = lane; chunk < chunks; chunk += group_width) {
            best = max(best, partials[row * chunks + chunk]);
        }
    }
    best = argmax_threadgroup_max(
        best, simd_lane, simd_group, partial_keys, partial_indices);
    if (lane == 0) {
        output[physical_index(output_layout, row)] = uint(best);
    }
}

kernel void sample_candidates_finalize(
    device const uchar *logits [[buffer(0)]],
    device const uint *sampling [[buffer(1)]],
    device const ulong *candidates [[buffer(2)]],
    device uint *output [[buffer(3)]],
    constant TensorLayout &logits_layout [[buffer(4)]],
    constant TensorLayout &sampling_layout [[buffer(5)]],
    constant TensorLayout &output_layout [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &position [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong ordered[1024];
    threadgroup float weights[1024];
    threadgroup uint keep;
    threadgroup uint partial_keys[32];
    threadgroup uint partial_indices[32];
    SamplingConfig config = sampling_config(sampling, sampling_layout);
    if (invalid_sampling(config) || !sampling_uses_candidates(config, width)) {
        return;
    }
    uint count = sampling_candidate_count(config, width);
    for (uint slot = lane; slot < 1024; slot += group_width) {
        ordered[slot] = slot < count ? candidates[row * 1024 + slot] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = 2; span <= 1024; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            for (uint slot = lane; slot < 1024; slot += group_width) {
                uint other = slot ^ stride;
                if (other > slot) {
                    ulong left = ordered[slot];
                    ulong right = ordered[other];
                    bool descending = (slot & span) == 0;
                    if ((descending && left < right) || (!descending && left > right)) {
                        ordered[slot] = right;
                        ordered[other] = left;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    keep = count;
    if (config.top_p < 1.0f) {
        float maximum = load_float(
            logits,
            physical_index(logits_layout, row * width + uint(ordered[0])),
            input0_dtype);
        for (uint slot = lane; slot < count; slot += group_width) {
            uint column = uint(ordered[slot]);
            float value = load_float(
                logits, physical_index(logits_layout, row * width + column), input0_dtype);
            weights[slot] = exp((value - maximum) / config.temperature);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            float total = 0.0f;
            for (uint slot = 0; slot < count; ++slot) {
                total += weights[slot];
            }
            keep = 1;
            if (isfinite(total) && total > 0.0f) {
                float cumulative = 0.0f;
                for (uint slot = 0; slot < count; ++slot) {
                    cumulative += weights[slot] / total;
                    keep = slot + 1;
                    if (cumulative >= config.top_p) {
                        break;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ulong best = 0;
    for (uint slot = lane; slot < keep; slot += group_width) {
        uint column = uint(ordered[slot]);
        float value = load_float(
            logits, physical_index(logits_layout, row * width + column), input0_dtype);
        float score = value / config.temperature + gumbel_noise(config.seed, position, column);
        best = max(best, argmax_candidate(score, column));
    }
    best = argmax_threadgroup_max(
        best, simd_lane, simd_group, partial_keys, partial_indices);
    if (lane == 0) {
        output[physical_index(output_layout, row)] = uint(best);
    }
}

struct RopeParams {
    uint heads;
    uint width;
    uint half_width;
};

kernel void rope(
    device const uchar *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &input_layout [[buffer(3)]],
    constant TensorLayout &positions_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    constant RopeParams &params [[buffer(6)]],
    device const float *inverse_frequencies [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    uint pair_count = output_layout.element_count / 2;
    if (index >= pair_count) {
        return;
    }
    uint frequency = index % params.half_width;
    uint row = index / params.half_width;
    uint sequence = row / params.heads;
    uint base = row * params.width;
    uint first_index = base + frequency;
    uint second_index = first_index + params.half_width;
    uint position = positions[physical_index(positions_layout, sequence)];
    float angle = float(position) * inverse_frequencies[frequency];
    float cosine = cos(angle);
    float sine = sin(angle);
    float first = load_float(
        input, physical_index(input_layout, first_index), input0_dtype);
    float second = load_float(
        input, physical_index(input_layout, second_index), input0_dtype);
    store_float(output, physical_index(output_layout, first_index), output_dtype,
                first * cosine - second * sine);
    store_float(output, physical_index(output_layout, second_index), output_dtype,
                second * cosine + first * sine);
}

struct EmbedParams {
    uint vocab;
    uint width;
};

kernel void embed(
    device const uchar *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &table_layout [[buffer(3)]],
    constant TensorLayout &ids_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    constant EmbedParams &params [[buffer(6)]],
    device atomic_uint *error_flag [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= output_layout.element_count) {
        return;
    }
    uint id_position = index / params.width;
    uint column = index % params.width;
    uint id = ids[physical_index(ids_layout, id_position)];
    if (id >= params.vocab) {
        store_float(output, physical_index(output_layout, index), output_dtype, 0.0f);
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, id, memory_order_relaxed);
        return;
    }
    uint table_index = id * params.width + column;
    float value = load_float(
        table, physical_index(table_layout, table_index), input0_dtype);
    store_float(output, physical_index(output_layout, index), output_dtype, value);
}

kernel void silu_mul(
    device const uchar *gate [[buffer(0)]],
    device const uchar *up [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant TensorLayout &gate_layout [[buffer(3)]],
    constant TensorLayout &up_layout [[buffer(4)]],
    constant TensorLayout &output_layout [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= output_layout.element_count) {
        return;
    }
    ulong gate_index = physical_index(gate_layout, index);
    ulong up_index = physical_index(up_layout, index);
    ulong output_index = physical_index(output_layout, index);
    float gate_value = load_float(gate, gate_index, input0_dtype);
    float up_value = load_float(up, up_index, input1_dtype);
    float result = gate_value / (1.0f + exp(-gate_value)) * up_value;
    store_float(output, output_index, output_dtype, result);
}
