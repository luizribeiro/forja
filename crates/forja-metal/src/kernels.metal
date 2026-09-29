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

kernel void argmax(
    device const uchar *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant TensorLayout &input_layout [[buffer(2)]],
    constant TensorLayout &output_layout [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]]) {
    threadgroup uint keys[1024];
    threadgroup uint indices[1024];
    if (lane < width) {
        uint best_index = lane;
        uint best_key = total_order_key(load_float(
            input, physical_index(input_layout, row * width + lane), input0_dtype));
        for (uint column = lane + group_width; column < width; column += group_width) {
            uint key = total_order_key(load_float(
                input, physical_index(input_layout, row * width + column), input0_dtype));
            if (key >= best_key) {
                best_key = key;
                best_index = column;
            }
        }
        keys[lane] = best_key;
        indices[lane] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint best_key = keys[0];
        uint best_index = indices[0];
        uint active = min(width, group_width);
        for (uint index = 1; index < active; ++index) {
            uint key = keys[index];
            uint candidate = indices[index];
            if (key > best_key || (key == best_key && candidate > best_index)) {
                best_key = key;
                best_index = candidate;
            }
        }
        output[physical_index(output_layout, row)] = best_index;
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
