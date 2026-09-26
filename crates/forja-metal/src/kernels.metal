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
