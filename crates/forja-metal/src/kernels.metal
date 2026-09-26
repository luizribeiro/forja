// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

kernel void hold(device uchar *buffer [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        buffer[0] = buffer[0];
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
