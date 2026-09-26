#include <metal_stdlib>
using namespace metal;

kernel void hold(device uchar *buffer [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        buffer[0] = buffer[0];
    }
}

constant uint gate_dtype [[function_constant(0)]];
constant uint up_dtype [[function_constant(1)]];
constant uint output_dtype [[function_constant(2)]];

struct TensorLayout {
    ulong offset;
    uint rank;
    uint shape[8];
    uint element_count;
    ulong strides[8];
};

ulong physical_index(constant TensorLayout &layout, uint linear) {
    ulong physical = layout.offset;
    for (uint axis = layout.rank; axis > 0; --axis) {
        uint extent = layout.shape[axis - 1];
        physical += ulong(linear % extent) * layout.strides[axis - 1];
        linear /= extent;
    }
    return physical;
}

float load_value(device const uchar *buffer, ulong index, uint dtype) {
    if (dtype == 0) {
        return *reinterpret_cast<device const float *>(buffer + index * 4);
    }
    if (dtype == 1) {
        return float(*reinterpret_cast<device const half *>(buffer + index * 2));
    }
    return float(*reinterpret_cast<device const bfloat *>(buffer + index * 2));
}

void store_value(device uchar *buffer, ulong index, uint dtype, float value) {
    if (dtype == 0) {
        *reinterpret_cast<device float *>(buffer + index * 4) = value;
    } else if (dtype == 1) {
        *reinterpret_cast<device half *>(buffer + index * 2) = half(value);
    } else {
        *reinterpret_cast<device bfloat *>(buffer + index * 2) = bfloat(value);
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
    float gate_value = load_value(gate, gate_index, gate_dtype);
    float up_value = load_value(up, up_index, up_dtype);
    float result = gate_value / (1.0f + exp(-gate_value)) * up_value;
    store_value(output, output_index, output_dtype, result);
}
