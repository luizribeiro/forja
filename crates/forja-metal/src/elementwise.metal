// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
using namespace metal;

constant uint input0_dtype [[function_constant(0)]];
constant uint input1_dtype [[function_constant(1)]];
constant uint output_dtype [[function_constant(2)]];
constant uint input2_dtype [[function_constant(3)]];
constant uint program_input3_dtype [[function_constant(4)]];
constant uint program_input4_dtype [[function_constant(5)]];
constant uint program_input5_dtype [[function_constant(6)]];
constant uint program_input6_dtype [[function_constant(7)]];
constant uint program_input7_dtype [[function_constant(8)]];
constant uint program_output1_dtype [[function_constant(9)]];
constant uint program_output2_dtype [[function_constant(10)]];
constant uint program_output3_dtype [[function_constant(11)]];

float negative_finite_sentinel() {
    return as_type<float>(0xff7fffffu);
}

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

float load_float(device const uchar *buffer, ulong index, uint dtype) {
    if (dtype == 0) {
        return *reinterpret_cast<device const float *>(buffer + index * 4);
    }
    if (dtype == 1) {
        return float(*reinterpret_cast<device const half *>(buffer + index * 2));
    }
    return float(*reinterpret_cast<device const bfloat *>(buffer + index * 2));
}

uint load_uint(device const uchar *buffer, ulong index) {
    return reinterpret_cast<device const uint *>(buffer)[index];
}

float load_bfloat_bits(device const uchar *buffer, ulong index) {
    uint bits = uint(reinterpret_cast<device const ushort *>(buffer)[index]) << 16;
    return as_type<float>(bits);
}

void store_float(device uchar *buffer, ulong index, uint dtype, float value) {
    if (dtype == 0) {
        *reinterpret_cast<device float *>(buffer + index * 4) = value;
    } else if (dtype == 1) {
        *reinterpret_cast<device half *>(buffer + index * 2) = half(value);
    } else {
        *reinterpret_cast<device bfloat *>(buffer + index * 2) = bfloat(value);
    }
}

void store_bfloat_bits(device uchar *buffer, ulong index, float value) {
    uint bits = as_type<uint>(value);
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    reinterpret_cast<device ushort *>(buffer)[index] = ushort(rounded >> 16);
}

void copy_value(
    device const uchar *input,
    device uchar *output,
    ulong input_index,
    ulong output_index) {
    if (input0_dtype < 3) {
        store_float(output, output_index, output_dtype,
                    load_float(input, input_index, input0_dtype));
    } else if (input0_dtype == 3) {
        reinterpret_cast<device int *>(output)[output_index] =
            reinterpret_cast<device const int *>(input)[input_index];
    } else {
        reinterpret_cast<device uint *>(output)[output_index] =
            reinterpret_cast<device const uint *>(input)[input_index];
    }
}
