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

bool f32_is_nan(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}

bool f32_is_non_finite(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) >= 0x7f800000u;
}

bool f32_lt(float left, float right) {
    return !f32_is_nan(left) && !f32_is_nan(right) && left < right;
}

bool f32_le(float left, float right) {
    return !f32_is_nan(left) && !f32_is_nan(right) && left <= right;
}

bool f32_eq(float left, float right) {
    return !f32_is_nan(left) && !f32_is_nan(right) && left == right;
}

bool f32_ne(float left, float right) {
    return f32_is_nan(left) || f32_is_nan(right) || left != right;
}

bool f32_ge(float left, float right) {
    return !f32_is_nan(left) && !f32_is_nan(right) && left >= right;
}

bool f32_gt(float left, float right) {
    return !f32_is_nan(left) && !f32_is_nan(right) && left > right;
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

// NaN payloads are intentionally canonicalized to one quiet NaN.
ushort f32_to_f16_rne_bits(float value) {
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return ushort(sign | select(0x7c00u, 0x7e00u, magnitude > 0x7f800000u));
    }

    int exponent = int(magnitude >> 23) - 127;
    uint fraction = magnitude & 0x007fffffu;
    if (exponent > 15) {
        return ushort(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        uint rounded = fraction + 0x00000fffu + ((fraction >> 13) & 1u);
        uint half_exponent = uint(exponent + 15) << 10;
        if ((rounded & 0x00800000u) != 0u) {
            half_exponent += 0x0400u;
            rounded = 0u;
        }
        return ushort(sign | half_exponent | (rounded >> 13));
    }
    if (exponent < -25) {
        return ushort(sign);
    }

    uint significand = fraction | 0x00800000u;
    uint shift = uint(-exponent - 1);
    uint truncated = significand >> shift;
    uint remainder = significand & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    truncated += uint(remainder > halfway ||
                      (remainder == halfway && (truncated & 1u) != 0u));
    return ushort(sign | truncated);
}

ushort f32_to_bf16_rne_bits(float value) {
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 16) & 0x8000u;
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return ushort(sign | 0x7fc0u);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

void store_float(device uchar *buffer, ulong index, uint dtype, float value) {
    if (dtype == 0) {
        *reinterpret_cast<device float *>(buffer + index * 4) = value;
    } else if (dtype == 1) {
        reinterpret_cast<device ushort *>(buffer)[index] = f32_to_f16_rne_bits(value);
    } else {
        reinterpret_cast<device ushort *>(buffer)[index] = f32_to_bf16_rne_bits(value);
    }
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
