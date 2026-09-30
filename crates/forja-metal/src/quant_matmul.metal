// Copyright © 2023-2024 Apple Inc.
// SPDX-License-Identifier: MIT

constant uint quant_bits [[function_constant(12)]];
constant uint quant_group_size [[function_constant(13)]];

struct QuantEmbedParams {
    uint vocab;
    uint width;
};

kernel void quant_embed(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]],
    device const uint *ids [[buffer(3)]],
    device uchar *output [[buffer(4)]],
    constant TensorLayout &packed_layout [[buffer(5)]],
    constant TensorLayout &scale_layout [[buffer(6)]],
    constant TensorLayout &bias_layout [[buffer(7)]],
    constant TensorLayout &ids_layout [[buffer(8)]],
    constant TensorLayout &output_layout [[buffer(9)]],
    constant QuantEmbedParams &params [[buffer(10)]],
    device atomic_uint *error_flag [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= output_layout.element_count) {
        return;
    }
    uint token = index / params.width;
    uint column = index % params.width;
    uint id = ids[physical_index(ids_layout, token)];
    if (id >= params.vocab) {
        store_float(output, physical_index(output_layout, index), output_dtype, 0.0f);
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, id, memory_order_relaxed);
        return;
    }
    uint values_per_word = 32 / quant_bits;
    uint packed_width = params.width / values_per_word;
    uint groups = params.width / quant_group_size;
    ulong packed_index = ulong(id) * ulong(packed_width) + ulong(column / values_per_word);
    ulong group_index = ulong(id) * ulong(groups) + ulong(column / quant_group_size);
    if (packed_index > 0xfffffffful || group_index > 0xfffffffful) {
        store_float(output, physical_index(output_layout, index), output_dtype, 0.0f);
        atomic_store_explicit(error_flag, 2, memory_order_relaxed);
        return;
    }
    uint word = load_uint(
        packed,
        physical_index(packed_layout, uint(packed_index)));
    uint shift = (column % values_per_word) * quant_bits;
    uint quantized = (word >> shift) & ((1u << quant_bits) - 1u);
    float scale = load_float(
        scales, physical_index(scale_layout, uint(group_index)), input1_dtype);
    float bias = load_float(
        biases, physical_index(bias_layout, uint(group_index)), input1_dtype);
    store_float(
        output,
        physical_index(output_layout, index),
        output_dtype,
        scale * float(quantized) + bias);
}

struct QuantMatmulParams {
    ulong input_offset;
    ulong packed_offset;
    ulong scale_offset;
    ulong bias_offset;
    ulong output_offset;
    ulong input_row_stride;
    ulong input_inner_stride;
    ulong packed_row_stride;
    ulong packed_word_stride;
    ulong scale_row_stride;
    ulong scale_group_stride;
    ulong bias_row_stride;
    ulong bias_group_stride;
    ulong output_row_stride;
    ulong output_column_stride;
    uint rows;
    uint inner;
    uint columns;
    uint packed_width;
    uint bits;
    uint group_size;
};

float4 load_float4_contiguous(device const uchar *buffer, ulong index, uint dtype) {
    if (dtype == 0) {
        return *reinterpret_cast<device const float4 *>(buffer + index * 4);
    }
    if (dtype == 1) {
        return float4(*reinterpret_cast<device const half4 *>(buffer + index * 2));
    }
    return float4(*reinterpret_cast<device const bfloat4 *>(buffer + index * 2));
}

kernel void quantized_gemv(
    device const uchar *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device uchar *output [[buffer(4)]],
    constant QuantMatmulParams &params [[buffer(5)]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint first_column = tile.x * 16 + uint(simdgroup) * 8;
    uint values_per_word = 32 / quant_bits;
    uint words_per_group = quant_group_size / values_per_word;
    uint mask = (1u << quant_bits) - 1u;
    float sums[8] = {0.0f};
    for (uint first_word = uint(lane) * 4;
         first_word < params.packed_width;
         first_word += 128) {
        uint group = first_word / words_per_group;
        float input_sum = 0.0f;
        float quantized_sums[8] = {0.0f};
        uint4 quantized_words[8] = {uint4(0u)};
        for (uint row = 0; row < 8; ++row) {
            uint column = first_column + row;
            if (column < params.columns) {
                ulong index = params.packed_offset +
                    ulong(column) * params.packed_row_stride +
                    ulong(first_word) * params.packed_word_stride;
                if (params.packed_word_stride == 1 && (index & 3) == 0) {
                    quantized_words[row] =
                        *(device const uint4 *)(packed + index * sizeof(uint));
                } else {
                    for (uint word = 0; word < 4; ++word) {
                        quantized_words[row][word] = load_uint(
                            packed,
                            index + ulong(word) * params.packed_word_stride);
                    }
                }
            }
        }
        for (uint word = 0; word < 4; ++word) {
            uint word_index = first_word + word;
            uint inner = word_index * values_per_word;
            if (params.input_inner_stride == 1 &&
                ((params.input_offset + ulong(inner)) & 3) == 0) {
                float4 first = load_float4_contiguous(
                    input,
                    params.input_offset + ulong(inner),
                    input0_dtype);
                input_sum += dot(first, float4(1.0f));
                if (quant_bits == 4) {
                    float4 second = load_float4_contiguous(
                        input,
                        params.input_offset + ulong(inner + 4),
                        input0_dtype);
                    input_sum += dot(second, float4(1.0f));
                    float4 shifts = float4(1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
                    first *= shifts;
                    second *= shifts;
                    for (uint row = 0; row < 8; ++row) {
                        ushort2 packed_halves =
                            as_type<ushort2>(quantized_words[row][word]);
                        uint4 low = uint4(
                            packed_halves[0] & 0x000fu,
                            packed_halves[0] & 0x00f0u,
                            packed_halves[0] & 0x0f00u,
                            packed_halves[0] & 0xf000u);
                        uint4 high = uint4(
                            packed_halves[1] & 0x000fu,
                            packed_halves[1] & 0x00f0u,
                            packed_halves[1] & 0x0f00u,
                            packed_halves[1] & 0xf000u);
                        quantized_sums[row] +=
                            dot(first, float4(low)) + dot(second, float4(high));
                    }
                } else {
                    for (uint row = 0; row < 8; ++row) {
                        uchar4 values = as_type<uchar4>(quantized_words[row][word]);
                        quantized_sums[row] += dot(first, float4(values));
                    }
                }
            } else {
                for (uint element = 0; element < values_per_word; ++element) {
                    float value = load_float(
                        input,
                        params.input_offset +
                            ulong(inner + element) * params.input_inner_stride,
                        input0_dtype);
                    input_sum += value;
                    uint shift = element * quant_bits;
                    for (uint row = 0; row < 8; ++row) {
                        uint quantized = (quantized_words[row][word] >> shift) & mask;
                        quantized_sums[row] += value * float(quantized);
                    }
                }
            }
        }
        for (uint row = 0; row < 8; ++row) {
            uint column = first_column + row;
            if (column < params.columns) {
                ulong scale_index = params.scale_offset +
                    ulong(column) * params.scale_row_stride +
                    ulong(group) * params.scale_group_stride;
                ulong bias_index = params.bias_offset +
                    ulong(column) * params.bias_row_stride +
                    ulong(group) * params.bias_group_stride;
                sums[row] += load_float(scales, scale_index, input1_dtype) *
                    quantized_sums[row] +
                    load_float(biases, bias_index, input1_dtype) * input_sum;
            }
        }
    }
    for (uint row = 0; row < 8; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    if (lane == 0) {
        for (uint row = 0; row < 8; ++row) {
            uint column = first_column + row;
            if (column < params.columns) {
                store_float(
                    output,
                    params.output_offset + ulong(column) * params.output_column_stride,
                    output_dtype,
                    sums[row]);
            }
        }
    }
}

struct GatherQuantMatmulParams {
    ulong input_offset;
    ulong packed_offset;
    ulong scale_offset;
    ulong bias_offset;
    ulong indices_offset;
    ulong output_offset;
    ulong input_row_stride;
    ulong input_inner_stride;
    ulong packed_expert_stride;
    ulong packed_row_stride;
    ulong packed_word_stride;
    ulong scale_expert_stride;
    ulong scale_row_stride;
    ulong scale_group_stride;
    ulong bias_expert_stride;
    ulong bias_row_stride;
    ulong bias_group_stride;
    ulong indices_row_stride;
    ulong indices_slot_stride;
    ulong output_row_stride;
    ulong output_slot_stride;
    ulong output_column_stride;
    uint rows;
    uint routes;
    uint inner;
    uint experts;
    uint columns;
    uint packed_width;
    uint bits;
    uint group_size;
};

void gather_quantized_sums(
    device const uchar *input,
    device const uchar *packed,
    device const uchar *scales,
    device const uchar *biases,
    constant GatherQuantMatmulParams &params,
    uint row,
    uint expert,
    uint first_column,
    ushort lane,
    thread float (&sums)[8]) {
    uint values_per_word = 32 / quant_bits;
    uint words_per_group = quant_group_size / values_per_word;
    uint mask = (1u << quant_bits) - 1u;
    for (uint first_word = uint(lane) * 4;
         first_word < params.packed_width;
         first_word += 128) {
        uint group = first_word / words_per_group;
        float input_sum = 0.0f;
        float quantized_sums[8] = {0.0f};
        uint4 quantized_words[8] = {uint4(0u)};
        for (uint item = 0; item < 8; ++item) {
            uint column = first_column + item;
            if (column < params.columns) {
                ulong index = params.packed_offset +
                    ulong(expert) * params.packed_expert_stride +
                    ulong(column) * params.packed_row_stride +
                    ulong(first_word) * params.packed_word_stride;
                if (params.packed_word_stride == 1 && (index & 3) == 0) {
                    quantized_words[item] =
                        *(device const uint4 *)(packed + index * sizeof(uint));
                } else {
                    for (uint word = 0; word < 4; ++word) {
                        quantized_words[item][word] = load_uint(
                            packed,
                            index + ulong(word) * params.packed_word_stride);
                    }
                }
            }
        }
        for (uint word = 0; word < 4; ++word) {
            uint word_index = first_word + word;
            uint inner = word_index * values_per_word;
            if (params.input_inner_stride == 1 &&
                ((params.input_offset + ulong(row) * params.input_row_stride +
                  ulong(inner)) & 3) == 0) {
                ulong input_index = params.input_offset +
                    ulong(row) * params.input_row_stride + ulong(inner);
                float4 first = load_float4_contiguous(input, input_index, input0_dtype);
                input_sum += dot(first, float4(1.0f));
                if (quant_bits == 4) {
                    float4 second = load_float4_contiguous(
                        input, input_index + 4, input0_dtype);
                    input_sum += dot(second, float4(1.0f));
                    float4 shifts = float4(
                        1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
                    first *= shifts;
                    second *= shifts;
                    for (uint item = 0; item < 8; ++item) {
                        ushort2 halves = as_type<ushort2>(quantized_words[item][word]);
                        uint4 low = uint4(
                            halves[0] & 0x000fu, halves[0] & 0x00f0u,
                            halves[0] & 0x0f00u, halves[0] & 0xf000u);
                        uint4 high = uint4(
                            halves[1] & 0x000fu, halves[1] & 0x00f0u,
                            halves[1] & 0x0f00u, halves[1] & 0xf000u);
                        quantized_sums[item] +=
                            dot(first, float4(low)) + dot(second, float4(high));
                    }
                } else {
                    for (uint item = 0; item < 8; ++item) {
                        uchar4 values = as_type<uchar4>(quantized_words[item][word]);
                        quantized_sums[item] += dot(first, float4(values));
                    }
                }
            } else {
                for (uint element = 0; element < values_per_word; ++element) {
                    float value = load_float(
                        input,
                        params.input_offset + ulong(row) * params.input_row_stride +
                            ulong(inner + element) * params.input_inner_stride,
                        input0_dtype);
                    input_sum += value;
                    uint shift = element * quant_bits;
                    for (uint item = 0; item < 8; ++item) {
                        uint quantized = (quantized_words[item][word] >> shift) & mask;
                        quantized_sums[item] += value * float(quantized);
                    }
                }
            }
        }
        for (uint item = 0; item < 8; ++item) {
            uint column = first_column + item;
            if (column < params.columns) {
                ulong scale_index = params.scale_offset +
                    ulong(expert) * params.scale_expert_stride +
                    ulong(column) * params.scale_row_stride +
                    ulong(group) * params.scale_group_stride;
                ulong bias_index = params.bias_offset +
                    ulong(expert) * params.bias_expert_stride +
                    ulong(column) * params.bias_row_stride +
                    ulong(group) * params.bias_group_stride;
                sums[item] += load_float(scales, scale_index, input1_dtype) *
                    quantized_sums[item] +
                    load_float(biases, bias_index, input1_dtype) * input_sum;
            }
        }
    }
    for (uint item = 0; item < 8; ++item) {
        sums[item] = simd_sum(sums[item]);
    }
}

kernel void gather_quantized_gemv(
    device const uchar *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device const uint *indices [[buffer(4)]],
    device uchar *output [[buffer(5)]],
    constant GatherQuantMatmulParams &params [[buffer(6)]],
    device atomic_uint *error_flag [[buffer(7)]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint route = tile.y;
    uint row = route / params.routes;
    uint slot = route % params.routes;
    uint expert = indices[
        params.indices_offset + ulong(row) * params.indices_row_stride +
        ulong(slot) * params.indices_slot_stride];
    bool valid = expert < params.experts;
    if (!valid && lane == 0 && simdgroup == 0) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, expert, memory_order_relaxed);
    }
    uint first_column = tile.x * 16 + uint(simdgroup) * 8;
    float sums[8] = {0.0f};
    if (valid) {
        gather_quantized_sums(
            input, packed, scales, biases, params, row, expert, first_column, lane, sums);
    }
    if (lane == 0) {
        for (uint item = 0; item < 8; ++item) {
            uint column = first_column + item;
            if (column < params.columns) {
                ulong output_index = params.output_offset +
                    ulong(row) * params.output_row_stride +
                    ulong(slot) * params.output_slot_stride +
                    ulong(column) * params.output_column_stride;
                store_float(output, output_index, output_dtype, sums[item]);
            }
        }
    }
}

kernel void gather_quantized_silu_mul(
    device const uchar *input [[buffer(0)]],
    device const uchar *gate_packed [[buffer(1)]],
    device const uchar *gate_scales [[buffer(2)]],
    device const uchar *gate_biases [[buffer(3)]],
    device const uchar *up_packed [[buffer(4)]],
    device const uchar *up_scales [[buffer(5)]],
    device const uchar *up_biases [[buffer(6)]],
    device const uint *indices [[buffer(7)]],
    device uchar *output [[buffer(8)]],
    constant GatherQuantMatmulParams &gate_params [[buffer(9)]],
    constant GatherQuantMatmulParams &up_params [[buffer(10)]],
    device atomic_uint *error_flag [[buffer(11)]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint route = tile.y;
    uint row = route / gate_params.routes;
    uint slot = route % gate_params.routes;
    uint expert = indices[
        gate_params.indices_offset + ulong(row) * gate_params.indices_row_stride +
        ulong(slot) * gate_params.indices_slot_stride];
    bool valid = expert < gate_params.experts;
    if (!valid && lane == 0 && simdgroup == 0) {
        atomic_store_explicit(error_flag, 1, memory_order_relaxed);
        atomic_fetch_min_explicit(error_flag + 1, expert, memory_order_relaxed);
    }
    bool gate_projection = simdgroup < 2;
    uint projection_simdgroup = uint(simdgroup) & 1u;
    uint first_column = tile.x * 16 + projection_simdgroup * 8;
    float sums[8] = {0.0f};
    if (valid) {
        if (gate_projection) {
            gather_quantized_sums(
                input, gate_packed, gate_scales, gate_biases, gate_params,
                row, expert, first_column, lane, sums);
        } else {
            gather_quantized_sums(
                input, up_packed, up_scales, up_biases, up_params,
                row, expert, first_column, lane, sums);
        }
    }
    threadgroup float gate_values[16];
    threadgroup float up_values[16];
    if (lane == 0) {
        for (uint item = 0; item < 8; ++item) {
            uint local_column = projection_simdgroup * 8 + item;
            if (gate_projection) {
                gate_values[local_column] = sums[item];
            } else {
                up_values[local_column] = sums[item];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && simdgroup < 2) {
        for (uint item = 0; item < 8; ++item) {
            uint local_column = uint(simdgroup) * 8 + item;
            uint column = tile.x * 16 + local_column;
            if (column < gate_params.columns) {
                float gate = gate_values[local_column];
                float result = gate / (1.0f + exp(-gate)) * up_values[local_column];
                ulong output_index = gate_params.output_offset +
                    ulong(row) * gate_params.output_row_stride +
                    ulong(slot) * gate_params.output_slot_stride +
                    ulong(column) * gate_params.output_column_stride;
                store_float(output, output_index, output_dtype, result);
            }
        }
    }
}

kernel void quantized_gemm_small_m(
    device const uchar *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device uchar *output [[buffer(4)]],
    constant QuantMatmulParams &params [[buffer(5)]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint column = tile.x * 8 + uint(simdgroup);
    uint first_row = tile.y * 8;
    if (column >= params.columns) {
        return;
    }
    uint values_per_word = 32 / quant_bits;
    uint words_per_group = quant_group_size / values_per_word;
    uint groups = params.inner / quant_group_size;
    float sums[8] = {0.0f};
    for (uint group = uint(lane); group < groups; group += 32) {
        float input_sums[8] = {0.0f};
        float quantized_sums[8] = {0.0f};
        uint first_word = group * words_per_group;
        for (uint word_in_group = 0; word_in_group < words_per_group; ++word_in_group) {
            uint word_index = first_word + word_in_group;
            ulong packed_index = params.packed_offset +
                ulong(column) * params.packed_row_stride +
                ulong(word_index) * params.packed_word_stride;
            uint quantized_word = load_uint(packed, packed_index);
            for (uint element = 0; element < values_per_word; ++element) {
                uint inner = word_index * values_per_word + element;
                uint shift = element * quant_bits;
                uint quantized = (quantized_word >> shift) & ((1u << quant_bits) - 1u);
                for (uint row = 0; row < 8; ++row) {
                    uint input_row = first_row + row;
                    if (input_row < params.rows) {
                        ulong input_index = params.input_offset +
                            ulong(input_row) * params.input_row_stride +
                            ulong(inner) * params.input_inner_stride;
                        float value = load_float(input, input_index, input0_dtype);
                        input_sums[row] += value;
                        quantized_sums[row] += value * float(quantized);
                    }
                }
            }
        }
        ulong scale_index = params.scale_offset +
            ulong(column) * params.scale_row_stride +
            ulong(group) * params.scale_group_stride;
        ulong bias_index = params.bias_offset +
            ulong(column) * params.bias_row_stride +
            ulong(group) * params.bias_group_stride;
        float scale = load_float(scales, scale_index, input1_dtype);
        float bias = load_float(biases, bias_index, input1_dtype);
        for (uint row = 0; row < 8; ++row) {
            sums[row] += scale * quantized_sums[row] + bias * input_sums[row];
        }
    }
    for (uint row = 0; row < 8; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    if (lane == 0) {
        for (uint row = 0; row < 8; ++row) {
            uint output_row = first_row + row;
            if (output_row < params.rows) {
                ulong output_index = params.output_offset +
                    ulong(output_row) * params.output_row_stride +
                    ulong(column) * params.output_column_stride;
                store_float(output, output_index, output_dtype, sums[row]);
            }
        }
    }
}
