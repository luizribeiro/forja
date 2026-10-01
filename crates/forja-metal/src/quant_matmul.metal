// Copyright © 2023-2024 Apple Inc.
// SPDX-License-Identifier: MIT

constant uint quant_bits [[function_constant(12)]];
constant uint quant_group_size [[function_constant(13)]];
constant uint gather_words_per_thread [[function_constant(14)]];

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
    uint block_capacity;
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
    thread float (&sums)[4]) {
    uint values_per_word = 32 / quant_bits;
    uint values_per_thread = values_per_word * gather_words_per_thread;
    uint block_size = values_per_thread * 32;
    for (uint block = 0; block < params.inner; block += block_size) {
        uint first_inner = block + uint(lane) * values_per_thread;
        if (first_inner >= params.inner) {
            continue;
        }
        uint group = first_inner / quant_group_size;
        float input_sum = 0.0f;
        float activations[16] = {0.0f};
        for (uint element = 0; element < values_per_thread; element += 4) {
            ulong input_index = params.input_offset +
                ulong(row) * params.input_row_stride +
                ulong(first_inner + element) * params.input_inner_stride;
            float4 values;
            if (params.input_inner_stride == 1 && (input_index & 3) == 0) {
                values = load_float4_contiguous(input, input_index, input0_dtype);
            } else {
                for (uint item = 0; item < 4; ++item) {
                    values[item] = load_float(
                        input,
                        input_index + ulong(item) * params.input_inner_stride,
                        input0_dtype);
                }
            }
            input_sum += dot(values, float4(1.0f));
            if (quant_bits == 4) {
                values *= float4(
                    1.0f, 1.0f / 16.0f, 1.0f / 256.0f, 1.0f / 4096.0f);
            }
            for (uint item = 0; item < 4; ++item) {
                activations[element + item] = values[item];
            }
        }
        for (uint item = 0; item < 4; ++item) {
            uint column = first_column + item;
            if (column < params.columns) {
                uint first_word = first_inner / values_per_word;
                ulong index = params.packed_offset +
                    ulong(expert) * params.packed_expert_stride +
                    ulong(column) * params.packed_row_stride +
                    ulong(first_word) * params.packed_word_stride;
                uint2 words = uint2(0u);
                if (params.packed_word_stride == 1 && gather_words_per_thread == 2) {
                    words = *(device const uint2 *)(packed + index * sizeof(uint));
                } else {
                    for (uint word = 0; word < gather_words_per_thread; ++word) {
                        words[word] = load_uint(
                            packed,
                            index + ulong(word) * params.packed_word_stride);
                    }
                }
                float quantized_sum = 0.0f;
                if (quant_bits == 4) {
                    for (uint word = 0; word < gather_words_per_thread; ++word) {
                        ushort2 halves = as_type<ushort2>(words[word]);
                        for (uint part = 0; part < 2; ++part) {
                            uint activation = word * 8 + part * 4;
                            float4 values = float4(
                                halves[part] & 0x000fu,
                                halves[part] & 0x00f0u,
                                halves[part] & 0x0f00u,
                                halves[part] & 0xf000u);
                            quantized_sum += dot(
                                float4(
                                    activations[activation],
                                    activations[activation + 1],
                                    activations[activation + 2],
                                    activations[activation + 3]),
                                values);
                        }
                    }
                } else {
                    for (uint word = 0; word < gather_words_per_thread; ++word) {
                        uchar4 values = as_type<uchar4>(words[word]);
                        uint activation = word * 4;
                        quantized_sum += dot(
                            float4(
                                activations[activation],
                                activations[activation + 1],
                                activations[activation + 2],
                                activations[activation + 3]),
                            float4(values));
                    }
                }
                ulong scale_index = params.scale_offset +
                    ulong(expert) * params.scale_expert_stride +
                    ulong(column) * params.scale_row_stride +
                    ulong(group) * params.scale_group_stride;
                ulong bias_index = params.bias_offset +
                    ulong(expert) * params.bias_expert_stride +
                    ulong(column) * params.bias_row_stride +
                    ulong(group) * params.bias_group_stride;
                sums[item] += load_float(scales, scale_index, input1_dtype) *
                    quantized_sum +
                    load_float(biases, bias_index, input1_dtype) * input_sum;
            }
        }
    }
    for (uint item = 0; item < 4; ++item) {
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
    uint first_column = tile.x * 8 + uint(simdgroup) * 4;
    float sums[4] = {0.0f};
    if (valid) {
        gather_quantized_sums(
            input, packed, scales, biases, params, row, expert, first_column, lane, sums);
    }
    if (lane == 0) {
        for (uint item = 0; item < 4; ++item) {
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
    uint first_column = tile.x * 8 + projection_simdgroup * 4;
    float sums[4] = {0.0f};
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
    threadgroup float gate_values[8];
    threadgroup float up_values[8];
    if (lane == 0) {
        for (uint item = 0; item < 4; ++item) {
            uint local_column = projection_simdgroup * 4 + item;
            if (gate_projection) {
                gate_values[local_column] = sums[item];
            } else {
                up_values[local_column] = sums[item];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && simdgroup < 2) {
        for (uint item = 0; item < 4; ++item) {
            uint local_column = uint(simdgroup) * 4 + item;
            uint column = tile.x * 8 + local_column;
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

template <
    typename T, uint block_inner, uint block_columns,
    uint tile_stride, bool transpose>
void load_quantized_weight_tile(
    device const uchar *packed,
    device const uchar *scales,
    device const uchar *biases,
    ulong packed_base,
    ulong scale_base,
    ulong bias_base,
    ulong packed_row_stride,
    ulong packed_word_stride,
    ulong scale_row_stride,
    ulong scale_group_stride,
    ulong bias_row_stride,
    ulong bias_group_stride,
    uint inner_origin,
    uint column_origin,
    uint inner_extent,
    uint column_extent,
    threadgroup T *tile,
    uint thread_index,
    uint thread_count);

kernel void count_expert_routes(
    device const uint *indices [[buffer(0)]],
    device atomic_uint *counts [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    constant GatherQuantMatmulParams &params [[buffer(3)]],
    device atomic_uint *error_flag [[buffer(4)]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]) {
    uint route_count = params.rows * params.routes;
    for (uint expert = thread_index; expert <= params.experts; expert += thread_count) {
        atomic_store_explicit(counts + expert, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint route = thread_index; route < route_count; route += thread_count) {
        uint row = route / params.routes;
        uint slot = route % params.routes;
        uint selected = indices[
            params.indices_offset + ulong(row) * params.indices_row_stride +
            ulong(slot) * params.indices_slot_stride];
        if (selected < params.experts) {
            atomic_fetch_add_explicit(counts + selected, 1, memory_order_relaxed);
        } else {
            for (uint column = 0; column < params.columns; ++column) {
                store_float(
                    output,
                    params.output_offset + ulong(row) * params.output_row_stride +
                        ulong(slot) * params.output_slot_stride +
                        ulong(column) * params.output_column_stride,
                    output_dtype,
                    0.0f);
            }
            atomic_store_explicit(error_flag, 1, memory_order_relaxed);
            atomic_fetch_min_explicit(error_flag + 1, selected, memory_order_relaxed);
        }
    }
}

kernel void arrange_expert_routes(
    device const uint *indices [[buffer(0)]],
    device uint *sorted_routes [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    device uint3 *blocks [[buffer(3)]],
    constant GatherQuantMatmulParams &params [[buffer(4)]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]) {
    threadgroup uint scan[256];
    uint route_count = params.rows * params.routes;
    uint count = thread_index < params.experts
        ? atomic_load_explicit(counts + thread_index, memory_order_relaxed)
        : 0;
    scan[thread_index] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint shift = 1; shift < 256; shift <<= 1) {
        uint prefix = scan[thread_index];
        if (thread_index >= shift) {
            prefix += scan[thread_index - shift];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[thread_index] = prefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint route_offset = thread_index == 0 ? 0 : scan[thread_index - 1];
    if (thread_index < params.experts) {
        atomic_store_explicit(counts + thread_index, route_offset, memory_order_relaxed);
    }
    if (thread_index + 1 == params.experts) {
        atomic_store_explicit(counts + params.experts, scan[thread_index], memory_order_relaxed);
    }
    scan[thread_index] = count == 0 ? 0 : (count + 15) / 16;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint shift = 1; shift < 256; shift <<= 1) {
        uint prefix = scan[thread_index];
        if (thread_index >= shift) {
            prefix += scan[thread_index - shift];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan[thread_index] = prefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint block = thread_index; block < params.block_capacity; block += thread_count) {
        blocks[block] = uint3(0xffffffffu, 0, 0);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index < params.experts) {
        uint sorted = route_offset;
        for (uint route = 0; route < route_count; ++route) {
            uint row = route / params.routes;
            uint slot = route % params.routes;
            uint selected = indices[
                params.indices_offset + ulong(row) * params.indices_row_stride +
                ulong(slot) * params.indices_slot_stride];
            if (selected == thread_index) {
                sorted_routes[sorted++] = route;
            }
        }
        uint block = thread_index == 0 ? 0 : scan[thread_index - 1];
        for (uint start = 0; start < count; start += 16) {
            blocks[block++] = uint3(thread_index, route_offset + start, min(16u, count - start));
        }
    }
}

kernel void grouped_quantized_gemm(
    device const uchar *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device const uint *sorted_routes [[buffer(4)]],
    device const uint3 *blocks [[buffer(5)]],
    device uchar *output [[buffer(6)]],
    constant GatherQuantMatmulParams &params [[buffer(7)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_size [[threads_per_threadgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tile [[threadgroup_position_in_grid]]) {
    constexpr uint block_rows = 16;
    constexpr uint block_columns = 32;
    constexpr uint block_inner = 32;
    constexpr uint simdgroups_rows = 1;
    constexpr uint simdgroups_columns = 2;
    constexpr uint tile_stride = block_inner + 4;
    threadgroup float input_tile[block_rows * tile_stride];
    threadgroup float weight_tile[block_columns * tile_stride];
    uint3 block = blocks[tile.y];
    if (block.x == 0xffffffffu) {
        return;
    }
    uint expert = block.x;
    uint sorted_origin = block.y;
    uint column_origin = tile.x * block_columns;
    uint row_count = block.z;
    uint thread_count = threadgroup_size.x * threadgroup_size.y * threadgroup_size.z;
    BlockMMA<
        float, block_rows, block_columns, block_inner,
        simdgroups_rows, simdgroups_columns,
        tile_stride, tile_stride, true> mma(simdgroup, lane);
    for (uint inner_origin = 0; inner_origin < params.inner; inner_origin += block_inner) {
        if (thread_count == 64 && input0_dtype == 0 &&
            inner_origin + block_inner <= params.inner &&
            params.input_offset % 4 == 0 && params.input_row_stride % 4 == 0 &&
            params.input_inner_stride == 1) {
            uint local_row = thread_index / 4;
            uint local_inner = (thread_index % 4) * 8;
            threadgroup float *destination =
                input_tile + local_row * tile_stride + local_inner;
            if (local_row < row_count) {
                uint route = sorted_routes[sorted_origin + local_row];
                uint input_row = route / params.routes;
                device const float *source = reinterpret_cast<device const float *>(input) +
                    params.input_offset + ulong(input_row) * params.input_row_stride +
                    inner_origin + local_inner;
                device const float4 *source_vectors =
                    reinterpret_cast<device const float4 *>(source);
                threadgroup float4 *destination_vectors =
                    reinterpret_cast<threadgroup float4 *>(destination);
#pragma clang loop unroll(full)
                for (uint read = 0; read < 2; ++read) {
                    destination_vectors[read] = source_vectors[read];
                }
            } else {
#pragma clang loop unroll(full)
                for (uint element = 0; element < 8; ++element) {
                    destination[element] = 0.0f;
                }
            }
        } else {
            for (uint linear = thread_index;
                 linear < block_rows * block_inner;
                 linear += thread_count) {
                uint local_row = linear / block_inner;
                uint local_inner = linear % block_inner;
                uint inner = inner_origin + local_inner;
                float value = 0.0f;
                if (local_row < row_count && inner < params.inner) {
                    uint route = sorted_routes[sorted_origin + local_row];
                    uint input_row = route / params.routes;
                    value = load_float(
                        input,
                        params.input_offset + ulong(input_row) * params.input_row_stride +
                            ulong(inner) * params.input_inner_stride,
                        input0_dtype);
                }
                input_tile[local_row * tile_stride + local_inner] = value;
            }
        }
        load_quantized_weight_tile<
            float, block_inner, block_columns, tile_stride, true>(
            packed, scales, biases,
            params.packed_offset + ulong(expert) * params.packed_expert_stride,
            params.scale_offset + ulong(expert) * params.scale_expert_stride,
            params.bias_offset + ulong(expert) * params.bias_expert_stride,
            params.packed_row_stride, params.packed_word_stride,
            params.scale_row_stride, params.scale_group_stride,
            params.bias_row_stride, params.bias_group_stride,
            inner_origin, column_origin, params.inner, params.columns,
            weight_tile, thread_index, thread_count);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.multiply(input_tile, weight_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row_fragment = 0;
         row_fragment < block_rows / (8 * simdgroups_rows);
         ++row_fragment) {
            uint local_row = uint(mma.simdgroup_row) * 8 + uint(mma.fragment_row) +
                row_fragment * 8 * simdgroups_rows;
            if (local_row < row_count) {
                uint route = sorted_routes[sorted_origin + local_row];
                uint output_row = route / params.routes;
                uint output_slot = route % params.routes;
                for (uint column_fragment = 0;
                     column_fragment < block_columns / (8 * simdgroups_columns);
                     ++column_fragment) {
                    uint column = column_origin + uint(mma.simdgroup_column) * 8 +
                        uint(mma.fragment_column) +
                        column_fragment * 8 * simdgroups_columns;
                    uint fragment_index =
                        row_fragment * (block_columns / (8 * simdgroups_columns)) +
                        column_fragment;
                    for (uint element = 0; element < 2; ++element) {
                        if (column + element < params.columns) {
                            ulong output_index = params.output_offset +
                                ulong(output_row) * params.output_row_stride +
                                ulong(output_slot) * params.output_slot_stride +
                                ulong(column + element) * params.output_column_stride;
                            store_float(
                                output, output_index, output_dtype,
                                float(mma.accumulators[fragment_index * 2 + element]));
                        }
                    }
                }
            }
    }
}

kernel void gathered_silu_from_projections(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device const uint *indices [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant GatherQuantMatmulParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    uint element_count = params.rows * params.routes * params.columns;
    if (index >= element_count) {
        return;
    }
    uint column = index % params.columns;
    uint route = index / params.columns;
    uint row = route / params.routes;
    uint slot = route % params.routes;
    uint expert = indices[
        params.indices_offset + ulong(row) * params.indices_row_stride +
        ulong(slot) * params.indices_slot_stride];
    if (expert >= params.experts) {
        store_float(
            output,
            params.output_offset + ulong(row) * params.output_row_stride +
                ulong(slot) * params.output_slot_stride +
                ulong(column) * params.output_column_stride,
            output_dtype,
            0.0f);
        return;
    }
    float gate_value = gate[index];
    float result = gate_value / (1.0f + exp(-gate_value)) * up[index];
    store_float(
        output,
        params.output_offset + ulong(row) * params.output_row_stride +
            ulong(slot) * params.output_slot_stride +
            ulong(column) * params.output_column_stride,
        output_dtype,
        result);
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

template <
    typename T, uint block_inner, uint block_columns,
    uint tile_stride, bool transpose>
void load_quantized_weight_tile(
    device const uchar *packed,
    device const uchar *scales,
    device const uchar *biases,
    ulong packed_base,
    ulong scale_base,
    ulong bias_base,
    ulong packed_row_stride,
    ulong packed_word_stride,
    ulong scale_row_stride,
    ulong scale_group_stride,
    ulong bias_row_stride,
    ulong bias_group_stride,
    uint inner_origin,
    uint column_origin,
    uint inner_extent,
    uint column_extent,
    threadgroup T *tile,
    uint thread_index,
    uint thread_count) {
    uint values_per_word = 32 / quant_bits;
    uint words_per_column = block_inner / values_per_word;
    uint mask = (1u << quant_bits) - 1u;
    for (uint linear = thread_index;
         linear < block_columns * words_per_column;
         linear += thread_count) {
        uint local_column = linear / words_per_column;
        uint local_word = linear % words_per_column;
        uint column = column_origin + local_column;
        uint inner = inner_origin + local_word * values_per_word;
        uint word = 0;
        float scale = 0.0f;
        float bias = 0.0f;
        if (inner < inner_extent && column < column_extent) {
            word = load_uint(
                packed,
                packed_base + ulong(column) * packed_row_stride +
                    ulong(inner / values_per_word) * packed_word_stride);
            uint group = inner / quant_group_size;
            scale = load_float(
                scales,
                scale_base + ulong(column) * scale_row_stride +
                    ulong(group) * scale_group_stride,
                input1_dtype);
            bias = load_float(
                biases,
                bias_base + ulong(column) * bias_row_stride +
                    ulong(group) * bias_group_stride,
                input1_dtype);
        }
        for (uint element = 0; element < values_per_word; ++element) {
            uint local_inner = local_word * values_per_word + element;
            float value = inner + element < inner_extent
                ? scale * float((word >> (element * quant_bits)) & mask) + bias
                : 0.0f;
            uint tile_index = transpose
                ? local_column * tile_stride + local_inner
                : local_inner * tile_stride + local_column;
            tile[tile_index] = T(value);
        }
    }
}

template <typename T, uint block_inner, uint block_columns, uint tile_stride>
void load_transposed_quantized_weight_tile(
    device const uchar *packed,
    device const uchar *scales,
    device const uchar *biases,
    constant QuantMatmulParams &params,
    uint inner_origin,
    uint column_origin,
    threadgroup T *tile,
    uint thread_index,
    uint thread_count) {
    uint values_per_word = 32 / quant_bits;
    uint words_per_column = block_inner / values_per_word;
    uint mask = (1u << quant_bits) - 1u;
    for (uint linear = thread_index;
         linear < block_columns * words_per_column;
         linear += thread_count) {
        uint local_column = linear / words_per_column;
        uint local_word = linear % words_per_column;
        uint column = column_origin + local_column;
        uint inner = inner_origin + local_word * values_per_word;
        uint word = 0;
        float scale = 0.0f;
        float bias = 0.0f;
        if (inner < params.inner && column < params.columns) {
            word = load_uint(
                packed,
                params.packed_offset + ulong(column) * params.packed_row_stride +
                    ulong(inner / values_per_word) * params.packed_word_stride);
            uint group = inner / quant_group_size;
            scale = load_float(
                scales,
                params.scale_offset + ulong(column) * params.scale_row_stride +
                    ulong(group) * params.scale_group_stride,
                input1_dtype);
            bias = load_float(
                biases,
                params.bias_offset + ulong(column) * params.bias_row_stride +
                    ulong(group) * params.bias_group_stride,
                input1_dtype);
        }
        #pragma clang loop unroll(full)
        for (uint element = 0; element < values_per_word; ++element) {
            float value = inner + element < params.inner
                ? scale * float((word >> (element * quant_bits)) & mask) + bias
                : 0.0f;
            tile[local_column * tile_stride + local_word * values_per_word + element] =
                T(value);
        }
    }
}

struct alignas(16) QuantizedGemmRead16 {
    uchar bytes[16];
};

template <typename T, uint tile_stride>
void load_quantized_gemm_input_tile(
    device const uchar *input,
    constant QuantMatmulParams &params,
    uint row_origin,
    uint inner_origin,
    threadgroup T *tile,
    uint thread_index,
    uint thread_count) {
    constexpr uint block_rows = 32;
    constexpr uint block_inner = 32;
    constexpr uint elements_per_thread = block_rows * block_inner / 128;
    constexpr uint vector_elements = 16 / sizeof(T);
    constexpr uint reads_per_thread = elements_per_thread / vector_elements;
    if (thread_count == 128 && row_origin + block_rows <= params.rows &&
        inner_origin + block_inner <= params.inner &&
        params.input_offset % vector_elements == 0 &&
        params.input_row_stride % vector_elements == 0 &&
        params.input_inner_stride == 1) {
        uint local_row = thread_index / 4;
        uint local_inner = (thread_index % 4) * elements_per_thread;
        device const T *source = reinterpret_cast<device const T *>(input) +
            params.input_offset + ulong(row_origin + local_row) * params.input_row_stride +
            inner_origin + local_inner;
        threadgroup T *destination = tile + local_row * tile_stride + local_inner;
        #pragma clang loop unroll(full)
        for (uint read = 0; read < reads_per_thread; ++read) {
            *reinterpret_cast<threadgroup QuantizedGemmRead16 *>(
                destination + read * vector_elements) =
                *reinterpret_cast<device const QuantizedGemmRead16 *>(
                    source + read * vector_elements);
        }
        return;
    }
    for (uint linear = thread_index;
         linear < block_rows * block_inner;
         linear += thread_count) {
        uint local_row = linear / block_inner;
        uint local_inner = linear % block_inner;
        uint row = row_origin + local_row;
        uint inner = inner_origin + local_inner;
        tile[local_row * tile_stride + local_inner] =
            row < params.rows && inner < params.inner
                ? T(load_float(
                    input,
                    params.input_offset + ulong(row) * params.input_row_stride +
                        ulong(inner) * params.input_inner_stride,
                    input0_dtype))
                : T(0.0f);
    }
}

template <typename T>
void quantized_gemm_tiled_impl(
    device const uchar *input [[buffer(0)]],
    device const uchar *packed [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device uchar *output [[buffer(4)]],
    constant QuantMatmulParams &params [[buffer(5)]],
    uint thread_index,
    uint thread_count,
    ushort simdgroup,
    ushort lane,
    uint3 tile,
    threadgroup T *input_tile,
    threadgroup T *weight_tile) {
    constexpr uint block_rows = 32;
    constexpr uint block_columns = 32;
    constexpr uint block_inner = 32;
    constexpr uint tile_stride = block_inner + 16 / sizeof(T);
    BlockMMA<
        T, block_rows, block_columns, block_inner, 2, 2,
        tile_stride, tile_stride, true, float> mma(simdgroup, lane);
    uint row_origin = tile.y * block_rows;
    uint column_origin = tile.x * block_columns;
    for (uint inner_origin = 0; inner_origin < params.inner; inner_origin += block_inner) {
        load_quantized_gemm_input_tile<T, tile_stride>(
            input, params, row_origin, inner_origin,
            input_tile, thread_index, thread_count);
        load_transposed_quantized_weight_tile<
            T, block_inner, block_columns, tile_stride>(
            packed, scales, biases, params,
            inner_origin, column_origin, weight_tile, thread_index, thread_count);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma.multiply(input_tile, weight_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mma.store(
        output, params.output_offset, params.output_row_stride,
        row_origin, column_origin, params.rows, params.columns, output_dtype);
}

#define quantized_gemm_tiled_kernel(name, type)                                    \
kernel void name(                                                                  \
    device const uchar *input [[buffer(0)]],                                        \
    device const uchar *packed [[buffer(1)]],                                       \
    device const uchar *scales [[buffer(2)]],                                       \
    device const uchar *biases [[buffer(3)]],                                       \
    device uchar *output [[buffer(4)]],                                             \
    constant QuantMatmulParams &params [[buffer(5)]],                               \
    uint thread_index [[thread_index_in_threadgroup]],                              \
    uint3 threadgroup_size [[threads_per_threadgroup]],                             \
    ushort simdgroup [[simdgroup_index_in_threadgroup]],                            \
    ushort lane [[thread_index_in_simdgroup]],                                      \
    uint3 tile [[threadgroup_position_in_grid]]) {                                  \
    constexpr uint tile_stride = 32 + 16 / sizeof(type);                            \
    threadgroup type input_tile[32 * tile_stride];                                  \
    threadgroup type weight_tile[32 * tile_stride];                                 \
    quantized_gemm_tiled_impl<type>(                                                \
        input, packed, scales, biases, output, params, thread_index,                \
        threadgroup_size.x * threadgroup_size.y * threadgroup_size.z,               \
        simdgroup, lane, tile, input_tile, weight_tile);                            \
}

quantized_gemm_tiled_kernel(quantized_gemm_tiled_f32, float)
quantized_gemm_tiled_kernel(quantized_gemm_tiled_f16, half)
quantized_gemm_tiled_kernel(quantized_gemm_tiled_bf16, bfloat)
