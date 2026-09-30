constant uint quant_bits [[function_constant(12)]];
constant uint quant_group_size [[function_constant(13)]];

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
    uint first_column = tile.x * 32 + uint(simdgroup) * 4;
    uint values_per_word = 32 / quant_bits;
    uint words_per_group = quant_group_size / values_per_word;
    uint groups = params.inner / quant_group_size;
    float sums[4] = {0.0f};
    for (uint group = uint(lane); group < groups; group += 32) {
        float input_sum = 0.0f;
        float quantized_sums[4] = {0.0f};
        uint first_word = group * words_per_group;
        for (uint word_in_group = 0; word_in_group < words_per_group; ++word_in_group) {
            uint word_index = first_word + word_in_group;
            uint quantized_words[4] = {0u};
            for (uint row = 0; row < 4; ++row) {
                uint column = first_column + row;
                if (column < params.columns) {
                    ulong index = params.packed_offset +
                        ulong(column) * params.packed_row_stride +
                        ulong(word_index) * params.packed_word_stride;
                    quantized_words[row] = load_uint(packed, index);
                }
            }
            for (uint element = 0; element < values_per_word; ++element) {
                uint inner = word_index * values_per_word + element;
                float value = load_float(
                    input,
                    params.input_offset + ulong(inner) * params.input_inner_stride,
                    input0_dtype);
                input_sum += value;
                uint shift = element * quant_bits;
                uint mask = (1u << quant_bits) - 1u;
                for (uint row = 0; row < 4; ++row) {
                    uint quantized = (quantized_words[row] >> shift) & mask;
                    quantized_sums[row] += value * float(quantized);
                }
            }
        }
        for (uint row = 0; row < 4; ++row) {
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
    for (uint row = 0; row < 4; ++row) {
        sums[row] = simd_sum(sums[row]);
    }
    if (lane == 0) {
        for (uint row = 0; row < 4; ++row) {
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
