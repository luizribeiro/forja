constant uint calibration_iterations [[function_constant(220)]];
constant uint calibration_chains [[function_constant(221)]];

template <typename Stored, typename Arithmetic>
void calibration_mma_impl(device const float *input, device float *output,
                          uint index, ushort lane) {
    simdgroup_matrix<Arithmetic, 8, 8> left[16], right[16];
    simdgroup_matrix<float, 8, 8> accumulators[16];
    float seed = input[index];
    float right_seed = input[index ^ 1u];
    for (uint element = 0; element < 2; ++element) {

        #pragma unroll
        for (uint chain = 0; chain < calibration_chains; ++chain) {
            left[chain].thread_elements()[element] = Arithmetic(Stored(seed * (0.001f + float(chain) * 0.00001f)));
            right[chain].thread_elements()[element] = Arithmetic(Stored(right_seed * 0.002f));
            accumulators[chain].thread_elements()[element] = seed * float(chain + 1);
        }
    }
    for (uint iteration = 0; iteration < calibration_iterations; iteration += 8) {
        #pragma unroll
        for (uint repeat = 0; repeat < 8; ++repeat) {
        #pragma unroll
        for (uint chain = 0; chain < calibration_chains; ++chain) {
            simdgroup_matrix<float, 8, 8> result;
            simdgroup_multiply_accumulate(result, left[chain], right[chain], accumulators[chain]);
            accumulators[chain] = result;
        }
        }
    }
    // Every fragment and chain remains observable, preventing dead-code removal.
    float result = 0;
    #pragma unroll
    for (uint chain = 0; chain < calibration_chains; ++chain) {
        result += accumulators[chain].thread_elements()[0] + accumulators[chain].thread_elements()[1];
    }
    output[index] = result + float(lane) * 0.000001f;
}

#define calibration_mma_kernel(name, stored, arithmetic) \
kernel void name(device const float *input [[buffer(0)]], device float *output [[buffer(1)]], \
                 uint index [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    calibration_mma_impl<stored, arithmetic>(input, output, index, lane); \
}
calibration_mma_kernel(calibration_mma_f32, float, float)
calibration_mma_kernel(calibration_mma_f16, half, half)
calibration_mma_kernel(calibration_mma_bf16, bfloat, float)
#ifdef FORJA_CALIBRATION_NATIVE_BF16
calibration_mma_kernel(calibration_mma_bf16_native, bfloat, bfloat)
#endif

kernel void calibration_threadgroup(device const float *input [[buffer(0)]],
                                   device float *output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]],
                                   uint lane [[thread_position_in_threadgroup]]) {
    threadgroup volatile float4 scratch[128];
    scratch[lane] = float4(input[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sum = 0;
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        sum += scratch[(lane + iteration) & 127];
    }
    output[index] = sum.x + sum.y + sum.z + sum.w;
}

kernel void calibration_scalar(device const float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               uint index [[thread_position_in_grid]]) {
    float seed = input[index];
    float value = seed;
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        value = fma(value, 0.9999999f, seed * 0.0000001f);
    }
    output[index] = value;
}

kernel void calibration_sfu(device const float *input [[buffer(0)]],
                            device float *output [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
    float value = input[index];
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        value = fast::exp(-value);
    }
    output[index] = value;
}

kernel void calibration_gather(device const float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               uint index [[thread_position_in_grid]]) {
    uint address = (index * 0x9e3779b9u) & calibration_iterations;
    output[index] = input[address];
}

#define calibration_serial_kernel(name, expression, lanes) \
kernel void name(device const float *input [[buffer(0)]], device float *output [[buffer(1)]], \
                 uint index [[thread_position_in_grid]]) { \
    float value = input[index]; \
    if (index < lanes) { \
        for (uint iteration = 0; iteration < calibration_iterations; ++iteration) { \
            value = expression; \
        } \
    } \
    output[index] = value; \
}
calibration_serial_kernel(calibration_scalar_serial, fma(value, 0.9999999f, 0.0000001f), 1)
calibration_serial_kernel(calibration_sfu_serial, fast::exp(-value), 1)

kernel void calibration_dependent_read(device const uint *input [[buffer(0)]],
                                       device uint *output [[buffer(1)]],
                                       uint index [[thread_position_in_grid]]) {
    if (index != 0) { output[index] = input[index]; return; }
    uint cursor = 0;
    uint checksum = 0;
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        cursor = input[cursor];
        checksum += cursor;
    }
    output[0] = checksum;
}

kernel void calibration_read_serial(device const float *input [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index != 0) { output[index] = 0; return; }
    float sum = 0;
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        sum = fma(sum, 0.99999f, input[iteration]);
    }
    output[0] = sum;
}

calibration_serial_kernel(calibration_scalar_group, fma(value, 0.9999999f, 0.0000001f), 32)
calibration_serial_kernel(calibration_sfu_group, fast::exp(-value), 32)

kernel void calibration_read_group(device const float *input [[buffer(0)]],
                                   device float *output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
    if (index >= 32) { output[index] = 0; return; }
    float sum = 0;
    for (uint iteration = index; iteration < calibration_iterations; iteration += 32) {
        sum = fma(sum, 0.99999f, input[iteration]);
    }
    output[index] = sum;
}

kernel void calibration_gather_group(device const uint *input [[buffer(0)]],
                                     device uint *output [[buffer(1)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index >= 32) { output[index] = 0; return; }
    uint cursor = index;
    uint sum = 0;
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        cursor = input[cursor];
        sum += cursor;
    }
    output[index] = sum;
}

kernel void calibration_group_dispatch(device const float *input [[buffer(0)]],
                                       device float *output [[buffer(1)]],
                                       uint index [[thread_position_in_grid]],
                                       uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float scratch[128];
    float value = input[index];
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        scratch[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value = scratch[(lane + 1) & 127] + 0.00001f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[index] = value;
}

kernel void calibration_read_wide_group(device const float *input [[buffer(0)]],
                                        device float *output [[buffer(1)]],
                                        uint index [[thread_position_in_grid]]) {
    float4 sum = 0;
    for (uint i = index * 4; i < calibration_iterations; i += 128 * 4) {
        sum += float4(input[i], input[i + 1], input[i + 2], input[i + 3]);
    }
    output[index] = sum.x + sum.y + sum.z + sum.w;
}

calibration_serial_kernel(calibration_scalar_wide_group, fma(value, 0.9999999f, 0.0000001f), 128)
calibration_serial_kernel(calibration_sfu_wide_group, fast::exp(-value), 128)

kernel void calibration_scalar_parallel(device const float *input [[buffer(0)]],
                                        device float *output [[buffer(1)]],
                                        uint index [[thread_position_in_grid]]) {
    float values[8];
    float seed = input[index];
    #pragma unroll
    for (uint chain = 0; chain < 8; ++chain) {
        values[chain] = seed * (0.5f + float(chain) * 0.0625f);
    }
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        #pragma unroll
        for (uint chain = 0; chain < 8; ++chain) {
            values[chain] = fma(values[chain], 0.9999999f, seed * 0.0000001f);
        }
    }
    float sum = 0;
    #pragma unroll
    for (uint chain = 0; chain < 8; ++chain) { sum += values[chain]; }
    output[index] = sum / 8.0f;
}
