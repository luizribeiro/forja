constant uint calibration_iterations [[function_constant(220)]];

template <typename Stored, typename Arithmetic>
void calibration_mma_impl(device const float *input, device float *output,
                          uint index, ushort lane) {
    simdgroup_matrix<Arithmetic, 8, 8> left, right;
    simdgroup_matrix<float, 8, 8> accumulators[4];
    float seed = input[index];
    for (uint element = 0; element < 2; ++element) {
        left.thread_elements()[element] = Arithmetic(Stored(seed * 0.001f));
        right.thread_elements()[element] = Arithmetic(Stored(seed * 0.001f));
        for (uint chain = 0; chain < 4; ++chain) {
            accumulators[chain].thread_elements()[element] = seed * float(chain + 1);
        }
    }
    for (uint iteration = 0; iteration < calibration_iterations; ++iteration) {
        for (uint chain = 0; chain < 4; ++chain) {
            simdgroup_matrix<float, 8, 8> result;
            simdgroup_multiply_accumulate(result, left, right, accumulators[chain]);
            accumulators[chain] = result;
        }
    }
    // Every fragment and chain remains observable, preventing dead-code removal.
    float result = 0;
    for (uint chain = 0; chain < 4; ++chain) {
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
