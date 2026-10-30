kernel void state_block_output(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *log_decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], device float *scratch [[buffer(11)]],
    constant TensorLayout *layouts [[buffer(12)]],
    constant DeltaBlockParams &params [[buffer(13)]], device atomic_uint *error_flag [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    uint count = params.tokens*params.heads*params.values;
    if (index >= count) { return; }
    if (activity[0] == 0) {
        store_float(output,physical_index(layouts[6],index),output_dtype,0.0f);
        return;
    }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint column = index % params.values;
    uint head = (index / params.values) % params.heads;
    uint absolute_row = index / (params.values*params.heads);
    uint block = absolute_row / FORJA_DELTA_BLOCK_TOKENS;
    uint row = absolute_row % FORJA_DELTA_BLOCK_TOKENS;
    uint base = delta_block_base(params,block,head);
    float lanes[32];
    for (uint lane=0; lane<32; ++lane) {
        float sum = 0.0f;
        for (uint item=0; item<2; ++item) {
            uint previous_row = lane*2+item;
            if (previous_row <= row) {
                sum += scratch[base+params.offsets[1]+row*FORJA_DELTA_BLOCK_TOKENS+previous_row]
                    * scratch[base+params.offsets[6]+previous_row*params.values+column];
            }
        }
        lanes[lane] = sum;
    }
    for (uint offset=16; offset>0; offset>>=1) {
        for (uint lane=0; lane<offset; ++lane) { lanes[lane] += lanes[lane+offset]; }
    }
    float result = scratch[base+params.offsets[7]+row*params.values+column]+lanes[0];
    store_float(output,physical_index(layouts[6],index),output_dtype,result);
}
