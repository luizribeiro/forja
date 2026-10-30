// A diagonal fragment must not multiply a masked future coefficient by a NaN value.
float delta_diagonal_output(device const float *scratch, constant DeltaBlockParams &params,
    uint base, uint row, uint column) {
    float terms[8];
    uint start = (row/8)*8;
    for (uint item=0; item<8; ++item) {
        uint previous = start+item;
        terms[item] = previous <= row
            ? scratch[base+params.offsets[1]+row*FORJA_DELTA_BLOCK_TOKENS+previous]
                *scratch[base+params.offsets[6]+previous*params.values+column]
            : 0.0f;
    }
    for (uint offset=4; offset>0; offset>>=1) {
        for (uint item=0; item<offset; ++item) { terms[item] += terms[item+offset]; }
    }
    return terms[0];
}

kernel void state_block_output_mma(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *log_decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], device float *scratch [[buffer(11)]],
    constant TensorLayout *layouts [[buffer(12)]],
    constant DeltaBlockParams &params [[buffer(13)]], device atomic_uint *error_flag [[buffer(14)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint thread_index [[thread_index_in_threadgroup]]) {
    uint block = group.z/params.heads;
    uint head = group.z%params.heads;
    if (block >= params.blocks) { return; }
    uint rows = delta_block_rows(params,block);
    // Inactive lanes publish zeros without accessing authenticated private state.
    if (activity[0] == 0) {
        for (uint cell=thread_index; cell<1024; cell+=128) {
            uint row = group.y*32+cell/32;
            uint column = group.x*32+cell%32;
            if (row < rows && column < params.values) {
                uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.heads+head)*params.values+column;
                store_float(output,physical_index(layouts[6],index),output_dtype,0.0f);
            }
        }
        return;
    }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint base = delta_block_base(params,block,head);
    threadgroup float gram_panel[1024];
    threadgroup float delta_panel[1024];
    float2 accumulators[4];
    for (uint part=0; part<4; ++part) { accumulators[part] = float2(0.0f); }
    uint fragment_row = (simd_group/2)*16;
    uint fragment_column = (simd_group%2)*16;
    for (uint inner=0; inner<FORJA_DELTA_BLOCK_TOKENS; inner+=32) {
        for (uint cell=thread_index; cell<1024; cell+=128) {
            uint row = group.y*32+cell/32;
            uint previous_row = inner+cell%32;
            gram_panel[cell] = row < rows && previous_row < rows && previous_row <= row
                ? scratch[base+params.offsets[1]+row*FORJA_DELTA_BLOCK_TOKENS+previous_row] : 0.0f;
            previous_row = inner+cell/32;
            uint column = group.x*32+cell%32;
            delta_panel[cell] = previous_row < rows && column < params.values
                ? scratch[base+params.offsets[6]+previous_row*params.values+column] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint fragment_inner=0; fragment_inner<32; fragment_inner+=8) {
            for (uint row=0; row<2; ++row) {
                uint absolute_start = group.y*32+fragment_row+row*8;
                if (inner+fragment_inner+8 <= absolute_start) {
                    float2 left = delta_fragment_load(gram_panel,fragment_row+row*8,fragment_inner,32,lane);
                    for (uint column=0; column<2; ++column) {
                        float2 right = delta_fragment_load(delta_panel,fragment_inner,fragment_column+column*8,32,lane);
                        delta_fragment_multiply(accumulators[row*2+column],left,right);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row=0; row<2; ++row) {
        for (uint column=0; column<2; ++column) {
            delta_fragment_store(gram_panel,fragment_row+row*8,fragment_column+column*8,32,lane,accumulators[row*2+column]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell=thread_index; cell<1024; cell+=128) {
        uint row = group.y*32+cell/32;
        uint column = group.x*32+cell%32;
        if (row < rows && column < params.values) {
            float correction = gram_panel[cell]+delta_diagonal_output(scratch,params,base,row,column);
            float result = scratch[base+params.offsets[7]+row*params.values+column]+correction;
            uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.heads+head)*params.values+column;
            store_float(output,physical_index(layouts[6],index),output_dtype,result);
        }
    }
}
