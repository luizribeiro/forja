kernel void state_block_solve_panel(
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
    uint head = group.y;
    uint block = group.z;
    if (block >= params.blocks || head >= params.heads || activity[0] == 0) { return; }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint base = delta_block_base(params,block,head);
    uint rows = delta_block_rows(params,block);
    uint query_head = head/(params.heads/params.query_heads);
    threadgroup float coefficients[4096];
    threadgroup float rhs[2048];
    for (uint cell=thread_index; cell<4096; cell+=128) {
        coefficients[cell] = scratch[base+params.offsets[0]+cell];
    }
    for (uint cell=thread_index; cell<2048; cell+=128) {
        uint row = cell/32;
        uint column = group.x*32+cell%32;
        float result = 0.0f;
        if (row < rows && column < params.values+params.keys) {
            uint absolute_row = block*FORJA_DELTA_BLOCK_TOKENS+row;
            float coefficient = load_float(beta,physical_index(layouts[4],absolute_row*params.heads+head),program_input4_dtype);
            if (column < params.values) {
                uint index = (absolute_row*params.heads+head)*params.values+column;
                result = coefficient*load_float(value,physical_index(layouts[2],index),input2_dtype);
            } else {
                uint index = (absolute_row*params.query_heads+query_head)*params.keys+column-params.values;
                result = coefficient*scratch[base+params.offsets[2]+row]
                    *load_float(key,physical_index(layouts[1],index),input1_dtype);
            }
        }
        rhs[cell] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row=0; row<rows; ++row) {
        float prediction[8];
        for (uint column=0; column<8; ++column) { prediction[column] = 0.0f; }
        for (uint item=0; item<2; ++item) {
            uint previous_row = lane*2+item;
            if (previous_row < row) {
                float coefficient = coefficients[row*FORJA_DELTA_BLOCK_TOKENS+previous_row];
                for (uint column=0; column<8; ++column) {
                    prediction[column] += coefficient*rhs[previous_row*32+simd_group*8+column];
                }
            }
        }
        for (uint offset=16; offset>0; offset>>=1) {
            for (uint column=0; column<8; ++column) {
                prediction[column] += simd_shuffle_down(prediction[column],ushort(offset));
            }
        }
        if (lane == 0) {
            for (uint column=0; column<8; ++column) {
                rhs[row*32+simd_group*8+column] -= prediction[column];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint cell=thread_index; cell<2048; cell+=128) {
        uint row = cell/32;
        uint column = group.x*32+cell%32;
        if (column < params.values+params.keys) {
            bool key_panel = column >= params.values;
            uint width = key_panel ? params.keys : params.values;
            uint panel_column = column-(key_panel ? params.values : 0);
            scratch[base+params.offsets[key_panel ? 5 : 4]+row*width+panel_column] = rhs[cell];
        }
    }
}
