float2 delta_fragment_load(threadgroup const float *source, uint row, uint column,
    uint stride, uint lane) {
    short2 coordinate = SteelMMAFragment<float>::coordinate(ushort(lane));
    uint index = (row+uint(coordinate.y))*stride+column+uint(coordinate.x);
    return float2(source[index],source[index+1]);
}

void delta_fragment_store(threadgroup float *destination, uint row, uint column,
    uint stride, uint lane, float2 values) {
    short2 coordinate = SteelMMAFragment<float>::coordinate(ushort(lane));
    uint index = (row+uint(coordinate.y))*stride+column+uint(coordinate.x);
    destination[index] = values.x;
    destination[index+1] = values.y;
}

void delta_fragment_multiply(thread float2 &accumulator, thread float2 &left,
    thread float2 &right) {
    float2 result;
    SteelMMAFragment<float>::multiply(result,left,right,accumulator);
    accumulator = result;
}

kernel void state_block_gram_mma(
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
    uint block = group.z / params.heads;
    uint head = group.z % params.heads;
    if (block >= params.blocks || activity[0] == 0) { return; }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint rows = delta_block_rows(params,block);
    if (group.x > group.y || group.y*32 >= rows) {
        for (uint cell=thread_index; cell<1024; cell+=128) {
            delta_block_gram_terms(scratch,log_decay,beta,layouts,params,block,head,
                group.y*32+cell/32,group.x*32+cell%32,0.0f,0.0f);
        }
        return;
    }
    uint query_head = head / (params.heads/params.query_heads);
    threadgroup float query_panel[1024];
    threadgroup float key_panel[1024];
    threadgroup float right_panel[1024];
    float2 query_accumulators[4];
    float2 key_accumulators[4];
    for (uint part=0; part<4; ++part) {
        query_accumulators[part] = float2(0.0f);
        key_accumulators[part] = float2(0.0f);
    }
    uint fragment_row = (simd_group/2)*16;
    uint fragment_column = (simd_group%2)*16;
    for (uint inner=0; inner<params.keys; inner+=32) {
        for (uint cell=thread_index; cell<1024; cell+=128) {
            uint row = group.y*32+cell/32;
            uint key_index = inner+cell%32;
            float query_value = 0.0f;
            float key_value = 0.0f;
            if (row < rows && key_index < params.keys) {
                uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.query_heads+query_head)*params.keys+key_index;
                query_value = load_float(query,physical_index(layouts[0],index),input0_dtype);
                key_value = load_float(key,physical_index(layouts[1],index),input1_dtype);
            }
            query_panel[cell] = query_value;
            key_panel[cell] = key_value;
            uint column = group.x*32+cell%32;
            key_index = inner+cell/32;
            float right_value = 0.0f;
            if (column < rows && key_index < params.keys) {
                uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+column)*params.query_heads+query_head)*params.keys+key_index;
                right_value = load_float(key,physical_index(layouts[1],index),input1_dtype);
            }
            right_panel[cell] = right_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint fragment_inner=0; fragment_inner<32; fragment_inner+=8) {
            for (uint row=0; row<2; ++row) {
                float2 left_query = delta_fragment_load(query_panel,fragment_row+row*8,fragment_inner,32,lane);
                float2 left_key = delta_fragment_load(key_panel,fragment_row+row*8,fragment_inner,32,lane);
                for (uint column=0; column<2; ++column) {
                    float2 right = delta_fragment_load(right_panel,fragment_inner,fragment_column+column*8,32,lane);
                    delta_fragment_multiply(query_accumulators[row*2+column],left_query,right);
                    delta_fragment_multiply(key_accumulators[row*2+column],left_key,right);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row=0; row<2; ++row) {
        for (uint column=0; column<2; ++column) {
            delta_fragment_store(query_panel,fragment_row+row*8,fragment_column+column*8,32,lane,query_accumulators[row*2+column]);
            delta_fragment_store(key_panel,fragment_row+row*8,fragment_column+column*8,32,lane,key_accumulators[row*2+column]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint cell=thread_index; cell<1024; cell+=128) {
        delta_block_gram_terms(scratch,log_decay,beta,layouts,params,block,head,
            group.y*32+cell/32,group.x*32+cell%32,query_panel[cell],key_panel[cell]);
    }
}
