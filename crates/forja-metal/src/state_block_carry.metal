float delta_block_state_tree(device const uchar *query, device const float *scratch,
    constant TensorLayout *layouts, constant DeltaBlockParams &params, uint absolute_row,
    uint query_head, uint base, uint row, uint column, bool solved,
    thread const float *state) {
    float lanes[32];
    uint count = (params.keys+31)/32;
    for (uint lane=0; lane<32; ++lane) {
        float sum = 0.0f;
        for (uint item=0; item<count; ++item) {
            uint key_index = lane*count+item;
            if (key_index < params.keys) {
                float left = solved
                    ? scratch[base+params.offsets[5]+row*params.keys+key_index]
                    : load_float(query,physical_index(layouts[0],
                        (absolute_row*params.query_heads+query_head)*params.keys+key_index),input0_dtype);
                sum += left * state[key_index];
            }
        }
        lanes[lane] = sum;
    }
    for (uint offset=16; offset>0; offset>>=1) {
        for (uint lane=0; lane<offset; ++lane) { lanes[lane] += lanes[lane+offset]; }
    }
    return lanes[0];
}

float delta_block_carry_tree(device const uchar *key, device const float *scratch,
    constant TensorLayout *layouts, constant DeltaBlockParams &params,
    uint base, uint block, uint head, uint query_head, uint key_index, uint column, uint rows) {
    float lanes[32];
    for (uint lane=0; lane<32; ++lane) {
        float sum = 0.0f;
        for (uint item=0; item<2; ++item) {
            uint row = lane*2+item;
            if (row < rows) {
                uint source = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.query_heads
                    +query_head)*params.keys+key_index;
                float factor = scratch[base+params.offsets[3]+row]
                    * load_float(key,physical_index(layouts[1],source),input1_dtype);
                sum += factor * scratch[base+params.offsets[6]+row*params.values+column];
            }
        }
        lanes[lane] = sum;
    }
    for (uint offset=16; offset>0; offset>>=1) {
        for (uint lane=0; lane<offset; ++lane) { lanes[lane] += lanes[lane+offset]; }
    }
    return lanes[0];
}

kernel void state_block_carry(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *log_decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], device float *scratch [[buffer(11)]],
    constant TensorLayout *layouts [[buffer(12)]],
    constant DeltaBlockParams &params [[buffer(13)]], device atomic_uint *error_flag [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    uint head = index / params.values;
    if (head >= params.heads || activity[0] == 0) { return; }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint column = index % params.values;
    uint query_head = head / (params.heads / params.query_heads);
    float state[FORJA_DELTA_RULE_MAX_KEYS];
    for (uint key_index=0; key_index<params.keys; ++key_index) {
        uint source = (head*params.keys+key_index)*params.values+column;
        state[key_index] = previous[physical_index(layouts[5],source)];
    }
    for (uint block=0; block<params.blocks; ++block) {
        uint base = delta_block_base(params,block,head);
        uint rows = delta_block_rows(params,block);
        for (uint row=0; row<rows; ++row) {
            uint absolute_row = block*FORJA_DELTA_BLOCK_TOKENS+row;
            uint slot = base+params.offsets[6]+row*params.values+column;
            float correction = delta_block_state_tree(query,scratch,layouts,params,absolute_row,
                query_head,base,row,column,true,state);
            scratch[slot] = scratch[base+params.offsets[4]+row*params.values+column]-correction;
            scratch[base+params.offsets[7]+row*params.values+column] =
                scratch[base+params.offsets[2]+row]
                * delta_block_state_tree(query,scratch,layouts,params,absolute_row,
                    query_head,base,row,column,false,state);
        }
        float decay = scratch[base+params.offsets[2]+rows-1];
        for (uint key_index=0; key_index<params.keys; ++key_index) {
            state[key_index] = decay*state[key_index]
                +delta_block_carry_tree(key,scratch,layouts,params,base,block,head,
                    query_head,key_index,column,rows);
        }
    }
    for (uint key_index=0; key_index<params.keys; ++key_index) {
        uint destination = (head*params.keys+key_index)*params.values+column;
        next[physical_index(layouts[7],destination)] = state[key_index];
    }
}
