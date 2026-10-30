struct DeltaBlockParams {
    StateAuth access;
    uint tokens;
    uint query_heads;
    uint heads;
    uint keys;
    uint values;
    uint blocks;
    uint stride;
    uint offsets[8];
};

uint delta_block_base(constant DeltaBlockParams &params, uint block, uint head) {
    return (block * params.heads + head) * params.stride;
}

uint delta_block_rows(constant DeltaBlockParams &params, uint block) {
    return min(uint(FORJA_DELTA_BLOCK_TOKENS), params.tokens - block * FORJA_DELTA_BLOCK_TOKENS);
}

float delta_block_tree(device const uchar *left, device const uchar *right,
    constant TensorLayout *layouts, uint left_layout, uint right_layout,
    uint left_row, uint right_row, uint left_head,
    uint right_head, constant DeltaBlockParams &params) {
    float lanes[32];
    uint count = (params.keys + 31) / 32;
    for (uint lane = 0; lane < 32; ++lane) {
        float sum = 0.0f;
        for (uint item = 0; item < count; ++item) {
            uint key_index = lane * count + item;
            if (key_index < params.keys) {
                uint left_index = (left_row * params.query_heads + left_head) * params.keys + key_index;
                uint right_index = (right_row * params.query_heads + right_head) * params.keys + key_index;
                sum += load_float(left,physical_index(layouts[left_layout],left_index),input0_dtype)
                    * load_float(right,physical_index(layouts[right_layout],right_index),input1_dtype);
            }
        }
        lanes[lane] = sum;
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
        for (uint lane = 0; lane < offset; ++lane) { lanes[lane] += lanes[lane+offset]; }
    }
    return lanes[0];
}

kernel void state_block_gram(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *log_decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], device float *scratch [[buffer(11)]],
    constant TensorLayout *layouts [[buffer(12)]],
    constant DeltaBlockParams &params [[buffer(13)]], device atomic_uint *error_flag [[buffer(14)]],
    uint index [[thread_position_in_grid]]) {
    uint matrix = FORJA_DELTA_BLOCK_TOKENS * FORJA_DELTA_BLOCK_TOKENS;
    uint head = index / (params.blocks * matrix);
    if (head >= params.heads || activity[0] == 0) { return; }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint block = (index / matrix) % params.blocks;
    uint row = (index / FORJA_DELTA_BLOCK_TOKENS) % FORJA_DELTA_BLOCK_TOKENS;
    uint column = index % FORJA_DELTA_BLOCK_TOKENS;
    uint rows = delta_block_rows(params,block);
    uint base = delta_block_base(params,block,head);
    uint lower = base + params.offsets[0] + row * FORJA_DELTA_BLOCK_TOKENS + column;
    uint query_gram = base + params.offsets[1] + row * FORJA_DELTA_BLOCK_TOKENS + column;
    scratch[lower] = 0.0f;
    scratch[query_gram] = 0.0f;
    if (row >= rows || column > row) { return; }
    uint query_head = head / (params.heads / params.query_heads);
    uint absolute_row = block * FORJA_DELTA_BLOCK_TOKENS + row;
    uint absolute_column = block * FORJA_DELTA_BLOCK_TOKENS + column;
    float interval = 0.0f;
    for (uint at = row; at > column; --at) {
        uint decay_index = (block * FORJA_DELTA_BLOCK_TOKENS + at) * params.heads + head;
        interval += load_float(log_decay,physical_index(layouts[3],decay_index),program_input3_dtype);
    }
    float factor = f32_exp_rounded(interval);
    if (row == column) { scratch[lower] = 1.0f; }
    else {
        float dot = delta_block_tree(key,key,layouts,1,1,absolute_row,absolute_column,
            query_head,query_head,params);
        float coefficient = load_float(beta,physical_index(layouts[4],absolute_row*params.heads+head),program_input4_dtype);
        scratch[lower] = coefficient * dot * factor;
    }
    scratch[query_gram] = delta_block_tree(query,key,layouts,0,1,absolute_row,absolute_column,
        query_head,query_head,params) * factor;
    if (row == rows-1) { scratch[base+params.offsets[3]+column] = factor; }
    if (column == 0) {
        float prefix = 0.0f;
        for (uint at=0; at<=row; ++at) {
            uint decay_index = (block*FORJA_DELTA_BLOCK_TOKENS+at)*params.heads+head;
            prefix += load_float(log_decay,physical_index(layouts[3],decay_index),program_input3_dtype);
        }
        scratch[base+params.offsets[2]+row] = f32_exp_rounded(prefix);
    }
}
