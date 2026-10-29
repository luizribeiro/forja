struct StateAuth {
    uint source;
    uint destination;
    uint records;
};

struct StateParams {
    StateAuth access;
    uint q_heads;
    uint heads;
    uint keys;
    uint values;
};

bool authenticate_state(
    device const uint2 *table, device const uint *generations,
    constant StateAuth &params, device atomic_uint *error_flag) {
    uint2 source = table[0];
    uint2 destination = table[1];
    bool valid = source.x == params.source && destination.x == params.destination
        && source.x != destination.x && source.x < params.records && destination.x < params.records;
    if (valid) {
        valid = generations[source.x] == source.y && generations[destination.x] == destination.y;
    }
    if (!valid) { atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed); }
    return valid;
}

kernel void state_delta_rule(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], constant TensorLayout *layouts [[buffer(11)]],
    constant StateParams &params [[buffer(12)]], device atomic_uint *error_flag [[buffer(13)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.y;
    uint column = group.x * 4 + simd_group;
    if (column >= params.values) { return; }
    uint out_index = head * params.values + column;
    if (activity[0] == 0) {
        if (lane == 0) { store_float(output, physical_index(layouts[6],out_index),output_dtype,0.0f); }
        return;
    }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    uint q_head = head / (params.heads / params.q_heads);
    uint per_thread = (params.keys + 31) / 32;
    float state[(FORJA_DELTA_RULE_MAX_KEYS + 31) / 32];
    float prediction = 0.0f;
    float d = load_float(decay,physical_index(layouts[3],head),program_input3_dtype);
    for (uint i=0; i<per_thread; ++i) {
        uint k = lane * per_thread + i;
        state[i] = 0.0f;
        if (k < params.keys) {
            uint index = (head * params.keys + k) * params.values + column;
            state[i] = previous[physical_index(layouts[5],index)] * d;
            prediction += state[i] * load_float(key,physical_index(layouts[1],q_head * params.keys+k),input1_dtype);
        }
    }
    prediction = simd_sum(prediction);
    float delta = (load_float(value,physical_index(layouts[2],out_index),input2_dtype)-prediction)
        * load_float(beta,physical_index(layouts[4],head),program_input4_dtype);
    float result = 0.0f;
    for (uint i=0; i<per_thread; ++i) {
        uint k = lane * per_thread + i;
        if (k < params.keys) {
            uint index = (head * params.keys + k) * params.values + column;
            state[i] += load_float(key,physical_index(layouts[1],q_head*params.keys+k),input1_dtype) * delta;
            next[physical_index(layouts[7],index)] = state[i];
            result += state[i] * load_float(query,physical_index(layouts[0],q_head*params.keys+k),input0_dtype);
        }
    }
    result = simd_sum(result);
    if (lane == 0) { store_float(output,physical_index(layouts[6],out_index),output_dtype,result); }
}

struct ConvStateParams {
    StateAuth access;
    uint tokens;
    uint channels;
    uint taps;
    uint reserved;
};

float causal_state_value(device const uchar *input, device const uchar *history,
    constant TensorLayout *layouts, ulong row, uint channel, constant ConvStateParams &params) {
    ulong history_rows = params.taps - 1;
    if (row < history_rows) {
        return load_float(history,physical_index(layouts[2],uint(row*params.channels+channel)),input0_dtype);
    }
    return load_float(input,physical_index(layouts[0],uint((row-history_rows)*params.channels+channel)),input0_dtype);
}

kernel void state_causal_conv(
    device const uchar *input [[buffer(0)]], device const uchar *weight [[buffer(1)]],
    device const uchar *history [[buffer(2)]], device const uint2 *table [[buffer(3)]],
    device const uint *generations [[buffer(4)]], device const uint *activity [[buffer(5)]],
    device uchar *output [[buffer(6)]], device uchar *next [[buffer(7)]],
    constant TensorLayout *layouts [[buffer(8)]], constant ConvStateParams &params [[buffer(9)]],
    device atomic_uint *error_flag [[buffer(10)]], uint index [[thread_position_in_grid]]) {
    uint outputs = layouts[3].element_count;
    uint states = layouts[4].element_count;
    if (index >= max(outputs,states)) { return; }
    if (activity[0] == 0) {
        if (index < outputs) { store_float(output,physical_index(layouts[3],index),output_dtype,0.0f); }
        return;
    }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    uint channel = index % params.channels;
    if (index < outputs) {
        ulong row = index / params.channels;
        float sum = 0.0f;
        for (uint tap=0;tap<params.taps;++tap) {
            sum += causal_state_value(input,history,layouts,row+tap,channel,params)
                * load_float(weight,physical_index(layouts[1],tap*params.channels+channel),input1_dtype);
        }
        store_float(output,physical_index(layouts[3],index),output_dtype,sum);
    }
    if (index < states) {
        ulong row = ulong(params.tokens) + index / params.channels;
        store_float(next,physical_index(layouts[4],index),input0_dtype,
            causal_state_value(input,history,layouts,row,channel,params));
    }
}
