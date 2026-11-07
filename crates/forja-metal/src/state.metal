constant uint state_linear_layout [[function_constant(227)]];
constant uint state_linear_views [[function_constant(228)]];

struct StateAuth {
    uint source;
    uint destination;
    uint records;
    uint active;
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
    bool valid = params.active != 0 && source.x == params.source && destination.x == params.destination
        && source.x != destination.x && source.x < params.records && destination.x < params.records;
    if (valid) {
        valid = generations[source.x] == source.y && generations[destination.x] == destination.y;
    }
    if (!valid) { atomic_fetch_or_explicit(error_flag, 1u, memory_order_relaxed); }
    return valid;
}

ulong delta_state_index(constant TensorLayout *layouts, uint slot, uint index) {
    return state_linear_layout != 0 && (state_linear_views & (1u << slot)) != 0
        ? layouts[slot].offset + index : physical_index(layouts[slot], index);
}

void state_delta_rule_compute(
    device const uchar *query, device const uchar *key,
    device const uchar *value, device const uchar *decay,
    device const uchar *beta, device const float *previous,
    device const uint2 *table, device const uint *generations,
    device const uint *activity, device uchar *output,
    device float *next, constant TensorLayout *layouts,
    constant StateParams &params, device atomic_uint *error_flag,
    uint3 group, uint simd_group,
    uint lane, thread float *state, bool carried) {
    uint head = group.y;
    uint column = group.x * 4 + simd_group;
    uint out_index = head * params.values + column;
    uint q_head = head / (params.heads / params.q_heads);
    uint per_thread = (params.keys + 31) / 32;
    float prediction = 0.0f;
    float d = load_float(decay,delta_state_index(layouts,3,head),program_input3_dtype);
    for (uint i=0; i<per_thread; ++i) {
        uint k = lane * per_thread + i;
        if (k < params.keys) {
            uint index = (head * params.keys + k) * params.values + column;
            ulong source_index = delta_state_index(layouts,5,index);
            state[i] = (carried ? state[i] : previous[source_index]) * d;
            prediction += state[i] * load_float(key,delta_state_index(layouts,1,q_head * params.keys+k),input1_dtype);
        }
    }
    prediction = simd_sum(prediction);
    float delta = (load_float(value,delta_state_index(layouts,2,out_index),input2_dtype)-prediction)
        * load_float(beta,delta_state_index(layouts,4,head),program_input4_dtype);
    float result = 0.0f;
    for (uint i=0; i<per_thread; ++i) {
        uint k = lane * per_thread + i;
        if (k < params.keys) {
            uint index = (head * params.keys + k) * params.values + column;
            state[i] += load_float(key,delta_state_index(layouts,1,q_head*params.keys+k),input1_dtype) * delta;
            ulong next_index = delta_state_index(layouts,7,index);
            next[next_index] = state[i];
            result += state[i] * load_float(query,delta_state_index(layouts,0,q_head*params.keys+k),input0_dtype);
        }
    }
    result = simd_sum(result);
    if (lane == 0) { store_float(output,delta_state_index(layouts,6,out_index),output_dtype,result); }
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
    float state[(FORJA_DELTA_RULE_MAX_KEYS + 31) / 32];
    state_delta_rule_compute(query, key, value, decay, beta, previous, table, generations,
        activity, output, next, layouts, params, error_flag, group, simd_group, lane, state, false);
}

bool state_delta_rule_indirect(
    constant ulong *pointers, constant TensorLayout *all_layouts,
    constant StateParams *all_params, device atomic_uint *error_flag,
    uint3 group, uint simd_group, uint lane, thread float *state, bool carried) {
    constant ulong *p = pointers + group.z * 11;
    constant TensorLayout *layouts = all_layouts + group.z * 8;
    constant StateParams &params = all_params[group.z];
    uint column = group.x * 4 + simd_group;
    if (column >= params.values) return false;
    device uchar *output = reinterpret_cast<device uchar *>(p[9]);
    device const uint *activity = reinterpret_cast<device const uint *>(p[8]);
    if (activity[0] == 0) {
        if (lane == 0) store_float(output, physical_index(layouts[6], group.y * params.values + column), output_dtype, 0.0f);
        return false;
    }
    device const uint2 *table = reinterpret_cast<device const uint2 *>(p[6]);
    device const uint *generations = reinterpret_cast<device const uint *>(p[7]);
    if (!authenticate_state(table, generations, params.access, error_flag)) return false;
    state_delta_rule_compute(reinterpret_cast<device const uchar *>(p[0]),
        reinterpret_cast<device const uchar *>(p[1]), reinterpret_cast<device const uchar *>(p[2]),
        reinterpret_cast<device const uchar *>(p[3]), reinterpret_cast<device const uchar *>(p[4]),
        reinterpret_cast<device const float *>(p[5]), table, generations, activity, output,
        reinterpret_cast<device float *>(p[10]), layouts, params, error_flag, group, simd_group, lane, state, carried);
    return true;
}

kernel void state_delta_rule_lanes(
    constant ulong *pointers [[buffer(0)]], constant TensorLayout *all_layouts [[buffer(1)]],
    constant StateParams *all_params [[buffer(2)]], device atomic_uint *error_flag [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    float state[(FORJA_DELTA_RULE_MAX_KEYS + 31) / 32];
    state_delta_rule_indirect(pointers, all_layouts, all_params, error_flag, group, simd_group, lane, state, false);
}

kernel void state_delta_rule_steps(
    constant ulong *pointers [[buffer(0)]], constant TensorLayout *all_layouts [[buffer(1)]],
    constant StateParams *all_params [[buffer(2)]], device atomic_uint *error_flag [[buffer(3)]],
    constant uint &steps [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float state[(FORJA_DELTA_RULE_MAX_KEYS + 31) / 32];
    bool carried = false;
    for (uint step = 0; step < steps; ++step) {
        group.z = step;
        carried = state_delta_rule_indirect(pointers, all_layouts, all_params, error_flag, group, simd_group, lane, state, carried);
    }
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

void state_causal_conv_compute(
    device const uchar *input, device const uchar *weight,
    device const uchar *history, device const uint2 *table,
    device const uint *generations, device const uint *activity,
    device uchar *output, device uchar *next,
    constant TensorLayout *layouts, constant ConvStateParams &params,
    device atomic_uint *error_flag, uint index) {
    uint outputs = layouts[3].element_count;
    uint states = layouts[4].element_count;
    uint channel = index % params.channels;
    if (index < outputs) {
        ulong row = index / params.channels;
        float sum = 0.0f;
        for (uint tap=0;tap<params.taps;++tap) {
#ifdef FORJA_SAFE_STATE_CONV
            sum = fma(causal_state_value(input,history,layouts,row+tap,channel,params),
                load_float(weight,physical_index(layouts[1],tap*params.channels+channel),input1_dtype),sum);
#else
            sum += causal_state_value(input,history,layouts,row+tap,channel,params)
                * load_float(weight,physical_index(layouts[1],tap*params.channels+channel),input1_dtype);
#endif
        }
        store_float(output,physical_index(layouts[3],index),output_dtype,sum);
    }
    if (index < states) {
        ulong row = ulong(params.tokens) + index / params.channels;
        store_float(next,physical_index(layouts[4],index),input0_dtype,
            causal_state_value(input,history,layouts,row,channel,params));
    }
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
    state_causal_conv_compute(input, weight, history, table, generations, activity, output,
        next, layouts, params, error_flag, index);
}

kernel void state_causal_conv_lanes(
    constant ulong *pointers [[buffer(0)]], constant TensorLayout *all_layouts [[buffer(1)]],
    constant ConvStateParams *all_params [[buffer(2)]], device atomic_uint *error_flag [[buffer(3)]],
    uint3 position [[thread_position_in_grid]]) {
    constant ulong *p = pointers + position.z * 11;
    constant TensorLayout *layouts = all_layouts + position.z * 8;
    constant ConvStateParams &params = all_params[position.z];
    uint index = position.x;
    uint outputs = layouts[3].element_count;
    if (index >= max(outputs, layouts[4].element_count)) return;
    device uchar *output = reinterpret_cast<device uchar *>(p[6]);
    device const uint *activity = reinterpret_cast<device const uint *>(p[5]);
    if (activity[0] == 0) {
        if (index < outputs) store_float(output, physical_index(layouts[3], index), output_dtype, 0.0f);
        return;
    }
    device const uint2 *table = reinterpret_cast<device const uint2 *>(p[3]);
    device const uint *generations = reinterpret_cast<device const uint *>(p[4]);
    if (!authenticate_state(table, generations, params.access, error_flag)) return;
    state_causal_conv_compute(reinterpret_cast<device const uchar *>(p[0]),
        reinterpret_cast<device const uchar *>(p[1]), reinterpret_cast<device const uchar *>(p[2]),
        table, generations, activity, output, reinterpret_cast<device uchar *>(p[7]),
        layouts, params, error_flag, index);
}
