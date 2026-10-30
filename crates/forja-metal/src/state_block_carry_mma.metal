kernel void state_block_carry_mma(
    device const uchar *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const uchar *log_decay [[buffer(3)]],
    device const uchar *beta [[buffer(4)]], device const float *previous [[buffer(5)]],
    device const uint2 *table [[buffer(6)]], device const uint *generations [[buffer(7)]],
    device const uint *activity [[buffer(8)]], device uchar *output [[buffer(9)]],
    device float *next [[buffer(10)]], device float *scratch [[buffer(11)]],
    constant TensorLayout *layouts [[buffer(12)]],
    constant DeltaBlockParams &params [[buffer(13)]], device atomic_uint *error_flag [[buffer(14)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
    uint head = group.y;
    if (head >= params.heads || activity[0] == 0) { return; }
    if (!authenticate_state(table,generations,params.access,error_flag)) { return; }
    scratch += layouts[8].offset;
    uint query_head = head/(params.heads/params.query_heads);
    uint column_start = group.x*params.panel+simd_group*8;
    short2 coordinate = SteelMMAFragment<float>::coordinate(ushort(lane));
    float2 state[(FORJA_DELTA_RULE_MAX_KEYS+7)/8];
    #pragma clang loop unroll(full)
    for (uint part=0; part<(FORJA_DELTA_RULE_MAX_KEYS+7)/8; ++part) {
        for (uint element=0; element<2; ++element) {
            uint key_index = part*8+uint(coordinate.y);
            uint column = column_start+uint(coordinate.x)+element;
            state[part][element] = key_index < params.keys && column < params.values
                ? previous[physical_index(layouts[5],(head*params.keys+key_index)*params.values+column)]
                : 0.0f;
        }
    }
    threadgroup float query_panel[1024];
    threadgroup float key_panel[1024];
    threadgroup float delta_panel[2048];
    for (uint block=0; block<params.blocks; ++block) {
        uint rows = delta_block_rows(params,block);
        uint base = delta_block_base(params,block,head);
        for (uint row_tile=0; row_tile<FORJA_DELTA_BLOCK_TOKENS; row_tile+=32) {
            float2 query_accumulators[4];
            float2 correction_accumulators[4];
            for (uint part=0; part<4; ++part) {
                query_accumulators[part] = float2(0.0f);
                correction_accumulators[part] = float2(0.0f);
            }
            #pragma clang loop unroll(full)
            for (uint inner=0; inner<FORJA_DELTA_RULE_MAX_KEYS; inner+=32) {
                if (inner < params.keys) {
                    for (uint cell=thread_index; cell<1024; cell+=group_size.x) {
                        uint row = row_tile+cell/32;
                        uint key_index = inner+cell%32;
                        float query_value = 0.0f;
                        float correction_value = 0.0f;
                        if (row < rows && key_index < params.keys) {
                            uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.query_heads+query_head)*params.keys+key_index;
                            query_value = load_float(query,physical_index(layouts[0],index),input0_dtype);
                            correction_value = scratch[base+params.offsets[5]+row*params.keys+key_index];
                        }
                        query_panel[cell] = query_value;
                        key_panel[cell] = correction_value;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    #pragma clang loop unroll(full)
                    for (uint fragment_inner=0; fragment_inner<32; fragment_inner+=8) {
                        #pragma clang loop unroll(full)
                        for (uint part=0; part<4; ++part) {
                            float2 left_query = delta_fragment_load(query_panel,part*8,fragment_inner,32,lane);
                            float2 left_correction = delta_fragment_load(key_panel,part*8,fragment_inner,32,lane);
                            delta_fragment_multiply(query_accumulators[part],left_query,state[inner/8+fragment_inner/8]);
                            delta_fragment_multiply(correction_accumulators[part],left_correction,state[inner/8+fragment_inner/8]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint part=0; part<4; ++part) {
                for (uint element=0; element<2; ++element) {
                    uint row = row_tile+part*8+uint(coordinate.y);
                    uint panel_column = simd_group*8+uint(coordinate.x)+element;
                    uint column = group.x*params.panel+panel_column;
                    float delta = 0.0f;
                    if (row < rows && column < params.values) {
                        uint slot = row*params.values+column;
                        delta = scratch[base+params.offsets[4]+slot]-correction_accumulators[part][element];
                        scratch[base+params.offsets[6]+slot] = delta;
                        scratch[base+params.offsets[7]+slot] = scratch[base+params.offsets[2]+row]*query_accumulators[part][element];
                    }
                    delta_panel[row*32+panel_column] = delta;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float decay = scratch[base+params.offsets[2]+rows-1];
        #pragma clang loop unroll(full)
        for (uint key_tile=0; key_tile<FORJA_DELTA_RULE_MAX_KEYS; key_tile+=32) {
            if (key_tile < params.keys) {
                float2 updates[4];
                for (uint part=0; part<4; ++part) { updates[part] = float2(0.0f); }
                for (uint row_tile=0; row_tile<FORJA_DELTA_BLOCK_TOKENS; row_tile+=32) {
                    if (row_tile < rows) {
                        for (uint cell=thread_index; cell<1024; cell+=group_size.x) {
                            uint key_index = key_tile+cell/32;
                            uint row = row_tile+cell%32;
                            float left = 0.0f;
                            if (key_index < params.keys && row < rows) {
                                uint index = ((block*FORJA_DELTA_BLOCK_TOKENS+row)*params.query_heads+query_head)*params.keys+key_index;
                                left = scratch[base+params.offsets[3]+row]*load_float(key,physical_index(layouts[1],index),input1_dtype);
                            }
                            key_panel[cell] = left;
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint fragment_inner=0; fragment_inner<32; fragment_inner+=8) {
                            float2 right = delta_fragment_load(delta_panel,row_tile+fragment_inner,simd_group*8,32,lane);
                            #pragma clang loop unroll(full)
                            for (uint part=0; part<4; ++part) {
                                float2 left = delta_fragment_load(key_panel,part*8,fragment_inner,32,lane);
                                delta_fragment_multiply(updates[part],left,right);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
                #pragma clang loop unroll(full)
                for (uint part=0; part<4; ++part) {
                    for (uint element=0; element<2; ++element) {
                        uint key_index = key_tile+part*8+uint(coordinate.y);
                        uint column = column_start+uint(coordinate.x)+element;
                        state[key_tile/8+part][element] = key_index < params.keys && column < params.values
                            ? decay*state[key_tile/8+part][element]+updates[part][element] : 0.0f;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint part=0; part<(FORJA_DELTA_RULE_MAX_KEYS+7)/8; ++part) {
        for (uint element=0; element<2; ++element) {
            uint key_index = part*8+uint(coordinate.y);
            uint column = column_start+uint(coordinate.x)+element;
            if (key_index < params.keys && column < params.values) {
                next[physical_index(layouts[7],(head*params.keys+key_index)*params.values+column)] = state[part][element];
            }
        }
    }
}
