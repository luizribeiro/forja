// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

    void multiply(
        threadgroup const T *a,
        threadgroup const T *b) thread {
        for (int inner = 0; inner < block_inner; inner += fragment_size) {
            for (int row_fragment = 0; row_fragment < row_fragments; ++row_fragment) {
                simdgroup_matrix<mma_type, 8, 8> a_matrix;
                uint a_row = simdgroup_row * fragment_size + fragment_row +
                    row_fragment * fragment_size * simdgroups_rows;
                for (int element = 0; element < 2; ++element) {
                    a_matrix.thread_elements()[element] = mma_type(
                        a[a_row * left_stride + inner + fragment_column + element]);
                }
                for (int column_fragment = 0;
                     column_fragment < column_fragments;
                     ++column_fragment) {
                    simdgroup_matrix<mma_type, 8, 8> b_matrix;
                    simdgroup_matrix<float, 8, 8> accumulator;
                    simdgroup_matrix<float, 8, 8> result;
                    uint b_column = simdgroup_column * fragment_size + fragment_column +
                        column_fragment * fragment_size * simdgroups_columns;
                    int fragment_index = row_fragment * column_fragments + column_fragment;
                    for (int element = 0; element < 2; ++element) {
                        b_matrix.thread_elements()[element] = mma_type(
                            right_transposed
                                ? b[(b_column + element) * right_stride + inner + fragment_row]
                                : b[(inner + fragment_row) * right_stride + b_column + element]);
                        accumulator.thread_elements()[element] =
                            accumulators[fragment_index * 2 + element];
                    }
                    simdgroup_multiply_accumulate(
                        result, a_matrix, b_matrix, accumulator);
                    for (int element = 0; element < 2; ++element) {
                        accumulators[fragment_index * 2 + element] =
                            result.thread_elements()[element];
                    }
                }
            }
        }
    }
