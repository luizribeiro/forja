// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

#include <metal_simdgroup_matrix>

template <typename T>
struct SteelMMAFragment {
    static constant constexpr int rows = 8;
    static constant constexpr int columns = 8;
    static constant constexpr int elements = 2;

    using fragment_type = metal::vec<T, elements>;
    using matrix_type = metal::simdgroup_matrix<T, rows, columns>;

    static short2 coordinate(ushort lane) {
        short quad = lane / 4;
        short row = (quad & 4) + ((lane / 2) % 4);
        short column = (quad & 2) * 2 + (lane % 2) * 2;
        return short2(column, row);
    }

    template <typename U>
    static void load(
        thread fragment_type &values,
        threadgroup const U *source,
        int row_stride,
        int column_stride) {
        for (short element = 0; element < elements; ++element) {
            values[element] = T(source[element * column_stride]);
        }
    }

    template <typename A, typename B, typename C>
    static void multiply(
        thread fragment_type &result,
        thread metal::vec<A, elements> &left,
        thread metal::vec<B, elements> &right,
        thread metal::vec<C, elements> &accumulator) {
        metal::simdgroup_matrix<T, rows, columns> result_matrix;
        metal::simdgroup_matrix<A, rows, columns> left_matrix;
        metal::simdgroup_matrix<B, rows, columns> right_matrix;
        metal::simdgroup_matrix<C, rows, columns> accumulator_matrix;
        left_matrix.thread_elements() = left;
        right_matrix.thread_elements() = right;
        accumulator_matrix.thread_elements() = accumulator;
        simdgroup_multiply_accumulate(
            result_matrix, left_matrix, right_matrix, accumulator_matrix);
        result = result_matrix.thread_elements();
    }

    template <typename Operation>
    static void row_reduce(
        thread const fragment_type &values,
        thread T &result) {
        T pair = Operation::apply(values[0], values[1]);
        T quad = Operation::apply(pair, simd_shuffle_xor(pair, ushort(1)));
        result = Operation::apply(result, simd_shuffle_xor(quad, ushort(8)));
    }

    template <typename Operation>
    static void row_apply(
        thread fragment_type &values,
        thread const T &row_value) {
        for (short element = 0; element < elements; ++element) {
            values[element] = Operation::apply(values[element], row_value);
        }
    }
};

template <typename T, int tile_rows, int tile_columns>
struct SteelMMATile {
    using Fragment = SteelMMAFragment<T>;
    using fragment_type = typename Fragment::fragment_type;
    static constant constexpr int fragment_count = tile_rows * tile_columns;
    static constant constexpr int element_count = fragment_count * Fragment::elements;

    fragment_type fragments[fragment_count];

    void clear() thread {
        for (short index = 0; index < fragment_count; ++index) {
            fragments[index] = fragment_type(0);
        }
    }

    thread fragment_type &at(short row, short column) thread {
        return fragments[row * tile_columns + column];
    }

    thread T *elements() thread {
        return reinterpret_cast<thread T *>(fragments);
    }

    template <typename Operation>
    void row_reduce(thread T *values) const thread {
        for (short row = 0; row < tile_rows; ++row) {
            for (short column = 0; column < tile_columns; ++column) {
                Fragment::template row_reduce<Operation>(
                    fragments[row * tile_columns + column], values[row]);
            }
        }
    }

    template <typename Operation>
    void row_apply(thread T *values) thread {
        for (short row = 0; row < tile_rows; ++row) {
            for (short column = 0; column < tile_columns; ++column) {
                Fragment::template row_apply<Operation>(at(row, column), values[row]);
            }
        }
    }
};

template <typename T, int rows, int columns, int inner>
void steel_tile_multiply(
    thread SteelMMATile<T, rows, columns> &result,
    thread SteelMMATile<T, rows, inner> &left,
    thread SteelMMATile<T, inner, columns> &right,
    thread SteelMMATile<T, rows, columns> &accumulator) {
    for (short row = 0; row < rows; ++row) {
        for (short column = 0; column < columns; ++column) {
            for (short reduction = 0; reduction < inner; ++reduction) {
                SteelMMAFragment<T>::multiply(
                    result.at(row, column), left.at(row, reduction),
                    right.at(reduction, column), accumulator.at(row, column));
            }
        }
    }
}

template <int block_rows, int block_columns, int block_inner,
          int simdgroups_rows, int simdgroups_columns>
struct BlockMMA {
    static constant constexpr int fragment_size = 8;
    static constant constexpr int row_fragments =
        block_rows / (fragment_size * simdgroups_rows);
    static constant constexpr int column_fragments =
        block_columns / (fragment_size * simdgroups_columns);
    static constant constexpr int fragment_count = row_fragments * column_fragments;

    float accumulators[fragment_count * 2];
    ushort simdgroup_row;
    ushort simdgroup_column;
    ushort fragment_row;
    ushort fragment_column;

    BlockMMA(
        ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) thread
        : simdgroup_row(simdgroup_index / simdgroups_columns),
          simdgroup_column(simdgroup_index % simdgroups_columns) {
        ushort quad = lane / 4;
        fragment_row = (quad & 4) + ((lane / 2) % 4);
        fragment_column = (quad & 2) * 2 + (lane % 2) * 2;
        for (int element = 0; element < fragment_count * 2; ++element) {
            accumulators[element] = 0.0f;
        }
    }

    void multiply(
        threadgroup const float *a,
        threadgroup const float *b) thread {
        for (int inner = 0; inner < block_inner; inner += fragment_size) {
            for (int row_fragment = 0; row_fragment < row_fragments; ++row_fragment) {
                simdgroup_matrix<float, 8, 8> a_matrix;
                uint a_row = simdgroup_row * fragment_size + fragment_row +
                    row_fragment * fragment_size * simdgroups_rows;
                for (int element = 0; element < 2; ++element) {
                    a_matrix.thread_elements()[element] =
                        a[a_row * block_inner + inner + fragment_column + element];
                }
                for (int column_fragment = 0;
                     column_fragment < column_fragments;
                     ++column_fragment) {
                    simdgroup_matrix<float, 8, 8> b_matrix;
                    simdgroup_matrix<float, 8, 8> accumulator;
                    simdgroup_matrix<float, 8, 8> result;
                    uint b_column = simdgroup_column * fragment_size + fragment_column +
                        column_fragment * fragment_size * simdgroups_columns;
                    int fragment_index = row_fragment * column_fragments + column_fragment;
                    for (int element = 0; element < 2; ++element) {
                        b_matrix.thread_elements()[element] =
                            b[(inner + fragment_row) * block_columns + b_column + element];
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

    void store(
        device uchar *output,
        ulong base,
        ulong leading_dimension,
        uint row_origin,
        uint column_origin,
        uint row_extent,
        uint column_extent,
        uint dtype) const thread {
        for (int row_fragment = 0; row_fragment < row_fragments; ++row_fragment) {
            uint row = simdgroup_row * fragment_size + fragment_row +
                row_fragment * fragment_size * simdgroups_rows;
            for (int column_fragment = 0;
                 column_fragment < column_fragments;
                 ++column_fragment) {
                uint column = simdgroup_column * fragment_size + fragment_column +
                    column_fragment * fragment_size * simdgroups_columns;
                int fragment_index = row_fragment * column_fragments + column_fragment;
                for (int element = 0; element < 2; ++element) {
                    if (row_origin + row < row_extent &&
                        column_origin + column + element < column_extent) {
                        ulong index = base + ulong(row_origin + row) * leading_dimension +
                            ulong(column_origin + column + element);
                        store_float(
                            output, index, dtype, accumulators[fragment_index * 2 + element]);
                    }
                }
            }
        }
    }
};
