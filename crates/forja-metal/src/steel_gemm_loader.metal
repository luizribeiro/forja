// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

template <int rows, int columns>
struct BlockLoader {
    template <typename T>
    static void load_to(
        device const uchar *source,
        ulong base,
        ulong row_stride,
        ulong column_stride,
        uint dtype,
        uint row_origin,
        uint column_origin,
        uint row_extent,
        uint column_extent,
        bool rows_aligned,
        bool columns_aligned,
        threadgroup T *destination,
        uint destination_row_stride,
        uint destination_column_stride,
        uint thread_index,
        uint thread_count) {
        for (uint linear = thread_index; linear < rows * columns; linear += thread_count) {
            uint row = linear / columns;
            uint column = linear % columns;
            bool valid = (rows_aligned || row_origin + row < row_extent) &&
                         (columns_aligned || column_origin + column < column_extent);
            ulong index = base + ulong(row_origin + row) * row_stride +
                          ulong(column_origin + column) * column_stride;
            uint destination_index = row * destination_row_stride
                + column * destination_column_stride;
            destination[destination_index] = valid ? T(load_float(source, index, dtype)) : T(0);
        }
    }

    template <typename T>
    static void load(
        device const uchar *source,
        ulong base,
        ulong row_stride,
        ulong column_stride,
        uint dtype,
        uint row_origin,
        uint column_origin,
        uint row_extent,
        uint column_extent,
        bool rows_aligned,
        bool columns_aligned,
        threadgroup T *destination,
        uint thread_index,
        uint thread_count) {
        load_to(
            source, base, row_stride, column_stride, dtype,
            row_origin, column_origin, row_extent, column_extent,
            rows_aligned, columns_aligned, destination, columns, 1,
            thread_index, thread_count);
    }
};
