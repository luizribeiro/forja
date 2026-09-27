// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT

template <int rows, int columns>
struct BlockLoader {
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
        threadgroup float *destination,
        uint thread_index,
        uint thread_count) {
        for (uint linear = thread_index; linear < rows * columns; linear += thread_count) {
            uint row = linear / columns;
            uint column = linear % columns;
            bool valid = (rows_aligned || row_origin + row < row_extent) &&
                         (columns_aligned || column_origin + column < column_extent);
            ulong index = base + ulong(row_origin + row) * row_stride +
                          ulong(column_origin + column) * column_stride;
            destination[linear] = valid ? load_float(source, index, dtype) : 0.0f;
        }
    }
};
