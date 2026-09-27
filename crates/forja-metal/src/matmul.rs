// Copyright © 2023-2026 Apple Inc.
//
// Licensed under the MIT License.

use forja_core::Layout;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
pub(super) enum MatrixLayout {
    RowMajor {
        leading_dimension: u64,
        batch_stride: u64,
    },
    ColumnMajor {
        leading_dimension: u64,
        batch_stride: u64,
    },
    NeedsCopy,
}

#[allow(dead_code)]
pub(super) fn classify(layout: &Layout) -> MatrixLayout {
    let rank = layout.shape().len();
    if !matches!(rank, 2 | 3) {
        return MatrixLayout::NeedsCopy;
    }
    let rows = u64::from(layout.shape()[rank - 2]);
    let columns = u64::from(layout.shape()[rank - 1]);
    let (Some(last_row), Some(last_column)) = (rows.checked_sub(1), columns.checked_sub(1)) else {
        return MatrixLayout::NeedsCopy;
    };
    let row_stride = layout.strides()[rank - 2];
    let column_stride = layout.strides()[rank - 1];
    let batch_stride = if rank == 3 { layout.strides()[0] } else { 0 };

    let row_span = row_stride
        .checked_mul(last_row)
        .and_then(|span| span.checked_add(columns));
    if column_stride == 1 && row_stride >= columns && batch_fits(layout, batch_stride, row_span) {
        return MatrixLayout::RowMajor {
            leading_dimension: row_stride,
            batch_stride,
        };
    }

    let column_span = column_stride
        .checked_mul(last_column)
        .and_then(|span| span.checked_add(rows));
    if row_stride == 1 && column_stride >= rows && batch_fits(layout, batch_stride, column_span) {
        return MatrixLayout::ColumnMajor {
            leading_dimension: column_stride,
            batch_stride,
        };
    }
    MatrixLayout::NeedsCopy
}

fn batch_fits(layout: &Layout, batch_stride: u64, matrix_span: Option<u64>) -> bool {
    layout.shape().len() == 2
        || layout.shape()[0] == 1
        || matrix_span.is_some_and(|span| batch_stride >= span)
}

#[cfg(test)]
mod tests {
    use forja_core::{DType, Layout, Slice};

    use super::{MatrixLayout, classify};

    #[test]
    fn classifies_dense_and_permuted_matrices() {
        let dense = Layout::contiguous(DType::F32, 0, vec![7, 33], 7 * 33 * 4).unwrap();
        assert_eq!(
            classify(&dense),
            MatrixLayout::RowMajor {
                leading_dimension: 33,
                batch_stride: 0
            }
        );
        let stored = Layout::contiguous(DType::F32, 0, vec![33, 7], 7 * 33 * 4).unwrap();
        assert_eq!(
            classify(&stored.permute(&[1, 0]).unwrap()),
            MatrixLayout::ColumnMajor {
                leading_dimension: 7,
                batch_stride: 0
            }
        );
    }

    #[test]
    fn preserves_padded_batch_and_rejects_irregular_views() {
        let cache =
            Layout::contiguous(DType::F16, 0, vec![16, 4097, 128], 16 * 4097 * 128 * 2).unwrap();
        let slice = cache
            .slice(&[
                Slice::new(0, 16, 1).unwrap(),
                Slice::new(37, 33, 1).unwrap(),
                Slice::new(0, 128, 1).unwrap(),
            ])
            .unwrap();
        assert_eq!(
            classify(&slice),
            MatrixLayout::RowMajor {
                leading_dimension: 128,
                batch_stride: 4097 * 128
            }
        );
        let stepped = cache
            .slice(&[
                Slice::new(0, 16, 1).unwrap(),
                Slice::new(0, 33, 1).unwrap(),
                Slice::new(0, 64, 2).unwrap(),
            ])
            .unwrap();
        assert_eq!(classify(&stepped), MatrixLayout::NeedsCopy);
        let single = Layout::contiguous(DType::F16, 0, vec![1, 33, 128], 33 * 128 * 2).unwrap();
        let broadcast = single.broadcast(&[16, 33, 128]).unwrap();
        assert_eq!(classify(&broadcast), MatrixLayout::NeedsCopy);
        let overlapping = Layout::new(DType::F16, 0, vec![2, 3, 4], vec![4, 4, 1], 32).unwrap();
        assert_eq!(classify(&overlapping), MatrixLayout::NeedsCopy);
    }

    #[test]
    fn rejects_zero_extent_matrices_without_arithmetic() {
        for shape in [vec![0, 33], vec![33, 0], vec![7, 0, 33]] {
            let layout = Layout::contiguous(DType::F32, 0, shape, 0).unwrap();
            assert_eq!(classify(&layout), MatrixLayout::NeedsCopy);
        }
    }
}
