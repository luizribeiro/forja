// Copyright © 2023-2026 Apple Inc.
//
// Licensed under the MIT License.

use forja_core::{DType, Layout};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GemmConfig {
    pub(super) kernel: &'static str,
    pub(super) block_rows: u32,
    pub(super) block_columns: u32,
    pub(super) block_inner: u32,
    pub(super) thread_count: usize,
}

pub(super) fn select_internal_gemm(
    dtype: DType,
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    left_column_major: bool,
    right_column_major: bool,
) -> Option<GemmConfig> {
    let output_elements = u64::from(batch)
        .checked_mul(u64::from(rows))?
        .checked_mul(u64::from(columns))?;
    let large = output_elements >= 1 << 20;
    let nt = !left_column_major && right_column_major;
    let half = matches!(dtype, DType::F16 | DType::BF16);
    if half && large && u64::from(rows.max(columns)) * 2 > u64::from(inner) {
        return Some(GEMM_64_64_HALF);
    }
    if half && nt {
        return Some(GEMM_64_32);
    }
    if half && large {
        return Some(GEMM_32_64);
    }
    if half {
        return Some(GEMM_64_64_HALF);
    }
    if !large && nt {
        return Some(GEMM_32_64);
    }
    if !large {
        return Some(GEMM_64_32);
    }
    Some(GEMM_64_64_FLOAT)
}

pub(super) const GEMM_64_64_FLOAT: GemmConfig = GemmConfig {
    kernel: "steel_gemm_64_64_16_2_2",
    block_rows: 64,
    block_columns: 64,
    block_inner: 16,
    thread_count: 128,
};
pub(super) const GEMM_64_64_HALF: GemmConfig = GemmConfig {
    kernel: "steel_gemm_64_64_16_1_2",
    block_rows: 64,
    block_columns: 64,
    block_inner: 16,
    thread_count: 64,
};
pub(super) const GEMM_64_32: GemmConfig = GemmConfig {
    kernel: "steel_gemm_64_32_32_2_2",
    block_rows: 64,
    block_columns: 32,
    block_inner: 32,
    thread_count: 128,
};
pub(super) const GEMM_32_64: GemmConfig = GemmConfig {
    kernel: "steel_gemm_32_64_16_1_2",
    block_rows: 32,
    block_columns: 64,
    block_inner: 16,
    thread_count: 64,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

impl MatrixLayout {
    pub(super) const fn kernel_strides(self) -> Option<(u32, u64, u64)> {
        match self {
            Self::RowMajor {
                leading_dimension,
                batch_stride,
            } => Some((0, leading_dimension, batch_stride)),
            Self::ColumnMajor {
                leading_dimension,
                batch_stride,
            } => Some((1, leading_dimension, batch_stride)),
            Self::NeedsCopy => None,
        }
    }
}

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

    use super::{
        GEMM_32_64, GEMM_64_32, GEMM_64_64_FLOAT, GEMM_64_64_HALF, MatrixLayout, classify,
        select_internal_gemm,
    };

    #[test]
    fn selects_internal_tile_configs_by_shape_and_dtype() {
        assert_eq!(
            select_internal_gemm(DType::BF16, 1, 512, 3072, 1024, false, true),
            Some(GEMM_64_64_HALF)
        );
        assert_eq!(
            select_internal_gemm(DType::F16, 1, 128, 3072, 1024, false, true),
            Some(GEMM_64_32)
        );
        assert_eq!(
            select_internal_gemm(DType::BF16, 1, 512, 3072, 8192, false, false),
            Some(GEMM_32_64)
        );
        assert_eq!(
            select_internal_gemm(DType::F32, 1, 128, 3072, 1024, false, true),
            Some(GEMM_32_64)
        );
        assert_eq!(
            select_internal_gemm(DType::F32, 1, 128, 3072, 1024, false, false),
            Some(GEMM_64_32)
        );
        assert_eq!(
            select_internal_gemm(DType::F32, 16, 128, 1024, 1024, false, false),
            Some(GEMM_64_64_FLOAT)
        );
    }

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
