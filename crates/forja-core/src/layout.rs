use std::{error::Error, fmt, ops::Range};

use crate::DType;

/// The largest number of axes accepted by a tensor layout.
pub const MAX_RANK: usize = 8;

/// A reason that tensor layout construction failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// The shape has more axes than [`MAX_RANK`].
    RankTooLarge,
    /// The shape and stride lists have different lengths.
    StrideRankMismatch,
    /// An element or byte calculation exceeded 64 bits.
    ArithmeticOverflow,
    /// The layout would touch bytes outside its buffer.
    OutOfBounds {
        /// The exclusive upper byte bound required by the layout.
        required: u64,
        /// The available buffer length in bytes.
        available: u64,
    },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankTooLarge => formatter.write_str("tensor rank exceeds the supported maximum"),
            Self::StrideRankMismatch => formatter.write_str("shape and stride ranks differ"),
            Self::ArithmeticOverflow => formatter.write_str("layout arithmetic overflowed"),
            Self::OutOfBounds {
                required,
                available,
            } => write!(
                formatter,
                "layout requires {required} bytes but its buffer has {available}"
            ),
        }
    }
}

impl Error for LayoutError {}

/// An immutable, buffer-validated tensor view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout {
    dtype: DType,
    offset: u64,
    shape: Vec<u32>,
    strides: Vec<u64>,
    buffer_len: u64,
    element_count: u64,
    byte_span: Range<u64>,
}

impl Layout {
    /// Creates and validates a layout with explicit element strides.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when ranks disagree, arithmetic overflows, or
    /// the resulting byte span exceeds `buffer_len`.
    pub fn new(
        dtype: DType,
        offset: u64,
        shape: Vec<u32>,
        strides: Vec<u64>,
        buffer_len: u64,
    ) -> Result<Self, LayoutError> {
        if shape.len() > MAX_RANK {
            return Err(LayoutError::RankTooLarge);
        }
        if shape.len() != strides.len() {
            return Err(LayoutError::StrideRankMismatch);
        }
        let element_count = element_count(&shape)?;
        let byte_span = validate_span(dtype, offset, &shape, &strides, buffer_len)?;
        Ok(Self {
            dtype,
            offset,
            shape,
            strides,
            buffer_len,
            element_count,
            byte_span,
        })
    }

    /// Creates and validates a row-major contiguous layout at an element offset.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when the rank is unsupported, arithmetic
    /// overflows, or the resulting byte span exceeds `buffer_len`.
    pub fn contiguous(
        dtype: DType,
        offset: u64,
        shape: Vec<u32>,
        buffer_len: u64,
    ) -> Result<Self, LayoutError> {
        if shape.len() > MAX_RANK {
            return Err(LayoutError::RankTooLarge);
        }
        let strides = contiguous_strides(&shape)?;
        Self::new(dtype, offset, shape, strides, buffer_len)
    }

    /// Returns the scalar type read by this layout.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the element offset of the layout origin.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the extent of each axis.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Returns the element stride of each axis.
    #[must_use]
    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    /// Returns the byte length of the buffer used during validation.
    #[must_use]
    pub const fn buffer_len(&self) -> u64 {
        self.buffer_len
    }

    /// Returns the number of logical elements in the view.
    #[must_use]
    pub const fn element_count(&self) -> u64 {
        self.element_count
    }

    /// Returns the first and one-past-last byte that the view may touch.
    #[must_use]
    pub fn byte_span(&self) -> Range<u64> {
        self.byte_span.clone()
    }
}

fn element_count(shape: &[u32]) -> Result<u64, LayoutError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1_u64, |count, &extent| {
        count
            .checked_mul(u64::from(extent))
            .ok_or(LayoutError::ArithmeticOverflow)
    })
}

fn contiguous_strides(shape: &[u32]) -> Result<Vec<u64>, LayoutError> {
    if shape.contains(&0) {
        return Ok(vec![0; shape.len()]);
    }
    let mut stride = 1_u64;
    let mut strides = vec![0; shape.len()];
    for (axis, &extent) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(u64::from(extent))
            .ok_or(LayoutError::ArithmeticOverflow)?;
    }
    Ok(strides)
}

fn validate_span(
    dtype: DType,
    offset: u64,
    shape: &[u32],
    strides: &[u64],
    buffer_len: u64,
) -> Result<Range<u64>, LayoutError> {
    if shape.contains(&0) {
        return Ok(0..0);
    }
    let last = shape
        .iter()
        .zip(strides)
        .try_fold(offset, |last, (&extent, &stride)| {
            u64::from(extent - 1)
                .checked_mul(stride)
                .and_then(|distance| last.checked_add(distance))
                .ok_or(LayoutError::ArithmeticOverflow)
        })?;
    let first_byte = offset
        .checked_mul(dtype.byte_size())
        .ok_or(LayoutError::ArithmeticOverflow)?;
    let end_byte = last
        .checked_add(1)
        .and_then(|end| end.checked_mul(dtype.byte_size()))
        .ok_or(LayoutError::ArithmeticOverflow)?;
    if end_byte > buffer_len {
        return Err(LayoutError::OutOfBounds {
            required: end_byte,
            available: buffer_len,
        });
    }
    Ok(first_byte..end_byte)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{Layout, LayoutError};
    use crate::DType;

    #[test]
    fn constructs_realistic_contiguous_layout() {
        let layout = Layout::contiguous(DType::F16, 7, vec![33, 16, 128], 135_182).unwrap();
        assert_eq!(layout.strides(), [2048, 128, 1]);
        assert_eq!(layout.element_count(), 67_584);
        assert_eq!(layout.byte_span(), 14..135_182);
    }

    #[test]
    fn reports_out_of_bounds_and_overflow() {
        assert_eq!(
            Layout::new(DType::U32, 0, vec![2], vec![2], 8),
            Err(LayoutError::OutOfBounds {
                required: 12,
                available: 8
            })
        );
        assert_eq!(
            Layout::new(DType::U32, u64::MAX, vec![1], vec![1], u64::MAX),
            Err(LayoutError::ArithmeticOverflow)
        );
    }

    proptest! {
        #[test]
        fn validation_matches_brute_force_maximum(
            axes in prop::collection::vec((0_u32..=4, 0_u64..=8), 0..=4),
            offset in 0_u64..=20,
            buffer_elements in 0_u64..=100,
        ) {
            let (shape, strides): (Vec<_>, Vec<_>) = axes.into_iter().unzip();
            let expected = brute_maximum(&shape, &strides, offset)
                .is_none_or(|maximum| maximum < buffer_elements);
            let actual = Layout::new(
                DType::U32,
                offset,
                shape,
                strides,
                buffer_elements * DType::U32.byte_size(),
            );
            prop_assert_eq!(actual.is_ok(), expected);
        }

        #[test]
        fn validation_matches_wide_u128_arithmetic(
            axes in prop::collection::vec((0_u32..=3, wide_u64()), 0..=4),
            offset in wide_u64(),
            buffer_len in wide_u64(),
            dtype in dtype_strategy(),
        ) {
            let (shape, strides): (Vec<_>, Vec<_>) = axes.into_iter().unzip();
            let expected = u128_layout_fits(dtype, offset, &shape, &strides, buffer_len);
            let actual = Layout::new(dtype, offset, shape, strides, buffer_len);
            prop_assert_eq!(actual.is_ok(), expected);
        }
    }

    fn wide_u64() -> impl Strategy<Value = u64> {
        prop_oneof![
            3 => 0_u64..=64,
            5 => (1_u64..=16).prop_map(|divisor| u64::MAX / divisor),
            2 => any::<u64>(),
        ]
    }

    fn dtype_strategy() -> impl Strategy<Value = DType> {
        prop_oneof![
            Just(DType::F32),
            Just(DType::F16),
            Just(DType::BF16),
            Just(DType::I32),
            Just(DType::U32),
        ]
    }

    fn u128_layout_fits(
        dtype: DType,
        offset: u64,
        shape: &[u32],
        strides: &[u64],
        buffer_len: u64,
    ) -> bool {
        if shape.contains(&0) {
            return true;
        }
        let last = shape
            .iter()
            .zip(strides)
            .fold(u128::from(offset), |last, (&extent, &stride)| {
                last + u128::from(extent - 1) * u128::from(stride)
            });
        (last + 1) * u128::from(dtype.byte_size()) <= u128::from(buffer_len)
    }

    fn brute_maximum(shape: &[u32], strides: &[u64], offset: u64) -> Option<u64> {
        let count = shape
            .iter()
            .map(|&extent| extent as usize)
            .product::<usize>();
        (0..count)
            .map(|mut linear| {
                let mut element = offset;
                for (&extent, &stride) in shape.iter().zip(strides).rev() {
                    element += (linear % extent as usize) as u64 * stride;
                    linear /= extent as usize;
                }
                element
            })
            .max()
    }
}
