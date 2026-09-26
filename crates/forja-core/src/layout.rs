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
    /// A slice step was zero.
    ZeroSliceStep,
    /// The number of slice specifications differs from the tensor rank.
    SliceRankMismatch,
    /// A slice selects an element outside its parent axis.
    SliceOutOfBounds {
        /// The axis containing the invalid slice.
        axis: usize,
    },
    /// The permutation length differs from the tensor rank.
    PermutationRankMismatch,
    /// A permutation references an axis outside the tensor rank.
    PermutationAxisOutOfRange {
        /// The invalid axis number.
        axis: u8,
    },
    /// A permutation references the same axis more than once.
    DuplicatePermutationAxis {
        /// The repeated axis number.
        axis: u8,
    },
    /// A broadcast target has fewer axes than its source.
    BroadcastRankReduction,
    /// A source extent cannot broadcast to its target extent.
    BroadcastDimensionMismatch {
        /// The target axis containing the mismatch.
        axis: usize,
        /// The source extent.
        source: u32,
        /// The requested target extent.
        target: u32,
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
            Self::ZeroSliceStep => formatter.write_str("slice step must be at least one"),
            Self::SliceRankMismatch => {
                formatter.write_str("slice specification rank differs from tensor rank")
            }
            Self::SliceOutOfBounds { axis } => {
                write!(formatter, "slice exceeds parent extent on axis {axis}")
            }
            Self::PermutationRankMismatch => {
                formatter.write_str("permutation length differs from tensor rank")
            }
            Self::PermutationAxisOutOfRange { axis } => {
                write!(formatter, "permutation axis {axis} is out of range")
            }
            Self::DuplicatePermutationAxis { axis } => {
                write!(formatter, "permutation axis {axis} appears more than once")
            }
            Self::BroadcastRankReduction => {
                formatter.write_str("broadcast target cannot remove axes")
            }
            Self::BroadcastDimensionMismatch {
                axis,
                source,
                target,
            } => write!(
                formatter,
                "source extent {source} cannot broadcast to {target} on axis {axis}"
            ),
        }
    }
}

impl Error for LayoutError {}

/// A validated selection along one tensor axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slice {
    start: u32,
    len: u32,
    step: u32,
}

impl Slice {
    /// Creates an axis slice with a nonzero step.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError::ZeroSliceStep`] when `step` is zero.
    pub const fn new(start: u32, len: u32, step: u32) -> Result<Self, LayoutError> {
        if step == 0 {
            return Err(LayoutError::ZeroSliceStep);
        }
        Ok(Self { start, len, step })
    }

    /// Returns the first selected index.
    #[must_use]
    pub const fn start(self) -> u32 {
        self.start
    }

    /// Returns the number of selected indices.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns whether this slice selects no indices.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Returns the distance between selected indices.
    #[must_use]
    pub const fn step(self) -> u32 {
        self.step
    }
}

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

    /// Creates a view by slicing every axis of this layout.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when the slice rank differs, a selection leaves
    /// its parent extent, or checked layout arithmetic fails.
    pub fn slice(&self, slices: &[Slice]) -> Result<Self, LayoutError> {
        if slices.len() != self.shape.len() {
            return Err(LayoutError::SliceRankMismatch);
        }
        for (axis, (&extent, slice)) in self.shape.iter().zip(slices).enumerate() {
            let last = if slice.is_empty() {
                u64::from(slice.start)
            } else {
                u64::from(slice.len - 1)
                    .checked_mul(u64::from(slice.step))
                    .and_then(|distance| u64::from(slice.start).checked_add(distance))
                    .ok_or(LayoutError::ArithmeticOverflow)?
            };
            let in_bounds = if slice.is_empty() {
                last <= u64::from(extent)
            } else {
                last < u64::from(extent)
            };
            if !in_bounds {
                return Err(LayoutError::SliceOutOfBounds { axis });
            }
        }
        let shape = slices.iter().map(|slice| slice.len).collect::<Vec<_>>();
        let strides = self
            .strides
            .iter()
            .zip(slices)
            .map(|(&stride, slice)| stride.checked_mul(u64::from(slice.step)))
            .collect::<Option<Vec<_>>>()
            .ok_or(LayoutError::ArithmeticOverflow)?;
        let offset = if shape.contains(&0) {
            self.offset
        } else {
            self.strides
                .iter()
                .zip(slices)
                .try_fold(self.offset, |offset, (&stride, slice)| {
                    stride
                        .checked_mul(u64::from(slice.start))
                        .and_then(|distance| offset.checked_add(distance))
                        .ok_or(LayoutError::ArithmeticOverflow)
                })?
        };
        Self::new(self.dtype, offset, shape, strides, self.buffer_len)
    }

    /// Creates a view with axes in the given order.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] unless `axes` is a permutation of every axis,
    /// or when revalidating the resulting layout fails.
    pub fn permute(&self, axes: &[u8]) -> Result<Self, LayoutError> {
        if axes.len() != self.shape.len() {
            return Err(LayoutError::PermutationRankMismatch);
        }
        let mut seen = [false; MAX_RANK];
        for &axis in axes {
            let index = usize::from(axis);
            if index >= self.shape.len() {
                return Err(LayoutError::PermutationAxisOutOfRange { axis });
            }
            if seen[index] {
                return Err(LayoutError::DuplicatePermutationAxis { axis });
            }
            seen[index] = true;
        }
        let shape = axes
            .iter()
            .map(|&axis| self.shape[usize::from(axis)])
            .collect();
        let strides = axes
            .iter()
            .map(|&axis| self.strides[usize::from(axis)])
            .collect();
        Self::new(self.dtype, self.offset, shape, strides, self.buffer_len)
    }

    /// Creates a view broadcast to `target_shape` using `NumPy` rules.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when the target removes axes, corresponding
    /// extents are neither equal nor size one, or revalidation fails.
    pub fn broadcast(&self, target_shape: &[u32]) -> Result<Self, LayoutError> {
        if target_shape.len() > MAX_RANK {
            return Err(LayoutError::RankTooLarge);
        }
        let leading = target_shape
            .len()
            .checked_sub(self.shape.len())
            .ok_or(LayoutError::BroadcastRankReduction)?;
        let mut strides = vec![0; leading];
        for (source_axis, (&source, &target)) in
            self.shape.iter().zip(&target_shape[leading..]).enumerate()
        {
            let stride = match (source, target) {
                (source, target) if source == target => self.strides[source_axis],
                (1, _) => 0,
                _ => {
                    return Err(LayoutError::BroadcastDimensionMismatch {
                        axis: leading + source_axis,
                        source,
                        target,
                    });
                }
            };
            strides.push(stride);
        }
        Self::new(
            self.dtype,
            self.offset,
            target_shape.to_vec(),
            strides,
            self.buffer_len,
        )
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

    use super::{Layout, LayoutError, Slice};
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

    #[test]
    fn slices_kv_cache_to_current_length() {
        let layout = Layout::contiguous(DType::F16, 0, vec![8, 4096, 128], 8_388_608).unwrap();
        let slices = [
            Slice::new(0, 8, 1).unwrap(),
            Slice::new(0, 33, 1).unwrap(),
            Slice::new(0, 128, 1).unwrap(),
        ];
        let current = layout.slice(&slices).unwrap();
        assert_eq!(current.shape(), [8, 33, 128]);
        assert_eq!(current.strides(), [524_288, 128, 1]);
        assert_eq!(current.byte_span(), 0..7_348_480);
    }

    #[test]
    fn rejects_invalid_slices() {
        assert_eq!(Slice::new(0, 1, 0), Err(LayoutError::ZeroSliceStep));
        let layout = Layout::contiguous(DType::F32, 0, vec![7], 28).unwrap();
        assert_eq!(
            layout.slice(&[Slice::new(6, 2, 1).unwrap()]),
            Err(LayoutError::SliceOutOfBounds { axis: 0 })
        );
    }

    #[test]
    fn permutes_q_heads_before_attention() {
        let layout = Layout::contiguous(DType::F16, 0, vec![33, 16, 128], 135_168).unwrap();
        let heads_first = layout.permute(&[1, 0, 2]).unwrap();
        assert_eq!(heads_first.shape(), [16, 33, 128]);
        assert_eq!(heads_first.strides(), [128, 2048, 1]);
        assert_eq!(heads_first.byte_span(), layout.byte_span());
    }

    #[test]
    fn rejects_invalid_permutations() {
        let layout = Layout::contiguous(DType::F32, 0, vec![7, 33], 924).unwrap();
        assert_eq!(
            layout.permute(&[0]),
            Err(LayoutError::PermutationRankMismatch)
        );
        assert_eq!(
            layout.permute(&[0, 2]),
            Err(LayoutError::PermutationAxisOutOfRange { axis: 2 })
        );
        assert_eq!(
            layout.permute(&[1, 1]),
            Err(LayoutError::DuplicatePermutationAxis { axis: 1 })
        );
    }

    #[test]
    fn broadcasts_q_heads_across_sequence() {
        let layout = Layout::contiguous(DType::F16, 0, vec![1, 16, 128], 4096).unwrap();
        let broadcast = layout.broadcast(&[33, 16, 128]).unwrap();
        assert_eq!(broadcast.shape(), [33, 16, 128]);
        assert_eq!(broadcast.strides(), [0, 128, 1]);
        assert_eq!(broadcast.byte_span(), layout.byte_span());
    }

    #[test]
    fn rejects_incompatible_broadcast() {
        let layout = Layout::contiguous(DType::F32, 0, vec![7, 33], 924).unwrap();
        assert_eq!(
            layout.broadcast(&[7, 1]),
            Err(LayoutError::BroadcastDimensionMismatch {
                axis: 1,
                source: 33,
                target: 1
            })
        );
    }

    #[test]
    fn rejects_broadcast_rank_before_building_strides() {
        let layout = Layout::contiguous(DType::F32, 0, vec![1], 4).unwrap();
        assert_eq!(layout.broadcast(&[1; 9]), Err(LayoutError::RankTooLarge));
    }

    proptest! {
        #[test]
        fn broadcast_only_zeroes_stretched_strides(
            axes in prop::collection::vec((1_u32..=4, 0_u64..=8, 1_u32..=4), 0..=4),
            leading in prop::collection::vec(1_u32..=4, 0..=4),
        ) {
            prop_assume!(axes.len() + leading.len() <= super::MAX_RANK);
            let shape = axes.iter().map(|axis| axis.0).collect::<Vec<_>>();
            let strides = axes.iter().map(|axis| axis.1).collect::<Vec<_>>();
            let source = Layout::new(DType::U32, 0, shape, strides, 4096).unwrap();
            let mut target = leading.clone();
            target.extend(axes.iter().map(|&(extent, _, stretch)| {
                if extent == 1 { stretch } else { extent }
            }));
            let result = source.broadcast(&target).unwrap();
            prop_assert!(result.strides()[..leading.len()].iter().all(|&stride| stride == 0));
            for (axis, (&before, &after)) in source.strides().iter()
                .zip(&result.strides()[leading.len()..]).enumerate()
            {
                if source.shape()[axis] == target[leading.len() + axis] {
                    prop_assert_eq!(after, before);
                } else {
                    prop_assert_eq!(after, 0);
                    prop_assert_eq!(source.shape()[axis], 1);
                }
            }
        }

        #[test]
        fn sliced_elements_map_to_parent(
            axes in prop::collection::vec((1_u32..=4, 0_u32..=10, 0_u32..=10, 1_u32..=3), 0..=4),
        ) {
            let shape = axes.iter().map(|axis| axis.0).collect::<Vec<_>>();
            let elements = shape.iter().map(|&extent| u64::from(extent)).product::<u64>();
            let parent = Layout::contiguous(DType::U32, 0, shape, elements * 4).unwrap();
            let slices = axes.iter().map(|&(extent, raw_start, raw_len, step)| {
                let start = raw_start % extent;
                let max_len = (extent - 1 - start) / step + 1;
                Slice::new(start, raw_len % (max_len + 1), step).unwrap()
            }).collect::<Vec<_>>();
            let child = parent.slice(&slices).unwrap();
            for mut linear in 0..child.element_count() {
                let mut child_offset = child.offset();
                let mut parent_offset = parent.offset();
                for ((&extent, &stride), (&parent_stride, slice)) in child.shape()
                    .iter().zip(child.strides()).zip(parent.strides().iter().zip(&slices)).rev()
                {
                    let index = linear % u64::from(extent);
                    linear /= u64::from(extent);
                    child_offset += index * stride;
                    parent_offset += (u64::from(slice.start()) + index * u64::from(slice.step()))
                        * parent_stride;
                }
                prop_assert_eq!(child_offset, parent_offset);
            }
        }

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
