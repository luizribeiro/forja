use std::{error::Error, fmt};

use crate::{DType, Layout};

/// A component of an affine quantized matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizedMatrixPart {
    /// Packed unsigned integer words.
    Packed,
    /// Per-group multiplicative factors.
    Scales,
    /// Per-group additive factors.
    Biases,
}

/// A reason that quantized matrix construction failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizedMatrixError {
    /// The bit width is neither 4 nor 8.
    UnsupportedBitWidth,
    /// The group size is not 32, 64, or 128.
    UnsupportedGroupSize,
    /// The column count is not divisible by the number packed per word.
    ColumnsNotPackable,
    /// The column count is not divisible by the group size.
    ColumnsNotGrouped,
    /// A component has an unsupported scalar type.
    DTypeMismatch {
        /// The component with the invalid type.
        part: QuantizedMatrixPart,
    },
    /// A component does not have its required rank-2 shape.
    ShapeMismatch {
        /// The component with the invalid shape.
        part: QuantizedMatrixPart,
    },
}

impl fmt::Display for QuantizedMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBitWidth => formatter.write_str("quantized bit width must be 4 or 8"),
            Self::UnsupportedGroupSize => {
                formatter.write_str("quantized group size must be 32, 64, or 128")
            }
            Self::ColumnsNotPackable => {
                formatter.write_str("column count does not fill packed words")
            }
            Self::ColumnsNotGrouped => {
                formatter.write_str("column count does not fill quantization groups")
            }
            Self::DTypeMismatch { part } => write!(formatter, "invalid {part:?} data type"),
            Self::ShapeMismatch { part } => write!(formatter, "invalid {part:?} shape"),
        }
    }
}

impl Error for QuantizedMatrixError {}

/// Three validated layouts that encode an affine quantized rank-2 tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedMatrix {
    shape: [u32; 2],
    bits: u8,
    group: u32,
    packed: Layout,
    scales: Layout,
    biases: Layout,
}

impl QuantizedMatrix {
    /// Groups compatible packed words, scales, and biases into one matrix.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizedMatrixError`] when quantization parameters, dtypes,
    /// or component shapes do not match the logical matrix dimensions.
    pub fn new(
        rows: u32,
        cols: u32,
        bits: u8,
        group: u32,
        packed: Layout,
        scales: Layout,
        biases: Layout,
    ) -> Result<Self, QuantizedMatrixError> {
        if !matches!(bits, 4 | 8) {
            return Err(QuantizedMatrixError::UnsupportedBitWidth);
        }
        if !matches!(group, 32 | 64 | 128) {
            return Err(QuantizedMatrixError::UnsupportedGroupSize);
        }
        let per_word = 32 / u32::from(bits);
        if !cols.is_multiple_of(per_word) {
            return Err(QuantizedMatrixError::ColumnsNotPackable);
        }
        if !cols.is_multiple_of(group) {
            return Err(QuantizedMatrixError::ColumnsNotGrouped);
        }
        validate_part(
            &packed,
            DType::U32,
            [rows, cols / per_word],
            QuantizedMatrixPart::Packed,
        )?;
        validate_parameter(&scales, [rows, cols / group], QuantizedMatrixPart::Scales)?;
        validate_parameter(&biases, [rows, cols / group], QuantizedMatrixPart::Biases)?;
        Ok(Self {
            shape: [rows, cols],
            bits,
            group,
            packed,
            scales,
            biases,
        })
    }

    /// Returns the logical row and column counts.
    #[must_use]
    pub const fn shape(&self) -> [u32; 2] {
        self.shape
    }

    /// Returns the number of bits stored for each logical value.
    #[must_use]
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// Returns the number of logical values sharing scale and bias values.
    #[must_use]
    pub const fn group(&self) -> u32 {
        self.group
    }

    /// Returns the packed-word layout used by quantized kernels.
    #[must_use]
    pub const fn packed(&self) -> &Layout {
        &self.packed
    }

    /// Returns the per-group scale layout used by quantized kernels.
    #[must_use]
    pub const fn scales(&self) -> &Layout {
        &self.scales
    }

    /// Returns the per-group bias layout used by quantized kernels.
    #[must_use]
    pub const fn biases(&self) -> &Layout {
        &self.biases
    }
}

fn validate_parameter(
    layout: &Layout,
    shape: [u32; 2],
    part: QuantizedMatrixPart,
) -> Result<(), QuantizedMatrixError> {
    if !matches!(layout.dtype(), DType::F16 | DType::BF16) {
        return Err(QuantizedMatrixError::DTypeMismatch { part });
    }
    validate_shape(layout, shape, part)
}

fn validate_part(
    layout: &Layout,
    dtype: DType,
    shape: [u32; 2],
    part: QuantizedMatrixPart,
) -> Result<(), QuantizedMatrixError> {
    if layout.dtype() != dtype {
        return Err(QuantizedMatrixError::DTypeMismatch { part });
    }
    validate_shape(layout, shape, part)
}

fn validate_shape(
    layout: &Layout,
    shape: [u32; 2],
    part: QuantizedMatrixPart,
) -> Result<(), QuantizedMatrixError> {
    if layout.shape() != shape {
        return Err(QuantizedMatrixError::ShapeMismatch { part });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{QuantizedMatrix, QuantizedMatrixError, QuantizedMatrixPart};
    use crate::{DType, Layout};

    #[test]
    fn constructs_qwen_mlp_weights() {
        let q4 = matrix(4, 64, DType::F16).unwrap();
        assert_eq!(q4.shape(), [3072, 1024]);
        assert_eq!(q4.packed().shape(), [3072, 128]);
        assert_eq!(q4.scales().shape(), [3072, 16]);

        let q8 = matrix(8, 32, DType::BF16).unwrap();
        assert_eq!(q8.packed().shape(), [3072, 256]);
        assert_eq!(q8.biases().shape(), [3072, 32]);
    }

    #[test]
    fn rejects_invalid_parameters_and_shapes() {
        let packed = layout(DType::U32, [3072, 128]);
        let params = layout(DType::F16, [3072, 16]);
        assert_eq!(
            QuantizedMatrix::new(
                3072,
                1024,
                3,
                64,
                packed.clone(),
                params.clone(),
                params.clone()
            ),
            Err(QuantizedMatrixError::UnsupportedBitWidth)
        );
        assert_eq!(
            QuantizedMatrix::new(
                3072,
                1024,
                4,
                16,
                packed.clone(),
                params.clone(),
                params.clone()
            ),
            Err(QuantizedMatrixError::UnsupportedGroupSize)
        );
        assert_eq!(
            QuantizedMatrix::new(
                3072,
                1025,
                4,
                64,
                packed.clone(),
                params.clone(),
                params.clone()
            ),
            Err(QuantizedMatrixError::ColumnsNotPackable)
        );
        assert_eq!(
            QuantizedMatrix::new(3072, 1040, 4, 64, packed.clone(), params.clone(), params),
            Err(QuantizedMatrixError::ColumnsNotGrouped)
        );
        let wrong = layout(DType::U32, [3072, 127]);
        let params = layout(DType::F16, [3072, 16]);
        assert_eq!(
            QuantizedMatrix::new(3072, 1024, 4, 64, wrong, params.clone(), params),
            Err(QuantizedMatrixError::ShapeMismatch {
                part: QuantizedMatrixPart::Packed
            })
        );
    }

    fn matrix(
        bits: u8,
        group: u32,
        parameter_dtype: DType,
    ) -> Result<QuantizedMatrix, QuantizedMatrixError> {
        let packed = layout(DType::U32, [3072, 1024 * u32::from(bits) / 32]);
        let params = layout(parameter_dtype, [3072, 1024 / group]);
        QuantizedMatrix::new(3072, 1024, bits, group, packed, params.clone(), params)
    }

    fn layout(dtype: DType, shape: [u32; 2]) -> Layout {
        let bytes = u64::from(shape[0]) * u64::from(shape[1]) * dtype.byte_size();
        Layout::contiguous(dtype, 0, shape.to_vec(), bytes).unwrap()
    }
}
