//! Shared deterministic and numeric helpers for backend tests.

use std::{error::Error, fmt};

use forja_core::{Backend, BackendError, DType, Op, Slice, Submission, Tensor, ViewOp};
use half::{bf16, f16};

/// The f32 normwise relative-error limit.
pub const F32_TOLERANCE: f64 = 1e-5;
/// The f16 normwise relative-error limit.
pub const F16_TOLERANCE: f64 = 2e-3;
/// The bf16 normwise relative-error limit.
pub const BF16_TOLERANCE: f64 = 1e-2;
/// The quantized normwise relative-error limit.
pub const QUANTIZED_TOLERANCE: f64 = 1e-2;

/// A tensor allocation and optional metadata-only view used by backend tests.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    dtype: DType,
    allocation_shape: Vec<u32>,
    view: Option<ViewOp>,
}

impl TensorSpec {
    /// Describes a contiguous tensor.
    #[must_use]
    pub fn contiguous(dtype: DType, shape: &[u32]) -> Self {
        Self {
            dtype,
            allocation_shape: shape.to_vec(),
            view: None,
        }
    }

    /// Describes a tensor viewed through an axis permutation.
    #[must_use]
    pub fn permuted(dtype: DType, allocation_shape: &[u32], axes: &[u8]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            view: Some(ViewOp::Permute(axes.to_vec())),
        }
    }

    /// Describes a tensor viewed through independent axis slices.
    #[must_use]
    pub fn sliced(dtype: DType, allocation_shape: &[u32], slices: &[Slice]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            view: Some(ViewOp::Slice(slices.to_vec())),
        }
    }

    /// Describes a tensor viewed through zero-stride broadcasting.
    #[must_use]
    pub fn broadcast(dtype: DType, allocation_shape: &[u32], shape: &[u32]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            view: Some(ViewOp::Broadcast(shape.to_vec())),
        }
    }
}

/// A deterministic xorshift64 value source.
#[derive(Clone, Debug)]
pub struct DeterministicValues {
    state: u64,
}

impl DeterministicValues {
    /// Creates a generator from a nonzero seed, substituting a fixed state for zero.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    /// Returns the next value in the range `[-1, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let [a, b, c, d, _, _, _, _] = self.state.to_le_bytes();
        let mantissa = u32::from_le_bytes([a, b, c, d]) >> 9;
        let unit = f32::from_bits(0x3f80_0000 | mantissa) - 1.0;
        unit.mul_add(2.0, -1.0)
    }
}

/// A failure while constructing or comparing a backend differential test.
#[derive(Clone, Debug, PartialEq)]
pub enum AgreementError {
    /// A backend rejected or failed the test work.
    Backend(BackendError),
    /// Integer tensors do not have a floating-point tolerance.
    UnsupportedDType(DType),
    /// A tensor size exceeded host addressable memory.
    SizeOverflow,
    /// Backend output bytes did not encode the declared dtype.
    InvalidOutput,
    /// Candidate output exceeded the dtype tolerance.
    OutsideTolerance {
        /// Measured normwise relative error.
        error: f64,
        /// Maximum accepted normwise relative error.
        tolerance: f64,
    },
    /// Candidate integer output differed from the reference bytes.
    OutputMismatch,
}

impl fmt::Display for AgreementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "backend agreement failed: {self:?}")
    }
}

impl Error for AgreementError {}

impl From<BackendError> for AgreementError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

/// Returns the accepted normwise relative error for an unquantized dtype.
///
/// # Errors
///
/// Returns [`AgreementError::UnsupportedDType`] for integer dtypes.
pub const fn dtype_tolerance(dtype: DType) -> Result<f64, AgreementError> {
    match dtype {
        DType::F32 => Ok(F32_TOLERANCE),
        DType::F16 => Ok(F16_TOLERANCE),
        DType::BF16 => Ok(BF16_TOLERANCE),
        DType::I32 | DType::U32 => Err(AgreementError::UnsupportedDType(dtype)),
    }
}

/// Computes `||candidate - reference|| / ||reference||` using f64 accumulation.
#[must_use]
pub fn normwise_relative_error(reference: &[f32], candidate: &[f32]) -> f64 {
    if reference.len() != candidate.len() {
        return f64::INFINITY;
    }
    let (difference, norm) = reference.iter().zip(candidate).fold(
        (0.0_f64, 0.0_f64),
        |(difference, norm), (&expected, &actual)| {
            let expected = f64::from(expected);
            let delta = f64::from(actual) - expected;
            (
                delta.mul_add(delta, difference),
                expected.mul_add(expected, norm),
            )
        },
    );
    if norm == 0.0 {
        if difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference / norm).sqrt()
    }
}

/// Runs identical initialized work on two backends and checks the output tolerance.
///
/// # Errors
///
/// Returns a backend, encoding, size, dtype, or tolerance error.
pub fn assert_backends_agree<R, C>(
    reference: &R,
    candidate: &C,
    op: Op,
    inputs: &[TensorSpec],
    output: &TensorSpec,
) -> Result<(), AgreementError>
where
    R: Backend,
    C: Backend,
{
    let mut values = DeterministicValues::new(0x6a09_e667_f3bc_c909);
    let mut reference_inputs = Vec::with_capacity(inputs.len());
    let mut candidate_inputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let bytes = generated_bytes(input, &mut values)?;
        reference_inputs.push(allocate_initialized(reference, input, &bytes)?);
        candidate_inputs.push(allocate_initialized(candidate, input, &bytes)?);
    }
    let reference_output = allocate(reference, output)?;
    let candidate_output = allocate(candidate, output)?;
    run(reference, op, &reference_inputs, &reference_output)?;
    run(candidate, op, &candidate_inputs, &candidate_output)?;
    let expected_bytes = reference.read(&reference_output)?;
    let actual_bytes = candidate.read(&candidate_output)?;
    if matches!(output.dtype, DType::I32 | DType::U32) {
        return (expected_bytes == actual_bytes)
            .then_some(())
            .ok_or(AgreementError::OutputMismatch);
    }
    let expected = decode(&expected_bytes, output.dtype)?;
    let actual = decode(&actual_bytes, output.dtype)?;
    let error = normwise_relative_error(&expected, &actual);
    let tolerance = dtype_tolerance(output.dtype)?;
    if error > tolerance {
        return Err(AgreementError::OutsideTolerance { error, tolerance });
    }
    Ok(())
}

fn run<B: Backend>(
    backend: &B,
    op: Op,
    inputs: &[Tensor],
    output: &Tensor,
) -> Result<(), AgreementError> {
    let mut commands = forja_core::CommandList::new();
    let inputs = inputs.iter().collect::<Vec<_>>();
    commands
        .dispatch(op, &inputs, output)
        .map_err(|_| BackendError::InvalidInput)?;
    backend.submit(commands)?.wait()?;
    Ok(())
}

fn allocate<B: Backend>(backend: &B, spec: &TensorSpec) -> Result<Tensor, AgreementError> {
    let tensor = backend.alloc(spec.dtype, &spec.allocation_shape)?;
    spec.view
        .clone()
        .map_or(Ok(tensor.clone()), |view| backend.view(&tensor, view))
        .map_err(Into::into)
}

fn allocate_initialized<B: Backend>(
    backend: &B,
    spec: &TensorSpec,
    bytes: &[u8],
) -> Result<Tensor, AgreementError> {
    let allocation = backend.alloc(spec.dtype, &spec.allocation_shape)?;
    backend.write(&allocation, bytes)?;
    spec.view
        .clone()
        .map_or(Ok(allocation.clone()), |view| {
            backend.view(&allocation, view)
        })
        .map_err(Into::into)
}

fn generated_bytes(
    spec: &TensorSpec,
    values: &mut DeterministicValues,
) -> Result<Vec<u8>, AgreementError> {
    let count = spec
        .allocation_shape
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
    let count = usize::try_from(count.ok_or(AgreementError::SizeOverflow)?)
        .map_err(|_| AgreementError::SizeOverflow)?;
    let width =
        usize::try_from(spec.dtype.byte_size()).map_err(|_| AgreementError::SizeOverflow)?;
    let capacity = count
        .checked_mul(width)
        .ok_or(AgreementError::SizeOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for _ in 0..count {
        let value = values.next_f32();
        match spec.dtype {
            DType::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
            DType::F16 => bytes.extend_from_slice(&f16::from_f32(value).to_le_bytes()),
            DType::BF16 => bytes.extend_from_slice(&bf16::from_f32(value).to_le_bytes()),
            DType::I32 => bytes.extend_from_slice(&value.to_bits().cast_signed().to_le_bytes()),
            DType::U32 => bytes.extend_from_slice(&value.to_bits().to_le_bytes()),
        }
    }
    Ok(bytes)
}

fn decode(bytes: &[u8], dtype: DType) -> Result<Vec<f32>, AgreementError> {
    let width = usize::try_from(dtype.byte_size()).map_err(|_| AgreementError::InvalidOutput)?;
    if !bytes.len().is_multiple_of(width) {
        return Err(AgreementError::InvalidOutput);
    }
    bytes
        .chunks_exact(width)
        .map(|bytes| match dtype {
            DType::F32 => bytes
                .try_into()
                .map(f32::from_le_bytes)
                .map_err(|_| AgreementError::InvalidOutput),
            DType::F16 => bytes
                .try_into()
                .map(f16::from_le_bytes)
                .map(f16::to_f32)
                .map_err(|_| AgreementError::InvalidOutput),
            DType::BF16 => bytes
                .try_into()
                .map(bf16::from_le_bytes)
                .map(bf16::to_f32)
                .map_err(|_| AgreementError::InvalidOutput),
            DType::I32 | DType::U32 => Err(AgreementError::UnsupportedDType(dtype)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_cpu::CpuBackend;

    #[test]
    fn generator_repeats_for_the_same_seed() {
        let mut first = DeterministicValues::new(7);
        let mut second = DeterministicValues::new(7);
        assert_eq!(first.next_f32().to_bits(), second.next_f32().to_bits());
        assert_eq!(first.next_f32().to_bits(), second.next_f32().to_bits());
    }

    #[test]
    fn normwise_error_handles_exact_zero_and_known_difference() {
        assert!(normwise_relative_error(&[0.0], &[0.0]).abs() < f64::EPSILON);
        assert!(normwise_relative_error(&[0.0], &[1.0]).is_infinite());
        assert!((normwise_relative_error(&[3.0, 4.0], &[0.0, 0.0]) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compares_non_contiguous_inputs_through_one_helper() {
        let reference = CpuBackend::new();
        let candidate = CpuBackend::new();
        let inputs = [
            TensorSpec::permuted(DType::F32, &[3072, 7], &[1, 0]),
            TensorSpec::contiguous(DType::F16, &[7, 3072]),
        ];
        let output = TensorSpec::contiguous(DType::BF16, &[7, 3072]);

        assert_backends_agree(&reference, &candidate, Op::SiluMul, &inputs, &output).unwrap();
    }
}
