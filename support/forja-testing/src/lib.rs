//! Shared deterministic and numeric helpers for backend tests.

pub mod program;
pub mod representative;

mod interval;

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

/// Prepares a validated program for the supplied concrete tensor bindings.
///
/// # Errors
///
/// Returns an error when the binding cannot form a valid kernel signature or preparation fails.
pub fn prepare_program<B: Backend>(
    backend: &B,
    program: &forja_core::program::ValidatedProgram,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
) -> Result<std::sync::Arc<forja_core::program::PreparedProgram>, forja_core::program::PrepareError>
{
    let rank = outputs
        .first()
        .or_else(|| inputs.first())
        .and_then(|tensor| u8::try_from(tensor.layout().shape().len()).ok())
        .ok_or(forja_core::program::PrepareError::Rank)?;
    let signature = forja_core::program::KernelSignature::new(
        rank,
        inputs
            .iter()
            .map(|tensor| tensor.layout().dtype())
            .collect(),
        outputs
            .iter()
            .map(|tensor| tensor.layout().dtype())
            .collect(),
        0,
    );
    forja_core::program::prepare_program(backend, program.clone(), signature)
}

/// A tensor allocation and metadata-only views used by backend tests.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    dtype: DType,
    allocation_shape: Vec<u32>,
    views: Vec<ViewOp>,
    initialized: Option<Vec<u8>>,
}

impl TensorSpec {
    /// Returns the allocation element type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the shape of the underlying allocation.
    #[must_use]
    pub fn allocation_shape(&self) -> &[u32] {
        &self.allocation_shape
    }

    /// Returns the metadata-only views applied in order.
    #[must_use]
    pub fn views(&self) -> &[ViewOp] {
        &self.views
    }

    /// Describes a contiguous tensor.
    #[must_use]
    pub fn contiguous(dtype: DType, shape: &[u32]) -> Self {
        Self {
            dtype,
            allocation_shape: shape.to_vec(),
            views: Vec::new(),
            initialized: None,
        }
    }

    /// Describes a tensor viewed through an axis permutation.
    #[must_use]
    pub fn permuted(dtype: DType, allocation_shape: &[u32], axes: &[u8]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            views: vec![ViewOp::Permute(axes.to_vec())],
            initialized: None,
        }
    }

    /// Describes a tensor viewed through independent axis slices.
    #[must_use]
    pub fn sliced(dtype: DType, allocation_shape: &[u32], slices: &[Slice]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            views: vec![ViewOp::Slice(slices.to_vec())],
            initialized: None,
        }
    }

    /// Describes a tensor viewed through a slice followed by an axis permutation.
    #[must_use]
    pub fn sliced_permuted(
        dtype: DType,
        allocation_shape: &[u32],
        slices: &[Slice],
        axes: &[u8],
    ) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            views: vec![
                ViewOp::Slice(slices.to_vec()),
                ViewOp::Permute(axes.to_vec()),
            ],
            initialized: None,
        }
    }

    /// Describes a tensor viewed through zero-stride broadcasting.
    #[must_use]
    pub fn broadcast(dtype: DType, allocation_shape: &[u32], shape: &[u32]) -> Self {
        Self {
            dtype,
            allocation_shape: allocation_shape.to_vec(),
            views: vec![ViewOp::Broadcast(shape.to_vec())],
            initialized: None,
        }
    }

    /// Describes a contiguous tensor with explicit allocation bytes.
    #[must_use]
    pub fn initialized(dtype: DType, shape: &[u32], bytes: Vec<u8>) -> Self {
        Self {
            dtype,
            allocation_shape: shape.to_vec(),
            views: Vec::new(),
            initialized: Some(bytes),
        }
    }
}

/// Generates deterministic encoded allocation bytes for a tensor specification.
///
/// # Errors
///
/// Returns an error when the allocation size exceeds addressable memory.
pub fn generated_tensor_bytes(
    spec: &TensorSpec,
    values: &mut DeterministicValues,
) -> Result<Vec<u8>, AgreementError> {
    generated_bytes(spec, values)
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
    /// Candidate output disagreed with the reference numeric classes.
    ClassMismatch {
        /// Number of positions with a NaN, infinity, or finite-class mismatch.
        count: usize,
    },
    /// Candidate integer output differed from the reference bytes.
    OutputMismatch,
    /// A generated-program output fell outside its conforming interval.
    OutsideInterval {
        /// Output element index within its tensor.
        index: usize,
        /// Value produced by the candidate backend.
        value: f32,
        /// Inclusive lower numeric bound.
        lo: f64,
        /// Inclusive upper numeric bound.
        hi: f64,
        /// Whether the interval admits NaN.
        may_nan: bool,
    },
    /// Generated-program intervals were too wide to judge the result.
    WideIntervals {
        /// Output dtype whose median was excessive.
        dtype: DType,
        /// Observed median relative width.
        median: f64,
        /// Maximum accepted median relative width.
        limit: f64,
    },
}

/// Interval-oracle activity observed while checking one generated program.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntervalReport {
    /// Ambiguous floating-point predicate evaluations.
    pub ambiguous_predicates: usize,
    /// Select evaluations whose condition admitted both branches.
    pub ambiguous_selects: usize,
    /// Rounded reduction steps included in the interval budget.
    pub reduction_steps: usize,
    /// Output elements whose numeric interval was vacuous.
    pub vacuous_elements: usize,
    /// Total output elements examined.
    pub total_elements: usize,
    /// Median relative output-interval width.
    pub median_relative_width: f64,
    /// Ninetieth-percentile relative output-interval width.
    pub p90_relative_width: f64,
    /// Largest relative output-interval width.
    pub max_relative_width: f64,
}

#[derive(Clone, Copy)]
struct WideOutput {
    dtype: DType,
    median: f64,
    limit: f64,
}

struct ProgramIntervals {
    input_bytes: Vec<Vec<u8>>,
    evaluation: interval::Evaluation,
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
    let (error, class_mismatches) = compare_float_values(reference, candidate);
    if class_mismatches == 0 {
        error
    } else {
        f64::INFINITY
    }
}

fn compare_float_values(reference: &[f32], candidate: &[f32]) -> (f64, usize) {
    let mut difference = 0.0_f64;
    let mut norm = 0.0_f64;
    let mut class_mismatches = 0;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        if expected.is_finite() && actual.is_finite() {
            let expected = f64::from(expected);
            let delta = f64::from(actual) - expected;
            difference = delta.mul_add(delta, difference);
            norm = expected.mul_add(expected, norm);
        } else if !nonfinite_values_agree(expected, actual) {
            class_mismatches += 1;
        }
    }
    let error = if norm == 0.0 {
        if difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference / norm).sqrt()
    };
    (error, class_mismatches)
}

/// Compares non-finite classes while admitting rounding at the f32 overflow edge.
///
/// Sequential CPU evaluation and parallel or fused GPU evaluation can round the
/// same mathematical result to a finite value or infinity on opposite sides of
/// the overflow boundary. A finite/infinite pair therefore agrees only when its
/// signs match and the finite magnitude is at least 2^126. NaN against any
/// non-NaN remains a mismatch.
fn nonfinite_values_agree(left: f32, right: f32) -> bool {
    const OVERFLOW_EDGE: f32 = f32::from_bits(0x7e80_0000);

    if left.is_nan() || right.is_nan() {
        return left.is_nan() && right.is_nan();
    }
    if left.is_infinite() && right.is_infinite() {
        return left.to_bits() == right.to_bits();
    }
    let (finite, infinite) = if left.is_finite() {
        (left, right)
    } else {
        (right, left)
    };
    infinite.is_infinite()
        && finite.abs() >= OVERFLOW_EDGE
        && finite.is_sign_negative() == infinite.is_sign_negative()
}

/// Checks f32 values with the shared normwise relative-error tolerance.
///
/// # Errors
///
/// Returns an error when lengths differ or the candidate exceeds [`F32_TOLERANCE`].
pub fn assert_f32_values_agree(reference: &[f32], candidate: &[f32]) -> Result<(), AgreementError> {
    if reference.len() != candidate.len() {
        return Err(AgreementError::OutputMismatch);
    }
    let (error, count) = compare_float_values(reference, candidate);
    if count != 0 {
        return Err(AgreementError::ClassMismatch { count });
    }
    if error > F32_TOLERANCE {
        return Err(AgreementError::OutsideTolerance {
            error,
            tolerance: F32_TOLERANCE,
        });
    }
    Ok(())
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
    assert_outputs_agree(output.dtype, &expected_bytes, &actual_bytes)
}

/// Runs a generated scalar program and checks each candidate output against an f64 interval.
///
/// The reference backend parameter is retained so differential-test call sites can share setup;
/// generated-program acceptance depends only on the conforming interval.
///
/// # Errors
///
/// Returns a validation, backend, encoding, size, dtype, interval, or meaningfulness error.
pub fn assert_program_backends_agree<R, C>(
    _reference: &R,
    candidate: &C,
    case: &program::ProgramCase,
) -> Result<IntervalReport, Box<dyn Error>>
where
    R: Backend,
    C: Backend,
{
    let program = case.program().validate()?;
    let ProgramIntervals {
        input_bytes,
        evaluation,
    } = evaluate_program_intervals(case)?;
    let (report, wide_output) = summarize_program_intervals(case, &evaluation)?;
    if let Some(wide) = wide_output {
        return Err(AgreementError::WideIntervals {
            dtype: wide.dtype,
            median: wide.median,
            limit: wide.limit,
        }
        .into());
    }

    let candidate_inputs = case
        .inputs()
        .iter()
        .zip(&input_bytes)
        .map(|(input, bytes)| allocate_initialized(candidate, input, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let candidate_outputs = case
        .outputs()
        .iter()
        .map(|output| allocate(candidate, output))
        .collect::<Result<Vec<_>, _>>()?;
    run_program(candidate, &program, &candidate_inputs, &candidate_outputs)?;

    for (output_slot, (spec, actual)) in case.outputs().iter().zip(candidate_outputs).enumerate() {
        let actual = decode(&candidate.read(&actual)?, spec.dtype())?;
        let intervals = evaluation
            .outputs
            .get(output_slot)
            .ok_or(AgreementError::InvalidOutput)?;
        if actual.len() != intervals.len() {
            return Err(AgreementError::OutputMismatch.into());
        }

        for (index, (&actual, interval)) in actual.iter().zip(intervals).enumerate() {
            let interval = interval.converted(spec.dtype())?;
            if !interval.contains(actual) {
                return Err(AgreementError::OutsideInterval {
                    index,
                    value: actual,
                    lo: interval.lo,
                    hi: interval.hi,
                    may_nan: interval.may_nan,
                }
                .into());
            }
        }
    }
    Ok(report)
}

/// Assesses a generated program's output intervals without running a backend.
///
/// # Errors
///
/// Returns a validation, encoding, size, or dtype error.
pub fn program_interval_report(
    case: &program::ProgramCase,
) -> Result<IntervalReport, Box<dyn Error>> {
    case.program().validate()?;
    let intervals = evaluate_program_intervals(case)?;
    let (report, _) = summarize_program_intervals(case, &intervals.evaluation)?;
    Ok(report)
}

fn evaluate_program_intervals(
    case: &program::ProgramCase,
) -> Result<ProgramIntervals, Box<dyn Error>> {
    let mut values = DeterministicValues::new(0xbb67_ae85_84ca_a73b);
    let mut input_bytes = Vec::with_capacity(case.inputs().len());
    for input in case.inputs() {
        input_bytes.push(generated_bytes(input, &mut values)?);
    }
    let evaluation = interval::evaluate(case, &input_bytes)?;
    Ok(ProgramIntervals {
        input_bytes,
        evaluation,
    })
}

fn summarize_program_intervals(
    case: &program::ProgramCase,
    evaluation: &interval::Evaluation,
) -> Result<(IntervalReport, Option<WideOutput>), AgreementError> {
    let mut report = IntervalReport {
        ambiguous_predicates: evaluation.ambiguous_predicates,
        ambiguous_selects: evaluation.ambiguous_selects,
        reduction_steps: evaluation.reduction_steps,
        ..IntervalReport::default()
    };
    let mut relative_widths = Vec::new();
    let mut wide_output = None;
    for (output_slot, spec) in case.outputs().iter().enumerate() {
        let limit = interval_width_limit(spec.dtype())?;
        let intervals = evaluation
            .outputs
            .get(output_slot)
            .ok_or(AgreementError::InvalidOutput)?;

        let mut output_widths = Vec::with_capacity(intervals.len());
        let mut output_vacuous = 0_usize;
        for &interval in intervals {
            let interval = interval.converted(spec.dtype())?;
            let width = interval.relative_width();
            output_widths.push(width);
            relative_widths.push(width);
            output_vacuous += usize::from(!interval.is_meaningful(limit));
            report.total_elements += 1;
        }
        report.vacuous_elements += output_vacuous;
        let median = percentile(&mut output_widths, 50);
        if output_vacuous > intervals.len() / 2 && wide_output.is_none() {
            wide_output = Some(WideOutput {
                dtype: spec.dtype(),
                median,
                limit,
            });
        }
    }

    report.median_relative_width = percentile(&mut relative_widths, 50);
    report.p90_relative_width = percentile(&mut relative_widths, 90);
    report.max_relative_width = relative_widths
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    Ok((report, wide_output))
}

fn interval_width_limit(dtype: DType) -> Result<f64, AgreementError> {
    match dtype {
        DType::F32 => Ok(6.0e-3),
        DType::F16 => Ok(2.0e-2),
        DType::BF16 => Ok(1.0e-1),
        DType::I32 | DType::U32 => Err(AgreementError::UnsupportedDType(dtype)),
    }
}

fn percentile(values: &mut [f64], percentage: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let index = values.len().saturating_sub(1) * percentage / 100;
    values[index]
}

/// Checks two encoded outputs using exact integer comparison or the dtype tolerance.
///
/// # Errors
///
/// Returns an encoding or tolerance error when the candidate differs from the reference.
pub fn assert_outputs_agree(
    dtype: DType,
    expected_bytes: &[u8],
    actual_bytes: &[u8],
) -> Result<(), AgreementError> {
    if matches!(dtype, DType::I32 | DType::U32) {
        return (expected_bytes == actual_bytes)
            .then_some(())
            .ok_or(AgreementError::OutputMismatch);
    }
    let expected = decode(expected_bytes, dtype)?;
    let actual = decode(actual_bytes, dtype)?;
    if expected.len() != actual.len() {
        return Err(AgreementError::OutputMismatch);
    }
    let (error, count) = compare_float_values(&expected, &actual);
    if count != 0 {
        return Err(AgreementError::ClassMismatch { count });
    }
    let tolerance = dtype_tolerance(dtype)?;
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

fn run_program<B: Backend>(
    backend: &B,
    program: &forja_core::program::ValidatedProgram,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<(), BackendError> {
    let input_refs = inputs.iter().collect::<Vec<_>>();
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let prepared = prepare_program(backend, program, &input_refs, &output_refs)
        .map_err(|_| BackendError::InvalidInput)?;
    let mut commands = forja_core::CommandList::new();
    commands
        .dispatch_kernel(&prepared, &input_refs, &output_refs)
        .map_err(|_| BackendError::InvalidInput)?;
    backend.submit(commands)?.wait()
}

fn allocate<B: Backend>(backend: &B, spec: &TensorSpec) -> Result<Tensor, AgreementError> {
    let tensor = backend.alloc(spec.dtype, &spec.allocation_shape)?;
    spec.views
        .iter()
        .cloned()
        .try_fold(tensor, |tensor, view| backend.view(&tensor, view))
        .map_err(Into::into)
}

fn allocate_initialized<B: Backend>(
    backend: &B,
    spec: &TensorSpec,
    bytes: &[u8],
) -> Result<Tensor, AgreementError> {
    let allocation = backend.alloc(spec.dtype, &spec.allocation_shape)?;
    backend.write(&allocation, bytes)?;
    spec.views
        .iter()
        .cloned()
        .try_fold(allocation, |tensor, view| backend.view(&tensor, view))
        .map_err(Into::into)
}

fn generated_bytes(
    spec: &TensorSpec,
    values: &mut DeterministicValues,
) -> Result<Vec<u8>, AgreementError> {
    if let Some(bytes) = &spec.initialized {
        return Ok(bytes.clone());
    }
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
    fn interval_width_limits_cover_observed_output_medians() {
        assert_eq!(interval_width_limit(DType::F32), Ok(6.0e-3));
        assert_eq!(interval_width_limit(DType::F16), Ok(2.0e-2));
        assert_eq!(interval_width_limit(DType::BF16), Ok(1.0e-1));
    }

    #[test]
    fn zero_scale_width_keeps_exact_zero_intervals_meaningful() {
        let subnormal_ulp = f64::from(f32::from_bits(1));
        let interval = interval::FloatInterval {
            lo: -subnormal_ulp,
            hi: subnormal_ulp,
            may_nan: false,
        };

        assert!(interval.relative_width().is_infinite());
        assert!(interval.is_meaningful(interval_width_limit(DType::F32).unwrap()));
    }

    #[test]
    fn zero_scale_width_does_not_weaken_nonzero_relative_limit() {
        let interval = interval::FloatInterval {
            lo: 1.0,
            hi: 1.01,
            may_nan: false,
        };

        assert!(!interval.is_meaningful(interval_width_limit(DType::F32).unwrap()));
    }

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
    fn normwise_error_matches_nonfinite_values_by_kind() {
        assert!(
            normwise_relative_error(
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            )
            .abs()
                < f64::EPSILON
        );
        assert!(normwise_relative_error(&[0.0], &[f32::NAN]).is_infinite());
        assert!(normwise_relative_error(&[f32::INFINITY], &[f32::NEG_INFINITY]).is_infinite());
    }

    #[test]
    fn overflow_edge_accepts_only_matching_nearby_finite_values() {
        let edge = f32::from_bits(0x7e80_0000);
        let below_edge = f32::from_bits(edge.to_bits() - 1);

        for (reference, candidate) in [
            (edge, f32::INFINITY),
            (f32::INFINITY, edge),
            (-edge, f32::NEG_INFINITY),
            (f32::NEG_INFINITY, -edge),
        ] {
            assert_f32_values_agree(&[reference], &[candidate]).unwrap();
        }
        for (reference, candidate) in [
            (below_edge, f32::INFINITY),
            (-below_edge, f32::NEG_INFINITY),
            (edge, f32::NEG_INFINITY),
            (-edge, f32::INFINITY),
            (f32::NAN, f32::INFINITY),
            (f32::NAN, edge),
        ] {
            assert_eq!(
                assert_f32_values_agree(&[reference], &[candidate]),
                Err(AgreementError::ClassMismatch { count: 1 })
            );
        }
    }

    #[test]
    fn reports_class_mismatches_separately_from_finite_error() {
        let reference = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.0, 4.0];
        let candidate = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, 0.0];
        assert_eq!(
            assert_f32_values_agree(&reference, &candidate),
            Err(AgreementError::ClassMismatch { count: 2 })
        );
        assert!((compare_float_values(&reference, &candidate).0 - 1.0).abs() < f64::EPSILON);
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
