use std::{error::Error, fmt};

/// The accepted per-layer bf16 error against float32 transformer fixtures.
pub const BF16_HIDDEN_STATE_TOLERANCE: f64 = 5e-2;
/// The accepted teacher-forced bf16 mean KL divergence.
pub const BF16_LOGIT_KL_TOLERANCE: f64 = 1e-2;
/// The accepted mean KL divergence between end-to-end logit distributions.
pub const LOGIT_KL_TOLERANCE: f64 = 5e-3;

/// The result of comparing corresponding hidden states.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerComparison {
    errors: Vec<f64>,
    first_failing_layer: Option<usize>,
    tolerance: f64,
}

impl LayerComparison {
    /// Returns the normwise relative error for every hidden-state index.
    #[must_use]
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }

    /// Returns the first hidden-state index exceeding the tolerance.
    #[must_use]
    pub const fn first_failing_layer(&self) -> Option<usize> {
        self.first_failing_layer
    }

    /// Returns whether every hidden state is within tolerance.
    #[must_use]
    pub const fn passed(&self) -> bool {
        self.first_failing_layer.is_none()
    }
}

impl fmt::Display for LayerComparison {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.first_failing_layer {
            Some(layer) => write!(
                formatter,
                "hidden state {layer} first exceeded tolerance {}; per-layer errors: {:?}",
                self.tolerance, self.errors
            ),
            None => write!(
                formatter,
                "all hidden states are within tolerance {}; per-layer errors: {:?}",
                self.tolerance, self.errors
            ),
        }
    }
}

/// Compares corresponding hidden states with normwise relative error.
///
/// # Errors
///
/// Returns an error when the layer counts differ or a tolerance is invalid.
pub fn compare_hidden_states(
    reference: &[&[f32]],
    candidate: &[&[f32]],
    tolerance: f64,
) -> Result<LayerComparison, ComparisonError> {
    if reference.len() != candidate.len() {
        return Err(ComparisonError::ShapeMismatch);
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ComparisonError::InvalidTolerance);
    }
    let errors = reference
        .iter()
        .zip(candidate)
        .map(|(&reference, &candidate)| normwise_relative_error(reference, candidate))
        .collect::<Result<Vec<_>, _>>()?;
    let first_failing_layer = errors.iter().position(|&error| error > tolerance);
    Ok(LayerComparison {
        errors,
        first_failing_layer,
        tolerance,
    })
}

/// Computes `||candidate - reference|| / ||reference||` in float64.
///
/// # Errors
///
/// Returns an error for mismatched lengths or non-finite values.
pub fn normwise_relative_error(
    reference: &[f32],
    candidate: &[f32],
) -> Result<f64, ComparisonError> {
    if reference.len() != candidate.len() {
        return Err(ComparisonError::ShapeMismatch);
    }
    let mut difference = 0.0_f64;
    let mut norm = 0.0_f64;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        let expected = f64::from(expected);
        let actual = f64::from(actual);
        if !expected.is_finite() || !actual.is_finite() {
            return Err(ComparisonError::NonFinite);
        }
        let delta = actual - expected;
        difference = delta.mul_add(delta, difference);
        norm = expected.mul_add(expected, norm);
    }
    Ok(if norm == 0.0 {
        if difference == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (difference / norm).sqrt()
    })
}

/// Computes mean `KL(reference || candidate)` over rows of logits in float64.
///
/// # Errors
///
/// Returns an error for invalid row widths, mismatched shapes, or non-finite logits.
pub fn mean_logit_kl_divergence(
    reference: &[f32],
    candidate: &[f32],
    row_width: usize,
) -> Result<f64, ComparisonError> {
    if row_width == 0
        || reference.is_empty()
        || reference.len() != candidate.len()
        || !reference.len().is_multiple_of(row_width)
    {
        return Err(ComparisonError::ShapeMismatch);
    }
    let mut total = 0.0;
    for (reference, candidate) in reference
        .chunks_exact(row_width)
        .zip(candidate.chunks_exact(row_width))
    {
        total += row_kl(reference, candidate)?;
    }
    let rows = reference.len() / row_width;
    let rows = u32::try_from(rows).map_err(|_| ComparisonError::ShapeMismatch)?;
    Ok(total / f64::from(rows))
}

fn row_kl(reference: &[f32], candidate: &[f32]) -> Result<f64, ComparisonError> {
    let reference_log_partition = log_sum_exp(reference)?;
    let candidate_log_partition = log_sum_exp(candidate)?;
    reference
        .iter()
        .zip(candidate)
        .try_fold(0.0, |divergence, (&reference, &candidate)| {
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let log_probability = reference - reference_log_partition;
            Ok(divergence
                + log_probability.exp() * (log_probability - (candidate - candidate_log_partition)))
        })
}

fn log_sum_exp(values: &[f32]) -> Result<f64, ComparisonError> {
    let maximum = values
        .iter()
        .try_fold(f64::NEG_INFINITY, |maximum, &value| {
            let value = f64::from(value);
            value
                .is_finite()
                .then_some(maximum.max(value))
                .ok_or(ComparisonError::NonFinite)
        })?;
    Ok(maximum
        + values
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .sum::<f64>()
            .ln())
}

/// A numeric comparison failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonError {
    /// The inputs do not describe corresponding tensor shapes.
    ShapeMismatch,
    /// An input includes NaN or infinity.
    NonFinite,
    /// A tolerance is negative, NaN, or infinite.
    InvalidTolerance,
}

impl fmt::Display for ComparisonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "fixture comparison failed: {self:?}")
    }
}

impl Error for ComparisonError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normwise_error_matches_three_four_five_triangle() {
        let error = normwise_relative_error(&[3.0, 4.0], &[0.0, 0.0]).unwrap();
        assert!((error - 1.0).abs() < f64::EPSILON);
        assert!(normwise_relative_error(&[0.0], &[0.0]).unwrap().abs() < f64::EPSILON);
        assert!(
            normwise_relative_error(&[0.0], &[1.0])
                .unwrap()
                .is_infinite()
        );
    }

    #[test]
    fn mean_kl_matches_known_binary_distributions() {
        let reference = [1.098_612_3_f32, 0.0];
        let candidate = [0.0, 0.0];
        let actual = mean_logit_kl_divergence(&reference, &candidate, 2).unwrap();
        let expected = 0.75 * 1.5_f64.ln() + 0.25 * 0.5_f64.ln();
        assert!((actual - expected).abs() < 1e-7);
    }

    #[test]
    fn layer_report_identifies_the_first_failure() {
        let reference = [&[1.0_f32, 0.0][..], &[1.0_f32, 0.0][..]];
        let candidate = [&[1.01_f32, 0.0][..], &[1.1_f32, 0.0][..]];
        let report = compare_hidden_states(&reference, &candidate, 0.05).unwrap();
        assert_eq!(report.first_failing_layer(), Some(1));
        assert_eq!(report.errors().len(), 2);
        assert!(report.to_string().contains("per-layer errors"));
    }
}
