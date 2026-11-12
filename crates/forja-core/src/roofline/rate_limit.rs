use crate::BackendError;
use std::time::Duration;

/// A calibrated peak with an explicit fractional measurement margin.
#[derive(Clone, Copy, Debug)]
pub struct RateLimit {
    resource: &'static str,
    limit: f64,
}

impl RateLimit {
    /// Validates the peak and margin, refusing nonfinite or nonpositive limits.
    #[must_use]
    pub fn new(resource: &'static str, peak: f64, margin: f64) -> Option<Self> {
        let limit = peak * (1.0 + margin);
        (peak.is_finite() && peak > 0.0 && margin.is_finite() && margin >= 0.0 && limit.is_finite())
            .then_some(Self { resource, limit })
    }

    /// Refuses a completed observation exceeding the calibrated limit.
    ///
    /// # Errors
    /// Returns an impossible-rate error for positive work with zero duration, or excessive throughput.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn check(self, units: u64, elapsed: Duration) -> Result<(), BackendError> {
        if units == 0 {
            return Ok(());
        }
        let achieved = units as f64 / elapsed.as_secs_f64();
        if elapsed.is_zero() || !achieved.is_finite() || achieved > self.limit {
            return Err(BackendError::ImpossibleRate {
                resource: self.resource,
                achieved: if elapsed.is_zero() {
                    u64::MAX
                } else {
                    achieved as u64
                },
                limit: self.limit as u64,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_allows_margin_and_refuses_impossible_observations() {
        let limit = RateLimit::new("bytes", 100.0, 0.1).unwrap();
        for units in [0, 100, 110] {
            limit.check(units, Duration::from_secs(1)).unwrap();
        }
        let error = limit.check(111, Duration::from_secs(1)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("impossible benchmark record: bytes")
        );
        assert!(limit.check(0, Duration::ZERO).is_ok());
        assert!(limit.check(1, Duration::ZERO).is_err());
        assert!(limit.check(u64::MAX, Duration::from_nanos(1)).is_err());
        for (peak, margin) in [
            (0.0, 0.1),
            (f64::NAN, 0.1),
            (1.0, -0.1),
            (1.0, f64::INFINITY),
            (f64::MAX, 1.0),
        ] {
            assert!(RateLimit::new("FLOPs", peak, margin).is_none());
        }
    }
}
