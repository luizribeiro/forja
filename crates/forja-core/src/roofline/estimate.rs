/// A positive measurement with an ordered uncertainty interval in the same units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    median: f64,
    interval: [f64; 2],
}

impl Estimate {
    /// Validates finite positive values and an interval containing the median.
    #[must_use]
    pub fn new(median: f64, interval: [f64; 2]) -> Option<Self> {
        [median, interval[0], interval[1]]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            .then_some(())?;
        (interval[0] <= median && median <= interval[1]).then_some(Self { median, interval })
    }

    /// Returns the measured median.
    #[must_use]
    pub const fn median(self) -> f64 {
        self.median
    }

    /// Returns the lower end of the measurement interval.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.interval[0]
    }

    /// Returns the upper end of the measurement interval.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.interval[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_require_positive_ordered_finite_intervals() {
        assert!(Estimate::new(2.0, [1.0, 3.0]).is_some());
        for (median, interval) in [
            (2.0, [3.0, 4.0]),
            (2.0, [1.0, 1.5]),
            (0.0, [0.0, 1.0]),
            (2.0, [1.0, f64::INFINITY]),
            (f64::NAN, [1.0, 3.0]),
        ] {
            assert!(Estimate::new(median, interval).is_none());
        }
    }
}
