use std::{
    error::Error,
    fmt,
    ops::RangeInclusive,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PARAM_SPACE_ID: AtomicU64 = AtomicU64::new(0);

/// The largest number of parameters accepted by a symbolic layout.
pub const MAX_PARAMS: usize = 4;

/// A reason that symbolic parameter construction or evaluation failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParamError {
    /// A parameter space contains more than [`MAX_PARAMS`] ranges.
    TooManyParameters,
    /// A parameter range has its lower bound above its upper bound.
    InvalidRange {
        /// The index of the invalid range.
        index: usize,
    },
    /// A value list has the wrong number of entries.
    CountMismatch {
        /// The number of values required by the space.
        expected: usize,
        /// The number of values supplied.
        actual: usize,
    },
    /// A parameter value falls outside its declared range.
    OutOfRange {
        /// The index of the invalid value.
        index: usize,
        /// The value that was supplied.
        value: u32,
    },
    /// No unused process-unique identity remains.
    IdentityExhausted,
}

impl fmt::Display for ParamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyParameters => formatter.write_str("parameter count exceeds four"),
            Self::InvalidRange { index } => {
                write!(formatter, "parameter {index} has an empty range")
            }
            Self::CountMismatch { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} parameter values, got {actual}"
                )
            }
            Self::OutOfRange { index, value } => {
                write!(
                    formatter,
                    "value {value} is outside parameter {index}'s range"
                )
            }
            Self::IdentityExhausted => formatter.write_str("parameter space identities exhausted"),
        }
    }
}

impl Error for ParamError {}

/// A checked box of unsigned parameter ranges.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamSpace {
    id: u64,
    ranges: Vec<RangeInclusive<u32>>,
}

impl ParamSpace {
    /// Creates a parameter space containing at most four nonempty ranges.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when there are too many ranges or any range is empty.
    pub fn new(ranges: Vec<RangeInclusive<u32>>) -> Result<Self, ParamError> {
        if ranges.len() > MAX_PARAMS {
            return Err(ParamError::TooManyParameters);
        }
        if let Some(index) = ranges.iter().position(RangeInclusive::is_empty) {
            return Err(ParamError::InvalidRange { index });
        }
        let id = NEXT_PARAM_SPACE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| ParamError::IdentityExhausted)?;
        Ok(Self { id, ranges })
    }

    /// Returns the declared ranges in parameter order.
    #[must_use]
    pub fn ranges(&self) -> &[RangeInclusive<u32>] {
        &self.ranges
    }

    /// Checks concrete values against this parameter space.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when the count differs or a value is out of range.
    pub fn values(&self, values: Vec<u32>) -> Result<ParamValues, ParamError> {
        if values.len() != self.ranges.len() {
            return Err(ParamError::CountMismatch {
                expected: self.ranges.len(),
                actual: values.len(),
            });
        }
        if let Some((index, &value)) = values
            .iter()
            .enumerate()
            .find(|(index, value)| !self.ranges[*index].contains(value))
        {
            return Err(ParamError::OutOfRange { index, value });
        }
        Ok(ParamValues {
            space_id: self.id,
            values,
        })
    }
}

/// Concrete parameter values checked against their originating space.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamValues {
    space_id: u64,
    values: Vec<u32>,
}

impl ParamValues {
    /// Returns the checked values in parameter order.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::{MAX_PARAMS, ParamError, ParamSpace};

    #[test]
    fn checks_parameter_spaces_and_values() {
        let space = ParamSpace::new(vec![1..=7, 33..=4097]).unwrap();
        assert_eq!(space.values(vec![7, 33]).unwrap().as_slice(), [7, 33]);
        assert_eq!(
            space.values(vec![0, 33]),
            Err(ParamError::OutOfRange { index: 0, value: 0 })
        );
        assert!(matches!(
            space.values(vec![1]),
            Err(ParamError::CountMismatch { .. })
        ));

        let cloned = space.clone();
        let equal_ranges = ParamSpace::new(vec![1..=7, 33..=4097]).unwrap();
        assert_eq!(space.values(vec![1, 33]), cloned.values(vec![1, 33]));
        assert_ne!(space.values(vec![1, 33]), equal_ranges.values(vec![1, 33]));
    }

    #[test]
    fn refuses_invalid_parameter_spaces() {
        let invalid = RangeInclusive::new(2, 1);
        assert_eq!(
            ParamSpace::new(vec![invalid]),
            Err(ParamError::InvalidRange { index: 0 })
        );
        assert_eq!(
            ParamSpace::new(vec![0..=0; MAX_PARAMS + 1]),
            Err(ParamError::TooManyParameters)
        );
    }
}
