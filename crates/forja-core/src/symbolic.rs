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
    /// An affine expression references a parameter absent from the value list.
    UnknownParameter {
        /// The referenced parameter index.
        index: u8,
    },
    /// An affine expression cannot be represented as a `u32`.
    ArithmeticOverflow,
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
            Self::UnknownParameter { index } => {
                write!(formatter, "parameter {index} is not declared")
            }
            Self::ArithmeticOverflow => formatter.write_str("affine arithmetic overflowed"),
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

/// A nonnegative affine expression over at most one parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Affine {
    offset: u32,
    term: Option<(u8, u32)>,
}

impl Affine {
    /// Creates a constant expression.
    #[must_use]
    pub const fn constant(value: u32) -> Self {
        Self {
            offset: value,
            term: None,
        }
    }

    /// Creates `offset + scale * parameter`.
    #[must_use]
    pub const fn parameter(parameter: u8, offset: u32, scale: u32) -> Self {
        Self {
            offset,
            term: Some((parameter, scale)),
        }
    }

    /// Returns whether this expression is independent of all parameters.
    #[must_use]
    pub const fn is_constant(self) -> bool {
        matches!(self.term, None | Some((_, 0)))
    }

    /// Evaluates the expression using checked `u64` arithmetic and narrows to `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamError`] when the parameter is absent or the result exceeds `u32`.
    pub fn evaluate(self, values: &ParamValues) -> Result<u32, ParamError> {
        let Some((parameter, scale)) = self.term else {
            return Ok(self.offset);
        };
        let value = values
            .as_slice()
            .get(usize::from(parameter))
            .ok_or(ParamError::UnknownParameter { index: parameter })?;
        u64::from(scale)
            .checked_mul(u64::from(*value))
            .and_then(|term| term.checked_add(u64::from(self.offset)))
            .and_then(|result| u32::try_from(result).ok())
            .ok_or(ParamError::ArithmeticOverflow)
    }
}

impl From<u32> for Affine {
    fn from(value: u32) -> Self {
        Self::constant(value)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::{Affine, MAX_PARAMS, ParamError, ParamSpace};

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

    #[test]
    fn evaluates_affine_expressions_with_checked_narrowing() {
        let range = RangeInclusive::new(7, u32::MAX);
        let space = ParamSpace::new(vec![range]).unwrap();
        let values = space.values(vec![7]).unwrap();
        assert_eq!(Affine::parameter(0, 1, 2).evaluate(&values), Ok(15));
        assert_eq!(Affine::constant(33).evaluate(&values), Ok(33));
        assert_eq!(Affine::parameter(0, 33, 0).evaluate(&values), Ok(33));
        assert_eq!(
            Affine::parameter(1, 0, 1).evaluate(&values),
            Err(ParamError::UnknownParameter { index: 1 })
        );
        assert_eq!(
            Affine::parameter(1, 33, 0).evaluate(&values),
            Err(ParamError::UnknownParameter { index: 1 })
        );
        let largest = space.values(vec![u32::MAX]).unwrap();
        assert_eq!(
            Affine::parameter(0, 1, 1).evaluate(&largest),
            Err(ParamError::ArithmeticOverflow)
        );
    }
}
