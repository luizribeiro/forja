use std::{
    error::Error,
    fmt,
    ops::{Range, RangeInclusive},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PARAM_SPACE_ID: AtomicU64 = AtomicU64::new(0);

use crate::{Layout, LayoutError, Slice};

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

    pub(crate) const fn identity(&self) -> u64 {
        self.id
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

    pub(crate) fn corners(&self) -> Vec<ParamValues> {
        let count = 1_usize << self.ranges.len();
        (0..count)
            .map(|corner| ParamValues {
                space_id: self.id,
                values: self
                    .ranges
                    .iter()
                    .enumerate()
                    .map(|(index, range)| {
                        if corner & (1 << index) == 0 {
                            *range.start()
                        } else {
                            *range.end()
                        }
                    })
                    .collect(),
            })
            .collect()
    }

    fn endpoint_values(&self, upper: bool) -> ParamValues {
        ParamValues {
            space_id: self.id,
            values: self
                .ranges
                .iter()
                .map(|range| if upper { *range.end() } else { *range.start() })
                .collect(),
        }
    }

    pub(crate) fn contains_values(&self, values: &ParamValues) -> bool {
        self.id == values.space_id
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

    pub(crate) const fn parts(self) -> (u32, Option<(u8, u32)>) {
        (self.offset, self.term)
    }
}

impl From<u32> for Affine {
    fn from(value: u32) -> Self {
        Self::constant(value)
    }
}

/// A reason that symbolic layout construction or instantiation failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolicLayoutError {
    /// Parameter evaluation failed.
    Parameter(ParamError),
    /// Concrete layout validation failed.
    Layout(LayoutError),
    /// Values were checked against a different parameter space.
    ParameterSpaceMismatch,
    /// A slice references an axis outside the current rank.
    AxisOutOfRange {
        /// The invalid axis.
        axis: u8,
    },
    /// A constant slice targeted an axis whose extent is symbolic.
    ConstantSliceOnSymbolicAxis {
        /// The axis that cannot accept the slice.
        axis: u8,
    },
    /// A shape-changing operation requires concrete extents.
    SymbolicExtentTransform,
}

impl fmt::Display for SymbolicLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parameter(error) => error.fmt(formatter),
            Self::Layout(error) => error.fmt(formatter),
            Self::ParameterSpaceMismatch => {
                formatter.write_str("values belong to a different parameter space")
            }
            Self::AxisOutOfRange { axis } => write!(formatter, "axis {axis} is out of range"),
            Self::ConstantSliceOnSymbolicAxis { axis } => {
                write!(formatter, "axis {axis} has a symbolic extent")
            }
            Self::SymbolicExtentTransform => {
                formatter.write_str("operation requires concrete extents")
            }
        }
    }
}

impl Error for SymbolicLayoutError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parameter(error) => Some(error),
            Self::Layout(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ParamError> for SymbolicLayoutError {
    fn from(error: ParamError) -> Self {
        Self::Parameter(error)
    }
}

impl From<LayoutError> for SymbolicLayoutError {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecipeOp {
    Slice {
        axis: u8,
        start: Affine,
        len: Affine,
        step: u32,
    },
    Permute(Vec<u8>),
    Broadcast(Vec<u32>),
    Reshape(Vec<u32>),
}

/// A concrete base layout and a checked recipe for parameterized views.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicLayout {
    base: Layout,
    space: ParamSpace,
    recipe: Vec<RecipeOp>,
    symbolic_extents: Vec<bool>,
}

impl SymbolicLayout {
    /// Starts an empty recipe from a validated concrete layout.
    #[must_use]
    pub fn new(base: Layout, space: ParamSpace) -> Self {
        let symbolic_extents = vec![false; base.shape().len()];
        Self {
            base,
            space,
            recipe: Vec::new(),
            symbolic_extents,
        }
    }

    /// Returns the concrete layout from which the recipe starts.
    #[must_use]
    pub const fn base(&self) -> &Layout {
        &self.base
    }

    /// Returns the parameter space accepted by this layout.
    #[must_use]
    pub const fn space(&self) -> &ParamSpace {
        &self.space
    }

    /// Appends a slice along one axis and validates every parameter-space corner.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] for an invalid composition or corner layout.
    pub fn slice(
        &self,
        axis: u8,
        start: Affine,
        len: Affine,
        step: u32,
    ) -> Result<Self, SymbolicLayoutError> {
        if usize::from(axis) >= self.symbolic_extents.len() {
            return Err(SymbolicLayoutError::AxisOutOfRange { axis });
        }
        if start.is_constant()
            && len.is_constant()
            && self.symbolic_extents.get(usize::from(axis)) == Some(&true)
        {
            return Err(SymbolicLayoutError::ConstantSliceOnSymbolicAxis { axis });
        }
        let mut result = self.clone();
        result.recipe.push(RecipeOp::Slice {
            axis,
            start,
            len,
            step,
        });
        result.validate_corners()?;
        result.symbolic_extents[usize::from(axis)] = !len.is_constant();
        Ok(result)
    }

    /// Appends an axis permutation and validates every parameter-space corner.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] when the permutation or a corner is invalid.
    pub fn permute(&self, axes: &[u8]) -> Result<Self, SymbolicLayoutError> {
        let mut result = self.clone();
        result.recipe.push(RecipeOp::Permute(axes.to_owned()));
        result.validate_corners()?;
        result.symbolic_extents = axes
            .iter()
            .map(|&axis| self.symbolic_extents[usize::from(axis)])
            .collect();
        Ok(result)
    }

    /// Appends a broadcast when all extents are concrete.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] for symbolic extents or invalid corners.
    pub fn broadcast(&self, shape: Vec<u32>) -> Result<Self, SymbolicLayoutError> {
        self.with_concrete_extent_op(RecipeOp::Broadcast(shape))
    }

    /// Appends a reshape when all extents are concrete.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] for symbolic extents or invalid corners.
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Self, SymbolicLayoutError> {
        self.with_concrete_extent_op(RecipeOp::Reshape(shape))
    }

    /// Instantiates and revalidates the complete recipe with trusted [`Layout`] operations.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] when values belong to another space, affine
    /// evaluation fails, or any concrete layout operation refuses the result.
    pub fn instantiate(&self, values: &ParamValues) -> Result<Layout, SymbolicLayoutError> {
        if !self.space.contains_values(values) {
            return Err(SymbolicLayoutError::ParameterSpaceMismatch);
        }
        self.recipe
            .iter()
            .try_fold(self.base.clone(), |layout, op| {
                Self::apply_op(&layout, op, values)
            })
    }

    /// Instantiates every corner for early refusal, not as a proof of replay safety.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] when any corner cannot be instantiated.
    pub fn corner_layouts(&self) -> Result<Vec<Layout>, SymbolicLayoutError> {
        self.space
            .corners()
            .iter()
            .map(|values| self.instantiate(values))
            .collect()
    }

    /// Returns a conservative byte-span envelope over the entire parameter box.
    ///
    /// Empty layouts contribute no bytes. The recipe grammar makes offsets and
    /// nonempty span ends monotone, but this method retains the base start when
    /// the lower corner is empty. Corner validation alone is not a safety proof.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicLayoutError`] if an endpoint cannot be instantiated.
    pub fn byte_hull(&self) -> Result<ByteHull, SymbolicLayoutError> {
        let upper = self.instantiate(&self.space.endpoint_values(true))?;
        if upper.element_count() == 0 {
            return Ok(ByteHull::empty());
        }
        let lower = self.instantiate(&self.space.endpoint_values(false))?;
        let start = if lower.element_count() == 0 {
            self.base.byte_span().start
        } else {
            lower.byte_span().start
        };
        Ok(ByteHull {
            span: Some(start..upper.byte_span().end),
        })
    }

    fn with_concrete_extent_op(&self, op: RecipeOp) -> Result<Self, SymbolicLayoutError> {
        if self.symbolic_extents.contains(&true) {
            return Err(SymbolicLayoutError::SymbolicExtentTransform);
        }
        let rank = match &op {
            RecipeOp::Broadcast(shape) | RecipeOp::Reshape(shape) => shape.len(),
            _ => self.symbolic_extents.len(),
        };
        let mut result = self.clone();
        result.recipe.push(op);
        result.validate_corners()?;
        result.symbolic_extents = vec![false; rank];
        Ok(result)
    }

    fn validate_corners(&self) -> Result<(), SymbolicLayoutError> {
        self.corner_layouts().map(|_| ())
    }

    fn apply_op(
        layout: &Layout,
        op: &RecipeOp,
        values: &ParamValues,
    ) -> Result<Layout, SymbolicLayoutError> {
        match op {
            RecipeOp::Slice {
                axis,
                start,
                len,
                step,
            } => {
                let mut slices = layout
                    .shape()
                    .iter()
                    .map(|&extent| Slice::new(0, extent, 1))
                    .collect::<Result<Vec<_>, _>>()?;
                let selected = slices
                    .get_mut(usize::from(*axis))
                    .ok_or(SymbolicLayoutError::AxisOutOfRange { axis: *axis })?;
                *selected = Slice::new(start.evaluate(values)?, len.evaluate(values)?, *step)?;
                Ok(layout.slice(&slices)?)
            }
            RecipeOp::Permute(axes) => Ok(layout.permute(axes)?),
            RecipeOp::Broadcast(shape) => Ok(layout.broadcast(shape)?),
            RecipeOp::Reshape(shape) => Ok(layout.reshape(shape)?),
        }
    }

    pub(crate) fn constraint_facts(&self) -> Result<SymbolicLayoutFacts, SymbolicLayoutError> {
        let mut dimensions = self
            .base
            .shape()
            .iter()
            .copied()
            .map(Affine::constant)
            .collect::<Vec<_>>();
        let mut strides = self.base.strides().to_vec();
        let mut offset = LinearAffine::constant(self.base.offset());
        let lower = self.space.endpoint_values(false);
        let mut representative = self.base.clone();
        for operation in &self.recipe {
            representative = Self::apply_op(&representative, operation, &lower)?;
            match operation {
                RecipeOp::Slice {
                    axis,
                    start,
                    len,
                    step,
                } => {
                    let index = usize::from(*axis);
                    let stride = *strides
                        .get(index)
                        .ok_or(SymbolicLayoutError::AxisOutOfRange { axis: *axis })?;
                    offset = offset.checked_add_scaled(*start, stride)?;
                    dimensions[index] = *len;
                    strides[index] = stride
                        .checked_mul(u64::from(*step))
                        .ok_or(ParamError::ArithmeticOverflow)?;
                }
                RecipeOp::Permute(axes) => {
                    dimensions = axes
                        .iter()
                        .map(|&axis| dimensions[usize::from(axis)])
                        .collect();
                    strides = axes
                        .iter()
                        .map(|&axis| strides[usize::from(axis)])
                        .collect();
                }
                RecipeOp::Broadcast(shape) | RecipeOp::Reshape(shape) => {
                    dimensions = shape.iter().copied().map(Affine::constant).collect();
                    strides = representative.strides().to_vec();
                }
            }
        }
        Ok(SymbolicLayoutFacts {
            dtype: self.base.dtype(),
            dimensions,
            strides,
            offset,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SymbolicLayoutFacts {
    pub(crate) dtype: crate::DType,
    pub(crate) dimensions: Vec<Affine>,
    pub(crate) strides: Vec<u64>,
    pub(crate) offset: LinearAffine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LinearAffine {
    pub(crate) offset: u64,
    pub(crate) coefficients: [u64; MAX_PARAMS],
}

impl LinearAffine {
    const fn constant(value: u64) -> Self {
        Self {
            offset: value,
            coefficients: [0; MAX_PARAMS],
        }
    }

    fn checked_add_scaled(mut self, affine: Affine, scale: u64) -> Result<Self, ParamError> {
        let (constant, term) = affine.parts();
        self.offset = u64::from(constant)
            .checked_mul(scale)
            .and_then(|value| self.offset.checked_add(value))
            .ok_or(ParamError::ArithmeticOverflow)?;
        if let Some((parameter, coefficient)) = term {
            let slot = self
                .coefficients
                .get_mut(usize::from(parameter))
                .ok_or(ParamError::UnknownParameter { index: parameter })?;
            *slot = u64::from(coefficient)
                .checked_mul(scale)
                .and_then(|value| slot.checked_add(value))
                .ok_or(ParamError::ArithmeticOverflow)?;
        }
        Ok(self)
    }
}

/// A conservative byte-span envelope used for overlap analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteHull {
    span: Option<Range<u64>>,
}

impl ByteHull {
    /// Creates a hull containing no bytes.
    #[must_use]
    pub const fn empty() -> Self {
        Self { span: None }
    }

    /// Encloses the nonempty layouts, skipping empty layouts.
    #[must_use]
    pub fn from_layouts<'a>(layouts: impl IntoIterator<Item = &'a Layout>) -> Self {
        let span = layouts
            .into_iter()
            .filter(|layout| layout.element_count() != 0)
            .map(Layout::byte_span)
            .reduce(|left, right| left.start.min(right.start)..left.end.max(right.end));
        Self { span }
    }

    /// Returns the hull's byte range, or `None` when every layout was empty.
    #[must_use]
    pub fn byte_span(&self) -> Option<Range<u64>> {
        self.span.clone()
    }

    /// Returns whether this hull contains every byte touched by `layout`.
    #[must_use]
    pub fn contains(&self, layout: &Layout) -> bool {
        if layout.element_count() == 0 {
            return true;
        }
        let layout_span = layout.byte_span();
        self.span
            .as_ref()
            .is_some_and(|span| span.start <= layout_span.start && span.end >= layout_span.end)
    }

    /// Conservatively reports whether this hull and a concrete layout may overlap.
    #[must_use]
    pub fn overlaps(&self, layout: &Layout) -> bool {
        if layout.element_count() == 0 {
            return false;
        }
        let layout_span = layout.byte_span();
        self.span
            .as_ref()
            .is_some_and(|span| span.start < layout_span.end && layout_span.start < span.end)
    }

    /// Conservatively reports whether two hulls of the same buffer overlap.
    #[must_use]
    pub fn overlaps_hull(&self, other: &Self) -> bool {
        self.span.as_ref().is_some_and(|left| {
            other
                .span
                .as_ref()
                .is_some_and(|right| left.start < right.end && right.start < left.end)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use proptest::prelude::*;

    use super::{
        Affine, ByteHull, MAX_PARAMS, ParamError, ParamSpace, SymbolicLayout, SymbolicLayoutError,
    };
    use crate::{DType, Layout, LayoutError};

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

    #[test]
    fn instantiates_qwen_cache_views_through_layout_validation() {
        let range = RangeInclusive::new(0, 4095);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F16, 0, vec![8, 4096, 128], 8_388_608).unwrap();
        let appended = SymbolicLayout::new(base.clone(), space.clone())
            .slice(1, Affine::parameter(0, 0, 1), 1.into(), 1)
            .unwrap();
        let attended = SymbolicLayout::new(base, space.clone())
            .slice(1, 0.into(), Affine::parameter(0, 1, 1), 1)
            .unwrap();
        let values = space.values(vec![33]).unwrap();

        assert_eq!(appended.instantiate(&values).unwrap().shape(), [8, 1, 128]);
        assert_eq!(appended.instantiate(&values).unwrap().offset(), 4224);
        assert_eq!(attended.instantiate(&values).unwrap().shape(), [8, 34, 128]);
    }

    #[test]
    fn parameter_spaces_are_not_interchangeable() {
        let first = ParamSpace::new(vec![0..=3, 7..=7]).unwrap();
        let equal_ranges = ParamSpace::new(vec![0..=3, 7..=7]).unwrap();
        let different_ranges = ParamSpace::new(vec![1..=4, 7..=7]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![4], 16).unwrap();
        let layout = SymbolicLayout::new(base, first.clone());

        assert!(
            layout
                .instantiate(&first.values(vec![2, 7]).unwrap())
                .is_ok()
        );
        assert_eq!(
            layout.instantiate(&equal_ranges.values(vec![2, 7]).unwrap()),
            Err(SymbolicLayoutError::ParameterSpaceMismatch)
        );
        assert_eq!(
            layout.instantiate(&different_ranges.values(vec![2, 7]).unwrap()),
            Err(SymbolicLayoutError::ParameterSpaceMismatch)
        );
    }

    #[test]
    fn enforces_symbolic_composition_rules() {
        let range = RangeInclusive::new(0, 3);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![3, 3], 36).unwrap();
        let variable = SymbolicLayout::new(base, space)
            .slice(1, 0.into(), Affine::parameter(0, 0, 1), 1)
            .unwrap();

        assert!(matches!(
            variable.slice(1, 0.into(), 1.into(), 1),
            Err(SymbolicLayoutError::ConstantSliceOnSymbolicAxis { axis: 1 })
        ));
        assert_eq!(
            variable.reshape(vec![9]),
            Err(SymbolicLayoutError::SymbolicExtentTransform)
        );
        assert_eq!(
            variable.broadcast(vec![3, 3]),
            Err(SymbolicLayoutError::SymbolicExtentTransform)
        );
        let permuted = variable.permute(&[1, 0]).unwrap();
        assert!(matches!(
            permuted.slice(0, 0.into(), 1.into(), 1),
            Err(SymbolicLayoutError::ConstantSliceOnSymbolicAxis { axis: 0 })
        ));
    }

    #[test]
    fn allows_shape_changes_with_only_a_symbolic_offset() {
        let range = RangeInclusive::new(0, 6);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![7], 28).unwrap();
        let layout = SymbolicLayout::new(base, space.clone())
            .slice(0, Affine::parameter(0, 0, 1), 1.into(), 1)
            .unwrap()
            .reshape(vec![1, 1])
            .unwrap()
            .broadcast(vec![7, 1])
            .unwrap();

        assert_eq!(
            layout
                .instantiate(&space.values(vec![6]).unwrap())
                .unwrap()
                .byte_span(),
            24..28
        );
    }

    #[test]
    fn corners_refuse_invalid_recipes_early() {
        let range = RangeInclusive::new(0, 7);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![7], 28).unwrap();
        let result =
            SymbolicLayout::new(base, space).slice(0, Affine::parameter(0, 0, 1), 1.into(), 1);

        assert!(matches!(
            result,
            Err(SymbolicLayoutError::Layout(LayoutError::SliceOutOfBounds {
                axis: 0
            }))
        ));
    }

    #[test]
    fn enumerates_at_most_sixteen_corners() {
        let space = ParamSpace::new(vec![0..=1, 2..=3, 4..=5, 6..=7]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![1], 4).unwrap();
        assert_eq!(
            SymbolicLayout::new(base, space)
                .corner_layouts()
                .unwrap()
                .len(),
            16
        );
    }

    #[test]
    fn hulls_skip_empty_layouts_and_detect_overlap() {
        let empty = Layout::contiguous(DType::F32, 0, vec![0], 28).unwrap();
        let middle = Layout::contiguous(DType::F32, 2, vec![3], 28).unwrap();
        let hull = ByteHull::from_layouts([&empty, &middle]);
        let touching = Layout::contiguous(DType::F32, 5, vec![1], 28).unwrap();
        let overlapping = Layout::contiguous(DType::F32, 4, vec![1], 28).unwrap();

        assert_eq!(hull.byte_span(), Some(8..20));
        assert!(!hull.overlaps(&empty));
        assert!(!hull.overlaps(&touching));
        assert!(hull.overlaps(&overlapping));
        assert_eq!(ByteHull::from_layouts([&empty]).byte_span(), None);
        assert!(hull.overlaps_hull(&ByteHull::from_layouts([&overlapping])));
        assert!(!hull.overlaps_hull(&ByteHull::from_layouts([&touching])));
        assert!(!hull.overlaps_hull(&ByteHull::empty()));

        let range = RangeInclusive::new(1, 3);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![7], 28).unwrap();
        let symbolic = SymbolicLayout::new(base, space)
            .slice(0, Affine::parameter(0, 0, 1), 1.into(), 1)
            .unwrap();
        assert_eq!(symbolic.byte_hull().unwrap().byte_span(), Some(4..16));
    }

    #[test]
    fn corner_contiguity_does_not_prove_interior_contiguity() {
        let range = RangeInclusive::new(0, 3);
        let space = ParamSpace::new(vec![range]).unwrap();
        let base = Layout::contiguous(DType::F32, 0, vec![3, 3], 36).unwrap();
        let layout = SymbolicLayout::new(base, space.clone())
            .slice(1, 0.into(), Affine::parameter(0, 0, 1), 1)
            .unwrap();

        assert!(
            layout
                .corner_layouts()
                .unwrap()
                .iter()
                .all(Layout::is_contiguous)
        );
        assert!(
            !layout
                .instantiate(&space.values(vec![2]).unwrap())
                .unwrap()
                .is_contiguous()
        );
    }

    #[test]
    fn constraint_facts_follow_shape_transforms() {
        let empty = ParamSpace::new(Vec::new()).unwrap();
        let permuted = SymbolicLayout::new(
            Layout::contiguous(DType::F32, 0, vec![2, 3, 4], 96).unwrap(),
            empty.clone(),
        )
        .permute(&[1, 0, 2])
        .unwrap()
        .constraint_facts()
        .unwrap();
        assert_eq!(permuted.dimensions, [3.into(), 2.into(), 4.into()]);
        assert_eq!(permuted.strides, [4, 12, 1]);

        let broadcast = SymbolicLayout::new(
            Layout::contiguous(DType::F16, 0, vec![1, 4], 8).unwrap(),
            empty.clone(),
        )
        .broadcast(vec![7, 4])
        .unwrap()
        .constraint_facts()
        .unwrap();
        assert_eq!(broadcast.dimensions, [7.into(), 4.into()]);
        assert_eq!(broadcast.strides, [0, 1]);

        let reshaped = SymbolicLayout::new(
            Layout::contiguous(DType::BF16, 0, vec![2, 3, 4], 48).unwrap(),
            empty,
        )
        .reshape(vec![6, 4])
        .unwrap()
        .constraint_facts()
        .unwrap();
        assert_eq!(reshaped.dimensions, [6.into(), 4.into()]);
        assert_eq!(reshaped.strides, [4, 1]);
    }

    fn chained_recipe(
        hi0: u32,
        hi1: u32,
        step: u32,
        constant_start: u32,
    ) -> (ParamSpace, SymbolicLayout) {
        let space = ParamSpace::new(vec![0..=hi0, 0..=hi1]).unwrap();
        let symbolic_len_offset = hi0 + 1;
        let first_extent = hi0 + (hi0 + hi1) * step + 1;
        let buffer_len = u64::from(first_extent) * 7 * 3 * DType::F32.byte_size();
        let base = Layout::contiguous(DType::F32, 0, vec![first_extent, 7, 3], buffer_len).unwrap();
        let layout = SymbolicLayout::new(base, space.clone())
            .slice(
                0,
                Affine::parameter(0, 0, 1),
                Affine::parameter(1, symbolic_len_offset, 1),
                step,
            )
            .unwrap()
            .slice(0, Affine::parameter(0, 0, 1), Affine::parameter(1, 1, 1), 1)
            .unwrap()
            .slice(1, constant_start.into(), (7 - constant_start).into(), 1)
            .unwrap()
            .permute(&[1, 0, 2])
            .unwrap();
        (space, layout)
    }

    proptest! {
        #[test]
        fn chained_recipe_instantiations_stay_in_the_base_buffer(
            hi0 in 0_u32..=3,
            hi1 in 0_u32..=3,
            step in 2_u32..=3,
            constant_start in 0_u32..=2,
        ) {
            let (space, symbolic) = chained_recipe(hi0, hi1, step, constant_start);
            for value0 in 0..=hi0 {
                for value1 in 0..=hi1 {
                    if let Ok(layout) =
                        symbolic.instantiate(&space.values(vec![value0, value1]).unwrap())
                    {
                        prop_assert!(layout.byte_span().end <= symbolic.base().buffer_len());
                    }
                }
            }
        }

        #[test]
        fn chained_recipe_hull_contains_every_instantiation(
            hi0 in 0_u32..=3,
            hi1 in 0_u32..=3,
            step in 2_u32..=3,
            constant_start in 0_u32..=2,
        ) {
            let (space, symbolic) = chained_recipe(hi0, hi1, step, constant_start);
            let hull = symbolic.byte_hull().unwrap();
            for value0 in 0..=hi0 {
                for value1 in 0..=hi1 {
                    if let Ok(layout) =
                        symbolic.instantiate(&space.values(vec![value0, value1]).unwrap())
                    {
                        prop_assert!(hull.contains(&layout));
                    }
                }
            }
        }
    }
}
