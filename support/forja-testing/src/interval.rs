#![allow(clippy::float_cmp)]

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use forja_core::{
    DType, Layout, ViewOp,
    program::{BinOp, Inst, ProgramKind, RedOp, UnOp, ValueType},
};
use half::{bf16, f16};

use crate::{AgreementError, TensorSpec, program::ProgramCase};

const F32_ULPS: u32 = 1;
const PRECISE_ULPS: u32 = 4;
const ZERO_SCALE_NORMALS: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatInterval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
    pub(crate) may_nan: bool,
}

impl FloatInterval {
    fn point(value: f64) -> Self {
        if value.is_nan() {
            Self::nan()
        } else {
            Self {
                lo: value,
                hi: value,
                may_nan: false,
            }
        }
    }

    const fn nan() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
            may_nan: true,
        }
    }

    const fn has_numbers(self) -> bool {
        self.lo <= self.hi
    }

    fn contains_zero(self) -> bool {
        self.has_numbers() && self.lo <= 0.0 && self.hi >= 0.0
    }

    fn hull(self, other: Self) -> Self {
        let (lo, hi) = match (self.has_numbers(), other.has_numbers()) {
            (true, true) => (self.lo.min(other.lo), self.hi.max(other.hi)),
            (true, false) => (self.lo, self.hi),
            (false, true) => (other.lo, other.hi),
            (false, false) => (f64::INFINITY, f64::NEG_INFINITY),
        };
        Self {
            lo,
            hi,
            may_nan: self.may_nan || other.may_nan,
        }
    }

    pub(crate) fn contains(self, value: f32) -> bool {
        if value.is_nan() {
            self.may_nan
        } else {
            self.has_numbers() && f64::from(value) >= self.lo && f64::from(value) <= self.hi
        }
    }

    pub(crate) fn relative_width(self) -> f64 {
        if !self.has_numbers() {
            return 0.0;
        }
        if self.lo.to_bits() == self.hi.to_bits() {
            return 0.0;
        }
        if !self.lo.is_finite() || !self.hi.is_finite() {
            return f64::INFINITY;
        }
        let width = self.hi - self.lo;
        let center = self.lo + width / 2.0;
        if center == 0.0 {
            if width == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            width / center.abs()
        }
    }

    pub(crate) fn is_meaningful(self, relative_limit: f64) -> bool {
        self.relative_width() <= relative_limit || self.absolute_width() <= zero_scale_width_limit()
    }

    fn absolute_width(self) -> f64 {
        if self.has_numbers() {
            self.hi - self.lo
        } else {
            0.0
        }
    }

    pub(crate) fn converted(self, dtype: DType) -> Result<Self, AgreementError> {
        if matches!(dtype, DType::I32 | DType::U32) {
            return Err(AgreementError::UnsupportedDType(dtype));
        }
        if !self.has_numbers() {
            return Ok(self);
        }
        Ok(Self {
            lo: quantize(self.lo, dtype),
            hi: quantize(self.hi, dtype),
            may_nan: self.may_nan,
        })
    }

    fn apply_dtype_overflow(&mut self, dtype: DType) {
        let maximum = match dtype {
            DType::F32 => f64::from(f32::MAX),
            DType::F16 => f64::from(f16::MAX.to_f32()),
            DType::BF16 => f64::from(bf16::MAX.to_f32()),
            DType::I32 | DType::U32 => return,
        };
        self.lo = overflow_endpoint(self.lo, maximum);
        self.hi = overflow_endpoint(self.hi, maximum);
    }
}

fn zero_scale_width_limit() -> f64 {
    // One minimum-normal f32 keeps the exception inside the subnormal scale.
    f64::from(f32::MIN_POSITIVE) * ZERO_SCALE_NORMALS
}

#[derive(Clone, Copy)]
struct U32Interval {
    lo: u32,
    hi: u32,
}

impl U32Interval {
    const fn point(value: u32) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    const fn full() -> Self {
        Self {
            lo: 0,
            hi: u32::MAX,
        }
    }

    const fn is_point(self) -> bool {
        self.lo == self.hi
    }

    fn hull(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
}

#[derive(Clone, Copy)]
struct BoolInterval {
    may_be_false: bool,
    may_be_true: bool,
}

impl BoolInterval {
    const fn ambiguous(self) -> bool {
        self.may_be_false && self.may_be_true
    }

    fn hull(self, other: Self) -> Self {
        Self {
            may_be_false: self.may_be_false || other.may_be_false,
            may_be_true: self.may_be_true || other.may_be_true,
        }
    }
}

#[derive(Clone, Copy)]
enum ScalarInterval {
    Float(FloatInterval),
    U32(U32Interval),
    Bool(BoolInterval),
}

#[derive(Clone, Copy)]
struct ValueInterval {
    scalar: ScalarInterval,
    origin: Option<u32>,
}

struct EvalLocation<'a> {
    coordinates: &'a [u32],
    shape: &'a [u32],
    inputs: &'a [Vec<ScalarInterval>],
    linear: usize,
    inst_index: u32,
}

pub(crate) struct Evaluation {
    pub(crate) outputs: Vec<Vec<FloatInterval>>,
    pub(crate) ambiguous_predicates: usize,
    pub(crate) ambiguous_selects: usize,
    pub(crate) reduction_steps: usize,
}

pub(crate) fn evaluate(
    case: &ProgramCase,
    input_bytes: &[Vec<u8>],
) -> Result<Evaluation, AgreementError> {
    let elements = element_count(case.shape())?;
    let width = if case.program().kind == ProgramKind::Row {
        usize::try_from(*case.shape().last().ok_or(AgreementError::InvalidOutput)?)
            .map_err(|_| AgreementError::SizeOverflow)?
    } else {
        1
    };
    let scopes = elements
        .checked_div(width)
        .ok_or(AgreementError::InvalidOutput)?;
    let inputs = case
        .inputs()
        .iter()
        .zip(input_bytes)
        .map(|(spec, bytes)| logical_values(spec, bytes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut evaluation = Evaluation {
        outputs: (0..case.outputs().len())
            .map(|_| Vec::with_capacity(elements))
            .collect(),
        ambiguous_predicates: 0,
        ambiguous_selects: 0,
        reduction_steps: 0,
    };
    let mut coordinates = vec![0_u32; case.shape().len()];
    for scope in 0..scopes {
        let start = scope
            .checked_mul(width)
            .ok_or(AgreementError::SizeOverflow)?;
        let mut columns = Vec::<Vec<ValueInterval>>::with_capacity(case.program().insts.len());
        for (inst_index, &inst) in case.program().insts.iter().enumerate() {
            let inst_index = u32::try_from(inst_index).map_err(|_| AgreementError::SizeOverflow)?;
            let column = if let Inst::Reduce(op, operand) = inst {
                let values = column(&columns, operand)?;
                evaluation.reduction_steps = evaluation
                    .reduction_steps
                    .checked_add(values.len().saturating_sub(1))
                    .ok_or(AgreementError::SizeOverflow)?;
                vec![
                    ValueInterval {
                        scalar: reduce(op, values)?,
                        origin: Some(inst_index),
                    };
                    width
                ]
            } else {
                let mut values = Vec::with_capacity(width);
                for lane in 0..width {
                    let linear = start
                        .checked_add(lane)
                        .ok_or(AgreementError::SizeOverflow)?;
                    decode_coordinates(linear, case.shape(), &mut coordinates)?;
                    values.push(eval_inst(
                        inst,
                        &EvalLocation {
                            coordinates: &coordinates,
                            shape: case.shape(),
                            inputs: &inputs,
                            linear,
                            inst_index,
                        },
                        |operand| value(&columns, operand, lane),
                        &mut evaluation,
                    )?);
                }
                values
            };
            columns.push(column);
        }
        for &(slot, output_value) in &case.program().outputs {
            let output = evaluation
                .outputs
                .get_mut(usize::try_from(slot).map_err(|_| AgreementError::InvalidOutput)?)
                .ok_or(AgreementError::InvalidOutput)?;
            for &value in column(&columns, output_value)? {
                let ScalarInterval::Float(value) = value.scalar else {
                    return Err(AgreementError::InvalidOutput);
                };
                output.push(value);
            }
        }
    }
    Ok(evaluation)
}

fn eval_inst(
    inst: Inst,
    location: &EvalLocation<'_>,
    operand: impl Fn(u32) -> Result<ValueInterval, AgreementError>,
    evaluation: &mut Evaluation,
) -> Result<ValueInterval, AgreementError> {
    let produced = |scalar| ValueInterval {
        scalar,
        origin: Some(location.inst_index),
    };
    match inst {
        Inst::Input(slot) => location
            .inputs
            .get(usize::try_from(slot).map_err(|_| AgreementError::InvalidOutput)?)
            .and_then(|input| input.get(location.linear))
            .copied()
            .map(produced)
            .ok_or(AgreementError::InvalidOutput),
        Inst::Const(value) => Ok(produced(ScalarInterval::Float(FloatInterval::point(
            f64::from(value),
        )))),
        Inst::Index(axis) => location
            .coordinates
            .get(usize::from(axis))
            .copied()
            .map(U32Interval::point)
            .map(ScalarInterval::U32)
            .map(produced)
            .ok_or(AgreementError::InvalidOutput),
        Inst::Extent(axis) => location
            .shape
            .get(usize::from(axis))
            .copied()
            .map(U32Interval::point)
            .map(ScalarInterval::U32)
            .map(produced)
            .ok_or(AgreementError::InvalidOutput),
        Inst::Unary(op, input) => unary(op, operand(input)?.scalar).map(produced),
        Inst::Binary(op, left, right) => {
            let left = operand(left)?;
            let right = operand(right)?;
            let result = if left.origin == right.origin {
                binary_same(op, left.scalar)?
            } else {
                binary(op, left.scalar, right.scalar)?
            };
            if matches!(result, ScalarInterval::Bool(value) if value.ambiguous()) {
                evaluation.ambiguous_predicates += 1;
            }
            Ok(produced(result))
        }
        Inst::Select(condition, accepted, rejected) => {
            let ScalarInterval::Bool(condition) = operand(condition)?.scalar else {
                return Err(AgreementError::InvalidOutput);
            };
            if condition.ambiguous() {
                evaluation.ambiguous_selects += 1;
            }
            select(condition, operand(accepted)?, operand(rejected)?)
        }
        Inst::Cast(to, input) => {
            let input = operand(input)?;
            if to == ValueType::F32 && matches!(input.scalar, ScalarInterval::Float(_)) {
                Ok(input)
            } else {
                Ok(produced(cast(to, input.scalar)))
            }
        }
        Inst::Reduce(_, _) => Err(AgreementError::InvalidOutput),
    }
}

fn unary(op: UnOp, input: ScalarInterval) -> Result<ScalarInterval, AgreementError> {
    let ScalarInterval::Float(input) = input else {
        return Err(AgreementError::InvalidOutput);
    };
    if let Some(value) = exact_transcendental(op, input) {
        return Ok(ScalarInterval::Float(value));
    }
    let result = match op {
        UnOp::Neg => map_monotone(input, |value| -value, true, 0),
        UnOp::Abs => absolute(input),
        UnOp::Exp => map_monotone(input, f64::exp, false, PRECISE_ULPS),
        UnOp::Log => domain_monotone(input, 0.0, false, f64::ln, PRECISE_ULPS),
        UnOp::Sqrt => domain_monotone(input, 0.0, true, f64::sqrt, PRECISE_ULPS),
        UnOp::Rsqrt => reciprocal_sqrt(input),
        UnOp::Sin => trigonometric(input, true),
        UnOp::Cos => trigonometric(input, false),
        UnOp::Tanh => map_monotone(input, f64::tanh, false, PRECISE_ULPS),
        UnOp::Sigmoid => map_monotone(
            input,
            |value| 1.0 / (1.0 + (-value).exp()),
            false,
            PRECISE_ULPS,
        ),
        UnOp::Recip => rounded(reciprocal(input), F32_ULPS),
        UnOp::Floor => map_monotone(input, f64::floor, false, 0),
    };
    Ok(ScalarInterval::Float(result))
}

fn exact_transcendental(op: UnOp, input: FloatInterval) -> Option<FloatInterval> {
    if input.may_nan || input.lo.to_bits() != input.hi.to_bits() {
        return None;
    }
    let value = match (op, input.lo) {
        (UnOp::Exp | UnOp::Cos, 0.0) => 1.0_f64,
        (UnOp::Log, 1.0) => 0.0,
        (UnOp::Sqrt | UnOp::Sin | UnOp::Tanh, 0.0) => input.lo,
        (UnOp::Sigmoid, 0.0) => 0.5,
        _ => return None,
    };
    Some(FloatInterval::point(value))
}

fn binary(
    op: BinOp,
    left: ScalarInterval,
    right: ScalarInterval,
) -> Result<ScalarInterval, AgreementError> {
    match (left, right) {
        (ScalarInterval::Float(left), ScalarInterval::Float(right)) => {
            Ok(float_binary(op, left, right))
        }
        (ScalarInterval::U32(left), ScalarInterval::U32(right)) => integer_binary(op, left, right),
        _ => Err(AgreementError::InvalidOutput),
    }
}

fn binary_same(op: BinOp, input: ScalarInterval) -> Result<ScalarInterval, AgreementError> {
    match input {
        ScalarInterval::Float(value) => match op {
            BinOp::Sub => Ok(ScalarInterval::Float(if value.has_numbers() {
                FloatInterval {
                    lo: 0.0,
                    hi: 0.0,
                    may_nan: value.may_nan
                        || value.lo == f64::NEG_INFINITY
                        || value.hi == f64::INFINITY,
                }
            } else {
                FloatInterval::nan()
            })),
            BinOp::Mul => Ok(ScalarInterval::Float(rounded(
                FloatInterval {
                    lo: if value.contains_zero() {
                        0.0
                    } else {
                        value.lo.abs().min(value.hi.abs()).powi(2)
                    },
                    hi: value.lo.abs().max(value.hi.abs()).powi(2),
                    may_nan: value.may_nan,
                },
                F32_ULPS,
            ))),
            BinOp::Min | BinOp::Max => Ok(ScalarInterval::Float(value)),
            BinOp::Lt | BinOp::Gt => Ok(ScalarInterval::Bool(BoolInterval {
                may_be_false: true,
                may_be_true: false,
            })),
            BinOp::Le | BinOp::Eq | BinOp::Ge => Ok(ScalarInterval::Bool(BoolInterval {
                may_be_false: value.may_nan,
                may_be_true: value.has_numbers(),
            })),
            BinOp::Ne => Ok(ScalarInterval::Bool(BoolInterval {
                may_be_false: value.has_numbers(),
                may_be_true: value.may_nan,
            })),
            BinOp::Add | BinOp::Div | BinOp::Pow => binary(
                op,
                ScalarInterval::Float(value),
                ScalarInterval::Float(value),
            ),
        },
        ScalarInterval::U32(value) => match op {
            BinOp::Sub => Ok(ScalarInterval::U32(U32Interval::point(0))),
            BinOp::Min | BinOp::Max => Ok(ScalarInterval::U32(value)),
            BinOp::Lt | BinOp::Gt | BinOp::Ne => Ok(ScalarInterval::Bool(BoolInterval {
                may_be_false: true,
                may_be_true: false,
            })),
            BinOp::Le | BinOp::Eq | BinOp::Ge => Ok(ScalarInterval::Bool(BoolInterval {
                may_be_false: false,
                may_be_true: true,
            })),
            BinOp::Add | BinOp::Mul => {
                binary(op, ScalarInterval::U32(value), ScalarInterval::U32(value))
            }
            BinOp::Div | BinOp::Pow => Err(AgreementError::InvalidOutput),
        },
        ScalarInterval::Bool(_) => Err(AgreementError::InvalidOutput),
    }
}

fn float_binary(op: BinOp, left: FloatInterval, right: FloatInterval) -> ScalarInterval {
    let value = match op {
        BinOp::Add => rounded(add(left, right), F32_ULPS),
        BinOp::Sub => rounded(subtract(left, right), F32_ULPS),
        BinOp::Mul => rounded(multiply(left, right), F32_ULPS),
        BinOp::Div => rounded(divide(left, right), F32_ULPS),
        BinOp::Min => extreme(left, right, false),
        BinOp::Max => extreme(left, right, true),
        BinOp::Pow => power(left, right),
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            return ScalarInterval::Bool(float_comparison(op, left, right));
        }
    };
    ScalarInterval::Float(value)
}

fn integer_binary(
    op: BinOp,
    left: U32Interval,
    right: U32Interval,
) -> Result<ScalarInterval, AgreementError> {
    let value = match op {
        BinOp::Add => integer_add(left, right),
        BinOp::Sub => integer_subtract(left, right),
        BinOp::Mul => integer_multiply(left, right),
        BinOp::Min => U32Interval {
            lo: left.lo.min(right.lo),
            hi: left.hi.min(right.hi),
        },
        BinOp::Max => U32Interval {
            lo: left.lo.max(right.lo),
            hi: left.hi.max(right.hi),
        },
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            return Ok(ScalarInterval::Bool(integer_comparison(op, left, right)));
        }
        BinOp::Div | BinOp::Pow => return Err(AgreementError::InvalidOutput),
    };
    Ok(ScalarInterval::U32(value))
}

fn select(
    condition: BoolInterval,
    accepted: ValueInterval,
    rejected: ValueInterval,
) -> Result<ValueInterval, AgreementError> {
    match (condition.may_be_true, condition.may_be_false) {
        (true, false) => Ok(accepted),
        (false, true) => Ok(rejected),
        (true, true) => Ok(ValueInterval {
            scalar: hull(accepted.scalar, rejected.scalar)?,
            origin: (accepted.origin == rejected.origin)
                .then_some(accepted.origin)
                .flatten(),
        }),
        (false, false) => Err(AgreementError::InvalidOutput),
    }
}

fn hull(left: ScalarInterval, right: ScalarInterval) -> Result<ScalarInterval, AgreementError> {
    match (left, right) {
        (ScalarInterval::Float(left), ScalarInterval::Float(right)) => {
            Ok(ScalarInterval::Float(left.hull(right)))
        }
        (ScalarInterval::U32(left), ScalarInterval::U32(right)) => {
            Ok(ScalarInterval::U32(left.hull(right)))
        }
        (ScalarInterval::Bool(left), ScalarInterval::Bool(right)) => {
            Ok(ScalarInterval::Bool(left.hull(right)))
        }
        _ => Err(AgreementError::InvalidOutput),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn cast(to: ValueType, input: ScalarInterval) -> ScalarInterval {
    match to {
        ValueType::F32 => ScalarInterval::Float(match input {
            ScalarInterval::Float(value) => value,
            ScalarInterval::U32(value) => FloatInterval {
                lo: quantize(f64::from(value.lo), DType::F32),
                hi: quantize(f64::from(value.hi), DType::F32),
                may_nan: false,
            },
            ScalarInterval::Bool(value) => FloatInterval {
                lo: f64::from(u8::from(!value.may_be_false)),
                hi: f64::from(u8::from(value.may_be_true)),
                may_nan: false,
            },
        }),
        ValueType::U32 => ScalarInterval::U32(match input {
            ScalarInterval::Float(value) => float_to_u32(value),
            ScalarInterval::U32(value) => value,
            ScalarInterval::Bool(value) => U32Interval {
                lo: u32::from(!value.may_be_false),
                hi: u32::from(value.may_be_true),
            },
        }),
        ValueType::Bool => ScalarInterval::Bool(match input {
            ScalarInterval::Float(value) => BoolInterval {
                may_be_false: value.contains_zero(),
                may_be_true: value.may_nan
                    || (value.has_numbers() && (value.lo < 0.0 || value.hi > 0.0)),
            },
            ScalarInterval::U32(value) => BoolInterval {
                may_be_false: value.lo == 0,
                may_be_true: value.hi > 0,
            },
            ScalarInterval::Bool(value) => value,
        }),
    }
}

fn reduce(op: RedOp, values: &[ValueInterval]) -> Result<ScalarInterval, AgreementError> {
    let floats = values
        .iter()
        .map(|value| match value.scalar {
            ScalarInterval::Float(value) => Ok(value),
            ScalarInterval::U32(_) | ScalarInterval::Bool(_) => Err(AgreementError::InvalidOutput),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let value = match op {
        RedOp::Sum => sum(&floats)?,
        RedOp::Max => reduce_extreme(&floats, true)?,
        RedOp::Min => reduce_extreme(&floats, false)?,
    };
    Ok(ScalarInterval::Float(value))
}

fn sum(values: &[FloatInterval]) -> Result<FloatInterval, AgreementError> {
    let mut lo = 0.0;
    let mut hi = 0.0;
    let mut magnitude = 0.0;
    let mut may_nan = false;
    for value in values {
        may_nan |= value.may_nan;
        if value.has_numbers() {
            lo += value.lo;
            hi += value.hi;
            magnitude += value.lo.abs().max(value.hi.abs());
        } else {
            may_nan = true;
        }
    }
    let steps =
        u32::try_from(values.len().saturating_sub(1)).map_err(|_| AgreementError::SizeOverflow)?;
    let product = f64::from(steps) * f64::from(f32::EPSILON);
    let error = if product < 1.0 {
        product / (1.0 - product) * magnitude
    } else {
        f64::INFINITY
    };
    Ok(FloatInterval {
        lo: lo - error,
        hi: hi + error,
        may_nan: may_nan || (lo.is_infinite() && hi.is_infinite()),
    })
}

fn reduce_extreme(
    values: &[FloatInterval],
    maximum: bool,
) -> Result<FloatInterval, AgreementError> {
    let first = values.first().ok_or(AgreementError::InvalidOutput)?;
    let mut lo = first.lo;
    let mut hi = first.hi;
    let mut may_nan = first.may_nan;
    for value in &values[1..] {
        may_nan |= value.may_nan;
        if maximum {
            lo = lo.max(value.lo);
            hi = hi.max(value.hi);
        } else {
            lo = lo.min(value.lo);
            hi = hi.min(value.hi);
        }
    }
    Ok(FloatInterval { lo, hi, may_nan })
}

fn map_monotone(
    input: FloatInterval,
    operation: impl Fn(f64) -> f64,
    decreasing: bool,
    ulps: u32,
) -> FloatInterval {
    if !input.has_numbers() {
        return FloatInterval::nan();
    }
    let (lo, hi) = if decreasing {
        (operation(input.hi), operation(input.lo))
    } else {
        (operation(input.lo), operation(input.hi))
    };
    rounded(
        FloatInterval {
            lo,
            hi,
            may_nan: input.may_nan || lo.is_nan() || hi.is_nan(),
        },
        ulps,
    )
}

fn domain_monotone(
    input: FloatInterval,
    minimum: f64,
    inclusive: bool,
    operation: impl Fn(f64) -> f64,
    ulps: u32,
) -> FloatInterval {
    if !input.has_numbers() || input.hi < minimum || (!inclusive && input.hi == minimum) {
        return FloatInterval::nan();
    }
    let lower = input.lo.max(minimum);
    let lo = operation(lower);
    let hi = operation(input.hi);
    rounded(
        FloatInterval {
            lo,
            hi,
            may_nan: input.may_nan || input.lo < minimum || (!inclusive && input.lo <= minimum),
        },
        ulps,
    )
}

fn absolute(input: FloatInterval) -> FloatInterval {
    if !input.has_numbers() {
        return input;
    }
    FloatInterval {
        lo: if input.contains_zero() {
            0.0
        } else {
            input.lo.abs().min(input.hi.abs())
        },
        hi: input.lo.abs().max(input.hi.abs()),
        may_nan: input.may_nan,
    }
}

fn reciprocal(input: FloatInterval) -> FloatInterval {
    if !input.has_numbers() {
        return input;
    }
    if input.contains_zero() {
        return FloatInterval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
            may_nan: input.may_nan,
        };
    }
    FloatInterval {
        lo: input.hi.recip(),
        hi: input.lo.recip(),
        may_nan: input.may_nan,
    }
}

fn reciprocal_sqrt(input: FloatInterval) -> FloatInterval {
    if !input.has_numbers() || input.hi < 0.0 {
        return FloatInterval::nan();
    }
    rounded(
        FloatInterval {
            lo: input.hi.sqrt().recip(),
            hi: input.lo.max(0.0).sqrt().recip(),
            may_nan: input.may_nan || input.lo < 0.0,
        },
        PRECISE_ULPS,
    )
}

fn trigonometric(input: FloatInterval, sine: bool) -> FloatInterval {
    if !input.has_numbers() {
        return input;
    }
    if !input.lo.is_finite() || !input.hi.is_finite() {
        return FloatInterval {
            lo: -1.0,
            hi: 1.0,
            may_nan: true,
        };
    }
    let operation = if sine { f64::sin } else { f64::cos };
    let mut lo = operation(input.lo).min(operation(input.hi));
    let mut hi = operation(input.lo).max(operation(input.hi));
    let offset = if sine { FRAC_PI_2 } else { 0.0 };
    if input.hi - input.lo >= TAU {
        lo = -1.0;
        hi = 1.0;
    } else {
        let first = ((input.lo - offset) / PI).ceil();
        let last = ((input.hi - offset) / PI).floor();
        if first <= last {
            if first.rem_euclid(2.0) == 0.0 || last > first {
                hi = 1.0;
            }
            if first.rem_euclid(2.0) == 1.0 || last > first {
                lo = -1.0;
            }
        }
    }
    rounded(
        FloatInterval {
            lo,
            hi,
            may_nan: input.may_nan,
        },
        PRECISE_ULPS,
    )
}

fn add(left: FloatInterval, right: FloatInterval) -> FloatInterval {
    FloatInterval {
        lo: left.lo + right.lo,
        hi: left.hi + right.hi,
        may_nan: left.may_nan
            || right.may_nan
            || (left.lo + right.lo).is_nan()
            || (left.hi + right.hi).is_nan(),
    }
}

fn subtract(left: FloatInterval, right: FloatInterval) -> FloatInterval {
    FloatInterval {
        lo: left.lo - right.hi,
        hi: left.hi - right.lo,
        may_nan: left.may_nan
            || right.may_nan
            || (left.lo - right.hi).is_nan()
            || (left.hi - right.lo).is_nan(),
    }
}

fn multiply(left: FloatInterval, right: FloatInterval) -> FloatInterval {
    if (is_zero_point(left) && finite_numbers(right))
        || (is_zero_point(right) && finite_numbers(left))
    {
        return FloatInterval {
            lo: 0.0,
            hi: 0.0,
            may_nan: left.may_nan || right.may_nan,
        };
    }
    bounds(
        [
            left.lo * right.lo,
            left.lo * right.hi,
            left.hi * right.lo,
            left.hi * right.hi,
        ],
        left.may_nan || right.may_nan,
    )
}

fn divide(left: FloatInterval, right: FloatInterval) -> FloatInterval {
    if is_zero_point(left) && finite_numbers(right) && !right.contains_zero() {
        return FloatInterval {
            lo: 0.0,
            hi: 0.0,
            may_nan: left.may_nan || right.may_nan,
        };
    }
    if right.contains_zero() {
        return FloatInterval {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
            may_nan: left.may_nan || right.may_nan || left.contains_zero(),
        };
    }
    bounds(
        [
            left.lo / right.lo,
            left.lo / right.hi,
            left.hi / right.lo,
            left.hi / right.hi,
        ],
        left.may_nan || right.may_nan,
    )
}

fn is_zero_point(value: FloatInterval) -> bool {
    value.has_numbers() && value.lo == 0.0 && value.hi == 0.0
}

fn finite_numbers(value: FloatInterval) -> bool {
    value.has_numbers() && value.lo.is_finite() && value.hi.is_finite()
}

fn bounds(values: [f64; 4], inherited_nan: bool) -> FloatInterval {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut may_nan = inherited_nan;
    for value in values {
        if value.is_nan() {
            may_nan = true;
        } else {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    FloatInterval { lo, hi, may_nan }
}

fn extreme(left: FloatInterval, right: FloatInterval, maximum: bool) -> FloatInterval {
    if !left.has_numbers() || !right.has_numbers() {
        return FloatInterval::nan();
    }
    FloatInterval {
        lo: if maximum {
            left.lo.max(right.lo)
        } else {
            left.lo.min(right.lo)
        },
        hi: if maximum {
            left.hi.max(right.hi)
        } else {
            left.hi.min(right.hi)
        },
        may_nan: left.may_nan || right.may_nan,
    }
}

fn power(base: FloatInterval, exponent: FloatInterval) -> FloatInterval {
    if exponent.has_numbers() && exponent.lo == 0.0 && exponent.hi == 0.0 {
        return FloatInterval::point(1.0);
    }
    if !base.has_numbers() || !exponent.has_numbers() {
        return FloatInterval::nan();
    }
    if base.lo == base.hi && exponent.lo == exponent.hi {
        let value = base.lo.powf(exponent.lo);
        return rounded(
            FloatInterval {
                lo: value,
                hi: value,
                may_nan: base.may_nan || exponent.may_nan || value.is_nan(),
            },
            PRECISE_ULPS,
        );
    }
    if exponent.lo == exponent.hi
        && exponent.lo.fract() == 0.0
        && exponent.lo > f64::from(i32::MIN)
        && exponent.lo <= f64::from(i32::MAX)
    {
        #[allow(clippy::cast_possible_truncation)]
        let exponent = exponent.lo as i32;
        let value = if exponent == 0 {
            FloatInterval::point(1.0)
        } else if exponent > 0 && exponent % 2 == 0 {
            let smallest = if base.contains_zero() {
                0.0
            } else {
                base.lo.abs().min(base.hi.abs()).powi(exponent)
            };
            FloatInterval {
                lo: smallest,
                hi: base.lo.abs().max(base.hi.abs()).powi(exponent),
                may_nan: base.may_nan,
            }
        } else {
            let positive = exponent.abs();
            let powered = FloatInterval {
                lo: base.lo.powi(positive),
                hi: base.hi.powi(positive),
                may_nan: base.may_nan,
            };
            if exponent < 0 {
                reciprocal(powered)
            } else {
                powered
            }
        };
        return rounded(value, PRECISE_ULPS);
    }
    if base.lo > 0.0 {
        let logarithm = map_monotone(base, f64::ln, false, 0);
        let product = multiply(logarithm, exponent);
        return map_monotone(product, f64::exp, false, PRECISE_ULPS);
    }
    FloatInterval {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
        may_nan: true,
    }
}

fn float_comparison(op: BinOp, left: FloatInterval, right: FloatInterval) -> BoolInterval {
    let nan_true = matches!(op, BinOp::Ne) && (left.may_nan || right.may_nan);
    let nan_false = !matches!(op, BinOp::Ne) && (left.may_nan || right.may_nan);
    if !left.has_numbers() || !right.has_numbers() {
        return BoolInterval {
            may_be_false: nan_false,
            may_be_true: nan_true,
        };
    }
    let (may_be_true, may_be_false) = match op {
        BinOp::Lt => (left.lo < right.hi, left.hi >= right.lo),
        BinOp::Le => (left.lo <= right.hi, left.hi > right.lo),
        BinOp::Eq => (
            left.lo <= right.hi && right.lo <= left.hi,
            left.lo != left.hi || right.lo != right.hi || left.lo != right.lo,
        ),
        BinOp::Ne => (
            left.lo != left.hi || right.lo != right.hi || left.lo != right.lo,
            left.lo <= right.hi && right.lo <= left.hi,
        ),
        BinOp::Ge => (left.hi >= right.lo, left.lo < right.hi),
        BinOp::Gt => (left.hi > right.lo, left.lo <= right.hi),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Min
        | BinOp::Max
        | BinOp::Pow => (false, false),
    };
    BoolInterval {
        may_be_false: may_be_false || nan_false,
        may_be_true: may_be_true || nan_true,
    }
}

fn integer_comparison(op: BinOp, left: U32Interval, right: U32Interval) -> BoolInterval {
    let (may_be_true, may_be_false) = match op {
        BinOp::Lt => (left.lo < right.hi, left.hi >= right.lo),
        BinOp::Le => (left.lo <= right.hi, left.hi > right.lo),
        BinOp::Eq => (
            left.lo <= right.hi && right.lo <= left.hi,
            !left.is_point() || !right.is_point() || left.lo != right.lo,
        ),
        BinOp::Ne => (
            !left.is_point() || !right.is_point() || left.lo != right.lo,
            left.lo <= right.hi && right.lo <= left.hi,
        ),
        BinOp::Ge => (left.hi >= right.lo, left.lo < right.hi),
        BinOp::Gt => (left.hi > right.lo, left.lo <= right.hi),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Min
        | BinOp::Max
        | BinOp::Pow => (false, false),
    };
    BoolInterval {
        may_be_false,
        may_be_true,
    }
}

fn integer_add(left: U32Interval, right: U32Interval) -> U32Interval {
    match (left.lo.checked_add(right.lo), left.hi.checked_add(right.hi)) {
        (Some(lo), Some(hi)) => U32Interval { lo, hi },
        _ if left.is_point() && right.is_point() => {
            U32Interval::point(left.lo.wrapping_add(right.lo))
        }
        _ => U32Interval::full(),
    }
}

fn integer_subtract(left: U32Interval, right: U32Interval) -> U32Interval {
    if left.lo >= right.hi {
        U32Interval {
            lo: left.lo - right.hi,
            hi: left.hi - right.lo,
        }
    } else if left.is_point() && right.is_point() {
        U32Interval::point(left.lo.wrapping_sub(right.lo))
    } else {
        U32Interval::full()
    }
}

fn integer_multiply(left: U32Interval, right: U32Interval) -> U32Interval {
    match (left.lo.checked_mul(right.lo), left.hi.checked_mul(right.hi)) {
        (Some(lo), Some(hi)) => U32Interval { lo, hi },
        _ if left.is_point() && right.is_point() => {
            U32Interval::point(left.lo.wrapping_mul(right.lo))
        }
        _ => U32Interval::full(),
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_to_u32(value: FloatInterval) -> U32Interval {
    if !value.has_numbers() {
        return U32Interval::point(0);
    }
    let mut lo = value.lo as u32;
    let hi = value.hi as u32;
    if value.may_nan {
        lo = 0;
    }
    U32Interval { lo, hi }
}

fn rounded(mut value: FloatInterval, ulps: u32) -> FloatInterval {
    if !value.has_numbers() || ulps == 0 {
        return value;
    }
    value.apply_dtype_overflow(DType::F32);
    if value.lo.is_finite() {
        value.lo -= f64::from(ulps) * f32_ulp(value.lo.abs());
    }
    if value.hi.is_finite() {
        value.hi += f64::from(ulps) * f32_ulp(value.hi.abs());
    }
    value.apply_dtype_overflow(DType::F32);
    value
}

fn overflow_endpoint(value: f64, maximum: f64) -> f64 {
    if value < -maximum {
        f64::NEG_INFINITY
    } else if value > maximum {
        f64::INFINITY
    } else {
        value
    }
}

fn f32_ulp(magnitude: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let value = magnitude as f32;
    if !value.is_finite() {
        return f64::INFINITY;
    }
    let next = f32::from_bits(value.to_bits().saturating_add(1));
    f64::from(next - value)
}

#[allow(clippy::cast_possible_truncation)]
fn quantize(value: f64, dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f64::from(value as f32),
        DType::F16 => f64::from(f16::from_f32(value as f32).to_f32()),
        DType::BF16 => f64::from(bf16::from_f32(value as f32).to_f32()),
        DType::I32 | DType::U32 => value,
    }
}

fn logical_values(spec: &TensorSpec, bytes: &[u8]) -> Result<Vec<ScalarInterval>, AgreementError> {
    let buffer_len = u64::try_from(bytes.len()).map_err(|_| AgreementError::SizeOverflow)?;
    let layout = Layout::contiguous(
        spec.dtype(),
        0,
        spec.allocation_shape().to_vec(),
        buffer_len,
    )
    .map_err(|_| AgreementError::InvalidOutput)?;
    let layout = spec
        .views()
        .iter()
        .try_fold(layout, |layout, view| match view {
            ViewOp::Slice(slices) => layout.slice(slices),
            ViewOp::Reshape(shape) => layout.reshape(shape),
            ViewOp::Permute(axes) => layout.permute(axes),
            ViewOp::Broadcast(shape) => layout.broadcast(shape),
        })
        .map_err(|_| AgreementError::InvalidOutput)?;
    let elements =
        usize::try_from(layout.element_count()).map_err(|_| AgreementError::SizeOverflow)?;
    let mut coordinates = vec![0_u32; layout.shape().len()];
    (0..elements)
        .map(|linear| {
            decode_coordinates(linear, layout.shape(), &mut coordinates)?;
            let index = coordinates.iter().zip(layout.strides()).try_fold(
                layout.offset(),
                |index, (&coordinate, &stride)| {
                    u64::from(coordinate)
                        .checked_mul(stride)
                        .and_then(|offset| index.checked_add(offset))
                        .ok_or(AgreementError::SizeOverflow)
                },
            )?;
            decode_scalar(spec.dtype(), bytes, index)
        })
        .collect()
}

fn decode_scalar(dtype: DType, bytes: &[u8], index: u64) -> Result<ScalarInterval, AgreementError> {
    let width = dtype.byte_size();
    let start = usize::try_from(
        index
            .checked_mul(width)
            .ok_or(AgreementError::SizeOverflow)?,
    )
    .map_err(|_| AgreementError::SizeOverflow)?;
    let end = start
        .checked_add(usize::try_from(width).map_err(|_| AgreementError::SizeOverflow)?)
        .ok_or(AgreementError::SizeOverflow)?;
    let bytes = bytes.get(start..end).ok_or(AgreementError::InvalidOutput)?;
    let value = match dtype {
        DType::F32 => bytes
            .try_into()
            .map(f32::from_le_bytes)
            .map(f64::from)
            .map(FloatInterval::point)
            .map(ScalarInterval::Float),
        DType::F16 => bytes
            .try_into()
            .map(f16::from_le_bytes)
            .map(f16::to_f32)
            .map(f64::from)
            .map(FloatInterval::point)
            .map(ScalarInterval::Float),
        DType::BF16 => bytes
            .try_into()
            .map(bf16::from_le_bytes)
            .map(bf16::to_f32)
            .map(f64::from)
            .map(FloatInterval::point)
            .map(ScalarInterval::Float),
        DType::U32 => bytes
            .try_into()
            .map(u32::from_le_bytes)
            .map(U32Interval::point)
            .map(ScalarInterval::U32),
        DType::I32 => return Err(AgreementError::UnsupportedDType(dtype)),
    };
    value.map_err(|_| AgreementError::InvalidOutput)
}

fn column(
    columns: &[Vec<ValueInterval>],
    operand: u32,
) -> Result<&[ValueInterval], AgreementError> {
    columns
        .get(usize::try_from(operand).map_err(|_| AgreementError::InvalidOutput)?)
        .map(Vec::as_slice)
        .ok_or(AgreementError::InvalidOutput)
}

fn value(
    columns: &[Vec<ValueInterval>],
    operand: u32,
    lane: usize,
) -> Result<ValueInterval, AgreementError> {
    column(columns, operand)?
        .get(lane)
        .copied()
        .ok_or(AgreementError::InvalidOutput)
}

fn element_count(shape: &[u32]) -> Result<usize, AgreementError> {
    shape.iter().try_fold(1_usize, |count, &extent| {
        count
            .checked_mul(usize::try_from(extent).map_err(|_| AgreementError::SizeOverflow)?)
            .ok_or(AgreementError::SizeOverflow)
    })
}

fn decode_coordinates(
    mut linear: usize,
    shape: &[u32],
    coordinates: &mut [u32],
) -> Result<(), AgreementError> {
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape).rev() {
        let extent = usize::try_from(extent).map_err(|_| AgreementError::SizeOverflow)?;
        if extent == 0 {
            return Err(AgreementError::InvalidOutput);
        }
        *coordinate = u32::try_from(linear % extent).map_err(|_| AgreementError::SizeOverflow)?;
        linear /= extent;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use forja_core::program::{Inst, Program};
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn rounded_inputs_near_one_use_their_stored_value() {
        for dtype in [DType::F16, DType::BF16] {
            for raw in [
                f32::from_bits(0x3f7f_c4c8),
                f32::from_bits(0x3f7f_f000),
                f32::from_bits(0x3f80_1000),
            ] {
                let (bytes, stored) = stored_value(dtype, raw);
                for op in [UnOp::Log, UnOp::Exp, UnOp::Sqrt] {
                    let interval = evaluate_unary(dtype, bytes.clone(), op);
                    assert_contains_truth(interval, unary_truth(op, stored));
                }
            }
        }
    }

    proptest! {
        #[test]
        fn single_unary_intervals_contain_stored_truth(
            raw in 0.001_f32..10.0,
            bf16_input in any::<bool>(),
            operation in 0_u8..3,
        ) {
            let dtype = if bf16_input { DType::BF16 } else { DType::F16 };
            let op = [UnOp::Log, UnOp::Exp, UnOp::Sqrt][usize::from(operation)];
            let (bytes, stored) = stored_value(dtype, raw);
            let interval = evaluate_unary(dtype, bytes, op);
            assert_contains_truth(interval, unary_truth(op, stored));
        }
    }

    fn evaluate_unary(dtype: DType, bytes: Vec<u8>, op: UnOp) -> FloatInterval {
        let case = ProgramCase::new(
            Program {
                kind: ProgramKind::Map,
                insts: vec![Inst::Input(0), Inst::Unary(op, 0)],
                outputs: vec![(0, 1)],
            },
            vec![1],
            vec![TensorSpec::contiguous(dtype, &[1])],
            vec![TensorSpec::contiguous(DType::F32, &[1])],
        );
        evaluate(&case, &[bytes]).unwrap().outputs[0][0]
    }

    fn stored_value(dtype: DType, raw: f32) -> (Vec<u8>, f64) {
        match dtype {
            DType::F16 => {
                let stored = f16::from_f32(raw);
                (stored.to_le_bytes().to_vec(), f64::from(stored.to_f32()))
            }
            DType::BF16 => {
                let stored = bf16::from_f32(raw);
                (stored.to_le_bytes().to_vec(), f64::from(stored.to_f32()))
            }
            DType::F32 | DType::I32 | DType::U32 => unreachable!(),
        }
    }

    fn unary_truth(op: UnOp, input: f64) -> f64 {
        match op {
            UnOp::Log => input.ln(),
            UnOp::Exp => input.exp(),
            UnOp::Sqrt => input.sqrt(),
            UnOp::Neg
            | UnOp::Abs
            | UnOp::Rsqrt
            | UnOp::Sin
            | UnOp::Cos
            | UnOp::Tanh
            | UnOp::Sigmoid
            | UnOp::Recip
            | UnOp::Floor => unreachable!(),
        }
    }

    fn assert_contains_truth(interval: FloatInterval, truth: f64) {
        assert!(
            interval.lo <= truth && truth <= interval.hi,
            "{truth} escaped [{}, {}]",
            interval.lo,
            interval.hi,
        );
    }
}
