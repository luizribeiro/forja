//! Guest-side construction of validated scalar programs.

use std::{
    cell::RefCell,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{Error, Result};

/// A scalar program's iteration strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramKind {
    /// Evaluates each output element independently.
    Map,
    /// Evaluates rows and permits reductions over the last axis.
    Row,
}

/// A row reduction operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    /// Sum.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Recip,
    Floor,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Instruction {
    Input(u32),
    Constant(f32),
    Index(i32),
    Extent(i32),
    Unary(UnaryOp, u32),
    Binary(BinaryOp, u32, u32),
    Select(u32, u32, u32),
    Reduce(ReduceOp, u32),
}

#[derive(Debug)]
struct State {
    instructions: Vec<Instruction>,
    outputs: Vec<(u32, u32)>,
    error: Option<Error>,
}

/// A builder for a flat scalar program.
#[derive(Debug)]
pub struct Program {
    kind: ProgramKind,
    state: RefCell<State>,
}

impl Program {
    /// Creates a builder with the selected iteration strategy.
    #[must_use]
    pub fn new(kind: ProgramKind) -> Self {
        Self {
            kind,
            state: RefCell::new(State {
                instructions: Vec::new(),
                outputs: Vec::new(),
                error: None,
            }),
        }
    }

    /// Creates an elementwise program builder.
    #[must_use]
    pub fn map() -> Self {
        Self::new(ProgramKind::Map)
    }

    /// Creates a row-reduction program builder.
    #[must_use]
    pub fn row() -> Self {
        Self::new(ProgramKind::Row)
    }

    /// Loads a bound input slot.
    #[must_use]
    pub fn input(&self, slot: u32) -> Value<'_> {
        self.push_value(Instruction::Input(slot))
    }

    /// Creates an f32 constant.
    #[must_use]
    pub fn constant(&self, value: f32) -> Value<'_> {
        self.push_value(Instruction::Constant(value))
    }

    /// Reads an axis coordinate as f32. Negative axes count from the end.
    #[must_use]
    pub fn index(&self, axis: i32) -> Value<'_> {
        self.push_value(Instruction::Index(axis))
    }

    /// Reads an axis extent as f32. Negative axes count from the end.
    #[must_use]
    pub fn extent(&self, axis: i32) -> Value<'_> {
        self.push_value(Instruction::Extent(axis))
    }

    /// Reduces a value across the last axis and broadcasts the result.
    #[must_use]
    pub fn reduce<'a>(&'a self, op: ReduceOp, value: Value<'a>) -> Value<'a> {
        self.check_value(value.state);
        self.push_value(Instruction::Reduce(op, value.instruction))
    }

    /// Selects between two values using a Boolean condition.
    #[must_use]
    pub fn select<'a>(
        &'a self,
        condition: BoolValue<'a>,
        accepted: Value<'a>,
        rejected: Value<'a>,
    ) -> Value<'a> {
        self.check_value(condition.state);
        self.check_value(accepted.state);
        self.check_value(rejected.state);
        self.push_value(Instruction::Select(
            condition.instruction,
            accepted.instruction,
            rejected.instruction,
        ))
    }

    /// Assigns a value to an output slot.
    pub fn output(&self, slot: u32, value: Value<'_>) {
        self.check_value(value.state);
        self.state
            .borrow_mut()
            .outputs
            .push((slot, value.instruction));
    }

    pub(crate) fn parts(&self) -> Result<ProgramParts> {
        let state = self.state.borrow();
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        Ok((self.kind, state.instructions.clone(), state.outputs.clone()))
    }

    pub(crate) fn definition(&self, rank: usize) -> Result<crate::sys::Program> {
        let (kind, instructions, outputs) = self.parts()?;
        let mut lowered = Vec::with_capacity(instructions.len());
        let mut values = Vec::with_capacity(instructions.len());
        for instruction in instructions {
            let instruction = match instruction {
                Instruction::Input(slot) => crate::sys::ProgramInst::Input(slot),
                Instruction::Constant(value) => crate::sys::ProgramInst::Constant(value),
                Instruction::Index(axis) => {
                    let index = push_lowered(
                        &mut lowered,
                        crate::sys::ProgramInst::Index(resolve_axis(axis, rank)?),
                    )?;
                    crate::sys::ProgramInst::CastF32(index)
                }
                Instruction::Extent(axis) => {
                    let extent = push_lowered(
                        &mut lowered,
                        crate::sys::ProgramInst::Extent(resolve_axis(axis, rank)?),
                    )?;
                    crate::sys::ProgramInst::CastF32(extent)
                }
                Instruction::Unary(op, value) => {
                    crate::sys::ProgramInst::Unary(op, lowered_value(&values, value)?)
                }
                Instruction::Binary(op, left, right) => crate::sys::ProgramInst::Binary(
                    op,
                    lowered_value(&values, left)?,
                    lowered_value(&values, right)?,
                ),
                Instruction::Select(condition, accepted, rejected) => {
                    crate::sys::ProgramInst::Select(
                        lowered_value(&values, condition)?,
                        lowered_value(&values, accepted)?,
                        lowered_value(&values, rejected)?,
                    )
                }
                Instruction::Reduce(op, value) => {
                    crate::sys::ProgramInst::Reduce(op, lowered_value(&values, value)?)
                }
            };
            values.push(push_lowered(&mut lowered, instruction)?);
        }
        let outputs = outputs
            .into_iter()
            .map(|(slot, value)| Ok((slot, lowered_value(&values, value)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(crate::sys::Program {
            kind,
            instructions: lowered,
            outputs,
        })
    }

    fn push_value(&self, instruction: Instruction) -> Value<'_> {
        let instruction = push(&self.state, instruction);
        Value {
            state: &self.state,
            instruction,
            marker: PhantomData,
        }
    }

    fn check_value(&self, state: &RefCell<State>) {
        if !std::ptr::eq(&raw const self.state, state) {
            record_error(&self.state, "program values belong to different builders");
        }
    }
}

type ProgramParts = (ProgramKind, Vec<Instruction>, Vec<(u32, u32)>);

fn resolve_axis(axis: i32, rank: usize) -> Result<u8> {
    let rank = i64::try_from(rank).map_err(|_| Error::new("tensor rank is too large"))?;
    let axis = i64::from(axis);
    let axis = if axis < 0 {
        rank.checked_add(axis)
    } else {
        Some(axis)
    }
    .filter(|&axis| axis >= 0 && axis < rank)
    .ok_or_else(|| Error::new("program axis is out of range"))?;
    u8::try_from(axis).map_err(|_| Error::new("program axis is out of range"))
}

fn push_lowered(
    instructions: &mut Vec<crate::sys::ProgramInst>,
    instruction: crate::sys::ProgramInst,
) -> Result<u32> {
    let index = u32::try_from(instructions.len())
        .map_err(|_| Error::new("program has too many instructions"))?;
    instructions.push(instruction);
    Ok(index)
}

fn lowered_value(values: &[u32], value: u32) -> Result<u32> {
    usize::try_from(value)
        .ok()
        .and_then(|value| values.get(value).copied())
        .ok_or_else(|| Error::new("program value is invalid"))
}

/// An f32 value produced by a program instruction.
#[derive(Clone, Copy)]
pub struct Value<'a> {
    state: &'a RefCell<State>,
    instruction: u32,
    marker: PhantomData<&'a Program>,
}

impl<'a> Value<'a> {
    /// Computes the absolute value.
    #[must_use]
    pub fn abs(self) -> Self {
        self.unary(UnaryOp::Abs)
    }

    /// Computes the base-e exponential.
    #[must_use]
    pub fn exp(self) -> Self {
        self.unary(UnaryOp::Exp)
    }

    /// Computes the natural logarithm.
    #[must_use]
    pub fn log(self) -> Self {
        self.unary(UnaryOp::Log)
    }

    /// Computes the square root.
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.unary(UnaryOp::Sqrt)
    }

    /// Computes the reciprocal square root.
    #[must_use]
    pub fn rsqrt(self) -> Self {
        self.unary(UnaryOp::Rsqrt)
    }

    /// Computes sine.
    #[must_use]
    pub fn sin(self) -> Self {
        self.unary(UnaryOp::Sin)
    }

    /// Computes cosine.
    #[must_use]
    pub fn cos(self) -> Self {
        self.unary(UnaryOp::Cos)
    }

    /// Computes hyperbolic tangent.
    #[must_use]
    pub fn tanh(self) -> Self {
        self.unary(UnaryOp::Tanh)
    }

    /// Computes the logistic sigmoid.
    #[must_use]
    pub fn sigmoid(self) -> Self {
        self.unary(UnaryOp::Sigmoid)
    }

    /// Computes the reciprocal.
    #[must_use]
    pub fn recip(self) -> Self {
        self.unary(UnaryOp::Recip)
    }

    /// Rounds toward negative infinity.
    #[must_use]
    pub fn floor(self) -> Self {
        self.unary(UnaryOp::Floor)
    }

    /// Computes the minimum.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        self.binary(BinaryOp::Min, other)
    }

    /// Computes the maximum.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        self.binary(BinaryOp::Max, other)
    }

    /// Raises this value to a power.
    #[must_use]
    pub fn pow(self, other: Self) -> Self {
        self.binary(BinaryOp::Pow, other)
    }

    /// Compares two values with `<`.
    #[must_use]
    pub fn lt(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Lt, other)
    }

    /// Compares two values with `<=`.
    #[must_use]
    pub fn le(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Le, other)
    }

    /// Compares two values for equality.
    #[must_use]
    pub fn equal(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Eq, other)
    }

    /// Compares two values for inequality.
    #[must_use]
    pub fn not_equal(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Ne, other)
    }

    /// Compares two values with `>=`.
    #[must_use]
    pub fn ge(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Ge, other)
    }

    /// Compares two values with `>`.
    #[must_use]
    pub fn gt(self, other: Self) -> BoolValue<'a> {
        self.compare(BinaryOp::Gt, other)
    }

    fn unary(self, op: UnaryOp) -> Self {
        Self {
            instruction: push(self.state, Instruction::Unary(op, self.instruction)),
            ..self
        }
    }

    fn binary(self, op: BinaryOp, other: Self) -> Self {
        check_same_state(self.state, other.state);
        Self {
            instruction: push(
                self.state,
                Instruction::Binary(op, self.instruction, other.instruction),
            ),
            ..self
        }
    }

    fn compare(self, op: BinaryOp, other: Self) -> BoolValue<'a> {
        check_same_state(self.state, other.state);
        let instruction = push(
            self.state,
            Instruction::Binary(op, self.instruction, other.instruction),
        );
        BoolValue {
            state: self.state,
            instruction,
            marker: PhantomData,
        }
    }

    fn scalar(self, op: BinaryOp, scalar: f32) -> Self {
        let scalar = Self {
            state: self.state,
            instruction: push(self.state, Instruction::Constant(scalar)),
            marker: PhantomData,
        };
        self.binary(op, scalar)
    }
}

/// A Boolean value produced by a comparison instruction.
#[derive(Clone, Copy)]
pub struct BoolValue<'a> {
    state: &'a RefCell<State>,
    instruction: u32,
    marker: PhantomData<&'a Program>,
}

impl<'a> BoolValue<'a> {
    /// Selects `accepted` when true and `rejected` otherwise.
    #[must_use]
    pub fn select(self, accepted: Value<'a>, rejected: Value<'a>) -> Value<'a> {
        check_same_state(self.state, accepted.state);
        check_same_state(self.state, rejected.state);
        Value {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Select(self.instruction, accepted.instruction, rejected.instruction),
            ),
            marker: PhantomData,
        }
    }
}

fn push(state: &RefCell<State>, instruction: Instruction) -> u32 {
    let mut state = state.borrow_mut();
    let index = u32::try_from(state.instructions.len()).unwrap_or(u32::MAX);
    state.instructions.push(instruction);
    index
}

fn check_same_state(left: &RefCell<State>, right: &RefCell<State>) {
    if !std::ptr::eq(left, right) {
        record_error(left, "program values belong to different builders");
    }
}

fn record_error(state: &RefCell<State>, message: &str) {
    let mut state = state.borrow_mut();
    if state.error.is_none() {
        state.error = Some(Error::new(message));
    }
}

macro_rules! value_operator {
    ($trait:ident, $method:ident, $op:ident) => {
        impl<'a> $trait for Value<'a> {
            type Output = Self;

            fn $method(self, other: Self) -> Self::Output {
                self.binary(BinaryOp::$op, other)
            }
        }

        impl<'a> $trait<f32> for Value<'a> {
            type Output = Self;

            fn $method(self, other: f32) -> Self::Output {
                self.scalar(BinaryOp::$op, other)
            }
        }
    };
}

value_operator!(Add, add, Add);
value_operator!(Sub, sub, Sub);
value_operator!(Mul, mul, Mul);
value_operator!(Div, div, Div);

impl Neg for Value<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_the_documented_softmax_shape() {
        let program = Program::row();
        let value = program.input(0);
        let maximum = program.reduce(ReduceOp::Max, value);
        let exponent = (value - maximum).exp();
        let sum = program.reduce(ReduceOp::Sum, exponent);
        program.output(0, exponent / sum);

        let (kind, instructions, outputs) = program.parts().unwrap();
        assert_eq!(kind, ProgramKind::Row);
        assert_eq!(instructions.len(), 6);
        assert_eq!(outputs, [(0, 5)]);
    }

    #[test]
    fn remembers_cross_program_value_errors() {
        let first = Program::map();
        let second = Program::map();
        first.output(0, second.constant(1.0));
        assert!(first.parts().is_err());
    }
}
