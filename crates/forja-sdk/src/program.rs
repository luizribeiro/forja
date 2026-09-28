//! Guest-side construction of validated scalar programs.

use std::{
    cell::RefCell,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use crate::{DType, Error, Result, sys};

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
pub(crate) enum ValueType {
    F32,
    U32,
    Bool,
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
    Cast(ValueType, u32),
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

/// A prepared scalar program with a fixed tensor signature.
#[derive(Clone)]
pub struct Kernel {
    pub(crate) handle: Rc<sys::Kernel>,
    pub(crate) rank: u8,
    pub(crate) inputs: Vec<DType>,
    pub(crate) outputs: Vec<DType>,
}

impl Kernel {
    /// Prepares a program for tensors with the given rank and element types.
    ///
    /// # Errors
    ///
    /// Returns an error when the program or signature is invalid, preparation
    /// fails, or the host refuses another live kernel.
    pub fn new(
        program: &Program,
        rank: u8,
        input_dtypes: &[DType],
        output_dtypes: &[DType],
    ) -> Result<Self> {
        let definition = program.definition(usize::from(rank))?;
        let handle = sys::create_kernel(definition, rank, input_dtypes, output_dtypes)?;
        Ok(Self {
            handle: Rc::new(handle),
            rank,
            inputs: input_dtypes.to_vec(),
            outputs: output_dtypes.to_vec(),
        })
    }
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
    pub fn input(&self, slot: u32) -> F32<'_> {
        self.push_f32(Instruction::Input(slot))
    }

    /// Loads a bound unsigned integer input slot.
    #[must_use]
    pub fn input_u32(&self, slot: u32) -> U32<'_> {
        self.push_u32(Instruction::Input(slot))
    }

    /// Creates an f32 constant.
    #[must_use]
    pub fn constant(&self, value: f32) -> F32<'_> {
        self.push_f32(Instruction::Constant(value))
    }

    /// Creates a Boolean constant.
    #[must_use]
    pub fn boolean(&self, value: bool) -> Bool<'_> {
        let value = self.push_f32(Instruction::Constant(f32::from(u8::from(value))));
        value.cast_bool()
    }

    /// Reads an axis coordinate as u32. Negative axes count from the end.
    #[must_use]
    pub fn index(&self, axis: i32) -> U32<'_> {
        self.push_u32(Instruction::Index(axis))
    }

    /// Reads an axis extent as u32. Negative axes count from the end.
    #[must_use]
    pub fn extent(&self, axis: i32) -> U32<'_> {
        self.push_u32(Instruction::Extent(axis))
    }

    /// Reduces a value across the last axis and broadcasts the result.
    #[must_use]
    pub fn reduce<'a>(&'a self, op: ReduceOp, value: F32<'a>) -> F32<'a> {
        self.check_value(value.state);
        self.push_f32(Instruction::Reduce(op, value.instruction))
    }

    /// Selects between two values using a Boolean condition.
    #[must_use]
    pub fn select<'a>(
        &'a self,
        condition: Bool<'a>,
        accepted: F32<'a>,
        rejected: F32<'a>,
    ) -> F32<'a> {
        self.check_value(condition.state);
        self.check_value(accepted.state);
        self.check_value(rejected.state);
        self.push_f32(Instruction::Select(
            condition.instruction,
            accepted.instruction,
            rejected.instruction,
        ))
    }

    /// Assigns a value to an output slot.
    pub fn output(&self, slot: u32, value: F32<'_>) {
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
                    crate::sys::ProgramInst::Index(resolve_axis(axis, rank)?)
                }
                Instruction::Extent(axis) => {
                    crate::sys::ProgramInst::Extent(resolve_axis(axis, rank)?)
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
                Instruction::Cast(to, value) => {
                    crate::sys::ProgramInst::Cast(to, lowered_value(&values, value)?)
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

    fn push_f32(&self, instruction: Instruction) -> F32<'_> {
        let instruction = push(&self.state, instruction);
        F32 {
            state: &self.state,
            instruction,
            marker: PhantomData,
        }
    }

    fn push_u32(&self, instruction: Instruction) -> U32<'_> {
        let instruction = push(&self.state, instruction);
        U32 {
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
pub struct F32<'a> {
    state: &'a RefCell<State>,
    instruction: u32,
    marker: PhantomData<&'a Program>,
}

#[doc(hidden)]
pub type Value<'a> = F32<'a>;

impl<'a> F32<'a> {
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
    pub fn lt(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Lt, other)
    }

    /// Compares two values with `<=`.
    #[must_use]
    pub fn le(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Le, other)
    }

    /// Compares two values for equality.
    #[must_use]
    pub fn equal(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Eq, other)
    }

    /// Compares two values for inequality.
    #[must_use]
    pub fn not_equal(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Ne, other)
    }

    /// Compares two values with `>=`.
    #[must_use]
    pub fn ge(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Ge, other)
    }

    /// Compares two values with `>`.
    #[must_use]
    pub fn gt(self, other: Self) -> Bool<'a> {
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

    fn compare(self, op: BinaryOp, other: Self) -> Bool<'a> {
        compare_values(op, self, other)
    }

    /// Casts this value to u32 using the IR's saturating conversion.
    #[must_use]
    pub fn cast_u32(self) -> U32<'a> {
        U32 {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Cast(ValueType::U32, self.instruction),
            ),
            marker: PhantomData,
        }
    }

    fn cast_bool(self) -> Bool<'a> {
        Bool {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Cast(ValueType::Bool, self.instruction),
            ),
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
pub struct Bool<'a> {
    state: &'a RefCell<State>,
    instruction: u32,
    marker: PhantomData<&'a Program>,
}

impl<'a> Bool<'a> {
    /// Selects `accepted` when true and `rejected` otherwise.
    #[must_use]
    pub fn select(self, accepted: F32<'a>, rejected: F32<'a>) -> F32<'a> {
        check_same_state(self.state, accepted.state);
        check_same_state(self.state, rejected.state);
        F32 {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Select(self.instruction, accepted.instruction, rejected.instruction),
            ),
            marker: PhantomData,
        }
    }

    /// Casts this condition to zero or one as u32.
    #[must_use]
    pub fn cast_u32(self) -> U32<'a> {
        U32 {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Cast(ValueType::U32, self.instruction),
            ),
            marker: PhantomData,
        }
    }
}

/// An unsigned integer value produced by a program instruction.
///
/// Integer arithmetic is available only through the explicit wrapping methods;
/// using Rust's `+`, `-`, or `*` operators is a type error.
///
/// ```compile_fail
/// use forja_sdk::program::Program;
/// let program = Program::map();
/// let value = program.index(0);
/// let _ = value + value;
/// ```
#[derive(Clone, Copy)]
pub struct U32<'a> {
    state: &'a RefCell<State>,
    instruction: u32,
    marker: PhantomData<&'a Program>,
}

impl<'a> U32<'a> {
    /// Adds with wrapping at the u32 boundary.
    #[must_use]
    pub fn wrapping_add(self, other: Self) -> Self {
        self.binary(BinaryOp::Add, other)
    }

    /// Subtracts with wrapping at the u32 boundary.
    #[must_use]
    pub fn wrapping_sub(self, other: Self) -> Self {
        self.binary(BinaryOp::Sub, other)
    }

    /// Multiplies with wrapping at the u32 boundary.
    #[must_use]
    pub fn wrapping_mul(self, other: Self) -> Self {
        self.binary(BinaryOp::Mul, other)
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

    /// Compares two values with `<`.
    #[must_use]
    pub fn lt(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Lt, other)
    }

    /// Compares two values with `<=`.
    #[must_use]
    pub fn le(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Le, other)
    }

    /// Compares two values for equality.
    #[must_use]
    pub fn equal(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Eq, other)
    }

    /// Compares two values for inequality.
    #[must_use]
    pub fn not_equal(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Ne, other)
    }

    /// Compares two values with `>=`.
    #[must_use]
    pub fn ge(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Ge, other)
    }

    /// Compares two values with `>`.
    #[must_use]
    pub fn gt(self, other: Self) -> Bool<'a> {
        self.compare(BinaryOp::Gt, other)
    }

    /// Casts this value to f32.
    #[must_use]
    pub fn cast_f32(self) -> F32<'a> {
        F32 {
            state: self.state,
            instruction: push(
                self.state,
                Instruction::Cast(ValueType::F32, self.instruction),
            ),
            marker: PhantomData,
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

    fn compare(self, op: BinaryOp, other: Self) -> Bool<'a> {
        compare_values(op, self, other)
    }
}

trait TypedValue<'a>: Copy {
    fn state(self) -> &'a RefCell<State>;
    fn instruction(self) -> u32;
    fn from_instruction(state: &'a RefCell<State>, instruction: u32) -> Self;
}

macro_rules! typed_value {
    ($value:ident) => {
        impl<'a> TypedValue<'a> for $value<'a> {
            fn state(self) -> &'a RefCell<State> {
                self.state
            }

            fn instruction(self) -> u32 {
                self.instruction
            }

            fn from_instruction(state: &'a RefCell<State>, instruction: u32) -> Self {
                Self {
                    state,
                    instruction,
                    marker: PhantomData,
                }
            }
        }
    };
}

typed_value!(F32);
typed_value!(U32);
typed_value!(Bool);

fn compare_values<'a, T: TypedValue<'a>>(op: BinaryOp, left: T, right: T) -> Bool<'a> {
    check_same_state(left.state(), right.state());
    Bool::from_instruction(
        left.state(),
        push(
            left.state(),
            Instruction::Binary(op, left.instruction(), right.instruction()),
        ),
    )
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
        impl<'a> $trait for F32<'a> {
            type Output = Self;

            fn $method(self, other: Self) -> Self::Output {
                self.binary(BinaryOp::$op, other)
            }
        }

        impl<'a> $trait<f32> for F32<'a> {
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

impl Neg for F32<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

impl<'a> Mul<f32> for U32<'a> {
    type Output = F32<'a>;

    fn mul(self, other: f32) -> Self::Output {
        self.cast_f32() * other
    }
}

impl<'a> Div<U32<'a>> for F32<'a> {
    type Output = Self;

    fn div(self, other: U32<'a>) -> Self::Output {
        self / other.cast_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys;

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

    #[test]
    fn rejects_cross_program_unsigned_arithmetic() {
        let first = Program::new(ProgramKind::Map);
        let second = Program::new(ProgramKind::Map);
        let _ = first.index(0).wrapping_add(second.index(0));

        assert!(first.parts().is_err());
    }

    #[test]
    fn lowers_typed_values_and_explicit_casts() {
        let program = Program::map();
        let input = program.input_u32(0);
        let lane = program.index(-1);
        let wrapped = input
            .wrapping_add(lane)
            .wrapping_sub(lane)
            .wrapping_mul(program.constant(1.0).cast_u32());
        let condition = wrapped.lt(program.extent(-1));
        program.output(
            0,
            condition.select(wrapped.cast_f32(), condition.cast_u32().cast_f32()),
        );

        let definition = program.definition(1).unwrap();
        assert!(matches!(
            definition.instructions[1],
            sys::ProgramInst::Index(0)
        ));
        assert!(matches!(
            definition.instructions[2],
            sys::ProgramInst::Binary(BinaryOp::Add, 0, 1)
        ));
        assert!(matches!(
            definition.instructions[5],
            sys::ProgramInst::Cast(ValueType::U32, 4)
        ));
        assert!(matches!(
            definition.instructions[12],
            sys::ProgramInst::Select(8, 9, 11)
        ));
    }

    #[cfg(feature = "native")]
    #[test]
    fn prepares_an_explicit_kernel_signature() {
        let program = Program::map();
        program.output(0, program.input(0));

        Kernel::new(&program, 2, &[DType::F32], &[DType::F32]).unwrap();
    }

    #[cfg(feature = "native")]
    #[test]
    fn prepares_an_unsigned_input_signature() {
        let program = Program::map();
        program.output(0, program.input_u32(0).cast_f32());

        Kernel::new(&program, 1, &[DType::U32], &[DType::F32]).unwrap();
    }
}
