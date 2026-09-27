//! Validated scalar programs over tensor iteration spaces.

use std::{error::Error, fmt};

use crate::MAX_RANK;

/// The largest accepted instruction count.
pub const MAX_INSTRUCTIONS: usize = 256;
/// The largest accepted reduction count.
pub const MAX_REDUCTIONS: usize = 4;
/// The largest accepted input slot count.
pub const MAX_INPUTS: usize = 8;
/// The largest accepted output slot count.
pub const MAX_OUTPUTS: usize = 4;

/// The iteration strategy for a scalar program.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramKind {
    /// Evaluates each output element independently.
    Map,
    /// Evaluates rows and permits reductions over their last axis.
    Row,
}

/// A scalar value type produced by an instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    /// A 32-bit floating-point value.
    F32,
    /// An unsigned 32-bit integer value.
    U32,
    /// A Boolean value.
    Bool,
}

/// A unary scalar operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnOp {
    /// Negation.
    Neg,
    /// Absolute value.
    Abs,
    /// Base-e exponentiation.
    Exp,
    /// Natural logarithm.
    Log,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid.
    Sigmoid,
    /// Reciprocal.
    Recip,
    /// Round toward negative infinity.
    Floor,
}

/// A binary scalar operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Floating-point division.
    Div,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Exponentiation.
    Pow,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than or equal.
    Ge,
    /// Greater than.
    Gt,
}

/// A row reduction operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RedOp {
    /// Sum.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

/// One instruction in a flat SSA program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inst {
    /// Loads an input slot at the current logical index.
    Input(u32),
    /// Produces a floating-point constant.
    Const(f32),
    /// Produces the coordinate along an axis.
    Index(u8),
    /// Produces the extent of an axis.
    Extent(u8),
    /// Applies a unary operation.
    Unary(UnOp, u32),
    /// Applies a binary operation.
    Binary(BinOp, u32, u32),
    /// Chooses between two values using a Boolean condition.
    Select(u32, u32, u32),
    /// Converts a value to another scalar type.
    Cast(ValueType, u32),
    /// Reduces a floating-point value over a row.
    Reduce(RedOp, u32),
}

impl Inst {
    fn operands(self) -> [Option<u32>; 3] {
        match self {
            Self::Unary(_, a) | Self::Cast(_, a) | Self::Reduce(_, a) => [Some(a), None, None],
            Self::Binary(_, a, b) => [Some(a), Some(b), None],
            Self::Select(condition, a, b) => [Some(condition), Some(a), Some(b)],
            Self::Input(_) | Self::Const(_) | Self::Index(_) | Self::Extent(_) => {
                [None, None, None]
            }
        }
    }
}

/// A scalar program before validation.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    /// The program's iteration strategy.
    pub kind: ProgramKind,
    /// Instructions in evaluation order.
    pub insts: Vec<Inst>,
    /// Pairs of output slots and instruction indices to store.
    pub outputs: Vec<(u32, u32)>,
}

impl Program {
    /// Validates structural safety and resource limits.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError`] when an operand is not backward, a slot is
    /// missing or repeated, or a resource limit is exceeded.
    pub fn validate(&self) -> Result<ValidatedProgram, ProgramError> {
        let (input_count, output_count) = validate_structure(self)?;
        Ok(ValidatedProgram {
            program: self.clone(),
            input_count,
            output_count,
        })
    }
}

/// A program whose structure and resource use have been validated.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedProgram {
    program: Program,
    input_count: usize,
    output_count: usize,
}

impl ValidatedProgram {
    /// Returns the validated source program.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Returns the number of input slots used by the program.
    #[must_use]
    pub const fn input_count(&self) -> usize {
        self.input_count
    }

    /// Returns the number of output slots written by the program.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }
}

/// A reason a scalar program failed validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgramError {
    /// The instruction limit was exceeded.
    TooManyInstructions,
    /// The reduction limit was exceeded.
    TooManyReductions,
    /// An input slot exceeds the supported range.
    TooManyInputs,
    /// An output slot exceeds the supported range.
    TooManyOutputs,
    /// The program does not write any output.
    NoOutputs,
    /// An operand does not refer to an earlier instruction.
    OperandNotEarlier {
        /// The instruction containing the operand.
        instruction: usize,
        /// The invalid operand index.
        operand: u32,
    },
    /// An output refers to a nonexistent instruction.
    InvalidOutputValue {
        /// The output slot being written.
        slot: u32,
        /// The invalid instruction index.
        instruction: u32,
    },
    /// A floating-point constant is not finite.
    NonFiniteConstant {
        /// The instruction containing the constant.
        instruction: usize,
    },
    /// An axis exceeds the maximum tensor rank.
    AxisOutOfRange {
        /// The instruction containing the axis.
        instruction: usize,
        /// The invalid axis.
        axis: u8,
    },
    /// A map program contains a reduction.
    ReduceInMap {
        /// The reduction instruction.
        instruction: usize,
    },
    /// An input slot below the largest used slot is unused.
    MissingInputSlot {
        /// The unused slot.
        slot: u32,
    },
    /// An output slot is written more than once.
    DuplicateOutputSlot {
        /// The repeated slot.
        slot: u32,
    },
    /// An output slot below the largest written slot is unwritten.
    MissingOutputSlot {
        /// The unwritten slot.
        slot: u32,
    },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid program: {self:?}")
    }
}

impl Error for ProgramError {}

fn validate_structure(program: &Program) -> Result<(usize, usize), ProgramError> {
    if program.insts.len() > MAX_INSTRUCTIONS {
        return Err(ProgramError::TooManyInstructions);
    }

    let mut inputs = [false; MAX_INPUTS];
    let mut reductions = 0_usize;
    for (instruction, &inst) in program.insts.iter().enumerate() {
        for operand in inst.operands().into_iter().flatten() {
            if u64::from(operand) >= instruction as u64 {
                return Err(ProgramError::OperandNotEarlier {
                    instruction,
                    operand,
                });
            }
        }
        match inst {
            Inst::Input(slot) => {
                let Ok(slot) = usize::try_from(slot) else {
                    return Err(ProgramError::TooManyInputs);
                };
                if slot >= MAX_INPUTS {
                    return Err(ProgramError::TooManyInputs);
                }
                inputs[slot] = true;
            }
            Inst::Const(value) if !value.is_finite() => {
                return Err(ProgramError::NonFiniteConstant { instruction });
            }
            Inst::Index(axis) | Inst::Extent(axis) if usize::from(axis) >= MAX_RANK => {
                return Err(ProgramError::AxisOutOfRange { instruction, axis });
            }
            Inst::Reduce(_, _) => {
                if program.kind == ProgramKind::Map {
                    return Err(ProgramError::ReduceInMap { instruction });
                }
                reductions += 1;
                if reductions > MAX_REDUCTIONS {
                    return Err(ProgramError::TooManyReductions);
                }
            }
            _ => {}
        }
    }
    let input_count =
        contiguous_slot_count(&inputs, |slot| ProgramError::MissingInputSlot { slot })?;

    if program.outputs.is_empty() {
        return Err(ProgramError::NoOutputs);
    }
    let mut outputs = [false; MAX_OUTPUTS];
    for &(slot, instruction) in &program.outputs {
        let Ok(slot_index) = usize::try_from(slot) else {
            return Err(ProgramError::TooManyOutputs);
        };
        if slot_index >= MAX_OUTPUTS {
            return Err(ProgramError::TooManyOutputs);
        }
        if outputs[slot_index] {
            return Err(ProgramError::DuplicateOutputSlot { slot });
        }
        if u64::from(instruction) >= program.insts.len() as u64 {
            return Err(ProgramError::InvalidOutputValue { slot, instruction });
        }
        outputs[slot_index] = true;
    }
    let output_count =
        contiguous_slot_count(&outputs, |slot| ProgramError::MissingOutputSlot { slot })?;
    Ok((input_count, output_count))
}

fn contiguous_slot_count<const N: usize>(
    slots: &[bool; N],
    missing: impl FnOnce(u32) -> ProgramError + Copy,
) -> Result<usize, ProgramError> {
    let count = slots
        .iter()
        .rposition(|&used| used)
        .map_or(0, |slot| slot + 1);
    for (slot, &used) in slots[..count].iter().enumerate() {
        if !used {
            return Err(missing(u32::try_from(slot).unwrap_or(u32::MAX)));
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(insts: Vec<Inst>, outputs: Vec<(u32, u32)>) -> Program {
        Program {
            kind: ProgramKind::Map,
            insts,
            outputs,
        }
    }

    #[test]
    fn validates_structural_limits_and_slot_counts() {
        let validated = program(
            vec![
                Inst::Input(1),
                Inst::Input(0),
                Inst::Binary(BinOp::Add, 0, 1),
            ],
            vec![(0, 2)],
        )
        .validate()
        .unwrap();
        assert_eq!(validated.input_count(), 2);
        assert_eq!(validated.output_count(), 1);

        assert_eq!(
            program(vec![Inst::Const(0.0)], vec![]).validate(),
            Err(ProgramError::NoOutputs)
        );

        assert_eq!(
            program(vec![Inst::Input(1)], vec![(0, 0)]).validate(),
            Err(ProgramError::MissingInputSlot { slot: 0 })
        );
        assert_eq!(
            program(vec![Inst::Const(0.0)], vec![(1, 0)]).validate(),
            Err(ProgramError::MissingOutputSlot { slot: 0 })
        );
        assert_eq!(
            program(vec![Inst::Const(0.0)], vec![(0, 0), (0, 0)]).validate(),
            Err(ProgramError::DuplicateOutputSlot { slot: 0 })
        );
    }

    #[test]
    fn rejects_non_backward_operands_and_invalid_outputs() {
        assert_eq!(
            program(vec![Inst::Unary(UnOp::Neg, 0)], vec![]).validate(),
            Err(ProgramError::OperandNotEarlier {
                instruction: 0,
                operand: 0,
            })
        );
        assert_eq!(
            program(vec![Inst::Const(0.0)], vec![(0, 1)]).validate(),
            Err(ProgramError::InvalidOutputValue {
                slot: 0,
                instruction: 1,
            })
        );
    }

    #[test]
    fn rejects_invalid_constants_axes_and_reductions() {
        assert_eq!(
            program(vec![Inst::Const(f32::INFINITY)], vec![]).validate(),
            Err(ProgramError::NonFiniteConstant { instruction: 0 })
        );
        assert_eq!(
            program(vec![Inst::Index(8)], vec![]).validate(),
            Err(ProgramError::AxisOutOfRange {
                instruction: 0,
                axis: 8,
            })
        );
        assert_eq!(
            program(vec![Inst::Const(1.0), Inst::Reduce(RedOp::Sum, 0)], vec![],).validate(),
            Err(ProgramError::ReduceInMap { instruction: 1 })
        );
    }

    #[test]
    fn rejects_each_resource_cap() {
        assert_eq!(
            program(vec![Inst::Const(0.0); MAX_INSTRUCTIONS + 1], vec![]).validate(),
            Err(ProgramError::TooManyInstructions)
        );
        assert_eq!(
            program(vec![Inst::Input(8)], vec![]).validate(),
            Err(ProgramError::TooManyInputs)
        );
        assert_eq!(
            program(vec![Inst::Const(0.0)], vec![(4, 0)]).validate(),
            Err(ProgramError::TooManyOutputs)
        );

        let mut row = Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Const(0.0)],
            outputs: vec![],
        };
        row.insts
            .extend((0..=MAX_REDUCTIONS).map(|_| Inst::Reduce(RedOp::Sum, 0)));
        assert_eq!(row.validate(), Err(ProgramError::TooManyReductions));
    }
}
