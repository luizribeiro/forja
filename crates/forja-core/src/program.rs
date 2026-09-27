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
        typecheck(self)?;
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
    /// A unary operation received a non-floating-point value.
    InvalidUnaryType {
        /// The invalid instruction.
        instruction: usize,
    },
    /// A binary operation received incompatible value types.
    InvalidBinaryType {
        /// The invalid instruction.
        instruction: usize,
    },
    /// A select condition is not Boolean.
    InvalidSelectCondition {
        /// The invalid instruction.
        instruction: usize,
    },
    /// The two select alternatives have different value types.
    InvalidSelectBranch {
        /// The invalid instruction.
        instruction: usize,
    },
    /// A reduction operand is not floating-point.
    InvalidReduceType {
        /// The invalid instruction.
        instruction: usize,
    },
    /// An output value is not floating-point.
    InvalidOutputType {
        /// The output slot being written.
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

#[derive(Clone, Copy)]
struct TypeSet(u8);

impl TypeSet {
    const F32: Self = Self(1);
    const U32: Self = Self(2);
    const BOOL: Self = Self(4);
    const NUMERIC: Self = Self(Self::F32.0 | Self::U32.0);
    const ANY: Self = Self(Self::NUMERIC.0 | Self::BOOL.0);

    const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

struct TypeConstraints {
    parent: Vec<usize>,
    allowed: Vec<TypeSet>,
}

impl TypeConstraints {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            allowed: vec![TypeSet::ANY; len],
        }
    }

    fn root(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent == value {
            value
        } else {
            let root = self.root(parent);
            self.parent[value] = root;
            root
        }
    }

    fn constrain(&mut self, value: usize, allowed: TypeSet) -> bool {
        let root = self.root(value);
        let narrowed = self.allowed[root].intersect(allowed);
        self.allowed[root] = narrowed;
        !narrowed.is_empty()
    }

    fn unify(&mut self, left: usize, right: usize, allowed: TypeSet) -> bool {
        let left = self.root(left);
        let right = self.root(right);
        let narrowed = self.allowed[left]
            .intersect(self.allowed[right])
            .intersect(allowed);
        if narrowed.is_empty() {
            return false;
        }
        self.parent[right] = left;
        self.allowed[left] = narrowed;
        true
    }

    #[cfg(test)]
    fn resolve(mut self) -> Vec<ValueType> {
        (0..self.parent.len())
            .map(|value| {
                let root = self.root(value);
                let allowed = self.allowed[root];
                if !allowed.intersect(TypeSet::F32).is_empty() {
                    ValueType::F32
                } else if !allowed.intersect(TypeSet::U32).is_empty() {
                    ValueType::U32
                } else {
                    ValueType::Bool
                }
            })
            .collect()
    }
}

fn typecheck(program: &Program) -> Result<(), ProgramError> {
    type_constraints(program).map(|_| ())
}

fn type_constraints(program: &Program) -> Result<TypeConstraints, ProgramError> {
    let mut constraints = TypeConstraints::new(program.insts.len());
    let mut input_values = [None; MAX_INPUTS];
    for (instruction, &inst) in program.insts.iter().enumerate() {
        match inst {
            Inst::Input(slot) => {
                let slot = usize::try_from(slot).map_err(|_| ProgramError::TooManyInputs)?;
                if let Some(previous) = input_values[slot] {
                    constraints.unify(instruction, previous, TypeSet::ANY);
                } else {
                    input_values[slot] = Some(instruction);
                }
            }
            Inst::Const(_) => {
                constraints.constrain(instruction, TypeSet::F32);
            }
            Inst::Index(_) | Inst::Extent(_) => {
                constraints.constrain(instruction, TypeSet::U32);
            }
            Inst::Unary(_, operand) => {
                let operand = operand as usize;
                if !constraints.constrain(operand, TypeSet::F32)
                    || !constraints.constrain(instruction, TypeSet::F32)
                {
                    return Err(ProgramError::InvalidUnaryType { instruction });
                }
            }
            Inst::Binary(op, left, right) => {
                let operands = match op {
                    BinOp::Div | BinOp::Pow => TypeSet::F32,
                    _ => TypeSet::NUMERIC,
                };
                let result = match op {
                    BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                        TypeSet::BOOL
                    }
                    _ => operands,
                };
                if !constraints.unify(left as usize, right as usize, operands)
                    || !constraints.constrain(instruction, result)
                    || (result.0 != TypeSet::BOOL.0
                        && !constraints.unify(instruction, left as usize, operands))
                {
                    return Err(ProgramError::InvalidBinaryType { instruction });
                }
            }
            Inst::Select(condition, accepted, rejected) => {
                if !constraints.constrain(condition as usize, TypeSet::BOOL) {
                    return Err(ProgramError::InvalidSelectCondition { instruction });
                }
                if !constraints.unify(accepted as usize, rejected as usize, TypeSet::ANY)
                    || !constraints.unify(instruction, accepted as usize, TypeSet::ANY)
                {
                    return Err(ProgramError::InvalidSelectBranch { instruction });
                }
            }
            Inst::Cast(to, _) => {
                constraints.constrain(instruction, type_set(to));
            }
            Inst::Reduce(_, operand) => {
                if !constraints.constrain(operand as usize, TypeSet::F32)
                    || !constraints.constrain(instruction, TypeSet::F32)
                {
                    return Err(ProgramError::InvalidReduceType { instruction });
                }
            }
        }
    }
    for &(slot, instruction) in &program.outputs {
        if !constraints.constrain(instruction as usize, TypeSet::F32) {
            return Err(ProgramError::InvalidOutputType { slot });
        }
    }
    Ok(constraints)
}

const fn type_set(value_type: ValueType) -> TypeSet {
    match value_type {
        ValueType::F32 => TypeSet::F32,
        ValueType::U32 => TypeSet::U32,
        ValueType::Bool => TypeSet::BOOL,
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

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

    #[test]
    fn typechecks_arithmetic_casts_comparisons_and_selects() {
        program(
            vec![
                Inst::Index(0),
                Inst::Cast(ValueType::F32, 0),
                Inst::Const(1.0),
                Inst::Binary(BinOp::Lt, 1, 2),
                Inst::Select(3, 1, 2),
            ],
            vec![(0, 4)],
        )
        .validate()
        .unwrap();

        assert_eq!(
            program(
                vec![Inst::Index(0), Inst::Unary(UnOp::Neg, 0)],
                vec![(0, 1)],
            )
            .validate(),
            Err(ProgramError::InvalidUnaryType { instruction: 1 })
        );
        assert_eq!(
            program(
                vec![
                    Inst::Index(0),
                    Inst::Const(1.0),
                    Inst::Binary(BinOp::Add, 0, 1),
                ],
                vec![(0, 2)],
            )
            .validate(),
            Err(ProgramError::InvalidBinaryType { instruction: 2 })
        );
        assert_eq!(
            program(
                vec![
                    Inst::Index(0),
                    Inst::Index(0),
                    Inst::Binary(BinOp::Div, 0, 1)
                ],
                vec![(0, 2)],
            )
            .validate(),
            Err(ProgramError::InvalidBinaryType { instruction: 2 })
        );
    }

    #[test]
    fn rejects_invalid_select_reduction_and_output_types() {
        assert_eq!(
            program(
                vec![Inst::Const(0.0), Inst::Const(1.0), Inst::Select(0, 0, 1),],
                vec![(0, 2)],
            )
            .validate(),
            Err(ProgramError::InvalidSelectCondition { instruction: 2 })
        );
        assert_eq!(
            program(
                vec![
                    Inst::Input(0),
                    Inst::Unary(UnOp::Neg, 0),
                    Inst::Index(0),
                    Inst::Const(1.0),
                    Inst::Binary(BinOp::Lt, 3, 3),
                    Inst::Select(4, 1, 2),
                ],
                vec![(0, 1)],
            )
            .validate(),
            Err(ProgramError::InvalidSelectBranch { instruction: 5 })
        );
        let row = Program {
            kind: ProgramKind::Row,
            insts: vec![Inst::Index(0), Inst::Reduce(RedOp::Sum, 0)],
            outputs: vec![(0, 1)],
        };
        assert_eq!(
            row.validate(),
            Err(ProgramError::InvalidReduceType { instruction: 1 })
        );
        assert_eq!(
            program(vec![Inst::Index(0)], vec![(0, 0)]).validate(),
            Err(ProgramError::InvalidOutputType { slot: 0 })
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        #[test]
        fn garbage_instruction_lists_never_panic(
            candidate in arbitrary_program(),
        ) {
            let _ = candidate.validate();
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1))]

        #[test]
        fn biased_programs_exercise_accepted_invariants(
            candidates in prop::collection::vec(biased_program(), 1024..=1024),
        ) {
            let mut accepted = 0_usize;
            for candidate in &candidates {
                if candidate.validate().is_ok() {
                    accepted += 1;
                    prop_assert!(accepted_invariants_hold(candidate));
                }
            }
            let percentage = accepted * 100 / candidates.len();
            eprintln!(
                "biased validator acceptance: {accepted}/{} ({percentage}%)",
                candidates.len()
            );
            prop_assert!(percentage >= 50, "acceptance was {percentage}%");
        }
    }

    fn biased_program() -> impl Strategy<Value = Program> {
        (
            prop_oneof![Just(ProgramKind::Map), Just(ProgramKind::Row)],
            prop::collection::vec(any::<u64>(), 1..=64),
        )
            .prop_map(|(kind, words)| build_biased_program(kind, &words))
    }

    fn build_biased_program(kind: ProgramKind, words: &[u64]) -> Program {
        let mut insts = Vec::with_capacity(words.len());
        let mut values = [Vec::new(), Vec::new(), Vec::new()];
        let mut reductions = 0;
        for (index, &word) in words.iter().enumerate() {
            let (inst, value_type) = if index == 0 {
                (Inst::Const(finite_value(word)), ValueType::F32)
            } else {
                biased_inst(kind, word, &values, &mut reductions)
            };
            insts.push(inst);
            values[type_bucket(value_type)].push(u32::try_from(index).unwrap());
        }
        let output = choose_value(&values[type_bucket(ValueType::F32)], words[0]);
        Program {
            kind,
            insts,
            outputs: vec![(0, output)],
        }
    }

    fn biased_inst(
        kind: ProgramKind,
        word: u64,
        values: &[Vec<u32>; 3],
        reductions: &mut usize,
    ) -> (Inst, ValueType) {
        let floats = &values[type_bucket(ValueType::F32)];
        let integers = &values[type_bucket(ValueType::U32)];
        let booleans = &values[type_bucket(ValueType::Bool)];
        match word % 10 {
            0 => (Inst::Input(0), ValueType::F32),
            1 => (Inst::Const(finite_value(word)), ValueType::F32),
            2 => (Inst::Index(valid_axis(word)), ValueType::U32),
            3 => (Inst::Extent(valid_axis(word)), ValueType::U32),
            4 => (
                Inst::Unary(biased_unop(word), choose_value(floats, word >> 8)),
                ValueType::F32,
            ),
            5 => (
                Inst::Binary(
                    biased_float_binop(word),
                    choose_value(floats, word >> 8),
                    choose_value(floats, word >> 16),
                ),
                ValueType::F32,
            ),
            6 if !integers.is_empty() => (
                Inst::Binary(
                    biased_integer_binop(word),
                    choose_value(integers, word >> 8),
                    choose_value(integers, word >> 16),
                ),
                ValueType::U32,
            ),
            7 => (
                Inst::Binary(
                    biased_comparison(word),
                    choose_value(floats, word >> 8),
                    choose_value(floats, word >> 16),
                ),
                ValueType::Bool,
            ),
            8 if !booleans.is_empty() => (
                Inst::Select(
                    choose_value(booleans, word >> 8),
                    choose_value(floats, word >> 16),
                    choose_value(floats, word >> 24),
                ),
                ValueType::F32,
            ),
            9 if kind == ProgramKind::Row && *reductions < MAX_REDUCTIONS => {
                *reductions += 1;
                (
                    Inst::Reduce(biased_redop(word), choose_value(floats, word >> 8)),
                    ValueType::F32,
                )
            }
            _ => (
                Inst::Cast(ValueType::F32, choose_value(floats, word >> 8)),
                ValueType::F32,
            ),
        }
    }

    fn accepted_invariants_hold(program: &Program) -> bool {
        let operands_are_backward =
            program
                .insts
                .iter()
                .copied()
                .enumerate()
                .all(|(index, inst)| {
                    inst.operands()
                        .into_iter()
                        .flatten()
                        .all(|operand| u64::from(operand) < index as u64)
                });
        let kind_is_respected = program.kind == ProgramKind::Row
            || program
                .insts
                .iter()
                .all(|inst| !matches!(inst, Inst::Reduce(_, _)));
        let reduction_count = program
            .insts
            .iter()
            .filter(|inst| matches!(inst, Inst::Reduce(_, _)))
            .count();
        let Ok(constraints) = type_constraints(program) else {
            return false;
        };
        let types = constraints.resolve();
        program.insts.len() <= MAX_INSTRUCTIONS
            && operands_are_backward
            && kind_is_respected
            && reduction_count <= MAX_REDUCTIONS
            && assignment_is_consistent(program, &types)
    }

    fn choose_value(values: &[u32], word: u64) -> u32 {
        let len = u64::try_from(values.len()).unwrap();
        let index = usize::try_from(word % len).unwrap();
        values[index]
    }

    fn finite_value(word: u64) -> f32 {
        f32::from(u16::try_from(word % 2048).unwrap()) / 16.0
    }

    fn valid_axis(word: u64) -> u8 {
        let rank = u64::try_from(MAX_RANK).unwrap();
        u8::try_from(word % rank).unwrap()
    }

    const fn type_bucket(value_type: ValueType) -> usize {
        match value_type {
            ValueType::F32 => 0,
            ValueType::U32 => 1,
            ValueType::Bool => 2,
        }
    }

    const fn biased_unop(word: u64) -> UnOp {
        match word % 12 {
            0 => UnOp::Neg,
            1 => UnOp::Abs,
            2 => UnOp::Exp,
            3 => UnOp::Log,
            4 => UnOp::Sqrt,
            5 => UnOp::Rsqrt,
            6 => UnOp::Sin,
            7 => UnOp::Cos,
            8 => UnOp::Tanh,
            9 => UnOp::Sigmoid,
            10 => UnOp::Recip,
            _ => UnOp::Floor,
        }
    }

    const fn biased_float_binop(word: u64) -> BinOp {
        match word % 7 {
            0 => BinOp::Add,
            1 => BinOp::Sub,
            2 => BinOp::Mul,
            3 => BinOp::Div,
            4 => BinOp::Min,
            5 => BinOp::Max,
            _ => BinOp::Pow,
        }
    }

    const fn biased_integer_binop(word: u64) -> BinOp {
        match word % 5 {
            0 => BinOp::Add,
            1 => BinOp::Sub,
            2 => BinOp::Mul,
            3 => BinOp::Min,
            _ => BinOp::Max,
        }
    }

    const fn biased_comparison(word: u64) -> BinOp {
        match word % 6 {
            0 => BinOp::Lt,
            1 => BinOp::Le,
            2 => BinOp::Eq,
            3 => BinOp::Ne,
            4 => BinOp::Ge,
            _ => BinOp::Gt,
        }
    }

    const fn biased_redop(word: u64) -> RedOp {
        match word % 3 {
            0 => RedOp::Sum,
            1 => RedOp::Max,
            _ => RedOp::Min,
        }
    }

    fn arbitrary_program() -> impl Strategy<Value = Program> {
        (
            prop_oneof![Just(ProgramKind::Map), Just(ProgramKind::Row)],
            prop::collection::vec(arbitrary_inst(), 0..=260),
            prop::collection::vec((0_u32..=5, 0_u32..=260), 0..=6),
        )
            .prop_map(|(kind, insts, outputs)| Program {
                kind,
                insts,
                outputs,
            })
    }

    fn arbitrary_inst() -> impl Strategy<Value = Inst> {
        prop_oneof![
            (0_u32..=9).prop_map(Inst::Input),
            any::<f32>().prop_map(Inst::Const),
            (0_u8..=9).prop_map(Inst::Index),
            (0_u8..=9).prop_map(Inst::Extent),
            (arbitrary_unop(), 0_u32..=260).prop_map(|(op, a)| Inst::Unary(op, a)),
            (arbitrary_binop(), 0_u32..=260, 0_u32..=260)
                .prop_map(|(op, a, b)| Inst::Binary(op, a, b)),
            (0_u32..=260, 0_u32..=260, 0_u32..=260)
                .prop_map(|(condition, a, b)| Inst::Select(condition, a, b)),
            (arbitrary_type(), 0_u32..=260).prop_map(|(to, a)| Inst::Cast(to, a)),
            (arbitrary_redop(), 0_u32..=260).prop_map(|(op, a)| Inst::Reduce(op, a)),
        ]
    }

    fn arbitrary_unop() -> impl Strategy<Value = UnOp> {
        prop::sample::select(vec![
            UnOp::Neg,
            UnOp::Abs,
            UnOp::Exp,
            UnOp::Log,
            UnOp::Sqrt,
            UnOp::Rsqrt,
            UnOp::Sin,
            UnOp::Cos,
            UnOp::Tanh,
            UnOp::Sigmoid,
            UnOp::Recip,
            UnOp::Floor,
        ])
    }

    fn arbitrary_binop() -> impl Strategy<Value = BinOp> {
        prop::sample::select(vec![
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Min,
            BinOp::Max,
            BinOp::Pow,
            BinOp::Lt,
            BinOp::Le,
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Ge,
            BinOp::Gt,
        ])
    }

    fn arbitrary_redop() -> impl Strategy<Value = RedOp> {
        prop::sample::select(vec![RedOp::Sum, RedOp::Max, RedOp::Min])
    }

    fn arbitrary_type() -> impl Strategy<Value = ValueType> {
        prop::sample::select(vec![ValueType::F32, ValueType::U32, ValueType::Bool])
    }

    fn assignment_is_consistent(program: &Program, types: &[ValueType]) -> bool {
        program.insts.iter().enumerate().all(|(index, inst)| {
            let result = types[index];
            match *inst {
                Inst::Input(_) => true,
                Inst::Const(_) => result == ValueType::F32,
                Inst::Index(_) | Inst::Extent(_) => result == ValueType::U32,
                Inst::Unary(_, a) | Inst::Reduce(_, a) => {
                    result == ValueType::F32 && types[a as usize] == result
                }
                Inst::Binary(op, a, b) => {
                    let left = types[a as usize];
                    let right = types[b as usize];
                    let comparison = matches!(
                        op,
                        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt
                    );
                    left == right
                        && matches!(left, ValueType::F32 | ValueType::U32)
                        && (!matches!(op, BinOp::Div | BinOp::Pow) || left == ValueType::F32)
                        && result == if comparison { ValueType::Bool } else { left }
                }
                Inst::Select(condition, a, b) => {
                    types[condition as usize] == ValueType::Bool
                        && types[a as usize] == types[b as usize]
                        && result == types[a as usize]
                }
                Inst::Cast(to, _) => result == to,
            }
        }) && program
            .outputs
            .iter()
            .all(|&(_, instruction)| types[instruction as usize] == ValueType::F32)
    }
}
