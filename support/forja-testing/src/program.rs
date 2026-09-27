//! Well-typed scalar-program generation for backend differential tests.

use forja_core::{
    DType,
    program::{
        BinOp, Inst, MAX_INSTRUCTIONS, MAX_REDUCTIONS, Program, ProgramKind, RedOp, UnOp, ValueType,
    },
};
use proptest::prelude::*;

use crate::TensorSpec;

const BASE_MAP_LEN: usize = 11;
const BASE_ROW_LEN: usize = 12;

/// A generated program and concrete tensor-view signature that accepts it.
#[derive(Clone, Debug)]
pub struct ProgramCase {
    program: Program,
    shape: Vec<u32>,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

impl ProgramCase {
    /// Returns the unvalidated program.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Returns the iteration shape.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Returns input allocation and view specifications in slot order.
    #[must_use]
    pub fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }

    /// Returns output allocation specifications in slot order.
    #[must_use]
    pub fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }
}

/// Generates valid map and row programs up to a caller-selected instruction count.
///
/// Every case contains input, constant, index, extent, unary, binary, select,
/// and cast instructions. Row cases also contain reductions. Inputs independently
/// use contiguous, broadcast, or permuted views; shapes cover ranks one through
/// four and include last-axis length 4097. Each program writes one to four outputs.
pub fn well_typed_programs(max_instructions: usize) -> impl Strategy<Value = ProgramCase> {
    let max_instructions = max_instructions.clamp(BASE_ROW_LEN, MAX_INSTRUCTIONS);
    let word_count = max_instructions.max(16);
    (
        any::<bool>(),
        any::<u64>(),
        1_usize..=4,
        prop::collection::vec(any::<u64>(), word_count..=word_count),
    )
        .prop_map(move |(row, shape_word, output_count, words)| {
            build_case(row, shape_word, output_count, &words, max_instructions)
        })
}

/// Builds one deterministic well-typed case from a seed.
#[must_use]
pub fn case_from_seed(seed: u64, max_instructions: usize) -> ProgramCase {
    let max_instructions = max_instructions.clamp(BASE_ROW_LEN, MAX_INSTRUCTIONS);
    let mut state = seed.max(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let row = next() & 1 != 0;
    let shape_word = next();
    let output_count = usize::try_from(next() % 4 + 1).unwrap_or(1);
    let words = (0..max_instructions.max(16))
        .map(|_| next())
        .collect::<Vec<_>>();
    build_case(row, shape_word, output_count, &words, max_instructions)
}

fn build_case(
    row: bool,
    shape_word: u64,
    output_count: usize,
    words: &[u64],
    max_instructions: usize,
) -> ProgramCase {
    let kind = if row {
        ProgramKind::Row
    } else {
        ProgramKind::Map
    };
    let shape = generated_shape(shape_word);
    let program = generated_program(kind, output_count, words, max_instructions);
    let inputs = vec![
        generated_input(float_dtype(words[0]), &shape, words[1]),
        generated_input(DType::U32, &shape, words[2]),
    ];
    let outputs = (0..output_count)
        .map(|slot| TensorSpec::contiguous(float_dtype(words[3 + slot]), &shape))
        .collect();
    ProgramCase {
        program,
        shape,
        inputs,
        outputs,
    }
}

fn generated_shape(word: u64) -> Vec<u32> {
    match word % 10 {
        0 => vec![1],
        1 => vec![7],
        2 => vec![33],
        3 => vec![4097],
        4 => vec![2, 7],
        5 => vec![7, 33],
        6 => vec![1, 4097],
        7 => vec![2, 3, 7],
        8 => vec![1, 2, 33],
        _ => vec![1, 2, 1, 7],
    }
}

fn generated_input(dtype: DType, shape: &[u32], word: u64) -> TensorSpec {
    match word % 3 {
        0 => TensorSpec::contiguous(dtype, shape),
        1 => {
            let allocation = shape
                .iter()
                .enumerate()
                .map(|(axis, &extent)| {
                    if (word >> (axis % 8)) & 1 == 0 {
                        1
                    } else {
                        extent
                    }
                })
                .collect::<Vec<_>>();
            TensorSpec::broadcast(dtype, &allocation, shape)
        }
        _ => {
            let rank = shape.len();
            let shift = usize::try_from(word).unwrap_or(0) % rank;
            let mut axes = (0..rank)
                .map(|axis| u8::try_from(axis).unwrap_or(0))
                .collect::<Vec<_>>();
            axes.rotate_left(shift);
            if word & 4 != 0 {
                axes.reverse();
            }
            let mut allocation = vec![0; rank];
            for (output_axis, &input_axis) in axes.iter().enumerate() {
                allocation[usize::from(input_axis)] = shape[output_axis];
            }
            TensorSpec::permuted(dtype, &allocation, &axes)
        }
    }
}

fn generated_program(
    kind: ProgramKind,
    output_count: usize,
    words: &[u64],
    max_instructions: usize,
) -> Program {
    let mut insts = Vec::with_capacity(max_instructions);
    let mut floats = Vec::new();
    let mut integers = Vec::new();
    let mut booleans = Vec::new();
    push(&mut insts, &mut floats, Inst::Input(0), ValueType::F32);
    push(&mut insts, &mut integers, Inst::Input(1), ValueType::U32);
    push(&mut insts, &mut floats, Inst::Const(0.5), ValueType::F32);
    push(&mut insts, &mut integers, Inst::Index(0), ValueType::U32);
    push(&mut insts, &mut integers, Inst::Extent(0), ValueType::U32);
    push(
        &mut insts,
        &mut floats,
        Inst::Unary(unop(words[5]), 0),
        ValueType::F32,
    );
    push(
        &mut insts,
        &mut floats,
        Inst::Binary(float_binop(words[6]), 0, 2),
        ValueType::F32,
    );
    push(
        &mut insts,
        &mut integers,
        Inst::Binary(integer_binop(words[7]), 1, 3),
        ValueType::U32,
    );
    push(
        &mut insts,
        &mut booleans,
        Inst::Binary(comparison(words[8]), 0, 2),
        ValueType::Bool,
    );
    push(
        &mut insts,
        &mut floats,
        Inst::Select(8, 5, 6),
        ValueType::F32,
    );
    push(
        &mut insts,
        &mut floats,
        Inst::Cast(ValueType::F32, 7),
        ValueType::F32,
    );
    let mut reductions = 0;
    if kind == ProgramKind::Row {
        push(
            &mut insts,
            &mut floats,
            Inst::Reduce(redop(words[9]), 9),
            ValueType::F32,
        );
        reductions = 1;
    }

    let base = if kind == ProgramKind::Row {
        BASE_ROW_LEN
    } else {
        BASE_MAP_LEN
    };
    let extra = usize::try_from(words[10]).unwrap_or(0) % (max_instructions - base + 1);
    for &word in words.iter().skip(11).take(extra) {
        push_random(
            kind,
            word,
            &mut insts,
            &mut floats,
            &mut integers,
            &mut booleans,
            &mut reductions,
        );
    }
    let outputs = (0..output_count)
        .map(|slot| {
            (
                u32::try_from(slot).unwrap_or(0),
                choose(&floats, words[11 + slot]),
            )
        })
        .collect();
    Program {
        kind,
        insts,
        outputs,
    }
}

fn push_random(
    kind: ProgramKind,
    word: u64,
    insts: &mut Vec<Inst>,
    floats: &mut Vec<u32>,
    integers: &mut Vec<u32>,
    booleans: &mut Vec<u32>,
    reductions: &mut usize,
) {
    match word % 11 {
        0 => push(insts, floats, Inst::Input(0), ValueType::F32),
        1 => push(insts, integers, Inst::Input(1), ValueType::U32),
        2 => push(insts, floats, Inst::Const(finite(word)), ValueType::F32),
        3 => push(insts, integers, Inst::Index(0), ValueType::U32),
        4 => push(insts, integers, Inst::Extent(0), ValueType::U32),
        5 => push(
            insts,
            floats,
            Inst::Unary(unop(word), choose(floats, word >> 8)),
            ValueType::F32,
        ),
        6 => push(
            insts,
            floats,
            Inst::Binary(
                float_binop(word),
                choose(floats, word >> 8),
                choose(floats, word >> 16),
            ),
            ValueType::F32,
        ),
        7 => push(
            insts,
            integers,
            Inst::Binary(
                integer_binop(word),
                choose(integers, word >> 8),
                choose(integers, word >> 16),
            ),
            ValueType::U32,
        ),
        8 => push(
            insts,
            booleans,
            Inst::Binary(
                comparison(word),
                choose(floats, word >> 8),
                choose(floats, word >> 16),
            ),
            ValueType::Bool,
        ),
        9 => push(
            insts,
            floats,
            Inst::Select(
                choose(booleans, word >> 8),
                choose(floats, word >> 16),
                choose(floats, word >> 24),
            ),
            ValueType::F32,
        ),
        _ if kind == ProgramKind::Row && *reductions < MAX_REDUCTIONS => {
            *reductions += 1;
            push(
                insts,
                floats,
                Inst::Reduce(redop(word), choose(floats, word >> 8)),
                ValueType::F32,
            );
        }
        _ => push(
            insts,
            floats,
            Inst::Cast(ValueType::F32, choose(integers, word >> 8)),
            ValueType::F32,
        ),
    }
}

fn push(insts: &mut Vec<Inst>, values: &mut Vec<u32>, inst: Inst, _: ValueType) {
    values.push(u32::try_from(insts.len()).unwrap_or(0));
    insts.push(inst);
}

fn choose(values: &[u32], word: u64) -> u32 {
    let divisor = u64::try_from(values.len()).unwrap_or(1);
    values[usize::try_from(word % divisor).unwrap_or(0)]
}

fn float_dtype(word: u64) -> DType {
    match word % 3 {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    }
}

fn finite(word: u64) -> f32 {
    f32::from(u16::try_from(word % 2048).unwrap_or(0)) / 16.0 - 64.0
}

fn unop(word: u64) -> UnOp {
    const OPS: [UnOp; 12] = [
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
    ];
    OPS[usize::try_from(word % 12).unwrap_or(0)]
}

fn float_binop(word: u64) -> BinOp {
    const OPS: [BinOp; 7] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Min,
        BinOp::Max,
        BinOp::Pow,
    ];
    OPS[usize::try_from(word % 7).unwrap_or(0)]
}

fn integer_binop(word: u64) -> BinOp {
    const OPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Min, BinOp::Max];
    OPS[usize::try_from(word % 5).unwrap_or(0)]
}

fn comparison(word: u64) -> BinOp {
    const OPS: [BinOp; 6] = [
        BinOp::Lt,
        BinOp::Le,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::Ge,
        BinOp::Gt,
    ];
    OPS[usize::try_from(word % 6).unwrap_or(0)]
}

fn redop(word: u64) -> RedOp {
    const OPS: [RedOp; 3] = [RedOp::Sum, RedOp::Max, RedOp::Min];
    OPS[usize::try_from(word % 3).unwrap_or(0)]
}

#[cfg(test)]
mod tests {
    use forja_core::{Backend, CommandList, Submission};
    use forja_cpu::CpuBackend;
    use proptest::test_runner::TestCaseError;

    use super::*;
    use crate::{DeterministicValues, allocate, allocate_initialized, generated_bytes};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1))]

        #[test]
        fn validator_accepts_at_least_ninety_nine_percent(
            cases in prop::collection::vec(well_typed_programs(64), 1024..=1024),
        ) {
            let accepted = cases
                .iter()
                .filter(|case| case.program.validate().is_ok())
                .count();
            let percentage = accepted * 100 / cases.len();
            eprintln!(
                "well-typed generator acceptance: {accepted}/{} ({percentage}%)",
                cases.len()
            );
            prop_assert!(percentage >= 99, "acceptance was {percentage}%");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn interpreter_never_panics(case in well_typed_programs(64)) {
            execute(&case).map_err(|error| TestCaseError::fail(error.to_string()))?;
        }
    }

    fn execute(case: &ProgramCase) -> Result<(), Box<dyn std::error::Error>> {
        let backend = CpuBackend::new();
        let mut values = DeterministicValues::new(0x510e_527f_ade6_82d1);
        let inputs = case
            .inputs
            .iter()
            .map(|spec| {
                let bytes = generated_bytes(spec, &mut values)?;
                allocate_initialized(&backend, spec, &bytes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = case
            .outputs
            .iter()
            .map(|spec| allocate(&backend, spec))
            .collect::<Result<Vec<_>, _>>()?;
        let program = case.program.validate()?;
        let mut commands = CommandList::new();
        commands.dispatch_program(
            &program,
            &inputs.iter().collect::<Vec<_>>(),
            &outputs.iter().collect::<Vec<_>>(),
        )?;
        backend.submit(commands)?.wait()?;
        Ok(())
    }
}
