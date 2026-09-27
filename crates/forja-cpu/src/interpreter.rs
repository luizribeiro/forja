//! Scalar-program reference semantics.

use forja_core::{
    BackendError,
    program::{BinOp, Inst, ProgramKind, RedOp, UnOp, ValidatedProgram, ValueType},
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    F32(f32),
    U32(u32),
    Bool(bool),
}

/// Logical values for one interpreter input slot.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// Floating-point values, already converted to F32 from their storage type.
    F32(&'a [f32]),
    /// Unsigned integer values.
    U32(&'a [u32]),
}

/// Evaluates a map program in logical row-major coordinate order.
///
/// Every instruction is evaluated in SSA order at every coordinate, including
/// both alternatives of a select. Floating-point arithmetic uses Rust's IEEE
/// `f32` operations and propagates NaNs; division by zero therefore produces
/// infinities or NaN. Float-to-unsigned casts saturate, with NaN becoming zero.
/// Unsigned addition, subtraction, and multiplication wrap. `pow` is `f32::powf`,
/// including its behavior for negative bases. Output conversion is performed by
/// the backend after interpretation.
///
/// # Errors
///
/// Returns [`BackendError::ExecutionFailed`] if the program is not a map or
/// its logical inputs and iteration shape are inconsistent.
pub fn interpret_map(
    program: &ValidatedProgram,
    shape: &[u32],
    inputs: &[Input<'_>],
) -> Result<Vec<Vec<f32>>, BackendError> {
    if program.program().kind != ProgramKind::Map {
        return Err(BackendError::ExecutionFailed);
    }
    let elements = element_count(shape)?;
    if inputs.iter().any(|input| input.len() != elements) {
        return Err(BackendError::ExecutionFailed);
    }
    let mut outputs = vec![Vec::with_capacity(elements); program.output_count()];
    let mut coordinates = vec![0_u32; shape.len()];
    for linear in 0..elements {
        decode_coordinates(linear, shape, &mut coordinates)?;
        let mut values = Vec::with_capacity(program.program().insts.len());
        for &inst in &program.program().insts {
            values.push(eval(
                inst,
                &coordinates,
                shape,
                inputs,
                linear,
                |operand| value_at(&values, operand),
            )?);
        }
        for &(slot, value) in &program.program().outputs {
            let output = outputs
                .get_mut(usize::try_from(slot).map_err(|_| BackendError::ExecutionFailed)?)
                .ok_or(BackendError::ExecutionFailed)?;
            output.push(as_f32(value_at(&values, value)?)?);
        }
    }
    Ok(outputs)
}

/// Evaluates a row program in logical row-major coordinate order.
///
/// Reductions consume the complete last-axis row and combine F32 values
/// sequentially from left to right. Each result is broadcast to every lane of
/// that row. Evaluation then resumes, so subsequent instructions and reductions
/// can depend on any earlier reduction.
///
/// # Errors
///
/// Returns [`BackendError::ExecutionFailed`] if the program is not a row or
/// its logical inputs and nonempty iteration shape are inconsistent.
pub fn interpret_row(
    program: &ValidatedProgram,
    shape: &[u32],
    inputs: &[Input<'_>],
) -> Result<Vec<Vec<f32>>, BackendError> {
    if program.program().kind != ProgramKind::Row {
        return Err(BackendError::ExecutionFailed);
    }
    let elements = element_count(shape)?;
    if inputs.iter().any(|input| input.len() != elements) {
        return Err(BackendError::ExecutionFailed);
    }
    let (&last_extent, row_shape) = shape.split_last().ok_or(BackendError::ExecutionFailed)?;
    let row_width = usize::try_from(last_extent).map_err(|_| BackendError::ExecutionFailed)?;
    if row_width == 0 {
        return Err(BackendError::ExecutionFailed);
    }
    let row_count = element_count(row_shape)?;
    let mut outputs = vec![Vec::with_capacity(elements); program.output_count()];
    let mut coordinates = vec![0_u32; shape.len()];
    for row in 0..row_count {
        let row_start = row
            .checked_mul(row_width)
            .ok_or(BackendError::ExecutionFailed)?;
        let mut columns = Vec::<Vec<Scalar>>::with_capacity(program.program().insts.len());
        for &inst in &program.program().insts {
            let column = if let Inst::Reduce(op, operand) = inst {
                let operand = columns
                    .get(usize::try_from(operand).map_err(|_| BackendError::ExecutionFailed)?)
                    .ok_or(BackendError::ExecutionFailed)?;
                vec![reduce(op, operand)?; row_width]
            } else {
                let mut column = Vec::with_capacity(row_width);
                for lane in 0..row_width {
                    let linear = row_start
                        .checked_add(lane)
                        .ok_or(BackendError::ExecutionFailed)?;
                    decode_coordinates(linear, shape, &mut coordinates)?;
                    column.push(eval(
                        inst,
                        &coordinates,
                        shape,
                        inputs,
                        linear,
                        |operand| column_value(&columns, operand, lane),
                    )?);
                }
                column
            };
            columns.push(column);
        }
        for &(slot, value) in &program.program().outputs {
            let output = outputs
                .get_mut(usize::try_from(slot).map_err(|_| BackendError::ExecutionFailed)?)
                .ok_or(BackendError::ExecutionFailed)?;
            let column = columns
                .get(usize::try_from(value).map_err(|_| BackendError::ExecutionFailed)?)
                .ok_or(BackendError::ExecutionFailed)?;
            for &value in column {
                output.push(as_f32(value)?);
            }
        }
    }
    Ok(outputs)
}

impl Input<'_> {
    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    fn get(self, index: usize) -> Option<Scalar> {
        match self {
            Self::F32(values) => values.get(index).copied().map(Scalar::F32),
            Self::U32(values) => values.get(index).copied().map(Scalar::U32),
        }
    }
}

fn element_count(shape: &[u32]) -> Result<usize, BackendError> {
    shape.iter().try_fold(1_usize, |count, &extent| {
        count
            .checked_mul(usize::try_from(extent).map_err(|_| BackendError::ExecutionFailed)?)
            .ok_or(BackendError::ExecutionFailed)
    })
}

fn decode_coordinates(
    mut linear: usize,
    shape: &[u32],
    coordinates: &mut [u32],
) -> Result<(), BackendError> {
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape).rev() {
        let extent = usize::try_from(extent).map_err(|_| BackendError::ExecutionFailed)?;
        if extent == 0 {
            return Err(BackendError::ExecutionFailed);
        }
        *coordinate = u32::try_from(linear % extent).map_err(|_| BackendError::ExecutionFailed)?;
        linear /= extent;
    }
    Ok(())
}

fn eval(
    inst: Inst,
    coordinates: &[u32],
    shape: &[u32],
    inputs: &[Input<'_>],
    linear: usize,
    operand: impl Fn(u32) -> Result<Scalar, BackendError>,
) -> Result<Scalar, BackendError> {
    match inst {
        Inst::Input(slot) => inputs
            .get(usize::try_from(slot).map_err(|_| BackendError::ExecutionFailed)?)
            .and_then(|input| input.get(linear))
            .ok_or(BackendError::ExecutionFailed),
        Inst::Const(value) => Ok(Scalar::F32(value)),
        Inst::Index(axis) => coordinates
            .get(usize::from(axis))
            .copied()
            .map(Scalar::U32)
            .ok_or(BackendError::ExecutionFailed),
        Inst::Extent(axis) => shape
            .get(usize::from(axis))
            .copied()
            .map(Scalar::U32)
            .ok_or(BackendError::ExecutionFailed),
        Inst::Unary(op, value) => unary(op, operand(value)?),
        Inst::Binary(op, left, right) => binary(op, operand(left)?, operand(right)?),
        Inst::Select(condition, accepted, rejected) => match operand(condition)? {
            Scalar::Bool(true) => operand(accepted),
            Scalar::Bool(false) => operand(rejected),
            Scalar::F32(_) | Scalar::U32(_) => Err(BackendError::ExecutionFailed),
        },
        Inst::Cast(to, value) => Ok(cast(to, operand(value)?)),
        Inst::Reduce(_, _) => Err(BackendError::ExecutionFailed),
    }
}

fn value_at(values: &[Scalar], index: u32) -> Result<Scalar, BackendError> {
    values
        .get(usize::try_from(index).map_err(|_| BackendError::ExecutionFailed)?)
        .copied()
        .ok_or(BackendError::ExecutionFailed)
}

fn column_value(columns: &[Vec<Scalar>], index: u32, lane: usize) -> Result<Scalar, BackendError> {
    columns
        .get(usize::try_from(index).map_err(|_| BackendError::ExecutionFailed)?)
        .and_then(|column| column.get(lane))
        .copied()
        .ok_or(BackendError::ExecutionFailed)
}

fn reduce(op: RedOp, values: &[Scalar]) -> Result<Scalar, BackendError> {
    let mut values = values.iter().copied().map(as_f32);
    match op {
        RedOp::Sum => values
            .try_fold(0.0, |sum, value| value.map(|value| sum + value))
            .map(Scalar::F32),
        RedOp::Max | RedOp::Min => {
            let first = values.next().ok_or(BackendError::ExecutionFailed)??;
            values
                .try_fold(first, |accumulator, value| {
                    value.map(|value| match op {
                        RedOp::Max => propagating_max(accumulator, value),
                        RedOp::Min => propagating_min(accumulator, value),
                        RedOp::Sum => accumulator,
                    })
                })
                .map(Scalar::F32)
        }
    }
}

fn unary(op: UnOp, value: Scalar) -> Result<Scalar, BackendError> {
    let value = as_f32(value)?;
    Ok(Scalar::F32(match op {
        UnOp::Neg => -value,
        UnOp::Abs => value.abs(),
        UnOp::Exp => value.exp(),
        UnOp::Log => value.ln(),
        UnOp::Sqrt => value.sqrt(),
        UnOp::Rsqrt => value.sqrt().recip(),
        UnOp::Sin => value.sin(),
        UnOp::Cos => value.cos(),
        UnOp::Tanh => value.tanh(),
        UnOp::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        UnOp::Recip => value.recip(),
        UnOp::Floor => value.floor(),
    }))
}

fn binary(op: BinOp, left: Scalar, right: Scalar) -> Result<Scalar, BackendError> {
    match (left, right) {
        (Scalar::F32(left), Scalar::F32(right)) => Ok(float_binary(op, left, right)),
        (Scalar::U32(left), Scalar::U32(right)) => integer_binary(op, left, right),
        _ => Err(BackendError::ExecutionFailed),
    }
}

#[allow(clippy::float_cmp)]
fn float_binary(op: BinOp, left: f32, right: f32) -> Scalar {
    match op {
        BinOp::Add => Scalar::F32(left + right),
        BinOp::Sub => Scalar::F32(left - right),
        BinOp::Mul => Scalar::F32(left * right),
        BinOp::Div => Scalar::F32(left / right),
        BinOp::Min => Scalar::F32(propagating_min(left, right)),
        BinOp::Max => Scalar::F32(propagating_max(left, right)),
        BinOp::Pow => Scalar::F32(left.powf(right)),
        BinOp::Lt => Scalar::Bool(left < right),
        BinOp::Le => Scalar::Bool(left <= right),
        BinOp::Eq => Scalar::Bool(left == right),
        BinOp::Ne => Scalar::Bool(left != right),
        BinOp::Ge => Scalar::Bool(left >= right),
        BinOp::Gt => Scalar::Bool(left > right),
    }
}

fn integer_binary(op: BinOp, left: u32, right: u32) -> Result<Scalar, BackendError> {
    let value = match op {
        BinOp::Add => Scalar::U32(left.wrapping_add(right)),
        BinOp::Sub => Scalar::U32(left.wrapping_sub(right)),
        BinOp::Mul => Scalar::U32(left.wrapping_mul(right)),
        BinOp::Min => Scalar::U32(left.min(right)),
        BinOp::Max => Scalar::U32(left.max(right)),
        BinOp::Lt => Scalar::Bool(left < right),
        BinOp::Le => Scalar::Bool(left <= right),
        BinOp::Eq => Scalar::Bool(left == right),
        BinOp::Ne => Scalar::Bool(left != right),
        BinOp::Ge => Scalar::Bool(left >= right),
        BinOp::Gt => Scalar::Bool(left > right),
        BinOp::Div | BinOp::Pow => return Err(BackendError::ExecutionFailed),
    };
    Ok(value)
}

fn cast(to: ValueType, value: Scalar) -> Scalar {
    match to {
        ValueType::F32 => Scalar::F32(match value {
            Scalar::F32(value) => value,
            Scalar::U32(value) => u32_to_f32(value),
            Scalar::Bool(value) => u8::from(value).into(),
        }),
        ValueType::U32 => Scalar::U32(match value {
            Scalar::F32(value) => f32_to_u32(value),
            Scalar::U32(value) => value,
            Scalar::Bool(value) => u32::from(value),
        }),
        ValueType::Bool => Scalar::Bool(match value {
            Scalar::F32(value) => value != 0.0,
            Scalar::U32(value) => value != 0,
            Scalar::Bool(value) => value,
        }),
    }
}

#[allow(clippy::cast_precision_loss)]
fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f32_to_u32(value: f32) -> u32 {
    value as u32
}

fn as_f32(value: Scalar) -> Result<f32, BackendError> {
    match value {
        Scalar::F32(value) => Ok(value),
        Scalar::U32(_) | Scalar::Bool(_) => Err(BackendError::ExecutionFailed),
    }
}

fn propagating_min(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        f32::NAN
    } else {
        left.min(right)
    }
}

fn propagating_max(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        f32::NAN
    } else {
        left.max(right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::program::{Program, ValueType};

    fn run(insts: Vec<Inst>, output: u32, inputs: &[Input<'_>]) -> Vec<f32> {
        let program = Program {
            kind: ProgramKind::Map,
            insts,
            outputs: vec![(0, output)],
        }
        .validate()
        .unwrap();
        interpret_map(
            &program,
            &[u32::try_from(inputs.first().map_or(1, |input| input.len())).unwrap()],
            inputs,
        )
        .unwrap()
        .remove(0)
    }

    fn run_row(insts: Vec<Inst>, output: u32, input: &[f32]) -> Vec<f32> {
        let program = Program {
            kind: ProgramKind::Row,
            insts,
            outputs: vec![(0, output)],
        }
        .validate()
        .unwrap();
        interpret_row(
            &program,
            &[1, u32::try_from(input.len()).unwrap()],
            &[Input::F32(input)],
        )
        .unwrap()
        .remove(0)
    }

    #[test]
    fn evaluates_coordinates_inputs_and_every_unary_operation() {
        let input = [4.0_f32, 9.0];
        let operations = [
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
        for operation in operations {
            let result = run(
                vec![Inst::Input(0), Inst::Unary(operation, 0)],
                1,
                &[Input::F32(&input)],
            );
            assert_eq!(result.len(), 2);
        }
        let coordinates = run(
            vec![
                Inst::Index(0),
                Inst::Extent(0),
                Inst::Binary(BinOp::Add, 0, 1),
                Inst::Cast(ValueType::F32, 2),
            ],
            3,
            &[],
        );
        assert_eq!(coordinates, [1.0]);
    }

    #[test]
    fn defines_float_corner_cases() {
        let zero = [0.0_f32];
        let negative = [-2.0_f32];
        let half = [0.5_f32];
        let infinity = run(
            vec![
                Inst::Const(1.0),
                Inst::Input(0),
                Inst::Binary(BinOp::Div, 0, 1),
            ],
            2,
            &[Input::F32(&zero)],
        )[0];
        assert!(infinity.is_infinite() && infinity.is_sign_positive());
        let nan = run(
            vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Binary(BinOp::Pow, 0, 1),
            ],
            2,
            &[Input::F32(&negative), Input::F32(&half)],
        )[0];
        assert!(nan.is_nan());
        let propagated = run(
            vec![
                Inst::Input(0),
                Inst::Const(1.0),
                Inst::Binary(BinOp::Min, 0, 1),
            ],
            2,
            &[Input::F32(&[f32::NAN])],
        )[0];
        assert!(propagated.is_nan());
    }

    #[test]
    fn casts_saturate_and_unsigned_arithmetic_wraps() {
        let values = [f32::NAN, -1.0, f32::INFINITY];
        let casted = run(
            vec![
                Inst::Input(0),
                Inst::Cast(ValueType::U32, 0),
                Inst::Cast(ValueType::F32, 1),
            ],
            2,
            &[Input::F32(&values)],
        );
        assert_eq!(casted, [0.0, 0.0, 4_294_967_300.0]);

        let integers = [u32::MAX];
        let wrapped = run(
            vec![
                Inst::Input(0),
                Inst::Extent(0),
                Inst::Binary(BinOp::Add, 0, 1),
                Inst::Cast(ValueType::F32, 2),
            ],
            3,
            &[Input::U32(&integers)],
        );
        assert_eq!(wrapped, [0.0]);
    }

    #[test]
    fn select_uses_the_value_from_an_already_evaluated_branch() {
        let result = run(
            vec![
                Inst::Const(0.0),
                Inst::Const(0.0),
                Inst::Binary(BinOp::Div, 0, 1),
                Inst::Const(1.0),
                Inst::Binary(BinOp::Eq, 0, 0),
                Inst::Select(4, 3, 2),
            ],
            5,
            &[],
        );
        assert_eq!(result, [1.0]);
    }

    #[test]
    fn reductions_fold_left_to_right_and_broadcast() {
        let result = run_row(
            vec![Inst::Input(0), Inst::Reduce(RedOp::Sum, 0)],
            1,
            &[1.0e20, -1.0e20, 3.0],
        );
        assert_eq!(result, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn later_reductions_depend_on_earlier_stages() {
        let result = run_row(
            vec![
                Inst::Input(0),
                Inst::Reduce(RedOp::Sum, 0),
                Inst::Binary(BinOp::Mul, 0, 1),
                Inst::Reduce(RedOp::Sum, 2),
            ],
            3,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(result, [36.0, 36.0, 36.0]);
    }

    #[test]
    fn reduction_extrema_propagate_nan() {
        for operation in [RedOp::Min, RedOp::Max] {
            let result = run_row(
                vec![Inst::Input(0), Inst::Reduce(operation, 0)],
                1,
                &[1.0, f32::NAN, 2.0],
            );
            assert!(result.iter().all(|value| value.is_nan()));
        }
    }
}
