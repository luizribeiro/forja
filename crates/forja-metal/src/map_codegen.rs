#![allow(dead_code)]

use std::fmt::{self, Write};

use forja_core::{
    DType,
    program::{BinOp, BoundProgram, Inst, ProgramKind, UnOp, ValueType},
};

pub(super) const KERNEL_NAME: &str = "forja_map";
pub(super) const SHARED_HEADER: &str = include_str!("elementwise.metal");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CodegenError {
    UnsupportedProgramKind,
}

pub(super) fn generate(program: &BoundProgram) -> Result<String, CodegenError> {
    if program.program().program().kind != ProgramKind::Map {
        return Err(CodegenError::UnsupportedProgramKind);
    }
    let inputs = program.inputs();
    let outputs = program.outputs();
    let rank = outputs[0].layout().shape().len();
    let mut source = String::from(SHARED_HEADER);
    line(&mut source, format_args!("\nkernel void {KERNEL_NAME}("));
    let operand_count = inputs.len() + outputs.len();
    for (slot, _) in inputs.iter().enumerate() {
        line(
            &mut source,
            format_args!("    device const uchar *input{slot} [[buffer({slot})]],"),
        );
    }
    for (slot, _) in outputs.iter().enumerate() {
        let buffer = inputs.len() + slot;
        line(
            &mut source,
            format_args!("    device uchar *output{slot} [[buffer({buffer})]],"),
        );
    }
    for slot in 0..inputs.len() {
        let buffer = operand_count + slot;
        line(
            &mut source,
            format_args!("    constant TensorLayout &input{slot}_layout [[buffer({buffer})]],"),
        );
    }
    for slot in 0..outputs.len() {
        let buffer = operand_count + inputs.len() + slot;
        line(
            &mut source,
            format_args!("    constant TensorLayout &output{slot}_layout [[buffer({buffer})]],"),
        );
    }
    line(
        &mut source,
        format_args!("    uint index [[thread_position_in_grid]]) {{"),
    );
    line(
        &mut source,
        format_args!("    if (index >= output0_layout.element_count) return;"),
    );
    line(&mut source, format_args!("    uint remaining = index;"));
    for axis in (0..rank).rev() {
        line(
            &mut source,
            format_args!("    uint coord{axis} = remaining % output0_layout.shape[{axis}];"),
        );
        line(
            &mut source,
            format_args!("    remaining /= output0_layout.shape[{axis}];"),
        );
    }
    for slot in 0..inputs.len() {
        line(
            &mut source,
            format_args!(
                "    ulong input_address{slot} = physical_index(input{slot}_layout, index);"
            ),
        );
    }
    for slot in 0..outputs.len() {
        line(
            &mut source,
            format_args!(
                "    ulong output_address{slot} = physical_index(output{slot}_layout, index);"
            ),
        );
    }

    let mut types = Vec::with_capacity(program.program().program().insts.len());
    for (index, &instruction) in program.program().program().insts.iter().enumerate() {
        let (value_type, expression) = instruction_expression(instruction, inputs, &types);
        types.push(value_type);
        line(
            &mut source,
            format_args!("    {} v{index} = {expression};", type_name(value_type)),
        );
    }
    for &(slot, value) in &program.program().program().outputs {
        let slot = usize::try_from(slot).unwrap_or(usize::MAX);
        line(
            &mut source,
            format_args!(
                "    store_float(output{slot}, output_address{slot}, {}u, v{value});",
                dtype_code(outputs[slot].layout().dtype())
            ),
        );
    }
    line(&mut source, format_args!("}}"));
    Ok(source)
}

fn instruction_expression(
    instruction: Inst,
    inputs: &[forja_core::Tensor],
    types: &[ValueType],
) -> (ValueType, String) {
    match instruction {
        Inst::Input(slot) => {
            let slot = usize::try_from(slot).unwrap_or(usize::MAX);
            match inputs[slot].layout().dtype() {
                DType::F32 | DType::F16 | DType::BF16 => (
                    ValueType::F32,
                    format!(
                        "load_float(input{slot}, input_address{slot}, {}u)",
                        dtype_code(inputs[slot].layout().dtype())
                    ),
                ),
                DType::U32 => (
                    ValueType::U32,
                    format!("load_uint(input{slot}, input_address{slot})"),
                ),
                DType::I32 => unreachable!("bound programs reject i32 inputs"),
            }
        }
        Inst::Const(value) => (
            ValueType::F32,
            format!("as_type<float>(0x{:08x}u)", value.to_bits()),
        ),
        Inst::Index(axis) => (ValueType::U32, format!("coord{axis}")),
        Inst::Extent(axis) => (ValueType::U32, format!("output0_layout.shape[{axis}]")),
        Inst::Unary(op, operand) => (ValueType::F32, unary(op, operand)),
        Inst::Binary(op, left, right) => binary(op, left, right, types[left as usize]),
        Inst::Select(condition, accepted, rejected) => (
            types[accepted as usize],
            format!("select(v{rejected}, v{accepted}, v{condition})"),
        ),
        Inst::Cast(to, operand) => (to, cast(to, operand, types[operand as usize])),
        Inst::Reduce(_, _) => unreachable!("map programs cannot contain reductions"),
    }
}

fn unary(op: UnOp, operand: u32) -> String {
    match op {
        UnOp::Neg => format!("-v{operand}"),
        UnOp::Abs => format!("abs(v{operand})"),
        UnOp::Exp => format!("precise::exp(v{operand})"),
        UnOp::Log => format!("precise::log(v{operand})"),
        UnOp::Sqrt => format!("precise::sqrt(v{operand})"),
        UnOp::Rsqrt => format!("precise::rsqrt(v{operand})"),
        UnOp::Sin => format!("precise::sin(v{operand})"),
        UnOp::Cos => format!("precise::cos(v{operand})"),
        UnOp::Tanh => format!("precise::tanh(v{operand})"),
        UnOp::Sigmoid => format!("1.0f / (1.0f + precise::exp(-v{operand}))"),
        UnOp::Recip => format!("1.0f / v{operand}"),
        UnOp::Floor => format!("floor(v{operand})"),
    }
}

fn binary(op: BinOp, left: u32, right: u32, operand_type: ValueType) -> (ValueType, String) {
    let comparison = matches!(
        op,
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne | BinOp::Ge | BinOp::Gt
    );
    let value_type = if comparison {
        ValueType::Bool
    } else {
        operand_type
    };
    let expression = match (op, operand_type) {
        (BinOp::Add, _) => format!("v{left} + v{right}"),
        (BinOp::Sub, _) => format!("v{left} - v{right}"),
        (BinOp::Mul, _) => format!("v{left} * v{right}"),
        (BinOp::Div, _) => format!("v{left} / v{right}"),
        (BinOp::Pow, _) => format!("precise::pow(v{left}, v{right})"),
        (BinOp::Min, ValueType::F32) => propagating_extreme(left, right, "<"),
        (BinOp::Max, ValueType::F32) => propagating_extreme(left, right, ">"),
        (BinOp::Min, ValueType::U32) => format!("min(v{left}, v{right})"),
        (BinOp::Max, ValueType::U32) => format!("max(v{left}, v{right})"),
        (BinOp::Lt, _) => format!("v{left} < v{right}"),
        (BinOp::Le, _) => format!("v{left} <= v{right}"),
        (BinOp::Eq, _) => format!("v{left} == v{right}"),
        (BinOp::Ne, _) => format!("v{left} != v{right}"),
        (BinOp::Ge, _) => format!("v{left} >= v{right}"),
        (BinOp::Gt, _) => format!("v{left} > v{right}"),
        (BinOp::Min | BinOp::Max, ValueType::Bool) => unreachable!("invalid binary type"),
    };
    (value_type, expression)
}

fn propagating_extreme(left: u32, right: u32, comparison: &str) -> String {
    format!(
        "select(select(v{right}, v{left}, v{left} {comparison} v{right}), \
         as_type<float>(0x7fc00000u), isnan(v{left}) || isnan(v{right}))"
    )
}

fn cast(to: ValueType, operand: u32, from: ValueType) -> String {
    match (to, from) {
        (ValueType::F32, ValueType::F32)
        | (ValueType::U32, ValueType::U32)
        | (ValueType::Bool, ValueType::Bool) => format!("v{operand}"),
        (ValueType::F32, ValueType::U32 | ValueType::Bool) => format!("float(v{operand})"),
        (ValueType::U32, ValueType::Bool) => format!("uint(v{operand})"),
        (ValueType::U32, ValueType::F32) => format!(
            "select(select(uint(clamp(v{operand}, 0.0f, 4294967040.0f)), 0xffffffffu, \
             v{operand} >= 4294967296.0f), 0u, isnan(v{operand}))"
        ),
        (ValueType::Bool, ValueType::F32) => format!("v{operand} != 0.0f"),
        (ValueType::Bool, ValueType::U32) => format!("v{operand} != 0u"),
    }
}

const fn type_name(value_type: ValueType) -> &'static str {
    match value_type {
        ValueType::F32 => "float",
        ValueType::U32 => "uint",
        ValueType::Bool => "bool",
    }
}

const fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
    }
}

fn line(source: &mut String, arguments: fmt::Arguments<'_>) {
    let _ = source.write_fmt(arguments);
    source.push('\n');
}

#[cfg(test)]
mod tests {
    use forja_core::{
        DType,
        program::{BinOp, Inst, Program, ProgramKind, UnOp, ValueType, bind_program},
    };
    use forja_cpu::CpuBackend;

    use super::*;

    #[test]
    fn residual_add_has_stable_source() {
        let source = source(
            &Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Binary(BinOp::Add, 0, 1),
                ],
                outputs: vec![(0, 2)],
            },
            &[DType::F16, DType::BF16],
            &[DType::F32],
            &[7, 33],
        );
        assert_eq!(
            source.strip_prefix(SHARED_HEADER).unwrap(),
            r"
kernel void forja_map(
    device const uchar *input0 [[buffer(0)]],
    device const uchar *input1 [[buffer(1)]],
    device uchar *output0 [[buffer(2)]],
    constant TensorLayout &input0_layout [[buffer(3)]],
    constant TensorLayout &input1_layout [[buffer(4)]],
    constant TensorLayout &output0_layout [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= output0_layout.element_count) return;
    uint remaining = index;
    uint coord1 = remaining % output0_layout.shape[1];
    remaining /= output0_layout.shape[1];
    uint coord0 = remaining % output0_layout.shape[0];
    remaining /= output0_layout.shape[0];
    ulong input_address0 = physical_index(input0_layout, index);
    ulong input_address1 = physical_index(input1_layout, index);
    ulong output_address0 = physical_index(output0_layout, index);
    float v0 = load_float(input0, input_address0, 1u);
    float v1 = load_float(input1, input_address1, 2u);
    float v2 = v0 + v1;
    store_float(output0, output_address0, 0u, v2);
}
"
        );
    }

    #[test]
    fn special_semantics_are_explicit() {
        let source = source(
            &Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Binary(BinOp::Min, 0, 1),
                    Inst::Binary(BinOp::Max, 0, 1),
                    Inst::Cast(ValueType::U32, 2),
                    Inst::Binary(BinOp::Add, 4, 4),
                    Inst::Cast(ValueType::F32, 5),
                    Inst::Unary(UnOp::Exp, 3),
                    Inst::Binary(BinOp::Add, 6, 7),
                ],
                outputs: vec![(0, 8)],
            },
            &[DType::F32, DType::F32],
            &[DType::F32],
            &[1],
        );
        assert!(source.contains("isnan(v0) || isnan(v1)"));
        assert!(source.contains("clamp(v2, 0.0f, 4294967040.0f)"));
        assert!(source.contains("uint v5 = v4 + v4;"));
        assert!(source.contains("precise::exp(v3)"));
    }

    #[test]
    fn addresses_never_depend_on_program_values() {
        let source = source(
            &Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Index(0),
                    Inst::Cast(ValueType::F32, 1),
                ],
                outputs: vec![(0, 2)],
            },
            &[DType::F32],
            &[DType::F32],
            &[4097],
        );
        for line in source
            .lines()
            .filter(|line| line.contains("physical_index("))
        {
            assert!(
                !(0..3).any(|value| line.contains(&format!("v{value}"))),
                "program value in address expression: {line}"
            );
        }
    }

    fn source(program: &Program, inputs: &[DType], outputs: &[DType], shape: &[u32]) -> String {
        let backend = CpuBackend::new();
        let inputs = inputs
            .iter()
            .map(|&dtype| backend.alloc(dtype, shape).unwrap())
            .collect::<Vec<_>>();
        let outputs = outputs
            .iter()
            .map(|&dtype| backend.alloc(dtype, shape).unwrap())
            .collect::<Vec<_>>();
        let program = program.validate().unwrap();
        let bound = bind_program(
            &program,
            &inputs.iter().collect::<Vec<_>>(),
            &outputs.iter().collect::<Vec<_>>(),
        )
        .unwrap();
        generate(&bound).unwrap()
    }
}
