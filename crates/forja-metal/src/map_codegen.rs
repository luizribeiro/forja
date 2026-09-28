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
        format_args!("    uint3 position [[thread_position_in_grid]]) {{"),
    );
    emit_coordinates(&mut source, rank);
    emit_addresses(&mut source, inputs.len(), outputs.len(), rank);

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
        let expression = if outputs[slot].layout().dtype() == DType::BF16 {
            format!("store_bfloat_bits(output{slot}, output_address{slot}, v{value});")
        } else {
            format!(
                "store_float(output{slot}, output_address{slot}, {}, v{value});",
                output_dtype_name(slot)
            )
        };
        line(&mut source, format_args!("    {expression}"));
    }
    line(&mut source, format_args!("}}"));
    Ok(source)
}

fn address_expression(layout: &str, rank: usize) -> String {
    (0..rank).fold(format!("{layout}.offset"), |address, axis| {
        format!("{address} + ulong(coord{axis}) * {layout}.strides[{axis}]")
    })
}

fn emit_addresses(source: &mut String, input_count: usize, output_count: usize, rank: usize) {
    for (name, slot) in operands(input_count, output_count) {
        line(
            source,
            format_args!(
                "    ulong {name}_address{slot} = {};",
                address_expression(&format!("{name}{slot}_layout"), rank)
            ),
        );
    }
}

fn emit_coordinates(source: &mut String, rank: usize) {
    if rank == 0 {
        line(source, format_args!("    if (position.x != 0) return;"));
        return;
    }
    if rank == 1 {
        line(
            source,
            format_args!("    if (position.x >= output0_layout.shape[0]) return;"),
        );
        line(source, format_args!("    uint coord0 = position.x;"));
        return;
    }
    let last = rank - 1;
    let penultimate = rank - 2;
    line(
        source,
        format_args!(
            "    if (position.x >= output0_layout.shape[{last}] || \
             position.y >= output0_layout.shape[{penultimate}]) return;"
        ),
    );
    line(source, format_args!("    uint coord{last} = position.x;"));
    line(
        source,
        format_args!("    uint coord{penultimate} = position.y;"),
    );
    if penultimate > 0 {
        line(source, format_args!("    uint remaining = position.z;"));
        for axis in (0..penultimate).rev() {
            line(
                source,
                format_args!("    uint coord{axis} = remaining % output0_layout.shape[{axis}];"),
            );
            if axis > 0 {
                line(
                    source,
                    format_args!("    remaining /= output0_layout.shape[{axis}];"),
                );
            }
        }
    }
}

fn operands(
    input_count: usize,
    output_count: usize,
) -> impl Iterator<Item = (&'static str, usize)> {
    (0..input_count)
        .map(|slot| ("input", slot))
        .chain((0..output_count).map(|slot| ("output", slot)))
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
                DType::F32 | DType::F16 => (
                    ValueType::F32,
                    format!(
                        "load_float(input{slot}, input_address{slot}, {})",
                        input_dtype_name(slot)
                    ),
                ),
                DType::BF16 => (
                    ValueType::F32,
                    format!("load_bfloat_bits(input{slot}, input_address{slot})"),
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
    let nan = "as_type<float>(0x7fc00000u)";
    match op {
        UnOp::Neg => format!("-v{operand}"),
        UnOp::Abs => format!("abs(v{operand})"),
        UnOp::Exp => format!("select(precise::exp(v{operand}), {nan}, isnan(v{operand}))"),
        UnOp::Log => {
            format!(
                "select(precise::log(v{operand}), {nan}, isnan(v{operand}) || v{operand} < 0.0f)"
            )
        }
        UnOp::Sqrt => {
            format!(
                "select(precise::sqrt(v{operand}), {nan}, isnan(v{operand}) || v{operand} < 0.0f)"
            )
        }
        UnOp::Rsqrt => format!(
            "select(precise::rsqrt(v{operand}), {nan}, isnan(v{operand}) || v{operand} < 0.0f)"
        ),
        UnOp::Sin => format!("select(precise::sin(v{operand}), {nan}, !isfinite(v{operand}))"),
        UnOp::Cos => format!("select(precise::cos(v{operand}), {nan}, !isfinite(v{operand}))"),
        UnOp::Tanh => format!(
            "select(select(precise::tanh(v{operand}), copysign(1.0f, v{operand}), \
             abs(v{operand}) > 8.0f), {nan}, isnan(v{operand}))"
        ),
        UnOp::Sigmoid => {
            format!("select(1.0f / (1.0f + precise::exp(-v{operand})), {nan}, isnan(v{operand}))")
        }
        UnOp::Recip => format!("select(1.0f / v{operand}, {nan}, isnan(v{operand}))"),
        UnOp::Floor => format!("select(floor(v{operand}), {nan}, isnan(v{operand}))"),
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

fn input_dtype_name(slot: usize) -> String {
    match slot {
        0 => "input0_dtype".to_owned(),
        1 => "input1_dtype".to_owned(),
        2 => "input2_dtype".to_owned(),
        _ => format!("program_input{slot}_dtype"),
    }
}

fn output_dtype_name(slot: usize) -> String {
    if slot == 0 {
        "output_dtype".to_owned()
    } else {
        format!("program_output{slot}_dtype")
    }
}

const fn type_name(value_type: ValueType) -> &'static str {
    match value_type {
        ValueType::F32 => "float",
        ValueType::U32 => "uint",
        ValueType::Bool => "bool",
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
    use forja_testing::{TensorSpec, program::map_programs};
    use proptest::prelude::*;

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
    uint3 position [[thread_position_in_grid]]) {
    if (position.x >= output0_layout.shape[1] || position.y >= output0_layout.shape[0]) return;
    uint coord1 = position.x;
    uint coord0 = position.y;
    ulong input_address0 = input0_layout.offset + ulong(coord0) * input0_layout.strides[0] + ulong(coord1) * input0_layout.strides[1];
    ulong input_address1 = input1_layout.offset + ulong(coord0) * input1_layout.strides[0] + ulong(coord1) * input1_layout.strides[1];
    ulong output_address0 = output0_layout.offset + ulong(coord0) * output0_layout.strides[0] + ulong(coord1) * output0_layout.strides[1];
    float v0 = load_float(input0, input_address0, input0_dtype);
    float v1 = load_bfloat_bits(input1, input_address1);
    float v2 = v0 + v1;
    store_float(output0, output_address0, output_dtype, v2);
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn addresses_never_depend_on_program_values(case in map_programs(64)) {
            let inputs = case.inputs().iter().map(TensorSpec::dtype).collect::<Vec<_>>();
            let outputs = case.outputs().iter().map(TensorSpec::dtype).collect::<Vec<_>>();
            let source = source(case.program(), &inputs, &outputs, case.shape());
            for line in source
                .lines()
                .filter(|line| line.contains("ulong ") && line.contains("_address"))
            {
                prop_assert!(
                    !contains_program_value(line),
                    "program value in address expression: {line}"
                );
            }
        }
    }

    fn contains_program_value(expression: &str) -> bool {
        expression
            .as_bytes()
            .windows(2)
            .any(|pair| pair[0] == b'v' && pair[1].is_ascii_digit())
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
