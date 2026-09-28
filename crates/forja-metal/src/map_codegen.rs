use std::fmt::{self, Write};

use forja_core::{
    DType,
    program::{
        BinOp, Inst, KernelSignature, ProgramKind, RedOp, UnOp, ValidatedProgram, ValueType,
    },
};

pub(super) const KERNEL_NAME: &str = "forja_map";
pub(super) const ROW_KERNEL_NAME: &str = "forja_row";
pub(super) const REGISTER_RESIDENT_WIDTH: u32 = 1024;
pub(super) const SHARED_HEADER: &str = include_str!("elementwise.metal");

pub(super) fn generate(
    program: &ValidatedProgram,
    signature: &KernelSignature,
    resident: bool,
) -> String {
    match program.program().kind {
        ProgramKind::Map => generate_map(program, signature),
        ProgramKind::Row if resident => generate_resident_row(program, signature),
        ProgramKind::Row => generate_row(program, signature),
    }
}

fn generate_map(program: &ValidatedProgram, signature: &KernelSignature) -> String {
    let inputs = signature.input_dtypes();
    let rank = usize::from(signature.rank());
    let mut source = String::from(SHARED_HEADER);
    line(&mut source, format_args!("\nkernel void {KERNEL_NAME}("));
    emit_operand_parameters(&mut source, inputs.len(), signature.output_dtypes().len());
    line(
        &mut source,
        format_args!("    uint3 position [[thread_position_in_grid]]) {{"),
    );
    emit_coordinates(&mut source, rank);
    emit_addresses(
        &mut source,
        inputs.len(),
        signature.output_dtypes().len(),
        rank,
    );

    let mut types = Vec::with_capacity(program.program().insts.len());
    for (index, &instruction) in program.program().insts.iter().enumerate() {
        let (value_type, expression) = instruction_expression(instruction, inputs, &types);
        types.push(value_type);
        line(
            &mut source,
            format_args!("    {} v{index} = {expression};", type_name(value_type)),
        );
    }
    for &(slot, value) in &program.program().outputs {
        let slot = usize::try_from(slot).unwrap_or(usize::MAX);
        let expression = format!(
            "store_float(output{slot}, output_address{slot}, {}, v{value});",
            output_dtype_name(slot)
        );
        line(&mut source, format_args!("    {expression}"));
    }
    line(&mut source, format_args!("}}"));
    source
}

fn generate_row(program: &ValidatedProgram, signature: &KernelSignature) -> String {
    let mut source = row_source(signature);
    let rank = usize::from(signature.rank());
    let instructions = &program.program().insts;
    for (reduce_slot, (index, instruction)) in instructions
        .iter()
        .enumerate()
        .filter_map(|(index, instruction)| match instruction {
            Inst::Reduce(op, operand) => Some((index, (*op, *operand))),
            _ => None,
        })
        .enumerate()
    {
        emit_reduce_stage(
            &mut source,
            program,
            signature,
            index,
            reduce_slot,
            instruction.0,
            instruction.1,
        );
    }
    emit_output_stage(&mut source, program, signature, rank);
    line(&mut source, format_args!("}}"));
    source
}

fn generate_resident_row(program: &ValidatedProgram, signature: &KernelSignature) -> String {
    let mut source = row_source(signature);
    let rank = usize::from(signature.rank());
    line(&mut source, format_args!("    bool active = lane < width;"));
    line(
        &mut source,
        format_args!("    uint column = min(lane, width - 1u);"),
    );
    emit_lane_coordinates(&mut source, rank, 4);
    emit_addresses_indented(
        &mut source,
        signature.input_dtypes().len(),
        signature.output_dtypes().len(),
        rank,
        4,
    );
    let mut types = Vec::with_capacity(program.program().insts.len());
    let mut reduce_slot = 0;
    for (index, &instruction) in program.program().insts.iter().enumerate() {
        if let Inst::Reduce(op, operand) = instruction {
            emit_resident_reduce(&mut source, reduce_slot, op, operand);
            types.push(ValueType::F32);
            line(
                &mut source,
                format_args!("    float v{index} = row_values[{reduce_slot}];"),
            );
            reduce_slot += 1;
        } else {
            let (value_type, expression) =
                instruction_expression(instruction, signature.input_dtypes(), &types);
            types.push(value_type);
            line(
                &mut source,
                format_args!("    {} v{index} = {expression};", type_name(value_type)),
            );
        }
    }
    line(&mut source, format_args!("    if (active) {{"));
    emit_stores(&mut source, program, 8);
    line(&mut source, format_args!("    }}"));
    line(&mut source, format_args!("}}"));
    source
}

fn row_source(signature: &KernelSignature) -> String {
    let rank = usize::from(signature.rank());
    let mut source = String::from(SHARED_HEADER);
    line(
        &mut source,
        format_args!("\nkernel void {ROW_KERNEL_NAME}("),
    );
    emit_operand_parameters(
        &mut source,
        signature.input_dtypes().len(),
        signature.output_dtypes().len(),
    );
    line(
        &mut source,
        format_args!("    uint row [[threadgroup_position_in_grid]],"),
    );
    line(
        &mut source,
        format_args!("    uint lane [[thread_position_in_threadgroup]],"),
    );
    line(
        &mut source,
        format_args!("    uint group_width [[threads_per_threadgroup]],"),
    );
    line(
        &mut source,
        format_args!("    uint simd_lane [[thread_index_in_simdgroup]],"),
    );
    line(
        &mut source,
        format_args!("    uint simd_group [[simdgroup_index_in_threadgroup]]) {{"),
    );
    line(
        &mut source,
        format_args!("    threadgroup float partial[32];"),
    );
    line(
        &mut source,
        format_args!("    threadgroup uint nan_partial[32];"),
    );
    line(
        &mut source,
        format_args!("    threadgroup float row_values[4];"),
    );
    emit_row_coordinates(&mut source, rank);
    source
}

fn emit_resident_reduce(source: &mut String, reduce_slot: usize, op: RedOp, operand: u32) {
    let identity = match op {
        RedOp::Sum => "0.0f",
        RedOp::Max => "-INFINITY",
        RedOp::Min => "INFINITY",
    };
    let simd_reduce = match op {
        RedOp::Sum => "simd_sum",
        RedOp::Max => "simd_max",
        RedOp::Min => "simd_min",
    };
    line(source, format_args!("    {{"));
    line(
        source,
        format_args!("        float accumulator = select({identity}, v{operand}, active);"),
    );
    line(
        source,
        format_args!("        bool accumulator_nan = active && f32_is_nan(v{operand});"),
    );
    emit_threadgroup_reduce(source, reduce_slot, identity, simd_reduce);
    line(source, format_args!("    }}"));
}

fn emit_operand_parameters(source: &mut String, input_count: usize, output_count: usize) {
    let operand_count = input_count + output_count;
    for slot in 0..input_count {
        line(
            source,
            format_args!("    device const uchar *input{slot} [[buffer({slot})]],"),
        );
    }
    for slot in 0..output_count {
        let buffer = input_count + slot;
        line(
            source,
            format_args!("    device uchar *output{slot} [[buffer({buffer})]],"),
        );
    }
    for slot in 0..input_count {
        let buffer = operand_count + slot;
        line(
            source,
            format_args!("    constant TensorLayout &input{slot}_layout [[buffer({buffer})]],"),
        );
    }
    for slot in 0..output_count {
        let buffer = operand_count + input_count + slot;
        line(
            source,
            format_args!("    constant TensorLayout &output{slot}_layout [[buffer({buffer})]],"),
        );
    }
}

fn emit_row_coordinates(source: &mut String, rank: usize) {
    let last = rank - 1;
    line(
        source,
        format_args!("    uint width = output0_layout.shape[{last}];"),
    );
    if rank > 1 {
        line(source, format_args!("    uint remaining = row;"));
        for axis in (0..last).rev() {
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

fn emit_reduce_stage(
    source: &mut String,
    program: &ValidatedProgram,
    signature: &KernelSignature,
    instruction_end: usize,
    reduce_slot: usize,
    op: RedOp,
    operand: u32,
) {
    let rank = usize::from(signature.rank());
    let identity = match op {
        RedOp::Sum => "0.0f",
        RedOp::Max => "-INFINITY",
        RedOp::Min => "INFINITY",
    };
    line(source, format_args!("    {{"));
    line(
        source,
        format_args!("        float accumulator = {identity};"),
    );
    line(
        source,
        format_args!("        bool accumulator_nan = false;"),
    );
    line(
        source,
        format_args!("        for (uint column = lane; column < width; column += group_width) {{"),
    );
    emit_lane_coordinates(source, rank, 12);
    emit_addresses_indented(
        source,
        signature.input_dtypes().len(),
        signature.output_dtypes().len(),
        rank,
        12,
    );
    emit_values(source, program, signature, instruction_end, 12);
    line(
        source,
        format_args!("            accumulator_nan = accumulator_nan || f32_is_nan(v{operand});"),
    );
    let combine = match op {
        RedOp::Sum => format!("accumulator + v{operand}"),
        RedOp::Max => format!("max(accumulator, v{operand})"),
        RedOp::Min => format!("min(accumulator, v{operand})"),
    };
    line(source, format_args!("            accumulator = {combine};"));
    line(source, format_args!("        }}"));
    let simd_reduce = match op {
        RedOp::Sum => "simd_sum",
        RedOp::Max => "simd_max",
        RedOp::Min => "simd_min",
    };
    emit_threadgroup_reduce(source, reduce_slot, identity, simd_reduce);
    line(source, format_args!("    }}"));
}

fn emit_threadgroup_reduce(
    source: &mut String,
    reduce_slot: usize,
    identity: &str,
    simd_reduce: &str,
) {
    line(
        source,
        format_args!("        accumulator = {simd_reduce}(accumulator);"),
    );
    line(
        source,
        format_args!("        accumulator_nan = simd_any(accumulator_nan);"),
    );
    line(source, format_args!("        if (simd_group == 0) {{"));
    line(
        source,
        format_args!("            partial[simd_lane] = {identity};"),
    );
    line(
        source,
        format_args!("            nan_partial[simd_lane] = 0u;"),
    );
    line(source, format_args!("        }}"));
    line(
        source,
        format_args!("        threadgroup_barrier(mem_flags::mem_threadgroup);"),
    );
    line(source, format_args!("        if (simd_lane == 0) {{"));
    line(
        source,
        format_args!("            partial[simd_group] = accumulator;"),
    );
    line(
        source,
        format_args!("            nan_partial[simd_group] = uint(accumulator_nan);"),
    );
    line(source, format_args!("        }}"));
    line(
        source,
        format_args!("        threadgroup_barrier(mem_flags::mem_threadgroup);"),
    );
    line(source, format_args!("        if (simd_group == 0) {{"));
    line(
        source,
        format_args!("            accumulator = {simd_reduce}(partial[simd_lane]);"),
    );
    line(
        source,
        format_args!("            accumulator_nan = simd_any(nan_partial[simd_lane] != 0u);"),
    );
    line(source, format_args!("            if (simd_lane == 0) {{"));
    line(
        source,
        format_args!(
            "                row_values[{reduce_slot}] = select(accumulator, as_type<float>(0x7fc00000u), accumulator_nan);"
        ),
    );
    line(source, format_args!("            }}"));
    line(source, format_args!("        }}"));
    line(
        source,
        format_args!("        threadgroup_barrier(mem_flags::mem_threadgroup);"),
    );
}

fn emit_output_stage(
    source: &mut String,
    program: &ValidatedProgram,
    signature: &KernelSignature,
    rank: usize,
) {
    line(
        source,
        format_args!("    for (uint column = lane; column < width; column += group_width) {{"),
    );
    emit_lane_coordinates(source, rank, 8);
    emit_addresses_indented(
        source,
        signature.input_dtypes().len(),
        signature.output_dtypes().len(),
        rank,
        8,
    );
    emit_values(source, program, signature, program.program().insts.len(), 8);
    emit_stores(source, program, 8);
    line(source, format_args!("    }}"));
}

fn emit_stores(source: &mut String, program: &ValidatedProgram, indent: usize) {
    for &(slot, value) in &program.program().outputs {
        let slot = usize::try_from(slot).unwrap_or(usize::MAX);
        let expression = format!(
            "store_float(output{slot}, output_address{slot}, {}, v{value});",
            output_dtype_name(slot)
        );
        line(source, format_args!("{:indent$}{expression}", ""));
    }
}

fn emit_lane_coordinates(source: &mut String, rank: usize, indent: usize) {
    line(
        source,
        format_args!("{:indent$}uint coord{} = column;", "", rank - 1),
    );
}

fn emit_addresses_indented(
    source: &mut String,
    input_count: usize,
    output_count: usize,
    rank: usize,
    indent: usize,
) {
    for (name, slot) in operands(input_count, output_count) {
        line(
            source,
            format_args!(
                "{:indent$}ulong {name}_address{slot} = {};",
                "",
                address_expression(&format!("{name}{slot}_layout"), rank)
            ),
        );
    }
}

fn emit_values(
    source: &mut String,
    program: &ValidatedProgram,
    signature: &KernelSignature,
    end: usize,
    indent: usize,
) {
    let mut types = Vec::with_capacity(end);
    let mut reduction = 0;
    for (index, &instruction) in program.program().insts[..end].iter().enumerate() {
        if matches!(instruction, Inst::Reduce(_, _)) {
            types.push(ValueType::F32);
            line(
                source,
                format_args!("{:indent$}float v{index} = row_values[{reduction}];", ""),
            );
            reduction += 1;
            continue;
        }
        let (value_type, expression) =
            instruction_expression(instruction, signature.input_dtypes(), &types);
        types.push(value_type);
        line(
            source,
            format_args!(
                "{:indent$}{} v{index} = {expression};",
                "",
                type_name(value_type)
            ),
        );
    }
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
    inputs: &[DType],
    types: &[ValueType],
) -> (ValueType, String) {
    match instruction {
        Inst::Input(slot) => {
            let slot = usize::try_from(slot).unwrap_or(usize::MAX);
            match inputs[slot] {
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
        UnOp::Exp => format!("select(precise::exp(v{operand}), {nan}, f32_is_nan(v{operand}))"),
        UnOp::Log => {
            format!(
                "select(precise::log(v{operand}), {nan}, f32_is_nan(v{operand}) || v{operand} < 0.0f)"
            )
        }
        UnOp::Sqrt => {
            format!(
                "select(precise::sqrt(v{operand}), {nan}, f32_is_nan(v{operand}) || v{operand} < 0.0f)"
            )
        }
        UnOp::Rsqrt => format!(
            "select(precise::rsqrt(v{operand}), {nan}, f32_is_nan(v{operand}) || v{operand} < 0.0f)"
        ),
        UnOp::Sin => {
            format!("select(precise::sin(v{operand}), {nan}, f32_is_non_finite(v{operand}))")
        }
        UnOp::Cos => {
            format!("select(precise::cos(v{operand}), {nan}, f32_is_non_finite(v{operand}))")
        }
        UnOp::Tanh => format!(
            "select(select(precise::tanh(v{operand}), copysign(1.0f, v{operand}), \
             abs(v{operand}) > 8.0f), {nan}, f32_is_nan(v{operand}))"
        ),
        UnOp::Sigmoid => {
            format!(
                "select(1.0f / (1.0f + precise::exp(-v{operand})), {nan}, f32_is_nan(v{operand}))"
            )
        }
        UnOp::Recip => format!("select(1.0f / v{operand}, {nan}, f32_is_nan(v{operand}))"),
        UnOp::Floor => format!("select(floor(v{operand}), {nan}, f32_is_nan(v{operand}))"),
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
        (BinOp::Lt, ValueType::F32) => format!("f32_lt(v{left}, v{right})"),
        (BinOp::Le, ValueType::F32) => format!("f32_le(v{left}, v{right})"),
        (BinOp::Eq, ValueType::F32) => format!("f32_eq(v{left}, v{right})"),
        (BinOp::Ne, ValueType::F32) => format!("f32_ne(v{left}, v{right})"),
        (BinOp::Ge, ValueType::F32) => format!("f32_ge(v{left}, v{right})"),
        (BinOp::Gt, ValueType::F32) => format!("f32_gt(v{left}, v{right})"),
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
         as_type<float>(0x7fc00000u), f32_is_nan(v{left}) || f32_is_nan(v{right}))"
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
             v{operand} >= 4294967296.0f), 0u, f32_is_nan(v{operand}))"
        ),
        (ValueType::Bool, ValueType::F32) => format!("f32_ne(v{operand}, 0.0f)"),
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
    use crate::rounding_program;
    use forja_core::{
        DType,
        program::{BinOp, Inst, KernelSignature, Program, ProgramKind, RedOp, UnOp, ValueType},
    };
    use forja_testing::{TensorSpec, program::well_typed_programs};
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn sigmoid_rounding_context_uses_the_shared_store() {
        let source = bound_source(
            &rounding_program::context_program(),
            &[DType::BF16, DType::U32],
            &[DType::F16],
            &[1],
            true,
        );
        assert!(source.contains("float v19 = select(1.0f /"));
        assert!(source.contains("float v29 = select(v19, v5, v8);"));
        assert!(source.contains("store_float(output0, output_address0, output_dtype, v29);"));
        assert!(source.contains("f32_to_f16_rne_bits(value)"));
        assert!(!source.contains("half(value)"));
        assert!(!source.contains("store_bfloat_bits"));
    }

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
        assert!(source.contains("f32_is_nan(v0) || f32_is_nan(v1)"));
        assert!(source.contains("(as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u"));
        assert!(source.contains("clamp(v2, 0.0f, 4294967040.0f)"));
        assert!(source.contains("uint v5 = v4 + v4;"));
        assert!(source.contains("precise::exp(v3)"));
    }

    #[test]
    fn row_sum_has_stable_source() {
        let source = source(
            &Program {
                kind: ProgramKind::Row,
                insts: vec![Inst::Input(0), Inst::Reduce(RedOp::Sum, 0)],
                outputs: vec![(0, 1)],
            },
            &[DType::F32],
            &[DType::F32],
            &[7],
        );
        assert_eq!(
            source.strip_prefix(SHARED_HEADER).unwrap(),
            r"
kernel void forja_row(
    device const uchar *input0 [[buffer(0)]],
    device uchar *output0 [[buffer(1)]],
    constant TensorLayout &input0_layout [[buffer(2)]],
    constant TensorLayout &output0_layout [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    threadgroup uint nan_partial[32];
    threadgroup float row_values[4];
    uint width = output0_layout.shape[0];
    {
        float accumulator = 0.0f;
        bool accumulator_nan = false;
        for (uint column = lane; column < width; column += group_width) {
            uint coord0 = column;
            ulong input_address0 = input0_layout.offset + ulong(coord0) * input0_layout.strides[0];
            ulong output_address0 = output0_layout.offset + ulong(coord0) * output0_layout.strides[0];
            float v0 = load_float(input0, input_address0, input0_dtype);
            accumulator_nan = accumulator_nan || f32_is_nan(v0);
            accumulator = accumulator + v0;
        }
        accumulator = simd_sum(accumulator);
        accumulator_nan = simd_any(accumulator_nan);
        if (simd_group == 0) {
            partial[simd_lane] = 0.0f;
            nan_partial[simd_lane] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            partial[simd_group] = accumulator;
            nan_partial[simd_group] = uint(accumulator_nan);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            accumulator = simd_sum(partial[simd_lane]);
            accumulator_nan = simd_any(nan_partial[simd_lane] != 0u);
            if (simd_lane == 0) {
                row_values[0] = select(accumulator, as_type<float>(0x7fc00000u), accumulator_nan);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint column = lane; column < width; column += group_width) {
        uint coord0 = column;
        ulong input_address0 = input0_layout.offset + ulong(coord0) * input0_layout.strides[0];
        ulong output_address0 = output0_layout.offset + ulong(coord0) * output0_layout.strides[0];
        float v0 = load_float(input0, input_address0, input0_dtype);
        float v1 = row_values[0];
        store_float(output0, output_address0, output_dtype, v1);
    }
}
"
        );
    }

    #[test]
    fn resident_rows_keep_values_across_reductions() {
        let program = Program {
            kind: ProgramKind::Row,
            insts: vec![
                Inst::Input(0),
                Inst::Reduce(RedOp::Max, 0),
                Inst::Binary(BinOp::Sub, 0, 1),
            ],
            outputs: vec![(0, 2)],
        };
        let source = bound_source(
            &program,
            &[DType::F32],
            &[DType::F32],
            &[REGISTER_RESIDENT_WIDTH],
            true,
        );
        assert_eq!(source.matches("load_float(input0").count(), 1);
        assert!(source.contains("float accumulator = select(-INFINITY, v0, active);"));
        assert!(source.contains("float v2 = v0 - v1;"));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn addresses_never_depend_on_program_values(
            case in well_typed_programs(64),
            resident in any::<bool>(),
        ) {
            let inputs = case.inputs().iter().map(TensorSpec::dtype).collect::<Vec<_>>();
            let outputs = case.outputs().iter().map(TensorSpec::dtype).collect::<Vec<_>>();
            let source = bound_source(
                case.program(),
                &inputs,
                &outputs,
                case.shape(),
                resident,
            );
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
        bound_source(program, inputs, outputs, shape, false)
    }

    fn bound_source(
        program: &Program,
        inputs: &[DType],
        outputs: &[DType],
        shape: &[u32],
        resident: bool,
    ) -> String {
        let program = program.validate().unwrap();
        let signature = KernelSignature::new(
            u8::try_from(shape.len()).unwrap(),
            inputs.to_vec(),
            outputs.to_vec(),
            0,
        );
        generate(&program, &signature, resident)
    }
}
