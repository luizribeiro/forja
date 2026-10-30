use forja_core::program::{BinOp, Inst, UnOp, ValueType};

pub(super) fn instruction_expression<I, X, E>(
    instruction: Inst,
    types: &[ValueType],
    input: I,
    index: X,
    extent: E,
) -> (ValueType, String)
where
    I: FnOnce(u32) -> (ValueType, String),
    X: FnOnce(u8) -> String,
    E: FnOnce(u8) -> String,
{
    match instruction {
        Inst::Input(slot) => input(slot),
        Inst::Const(value) => (
            ValueType::F32,
            format!("as_type<float>(0x{:08x}u)", value.to_bits()),
        ),
        Inst::Index(axis) => (ValueType::U32, index(axis)),
        Inst::Extent(axis) => (ValueType::U32, extent(axis)),
        Inst::Unary(op, operand) => (ValueType::F32, unary(op, operand)),
        Inst::Binary(op, left, right) => binary(op, left, right, types[left as usize]),
        Inst::Select(condition, accepted, rejected) => (
            types[accepted as usize],
            format!("select(v{rejected}, v{accepted}, v{condition})"),
        ),
        Inst::Cast(to, operand) => (to, cast(to, operand, types[operand as usize])),
        Inst::Reduce(_, _) => unreachable!("scalar expressions cannot contain reductions"),
    }
}

pub(super) const fn type_name(value_type: ValueType) -> &'static str {
    match value_type {
        ValueType::F32 => "float",
        ValueType::U32 => "uint",
        ValueType::Bool => "bool",
    }
}

fn unary(op: UnOp, operand: u32) -> String {
    let nan = "as_type<float>(0x7fc00000u)";
    match op {
        UnOp::Neg => format!("-v{operand}"),
        UnOp::Abs => format!("abs(v{operand})"),
        UnOp::Exp => format!("select(precise::exp(v{operand}), {nan}, f32_is_nan(v{operand}))"),
        UnOp::Log1p => format!("f32_log1p(v{operand})"),
        UnOp::Log => format!(
            "select(precise::log(v{operand}), {nan}, f32_is_nan(v{operand}) || v{operand} < 0.0f)"
        ),
        UnOp::Sqrt => format!(
            "select(precise::sqrt(v{operand}), {nan}, f32_is_nan(v{operand}) || v{operand} < 0.0f)"
        ),
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
        UnOp::Sigmoid => format!(
            "select(1.0f / (1.0f + precise::exp(-v{operand})), {nan}, f32_is_nan(v{operand}))"
        ),
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
