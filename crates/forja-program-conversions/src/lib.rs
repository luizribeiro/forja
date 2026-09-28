//! Shared scalar-program enum conversion generation.

#![forbid(unsafe_code)]

/// Generates conversions for scalar-program operation enums whose variants have matching names.
#[macro_export]
macro_rules! program_op_conversions {
    (
        $visibility:vis fn $unop_fn:ident($source_unop:ident => $target_unop:ident);
        $visibility_binop:vis fn $binop_fn:ident($source_binop:ident => $target_binop:ident);
        $visibility_redop:vis fn $redop_fn:ident($source_redop:ident => $target_redop:ident);
    ) => {
        $visibility const fn $unop_fn(value: $source_unop) -> $target_unop {
            match value {
                $source_unop::Neg => $target_unop::Neg,
                $source_unop::Abs => $target_unop::Abs,
                $source_unop::Exp => $target_unop::Exp,
                $source_unop::Log => $target_unop::Log,
                $source_unop::Sqrt => $target_unop::Sqrt,
                $source_unop::Rsqrt => $target_unop::Rsqrt,
                $source_unop::Sin => $target_unop::Sin,
                $source_unop::Cos => $target_unop::Cos,
                $source_unop::Tanh => $target_unop::Tanh,
                $source_unop::Sigmoid => $target_unop::Sigmoid,
                $source_unop::Recip => $target_unop::Recip,
                $source_unop::Floor => $target_unop::Floor,
            }
        }

        $visibility_binop const fn $binop_fn(value: $source_binop) -> $target_binop {
            match value {
                $source_binop::Add => $target_binop::Add,
                $source_binop::Sub => $target_binop::Sub,
                $source_binop::Mul => $target_binop::Mul,
                $source_binop::Div => $target_binop::Div,
                $source_binop::Min => $target_binop::Min,
                $source_binop::Max => $target_binop::Max,
                $source_binop::Pow => $target_binop::Pow,
                $source_binop::Lt => $target_binop::Lt,
                $source_binop::Le => $target_binop::Le,
                $source_binop::Eq => $target_binop::Eq,
                $source_binop::Ne => $target_binop::Ne,
                $source_binop::Ge => $target_binop::Ge,
                $source_binop::Gt => $target_binop::Gt,
            }
        }

        $visibility_redop const fn $redop_fn(value: $source_redop) -> $target_redop {
            match value {
                $source_redop::Sum => $target_redop::Sum,
                $source_redop::Max => $target_redop::Max,
                $source_redop::Min => $target_redop::Min,
            }
        }
    };
}

/// Generates a conversion for scalar-program value-type enums with matching variants.
#[macro_export]
macro_rules! program_value_type_conversion {
    ($visibility:vis fn $function:ident($source:ident => $target:ident);) => {
        $visibility const fn $function(value: $source) -> $target {
            match value {
                $source::F32 => $target::F32,
                $source::U32 => $target::U32,
                $source::Bool => $target::Bool,
            }
        }
    };
}

/// Selects the f32 variant of a scalar-program value-type enum.
#[macro_export]
macro_rules! program_f32_value_type {
    ($target:ident) => {
        $target::F32
    };
}
