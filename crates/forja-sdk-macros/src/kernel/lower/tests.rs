use std::collections::HashMap;

use forja_core::program::{
    BinOp, Inst, Program, ProgramKind, RedOp, UnOp, ValueType as CoreValueType,
};
use forja_cpu::interpreter::Input;
use forja_testing::assert_f32_values_agree;
use proc_macro2::{Ident, Span, TokenStream};
use proptest::{
    prelude::*,
    test_runner::{RngSeed, TestCaseError},
};
use quote::quote;
use syn::{BinOp as SynBinOp, Expr, Lit, Pat, Stmt, UnOp as SynUnOp};

use super::{Lowered, Parameter, lower};
use crate::kernel::{ComputeType, KernelKind};

#[derive(Clone, Debug)]
enum Tree {
    Input,
    Constant(i8),
    Neg(Box<Self>),
    Abs(Box<Self>),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Minimum(Box<Self>, Box<Self>),
    Maximum(Box<Self>, Box<Self>),
    FromU32(Box<U32Tree>),
    Select(Box<BoolTree>, Box<Self>, Box<Self>),
    Helper(Box<Self>),
    Reduce(Reduction, Box<Self>),
}

#[derive(Clone, Debug)]
enum U32Tree {
    Input,
    Constant(u8),
    Index,
    Extent,
    Len,
    WrappingAdd(Box<Self>, Box<Self>),
    WrappingSub(Box<Self>, Box<Self>),
    WrappingMul(Box<Self>, Box<Self>),
    Min(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    FromF32(Box<Tree>),
    FromBool(Box<BoolTree>),
    Select(Box<BoolTree>, Box<Self>, Box<Self>),
}

#[derive(Clone, Debug)]
enum BoolTree {
    CompareF32(Comparison, Box<Tree>, Box<Tree>),
    CompareU32(Comparison, Box<U32Tree>, Box<U32Tree>),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Select(Box<Self>, Box<Self>, Box<Self>),
}

#[derive(Clone, Copy, Debug)]
enum Comparison {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Sum,
    Max,
    Min,
    Mean,
}

#[derive(Default)]
struct Coverage {
    u32_leaves: usize,
    wrapping_add: usize,
    wrapping_sub: usize,
    wrapping_mul: usize,
    u32_min: usize,
    u32_max: usize,
    comparisons: usize,
    logical_not: usize,
    logical_and: usize,
    logical_or: usize,
    selects: usize,
    f32_to_u32: usize,
    u32_to_f32: usize,
    bool_to_u32: usize,
    helpers: usize,
    row_sum: usize,
    row_max: usize,
    row_min: usize,
    row_mean: usize,
    index: usize,
    extent: usize,
    len: usize,
}

fn trees() -> impl Strategy<Value = Tree> {
    (-8_i8..=8, -8_i8..=8, 0_u8..=31, any::<u8>(), any::<u8>()).prop_map(
        |(left_constant, right_constant, integer, f_comparison, u_comparison)| {
            comprehensive_tree(
                left_constant,
                right_constant,
                integer,
                comparison(f_comparison),
                comparison(u_comparison),
            )
        },
    )
}

fn comprehensive_tree(
    left_constant: i8,
    right_constant: i8,
    integer: u8,
    f_comparison: Comparison,
    u_comparison: Comparison,
) -> Tree {
    let value = Tree::Add(
        Box::new(Tree::Input),
        Box::new(Tree::Constant(left_constant)),
    );
    let alternate = Tree::Maximum(
        Box::new(Tree::Abs(Box::new(Tree::Neg(Box::new(Tree::Sub(
            Box::new(value.clone()),
            Box::new(Tree::Constant(right_constant)),
        )))))),
        Box::new(Tree::Constant(right_constant)),
    );
    let wrapped = U32Tree::WrappingMul(
        Box::new(U32Tree::WrappingSub(
            Box::new(U32Tree::WrappingAdd(
                Box::new(U32Tree::Input),
                Box::new(U32Tree::Constant(integer)),
            )),
            Box::new(U32Tree::Index),
        )),
        Box::new(U32Tree::Constant(3)),
    );
    let bounded = U32Tree::Max(
        Box::new(U32Tree::Min(Box::new(wrapped), Box::new(U32Tree::Extent))),
        Box::new(U32Tree::Len),
    );
    let f_condition = BoolTree::CompareF32(
        f_comparison,
        Box::new(value.clone()),
        Box::new(alternate.clone()),
    );
    let u_condition = BoolTree::CompareU32(
        u_comparison,
        Box::new(bounded.clone()),
        Box::new(U32Tree::Constant(integer)),
    );
    let condition = BoolTree::Or(
        Box::new(BoolTree::And(
            Box::new(f_condition.clone()),
            Box::new(BoolTree::Not(Box::new(u_condition.clone()))),
        )),
        Box::new(BoolTree::Select(
            Box::new(f_condition.clone()),
            Box::new(u_condition.clone()),
            Box::new(BoolTree::Not(Box::new(f_condition.clone()))),
        )),
    );
    let selected_u32 = U32Tree::Select(
        Box::new(u_condition),
        Box::new(bounded.clone()),
        Box::new(U32Tree::FromF32(Box::new(value.clone()))),
    );
    let selected = Tree::Select(
        Box::new(condition.clone()),
        Box::new(Tree::Helper(Box::new(value.clone()))),
        Box::new(Tree::FromU32(Box::new(selected_u32))),
    );
    let casts = Tree::Add(
        Box::new(Tree::FromU32(Box::new(U32Tree::FromF32(Box::new(
            alternate.clone(),
        ))))),
        Box::new(Tree::FromU32(Box::new(U32Tree::FromBool(Box::new(
            f_condition,
        ))))),
    );
    let reductions = Tree::Add(
        Box::new(Tree::Reduce(
            Reduction::Sum,
            Box::new(Tree::Helper(Box::new(value))),
        )),
        Box::new(Tree::Add(
            Box::new(Tree::Reduce(Reduction::Max, Box::new(selected))),
            Box::new(Tree::Add(
                Box::new(Tree::Reduce(Reduction::Min, Box::new(alternate.clone()))),
                Box::new(Tree::Reduce(
                    Reduction::Mean,
                    Box::new(Tree::FromU32(Box::new(bounded))),
                )),
            )),
        )),
    );
    Tree::Add(
        Box::new(Tree::Minimum(Box::new(casts), Box::new(alternate))),
        Box::new(Tree::Add(
            Box::new(Tree::Mul(
                Box::new(Tree::Input),
                Box::new(Tree::Constant(0)),
            )),
            Box::new(reductions),
        )),
    )
}

proptest! {
    #![proptest_config(property_config())]

    #[test]
    fn accepted_syntax_validates_and_matches_rust(tree in trees()) {
        let coverage = tree.coverage();
        assert_complete_coverage(&coverage)?;
        let tokens = tree.tokens();
        let program = lower_program(quote!({ #tokens }))
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let program = program
            .validate()
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let floats = (-16_i16..=16)
            .map(|value| f32::from(value) / 8.0)
            .collect::<Vec<_>>();
        let integers = (0..floats.len())
            .map(|index| u32::MAX.wrapping_sub(u32::try_from(index).unwrap().wrapping_mul(7_919)))
            .collect::<Vec<_>>();
        let interpreted = forja_cpu::interpreter::interpret(
            &program,
            &[u32::try_from(floats.len()).unwrap()],
            &[Input::F32(&floats), Input::U32(&integers)],
        )
        .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let inputs = EvalInputs { floats: &floats, integers: &integers };
        let expected = (0..floats.len())
            .map(|lane| tree.evaluate(lane, inputs))
            .collect::<Vec<_>>();
        assert_f32_values_agree(&expected, &interpreted[0])
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
    }

    #[test]
    fn over_cap_syntax_is_rejected(excess in 1_usize..32) {
        let mut expression = quote!(x);
        for _ in 0..(128 + excess) {
            expression = quote!((#expression + 1.0));
        }
        let instruction_error = lower_body(quote!({ #expression }))
            .err()
            .expect("generated expression must exceed the instruction cap");
        prop_assert!(instruction_error.to_string().contains("kernel is too large"));

        let mut expression = quote!(x.row_sum());
        for _ in 0..(4 + excess) {
            expression = quote!((#expression + x.row_sum()));
        }
        let reduction_error = lower_body(quote!({ #expression }))
            .err()
            .expect("generated expression must exceed the reduction cap");
        prop_assert!(reduction_error.to_string().contains("too many reductions"));
    }
}

fn property_config() -> ProptestConfig {
    let explore = std::env::var_os("FORJA_PROPTEST_EXPLORE").is_some();
    ProptestConfig {
        cases: if explore { 512 } else { 64 },
        rng_seed: if explore {
            RngSeed::Random
        } else {
            RngSeed::Fixed(0xbb67_ae85_84ca_a73b)
        },
        ..ProptestConfig::default()
    }
}

fn assert_complete_coverage(coverage: &Coverage) -> Result<(), TestCaseError> {
    let counts = [
        ("u32 leaves", coverage.u32_leaves),
        ("wrapping_add", coverage.wrapping_add),
        ("wrapping_sub", coverage.wrapping_sub),
        ("wrapping_mul", coverage.wrapping_mul),
        ("u32 min", coverage.u32_min),
        ("u32 max", coverage.u32_max),
        ("comparisons", coverage.comparisons),
        ("logical not", coverage.logical_not),
        ("logical and", coverage.logical_and),
        ("logical or", coverage.logical_or),
        ("selects", coverage.selects),
        ("f32 to u32", coverage.f32_to_u32),
        ("u32 to f32", coverage.u32_to_f32),
        ("bool to u32", coverage.bool_to_u32),
        ("helpers", coverage.helpers),
        ("row_sum", coverage.row_sum),
        ("row_max", coverage.row_max),
        ("row_min", coverage.row_min),
        ("row_mean", coverage.row_mean),
        ("index", coverage.index),
        ("extent", coverage.extent),
        ("len", coverage.len),
    ];
    for (construct, count) in counts {
        if count == 0 {
            return Err(TestCaseError::fail(format!(
                "generated tree did not cover {construct}"
            )));
        }
    }
    Ok(())
}

impl Tree {
    fn tokens(&self) -> TokenStream {
        match self {
            Self::Input => quote!(x),
            Self::Constant(value) => constant_tokens(*value),
            Self::Neg(value) => {
                let value = value.tokens();
                quote!(-(#value))
            }
            Self::Abs(value) => {
                let value = value.tokens();
                quote!((#value).abs())
            }
            Self::Add(left, right) => binary_tokens(left, right, &quote!(+)),
            Self::Sub(left, right) => binary_tokens(left, right, &quote!(-)),
            Self::Mul(left, right) => binary_tokens(left, right, &quote!(*)),
            Self::Minimum(left, right) => method_tokens(left, right, "minimum"),
            Self::Maximum(left, right) => method_tokens(left, right, "maximum"),
            Self::FromU32(value) => {
                let value = value.tokens();
                quote!((#value) as f32)
            }
            Self::Select(condition, accepted, rejected) => {
                let condition = condition.tokens();
                let accepted = accepted.tokens();
                let rejected = rejected.tokens();
                quote!(if #condition { #accepted } else { #rejected })
            }
            Self::Helper(value) => {
                let value = value.tokens();
                quote!(property_square(#value))
            }
            Self::Reduce(reduction, value) => {
                let value = value.tokens();
                let method = Ident::new(reduction.method(), Span::call_site());
                quote!((#value).#method())
            }
        }
    }

    #[allow(clippy::cast_precision_loss, reason = "the oracle models kernel casts")]
    fn evaluate(&self, lane: usize, inputs: EvalInputs<'_>) -> f32 {
        match self {
            Self::Input => inputs.floats[lane],
            Self::Constant(value) => f32::from(*value) / 4.0,
            Self::Neg(value) => -value.evaluate(lane, inputs),
            Self::Abs(value) => value.evaluate(lane, inputs).abs(),
            Self::Add(left, right) => left.evaluate(lane, inputs) + right.evaluate(lane, inputs),
            Self::Sub(left, right) => left.evaluate(lane, inputs) - right.evaluate(lane, inputs),
            Self::Mul(left, right) => left.evaluate(lane, inputs) * right.evaluate(lane, inputs),
            Self::Minimum(left, right) => left
                .evaluate(lane, inputs)
                .min(right.evaluate(lane, inputs)),
            Self::Maximum(left, right) => left
                .evaluate(lane, inputs)
                .max(right.evaluate(lane, inputs)),
            Self::FromU32(value) => value.evaluate(lane, inputs) as f32,
            Self::Select(condition, accepted, rejected) => {
                if condition.evaluate(lane, inputs) {
                    accepted.evaluate(lane, inputs)
                } else {
                    rejected.evaluate(lane, inputs)
                }
            }
            Self::Helper(value) => {
                let value = value.evaluate(lane, inputs);
                value * value
            }
            Self::Reduce(reduction, value) => reduction.evaluate(value, inputs),
        }
    }

    fn coverage(&self) -> Coverage {
        let mut coverage = Coverage::default();
        self.record_coverage(&mut coverage);
        coverage
    }

    fn record_coverage(&self, coverage: &mut Coverage) {
        match self {
            Self::Input | Self::Constant(_) => {}
            Self::Neg(value) | Self::Abs(value) | Self::Helper(value) => {
                if matches!(self, Self::Helper(_)) {
                    coverage.helpers += 1;
                }
                value.record_coverage(coverage);
            }
            Self::Add(left, right)
            | Self::Sub(left, right)
            | Self::Mul(left, right)
            | Self::Minimum(left, right)
            | Self::Maximum(left, right) => {
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::FromU32(value) => {
                coverage.u32_to_f32 += 1;
                value.record_coverage(coverage);
            }
            Self::Select(condition, accepted, rejected) => {
                coverage.selects += 1;
                condition.record_coverage(coverage);
                accepted.record_coverage(coverage);
                rejected.record_coverage(coverage);
            }
            Self::Reduce(reduction, value) => {
                match reduction {
                    Reduction::Sum => coverage.row_sum += 1,
                    Reduction::Max => coverage.row_max += 1,
                    Reduction::Min => coverage.row_min += 1,
                    Reduction::Mean => coverage.row_mean += 1,
                }
                value.record_coverage(coverage);
            }
        }
    }
}

impl U32Tree {
    fn tokens(&self) -> TokenStream {
        match self {
            Self::Input => quote!(ids),
            Self::Constant(value) => {
                let value = u32::from(*value);
                quote!(#value)
            }
            Self::Index => quote!(forja_sdk::kernel::index(-1)),
            Self::Extent => quote!(forja_sdk::kernel::extent(-1)),
            Self::Len => quote!(x.len()),
            Self::WrappingAdd(left, right) => u32_method_tokens(left, right, "wrapping_add"),
            Self::WrappingSub(left, right) => u32_method_tokens(left, right, "wrapping_sub"),
            Self::WrappingMul(left, right) => u32_method_tokens(left, right, "wrapping_mul"),
            Self::Min(left, right) => u32_method_tokens(left, right, "min"),
            Self::Max(left, right) => u32_method_tokens(left, right, "max"),
            Self::FromF32(value) => {
                let value = value.tokens();
                quote!((#value) as u32)
            }
            Self::FromBool(value) => {
                let value = value.tokens();
                quote!((#value) as u32)
            }
            Self::Select(condition, accepted, rejected) => {
                let condition = condition.tokens();
                let accepted = accepted.tokens();
                let rejected = rejected.tokens();
                quote!(if #condition { #accepted } else { #rejected })
            }
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the oracle models kernel casts"
    )]
    fn evaluate(&self, lane: usize, inputs: EvalInputs<'_>) -> u32 {
        match self {
            Self::Input => inputs.integers[lane],
            Self::Constant(value) => u32::from(*value),
            Self::Index => u32::try_from(lane).unwrap(),
            Self::Extent | Self::Len => u32::try_from(inputs.floats.len()).unwrap(),
            Self::WrappingAdd(left, right) => left
                .evaluate(lane, inputs)
                .wrapping_add(right.evaluate(lane, inputs)),
            Self::WrappingSub(left, right) => left
                .evaluate(lane, inputs)
                .wrapping_sub(right.evaluate(lane, inputs)),
            Self::WrappingMul(left, right) => left
                .evaluate(lane, inputs)
                .wrapping_mul(right.evaluate(lane, inputs)),
            Self::Min(left, right) => left
                .evaluate(lane, inputs)
                .min(right.evaluate(lane, inputs)),
            Self::Max(left, right) => left
                .evaluate(lane, inputs)
                .max(right.evaluate(lane, inputs)),
            Self::FromF32(value) => value.evaluate(lane, inputs) as u32,
            Self::FromBool(value) => u32::from(value.evaluate(lane, inputs)),
            Self::Select(condition, accepted, rejected) => {
                if condition.evaluate(lane, inputs) {
                    accepted.evaluate(lane, inputs)
                } else {
                    rejected.evaluate(lane, inputs)
                }
            }
        }
    }

    fn record_coverage(&self, coverage: &mut Coverage) {
        match self {
            Self::Input | Self::Constant(_) => coverage.u32_leaves += 1,
            Self::Index => coverage.index += 1,
            Self::Extent => coverage.extent += 1,
            Self::Len => coverage.len += 1,
            Self::WrappingAdd(left, right)
            | Self::WrappingSub(left, right)
            | Self::WrappingMul(left, right)
            | Self::Min(left, right)
            | Self::Max(left, right) => {
                match self {
                    Self::WrappingAdd(_, _) => coverage.wrapping_add += 1,
                    Self::WrappingSub(_, _) => coverage.wrapping_sub += 1,
                    Self::WrappingMul(_, _) => coverage.wrapping_mul += 1,
                    Self::Min(_, _) => coverage.u32_min += 1,
                    Self::Max(_, _) => coverage.u32_max += 1,
                    _ => {}
                }
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::FromF32(value) => {
                coverage.f32_to_u32 += 1;
                value.record_coverage(coverage);
            }
            Self::FromBool(value) => {
                coverage.bool_to_u32 += 1;
                value.record_coverage(coverage);
            }
            Self::Select(condition, accepted, rejected) => {
                coverage.selects += 1;
                condition.record_coverage(coverage);
                accepted.record_coverage(coverage);
                rejected.record_coverage(coverage);
            }
        }
    }
}

impl BoolTree {
    fn tokens(&self) -> TokenStream {
        match self {
            Self::CompareF32(comparison, left, right) => {
                comparison.tokens(&left.tokens(), &right.tokens())
            }
            Self::CompareU32(comparison, left, right) => {
                comparison.tokens(&left.tokens(), &right.tokens())
            }
            Self::Not(value) => {
                let value = value.tokens();
                quote!(!(#value))
            }
            Self::And(left, right) => bool_binary_tokens(left, right, &quote!(&&)),
            Self::Or(left, right) => bool_binary_tokens(left, right, &quote!(||)),
            Self::Select(condition, accepted, rejected) => {
                let condition = condition.tokens();
                let accepted = accepted.tokens();
                let rejected = rejected.tokens();
                quote!(if #condition { #accepted } else { #rejected })
            }
        }
    }

    fn evaluate(&self, lane: usize, inputs: EvalInputs<'_>) -> bool {
        match self {
            Self::CompareF32(comparison, left, right) => {
                comparison.evaluate(&left.evaluate(lane, inputs), &right.evaluate(lane, inputs))
            }
            Self::CompareU32(comparison, left, right) => {
                comparison.evaluate(&left.evaluate(lane, inputs), &right.evaluate(lane, inputs))
            }
            Self::Not(value) => !value.evaluate(lane, inputs),
            Self::And(left, right) => left.evaluate(lane, inputs) & right.evaluate(lane, inputs),
            Self::Or(left, right) => left.evaluate(lane, inputs) | right.evaluate(lane, inputs),
            Self::Select(condition, accepted, rejected) => {
                if condition.evaluate(lane, inputs) {
                    accepted.evaluate(lane, inputs)
                } else {
                    rejected.evaluate(lane, inputs)
                }
            }
        }
    }

    fn record_coverage(&self, coverage: &mut Coverage) {
        match self {
            Self::CompareF32(_, left, right) => {
                coverage.comparisons += 1;
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::CompareU32(_, left, right) => {
                coverage.comparisons += 1;
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::Not(value) => {
                coverage.logical_not += 1;
                value.record_coverage(coverage);
            }
            Self::And(left, right) => {
                coverage.logical_and += 1;
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::Or(left, right) => {
                coverage.logical_or += 1;
                left.record_coverage(coverage);
                right.record_coverage(coverage);
            }
            Self::Select(condition, accepted, rejected) => {
                coverage.selects += 1;
                condition.record_coverage(coverage);
                accepted.record_coverage(coverage);
                rejected.record_coverage(coverage);
            }
        }
    }
}

impl Comparison {
    fn tokens(self, left: &TokenStream, right: &TokenStream) -> TokenStream {
        match self {
            Self::Lt => quote!((#left) < (#right)),
            Self::Le => quote!((#left) <= (#right)),
            Self::Eq => quote!((#left) == (#right)),
            Self::Ne => quote!((#left) != (#right)),
            Self::Ge => quote!((#left) >= (#right)),
            Self::Gt => quote!((#left) > (#right)),
        }
    }

    fn evaluate<T: PartialOrd + PartialEq>(self, left: &T, right: &T) -> bool {
        match self {
            Self::Lt => left < right,
            Self::Le => left <= right,
            Self::Eq => left == right,
            Self::Ne => left != right,
            Self::Ge => left >= right,
            Self::Gt => left > right,
        }
    }
}

impl Reduction {
    const fn method(self) -> &'static str {
        match self {
            Self::Sum => "row_sum",
            Self::Max => "row_max",
            Self::Min => "row_min",
            Self::Mean => "row_mean",
        }
    }

    fn evaluate(self, value: &Tree, inputs: EvalInputs<'_>) -> f32 {
        let mut values = (0..inputs.floats.len()).map(|lane| value.evaluate(lane, inputs));
        match self {
            Self::Sum | Self::Mean => {
                let sum = values.fold(0.0, |sum, value| sum + value);
                if matches!(self, Self::Mean) {
                    sum / f32::from(u16::try_from(inputs.floats.len()).unwrap())
                } else {
                    sum
                }
            }
            Self::Max => values
                .next()
                .map_or(f32::NAN, |first| values.fold(first, f32::max)),
            Self::Min => values
                .next()
                .map_or(f32::NAN, |first| values.fold(first, f32::min)),
        }
    }
}

#[derive(Clone, Copy)]
struct EvalInputs<'a> {
    floats: &'a [f32],
    integers: &'a [u32],
}

fn comparison(value: u8) -> Comparison {
    match value % 6 {
        0 => Comparison::Lt,
        1 => Comparison::Le,
        2 => Comparison::Eq,
        3 => Comparison::Ne,
        4 => Comparison::Ge,
        _ => Comparison::Gt,
    }
}

fn binary_tokens(left: &Tree, right: &Tree, operator: &TokenStream) -> TokenStream {
    let left = left.tokens();
    let right = right.tokens();
    quote!((#left) #operator (#right))
}

fn method_tokens(left: &Tree, right: &Tree, method: &str) -> TokenStream {
    let left = left.tokens();
    let right = right.tokens();
    let method = Ident::new(method, Span::call_site());
    quote!((#left).#method(#right))
}

fn u32_method_tokens(left: &U32Tree, right: &U32Tree, method: &str) -> TokenStream {
    let left = left.tokens();
    let right = right.tokens();
    let method = Ident::new(method, Span::call_site());
    quote!((#left).#method(#right))
}

fn bool_binary_tokens(left: &BoolTree, right: &BoolTree, operator: &TokenStream) -> TokenStream {
    let left = left.tokens();
    let right = right.tokens();
    quote!((#left) #operator (#right))
}

fn constant_tokens(value: i8) -> TokenStream {
    let magnitude = f32::from(value.unsigned_abs()) / 4.0;
    let magnitude = syn::LitFloat::new(&format!("{magnitude:.2}f32"), Span::call_site());
    if value < 0 {
        quote!(-#magnitude)
    } else {
        quote!(#magnitude)
    }
}

fn lower_body(tokens: TokenStream) -> syn::Result<Lowered> {
    let body = syn::parse2(tokens)?;
    lower(
        &body,
        HashMap::from([
            (
                String::from("x"),
                Parameter::Tensor {
                    slot: 0,
                    compute: ComputeType::F32,
                    span: Span::call_site(),
                },
            ),
            (
                String::from("ids"),
                Parameter::Tensor {
                    slot: 1,
                    compute: ComputeType::U32,
                    span: Span::call_site(),
                },
            ),
        ]),
        &Ident::new("context", Span::call_site()),
        KernelKind::Row,
        None,
    )
}

fn lower_program(tokens: TokenStream) -> syn::Result<Program> {
    let lowered = lower_body(tokens)?;
    let mut names = HashMap::new();
    let mut insts = Vec::with_capacity(lowered.statements.len());
    for statement in lowered.statements {
        let Stmt::Local(local) = syn::parse2(statement)? else {
            return Err(syn::Error::new(
                Span::call_site(),
                "lowered non-local statement",
            ));
        };
        let Pat::Ident(pattern) = local.pat else {
            return Err(syn::Error::new(Span::call_site(), "lowered non-identifier"));
        };
        let init = local
            .init
            .ok_or_else(|| syn::Error::new(Span::call_site(), "lowered uninitialized value"))?;
        let index = lowered_instructions(&init.expr, &names, &mut insts)?;
        names.insert(pattern.ident.to_string(), index);
    }
    let outputs = lowered
        .outputs
        .iter()
        .enumerate()
        .map(|(slot, output)| {
            Ok((
                u32::try_from(slot)
                    .map_err(|_| syn::Error::new(output.span(), "output slot overflow"))?,
                value_index(output, &names)?,
            ))
        })
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(Program {
        kind: ProgramKind::Row,
        insts,
        outputs,
    })
}

fn lowered_instructions(
    expression: &Expr,
    names: &HashMap<String, u32>,
    insts: &mut Vec<Inst>,
) -> syn::Result<u32> {
    match expression {
        Expr::Block(block) => push_inst(insts, lowered_constant(&block.block)?),
        Expr::Unary(unary) if matches!(unary.op, SynUnOp::Neg(_)) => push_inst(
            insts,
            Inst::Unary(UnOp::Neg, expression_index(&unary.expr, names)?),
        ),
        Expr::Binary(binary) => push_inst(
            insts,
            Inst::Binary(
                match binary.op {
                    SynBinOp::Add(_) => BinOp::Add,
                    SynBinOp::Sub(_) => BinOp::Sub,
                    SynBinOp::Mul(_) => BinOp::Mul,
                    SynBinOp::Div(_) => BinOp::Div,
                    _ => return Err(syn::Error::new_spanned(binary.op, "unexpected binary op")),
                },
                expression_index(&binary.left, names)?,
                expression_index(&binary.right, names)?,
            ),
        ),
        Expr::Call(call) => {
            let Some(function) = path_name(&call.func) else {
                return Err(syn::Error::new_spanned(call, "helper is not a path"));
            };
            if function != "property_square" {
                return Err(syn::Error::new_spanned(call, "unexpected helper"));
            }
            let argument = call
                .args
                .get(1)
                .ok_or_else(|| syn::Error::new_spanned(call, "helper argument is missing"))?;
            let value = expression_index(argument, names)?;
            push_inst(insts, Inst::Binary(BinOp::Mul, value, value))
        }
        Expr::MethodCall(call) => lowered_method(call, names, insts),
        _ => Err(syn::Error::new_spanned(
            expression,
            "unexpected lowered expression",
        )),
    }
}

fn lowered_method(
    call: &syn::ExprMethodCall,
    names: &HashMap<String, u32>,
    insts: &mut Vec<Inst>,
) -> syn::Result<u32> {
    let receiver_name = path_name(&call.receiver);
    let method = call.method.to_string();
    if receiver_name.as_deref() == Some("context") {
        return lower_context_method(call, &method, names, insts);
    }
    let receiver = expression_index(&call.receiver, names)?;
    if let Some(operation) = match method.as_str() {
        "abs" => Some(UnOp::Abs),
        "sin" => Some(UnOp::Sin),
        "cos" => Some(UnOp::Cos),
        "tanh" => Some(UnOp::Tanh),
        "sigmoid" => Some(UnOp::Sigmoid),
        _ => None,
    } {
        return push_inst(insts, Inst::Unary(operation, receiver));
    }
    if let Some(target) = match method.as_str() {
        "cast_f32" => Some(CoreValueType::F32),
        "cast_u32" => Some(CoreValueType::U32),
        _ => None,
    } {
        return push_inst(insts, Inst::Cast(target, receiver));
    }
    match method.as_str() {
        "not" => lower_not(receiver, insts),
        "and" => lower_and(receiver, method_argument(call, names)?, insts),
        "or" => lower_or(receiver, method_argument(call, names)?, insts),
        "select" | "select_u32" | "select_bool" => {
            let accepted = call
                .args
                .first()
                .ok_or_else(|| syn::Error::new_spanned(call, "select value is missing"))?;
            let rejected = call
                .args
                .get(1)
                .ok_or_else(|| syn::Error::new_spanned(call, "select value is missing"))?;
            push_inst(
                insts,
                Inst::Select(
                    receiver,
                    expression_index(accepted, names)?,
                    expression_index(rejected, names)?,
                ),
            )
        }
        _ => {
            let operation = match method.as_str() {
                "minimum" | "min" => BinOp::Min,
                "maximum" | "max" => BinOp::Max,
                "wrapping_add" => BinOp::Add,
                "wrapping_sub" => BinOp::Sub,
                "wrapping_mul" => BinOp::Mul,
                "lt" => BinOp::Lt,
                "le" => BinOp::Le,
                "equal" => BinOp::Eq,
                "not_equal" => BinOp::Ne,
                "ge" => BinOp::Ge,
                "gt" => BinOp::Gt,
                _ => return Err(syn::Error::new_spanned(&call.method, "unexpected method")),
            };
            push_inst(
                insts,
                Inst::Binary(operation, receiver, method_argument(call, names)?),
            )
        }
    }
}

fn lower_context_method(
    call: &syn::ExprMethodCall,
    method: &str,
    names: &HashMap<String, u32>,
    insts: &mut Vec<Inst>,
) -> syn::Result<u32> {
    match method {
        "input" | "input_u32" => {
            let Some(Expr::Lit(literal)) = call.args.first() else {
                return Err(syn::Error::new_spanned(call, "input slot is not literal"));
            };
            let Lit::Int(slot) = &literal.lit else {
                return Err(syn::Error::new_spanned(
                    literal,
                    "input slot is not integer",
                ));
            };
            push_inst(insts, Inst::Input(slot.base10_parse()?))
        }
        "constant" => {
            let value = call
                .args
                .first()
                .ok_or_else(|| syn::Error::new_spanned(call, "constant is missing"))?;
            push_inst(insts, Inst::Const(evaluate_constant(value)?))
        }
        "index" => push_inst(insts, Inst::Index(0)),
        "extent" => push_inst(insts, Inst::Extent(0)),
        "row_sum" | "row_max" | "row_min" => {
            let operation = match method {
                "row_sum" => RedOp::Sum,
                "row_max" => RedOp::Max,
                "row_min" => RedOp::Min,
                _ => return Err(syn::Error::new_spanned(call, "unexpected reduction")),
            };
            push_inst(
                insts,
                Inst::Reduce(operation, method_argument(call, names)?),
            )
        }
        _ => Err(syn::Error::new_spanned(
            &call.method,
            "unexpected context method",
        )),
    }
}

fn lower_not(receiver: u32, insts: &mut Vec<Inst>) -> syn::Result<u32> {
    let false_value = push_inst(insts, Inst::Const(0.0))?;
    let false_value = push_inst(insts, Inst::Cast(CoreValueType::Bool, false_value))?;
    let true_value = push_inst(insts, Inst::Const(1.0))?;
    let true_value = push_inst(insts, Inst::Cast(CoreValueType::Bool, true_value))?;
    push_inst(insts, Inst::Select(receiver, false_value, true_value))
}

fn lower_and(receiver: u32, other: u32, insts: &mut Vec<Inst>) -> syn::Result<u32> {
    let false_value = push_inst(insts, Inst::Const(0.0))?;
    let false_value = push_inst(insts, Inst::Cast(CoreValueType::Bool, false_value))?;
    push_inst(insts, Inst::Select(receiver, other, false_value))
}

fn lower_or(receiver: u32, other: u32, insts: &mut Vec<Inst>) -> syn::Result<u32> {
    let true_value = push_inst(insts, Inst::Const(1.0))?;
    let true_value = push_inst(insts, Inst::Cast(CoreValueType::Bool, true_value))?;
    push_inst(insts, Inst::Select(receiver, true_value, other))
}

fn method_argument(call: &syn::ExprMethodCall, names: &HashMap<String, u32>) -> syn::Result<u32> {
    call.args
        .first()
        .ok_or_else(|| syn::Error::new_spanned(call, "method argument is missing"))
        .and_then(|argument| expression_index(argument, names))
}

fn push_inst(insts: &mut Vec<Inst>, instruction: Inst) -> syn::Result<u32> {
    let index = u32::try_from(insts.len())
        .map_err(|_| syn::Error::new(Span::call_site(), "test program is too large"))?;
    insts.push(instruction);
    Ok(index)
}

fn lowered_constant(block: &syn::Block) -> syn::Result<Inst> {
    let Some(Stmt::Local(local)) = block.stmts.first() else {
        return Err(syn::Error::new_spanned(
            block,
            "constant block has no binding",
        ));
    };
    let init = local
        .init
        .as_ref()
        .ok_or_else(|| syn::Error::new_spanned(local, "constant binding has no value"))?;
    Ok(Inst::Const(evaluate_constant(&init.expr)?))
}

#[allow(clippy::cast_precision_loss, reason = "the oracle models kernel casts")]
fn evaluate_constant(expression: &Expr) -> syn::Result<f32> {
    match expression {
        Expr::Lit(literal) => match &literal.lit {
            Lit::Float(value) => value.base10_parse(),
            Lit::Int(value) => value.base10_parse::<u32>().map(|value| value as f32),
            _ => Err(syn::Error::new_spanned(literal, "constant is not numeric")),
        },
        Expr::Unary(unary) if matches!(unary.op, SynUnOp::Neg(_)) => {
            Ok(-evaluate_constant(&unary.expr)?)
        }
        Expr::Paren(paren) => evaluate_constant(&paren.expr),
        Expr::Cast(cast) => evaluate_constant(&cast.expr),
        Expr::Binary(binary) => {
            let left = evaluate_constant(&binary.left)?;
            let right = evaluate_constant(&binary.right)?;
            match binary.op {
                SynBinOp::Add(_) => Ok(left + right),
                SynBinOp::Sub(_) => Ok(left - right),
                SynBinOp::Mul(_) => Ok(left * right),
                SynBinOp::Div(_) => Ok(left / right),
                _ => Err(syn::Error::new_spanned(
                    binary.op,
                    "constant op is unsupported",
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "constant expression is unsupported",
        )),
    }
}

fn expression_index(expression: &Expr, names: &HashMap<String, u32>) -> syn::Result<u32> {
    let name = path_name(expression)
        .ok_or_else(|| syn::Error::new_spanned(expression, "value is not an identifier"))?;
    names
        .get(&name)
        .copied()
        .ok_or_else(|| syn::Error::new_spanned(expression, "value is not defined"))
}

fn value_index(ident: &Ident, names: &HashMap<String, u32>) -> syn::Result<u32> {
    names
        .get(&ident.to_string())
        .copied()
        .ok_or_else(|| syn::Error::new(ident.span(), "output is not defined"))
}

fn path_name(expression: &Expr) -> Option<String> {
    let Expr::Path(path) = expression else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}
