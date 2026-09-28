use std::collections::{HashMap, HashSet};

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{BinOp, Expr, ExprBinary, ExprMethodCall, Lit, Pat, Stmt, Type, UnOp, spanned::Spanned};

use super::{ComputeType, KernelKind};

const MAX_INSTRUCTIONS: usize = 256;
const MAX_REDUCTIONS: usize = 4;

#[derive(Clone, Copy)]
pub(super) enum Parameter {
    Tensor {
        slot: u32,
        compute: ComputeType,
        span: Span,
    },
    Scalar(ComputeType),
}

pub(super) struct Lowered {
    pub(super) statements: Vec<TokenStream>,
    pub(super) outputs: Vec<Ident>,
}

pub(super) fn lower(
    body: &syn::Block,
    parameters: HashMap<String, Parameter>,
    context: &Ident,
    kind: KernelKind,
) -> syn::Result<Lowered> {
    let tensors = parameters
        .iter()
        .filter_map(|(name, parameter)| match parameter {
            Parameter::Tensor { span, .. } => Some((name.clone(), *span)),
            Parameter::Scalar(_) => None,
        })
        .collect::<Vec<_>>();
    let mut lowerer = Lowerer {
        bindings: parameters
            .into_iter()
            .map(|(name, parameter)| (name, Binding::Parameter(parameter)))
            .collect(),
        inputs: HashMap::new(),
        used_tensors: HashSet::new(),
        statements: Vec::new(),
        instruction_spans: Vec::new(),
        reduction_spans: Vec::new(),
        context: context.clone(),
        kind,
    };
    let result = lowerer.lower_block(body)?;
    let outputs = match result {
        BlockResult::Value(value) => vec![value],
        BlockResult::Tuple(values) => values,
    };
    if outputs.is_empty() || outputs.len() > 4 {
        return Err(kernel_error(body, "kernel must return one to four values"));
    }
    for (index, value) in outputs.iter().enumerate() {
        if value.ty != ValueType::F32 {
            return Err(kernel_error(
                value.span,
                format!(
                    "output {index} has type `{}`; kernel outputs must be `f32`",
                    value.ty.name()
                ),
            ));
        }
    }
    if lowerer.instruction_spans.len() > MAX_INSTRUCTIONS {
        return Err(kernel_error(
            lowerer.instruction_spans[MAX_INSTRUCTIONS],
            format!(
                "kernel is too large: {} instructions (IR limit is {MAX_INSTRUCTIONS})",
                lowerer.instruction_spans.len()
            ),
        ));
    }
    if lowerer.reduction_spans.len() > MAX_REDUCTIONS {
        return Err(kernel_error(
            lowerer.reduction_spans[MAX_REDUCTIONS],
            format!(
                "too many reductions: {} (IR limit is {MAX_REDUCTIONS})",
                lowerer.reduction_spans.len()
            ),
        ));
    }
    if let Some((name, span)) = tensors
        .iter()
        .find(|(name, _)| !lowerer.used_tensors.contains(name))
    {
        return Err(kernel_error(
            *span,
            format!("tensor parameter `{name}` is never used (every input slot must be read)"),
        ));
    }
    Ok(Lowered {
        statements: lowerer.statements,
        outputs: outputs.into_iter().map(|value| value.ident).collect(),
    })
}

#[derive(Clone)]
enum Binding {
    Parameter(Parameter),
    Value(Value),
}

enum BlockResult {
    Value(Value),
    Tuple(Vec<Value>),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ValueType {
    F32,
    U32,
    Bool,
}

impl ValueType {
    const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::Bool => "bool",
        }
    }
}

#[derive(Clone)]
struct Value {
    ident: Ident,
    ty: ValueType,
    span: Span,
}

struct Lowerer {
    bindings: HashMap<String, Binding>,
    inputs: HashMap<u32, Value>,
    used_tensors: HashSet<String>,
    statements: Vec<TokenStream>,
    instruction_spans: Vec<Span>,
    reduction_spans: Vec<Span>,
    context: Ident,
    kind: KernelKind,
}

impl Lowerer {
    fn lower_block(&mut self, block: &syn::Block) -> syn::Result<BlockResult> {
        let saved = self.bindings.clone();
        let mut result = None;
        for statement in &block.stmts {
            match statement {
                Stmt::Local(local) => {
                    let (name, annotation) = binding_pattern(&local.pat)?;
                    let init = local.init.as_ref().ok_or_else(|| {
                        kernel_error(local, "kernel `let` bindings need an initializer")
                    })?;
                    if init.diverge.is_some() {
                        return Err(kernel_error(
                            &init.expr,
                            "`let else` is not supported in kernels",
                        ));
                    }
                    let value = self.lower_expr(&init.expr)?;
                    if let Some((ty, span)) = annotation
                        && ty != value.ty
                    {
                        return Err(type_mismatch(span, ty, value.ty));
                    }
                    self.bindings.insert(name, Binding::Value(value));
                }
                Stmt::Expr(expression, None) => {
                    result = Some(self.lower_result(expression)?);
                }
                Stmt::Expr(expression, Some(_)) => {
                    return Err(kernel_error(
                        expression,
                        "only `let` statements and a final expression are supported in kernels",
                    ));
                }
                Stmt::Item(item) => {
                    return Err(kernel_error(item, "items are not supported inside kernels"));
                }
                Stmt::Macro(mac) => {
                    return Err(kernel_error(
                        mac,
                        "macro calls are not supported in kernels",
                    ));
                }
            }
        }
        self.bindings = saved;
        result.ok_or_else(|| kernel_error(block, "kernel body needs a final expression"))
    }

    fn lower_result(&mut self, expression: &Expr) -> syn::Result<BlockResult> {
        if let Expr::Tuple(tuple) = expression {
            return tuple
                .elems
                .iter()
                .map(|item| self.lower_expr(item))
                .collect::<syn::Result<Vec<_>>>()
                .map(BlockResult::Tuple);
        }
        self.lower_expr(expression).map(BlockResult::Value)
    }

    fn lower_expr(&mut self, expression: &Expr) -> syn::Result<Value> {
        if let Some(constant) = self.constant_expression(expression)? {
            let context = self.context.clone();
            let constant_name = format_ident!(
                "__forja_constant_{}",
                self.statements.len(),
                span = Span::mixed_site()
            );
            return Ok(self.emit(
                quote!({
                    let #constant_name: f32 = #constant;
                    #context.constant(#constant_name)
                }),
                ValueType::F32,
                expression.span(),
            ));
        }
        match expression {
            Expr::Path(path) => self.lower_path(path),
            Expr::Lit(literal) => self.lower_literal(literal),
            Expr::Binary(binary) => self.lower_binary(binary),
            Expr::Unary(unary) => {
                let value = self.lower_expr(&unary.expr)?;
                match unary.op {
                    UnOp::Neg(_) if value.ty == ValueType::F32 => {
                        let ident = value.ident;
                        Ok(self.emit(quote!(-#ident), ValueType::F32, unary.span()))
                    }
                    UnOp::Not(_) if value.ty == ValueType::Bool => {
                        let ident = value.ident;
                        let value = self.emit(quote!(#ident.not()), ValueType::Bool, unary.span());
                        self.account_hidden_instructions(4, unary.span());
                        Ok(value)
                    }
                    _ => Err(kernel_error(
                        unary.op,
                        format!(
                            "operator is not supported for `{}` in kernels",
                            value.ty.name()
                        ),
                    )),
                }
            }
            Expr::MethodCall(call) => self.lower_method(call),
            Expr::Call(call) => self.lower_call(call),
            Expr::Cast(cast) => self.lower_cast(cast),
            Expr::If(if_expression) => self.lower_if(if_expression),
            Expr::Paren(paren) => self.lower_expr(&paren.expr),
            Expr::Group(group) => self.lower_expr(&group.expr),
            Expr::Block(block) => match self.lower_block(&block.block)? {
                BlockResult::Value(value) => Ok(value),
                BlockResult::Tuple(_) => Err(kernel_error(
                    block,
                    "tuple values are only supported as the kernel's final expression",
                )),
            },
            Expr::Tuple(tuple) => Err(kernel_error(
                tuple,
                "tuple values are only supported as the kernel's final expression",
            )),
            Expr::Loop(loop_expression) => Err(kernel_error(
                loop_expression.loop_token,
                "loops are not supported in kernels; express the computation per element, or reduce with `.row_sum()`/`.row_max()`/`.row_min()`",
            )),
            Expr::ForLoop(loop_expression) => Err(kernel_error(
                loop_expression.for_token,
                "loops are not supported in kernels; express the computation per element, or reduce with `.row_sum()`/`.row_max()`/`.row_min()`",
            )),
            Expr::While(loop_expression) => Err(kernel_error(
                loop_expression.while_token,
                "loops are not supported in kernels; express the computation per element, or reduce with `.row_sum()`/`.row_max()`/`.row_min()`",
            )),
            Expr::Closure(closure) => Err(kernel_error(
                closure,
                "closures are not supported in kernels; use a `#[kernel(helper)]` fn",
            )),
            Expr::Return(return_expression) => Err(kernel_error(
                return_expression.return_token,
                "early `return` is not supported; the last expression is the kernel's output",
            )),
            Expr::Index(index) => Err(kernel_error(
                index.bracket_token.span.open(),
                "indexing is not supported (no gather in step 3 IR); bind a sliced view at the call site",
            )),
            Expr::Assign(assign) => Err(kernel_error(
                assign.eq_token,
                "`let mut` and assignment are not supported; shadow with a new `let`",
            )),
            Expr::Macro(mac) => Err(kernel_error(
                mac,
                "macro calls are not supported in kernels",
            )),
            _ => Err(kernel_error(
                expression,
                "expression is not supported in this kernel subset",
            )),
        }
    }

    fn lower_literal(&mut self, literal: &syn::ExprLit) -> syn::Result<Value> {
        let context = self.context.clone();
        match &literal.lit {
            Lit::Int(value) => {
                if !matches!(value.suffix(), "" | "u32") {
                    return Err(kernel_error(value, "integer literals must have type `u32`"));
                }
                let parsed = value.base10_parse::<u64>()?;
                if parsed > 16_777_216 {
                    return Err(kernel_error(
                        value,
                        format!(
                            "integer literal `{value}` exceeds 2^24 and is not exactly representable"
                        ),
                    ));
                }
                let constant = self.emit(
                    quote!(#context.constant(#value as f32)),
                    ValueType::F32,
                    value.span(),
                );
                let constant = constant.ident;
                Ok(self.emit(quote!(#constant.cast_u32()), ValueType::U32, value.span()))
            }
            Lit::Bool(value) => {
                let lowered = self.emit(
                    quote!(#context.boolean(#value)),
                    ValueType::Bool,
                    value.span(),
                );
                self.account_hidden_instructions(1, value.span());
                Ok(lowered)
            }
            Lit::Float(value) => Err(kernel_error(
                value,
                "float literal could not be lowered as an f32 constant",
            )),
            _ => Err(kernel_error(literal, "literal is not supported in kernels")),
        }
    }

    fn lower_path(&mut self, path: &syn::ExprPath) -> syn::Result<Value> {
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return Err(kernel_error(path, "outer constants must have type `f32`"));
        }
        let name = path.path.segments[0].ident.to_string();
        match self.bindings.get(&name).cloned() {
            Some(Binding::Value(value)) => Ok(value),
            Some(Binding::Parameter(Parameter::Tensor { slot, compute, .. })) => {
                self.used_tensors.insert(name);
                if let Some(value) = self.inputs.get(&slot) {
                    return Ok(value.clone());
                }
                let context = self.context.clone();
                let ty = compute_type(compute);
                let expression = match compute {
                    ComputeType::F32 => quote!(#context.input(#slot)),
                    ComputeType::U32 => quote!(#context.input_u32(#slot)),
                };
                let value = self.emit(expression, ty, path.span());
                self.inputs.insert(slot, value.clone());
                Ok(value)
            }
            Some(Binding::Parameter(Parameter::Scalar(compute))) => {
                let ident = &path.path.segments[0].ident;
                let context = self.context.clone();
                match compute {
                    ComputeType::F32 => Ok(self.emit(
                        quote!(#context.constant(#ident)),
                        ValueType::F32,
                        path.span(),
                    )),
                    ComputeType::U32 => {
                        let constant = self.emit(
                            quote!(#context.constant(#ident as f32)),
                            ValueType::F32,
                            path.span(),
                        );
                        let constant = constant.ident;
                        Ok(self.emit(quote!(#constant.cast_u32()), ValueType::U32, path.span()))
                    }
                }
            }
            None => Err(kernel_error(path, "unknown kernel binding")),
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "all Rust binary operators are diagnosed from one type-directed table"
    )]
    fn lower_binary(&mut self, binary: &ExprBinary) -> syn::Result<Value> {
        let left = self.lower_expr(&binary.left)?;
        let right = self.lower_expr(&binary.right)?;
        match binary.op {
            BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) => {
                if left.ty == ValueType::U32 && right.ty == ValueType::U32 {
                    let method = match binary.op {
                        BinOp::Add(_) => "wrapping_add",
                        BinOp::Sub(_) => "wrapping_sub",
                        BinOp::Mul(_) => "wrapping_mul",
                        _ => {
                            return Err(kernel_error(
                                binary.op,
                                "operator is not supported in this kernel subset",
                            ));
                        }
                    };
                    return Err(kernel_error(
                        binary.op,
                        format!(
                            "u32 `{}` is not allowed; use `{method}` for explicit wrapping semantics",
                            binary_operator(&binary.op)
                        ),
                    ));
                }
                require_same_numeric(binary, &left, &right)?;
                let left = left.ident;
                let right = right.ident;
                let operator = &binary.op;
                Ok(self.emit(
                    quote!(#left #operator #right),
                    ValueType::F32,
                    binary.span(),
                ))
            }
            BinOp::Div(_) => {
                if left.ty == ValueType::U32 && right.ty == ValueType::U32 {
                    return Err(kernel_error(
                        binary.op,
                        "integer division is not supported by the IR; convert with `as f32` and use `(a / b).floor()`",
                    ));
                }
                require_same_numeric(binary, &left, &right)?;
                let left = left.ident;
                let right = right.ident;
                Ok(self.emit(quote!(#left / #right), ValueType::F32, binary.span()))
            }
            BinOp::Rem(_) => Err(kernel_error(
                binary.op,
                "remainder `%` is not supported; write `a - (a / b).floor() * b` on f32",
            )),
            BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Eq(_)
            | BinOp::Ne(_)
            | BinOp::Ge(_)
            | BinOp::Gt(_) => {
                require_same_comparable(binary, &left, &right)?;
                let method = comparison_method(&binary.op).ok_or_else(|| {
                    kernel_error(binary.op, "operator is not supported in this kernel subset")
                })?;
                let left = left.ident;
                let right = right.ident;
                Ok(self.emit(
                    quote!(#left.#method(#right)),
                    ValueType::Bool,
                    binary.span(),
                ))
            }
            BinOp::And(_) | BinOp::Or(_) => {
                if left.ty != ValueType::Bool || right.ty != ValueType::Bool {
                    return Err(type_mismatch(
                        binary.op.span(),
                        ValueType::Bool,
                        if left.ty == ValueType::Bool {
                            right.ty
                        } else {
                            left.ty
                        },
                    ));
                }
                let method = if matches!(binary.op, BinOp::And(_)) {
                    format_ident!("and")
                } else {
                    format_ident!("or")
                };
                let left = left.ident;
                let right = right.ident;
                let value = self.emit(
                    quote!(#left.#method(#right)),
                    ValueType::Bool,
                    binary.span(),
                );
                self.account_hidden_instructions(2, binary.span());
                Ok(value)
            }
            _ => Err(kernel_error(
                binary.op,
                "operator is not supported in this kernel subset",
            )),
        }
    }

    fn lower_method(&mut self, call: &ExprMethodCall) -> syn::Result<Value> {
        if let Some(turbofish) = &call.turbofish {
            return Err(kernel_error(
                turbofish,
                "turbofish is not supported in kernels",
            ));
        }
        let name = call.method.to_string();
        if name == "len" {
            return self.lower_len(call);
        }
        let receiver = self.lower_expr(&call.receiver)?;
        if matches!(name.as_str(), "sum" | "mean")
            && receiver.ty == ValueType::F32
            && self.kind == KernelKind::Row
        {
            let replacement = if name == "sum" { "row_sum" } else { "row_mean" };
            return Err(kernel_error(
                &call.method,
                format!("use `.{replacement}()` for row reductions"),
            ));
        }
        if matches!(name.as_str(), "max" | "min")
            && receiver.ty == ValueType::F32
            && call.args.is_empty()
        {
            let replacement = if name == "max" { "row_max" } else { "row_min" };
            return Err(kernel_error(
                &call.method,
                format!("use `.{replacement}()` for row reductions"),
            ));
        }
        if matches!(name.as_str(), "max" | "min") && receiver.ty == ValueType::F32 {
            let replacement = if name == "max" { "maximum" } else { "minimum" };
            return Err(kernel_error(
                &call.method,
                format!(
                    "`.{name}(b)` on f32 is not supported; use `f32::{replacement}` for NaN-propagating semantics"
                ),
            ));
        }
        if let Some(arity) = method_arity(&name, receiver.ty)
            && call.args.len() != arity
        {
            return Err(kernel_error(&call.method, method_arity_error(&name, arity)));
        }
        if name == "powi" {
            if receiver.ty != ValueType::F32 {
                return Err(unknown_method(&call.method, receiver.ty));
            }
            let exponent = powi_exponent(call.args.first().ok_or_else(|| {
                kernel_error(&call.method, "`powi` requires an integer literal")
            })?)?;
            let context = self.context.clone();
            let exponent = self.emit(
                quote!(#context.constant(#exponent as f32)),
                ValueType::F32,
                call.span(),
            );
            let receiver = receiver.ident;
            let exponent = exponent.ident;
            return Ok(self.emit(
                quote!(#receiver.powf(#exponent)),
                ValueType::F32,
                call.span(),
            ));
        }
        if matches!(
            name.as_str(),
            "row_sum" | "row_max" | "row_min" | "row_mean"
        ) {
            return self.lower_reduction(call, receiver);
        }
        let arguments = call
            .args
            .iter()
            .map(|argument| self.lower_expr(argument))
            .collect::<syn::Result<Vec<_>>>()?;
        self.lower_typed_method(call, receiver, &arguments)
    }

    fn lower_typed_method(
        &mut self,
        call: &ExprMethodCall,
        receiver: Value,
        arguments: &[Value],
    ) -> syn::Result<Value> {
        let name = call.method.to_string();
        match (receiver.ty, name.as_str(), arguments) {
            (
                ValueType::F32,
                "abs" | "exp" | "ln" | "sqrt" | "rsqrt" | "recip" | "sin" | "cos" | "tanh"
                | "sigmoid" | "floor",
                [],
            ) => {
                let receiver = receiver.ident;
                let method = &call.method;
                Ok(self.emit(quote!(#receiver.#method()), ValueType::F32, call.span()))
            }
            (ValueType::F32, "maximum" | "minimum" | "powf", [argument]) => {
                require_type(argument, ValueType::F32)?;
                let receiver = receiver.ident;
                let argument = &argument.ident;
                let method = &call.method;
                Ok(self.emit(
                    quote!(#receiver.#method(#argument)),
                    ValueType::F32,
                    call.span(),
                ))
            }
            (ValueType::F32, "clamp", [low, high]) => {
                require_type(low, ValueType::F32)?;
                require_type(high, ValueType::F32)?;
                let receiver = receiver.ident;
                let low = &low.ident;
                let high = &high.ident;
                let maximum =
                    self.emit(quote!(#receiver.maximum(#low)), ValueType::F32, call.span());
                let maximum = maximum.ident;
                Ok(self.emit(quote!(#maximum.minimum(#high)), ValueType::F32, call.span()))
            }
            (ValueType::F32, "ceil", []) => {
                let receiver = receiver.ident;
                let negative = self.emit(quote!(-#receiver), ValueType::F32, call.span());
                let negative = negative.ident;
                let floor = self.emit(quote!(#negative.floor()), ValueType::F32, call.span());
                let floor = floor.ident;
                Ok(self.emit(quote!(-#floor), ValueType::F32, call.span()))
            }
            (ValueType::F32, "log2" | "log10", []) => {
                let receiver = receiver.ident;
                let logarithm = self.emit(quote!(#receiver.ln()), ValueType::F32, call.span());
                let context = self.context.clone();
                let scale = if name == "log2" {
                    quote!(::core::f32::consts::LOG2_E)
                } else {
                    quote!(::core::f32::consts::LOG10_E)
                };
                let scale = self.emit(
                    quote!(#context.constant(#scale)),
                    ValueType::F32,
                    call.span(),
                );
                let logarithm = logarithm.ident;
                let scale = scale.ident;
                Ok(self.emit(quote!(#logarithm * #scale), ValueType::F32, call.span()))
            }
            (ValueType::F32, "exp2", []) => {
                let context = self.context.clone();
                let scale = self.emit(
                    quote!(#context.constant(::core::f32::consts::LN_2)),
                    ValueType::F32,
                    call.span(),
                );
                let receiver = receiver.ident;
                let scale = scale.ident;
                let product = self.emit(quote!(#receiver * #scale), ValueType::F32, call.span());
                let product = product.ident;
                Ok(self.emit(quote!(#product.exp()), ValueType::F32, call.span()))
            }
            (
                ValueType::U32,
                "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "min" | "max",
                [argument],
            ) => {
                require_type(argument, ValueType::U32)?;
                let receiver = receiver.ident;
                let argument = &argument.ident;
                let method = &call.method;
                Ok(self.emit(
                    quote!(#receiver.#method(#argument)),
                    ValueType::U32,
                    call.span(),
                ))
            }
            _ => Err(unknown_method(&call.method, receiver.ty)),
        }
    }

    fn lower_reduction(&mut self, call: &ExprMethodCall, receiver: Value) -> syn::Result<Value> {
        if receiver.ty != ValueType::F32 {
            return Err(unknown_method(&call.method, receiver.ty));
        }
        if self.kind != KernelKind::Row {
            return Err(kernel_error(
                &call.method,
                format!(
                    "reduction `.{}()` is only available in `#[kernel(row)]`",
                    call.method
                ),
            ));
        }
        self.reduction_spans.push(call.method.span());
        let context = self.context.clone();
        let receiver = receiver.ident;
        let reduction = match call.method.to_string().as_str() {
            "row_sum" | "row_mean" => quote!(#context.row_sum(#receiver)),
            "row_max" => quote!(#context.row_max(#receiver)),
            "row_min" => quote!(#context.row_min(#receiver)),
            _ => return Err(unknown_method(&call.method, ValueType::F32)),
        };
        let reduced = self.emit(reduction, ValueType::F32, call.span());
        if call.method != "row_mean" {
            return Ok(reduced);
        }
        let extent = self.emit(quote!(#context.extent(-1)), ValueType::U32, call.span());
        let extent = extent.ident;
        let extent = self.emit(quote!(#extent.cast_f32()), ValueType::F32, call.span());
        let reduced = reduced.ident;
        let extent = extent.ident;
        Ok(self.emit(quote!(#reduced / #extent), ValueType::F32, call.span()))
    }

    fn lower_len(&mut self, call: &ExprMethodCall) -> syn::Result<Value> {
        if !call.args.is_empty() {
            return Err(kernel_error(&call.method, "`len` takes no arguments"));
        }
        let Expr::Path(path) = &*call.receiver else {
            return Err(kernel_error(
                &call.method,
                "`.len()` is only available on a row tensor parameter",
            ));
        };
        let Some(name) = path.path.get_ident().map(ToString::to_string) else {
            return Err(kernel_error(
                &call.method,
                "`.len()` is only available on a row tensor parameter",
            ));
        };
        if self.kind != KernelKind::Row
            || !matches!(
                self.bindings.get(&name),
                Some(Binding::Parameter(Parameter::Tensor { .. }))
            )
        {
            return Err(kernel_error(
                &call.method,
                "`.len()` is only available on a row tensor parameter",
            ));
        }
        let context = self.context.clone();
        Ok(self.emit(quote!(#context.extent(-1)), ValueType::U32, call.span()))
    }

    fn lower_call(&mut self, call: &syn::ExprCall) -> syn::Result<Value> {
        let Expr::Path(path) = &*call.func else {
            return Err(kernel_error(
                call,
                "function calls are not supported in kernels",
            ));
        };
        let Some(segment) = path.path.segments.last() else {
            return Err(kernel_error(call, "unknown kernel function"));
        };
        let name = segment.ident.to_string();
        if !matches!(name.as_str(), "index" | "extent") {
            return Err(kernel_error(
                &segment.ident,
                format!("unknown kernel function `{name}`"),
            ));
        }
        if call.args.len() != 1 {
            return Err(kernel_error(
                &segment.ident,
                format!("`{name}` takes one literal axis"),
            ));
        }
        let axis = axis_literal(&call.args[0])?;
        let context = self.context.clone();
        let method = &segment.ident;
        Ok(self.emit(quote!(#context.#method(#axis)), ValueType::U32, call.span()))
    }

    fn lower_cast(&mut self, cast: &syn::ExprCast) -> syn::Result<Value> {
        let value = self.lower_expr(&cast.expr)?;
        let target = value_type(&cast.ty)
            .ok_or_else(|| kernel_error(&cast.ty, "kernel casts may target only `f32` or `u32`"))?;
        if value.ty == ValueType::Bool && target == ValueType::F32 {
            return Err(kernel_error(
                cast.as_token,
                "`bool as f32` is not allowed; use `if c { 1.0 } else { 0.0 }`",
            ));
        }
        let method = match (value.ty, target) {
            (ValueType::F32 | ValueType::Bool, ValueType::U32) => format_ident!("cast_u32"),
            (ValueType::U32, ValueType::F32) => format_ident!("cast_f32"),
            _ => {
                return Err(kernel_error(
                    cast.as_token,
                    format!(
                        "cast from `{}` to `{}` is not supported in kernels",
                        value.ty.name(),
                        target.name()
                    ),
                ));
            }
        };
        let value = value.ident;
        Ok(self.emit(quote!(#value.#method()), target, cast.span()))
    }

    fn lower_if(&mut self, if_expression: &syn::ExprIf) -> syn::Result<Value> {
        let condition = self.lower_expr(&if_expression.cond)?;
        require_type(&condition, ValueType::Bool)?;
        let accepted = match self.lower_block(&if_expression.then_branch)? {
            BlockResult::Value(value) => value,
            BlockResult::Tuple(_) => {
                return Err(kernel_error(
                    &if_expression.then_branch,
                    "tuple values are only supported as the kernel's final expression",
                ));
            }
        };
        let Some((_, rejected)) = &if_expression.else_branch else {
            return Err(kernel_error(
                if_expression.if_token,
                "`if` without `else` has no value; kernels need both branches (both are evaluated)",
            ));
        };
        let rejected = self.lower_expr(rejected)?;
        if accepted.ty != rejected.ty {
            return Err(type_mismatch(rejected.span, accepted.ty, rejected.ty));
        }
        let method = match accepted.ty {
            ValueType::F32 => format_ident!("select"),
            ValueType::U32 => format_ident!("select_u32"),
            ValueType::Bool => format_ident!("select_bool"),
        };
        let condition = condition.ident;
        let accepted_ident = accepted.ident;
        let rejected_ident = rejected.ident;
        Ok(self.emit(
            quote!(#condition.#method(#accepted_ident, #rejected_ident)),
            accepted.ty,
            if_expression.span(),
        ))
    }

    fn constant_expression(&self, expression: &Expr) -> syn::Result<Option<TokenStream>> {
        match expression {
            Expr::Lit(literal) => match &literal.lit {
                Lit::Float(value) => {
                    if !matches!(value.suffix(), "" | "f32") {
                        return Err(kernel_error(value, "float literals must have type `f32`"));
                    }
                    let parsed = value.base10_parse::<f32>()?;
                    if !parsed.is_finite() {
                        return Err(kernel_error(
                            value,
                            format!("float literal `{value}` is not finite in f32"),
                        ));
                    }
                    Ok(Some(quote!(#value)))
                }
                _ => Ok(None),
            },
            Expr::Path(path) if self.is_outer_constant(path) => Ok(Some(quote!(#path))),
            Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => Ok(self
                .constant_expression(&unary.expr)?
                .map(|value| quote!(-(#value)))),
            Expr::Binary(binary) => {
                let Some(left) = self.constant_expression(&binary.left)? else {
                    return Ok(None);
                };
                let Some(right) = self.constant_expression(&binary.right)? else {
                    return Ok(None);
                };
                let operator = match binary.op {
                    BinOp::Add(_) => quote!(+),
                    BinOp::Sub(_) => quote!(-),
                    BinOp::Mul(_) => quote!(*),
                    BinOp::Div(_) => quote!(/),
                    _ => return Ok(None),
                };
                Ok(Some(quote!((#left) #operator (#right))))
            }
            Expr::Paren(paren) => self.constant_expression(&paren.expr),
            Expr::Group(group) => self.constant_expression(&group.expr),
            _ => Ok(None),
        }
    }

    fn is_outer_constant(&self, path: &syn::ExprPath) -> bool {
        path.qself.is_none()
            && (path.path.segments.len() > 1
                || path
                    .path
                    .get_ident()
                    .is_some_and(|ident| !self.bindings.contains_key(&ident.to_string())))
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "each constructed token stream is transferred into one statement"
    )]
    fn emit(&mut self, expression: TokenStream, ty: ValueType, span: Span) -> Value {
        let ident = format_ident!(
            "__forja_value_{}",
            self.statements.len(),
            span = Span::mixed_site()
        );
        self.statements.push(quote!(let #ident = #expression;));
        self.instruction_spans.push(span);
        Value { ident, ty, span }
    }

    fn account_hidden_instructions(&mut self, count: usize, span: Span) {
        self.instruction_spans
            .extend(std::iter::repeat_n(span, count));
    }
}

fn compute_type(ty: ComputeType) -> ValueType {
    match ty {
        ComputeType::F32 => ValueType::F32,
        ComputeType::U32 => ValueType::U32,
    }
}

fn method_arity(name: &str, ty: ValueType) -> Option<usize> {
    match (ty, name) {
        (
            ValueType::F32,
            "abs" | "exp" | "ln" | "sqrt" | "rsqrt" | "recip" | "sin" | "cos" | "tanh" | "sigmoid"
            | "floor" | "ceil" | "log2" | "log10" | "exp2" | "row_sum" | "row_max" | "row_min"
            | "row_mean",
        ) => Some(0),
        (ValueType::F32, "maximum" | "minimum" | "powf" | "powi") => Some(1),
        (ValueType::F32, "clamp") => Some(2),
        (ValueType::U32, "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "min" | "max") => {
            Some(1)
        }
        _ => None,
    }
}

fn method_arity_error(name: &str, arity: usize) -> String {
    let arguments = match arity {
        0 => "no arguments",
        1 => "one argument",
        2 => "two arguments",
        _ => "the documented number of arguments",
    };
    format!("`{name}` takes {arguments}")
}

fn binding_pattern(pattern: &Pat) -> syn::Result<(String, Option<(ValueType, Span)>)> {
    let (pattern, ty) = match pattern {
        Pat::Type(pattern) => (&*pattern.pat, Some(&*pattern.ty)),
        pattern => (pattern, None),
    };
    let Pat::Ident(pattern) = pattern else {
        return Err(kernel_error(
            pattern,
            "kernel `let` patterns must be identifiers",
        ));
    };
    if let Some(mutability) = pattern.mutability {
        return Err(kernel_error(
            mutability,
            "`let mut` and assignment are not supported; shadow with a new `let`",
        ));
    }
    if pattern.by_ref.is_some() || pattern.subpat.is_some() {
        return Err(kernel_error(
            pattern,
            "kernel `let` bindings must be immutable identifiers",
        ));
    }
    let annotation = ty
        .map(|ty| {
            value_type(ty)
                .map(|value| (value, ty.span()))
                .ok_or_else(|| kernel_error(ty, "kernel bindings may be `f32`, `u32`, or `bool`"))
        })
        .transpose()?;
    Ok((pattern.ident.to_string(), annotation))
}

fn value_type(ty: &Type) -> Option<ValueType> {
    let Type::Path(path) = ty else { return None };
    let ident = path
        .qself
        .is_none()
        .then(|| path.path.get_ident())
        .flatten()?;
    match ident.to_string().as_str() {
        "f32" => Some(ValueType::F32),
        "u32" => Some(ValueType::U32),
        "bool" => Some(ValueType::Bool),
        _ => None,
    }
}

fn require_same_numeric(binary: &ExprBinary, left: &Value, right: &Value) -> syn::Result<()> {
    if left.ty == ValueType::F32 && right.ty == ValueType::F32 {
        return Ok(());
    }
    if left.ty == ValueType::F32 && right.ty == ValueType::U32 {
        return Err(f32_integer_mismatch(&binary.right, right.span));
    }
    if left.ty == ValueType::U32 && right.ty == ValueType::F32 {
        return Err(f32_integer_mismatch(&binary.left, left.span));
    }
    Err(type_mismatch(binary.op.span(), ValueType::F32, right.ty))
}

fn require_same_comparable(binary: &ExprBinary, left: &Value, right: &Value) -> syn::Result<()> {
    if left.ty == right.ty && matches!(left.ty, ValueType::F32 | ValueType::U32) {
        return Ok(());
    }
    if left.ty == ValueType::F32 && right.ty == ValueType::U32 {
        return Err(f32_integer_mismatch(&binary.right, right.span));
    }
    if left.ty == ValueType::U32 && right.ty == ValueType::F32 {
        return Err(f32_integer_mismatch(&binary.left, left.span));
    }
    Err(type_mismatch(binary.op.span(), left.ty, right.ty))
}

fn f32_integer_mismatch(expression: &Expr, fallback: Span) -> syn::Error {
    if let Some(literal) = integer_literal(expression) {
        kernel_error(
            literal,
            format!("integer literal `{literal}` in an f32 expression; write `{literal}.0`"),
        )
    } else {
        type_mismatch(fallback, ValueType::F32, ValueType::U32)
    }
}

fn integer_literal(expression: &Expr) -> Option<&syn::LitInt> {
    match expression {
        Expr::Lit(literal) => match &literal.lit {
            Lit::Int(value) => Some(value),
            _ => None,
        },
        Expr::Paren(paren) => integer_literal(&paren.expr),
        Expr::Group(group) => integer_literal(&group.expr),
        _ => None,
    }
}

fn require_type(value: &Value, expected: ValueType) -> syn::Result<()> {
    if value.ty == expected {
        Ok(())
    } else {
        Err(type_mismatch(value.span, expected, value.ty))
    }
}

fn type_mismatch(span: impl Spanned, expected: ValueType, found: ValueType) -> syn::Error {
    let help = match (expected, found) {
        (ValueType::F32, ValueType::U32) => "; add `as f32`",
        (ValueType::U32, ValueType::F32) => "; use an integer literal or `as u32`",
        _ => "",
    };
    kernel_error(
        span,
        format!(
            "mismatched types: expected `{}`, found `{}`{help}",
            expected.name(),
            found.name()
        ),
    )
}

fn comparison_method(operator: &BinOp) -> Option<Ident> {
    let name = match operator {
        BinOp::Lt(_) => "lt",
        BinOp::Le(_) => "le",
        BinOp::Eq(_) => "equal",
        BinOp::Ne(_) => "not_equal",
        BinOp::Ge(_) => "ge",
        BinOp::Gt(_) => "gt",
        _ => return None,
    };
    Some(format_ident!("{name}"))
}

fn binary_operator(operator: &BinOp) -> &'static str {
    match operator {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        _ => "?",
    }
}

fn unknown_method(method: &Ident, ty: ValueType) -> syn::Error {
    let supported = match ty {
        ValueType::F32 => {
            "abs, exp, ln, sqrt, rsqrt, recip, sin, cos, tanh, sigmoid, floor, maximum, minimum, powf, powi, clamp, ceil, log2, log10, exp2, row_sum, row_max, row_min, row_mean"
        }
        ValueType::U32 => "wrapping_add, wrapping_sub, wrapping_mul, min, max",
        ValueType::Bool => "none",
    };
    kernel_error(
        method,
        format!(
            "unknown method `{method}` on `{}`; supported: {supported}",
            ty.name()
        ),
    )
}

fn axis_literal(expression: &Expr) -> syn::Result<i32> {
    let (negative, literal) = match expression {
        Expr::Lit(literal) => (false, literal),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            let Expr::Lit(literal) = &*unary.expr else {
                return Err(kernel_error(expression, "axis must be an integer literal"));
            };
            (true, literal)
        }
        _ => return Err(kernel_error(expression, "axis must be an integer literal")),
    };
    let Lit::Int(literal) = &literal.lit else {
        return Err(kernel_error(expression, "axis must be an integer literal"));
    };
    let value = literal.base10_parse::<i32>()?;
    if negative {
        value
            .checked_neg()
            .ok_or_else(|| kernel_error(expression, "axis is out of range"))
    } else {
        Ok(value)
    }
}

fn powi_exponent(expression: &Expr) -> syn::Result<i32> {
    let (negative, literal) = match expression {
        Expr::Lit(literal) => (false, literal),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            let Expr::Lit(literal) = &*unary.expr else {
                return Err(kernel_error(
                    expression,
                    "`powi` requires an integer literal",
                ));
            };
            (true, literal)
        }
        _ => {
            return Err(kernel_error(
                expression,
                "`powi` requires an integer literal",
            ));
        }
    };
    let Lit::Int(literal) = &literal.lit else {
        return Err(kernel_error(
            expression,
            "`powi` requires an integer literal",
        ));
    };
    let value = literal.base10_parse::<i32>()?;
    if negative {
        value
            .checked_neg()
            .ok_or_else(|| kernel_error(expression, "`powi` exponent is out of range"))
    } else {
        Ok(value)
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "accepting tokens and spans by value keeps diagnostic call sites uniform"
)]
fn kernel_error(span: impl Spanned, message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(span.span(), format!("kernel: {message}"))
}
