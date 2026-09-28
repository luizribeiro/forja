use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{BinOp, Expr, ExprMethodCall, Lit, Pat, Stmt, Type, UnOp, spanned::Spanned};

use super::ComputeType;

#[derive(Clone, Copy)]
pub(super) enum Parameter {
    Tensor { slot: u32, compute: ComputeType },
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
) -> syn::Result<Lowered> {
    let mut lowerer = Lowerer {
        bindings: parameters
            .into_iter()
            .map(|(name, parameter)| (name, Binding::Parameter(parameter)))
            .collect(),
        inputs: HashMap::new(),
        statements: Vec::new(),
        next_value: 0,
        context: context.clone(),
    };
    let result = lowerer.lower_block(body)?;
    let outputs = match result {
        BlockResult::Value(value) => vec![value],
        BlockResult::Tuple(values) => values,
    };
    if outputs.is_empty() || outputs.len() > 4 {
        return Err(syn::Error::new(
            body.span(),
            "kernel must return one to four values",
        ));
    }
    Ok(Lowered {
        statements: lowerer.statements,
        outputs,
    })
}

#[derive(Clone)]
enum Binding {
    Parameter(Parameter),
    Value(Ident),
}

enum BlockResult {
    Value(Ident),
    Tuple(Vec<Ident>),
}

struct Lowerer {
    bindings: HashMap<String, Binding>,
    inputs: HashMap<u32, Ident>,
    statements: Vec<TokenStream>,
    next_value: usize,
    context: Ident,
}

impl Lowerer {
    fn lower_block(&mut self, block: &syn::Block) -> syn::Result<BlockResult> {
        let saved = self.bindings.clone();
        let mut result = None;
        for statement in &block.stmts {
            match statement {
                Stmt::Local(local) => {
                    let (name, ty) = binding_pattern(&local.pat)?;
                    if let Some(ty) = ty {
                        require_f32(ty)?;
                    }
                    let init = local.init.as_ref().ok_or_else(|| {
                        syn::Error::new_spanned(local, "kernel `let` bindings need an initializer")
                    })?;
                    if init.diverge.is_some() {
                        return Err(syn::Error::new_spanned(
                            &init.expr,
                            "`let else` is not supported in kernels",
                        ));
                    }
                    let value = self.lower_expr(&init.expr)?;
                    self.bindings.insert(name, Binding::Value(value));
                }
                Stmt::Expr(expression, None) => {
                    result = Some(self.lower_result(expression)?);
                }
                Stmt::Expr(expression, Some(_)) => {
                    return Err(syn::Error::new_spanned(
                        expression,
                        "only `let` statements and a final expression are supported in kernels",
                    ));
                }
                Stmt::Item(item) => {
                    return Err(syn::Error::new_spanned(
                        item,
                        "items are not supported inside kernels",
                    ));
                }
                Stmt::Macro(mac) => {
                    return Err(syn::Error::new_spanned(
                        mac,
                        "macro calls are not supported in kernels",
                    ));
                }
            }
        }
        self.bindings = saved;
        result.ok_or_else(|| syn::Error::new(block.span(), "kernel body needs a final expression"))
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

    fn lower_expr(&mut self, expression: &Expr) -> syn::Result<Ident> {
        if let Some(constant) = self.constant_expression(expression)? {
            let context = &self.context;
            let constant_name = format_ident!(
                "__forja_constant_{}",
                self.next_value,
                span = Span::mixed_site()
            );
            return Ok(self.emit(quote!(
                {
                    let #constant_name: f32 = #constant;
                    #context.constant(#constant_name)
                }
            )));
        }
        match expression {
            Expr::Path(path) => self.lower_path(path),
            Expr::Binary(binary) => {
                let left = self.lower_expr(&binary.left)?;
                let right = self.lower_expr(&binary.right)?;
                let operator = match binary.op {
                    BinOp::Add(_) => quote!(+),
                    BinOp::Sub(_) => quote!(-),
                    BinOp::Mul(_) => quote!(*),
                    BinOp::Div(_) => quote!(/),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            binary.op,
                            "operator is not supported in this kernel subset",
                        ));
                    }
                };
                Ok(self.emit(quote!(#left #operator #right)))
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
                let value = self.lower_expr(&unary.expr)?;
                Ok(self.emit(quote!(-#value)))
            }
            Expr::MethodCall(call) => self.lower_method(call),
            Expr::Paren(paren) => self.lower_expr(&paren.expr),
            Expr::Group(group) => self.lower_expr(&group.expr),
            Expr::Block(block) => match self.lower_block(&block.block)? {
                BlockResult::Value(value) => Ok(value),
                BlockResult::Tuple(_) => Err(syn::Error::new_spanned(
                    block,
                    "tuple values are only supported as the kernel's final expression",
                )),
            },
            Expr::Tuple(tuple) => Err(syn::Error::new_spanned(
                tuple,
                "tuple values are only supported as the kernel's final expression",
            )),
            _ => Err(syn::Error::new_spanned(
                expression,
                "expression is not supported in this kernel subset",
            )),
        }
    }

    fn lower_path(&mut self, path: &syn::ExprPath) -> syn::Result<Ident> {
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return Err(syn::Error::new_spanned(
                path,
                "outer constants must have type `f32`",
            ));
        }
        let name = path.path.segments[0].ident.to_string();
        match self.bindings.get(&name).cloned() {
            Some(Binding::Value(value)) => Ok(value),
            Some(Binding::Parameter(Parameter::Tensor {
                slot,
                compute: ComputeType::F32,
            })) => {
                if let Some(value) = self.inputs.get(&slot) {
                    return Ok(value.clone());
                }
                let context = &self.context;
                let value = self.emit(quote!(#context.input(#slot)));
                self.inputs.insert(slot, value.clone());
                Ok(value)
            }
            Some(Binding::Parameter(Parameter::Scalar(ComputeType::F32))) => {
                let ident = &path.path.segments[0].ident;
                let context = &self.context;
                Ok(self.emit(quote!(#context.constant(#ident))))
            }
            Some(Binding::Parameter(_)) => Err(syn::Error::new_spanned(
                path,
                "u32 expressions are outside the core kernel subset",
            )),
            None => Err(syn::Error::new_spanned(path, "unknown kernel binding")),
        }
    }

    fn lower_method(&mut self, call: &ExprMethodCall) -> syn::Result<Ident> {
        if call.turbofish.is_some() {
            return Err(syn::Error::new_spanned(
                call,
                "turbofish is not supported in kernels",
            ));
        }
        let name = call.method.to_string();
        if matches!(name.as_str(), "max" | "min") {
            return Err(syn::Error::new_spanned(
                call,
                "use `maximum`/`minimum` for NaN-propagating f32 semantics",
            ));
        }
        if let Some(arity) = method_arity(&name)
            && call.args.len() != arity
        {
            return Err(syn::Error::new_spanned(
                call,
                method_arity_error(&name, arity),
            ));
        }
        let receiver = self.lower_expr(&call.receiver)?;
        if name == "powi" {
            let exponent = powi_exponent(call.args.first().ok_or_else(|| {
                syn::Error::new_spanned(call, "`powi` requires an integer literal")
            })?)?;
            let context = &self.context;
            let exponent = self.emit(quote!(#context.constant(#exponent as f32)));
            return Ok(self.emit(quote!(#receiver.powf(#exponent))));
        }
        let arguments = call
            .args
            .iter()
            .map(|argument| self.lower_expr(argument))
            .collect::<syn::Result<Vec<_>>>()?;
        match (name.as_str(), arguments.as_slice()) {
            (
                "abs" | "exp" | "ln" | "sqrt" | "rsqrt" | "recip" | "sin" | "cos" | "tanh"
                | "sigmoid" | "floor",
                [],
            ) => {
                let method = &call.method;
                Ok(self.emit(quote!(#receiver.#method())))
            }
            ("maximum" | "minimum" | "powf", [argument]) => {
                let method = &call.method;
                Ok(self.emit(quote!(#receiver.#method(#argument))))
            }
            ("clamp", [low, high]) => {
                let maximum = self.emit(quote!(#receiver.maximum(#low)));
                Ok(self.emit(quote!(#maximum.minimum(#high))))
            }
            ("ceil", []) => {
                let negative = self.emit(quote!(-#receiver));
                let floor = self.emit(quote!(#negative.floor()));
                Ok(self.emit(quote!(-#floor)))
            }
            ("log2", []) => {
                let logarithm = self.emit(quote!(#receiver.ln()));
                let context = &self.context;
                let scale = self.emit(quote!(#context.constant(::core::f32::consts::LOG2_E)));
                Ok(self.emit(quote!(#logarithm * #scale)))
            }
            ("log10", []) => {
                let logarithm = self.emit(quote!(#receiver.ln()));
                let context = &self.context;
                let scale = self.emit(quote!(#context.constant(::core::f32::consts::LOG10_E)));
                Ok(self.emit(quote!(#logarithm * #scale)))
            }
            ("exp2", []) => {
                let context = &self.context;
                let scale = self.emit(quote!(#context.constant(::core::f32::consts::LN_2)));
                let product = self.emit(quote!(#receiver * #scale));
                Ok(self.emit(quote!(#product.exp())))
            }
            _ => Err(syn::Error::new_spanned(
                call,
                format!("method `{name}` is not supported in this kernel subset"),
            )),
        }
    }

    fn constant_expression(&self, expression: &Expr) -> syn::Result<Option<TokenStream>> {
        match expression {
            Expr::Lit(literal) => match &literal.lit {
                Lit::Float(value) => {
                    if !matches!(value.suffix(), "" | "f32") {
                        return Err(syn::Error::new_spanned(
                            value,
                            "kernel float literals must be f32",
                        ));
                    }
                    let parsed = value.base10_parse::<f32>()?;
                    if !parsed.is_finite() {
                        return Err(syn::Error::new_spanned(
                            value,
                            "kernel float literal is not finite in f32",
                        ));
                    }
                    Ok(Some(quote!(#value)))
                }
                _ => Err(syn::Error::new_spanned(
                    literal,
                    "only float literals are supported in the core kernel subset",
                )),
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
        reason = "lowering transfers each constructed token stream into one statement"
    )]
    fn emit(&mut self, expression: TokenStream) -> Ident {
        let value = format_ident!(
            "__forja_value_{}",
            self.next_value,
            span = Span::mixed_site()
        );
        self.next_value += 1;
        self.statements.push(quote!(let #value = #expression;));
        value
    }
}

fn method_arity(name: &str) -> Option<usize> {
    match name {
        "abs" | "exp" | "ln" | "sqrt" | "rsqrt" | "recip" | "sin" | "cos" | "tanh" | "sigmoid"
        | "floor" | "ceil" | "log2" | "log10" | "exp2" => Some(0),
        "maximum" | "minimum" | "powf" | "powi" => Some(1),
        "clamp" => Some(2),
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

fn binding_pattern(pattern: &Pat) -> syn::Result<(String, Option<&Type>)> {
    let (pattern, ty) = match pattern {
        Pat::Type(pattern) => (&*pattern.pat, Some(&*pattern.ty)),
        pattern => (pattern, None),
    };
    let Pat::Ident(pattern) = pattern else {
        return Err(syn::Error::new_spanned(
            pattern,
            "kernel `let` patterns must be identifiers",
        ));
    };
    if pattern.by_ref.is_some() || pattern.mutability.is_some() || pattern.subpat.is_some() {
        return Err(syn::Error::new_spanned(
            pattern,
            "kernel `let` bindings must be immutable identifiers",
        ));
    }
    Ok((pattern.ident.to_string(), ty))
}

fn require_f32(ty: &Type) -> syn::Result<()> {
    if type_ident(ty).is_some_and(|ident| ident == "f32") {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "core kernel `let` annotations must be `f32`",
        ))
    }
}

fn type_ident(ty: &Type) -> Option<&Ident> {
    let Type::Path(path) = ty else { return None };
    path.qself
        .is_none()
        .then(|| path.path.get_ident())
        .flatten()
}

fn powi_exponent(expression: &Expr) -> syn::Result<i32> {
    let (negative, literal) = match expression {
        Expr::Lit(literal) => (false, literal),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            let Expr::Lit(literal) = &*unary.expr else {
                return Err(syn::Error::new_spanned(
                    expression,
                    "`powi` requires an integer literal",
                ));
            };
            (true, literal)
        }
        _ => {
            return Err(syn::Error::new_spanned(
                expression,
                "`powi` requires an integer literal",
            ));
        }
    };
    let Lit::Int(literal) = &literal.lit else {
        return Err(syn::Error::new_spanned(
            expression,
            "`powi` requires an integer literal",
        ));
    };
    let value = literal.base10_parse::<i32>()?;
    if negative {
        value
            .checked_neg()
            .ok_or_else(|| syn::Error::new_spanned(expression, "`powi` exponent is out of range"))
    } else {
        Ok(value)
    }
}
