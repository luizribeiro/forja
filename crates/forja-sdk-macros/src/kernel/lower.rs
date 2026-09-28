use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Expr, spanned::Spanned};

use super::ComputeType;

#[derive(Clone, Copy)]
pub(super) enum Parameter {
    Tensor { slot: u32, compute: ComputeType },
    Scalar,
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
    let expression = body
        .stmts
        .last()
        .and_then(|statement| match statement {
            syn::Stmt::Expr(expression, None) => Some(expression),
            _ => None,
        })
        .ok_or_else(|| syn::Error::new(body.span(), "kernel body needs a final expression"))?;
    if body.stmts.len() != 1 {
        return Err(syn::Error::new_spanned(
            body,
            "only a final parameter or tuple is supported in this kernel subset",
        ));
    }
    let mut lowerer = Lowerer {
        parameters,
        inputs: HashMap::new(),
        statements: Vec::new(),
        next_value: 0,
        context: context.clone(),
    };
    let outputs = if let Expr::Tuple(tuple) = expression {
        tuple
            .elems
            .iter()
            .map(|item| lowerer.lower_parameter(item))
            .collect::<syn::Result<Vec<_>>>()?
    } else {
        vec![lowerer.lower_parameter(expression)?]
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

struct Lowerer {
    parameters: HashMap<String, Parameter>,
    inputs: HashMap<u32, Ident>,
    statements: Vec<TokenStream>,
    next_value: usize,
    context: Ident,
}

impl Lowerer {
    fn lower_parameter(&mut self, expression: &Expr) -> syn::Result<Ident> {
        let Expr::Path(path) = expression else {
            return Err(syn::Error::new_spanned(
                expression,
                "only tensor parameters are supported in this kernel subset",
            ));
        };
        let name = path
            .path
            .get_ident()
            .ok_or_else(|| syn::Error::new_spanned(path, "expected a tensor parameter"))?
            .to_string();
        let slot = match self.parameters.get(&name).copied() {
            Some(Parameter::Tensor {
                slot,
                compute: ComputeType::F32,
            }) => slot,
            Some(Parameter::Tensor { .. } | Parameter::Scalar) | None => {
                return Err(syn::Error::new_spanned(
                    path,
                    "expected an f32 tensor parameter",
                ));
            }
        };
        if let Some(value) = self.inputs.get(&slot) {
            return Ok(value.clone());
        }
        let value = format_ident!(
            "__forja_value_{}",
            self.next_value,
            span = Span::mixed_site()
        );
        self.next_value += 1;
        let context = &self.context;
        self.statements
            .push(quote!(let #value = #context.input(#slot);));
        self.inputs.insert(slot, value.clone());
        Ok(value)
    }
}
