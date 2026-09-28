mod lower;

use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, GenericArgument, ItemFn, Pat, PathArguments, ReturnType, Type, spanned::Spanned};

use lower::{Parameter, lower};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum KernelKind {
    Map,
    Row,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum ComputeType {
    F32,
    U32,
}

struct TensorParameter {
    ident: Ident,
    compute: ComputeType,
    slot: u32,
    generic: Option<Ident>,
}

struct ScalarParameter {
    ident: Ident,
    compute: ComputeType,
}

struct KernelFunction {
    item: ItemFn,
    kind: KernelKind,
    tensors: Vec<TensorParameter>,
    scalars: Vec<ScalarParameter>,
    output_count: usize,
}

pub(super) fn expand(attribute: TokenStream, item: ItemFn) -> syn::Result<TokenStream> {
    let kind = parse_kind(attribute)?;
    let function = KernelFunction::parse(item, kind)?;
    function.expand()
}

impl KernelFunction {
    fn parse(item: ItemFn, kind: KernelKind) -> syn::Result<Self> {
        validate_modifiers(&item)?;
        let mut tensors = Vec::new();
        let mut scalars = Vec::new();
        for argument in &item.sig.inputs {
            let FnArg::Typed(argument) = argument else {
                return Err(syn::Error::new_spanned(
                    argument,
                    "kernel methods are not supported",
                ));
            };
            let Pat::Ident(pattern) = &*argument.pat else {
                return Err(syn::Error::new_spanned(
                    &argument.pat,
                    "kernel parameters must be identifiers",
                ));
            };
            if pattern.by_ref.is_some() || pattern.mutability.is_some() || pattern.subpat.is_some()
            {
                return Err(syn::Error::new_spanned(
                    &argument.pat,
                    "kernel parameters must be plain identifiers",
                ));
            }
            if let Some((wrapper, compute)) = tensor_type(&argument.ty)? {
                let expected = match kind {
                    KernelKind::Map => "Elem",
                    KernelKind::Row => "Row",
                };
                if wrapper != expected {
                    return Err(syn::Error::new_spanned(
                        &argument.ty,
                        format!(
                            "`#[kernel({})]` tensor parameters must use `{expected}`",
                            kind.name()
                        ),
                    ));
                }
                let slot = u32::try_from(tensors.len()).map_err(|_| {
                    syn::Error::new_spanned(&argument.ty, "too many tensor parameters")
                })?;
                let generic = (compute == ComputeType::F32)
                    .then(|| format_ident!("__ForjaT{slot}", span = Span::mixed_site()));
                tensors.push(TensorParameter {
                    ident: pattern.ident.clone(),
                    compute,
                    slot,
                    generic,
                });
            } else if let Some(compute) = scalar_type(&argument.ty) {
                scalars.push(ScalarParameter {
                    ident: pattern.ident.clone(),
                    compute,
                });
            } else {
                return Err(syn::Error::new_spanned(
                    &argument.ty,
                    format!(
                        "kernel: parameter `{}: {}` is not a kernel type; use `Row`/`Elem` for tensors or `f32`/`u32` for constants",
                        pattern.ident,
                        type_name(&argument.ty),
                    ),
                ));
            }
        }
        if tensors.is_empty() {
            return Err(syn::Error::new_spanned(
                &item.sig,
                "kernel requires a tensor parameter",
            ));
        }
        if tensors.len() > 8 {
            return Err(syn::Error::new_spanned(
                &item.sig.inputs,
                "kernels accept at most eight tensor parameters",
            ));
        }
        if tensors[0].compute != ComputeType::F32 {
            return Err(syn::Error::new_spanned(
                &item.sig.inputs,
                "the leading tensor must have f32 compute type",
            ));
        }
        let output_count = output_count(&item.sig.output, kind)?;
        Ok(Self {
            item,
            kind,
            tensors,
            scalars,
            output_count,
        })
    }

    fn expand(&self) -> syn::Result<TokenStream> {
        let item = &self.item;
        let visibility = &item.vis;
        let attributes = &item.attrs;
        let name = &item.sig.ident;
        let into_name = format_ident!("{name}_into");
        let program_name = format_ident!("{name}_program");
        let out_argument = internal("__forja_out");
        let rank_argument = internal("__forja_rank_argument");
        let input_dtypes_argument = internal("__forja_input_dtypes_argument");
        let output_dtypes_argument = internal("__forja_output_dtypes_argument");
        let tensor_generics = self
            .tensors
            .iter()
            .filter_map(|tensor| tensor.generic.as_ref())
            .collect::<Vec<_>>();
        let into_tensor_generics = tensor_generics.clone();
        let tensor_arguments = self.tensors.iter().map(tensor_argument).collect::<Vec<_>>();
        let into_tensor_arguments = tensor_arguments.clone();
        let scalar_arguments = self.scalars.iter().map(scalar_argument).collect::<Vec<_>>();
        let into_scalar_arguments = scalar_arguments.clone();
        let first_generic = self.tensors[0].generic.as_ref().ok_or_else(|| {
            syn::Error::new(
                item.sig.span(),
                "leading tensor storage must be floating point",
            )
        })?;
        let output_type = output_type(first_generic, self.output_count);
        let allocated_body = self.allocated_body(&program_name, first_generic);

        let into_generics = (0..self.output_count)
            .map(|index| format_ident!("__ForjaO{index}", span = Span::mixed_site()))
            .collect::<Vec<_>>();
        let into_output_type = if self.output_count == 1 {
            let output = &into_generics[0];
            quote!(&::forja_sdk::Tensor<#output>)
        } else {
            quote!((#(&::forja_sdk::Tensor<#into_generics>),*))
        };
        let into_body = self.write_body(&program_name, &into_generics, &out_argument);
        let program = self.program_body(
            name,
            &rank_argument,
            &input_dtypes_argument,
            &output_dtypes_argument,
        )?;
        let scalar_program_arguments = self.scalars.iter().map(scalar_argument).collect::<Vec<_>>();

        Ok(quote! {
            #(#attributes)*
            #visibility fn #name<#(#tensor_generics: ::forja_sdk::FloatElement),*>(
                #(#tensor_arguments,)*
                #(#scalar_arguments),*
            ) -> ::forja_sdk::Result<#output_type> {
                #allocated_body
            }

            #[doc = "Writes this generated kernel into caller-supplied output views."]
            #visibility fn #into_name<
                #(#into_tensor_generics: ::forja_sdk::FloatElement,)*
                #(#into_generics: ::forja_sdk::FloatElement),*
            >(
                #(#into_tensor_arguments,)*
                #(#into_scalar_arguments,)*
                #out_argument: #into_output_type,
            ) -> ::forja_sdk::Result<()> {
                #into_body
            }

            #[doc(hidden)]
            #visibility fn #program_name(
                #rank_argument: usize,
                #input_dtypes_argument: &[::forja_sdk::DType],
                #output_dtypes_argument: &[::forja_sdk::DType],
                #(#scalar_program_arguments),*
            ) -> ::forja_sdk::Result<::forja_sdk::kernel::Kernel> {
                #program
            }
        })
    }

    fn allocated_body(&self, program_name: &Ident, first_generic: &Ident) -> TokenStream {
        let first = &self.tensors[0].ident;
        let shape = internal("__forja_shape");
        let inputs = internal("__forja_inputs");
        let input_dtypes = internal("__forja_input_dtypes");
        let output_dtypes = internal("__forja_output_dtypes");
        let kernel = internal("__forja_kernel");
        let binding = internal("__forja_binding");
        let broadcasts = self.tensors.iter().skip(1).map(|tensor| {
            let ident = &tensor.ident;
            let storage = format_ident!("__forja_broadcast_{}", ident, span = Span::mixed_site());
            let broadcast = internal("__forja_broadcast_value");
            quote! {
                let #storage = if #ident.shape() == #shape {
                    None
                } else {
                    Some(#ident.broadcast_as(#shape)?)
                };
                let #ident = match &#storage {
                    Some(#broadcast) => #broadcast,
                    None => #ident,
                };
            }
        });
        let tensor_names = self.tensors.iter().map(|tensor| &tensor.ident);
        let scalar_names = self.scalars.iter().map(|scalar| &scalar.ident);
        let outputs = (0..self.output_count)
            .map(|index| format_ident!("__forja_output_{index}", span = Span::mixed_site()))
            .collect::<Vec<_>>();
        let returned = if self.output_count == 1 {
            let output = &outputs[0];
            quote!(#output)
        } else {
            quote!((#(#outputs),*))
        };
        let output_count = self.output_count;
        quote! {
            let #shape = #first.shape();
            #(#broadcasts)*
            let #inputs = [#(::forja_sdk::kernel::TensorRef::new(#tensor_names)?),*];
            let #input_dtypes = #inputs
                .iter()
                .map(|#binding| #binding.dtype())
                .collect::<::std::vec::Vec<_>>();
            let #output_dtypes = [#inputs[0].dtype(); #output_count];
            let #kernel = #program_name(
                #shape.len(),
                &#input_dtypes,
                &#output_dtypes,
                #(#scalar_names),*
            )?;
            let [#(#outputs),*] = ::forja_sdk::kernel::run::<#first_generic, #output_count>(
                &#kernel,
                &#inputs,
            )?;
            Ok(#returned)
        }
    }

    fn write_body(
        &self,
        program_name: &Ident,
        output_generics: &[Ident],
        out_argument: &Ident,
    ) -> TokenStream {
        let first = &self.tensors[0].ident;
        let inputs = internal("__forja_inputs");
        let output_refs = internal("__forja_outputs");
        let input_dtypes = internal("__forja_input_dtypes");
        let output_dtypes = internal("__forja_output_dtypes");
        let kernel = internal("__forja_kernel");
        let binding = internal("__forja_binding");
        let tensor_names = self.tensors.iter().map(|tensor| &tensor.ident);
        let scalar_names = self.scalars.iter().map(|scalar| &scalar.ident);
        let outputs = output_generics
            .iter()
            .enumerate()
            .map(|(index, _)| format_ident!("__forja_output_{index}", span = Span::mixed_site()))
            .collect::<Vec<_>>();
        let bind_outputs = if self.output_count == 1 {
            let output = &outputs[0];
            quote!(let #output = #out_argument;)
        } else {
            quote!(let (#(#outputs),*) = #out_argument;)
        };
        quote! {
            #bind_outputs
            let #inputs = [#(::forja_sdk::kernel::TensorRef::new(#tensor_names)?),*];
            let #output_refs = [#(::forja_sdk::kernel::TensorRef::new(#outputs)?),*];
            let #input_dtypes = #inputs
                .iter()
                .map(|#binding| #binding.dtype())
                .collect::<::std::vec::Vec<_>>();
            let #output_dtypes = #output_refs
                .iter()
                .map(|#binding| #binding.dtype())
                .collect::<::std::vec::Vec<_>>();
            let #kernel = #program_name(
                #first.shape().len(),
                &#input_dtypes,
                &#output_dtypes,
                #(#scalar_names),*
            )?;
            ::forja_sdk::kernel::run_into(&#kernel, &#inputs, &#output_refs)
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the generated build path stays together so hygiene is reviewable"
    )]
    fn program_body(
        &self,
        name: &Ident,
        rank_argument: &Ident,
        input_dtypes_argument: &Ident,
        output_dtypes_argument: &Ident,
    ) -> syn::Result<TokenStream> {
        let context = internal("__forja_context");
        let rank = internal("__forja_rank");
        let scalar_bits_name = internal("__forja_scalar_bits");
        let cache_name = internal("__FORJA_CACHE");
        let cache = internal("__forja_cache");
        let program = internal("__forja_program");
        let parameters = self
            .tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.ident.to_string(),
                    Parameter::Tensor {
                        slot: tensor.slot,
                        compute: tensor.compute,
                        span: tensor.ident.span(),
                    },
                )
            })
            .chain(
                self.scalars
                    .iter()
                    .map(|scalar| (scalar.ident.to_string(), Parameter::Scalar(scalar.compute))),
            )
            .collect::<HashMap<_, _>>();
        let lowered = lower(&self.item.block, parameters, &context, self.kind)?;
        if lowered.outputs.len() != self.output_count {
            return Err(syn::Error::new_spanned(
                &self.item.sig.output,
                "kernel return type does not match its final expression",
            ));
        }
        let context_type = match self.kind {
            KernelKind::Map => quote!(::forja_sdk::program::Ctx),
            KernelKind::Row => quote!(::forja_sdk::program::RowCtx),
        };
        let statements = &lowered.statements;
        let output_statements = lowered.outputs.iter().enumerate().map(|(slot, value)| {
            let slot = u32::try_from(slot).unwrap_or(0);
            quote!(#context.output(#slot, #value);)
        });
        let scalar_checks = self.scalars.iter().map(|scalar| {
            let ident = &scalar.ident;
            match scalar.compute {
                ComputeType::F32 => quote! {
                    if !#ident.is_finite() {
                        return Err(::forja_sdk::Error::loading(concat!(
                            "kernel `", stringify!(#name), "`: scalar `", stringify!(#ident), "` is not finite"
                        )));
                    }
                },
                ComputeType::U32 => quote! {
                    if #ident > 16_777_216 {
                        return Err(::forja_sdk::Error::loading(concat!(
                            "kernel `", stringify!(#name), "`: scalar `", stringify!(#ident), "` is not exactly representable"
                        )));
                    }
                },
            }
        });
        let scalar_bits = self.scalars.iter().map(|scalar| {
            let ident = &scalar.ident;
            match scalar.compute {
                ComputeType::F32 => quote!(#ident.to_bits()),
                ComputeType::U32 => quote!(#ident),
            }
        });
        Ok(quote! {
            #(#scalar_checks)*
            let #rank = u8::try_from(#rank_argument).map_err(|_| {
                ::forja_sdk::Error::loading(concat!(
                    "kernel `", stringify!(#name), "`: tensor rank is too large"
                ))
            })?;
            let #scalar_bits_name = [#(#scalar_bits),*];
            ::std::thread_local! {
                static #cache_name: ::forja_sdk::kernel::Cache =
                    const { ::forja_sdk::kernel::Cache::rejecting(stringify!(#name)) };
            }
            #cache_name.with(|#cache| {
                #cache.get_or_try_insert_with(
                    #rank,
                    #input_dtypes_argument,
                    #output_dtypes_argument,
                    &#scalar_bits_name,
                    || {
                        let #context = #context_type::new();
                        #(#statements)*
                        #(#output_statements)*
                        let #program = #context.finish();
                        ::forja_sdk::kernel::Kernel::new(
                            &#program,
                            #rank,
                            #input_dtypes_argument,
                            #output_dtypes_argument,
                        )
                        .map_err(|#program| ::forja_sdk::Error::loading(::std::format!(
                            "kernel `{}`: {}",
                            stringify!(#name),
                            #program,
                        )))
                    },
                )
            })
        })
    }
}

fn internal(name: &str) -> Ident {
    Ident::new(name, Span::mixed_site())
}

impl KernelKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Map => "map",
            Self::Row => "row",
        }
    }
}

fn parse_kind(attribute: TokenStream) -> syn::Result<KernelKind> {
    let ident = syn::parse2::<Ident>(attribute)?;
    match ident.to_string().as_str() {
        "map" => Ok(KernelKind::Map),
        "row" => Ok(KernelKind::Row),
        _ => Err(syn::Error::new_spanned(ident, "expected `map` or `row`")),
    }
}

fn validate_modifiers(item: &ItemFn) -> syn::Result<()> {
    let signature = &item.sig;
    if !signature.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &signature.generics,
            "kernel: generic parameters are not allowed; storage dtypes are chosen at the call site",
        ));
    }
    if signature.constness.is_some()
        || signature.asyncness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
    {
        return Err(syn::Error::new_spanned(
            signature,
            "kernel functions cannot have modifiers or generics",
        ));
    }
    Ok(())
}

fn tensor_type(ty: &Type) -> syn::Result<Option<(String, ComputeType)>> {
    let Type::Path(path) = ty else {
        return Ok(None);
    };
    if path.qself.is_some() {
        return Ok(None);
    }
    let Some(segment) = path.path.segments.last() else {
        return Ok(None);
    };
    if !matches!(segment.ident.to_string().as_str(), "Elem" | "Row") {
        return Ok(None);
    }
    let compute = match &segment.arguments {
        PathArguments::None => ComputeType::F32,
        PathArguments::AngleBracketed(arguments) if arguments.args.len() == 1 => {
            let Some(GenericArgument::Type(ty)) = arguments.args.first() else {
                return Err(syn::Error::new_spanned(
                    arguments,
                    "tensor compute type must be `f32` or `u32`",
                ));
            };
            scalar_type(ty).ok_or_else(|| {
                syn::Error::new_spanned(ty, "tensor compute type must be `f32` or `u32`")
            })?
        }
        arguments => {
            return Err(syn::Error::new_spanned(
                arguments,
                "tensor type takes at most one compute type",
            ));
        }
    };
    Ok(Some((segment.ident.to_string(), compute)))
}

fn scalar_type(ty: &Type) -> Option<ComputeType> {
    let Type::Path(path) = ty else { return None };
    let ident = path
        .qself
        .is_none()
        .then(|| path.path.get_ident())
        .flatten()?;
    match ident.to_string().as_str() {
        "f32" => Some(ComputeType::F32),
        "u32" => Some(ComputeType::U32),
        _ => None,
    }
}

fn type_name(ty: &Type) -> String {
    ty.to_token_stream().to_string().replace(' ', "")
}

fn output_count(output: &ReturnType, kind: KernelKind) -> syn::Result<usize> {
    let ReturnType::Type(_, ty) = output else {
        return Err(syn::Error::new_spanned(
            output,
            "kernel requires an `Elem` or `Row` return type",
        ));
    };
    let outputs = match &**ty {
        Type::Tuple(tuple) => tuple.elems.iter().collect::<Vec<_>>(),
        ty => vec![ty],
    };
    if outputs.is_empty() || outputs.len() > 4 {
        return Err(syn::Error::new_spanned(
            ty,
            "kernel must return one to four values",
        ));
    }
    let expected = match kind {
        KernelKind::Map => "Elem",
        KernelKind::Row => "Row",
    };
    for output in &outputs {
        let Some((wrapper, compute)) = tensor_type(output)? else {
            return Err(syn::Error::new_spanned(
                output,
                format!("kernel outputs must be `{expected}`"),
            ));
        };
        if wrapper != expected || compute != ComputeType::F32 {
            return Err(syn::Error::new_spanned(
                output,
                format!("kernel outputs must be `{expected}` with f32 compute type"),
            ));
        }
    }
    Ok(outputs.len())
}

fn tensor_argument(tensor: &TensorParameter) -> TokenStream {
    let ident = &tensor.ident;
    match (&tensor.generic, tensor.compute) {
        (Some(generic), ComputeType::F32) => quote!(#ident: &::forja_sdk::Tensor<#generic>),
        (None, ComputeType::U32) => quote!(#ident: &::forja_sdk::Tensor<u32>),
        _ => quote!(#ident: &::forja_sdk::Tensor<f32>),
    }
}

fn scalar_argument(scalar: &ScalarParameter) -> TokenStream {
    let ident = &scalar.ident;
    match scalar.compute {
        ComputeType::F32 => quote!(#ident: f32),
        ComputeType::U32 => quote!(#ident: u32),
    }
}

fn output_type(first_generic: &Ident, output_count: usize) -> TokenStream {
    if output_count == 1 {
        quote!(::forja_sdk::Tensor<#first_generic>)
    } else {
        let outputs =
            std::iter::repeat_n(quote!(::forja_sdk::Tensor<#first_generic>), output_count);
        quote!((#(#outputs),*))
    }
}
