//! Derive and attribute macros for Forja inference engines.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Expr, Field, Fields, GenericArgument, ItemImpl, LitStr, PathArguments, Type,
    parse_macro_input,
};

const COMPUTE_WIT: &str = include_str!("../../../wit/compute.wit");
const ENGINE_WIT: &str = include_str!("../../../wit/engine.wit");

/// Exports one `forja_sdk::Engine` implementation as a component.
#[proc_macro_attribute]
pub fn export_engine(_attribute: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as ItemImpl);
    expand_engine(&item)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn validate_engine(item: &ItemImpl) -> syn::Result<()> {
    let Some((_, trait_path, _)) = &item.trait_ else {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "export_engine requires `impl Engine for Type`",
        ));
    };
    if trait_path
        .segments
        .last()
        .is_none_or(|segment| segment.ident != "Engine")
    {
        return Err(syn::Error::new_spanned(
            trait_path,
            "export_engine requires `impl Engine for Type`",
        ));
    }
    if !item.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "exported engines cannot have generic parameters",
        ));
    }
    Ok(())
}

fn expand_engine(item: &ItemImpl) -> syn::Result<proc_macro2::TokenStream> {
    validate_engine(item)?;
    let engine = &item.self_ty;
    let engine_wit = ENGINE_WIT
        .strip_prefix("package l9o:gpu@0.1.0;\n")
        .ok_or_else(|| syn::Error::new_spanned(&item.self_ty, "engine WIT package changed"))?;
    let wit = format!("{COMPUTE_WIT}\n{engine_wit}");
    Ok(quote! {
        #item

        #[cfg(target_family = "wasm")]
        mod __forja_engine_export {
            use ::std::cell::RefCell;
            use ::forja_sdk::__private::wit_bindgen as wit_bindgen;

            type ExportedEngine = super::#engine;

            mod bindings {
                use ::forja_sdk::__private::wit_bindgen as wit_bindgen;

                ::forja_sdk::__private::wit_bindgen::generate!({
                    inline: #wit,
                    world: "engine-component",
                    runtime_path: "::forja_sdk::__private::wit_bindgen::rt",
                    with: {
                        "l9o:gpu/compute@0.1.0": ::forja_sdk::__private::compute,
                    },
                });
            }

            use bindings::exports::l9o::gpu::engine::{
                EngineInfo as WitEngineInfo, Guest, StepIn, StepOut,
            };
            use ::forja_sdk::__private::compute;

            struct Component;

            ::std::thread_local! {
                static ENGINE: RefCell<Option<ExportedEngine>> = const { RefCell::new(None) };
            }

            impl Guest for Component {
                fn describe() -> WitEngineInfo {
                    let info = <ExportedEngine as ::forja_sdk::Engine>::describe();
                    WitEngineInfo {
                        vocab: info.vocab,
                        max_context: info.max_context,
                        tap_layers: info.tap_layers,
                    }
                }

                async fn load(
                    weights: &compute::Weights,
                ) -> ::std::result::Result<(), compute::Error> {
                    let weights = ::forja_sdk::Weights::from_guest(weights);
                    let engine = <ExportedEngine as ::forja_sdk::Engine>::load(&weights)
                        .map_err(wit_error)?;
                    ENGINE.with(|slot| {
                        let mut slot = slot.try_borrow_mut().map_err(|_| {
                            compute::Error::OpSignature("engine state is already borrowed".into())
                        })?;
                        *slot = Some(engine);
                        Ok(())
                    })
                }

                async fn step(input: StepIn) -> ::std::result::Result<StepOut, compute::Error> {
                    let tokens = ::forja_sdk::Tensor::from_slice(
                        &input.tokens,
                        &[u32::try_from(input.tokens.len()).map_err(|_| {
                            compute::Error::Layout("token count exceeds u32".into())
                        })?],
                    ).map_err(wit_error)?;
                    let output = ENGINE.with(|slot| {
                        let mut slot = slot.try_borrow_mut().map_err(|_| {
                            compute::Error::OpSignature("engine state is already borrowed".into())
                        })?;
                        let engine = slot.as_mut().ok_or_else(|| {
                            compute::Error::InvalidHandle("engine is not loaded".into())
                        })?;
                        <ExportedEngine as ::forja_sdk::Engine>::step(
                            engine,
                            ::forja_sdk::StepInput {
                                tokens,
                                start_pos: input.start_pos,
                                taps: input.taps,
                            },
                        ).map_err(wit_error)
                    })?;
                    ::forja_sdk::eval().map_err(wit_error)?;
                    Ok(StepOut {
                        logits: output.logits.into_guest(),
                        taps: output.taps.into_iter().map(::forja_sdk::Tensor::into_guest).collect(),
                    })
                }
            }

            fn wit_error(error: ::forja_sdk::Error) -> compute::Error {
                compute::Error::OpSignature(error.to_string())
            }

            bindings::export!(Component with_types_in bindings);
        }
    })
}

/// Derives recursive loading from a safetensors namespace.
///
/// Fields load from the current namespace unless `#[load(prefix)]` descends into the field name.
#[proc_macro_derive(Load, attributes(load))]
pub fn derive_load(input: TokenStream) -> TokenStream {
    expand_load(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_load(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let config = container_config(&input)?;
    let Data::Struct(data) = input.data else {
        return Err(syn::Error::new_spanned(
            input.ident,
            "Load requires a struct",
        ));
    };
    let Fields::Named(fields) = data.fields else {
        return Err(syn::Error::new_spanned(
            input.ident,
            "Load requires named fields",
        ));
    };
    let name = input.ident;
    let fields = fields
        .named
        .iter()
        .map(load_field)
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        impl ::forja_sdk::Load<#config> for #name {
            fn load(
                weights: &::forja_sdk::Weights<'_>,
                config: &#config,
            ) -> ::forja_sdk::Result<Self> {
                Ok(Self { #(#fields),* })
            }
        }
    })
}

fn container_config(input: &DeriveInput) -> syn::Result<Type> {
    let mut config = None;
    for attribute in &input.attrs {
        if !attribute.path().is_ident("load") {
            continue;
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("config") {
                config = Some(meta.value()?.parse()?);
                Ok(())
            } else {
                Err(meta.error("expected `config = Type`"))
            }
        })?;
    }
    Ok(config.unwrap_or_else(|| syn::parse_quote!(())))
}

fn load_field(field: &Field) -> syn::Result<proc_macro2::TokenStream> {
    let ident = field
        .ident
        .as_ref()
        .ok_or_else(|| syn::Error::new_spanned(field, "Load requires named fields"))?;
    let mut tensor_name = ident.to_string();
    let mut count = None;
    let mut field_config = None;
    let mut prefix = false;
    for attribute in &field.attrs {
        if !attribute.path().is_ident("load") {
            continue;
        }
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                tensor_name = meta.value()?.parse::<LitStr>()?.value();
            } else if meta.path.is_ident("count") {
                count = Some(meta.value()?.parse::<Expr>()?);
            } else if meta.path.is_ident("config") {
                field_config = Some(meta.value()?.parse::<Expr>()?);
            } else if meta.path.is_ident("prefix") {
                prefix = true;
            } else {
                return Err(meta.error("expected `name`, `prefix`, `count`, or `config`"));
            }
            Ok(())
        })?;
    }
    let scope = LitStr::new(&tensor_name, ident.span());
    let config = field_config.map_or_else(|| quote!(config), |expression| quote!(&(#expression)));
    if let Some(element) = vec_element(&field.ty) {
        let count = count.ok_or_else(|| {
            syn::Error::new_spanned(field, "Vec fields require `#[load(count = ...)]`")
        })?;
        let index = format_ident!("{}_index", ident);
        let weights = if prefix {
            quote!(&weights.scoped(#scope).scoped(#index.to_string()))
        } else {
            quote!(&weights.scoped(#index.to_string()))
        };
        Ok(quote! {
            #ident: {
                let count = ::core::convert::TryInto::<usize>::try_into(#count)
                    .map_err(|_| ::forja_sdk::Error::loading(
                        concat!("layer count for `", stringify!(#ident), "` does not fit usize")
                    ))?;
                (0..count)
                    .map(|#index| <#element as ::forja_sdk::Load<_>>::load(
                        #weights,
                        #config,
                    ))
                    .collect::<::forja_sdk::Result<::std::vec::Vec<_>>>()?
            }
        })
    } else {
        let ty = &field.ty;
        let weights = if prefix {
            quote!(&weights.scoped(#scope))
        } else {
            quote!(weights)
        };
        Ok(quote! {
            #ident: <#ty as ::forja_sdk::Load<_>>::load(#weights, #config)?
        })
    }
}

fn vec_element(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Vec" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.first()? {
        GenericArgument::Type(element) => Some(element),
        _ => None,
    }
}
