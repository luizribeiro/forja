//! Derive and attribute macros for Forja inference engines.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Expr, Field, Fields, GenericArgument, LitStr, PathArguments, Type,
    parse_macro_input,
};

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
