//! Derive and attribute macros for Forja inference engines.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Expr, Field, Fields, GenericArgument, ItemFn, ItemImpl, LitStr,
    PathArguments, Type, parse_macro_input,
};

mod kernel;

const COMPUTE_WIT: &str = include_str!("../../../wit/compute.wit");
const PAGED_STORAGE_WIT: &str = include_str!("../../../wit/paged-storage.wit");
const ENGINE_WIT: &str = include_str!("../../../wit/engine.wit");

/// Defines a restricted Rust-syntax scalar kernel.
#[proc_macro_attribute]
pub fn kernel(attribute: TokenStream, item: TokenStream) -> TokenStream {
    kernel::expand(attribute.into(), parse_macro_input!(item as ItemFn))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

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
    let paged_storage_wit = PAGED_STORAGE_WIT
        .strip_prefix("package l9o:gpu@0.1.0;\n")
        .ok_or_else(|| {
            syn::Error::new_spanned(&item.self_ty, "paged storage WIT package changed")
        })?;
    let wit = format!("{COMPUTE_WIT}\n{paged_storage_wit}\n{engine_wit}");
    let guest_impl = engine_guest_impl();
    Ok(quote! {
        #item

        #[cfg(target_family = "wasm")]
        mod __forja_engine_export {
            #![allow(clippy::same_length_and_capacity, clippy::unused_async_trait_impl)]

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
                        "l9o:gpu/paged-storage@0.1.0": ::forja_sdk::__private::paged_storage,
                    },
                });
            }

            use bindings::exports::l9o::gpu::engine::{
                BatchIn, BatchOut, BatchSelectIn, DecodeIn, DecodeOut,
                EngineInfo as WitEngineInfo, Guest, LoadConfig, StepIn, StepOut,
            };
            use ::forja_sdk::__private::compute;

            struct Component;

            ::std::thread_local! {
                static ENGINE: RefCell<Option<ExportedEngine>> = const { RefCell::new(None) };
            }

            #guest_impl

            fn wit_error(error: ::forja_sdk::Error) -> compute::Error {
                compute::Error::OpSignature(error.to_string())
            }

            bindings::export!(Component with_types_in bindings);
        }
    })
}

fn engine_guest_impl() -> proc_macro2::TokenStream {
    let batch_exports = engine_batch_exports();
    let batch_helpers = engine_batch_helpers();
    let load_config = engine_load_config();
    let decode_phases = engine_decode_phases();
    let serving_info = serving_info();
    quote! {
        impl Guest for Component {
            fn describe() -> WitEngineInfo {
                let info = <ExportedEngine as ::forja_sdk::Engine>::describe();
                WitEngineInfo {
                    vocab: info.vocab,
                    max_context: info.max_context,
                    tap_layers: info.tap_layers,
                    router_layers: info.router_layers,
                    serving: #serving_info,
                }
            }

            async fn load(
                weights: &compute::Weights,
                pool: Option<&::forja_sdk::__private::paged_storage::PagePool>,
                config: LoadConfig,
            ) -> ::std::result::Result<(), compute::Error> {
                let weights = ::forja_sdk::Weights::from_guest(weights);
                let engine = <ExportedEngine as ::forja_sdk::Engine>::load_paged(
                    &weights,
                    pool.map(::forja_sdk::PagedPool::from_guest),
                    #load_config,
                )
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
                let tokens = token_tensor(&input.tokens)?;
                let output = with_engine(|engine| {
                    <ExportedEngine as ::forja_sdk::Engine>::step(
                        engine,
                        ::forja_sdk::StepInput {
                            tokens,
                            start_pos: input.start_pos,
                            taps: input.taps,
                        },
                    )
                })?;
                ::forja_sdk::eval().map_err(wit_error)?;
                Ok(StepOut {
                    logits: output.logits.into_guest(),
                    taps: output.taps.into_iter().map(::forja_sdk::Tensor::into_guest).collect(),
                    router_logits: output.router_logits.into_iter().map(::forja_sdk::Tensor::into_guest).collect(),
                })
            }

            async fn decode(input: DecodeIn) -> ::std::result::Result<DecodeOut, compute::Error> {
                let tokens = input.tokens.as_deref().map(token_tensor).transpose()?;
                let output = with_engine(|engine| {
                    <ExportedEngine as ::forja_sdk::Engine>::decode(
                        engine,
                        ::forja_sdk::DecodeInput {
                            tokens,
                            start_pos: input.start_pos,
                            sampling: sampling(input.sampling),
                        },
                    )
                })?;
                ::forja_sdk::eval().map_err(wit_error)?;
                Ok(DecodeOut {
                    logits: output.logits.into_guest(),
                    token: output.token.into_guest(),
                })
            }

            #batch_exports

            #decode_phases
        }

        fn with_engine<T>(
            call: impl FnOnce(&mut ExportedEngine) -> ::forja_sdk::Result<T>,
        ) -> ::std::result::Result<T, compute::Error> {
            ENGINE.with(|slot| {
                let mut slot = slot.try_borrow_mut().map_err(|_| {
                    compute::Error::OpSignature("engine state is already borrowed".into())
                })?;
                let engine = slot.as_mut().ok_or_else(|| {
                    compute::Error::InvalidHandle("engine is not loaded".into())
                })?;
                call(engine).map_err(wit_error)
            })
        }

        fn token_tensor(
            tokens: &[u32],
        ) -> ::std::result::Result<::forja_sdk::Tensor<u32>, compute::Error> {
            let len = u32::try_from(tokens.len())
                .map_err(|_| compute::Error::Layout("token count exceeds u32".into()))?;
            ::forja_sdk::Tensor::from_slice(tokens, &[len]).map_err(wit_error)
        }

        #batch_helpers
    }
}

fn engine_batch_exports() -> proc_macro2::TokenStream {
    quote! {
        async fn batch_step(
            slot: &::forja_sdk::__private::paged_storage::BatchSlot,
            input: BatchIn,
        ) -> ::std::result::Result<BatchOut, compute::Error> {
            let input = batch_input(slot, input)?;
            let output = with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::batch_step(engine, input)
            })?;
            ::forja_sdk::eval().map_err(wit_error)?;
            Ok(BatchOut {
                logits: output.logits.into_guest(),
                tokens: output.tokens.into_guest(),
            })
        }

        async fn batch_forward(
            slot: &::forja_sdk::__private::paged_storage::BatchSlot,
            input: BatchIn,
        ) -> ::std::result::Result<compute::Tensor, compute::Error> {
            let input = batch_input(slot, input)?;
            let logits = with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::batch_forward(engine, input)
            })?;
            ::forja_sdk::eval().map_err(wit_error)?;
            Ok(logits.into_guest())
        }

        async fn batch_select(
            slot: &::forja_sdk::__private::paged_storage::BatchSlot,
            logits: &compute::Tensor,
            input: BatchSelectIn,
        ) -> ::std::result::Result<BatchOut, compute::Error> {
            let vocab = <ExportedEngine as ::forja_sdk::Engine>::describe().vocab;
            let shape = batch_shape(input.shape);
            let logits = logits.view(&compute::ViewOp::Reshape(vec![shape.bucket, vocab]))?;
            let logits = ::forja_sdk::Tensor::from_guest(logits, vec![shape.bucket, vocab]);
            let lanes = input.lanes.into_iter().map(|lane| lane.map(|lane| {
                ::forja_sdk::BatchSelectLane {
                    start_pos: lane.start_pos,
                    sampling: sampling(lane.sampling),
                    allow: lane.allow,
                }
            })).collect();
            let output = with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::batch_select(
                    engine,
                    ::forja_sdk::BatchSelectInput {
                        slot: ::forja_sdk::PagedSlot::from_guest(slot),
                        shape,
                        logits: &logits,
                        lanes,
                    },
                )
            })?;
            ::forja_sdk::eval().map_err(wit_error)?;
            Ok(BatchOut {
                logits: output.logits.into_guest(),
                tokens: output.tokens.into_guest(),
            })
        }
    }
}

fn engine_batch_helpers() -> proc_macro2::TokenStream {
    quote! {
        fn batch_input<'a>(
            slot: &'a ::forja_sdk::__private::paged_storage::BatchSlot,
            input: BatchIn,
        ) -> ::std::result::Result<::forja_sdk::BatchInput<'a>, compute::Error> {
            let lanes = input.lanes.into_iter().map(|lane| lane.map(|lane| {
                Ok(::forja_sdk::BatchLane {
                    slot: lane.slot,
                    tokens: lane.tokens.as_deref().map(token_tensor).transpose()?,
                    start_pos: lane.start_pos,
                    sampling: sampling(lane.sampling),
                })
            }).transpose()).collect::<::std::result::Result<Vec<_>, compute::Error>>()?;
            Ok(::forja_sdk::BatchInput {
                slot: ::forja_sdk::PagedSlot::from_guest(slot),
                shape: batch_shape(input.shape),
                lanes,
            })
        }

        fn sampling(
            value: bindings::exports::l9o::gpu::engine::SamplingParams,
        ) -> ::forja_sdk::SamplingParams {
            ::forja_sdk::SamplingParams {
                temperature: value.temperature,
                top_k: value.top_k,
                top_p: value.top_p,
                seed: value.seed,
            }
        }

        fn batch_shape(
            value: bindings::exports::l9o::gpu::engine::BatchShape,
        ) -> ::forja_sdk::BatchShape {
            ::forja_sdk::BatchShape {
                phase: match value.phase {
                    bindings::exports::l9o::gpu::engine::BatchPhase::Prompt => ::forja_sdk::BatchPhase::Prompt,
                    bindings::exports::l9o::gpu::engine::BatchPhase::Decode => ::forja_sdk::BatchPhase::Decode,
                },
                bucket: value.bucket,
                tokens: value.tokens,
                selection: match value.selection {
                    bindings::exports::l9o::gpu::engine::BatchSelection::Combined => ::forja_sdk::BatchSelection::Combined,
                    bindings::exports::l9o::gpu::engine::BatchSelection::Forward => ::forja_sdk::BatchSelection::Forward,
                    bindings::exports::l9o::gpu::engine::BatchSelection::Masked => ::forja_sdk::BatchSelection::Masked,
                },
            }
        }
    }
}

fn serving_info() -> proc_macro2::TokenStream {
    quote! {
        info.serving.map(|serving| bindings::exports::l9o::gpu::engine::ServingInfo {
            page_tokens: serving.page_tokens,
            planes: serving.planes.into_iter().map(|plane| bindings::exports::l9o::gpu::engine::CachePlane {
                elements_per_token: plane.elements_per_token,
                alignment: plane.alignment,
                label: plane.label,
            }).collect(),
            active_lanes: serving.active_lanes,
            max_context: serving.max_context,
            prefill_chunk_multiple: serving.prefill_chunk_multiple.map(::std::num::NonZeroU32::get),
            batch_shapes: serving.batch_shapes.into_iter().map(|shape| bindings::exports::l9o::gpu::engine::BatchShape {
                phase: match shape.phase {
                    ::forja_sdk::BatchPhase::Prompt => bindings::exports::l9o::gpu::engine::BatchPhase::Prompt,
                    ::forja_sdk::BatchPhase::Decode => bindings::exports::l9o::gpu::engine::BatchPhase::Decode,
                },
                bucket: shape.bucket,
                tokens: shape.tokens,
                selection: match shape.selection {
                    ::forja_sdk::BatchSelection::Combined => bindings::exports::l9o::gpu::engine::BatchSelection::Combined,
                    ::forja_sdk::BatchSelection::Forward => bindings::exports::l9o::gpu::engine::BatchSelection::Forward,
                    ::forja_sdk::BatchSelection::Masked => bindings::exports::l9o::gpu::engine::BatchSelection::Masked,
                },
            }).collect(),
        })
    }
}

fn engine_decode_phases() -> proc_macro2::TokenStream {
    quote! {
        async fn begin_decode(
            tokens: Option<Vec<u32>>,
            start_pos: u32,
        ) -> ::std::result::Result<compute::Tensor, compute::Error> {
            let tokens = tokens.as_deref().map(token_tensor).transpose()?;
            let logits = with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::begin_decode(
                    engine,
                    ::forja_sdk::BeginDecodeInput { tokens, start_pos },
                )
            })?;
            ::forja_sdk::eval().map_err(wit_error)?;
            Ok(logits.into_guest())
        }

        async fn finish_decode(
            logits: &compute::Tensor,
            start_pos: u32,
            sampling: bindings::exports::l9o::gpu::engine::SamplingParams,
            allow: Vec<u32>,
        ) -> ::std::result::Result<DecodeOut, compute::Error> {
            let vocab = <ExportedEngine as ::forja_sdk::Engine>::describe().vocab;
            let logits = logits.view(&compute::ViewOp::Reshape(vec![vocab]))?;
            let logits = ::forja_sdk::Tensor::from_guest(logits, vec![vocab]);
            let output = with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::finish_decode(
                    engine,
                    &logits,
                    start_pos,
                    ::forja_sdk::SamplingParams {
                        temperature: sampling.temperature,
                        top_k: sampling.top_k,
                        top_p: sampling.top_p,
                        seed: sampling.seed,
                    },
                    &allow,
                )
            })?;
            ::forja_sdk::eval().map_err(wit_error)?;
            Ok(DecodeOut {
                logits: output.logits.into_guest(),
                token: output.token.into_guest(),
            })
        }

        async fn rewind_decode(next_pos: u32) -> ::std::result::Result<(), compute::Error> {
            with_engine(|engine| {
                <ExportedEngine as ::forja_sdk::Engine>::rewind_decode(engine, next_pos)
            })?;
            ::forja_sdk::eval().map_err(wit_error)
        }
    }
}

fn engine_load_config() -> proc_macro2::TokenStream {
    quote! {
        ::forja_sdk::EngineLoadConfig {
            num_hidden_layers: config.num_hidden_layers,
            max_context: config.max_context,
            replay: config.replay,
            kv_cache: match config.kv_cache {
                bindings::exports::l9o::gpu::engine::KvCacheFormat::Float16 => {
                    ::forja_sdk::KvCacheFormat::F16
                }
                bindings::exports::l9o::gpu::engine::KvCacheFormat::Bfloat16 => {
                    ::forja_sdk::KvCacheFormat::Bf16
                }
                bindings::exports::l9o::gpu::engine::KvCacheFormat::Float32 => {
                    ::forja_sdk::KvCacheFormat::F32
                }
            },
            tunings: config.tunings,
            fixed_variant_picks: config.fixed_variant_picks.into_iter().map(|pick| {
                ::forja_sdk::FixedVariantPick { site: pick.site, name: pick.name }
            }).collect(),
            variant_rule_picks: config.variant_rule_picks.into_iter().map(|pick| {
                ::forja_sdk::VariantRulePick {
                    site: pick.site,
                    parameter: pick.parameter,
                    arms: pick.arms.into_iter().map(|arm| {
                        ::forja_sdk::VariantPickArm {
                            lo: arm.lo,
                            hi: arm.hi,
                            name: arm.name,
                        }
                    }).collect(),
                }
            }).collect(),
        }
    }
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
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let fields = fields
        .named
        .iter()
        .map(load_field)
        .collect::<syn::Result<Vec<_>>>()?;
    Ok(quote! {
        impl #impl_generics ::forja_sdk::Load<#config> for #name #type_generics #where_clause {
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
