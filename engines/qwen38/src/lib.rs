//! Text-only Qwen3.8 inference with affine four-bit weights.
#![deny(unsafe_code)]
#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]
mod config;
pub use config::Config;

/// Affine quantized projection storage.
pub mod quant;

mod deltanet;
mod numerics;
pub use deltanet::DeltaNet;

mod attention;
pub use attention::Attention;

mod ffn;
pub use ffn::FeedForward;

mod layer;
pub use layer::DecoderLayer;

// All delta-rule-block.wy64-* picks require this numerical prompt block width.
pub(crate) const PREFILL_BLOCK_TOKENS: u32 = 64;

mod model;
pub use model::Model;

#[cfg_attr(
    target_family = "wasm",
    allow(unsafe_code, reason = "wit-bindgen generates the component ABI shim")
)]
mod engine;
pub use engine::Qwen38;
