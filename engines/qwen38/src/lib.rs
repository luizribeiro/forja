//! Text-only Qwen3.8 inference with affine four-bit weights.
#![forbid(unsafe_code)]
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
