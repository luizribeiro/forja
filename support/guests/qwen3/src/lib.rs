//! Qwen3-0.6B inference engine.

#![allow(dead_code, reason = "model fields are exercised as forward paths land")]

use forja_sdk::{
    Engine, EngineInfo, Load, Result, StepInput, StepOutput, Tensor, Weights, bf16, export_engine,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
};

/// Vocabulary size reported by Qwen3-0.6B.
pub const VOCAB: u32 = 151_936;
/// Hidden width of Qwen3-0.6B.
pub const HIDDEN: u32 = 1_024;
/// Number of decoder layers.
pub const LAYERS: usize = 28;
/// Largest context accepted by this engine.
pub const MAX_CONTEXT: u32 = 4_096;

const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const INTERMEDIATE: u32 = 3_072;
const RMS_EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy)]
struct Config;

#[derive(Load)]
#[load(config = Config)]
struct Attention {
    #[load(prefix, config = LinearConfig::new(HIDDEN, QUERY_HEADS * HEAD_DIM))]
    q_proj: Linear<bf16>,
    #[load(prefix, config = LinearConfig::new(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    k_proj: Linear<bf16>,
    #[load(prefix, config = LinearConfig::new(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    v_proj: Linear<bf16>,
    #[load(prefix, config = LinearConfig::new(QUERY_HEADS * HEAD_DIM, HIDDEN))]
    o_proj: Linear<bf16>,
    #[load(prefix, config = RmsNormConfig::new(HEAD_DIM, RMS_EPSILON))]
    q_norm: RmsNorm<bf16>,
    #[load(prefix, config = RmsNormConfig::new(HEAD_DIM, RMS_EPSILON))]
    k_norm: RmsNorm<bf16>,
}

#[derive(Load)]
#[load(config = Config)]
struct Mlp {
    #[load(prefix, config = LinearConfig::new(HIDDEN, INTERMEDIATE))]
    gate_proj: Linear<bf16>,
    #[load(prefix, config = LinearConfig::new(HIDDEN, INTERMEDIATE))]
    up_proj: Linear<bf16>,
    #[load(prefix, config = LinearConfig::new(INTERMEDIATE, HIDDEN))]
    down_proj: Linear<bf16>,
}

#[derive(Load)]
#[load(config = Config)]
struct DecoderLayer {
    #[load(prefix, config = RmsNormConfig::new(HIDDEN, RMS_EPSILON))]
    input_layernorm: RmsNorm<bf16>,
    #[load(prefix)]
    self_attn: Attention,
    #[load(prefix, config = RmsNormConfig::new(HIDDEN, RMS_EPSILON))]
    post_attention_layernorm: RmsNorm<bf16>,
    #[load(prefix)]
    mlp: Mlp,
}

#[derive(Load)]
#[load(config = Config)]
struct Model {
    #[load(prefix, config = EmbeddingConfig::new(VOCAB, HIDDEN))]
    embed_tokens: Embedding<bf16>,
    #[load(prefix, count = LAYERS)]
    layers: Vec<DecoderLayer>,
    #[load(prefix, config = RmsNormConfig::new(HIDDEN, RMS_EPSILON))]
    norm: RmsNorm<bf16>,
}

#[derive(Load)]
#[load(config = Config)]
struct QwenWeights {
    #[load(prefix)]
    model: Model,
    #[load(prefix, config = LinearConfig::new(HIDDEN, VOCAB))]
    lm_head: Linear<bf16>,
}

struct LayerCache {
    key: Tensor<bf16>,
    value: Tensor<bf16>,
}

impl LayerCache {
    fn new() -> Result<Self> {
        let shape = [KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM];
        let count = usize::try_from(
            u64::from(KEY_VALUE_HEADS) * u64::from(MAX_CONTEXT) * u64::from(HEAD_DIM),
        )
        .map_err(|_| forja_sdk::Error::loading("KV cache size does not fit usize"))?;
        let zeros = vec![bf16::from_bits(0); count];
        Ok(Self {
            key: Tensor::from_slice(&zeros, &shape)?,
            value: Tensor::from_slice(&zeros, &shape)?,
        })
    }
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3 {
    weights: QwenWeights,
    caches: Vec<LayerCache>,
}

#[export_engine]
impl Engine for Qwen3 {
    /// Tap `i` is decoder layer `i`; tap 28 includes the final model norm.
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: (1..=28).collect(),
        }
    }

    fn load(weights: &Weights<'_>) -> Result<Self> {
        let weights = QwenWeights::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| LayerCache::new())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { weights, caches })
    }

    fn step(&mut self, _input: StepInput) -> Result<StepOutput> {
        Err(forja_sdk::Error::loading(
            "the full Qwen3 forward pass is not available",
        ))
    }
}
