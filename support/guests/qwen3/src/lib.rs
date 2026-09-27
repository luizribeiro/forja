//! Qwen3-0.6B inference engine.

#![allow(dead_code, reason = "model fields are exercised as forward paths land")]

use forja_sdk::{
    Engine, EngineInfo, Load, Result, StepInput, StepOutput, Tensor, Weights, bf16, export_engine,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, ops::sdpa},
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
const ROPE_THETA: f32 = 1.0e6;
const ATTENTION_SCALE: f32 = 0.088_388_35;

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

impl DecoderLayer {
    fn forward(
        &self,
        input: &Tensor<bf16>,
        positions: &Tensor<u32>,
        start_pos: u32,
        end_pos: u32,
        cache: &mut LayerCache,
    ) -> Result<Tensor<bf16>> {
        let sequence = end_pos - start_pos;
        let normalized = self.input_layernorm.forward(input)?;
        let query = self
            .self_attn
            .q_norm
            .forward(&self.self_attn.q_proj.forward(&normalized)?.reshape(&[
                sequence,
                QUERY_HEADS,
                HEAD_DIM,
            ])?)?
            .rope(positions, ROPE_THETA)?
            .permute(&[1, 0, 2])?;
        let key = self
            .self_attn
            .k_norm
            .forward(&self.self_attn.k_proj.forward(&normalized)?.reshape(&[
                sequence,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            ])?)?
            .rope(positions, ROPE_THETA)?
            .permute(&[1, 0, 2])?;
        let value = self
            .self_attn
            .v_proj
            .forward(&normalized)?
            .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?
            .permute(&[1, 0, 2])?;
        key.copy_into(&mut cache.key.narrow(1, start_pos, sequence)?)?;
        value.copy_into(&mut cache.value.narrow(1, start_pos, sequence)?)?;
        let attended = sdpa(
            &query,
            &cache.key.narrow(1, 0, end_pos)?,
            &cache.value.narrow(1, 0, end_pos)?,
            ATTENTION_SCALE,
            true,
            start_pos,
        )?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[sequence, QUERY_HEADS * HEAD_DIM])?;
        let hidden = (input + &self.self_attn.o_proj.forward(&attended)?)?;
        let normalized = self.post_attention_layernorm.forward(&hidden)?;
        let gate = self.mlp.gate_proj.forward(&normalized)?;
        let up = self.mlp.up_proj.forward(&normalized)?;
        let projected = self.mlp.down_proj.forward(&gate.silu_mul(&up)?)?;
        &hidden + &projected
    }
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3 {
    weights: QwenWeights,
    caches: Vec<LayerCache>,
}

impl Qwen3 {
    /// Runs the embedding and first decoder layer for native differential checks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token or operation inputs, or failed execution.
    pub fn first_layer(&mut self, tokens: &Tensor<u32>) -> Result<Tensor<f32>> {
        let sequence = tokens
            .shape()
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("tokens must have rank one"))?;
        let positions = Tensor::from_slice(&(0..sequence).collect::<Vec<_>>(), &[sequence])?;
        let hidden = self.weights.model.embed_tokens.forward(tokens)?;
        self.weights.model.layers[0]
            .forward(&hidden, &positions, 0, sequence, &mut self.caches[0])?
            .to_dtype()
    }
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
