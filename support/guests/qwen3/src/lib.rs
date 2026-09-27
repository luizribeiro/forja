//! Qwen3-0.6B inference engine.

use forja_sdk::{
    Engine, EngineInfo, Load, Result, StepInput, StepOutput, Tensor, Weights, bf16, export_engine,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, WeightElement,
        ops::sdpa,
    },
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

/// Activation precision supported by the Qwen3 engine.
pub trait Activation: WeightElement {
    /// Additive identity used to initialize the KV cache.
    const ZERO: Self;

    /// Selects the model's bf16 storage representation.
    fn linear_config(input: u32, output: u32) -> LinearConfig;

    /// Selects the model's bf16 storage representation.
    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig;

    /// Selects the model's bf16 storage representation.
    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig;

    /// Converts an engine result to the component's f32 output type.
    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>>;
}

impl Activation for f32 {
    const ZERO: Self = 0.0;

    fn linear_config(input: u32, output: u32) -> LinearConfig {
        LinearConfig::promoted_bf16(input, output)
    }

    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig {
        RmsNormConfig::promoted_bf16(hidden, epsilon)
    }

    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig {
        EmbeddingConfig::promoted_bf16(vocab, hidden)
    }

    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>> {
        Ok(tensor)
    }
}

impl Activation for bf16 {
    const ZERO: Self = bf16::ZERO;

    fn linear_config(input: u32, output: u32) -> LinearConfig {
        LinearConfig::new(input, output)
    }

    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig {
        RmsNormConfig::new(hidden, epsilon)
    }

    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig {
        EmbeddingConfig::new(vocab, hidden)
    }

    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>> {
        tensor.to_dtype()
    }
}

#[derive(Load)]
#[load(config = Config)]
struct Attention<T: Activation> {
    #[load(prefix, config = T::linear_config(HIDDEN, QUERY_HEADS * HEAD_DIM))]
    q_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    k_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    v_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(QUERY_HEADS * HEAD_DIM, HIDDEN))]
    o_proj: Linear<T>,
    #[load(prefix, config = T::norm_config(HEAD_DIM, RMS_EPSILON))]
    q_norm: RmsNorm<T>,
    #[load(prefix, config = T::norm_config(HEAD_DIM, RMS_EPSILON))]
    k_norm: RmsNorm<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct Mlp<T: Activation> {
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    gate_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    up_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(INTERMEDIATE, HIDDEN))]
    down_proj: Linear<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct DecoderLayer<T: Activation> {
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    input_layernorm: RmsNorm<T>,
    #[load(prefix)]
    self_attn: Attention<T>,
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    post_attention_layernorm: RmsNorm<T>,
    #[load(prefix)]
    mlp: Mlp<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct Model<T: Activation> {
    #[load(prefix, config = T::embedding_config(VOCAB, HIDDEN))]
    embed_tokens: Embedding<T>,
    #[load(prefix, count = LAYERS)]
    layers: Vec<DecoderLayer<T>>,
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    norm: RmsNorm<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct QwenWeights<T: Activation> {
    #[load(prefix)]
    model: Model<T>,
}

struct LayerCache<T: Activation> {
    key: Tensor<T>,
    value: Tensor<T>,
}

impl<T: Activation> LayerCache<T> {
    fn new() -> Result<Self> {
        let shape = [KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM];
        let count = usize::try_from(
            u64::from(KEY_VALUE_HEADS) * u64::from(MAX_CONTEXT) * u64::from(HEAD_DIM),
        )
        .map_err(|_| forja_sdk::Error::loading("KV cache size does not fit usize"))?;
        let zeros = vec![T::ZERO; count];
        Ok(Self {
            key: Tensor::from_slice(&zeros, &shape)?,
            value: Tensor::from_slice(&zeros, &shape)?,
        })
    }
}

impl<T: Activation> DecoderLayer<T> {
    fn forward(
        &self,
        input: &Tensor<T>,
        positions: &Tensor<u32>,
        start_pos: u32,
        end_pos: u32,
        cache: &mut LayerCache<T>,
    ) -> Result<Tensor<T>> {
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
        let attention = self.self_attn.o_proj.forward(&attended)?;
        let hidden = (input + &attention)?;
        let normalized = self.post_attention_layernorm.forward(&hidden)?;
        let gate = self.mlp.gate_proj.forward(&normalized)?;
        let up = self.mlp.up_proj.forward(&normalized)?;
        let projected = self.mlp.down_proj.forward(&gate.silu_mul(&up)?)?;
        &hidden + &projected
    }
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3<T: Activation = f32> {
    weights: QwenWeights<T>,
    caches: Vec<LayerCache<T>>,
}

impl<T: Activation> Qwen3<T> {
    fn output(tensor: Tensor<T>) -> Result<Tensor<f32>> {
        T::output(tensor)
    }

    /// Runs the embedding and first decoder layer for native differential checks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token or operation inputs, or failed execution.
    pub fn first_layer(&mut self, tokens: &Tensor<u32>) -> Result<Tensor<T>> {
        let sequence = tokens
            .shape()
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("tokens must have rank one"))?;
        let positions = positions(0, sequence)?;
        let hidden = self.weights.model.embed_tokens.forward(tokens)?;
        self.weights.model.layers[0]
            .forward(&hidden, &positions, 0, sequence, &mut self.caches[0])?
            .contiguous()
    }
}

#[cfg(all(target_family = "wasm", feature = "bf16"))]
type ExportedQwen3 = Qwen3<bf16>;
#[cfg(not(all(target_family = "wasm", feature = "bf16")))]
type ExportedQwen3 = Qwen3<f32>;

#[export_engine]
impl Engine for ExportedQwen3 {
    /// Tap `i` is decoder layer `i`; tap 28 includes the final model norm.
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: (1..=28).collect(),
        }
    }

    fn load(weights: &Weights<'_>) -> Result<Self> {
        let weights = QwenWeights::<_>::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| LayerCache::new())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { weights, caches })
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        let [sequence] = input
            .tokens
            .shape()
            .try_into()
            .map_err(|_| forja_sdk::Error::loading("tokens must have rank one"))?;
        let last = sequence
            .checked_sub(1)
            .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?;
        let end_pos = input
            .start_pos
            .checked_add(sequence)
            .filter(|&end| end <= MAX_CONTEXT)
            .ok_or_else(|| forja_sdk::Error::loading("tokens exceed the 4096-token context"))?;
        let positions = positions(input.start_pos, end_pos)?;
        let mut hidden = self.weights.model.embed_tokens.forward(&input.tokens)?;
        let mut taps = Vec::with_capacity(if input.taps { LAYERS } else { 0 });
        for (index, (layer, cache)) in self
            .weights
            .model
            .layers
            .iter()
            .zip(&mut self.caches)
            .enumerate()
        {
            hidden = layer.forward(&hidden, &positions, input.start_pos, end_pos, cache)?;
            if input.taps && index + 1 < LAYERS {
                taps.push(Self::output(hidden.contiguous()?)?);
            }
        }
        hidden = self.weights.model.norm.forward(&hidden)?;
        if input.taps {
            taps.push(Self::output(hidden.contiguous()?)?);
        }
        let logits = Self::output(
            self.weights
                .model
                .embed_tokens
                .project(&hidden)?
                .narrow(0, last, 1)?
                .reshape(&[VOCAB])?
                .contiguous()?,
        )?;
        Ok(StepOutput { logits, taps })
    }
}

fn positions(start: u32, end: u32) -> Result<Tensor<u32>> {
    Tensor::from_slice(&(start..end).collect::<Vec<_>>(), &[end - start])
}
