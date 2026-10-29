//! Llama in execution order: embeddings, pre-norm decoder layers, final norm,
//! and a vocabulary projection tied to the embedding table.
#![forbid(unsafe_code)]
use crate::{Attention, Config, DecoderLayer, Mlp, ModelWeights};
use forja_sdk::{
    Dim, Result, StepOutput, Tensor,
    nn::{
        Embedding, RmsNorm,
        blocks::{KvCache, cached_attention_with},
    },
    target::metal::Variant,
};
/// Unfused Llama with a contiguous KV cache.
pub struct Reference {
    weights: ModelWeights,
    config: Config,
    context: u32,
    caches: Vec<KvCache<f32>>,
    positions: Tensor<u32>,
}
impl Reference {
    /// Loads the shared bf16 weights promoted to f32.
    ///
    /// # Errors
    /// Returns an error for invalid dimensions, context, weights or allocations.
    pub fn load(weights: &forja_sdk::Weights<'_>, config: Config, context: u32) -> Result<Self> {
        config.validate()?;
        if context == 0 || context > config.context {
            return Err(forja_sdk::Error::loading("unsupported context"));
        }
        Ok(Self {
            weights: <ModelWeights as forja_sdk::Load<Config>>::load(weights, &config)?,
            config,
            context,
            caches: (0..config.layers)
                .map(|_| KvCache::new(config.kv_heads, context, config.head_dim, 0.0))
                .collect::<Result<_>>()?,
            positions: Tensor::constant(&(0..context).collect::<Vec<_>>(), &[context])?,
        })
    }
    /// Runs embeddings, all decoder layers and the tied vocabulary projection.
    ///
    /// # Errors
    /// Returns an error for invalid tokens, context, or refused tensor operations.
    pub fn forward(
        &mut self,
        tokens: &Tensor<u32>,
        start: u32,
        taps_enabled: bool,
    ) -> Result<StepOutput> {
        let sequence = input_length(tokens.shape(), start, self.context)?;
        let weights = &self.weights;
        let config = self.config;
        let positions = self.positions.narrow(0, start, sequence)?;
        let mut hidden = embed(&weights.model.embed_tokens, tokens)?;
        let mut taps = Vec::new();
        for (index, (layer, cache)) in weights
            .model
            .layers
            .iter()
            .zip(&mut self.caches)
            .enumerate()
        {
            hidden = decoder_layer(layer, &hidden, cache, &positions, config, start)?;
            if taps_enabled && index + 1 < config.layers {
                taps.push(hidden.contiguous()?);
            }
        }
        hidden = rms_norm(&weights.model.norm, &hidden)?;
        if taps_enabled {
            taps.push(hidden.contiguous()?);
        }
        let last = hidden.narrow(0, sequence - 1, 1)?;
        let logits = weights
            .model
            .embed_tokens
            .project_with(&last, &Variant::new("matmul.portable")?)?
            .reshape(&[config.vocab])?
            .contiguous()?;
        Ok(StepOutput {
            logits,
            taps,
            router_logits: Vec::new(),
        })
    }
}
fn embed(table: &Embedding<f32>, tokens: &Tensor<u32>) -> Result<Tensor<f32>> {
    table.forward(tokens)
}
fn rms_norm(norm: &RmsNorm<f32>, hidden: &Tensor<f32>) -> Result<Tensor<f32>> {
    norm.forward(hidden)
}
fn decoder_layer(
    layer: &DecoderLayer,
    hidden: &Tensor<f32>,
    cache: &mut KvCache<f32>,
    positions: &Tensor<u32>,
    config: Config,
    start: u32,
) -> Result<Tensor<f32>> {
    let normalized = rms_norm(&layer.input_layernorm, hidden)?;
    let attended = attention(
        &layer.self_attn,
        &normalized,
        cache,
        positions,
        config,
        start,
    )?;
    let residual = (hidden + &attended)?;
    let normalized = rms_norm(&layer.post_attention_layernorm, &residual)?;
    &residual + &mlp(&layer.mlp, &normalized)?
}
fn attention(
    weights: &Attention,
    hidden: &Tensor<f32>,
    cache: &mut KvCache<f32>,
    positions: &Tensor<u32>,
    config: Config,
    start: u32,
) -> Result<Tensor<f32>> {
    let sequence = hidden.shape()[0];
    let query = linear(&weights.q_proj, hidden)?
        .reshape(&[sequence, config.query_heads, config.head_dim])?
        .rope(positions, config.theta)?;
    let key = linear(&weights.k_proj, hidden)?
        .reshape(&[sequence, config.kv_heads, config.head_dim])?
        .rope(positions, config.theta)?;
    let value =
        linear(&weights.v_proj, hidden)?.reshape(&[sequence, config.kv_heads, config.head_dim])?;
    let end = Dim::from(
        start
            .checked_add(sequence)
            .ok_or_else(|| forja_sdk::Error::loading("attention position overflow"))?,
    );
    let start = Dim::from(start);
    let attended = cached_attention_with(
        &query,
        &key,
        &value,
        cache,
        config.attention_scale(),
        &start,
        &end,
        &Variant::new("sdpa.decomposed")?,
    )?;
    linear(&weights.o_proj, &attended)
}
fn mlp(weights: &Mlp, hidden: &Tensor<f32>) -> Result<Tensor<f32>> {
    let gate = linear(&weights.gate_proj, hidden)?;
    let up = linear(&weights.up_proj, hidden)?;
    linear(&weights.down_proj, &gate.silu_mul(&up)?)
}
fn linear(weights: &forja_sdk::nn::Linear<f32>, hidden: &Tensor<f32>) -> Result<Tensor<f32>> {
    weights.forward_with(hidden, &Variant::new("matmul.portable")?)
}
pub(crate) fn input_length(shape: &[u32], start: u32, context: u32) -> Result<u32> {
    match shape {
        [n] if *n > 0 && start.checked_add(*n).is_some_and(|end| end <= context) => Ok(*n),
        _ => Err(forja_sdk::Error::loading("invalid token shape or context")),
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn input_boundaries_are_checked_before_execution() {
        for n in [1, 7, 33] {
            assert_eq!(super::input_length(&[n], 0, 33).unwrap(), n);
        }
        for shape in [&[][..], &[0][..], &[1, 7][..], &[34][..]] {
            assert!(super::input_length(shape, 0, 33).is_err());
        }
        assert!(super::input_length(&[7], u32::MAX, 33).is_err());
    }
}
