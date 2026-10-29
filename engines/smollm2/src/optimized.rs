//! Dense Llama execution with profile-owned matrix and attention choices.

#![forbid(unsafe_code)]

use crate::{Config, ModelWeights};
use forja_sdk::{
    Dim, EngineLoadConfig, Load, Param, Result, StepOutput, Tensor, Weights,
    nn::{
        Linear,
        blocks::{KvCache, cached_attention_with},
    },
};

pub(crate) struct Optimized {
    weights: ModelWeights,
    config: Config,
    variants: crate::variants::Variants,
    caches: Vec<KvCache<f32>>,
    positions: Tensor<u32>,
}

impl Optimized {
    pub(crate) fn load(
        weights: &Weights<'_>,
        config: Config,
        context: u32,
        load: &EngineLoadConfig,
    ) -> Result<Self> {
        let caches = (0..config.layers)
            .map(|_| KvCache::new(config.kv_heads, context, config.head_dim, 0.0))
            .collect::<Result<_>>()?;
        Ok(Self {
            weights: ModelWeights::load(weights, &config)?,
            config,
            variants: crate::variants::Variants::new(load, context)?,
            caches,
            positions: Tensor::constant(&(0..context).collect::<Vec<_>>(), &[context])?,
        })
    }

    fn linear(&self, projection: &Linear<f32>, hidden: &Tensor<f32>) -> Result<Tensor<f32>> {
        projection.forward_with(hidden, &self.variants.linear(hidden.shape()[0])?)
    }

    fn add_norm(
        hidden: &Tensor<f32>,
        update: &Tensor<f32>,
        norm: &forja_sdk::nn::RmsNorm<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let hidden = (hidden + update)?;
        let normalized = norm.forward(&hidden)?;
        Ok((hidden, normalized))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        last: &Dim,
        pick_start: u32,
        parameter: Option<&Param>,
        taps_enabled: bool,
    ) -> Result<StepOutput> {
        let c = self.config;
        let pick = self.variants.attention(sequence, pick_start, parameter)?;
        let positions = self.positions.narrow(0, start, sequence)?;
        forja_sdk::profile_scope("embedding", "global", None)?;
        let mut hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let first = self
            .weights
            .model
            .layers
            .first()
            .ok_or_else(|| forja_sdk::Error::loading("model has no layers"))?;
        let mut normalized = first.input_layernorm.forward(&hidden)?;
        let mut taps = Vec::new();
        for (index, layer) in self.weights.model.layers.iter().enumerate() {
            forja_sdk::profile_scope(
                "decoder",
                "dense-decoder",
                Some(
                    u32::try_from(index)
                        .map_err(|_| forja_sdk::Error::loading("layer index exceeds u32"))?,
                ),
            )?;
            let a = &layer.self_attn;
            let query = self
                .linear(&a.q_proj, &normalized)?
                .reshape(&[sequence, c.query_heads, c.head_dim])?
                .rope(&positions, c.theta)?;
            let key = self
                .linear(&a.k_proj, &normalized)?
                .reshape(&[sequence, c.kv_heads, c.head_dim])?
                .rope(&positions, c.theta)?;
            let value = self
                .linear(&a.v_proj, &normalized)?
                .reshape(&[sequence, c.kv_heads, c.head_dim])?;
            let attended = cached_attention_with(
                &query,
                &key,
                &value,
                &mut self.caches[index],
                c.attention_scale(),
                start,
                end,
                pick.choice(),
            )?;
            let update = self.linear(&a.o_proj, &attended)?;
            let (residual, mlp_input) =
                Self::add_norm(&hidden, &update, &layer.post_attention_layernorm)?;
            let gate = self.linear(&layer.mlp.gate_proj, &mlp_input)?;
            let up = self.linear(&layer.mlp.up_proj, &mlp_input)?;
            let update = self.linear(&layer.mlp.down_proj, &gate.silu_mul(&up)?)?;
            if let Some(next) = self.weights.model.layers.get(index + 1) {
                (hidden, normalized) = Self::add_norm(&residual, &update, &next.input_layernorm)?;
            } else {
                hidden = (&residual + &update)?;
            }
            if taps_enabled && index + 1 < c.layers {
                taps.push(hidden.contiguous()?);
            }
        }
        forja_sdk::profile_scope("output", "global", None)?;
        hidden = self.weights.model.norm.forward(&hidden)?;
        if taps_enabled {
            taps.push(hidden.contiguous()?);
        }
        let logits = self
            .weights
            .model
            .embed_tokens
            .project_with(&hidden.narrow(0, last, 1)?, &self.variants.linear(1)?)?
            .reshape(&[c.vocab])?
            .contiguous()?;
        Ok(StepOutput {
            logits,
            taps,
            router_logits: Vec::new(),
        })
    }
}
