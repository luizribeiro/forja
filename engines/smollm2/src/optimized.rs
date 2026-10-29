//! Dense Llama execution with profile-owned matrix and attention choices.

#![forbid(unsafe_code)]

use crate::{Config, ModelWeights};
use forja_sdk::{
    DType, Dim, EngineLoadConfig, Load, PagedCacheSet, PagedPool, Param, Result, StepOutput,
    Tensor, Weights,
    kernel::{Epilogue, Kernel},
    nn::blocks::{KvCache, residual_norm, residual_norm_kernel},
};
#[cfg(target_family = "wasm")]
use forja_sdk::{kernel::TensorRef, nn::blocks::cached_attention_with};

pub(crate) struct Optimized {
    weights: ModelWeights,
    config: Config,
    #[cfg(target_family = "wasm")]
    pub(crate) context: u32,
    variants: crate::variants::Variants,
    caches: Caches,
    positions: Tensor<u32>,
    fused: bool,
    gate_up: Epilogue,
    residual_norm: Kernel,
}

enum Caches {
    Contiguous(Vec<KvCache<f32>>),
    Paged(PagedCacheSet),
}

impl Optimized {
    pub(crate) fn load(
        weights: &Weights<'_>,
        config: Config,
        context: u32,
        load: &EngineLoadConfig,
        pool: Option<PagedPool<'_>>,
    ) -> Result<Self> {
        let caches = if let Some(pool) = pool {
            Caches::Paged(
                pool.cache_set(
                    u32::try_from(config.layers)
                        .map_err(|_| forja_sdk::Error::loading("layer count exceeds u32"))?,
                )?,
            )
        } else {
            Caches::Contiguous(
                (0..config.layers)
                    .map(|_| KvCache::new(config.kv_heads, context, config.head_dim, 0.0))
                    .collect::<Result<_>>()?,
            )
        };
        #[cfg(not(target_family = "wasm"))]
        let weights = ModelWeights::load(weights, &config)?.for_cpu(config.cpu);
        Ok(Self {
            #[cfg(target_family = "wasm")]
            weights: ModelWeights::load(weights, &config)?,
            #[cfg(not(target_family = "wasm"))]
            weights,
            config,
            #[cfg(target_family = "wasm")]
            context,
            fused: load.tunings.iter().any(|name| name == "fused-epilogues"),
            gate_up: gated_up_epilogue(1, 2, &[DType::F32], DType::F32)?,
            residual_norm: residual_norm_kernel(DType::F32, config.epsilon)?,
            variants: crate::variants::Variants::new(load, context)?,
            caches,
            positions: Tensor::constant(&(0..context).collect::<Vec<_>>(), &[context])?,
        })
    }

    fn linear(&self, projection: &crate::Projection, hidden: &Tensor<f32>) -> Result<Tensor<f32>> {
        projection.forward_with(hidden, &self.variants.linear(hidden.shape()[0])?)
    }

    fn add_norm(
        &self,
        hidden: &Tensor<f32>,
        update: &Tensor<f32>,
        norm: &forja_sdk::nn::RmsNorm<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if self.fused {
            return residual_norm(&self.residual_norm, hidden, update, norm.weight());
        }
        let hidden = (hidden + update)?;
        let normalized = norm.forward(&hidden)?;
        Ok((hidden, normalized))
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "native test fallbacks retain the execution order"
    )]
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
        let pick = self.variants.attention(
            sequence,
            pick_start,
            parameter,
            matches!(self.caches, Caches::Paged(_)),
        )?;
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
            let attended = match &mut self.caches {
                Caches::Contiguous(caches) => {
                    #[cfg(not(target_family = "wasm"))]
                    {
                        crate::native_attention(
                            [&query, &key, &value],
                            &mut caches[index],
                            [start, end],
                            c,
                            &pick,
                        )?
                    }
                    #[cfg(target_family = "wasm")]
                    {
                        cached_attention_with(
                            &query,
                            &key,
                            &value,
                            &mut caches[index],
                            c.attention_scale(),
                            start,
                            end,
                            pick.choice(),
                        )?
                    }
                }
                Caches::Paged(caches) => {
                    let cache = caches.layer(
                        u32::try_from(index)
                            .map_err(|_| forja_sdk::Error::loading("cache index exceeds u32"))?,
                    )?;
                    cache.append(&key, &value, start)?;
                    cache.attention(&query, c.attention_scale(), start, end, pick.choice())?
                }
            };
            let update = self.linear(&a.o_proj, &attended)?;
            let (residual, mlp_input) =
                self.add_norm(&hidden, &update, &layer.post_attention_layernorm)?;
            let gate = self.linear(&layer.mlp.gate_proj, &mlp_input)?;
            let activated = if self.fused {
                #[cfg(not(target_family = "wasm"))]
                {
                    crate::native_gate_up(
                        &mlp_input,
                        &layer.mlp.up_proj,
                        &gate,
                        &self.gate_up,
                        &self.variants.linear(sequence)?,
                    )?
                }
                #[cfg(target_family = "wasm")]
                {
                    mlp_input.matmul_epilogue_with(
                        &layer.mlp.up_proj.weight.t()?,
                        &self.gate_up,
                        &[TensorRef::new(&gate)?],
                        &self.variants.linear(sequence)?,
                    )?
                }
            } else {
                gate.silu_mul(&self.linear(&layer.mlp.up_proj, &mlp_input)?)?
            };
            let update = self.linear(&layer.mlp.down_proj, &activated)?;
            if let Some(next) = self.weights.model.layers.get(index + 1) {
                (hidden, normalized) = self.add_norm(&residual, &update, &next.input_layernorm)?;
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
        #[cfg(not(target_family = "wasm"))]
        let logits = crate::native_project(
            &self.weights.model.embed_tokens,
            &hidden.narrow(0, last, 1)?,
            c.cpu,
            &self.variants.linear(1)?,
        )?
        .reshape(&[c.vocab])?
        .contiguous()?;
        #[cfg(target_family = "wasm")]
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

#[forja_sdk::kernel::kernel(map)]
fn gated_up(up: forja_sdk::kernel::Elem, gate: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    gate * gate.sigmoid() * up
}
