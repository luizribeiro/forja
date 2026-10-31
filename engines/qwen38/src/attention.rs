mod batch;
mod paged;
use crate::{Config, numerics, quant::Affine};
use forja_sdk::{
    Dim, Error, Result, Tensor, Weights, bf16, f16,
    nn::blocks::{KvCache, cached_attention, cached_attention_with},
    target::metal::Variant,
};

/// Grouped-query attention with per-head Q/K normalization and a sigmoid output gate.
pub struct Attention {
    config: Config,
    q: Affine,
    k: Affine,
    v: Affine,
    out: Affine,
    q_norm: Tensor<bf16>,
    k_norm: Tensor<bf16>,
    cache: Cache,
}
impl Attention {
    /// Processes a canonical prompt block using the pinned per-row attention reduction.
    ///
    /// # Errors
    /// Refuses invalid spans, cache ranges or failed tensor operations.
    pub fn prefill(
        &mut self,
        input: &Tensor<bf16>,
        position: u32,
        quant: Option<&Variant>,
        attention: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let tokens = input.shape().first().copied().unwrap_or(0);
        if input.shape() != [tokens, self.config.hidden]
            || !(1..=crate::PREFILL_BLOCK_TOKENS).contains(&tokens)
        {
            return Err(Error::loading("invalid attention prefill block"));
        }
        let output = Tensor::zeros(input.shape())?;
        for row in 0..tokens {
            let absolute = position
                .checked_add(row)
                .ok_or_else(|| Error::loading("attention position overflow"))?;
            self.attention_layer(&input.narrow(0, row, 1)?, absolute, quant, attention)?
                .copy_into(&mut output.narrow(0, row, 1)?)?;
        }
        Ok(output)
    }

    /// Loads a full attention layer and allocates a contiguous reduced-precision cache.
    ///
    /// # Errors
    /// Refuses invalid geometry, missing weights or failed allocations.
    pub fn load(weights: &Weights<'_>, config: Config, context: u32) -> Result<Self> {
        config.validate()?;
        let cache = Cache::Eager(KvCache::new(
            config.kv_heads,
            context,
            config.head_dim,
            f16::from_f32(0.0),
        )?);
        Self::load_with_cache(weights, config, cache)
    }

    fn load_with_cache(weights: &Weights<'_>, config: Config, cache: Cache) -> Result<Self> {
        Ok(Self {
            config,
            q: Affine::load(
                &weights.scoped("q_proj"),
                config.hidden,
                2 * config.query_heads * config.head_dim,
            )?,
            k: Affine::load(
                &weights.scoped("k_proj"),
                config.hidden,
                config.kv_heads * config.head_dim,
            )?,
            v: Affine::load(
                &weights.scoped("v_proj"),
                config.hidden,
                config.kv_heads * config.head_dim,
            )?,
            out: Affine::load(
                &weights.scoped("o_proj"),
                config.query_heads * config.head_dim,
                config.hidden,
            )?,
            q_norm: weights.tensor("q_norm.weight", &[config.head_dim])?,
            k_norm: weights.tensor("k_norm.weight", &[config.head_dim])?,
            cache,
        })
    }
    /// Processes one normalized token at its absolute position.
    ///
    /// # Errors
    /// Refuses invalid input/cache ranges or failed tensor operations.
    pub fn attention_layer(
        &mut self,
        input: &Tensor<bf16>,
        position: u32,
        quant: Option<&Variant>,
        attention: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        if input.shape() != [1, self.config.hidden] {
            return Err(Error::loading("attention requires one normalized token"));
        }
        let end = position
            .checked_add(1)
            .ok_or_else(|| Error::loading("attention position overflow"))?;
        let positions = Tensor::constant(&[position], &[1])?;
        let projected = self.project(input, &positions, quant)?;
        let scale = crate::numerics::inverse_sqrt_width(self.config.head_dim)?;
        let Cache::Eager(cache) = &mut self.cache else {
            return Err(Error::loading("paged attention requires a serving lane"));
        };
        let attended = match attention {
            Some(pick) => cached_attention_with(
                &projected.query,
                &projected.key,
                &projected.value,
                cache,
                scale,
                &Dim::from(position),
                &Dim::from(end),
                pick,
            )?,
            None => cached_attention(
                &projected.query,
                &projected.key,
                &projected.value,
                cache,
                scale,
                &Dim::from(position),
                &Dim::from(end),
            )?,
        };
        self.output(&attended, &projected.gate, quant)
    }

    fn project(
        &self,
        input: &Tensor<bf16>,
        positions: &Tensor<u32>,
        quant: Option<&Variant>,
    ) -> Result<AttentionProjection> {
        let c = self.config;
        let rows = input.shape()[0];
        let projected =
            self.q
                .forward(input, quant)?
                .reshape(&[rows, c.query_heads, 2 * c.head_dim])?;
        let query = projected
            .narrow(2, 0, c.head_dim)?
            .contiguous()?
            .rms_norm(&self.q_norm, 1.0e-6)?;
        let gate = projected
            .narrow(2, c.head_dim, c.head_dim)?
            .contiguous()?
            .reshape(&[rows, c.query_heads * c.head_dim])?;
        let key = self
            .k
            .forward(input, quant)?
            .reshape(&[rows, c.kv_heads, c.head_dim])?
            .rms_norm(&self.k_norm, 1.0e-6)?;
        let query = partial_rope(&query, positions, c.rotary_dim)?.to_dtype::<f32>()?;
        let key = partial_rope(&key, positions, c.rotary_dim)?.to_dtype::<f32>()?;
        let value = self
            .v
            .forward(input, quant)?
            .reshape(&[rows, c.kv_heads, c.head_dim])?
            .to_dtype::<f32>()?;
        Ok(AttentionProjection {
            query,
            key,
            value,
            gate,
        })
    }

    fn output(
        &self,
        attended: &Tensor<f32>,
        gate: &Tensor<bf16>,
        quant: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let attended = attended.to_dtype::<bf16>()?;
        let gated = numerics::multiply(&attended, &numerics::sigmoid_bf16(gate)?)?;
        self.out.forward(&gated, quant)
    }
}

struct AttentionProjection {
    query: Tensor<f32>,
    key: Tensor<f32>,
    value: Tensor<f32>,
    gate: Tensor<bf16>,
}
fn partial_rope(input: &Tensor<bf16>, positions: &Tensor<u32>, width: u32) -> Result<Tensor<bf16>> {
    let rotated = input.narrow(2, 0, width)?.rope(positions, 10_000_000.0)?;
    let output = input.contiguous()?;
    rotated.copy_into(&mut output.narrow(2, 0, width)?)?;
    Ok(output)
}

enum Cache {
    Eager(KvCache<f16>),
    Paged {
        set: std::rc::Rc<forja_sdk::PagedCacheSet>,
        layer: u32,
        lane: u32,
        positions: Tensor<u32>,
    },
}
