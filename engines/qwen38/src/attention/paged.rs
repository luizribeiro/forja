use super::{Attention, Cache};
use crate::Config;
use forja_sdk::{
    Dim, Error, PagedCacheSet, Result, Tensor, Weights, bf16, f16, target::metal::Variant,
};
use std::rc::Rc;

impl Attention {
    /// Loads weights with an opaque paged cache and shared absolute position storage.
    ///
    /// # Errors
    /// Refuses invalid cache indices, position geometry or weights.
    pub fn load_paged(
        weights: &Weights<'_>,
        config: Config,
        set: Rc<PagedCacheSet>,
        layer: u32,
        positions: Tensor<u32>,
    ) -> Result<Self> {
        config.validate()?;
        set.layer(layer)?;
        if positions.shape().len() != 1 || positions.shape()[0] == 0 {
            return Err(Error::loading(
                "attention positions require a nonempty vector",
            ));
        }
        Self::load_with_cache(
            weights,
            config,
            Cache::Paged {
                set,
                layer,
                lane: 0,
                positions,
            },
        )
    }

    /// Selects the physical serving lane for the next attention recording.
    ///
    /// # Errors
    /// Refuses use with a contiguous cache.
    pub fn select_lane(&mut self, lane: u32) -> Result<()> {
        let Cache::Paged { lane: selected, .. } = &mut self.cache else {
            return Err(Error::loading("attention has no serving slot"));
        };
        *selected = lane;
        Ok(())
    }

    /// Processes one admitted prompt piece or decode token using its paged lane.
    ///
    /// # Errors
    /// Refuses incompatible input, cache spans, variants or host operations.
    pub fn paged_layer(
        &self,
        input: &Tensor<bf16>,
        start: &Dim,
        prefill: bool,
        quant: Option<&Variant>,
        attention: &Variant,
    ) -> Result<Tensor<bf16>> {
        let rows = input.shape().first().copied().unwrap_or(0);
        if input.shape() != [rows, self.config.hidden]
            || !(1..=2048).contains(&rows)
            || (!prefill && rows != 1)
        {
            return Err(Error::loading("invalid paged attention span"));
        }
        let Cache::Paged {
            set,
            layer,
            lane,
            positions,
        } = &self.cache
        else {
            return Err(Error::loading("attention has no serving slot"));
        };
        let projected = self.project(input, &positions.narrow(0, start.clone(), rows)?, quant)?;
        let end = start.clone().checked_add(rows)?;
        let cache = set.layer_at(*lane, *layer)?;
        let scale = crate::numerics::inverse_sqrt_width(self.config.head_dim)?;
        cache.append(
            &projected.key.to_dtype::<f16>()?,
            &projected.value.to_dtype::<f16>()?,
            start,
        )?;
        let attended = if prefill && !self.config.cpu {
            cache
                .attention(
                    &projected.query.to_dtype::<f16>()?,
                    scale,
                    start,
                    &end,
                    attention,
                )?
                .to_dtype::<f32>()?
        } else {
            cache.attention(&projected.query, scale, start, &end, attention)?
        };
        self.output(&attended, &projected.gate, quant)
    }
}
