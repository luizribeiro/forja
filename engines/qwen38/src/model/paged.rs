use super::Model;
use crate::Config;
use forja_sdk::{
    BatchPhase, Dim, EngineLoadConfig, EngineSelections, Error, PagedCacheSet, PagedInputLayout,
    PagedPool, Result, StepOutput, Tensor, Weights, target::metal::Variant,
};
use std::rc::Rc;

pub(super) struct PagedModel {
    pub(super) set: Rc<PagedCacheSet>,
    pub(super) positions: Tensor<u32>,
    prompt: Variant,
    pub(super) decode: Variant,
}
impl PagedModel {
    pub(super) fn new(
        pool: PagedPool<'_>,
        config: Config,
        context: u32,
        picks: Option<&EngineSelections>,
    ) -> Result<Self> {
        let pairs = config.layers / config.attention_interval;
        let set = Rc::new(pool.cache_set_with_state(
            pairs.max(1),
            &config.state_planes()?,
            PagedInputLayout {
                vocab: config.vocab,
                prompt_tokens: context.min(2048),
            },
        )?);
        let choice = |site| {
            picks.map_or_else(
                || Variant::new("paged-sdpa.vector"),
                |picks| picks.at(site, 0),
            )
        };
        Ok(Self {
            set,
            positions: Tensor::constant(&(0..context).collect::<Vec<_>>(), &[context])?,
            prompt: choice("paged-attention.prefill")?,
            decode: choice("paged-attention.decode")?,
        })
    }
}
impl Model {
    #[cfg(target_family = "wasm")]
    pub(crate) fn serving_set(&self) -> Result<Rc<PagedCacheSet>> {
        self.paged
            .as_ref()
            .map(|paged| Rc::clone(&paged.set))
            .ok_or_else(|| Error::loading("model has no serving slot"))
    }

    /// Loads the complete hybrid model into a shared host serving slot.
    ///
    /// # Errors
    /// Refuses invalid architecture, capacity, selections or weights.
    pub fn load_paged(
        weights: &Weights<'_>,
        load: &EngineLoadConfig,
        config: Config,
        pool: PagedPool<'_>,
    ) -> Result<Self> {
        Self::load_inner(weights, load, config, Some(pool))
    }
    /// Records a lane's whole admitted piece, publishing each recurrent plane once.
    ///
    /// # Errors
    /// Refuses incompatible spans, phases, cache bindings or tensor operations.
    pub fn forward_paged(
        &mut self,
        tokens: &Tensor<u32>,
        start: &Dim,
        lane: u32,
        phase: BatchPhase,
        taps: bool,
    ) -> Result<StepOutput> {
        let [rows]: [u32; 1] = tokens
            .shape()
            .try_into()
            .map_err(|_| Error::loading("tokens must have rank one"))?;
        let prefill = phase == BatchPhase::Prompt;
        let end = start.clone().checked_add(rows)?.value();
        if !(1..=2048).contains(&rows)
            || end > self.context
            || (prefill && !start.value().is_multiple_of(crate::PREFILL_BLOCK_TOKENS))
            || (!prefill && rows != 1)
        {
            return Err(Error::loading("invalid serving token span"));
        }
        let paged = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::loading("model has no serving slot"))?;
        let attention = if prefill {
            &paged.prompt
        } else {
            &paged.decode
        };
        let mut captured = Vec::new();
        forja_sdk::profile_scope("embedding", "global", None)?;
        let mut hidden = self.embedding.embed(tokens)?;
        let layers = self.layers.len();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer.paged_layer(
                &hidden,
                lane,
                start,
                phase,
                [self.quant.as_ref(), Some(attention), self.delta.as_ref()],
            )?;
            if taps && index + 1 < layers {
                captured.push(hidden.to_dtype::<f32>()?);
            }
        }
        forja_sdk::profile_scope("output", "global", None)?;
        hidden = hidden.rms_norm(&self.norm, 1.0e-6)?;
        if taps {
            captured.push(hidden.to_dtype::<f32>()?);
        }
        Ok(StepOutput {
            logits: self
                .output
                .forward(&hidden.narrow(0, rows - 1, 1)?, self.quant.as_ref())?
                .to_dtype::<f32>()?
                .reshape(&[self.config.vocab])?,
            taps: captured,
            router_logits: Vec::new(),
        })
    }
}
