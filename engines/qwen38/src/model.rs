#[cfg(target_family = "wasm")]
mod batch;
mod paged;
use crate::{Config, DecoderLayer, quant::Affine};
use forja_sdk::{
    EngineLoadConfig, EngineSelections, Error, Result, StepOutput, Tensor, Weights, bf16,
    target::metal::Variant,
};
/// Hybrid text model with eager or host-owned serving state.
pub struct Model {
    config: Config,
    paged: Option<paged::PagedModel>,
    context: u32,
    position: u32,
    usable: bool,
    embedding: Affine,
    output: Affine,
    norm: Tensor<bf16>,
    layers: Vec<DecoderLayer>,
    quant: Option<Variant>,
    attention: Option<Variant>,
    delta: Option<Variant>,
    pub(crate) sample: Option<Variant>,
}
impl Model {
    /// Loads the text checkpoint and validates every tensor required by its declared layers.
    ///
    /// # Errors
    /// Refuses invalid dimensions, profile selections, context, weights or allocations.
    pub fn load(weights: &Weights<'_>, load: &EngineLoadConfig, config: Config) -> Result<Self> {
        Self::load_inner(weights, load, config, None)
    }

    fn load_inner(
        weights: &Weights<'_>,
        load: &EngineLoadConfig,
        config: Config,
        pool: Option<forja_sdk::PagedPool<'_>>,
    ) -> Result<Self> {
        let (config, context) = load_config(load, config)?;
        let mut sites = vec!["quant", "attention", "sample", "delta-prefill"];
        if pool.is_some() {
            sites.extend(["paged-attention.prefill", "paged-attention.decode"]);
        }

        let picks = if config.cpu {
            None
        } else {
            Some(EngineSelections::new(load, &[], &sites, context)?)
        };
        let paged = pool
            .map(|pool| paged::PagedModel::new(pool, config, context, picks.as_ref()))
            .transpose()?;
        let weights = weights.scoped("language_model");
        let model = weights.scoped("model");
        Ok(Self {
            config,
            context,
            position: 0,
            usable: true,
            embedding: Affine::load(&model.scoped("embed_tokens"), config.hidden, config.vocab)?,
            output: Affine::load(&weights.scoped("lm_head"), config.hidden, config.vocab)?,
            norm: model.tensor("norm.weight", &[config.hidden])?,
            layers: (0..config.layers)
                .map(|index| {
                    let weights = model.scoped(format!("layers.{index}"));
                    if let Some(paged) = &paged {
                        DecoderLayer::load_paged(
                            &weights,
                            config,
                            index,
                            paged.set.clone(),
                            &paged.positions,
                        )
                    } else {
                        DecoderLayer::load(&weights, config, index, context)
                    }
                })
                .collect::<Result<_>>()?,
            quant: picks
                .as_ref()
                .map(|picks| picks.at("quant", 0))
                .transpose()?,
            attention: picks
                .as_ref()
                .map(|picks| picks.at("attention", 0))
                .transpose()?,
            delta: picks
                .as_ref()
                .map(|picks| picks.at("delta-prefill", 0))
                .transpose()?,
            paged,
            sample: picks
                .as_ref()
                .map(|picks| picks.at("sample", 0))
                .transpose()?,
        })
    }
    /// Processes canonical prompt blocks, projecting only the final row.
    ///
    /// # Errors
    /// Refuses gaps, empty inputs, context overflow or use after failed execution.
    pub fn forward(&mut self, tokens: &Tensor<u32>, start: u32, taps: bool) -> Result<StepOutput> {
        self.forward_phase(tokens, start, taps, true)
    }
    /// Advances one generated token with the recurrent decode reduction class.
    ///
    /// # Errors
    /// Refuses gaps, invalid spans or use after failed execution.
    pub fn decode_token(
        &mut self,
        tokens: &Tensor<u32>,
        start: u32,
        taps: bool,
    ) -> Result<StepOutput> {
        self.forward_phase(tokens, start, taps, false)
    }
    fn forward_phase(
        &mut self,
        tokens: &Tensor<u32>,
        start: u32,
        taps: bool,
        prefill: bool,
    ) -> Result<StepOutput> {
        let [sequence]: [u32; 1] = tokens
            .shape()
            .try_into()
            .map_err(|_| Error::loading("tokens must have rank one"))?;
        let end = start
            .checked_add(sequence)
            .filter(|&end| sequence > 0 && end <= self.context)
            .ok_or_else(|| Error::loading("invalid token span"))?;
        if (prefill && !start.is_multiple_of(crate::PREFILL_BLOCK_TOKENS))
            || (!prefill && sequence != 1)
        {
            return Err(Error::loading("token span splits a canonical prompt block"));
        }
        if !self.usable || (start != 0 && start != self.position) {
            return Err(Error::loading(
                "sequence requires reload or contiguous positions",
            ));
        }
        self.usable = false;
        if start == 0 && self.position != 0 {
            for layer in &mut self.layers {
                layer.reset()?;
            }
        }
        let output = self.forward_tokens(tokens, start, sequence, taps, prefill)?;
        forja_sdk::eval()?;
        self.position = end;
        self.usable = true;
        Ok(output)
    }
    fn forward_tokens(
        &mut self,
        tokens: &Tensor<u32>,
        start: u32,
        sequence: u32,
        taps: bool,
        prefill: bool,
    ) -> Result<StepOutput> {
        let layers = self.layers.len();
        let captured = if taps {
            (0..self.config.layers)
                .map(|_| Tensor::<bf16>::zeros(&[sequence, self.config.hidden]))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let mut last = None;
        let block = if prefill {
            crate::PREFILL_BLOCK_TOKENS
        } else {
            1
        };
        for row in (0..sequence)
            .step_by(usize::try_from(block).map_err(|_| Error::loading("block width overflow"))?)
        {
            let rows = block.min(sequence - row);
            forja_sdk::profile_scope("embedding", "global", None)?;
            let mut hidden = self.embedding.embed(&tokens.narrow(0, row, rows)?)?;
            for (index, layer) in self.layers.iter_mut().enumerate() {
                hidden = if prefill {
                    layer.prefill(
                        &hidden,
                        start + row,
                        self.quant.as_ref(),
                        self.attention.as_ref(),
                        self.delta.as_ref(),
                    )?
                } else {
                    layer.forward(
                        &hidden,
                        start + row,
                        self.quant.as_ref(),
                        self.attention.as_ref(),
                    )?
                };
                if taps && index + 1 < layers {
                    hidden.copy_into(&mut captured[index].narrow(0, row, rows)?)?;
                }
            }
            forja_sdk::profile_scope("output", "global", None)?;
            hidden = hidden.rms_norm(&self.norm, 1.0e-6)?;
            if let Some(tap) = captured.last() {
                hidden.copy_into(&mut tap.narrow(0, row, rows)?)?;
            }
            last = Some(hidden.narrow(0, rows - 1, 1)?);
        }
        let hidden = last.ok_or_else(|| Error::loading("empty prompt"))?;
        Ok(StepOutput {
            logits: self
                .output
                .forward(&hidden, self.quant.as_ref())?
                .to_dtype::<f32>()?
                .reshape(&[self.config.vocab])?,
            taps: captured
                .iter()
                .map(Tensor::to_dtype::<f32>)
                .collect::<Result<_>>()?,
            router_logits: Vec::new(),
        })
    }
}

fn load_config(load: &EngineLoadConfig, mut config: Config) -> Result<(Config, u32)> {
    if let Some(layers) = load.num_hidden_layers {
        if layers == 0 || layers > config.layers {
            return Err(Error::loading("invalid decoder layer limit"));
        }
        config.layers = layers;
    }
    config.validate()?;
    if !load.variant_rule_picks.is_empty() {
        return Err(Error::loading(
            "the eager model requires fixed variant picks",
        ));
    }
    let context = load
        .max_context
        .filter(|n| (1..=262_144).contains(n))
        .ok_or_else(|| Error::loading("unsupported profile context"))?;
    if load.kv_cache != forja_sdk::KvCacheFormat::F16 {
        return Err(Error::loading(
            "the reference engine requires f16 KV storage",
        ));
    }
    Ok((config, context))
}
