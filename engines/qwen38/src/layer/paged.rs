use super::{DecoderLayer, Mixer};
use crate::{Attention, Config, DeltaNet};
use forja_sdk::{Dim, Error, PagedCacheSet, Result, Tensor, Weights, bf16, target::metal::Variant};
use std::rc::Rc;

impl DecoderLayer {
    /// Loads a decoder with its attention pair or protected recurrent plane pair.
    ///
    /// # Errors
    /// Refuses invalid indices, state descriptors or weights.
    pub fn load_paged(
        weights: &Weights<'_>,
        config: Config,
        index: u32,
        set: Rc<PagedCacheSet>,
        positions: &Tensor<u32>,
    ) -> Result<Self> {
        config.validate()?;
        if index >= config.layers {
            return Err(Error::loading("decoder layer index exceeds the model"));
        }
        let preceding_attention = index / config.attention_interval;
        let mixer = if config.full_attention(index) {
            Mixer::Attention(Box::new(Attention::load_paged(
                &weights.scoped("self_attn"),
                config,
                set,
                preceding_attention,
                positions.narrow(0, 0, positions.shape()[0])?,
            )?))
        } else {
            let plane = index
                .checked_sub(preceding_attention)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| Error::loading("recurrent plane index overflow"))?;
            Mixer::Recurrent(Box::new(DeltaNet::load_paged(
                &weights.scoped("linear_attn"),
                config,
                set,
                plane,
            )?))
        };
        Self::load_with_mixer(weights, config, index, mixer)
    }

    /// Records one serving lane's complete admitted piece through both residuals.
    ///
    /// # Errors
    /// Refuses invalid spans, cache bindings, selections or tensor operations.
    pub fn paged_layer(
        &mut self,
        input: &Tensor<bf16>,
        lane: u32,
        start: &Dim,
        phase: forja_sdk::BatchPhase,
        picks: [Option<&Variant>; 3],
    ) -> Result<Tensor<bf16>> {
        let [quant, attention, delta] = picks;
        let prefill = phase == forja_sdk::BatchPhase::Prompt;
        let normalized = self.normalize(input)?;
        let mixed = match &mut self.mixer {
            Mixer::Recurrent(layer) => {
                layer.select_lane(lane)?;
                if prefill {
                    layer.prefill(&normalized, quant, delta)?
                } else {
                    layer.deltanet_layer(&normalized, quant)?
                }
            }
            Mixer::Attention(layer) => {
                layer.select_lane(lane)?;
                layer.paged_layer(
                    &normalized,
                    start,
                    prefill,
                    quant,
                    attention.ok_or_else(|| Error::loading("paged attention pick is missing"))?,
                )?
            }
        };
        self.finish(input, &mixed, quant)
    }
}
