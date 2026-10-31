#[cfg(target_family = "wasm")]
mod batch;
mod paged;
use crate::{Attention, Config, DeltaNet, FeedForward};
use forja_sdk::{Error, Result, Tensor, Weights, bf16, target::metal::Variant};
enum Mixer {
    Recurrent(Box<DeltaNet>),
    Attention(Box<Attention>),
}
/// A pre-normalized hybrid decoder layer with bf16 residual boundaries.
pub struct DecoderLayer {
    index: u32,
    mixer: Mixer,
    ffn: FeedForward,
    input_norm: Tensor<bf16>,
    post_norm: Tensor<bf16>,
}
impl DecoderLayer {
    /// Loads the mixer selected by the model's fixed layer pattern.
    ///
    /// # Errors
    /// Returns invalid indices, metadata or state/cache allocation errors.
    pub fn load(weights: &Weights<'_>, config: Config, index: u32, context: u32) -> Result<Self> {
        config.validate()?;
        if index >= config.layers {
            return Err(Error::loading("decoder layer index exceeds the model"));
        }
        let mixer = if config.full_attention(index) {
            Mixer::Attention(Box::new(Attention::load(
                &weights.scoped("self_attn"),
                config,
                context,
            )?))
        } else {
            Mixer::Recurrent(Box::new(DeltaNet::load(
                &weights.scoped("linear_attn"),
                config,
            )?))
        };
        Self::load_with_mixer(weights, config, index, mixer)
    }

    fn load_with_mixer(
        weights: &Weights<'_>,
        config: Config,
        index: u32,
        mixer: Mixer,
    ) -> Result<Self> {
        Ok(Self {
            index,
            mixer,
            ffn: FeedForward::load(&weights.scoped("mlp"), config)?,
            input_norm: weights.tensor("input_layernorm.weight", &[config.hidden])?,
            post_norm: weights.tensor("post_attention_layernorm.weight", &[config.hidden])?,
        })
    }
    /// Processes one token through the mixer and dense feed-forward residuals.
    ///
    /// # Errors
    /// Refuses invalid input geometry, positions or failed tensor operations.
    pub fn forward(
        &mut self,
        input: &Tensor<bf16>,
        position: u32,
        quant: Option<&Variant>,
        attention: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        self.process(input, position, quant, attention, None, false)
    }
    /// Applies one canonical prompt block through both residual branches.
    ///
    /// # Errors
    /// Returns invalid block geometry, positions, picks or execution failures.
    pub fn prefill(
        &mut self,
        input: &Tensor<bf16>,
        position: u32,
        quant: Option<&Variant>,
        attention: Option<&Variant>,
        delta: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        self.process(input, position, quant, attention, delta, true)
    }
    fn process(
        &mut self,
        input: &Tensor<bf16>,
        position: u32,
        quant: Option<&Variant>,
        attention: Option<&Variant>,
        delta: Option<&Variant>,
        prefill: bool,
    ) -> Result<Tensor<bf16>> {
        let normalized = self.normalize(input)?;
        let mixed = match &mut self.mixer {
            Mixer::Recurrent(layer) if prefill => layer.prefill(&normalized, quant, delta)?,
            Mixer::Recurrent(layer) => layer.deltanet_layer(&normalized, quant)?,
            Mixer::Attention(layer) if prefill => {
                layer.prefill(&normalized, position, quant, attention)?
            }
            Mixer::Attention(layer) => {
                layer.attention_layer(&normalized, position, quant, attention)?
            }
        };
        self.finish(input, &mixed, quant)
    }

    fn normalize(&self, input: &Tensor<bf16>) -> Result<Tensor<bf16>> {
        let layer_type = match &self.mixer {
            Mixer::Recurrent(_) => "recurrent",
            Mixer::Attention(_) => "attention",
        };
        forja_sdk::profile_scope("mixer", layer_type, Some(self.index))?;
        input.rms_norm(&self.input_norm, 1.0e-6)
    }

    fn finish(
        &self,
        input: &Tensor<bf16>,
        mixed: &Tensor<bf16>,
        quant: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let residual = (input + mixed)?;
        forja_sdk::profile_scope("ffn", "ffn", Some(self.index))?;
        let normalized = residual.rms_norm(&self.post_norm, 1.0e-6)?;
        &residual + &self.ffn.ffn(&normalized, quant)?
    }
    /// Clears recurrent carry before a new sequence. Attention writes replace its live prefix.
    ///
    /// # Errors
    /// Returns private-state quota or allocation errors.
    pub fn reset(&mut self) -> Result<()> {
        if let Mixer::Recurrent(layer) = &mut self.mixer {
            layer.reset()?;
        }
        Ok(())
    }
}
