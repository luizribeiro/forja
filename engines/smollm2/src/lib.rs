//! Configuration for the `SmolLM2` Llama engine.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

mod optimized;
pub mod reference;
pub mod variants;

use forja_sdk::{
    Load,
    nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig},
};

/// Dimensions shared by the reference and optimized implementations.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    vocab: u32,
    hidden: u32,
    intermediate: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    layers: usize,
    context: u32,
    epsilon: f32,
    theta: f32,
}

impl Config {
    /// Pinned Hugging Face model architecture.
    #[must_use]
    pub const fn smollm2() -> Self {
        Self {
            vocab: 49_152,
            hidden: 576,
            intermediate: 1_536,
            query_heads: 9,
            kv_heads: 3,
            head_dim: 64,
            layers: 30,
            context: 8_192,
            epsilon: 1.0e-5,
            theta: 100_000.0,
        }
    }

    /// Small instance of the same GQA, rotary, tied-embedding architecture.
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            vocab: 33,
            hidden: 24,
            intermediate: 40,
            query_heads: 3,
            kv_heads: 1,
            head_dim: 8,
            layers: 2,
            context: 64,
            epsilon: 1.0e-5,
            theta: 100_000.0,
        }
    }

    /// Inverse square root of the attention head width.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn attention_scale(self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// Checks architecture arithmetic before loading or allocating tensors.
    ///
    /// # Errors
    /// Returns an error for incompatible or overflowing model dimensions.
    pub fn validate(self) -> forja_sdk::Result<()> {
        let heads = self.query_heads.checked_mul(self.head_dim);
        let kv = self.kv_heads.checked_mul(self.head_dim);
        if self.vocab == 0
            || self.hidden == 0
            || self.intermediate == 0
            || self.layers == 0
            || self.context == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || heads != Some(self.hidden)
            || kv.is_none()
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.theta.is_finite()
            || self.theta <= 0.0
        {
            return Err(forja_sdk::Error::loading("invalid Llama dimensions"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn real_and_tiny_configs_use_grouped_rotary_attention() {
        for config in [Config::smollm2(), Config::tiny()] {
            config.validate().unwrap();
            assert_eq!(config.query_heads / config.kv_heads, 3);
            assert_eq!(config.hidden, config.query_heads * config.head_dim);
        }
        let mut invalid = Config::tiny();
        invalid.kv_heads = 2;
        assert!(invalid.validate().is_err());
        invalid.kv_heads = 0;
        assert!(invalid.validate().is_err());
    }
}

impl Config {
    fn linear(input: u32, output: u32) -> [u32; 2] {
        [output, input]
    }
    fn norm(self) -> RmsNormConfig {
        RmsNormConfig::promoted_bf16(self.hidden, self.epsilon)
    }
}

#[derive(Load)]
#[load(config = Config)]
#[allow(clippy::struct_field_names)]
struct Attention {
    #[load(prefix, config = Config::linear(config.hidden, config.query_heads * config.head_dim))]
    q_proj: Projection,
    #[load(prefix, config = Config::linear(config.hidden, config.kv_heads * config.head_dim))]
    k_proj: Projection,
    #[load(prefix, config = Config::linear(config.hidden, config.kv_heads * config.head_dim))]
    v_proj: Projection,
    #[load(prefix, config = Config::linear(config.query_heads * config.head_dim, config.hidden))]
    o_proj: Projection,
}

#[derive(Load)]
#[load(config = Config)]
#[allow(clippy::struct_field_names)]
struct Mlp {
    #[load(prefix, config = Config::linear(config.hidden, config.intermediate))]
    gate_proj: Projection,
    #[load(prefix, config = Config::linear(config.hidden, config.intermediate))]
    up_proj: Projection,
    #[load(prefix, config = Config::linear(config.intermediate, config.hidden))]
    down_proj: Projection,
}

#[derive(Load)]
#[load(config = Config)]
struct DecoderLayer {
    #[load(prefix, config = config.norm())]
    input_layernorm: RmsNorm<f32>,
    #[load(prefix)]
    self_attn: Attention,
    #[load(prefix, config = config.norm())]
    post_attention_layernorm: RmsNorm<f32>,
    #[load(prefix)]
    mlp: Mlp,
}

#[derive(Load)]
#[load(config = Config)]
struct Model {
    #[load(prefix, config = EmbeddingConfig::promoted_bf16(config.vocab, config.hidden))]
    embed_tokens: Embedding<f32>,
    #[load(prefix, count = config.layers)]
    layers: Vec<DecoderLayer>,
    #[load(prefix, config = config.norm())]
    norm: RmsNorm<f32>,
}

#[derive(Load)]
#[load(config = Config)]
struct ModelWeights {
    #[load(prefix)]
    model: Model,
}

/// `SmolLM2` with a profile-selected execution path.
pub struct SmolLM2 {
    path: Execution,
    context: u32,
    #[cfg(target_family = "wasm")]
    decode: forja_sdk::nn::blocks::DecodeState,
}

enum Execution {
    Reference(reference::Reference),
    Optimized(optimized::Optimized),
}

impl SmolLM2 {
    /// Loads a real or tiny architecture from the same named bf16 tensors.
    ///
    /// # Errors
    /// Returns an error for invalid load settings, weights, or allocations.
    pub fn with_config(
        weights: &forja_sdk::Weights<'_>,
        load: &forja_sdk::EngineLoadConfig,
        mut config: Config,
    ) -> forja_sdk::Result<Self> {
        config.validate()?;
        let context = load
            .max_context
            .filter(|&n| n > 0 && n <= config.context)
            .ok_or_else(|| forja_sdk::Error::loading("unsupported profile context"))?;
        if let Some(layers) = load.num_hidden_layers {
            config.layers = usize::try_from(layers)
                .ok()
                .filter(|&n| n > 0 && n <= config.layers)
                .ok_or_else(|| forja_sdk::Error::loading("unsupported layer count"))?;
        }
        forja_sdk::EngineTunings::new(load, variants::TUNINGS)?;
        let path = if load.tunings.iter().any(|name| name == "optimized") {
            Execution::Optimized(optimized::Optimized::load(weights, config, context, load)?)
        } else {
            Execution::Reference(reference::Reference::load(weights, config, context)?)
        };
        Ok(Self {
            path,
            context,
            #[cfg(target_family = "wasm")]
            decode: forja_sdk::nn::blocks::DecodeState::new(context, config.vocab)?,
        })
    }
}

#[forja_sdk::export_engine]
impl forja_sdk::Engine for SmolLM2 {
    fn describe() -> forja_sdk::EngineInfo {
        forja_sdk::EngineInfo {
            vocab: 49_152,
            max_context: 8_192,
            tap_layers: (1..=30).collect(),
            router_layers: Vec::new(),
            serving: None,
        }
    }

    fn load(
        weights: &forja_sdk::Weights<'_>,
        config: forja_sdk::EngineLoadConfig,
    ) -> forja_sdk::Result<Self> {
        Self::with_config(weights, &config, Config::smollm2())
    }

    fn step(&mut self, input: forja_sdk::StepInput) -> forja_sdk::Result<forja_sdk::StepOutput> {
        reference::input_length(input.tokens.shape(), input.start_pos, self.context)?;
        match &mut self.path {
            Execution::Reference(engine) => {
                engine.forward(&input.tokens, input.start_pos, input.taps)
            }
            Execution::Optimized(engine) => {
                let sequence = input.tokens.shape()[0];
                engine.forward(
                    &input.tokens,
                    sequence,
                    &input.start_pos.into(),
                    &(input.start_pos + sequence).into(),
                    &(sequence - 1).into(),
                    input.start_pos,
                    None,
                    input.taps,
                )
            }
        }
    }

    #[cfg(target_family = "wasm")]
    fn decode(
        &mut self,
        input: forja_sdk::DecodeInput,
    ) -> forja_sdk::Result<forja_sdk::DecodeOutput> {
        self.decode.select(input.sampling)?;
        let tokens = match input.tokens {
            Some(tokens) => tokens,
            None => self.decode.token()?,
        };
        let sequence = reference::input_length(tokens.shape(), input.start_pos, self.context)?;
        let slot = input
            .start_pos
            .checked_add(sequence)
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| forja_sdk::Error::loading("decode position overflow"))?;
        let logits = self
            .step(forja_sdk::StepInput {
                tokens,
                start_pos: input.start_pos,
                taps: false,
            })?
            .logits;
        let selected = logits.sample(&self.decode.sampling()?, slot)?;
        let token = self.decode.store_selected(&selected, slot)?;
        Ok(forja_sdk::DecodeOutput { logits, token })
    }
}

struct Projection {
    weight: forja_sdk::Tensor<f32>,
}

impl Load<[u32; 2]> for Projection {
    fn load(weights: &forja_sdk::Weights<'_>, shape: &[u32; 2]) -> forja_sdk::Result<Self> {
        Ok(Self {
            weight: weights
                .tensor::<forja_sdk::bf16>("weight", shape)?
                .to_dtype()?,
        })
    }
}

impl Projection {
    fn forward_with(
        &self,
        input: &forja_sdk::Tensor<f32>,
        selection: &forja_sdk::target::metal::Variant,
    ) -> forja_sdk::Result<forja_sdk::Tensor<f32>> {
        input.matmul_with(&self.weight.t()?, selection)
    }
}
