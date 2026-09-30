//! A small two-layer MLP engine used for end-to-end backend checks.

use forja_sdk::{
    Engine, EngineInfo, Load, Result, StepInput, StepOutput, Weights, export_engine,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
};

/// Vocabulary size used by the toy model.
pub const VOCAB: u32 = 33;
/// Hidden width used by the toy model.
pub const HIDDEN: u32 = 7;
/// MLP width used by the toy model.
pub const INTERMEDIATE: u32 = 11;
/// Number of residual MLP blocks.
pub const LAYERS: usize = 2;
/// Maximum input position accepted by the toy model.
pub const MAX_CONTEXT: u32 = 33;

#[derive(Clone, Copy)]
struct Config;

#[derive(Load)]
#[load(config = Config)]
struct Block {
    #[load(prefix, config = RmsNormConfig::new(HIDDEN, 1.0e-6))]
    norm: RmsNorm<f32>,
    #[load(prefix, config = LinearConfig::new(HIDDEN, INTERMEDIATE * 2))]
    up: Linear<f32>,
    #[load(prefix, config = LinearConfig::new(INTERMEDIATE, HIDDEN))]
    down: Linear<f32>,
}

impl Block {
    fn forward(&self, input: &forja_sdk::Tensor<f32>) -> Result<forja_sdk::Tensor<f32>> {
        let fused = self.up.forward(&self.norm.forward(input)?)?;
        let gate = fused.narrow(1, 0, INTERMEDIATE)?;
        let up = fused.narrow(1, INTERMEDIATE, INTERMEDIATE)?;
        let projected = self.down.forward(&gate.silu_mul(&up)?)?;
        input + &projected
    }
}

/// Two-layer residual MLP with hidden-state taps.
#[derive(Load)]
#[load(config = Config)]
pub struct ToyEngine {
    #[load(prefix, config = EmbeddingConfig::new(VOCAB, HIDDEN))]
    embed: Embedding<f32>,
    #[load(prefix, count = LAYERS)]
    blocks: Vec<Block>,
    #[load(prefix, config = RmsNormConfig::new(HIDDEN, 1.0e-6))]
    norm: RmsNorm<f32>,
    #[load(prefix, config = LinearConfig::new(HIDDEN, VOCAB))]
    lm_head: Linear<f32>,
}

#[export_engine]
impl Engine for ToyEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: vec![0, 1],
            router_layers: Vec::new(),
        }
    }

    fn load(weights: &Weights<'_>) -> Result<Self> {
        <Self as Load<Config>>::load(weights, &Config)
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        let sequence = input
            .tokens
            .shape()
            .first()
            .copied()
            .and_then(|length| length.checked_sub(1))
            .ok_or_else(|| {
                forja_sdk::Error::loading("toy engine requires a nonempty token list")
            })?;
        let mut hidden = self.embed.forward(&input.tokens)?;
        let mut taps = Vec::with_capacity(if input.taps { LAYERS } else { 0 });
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
            if input.taps {
                taps.push(hidden.contiguous()?);
            }
        }
        let logits = self
            .lm_head
            .forward(&self.norm.forward(&hidden)?)?
            .narrow(0, sequence, 1)?
            .reshape(&[VOCAB])?;
        Ok(StepOutput {
            logits,
            taps,
            router_logits: Vec::new(),
        })
    }
}
