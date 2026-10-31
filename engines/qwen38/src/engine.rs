mod eager;
use crate::{Config, Model};
use eager::EagerFeedback;
use forja_sdk::{
    DecodeInput, DecodeOutput, Engine, EngineInfo, EngineLoadConfig, Error, Result, StepInput,
    StepOutput, Weights,
};
/// Single-lane text engine with eager feedback and no prefix or graph reuse.
pub struct Qwen38 {
    model: Model,
    feedback: EagerFeedback,
}
impl Qwen38 {
    /// Loads an explicit architecture for trusted native tests.
    ///
    /// # Errors
    /// Returns configuration, weight, state or selection allocation errors.
    pub fn with_config(
        weights: &Weights<'_>,
        load: &EngineLoadConfig,
        config: Config,
    ) -> Result<Self> {
        let model = Model::load(weights, load, config)?;
        let context = load
            .max_context
            .ok_or_else(|| Error::loading("profile context is missing"))?;
        Ok(Self {
            model,
            feedback: EagerFeedback::new(context, config.vocab)?,
        })
    }
}
#[cfg_attr(feature = "component", forja_sdk::export_engine)]
impl Engine for Qwen38 {
    fn describe() -> EngineInfo {
        let c = Config::model();
        EngineInfo {
            prompt_block_tokens: std::num::NonZeroU32::new(crate::PREFILL_BLOCK_TOKENS),
            vocab: c.vocab,
            max_context: 262_144,
            tap_layers: (1..=c.layers).collect(),
            router_layers: Vec::new(),
            serving: None,
        }
    }
    fn load(weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        Self::with_config(weights, &config, Config::model())
    }
    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        self.feedback.invalidate();
        match input.phase {
            forja_sdk::BatchPhase::Prompt => {
                self.model
                    .forward(&input.tokens, input.start_pos, input.taps)
            }
            forja_sdk::BatchPhase::Decode => {
                self.model
                    .decode_token(&input.tokens, input.start_pos, input.taps)
            }
        }
    }
    fn decode(&mut self, input: DecodeInput) -> Result<DecodeOutput> {
        self.feedback.decode(&mut self.model, input)
    }
}
