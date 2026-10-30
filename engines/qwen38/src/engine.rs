use crate::{Config, Model};
use forja_sdk::{
    DecodeInput, DecodeOutput, Engine, EngineInfo, EngineLoadConfig, Error, Result, StepInput,
    StepOutput, Weights, export_engine, nn::blocks::DecodeState,
};
/// Single-lane text engine with eager feedback and no prefix or graph reuse.
pub struct Qwen38 {
    model: Model,
    decode: DecodeState,
    feedback_position: Option<u32>,
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
            decode: DecodeState::new(context, config.vocab)?,
            feedback_position: None,
        })
    }
}
#[export_engine]
impl Engine for Qwen38 {
    fn describe() -> EngineInfo {
        let c = Config::model();
        EngineInfo {
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
        self.feedback_position = None;
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
        self.decode.select(input.sampling)?;
        let prefill = input.phase == forja_sdk::BatchPhase::Prompt;
        let tokens = match input.tokens {
            Some(tokens) => tokens,
            None if prefill => return Err(Error::loading("prompt phase requires explicit tokens")),
            None if self.feedback_position == Some(input.start_pos) => self.decode.token()?,
            None => return Err(Error::loading("decode has no token at this position")),
        };
        let [sequence]: [u32; 1] = tokens
            .shape()
            .try_into()
            .map_err(|_| Error::loading("tokens must have rank one"))?;
        let next = input
            .start_pos
            .checked_add(sequence)
            .filter(|_| sequence > 0)
            .ok_or_else(|| Error::loading("invalid decode span"))?;
        let logits = if prefill {
            self.model.forward(&tokens, input.start_pos, false)?
        } else {
            self.model.decode_token(&tokens, input.start_pos, false)?
        }
        .logits;
        let row = logits.reshape(&[1, logits.shape()[0]])?;
        let params = self.decode.sampling()?;
        let selected = match self.model.sample.as_ref() {
            Some(pick) => row.sample_with(&params, next - 1, pick)?,
            None => row.sample(&params, next - 1)?,
        };
        let token = self.decode.store_selected(&selected, next - 1)?;
        forja_sdk::eval()?;
        self.feedback_position = Some(next);
        Ok(DecodeOutput { logits, token })
    }
}
