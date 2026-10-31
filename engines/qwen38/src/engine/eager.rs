use crate::Model;
use forja_sdk::{DecodeInput, DecodeOutput, Error, Result, nn::blocks::DecodeState};

pub(super) struct EagerFeedback {
    state: DecodeState,
    position: Option<u32>,
}
impl EagerFeedback {
    pub(super) fn new(context: u32, vocab: u32) -> Result<Self> {
        Ok(Self {
            state: DecodeState::new(context, vocab)?,
            position: None,
        })
    }
    pub(super) fn invalidate(&mut self) {
        self.position = None;
    }
    pub(super) fn decode(&mut self, model: &mut Model, input: DecodeInput) -> Result<DecodeOutput> {
        self.state.select(input.sampling)?;
        let prefill = input.phase == forja_sdk::BatchPhase::Prompt;
        let tokens = match input.tokens {
            Some(tokens) => tokens,
            None if prefill => return Err(Error::loading("prompt phase requires explicit tokens")),
            None if self.position == Some(input.start_pos) => self.state.token()?,
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
            model.forward(&tokens, input.start_pos, false)?
        } else {
            model.decode_token(&tokens, input.start_pos, false)?
        }
        .logits;
        let row = logits.reshape(&[1, logits.shape()[0]])?;
        let params = self.state.sampling()?;
        let selected = match model.sample.as_ref() {
            Some(pick) => row.sample_with(&params, next - 1, pick)?,
            None => row.sample(&params, next - 1)?,
        };
        let token = self.state.store_selected(&selected, next - 1)?;
        forja_sdk::eval()?;
        self.position = Some(next);
        Ok(DecodeOutput { logits, token })
    }
}
