use super::StateInputBank;
use crate::{
    Backend, BatchPhase, EngineBatch, Host, SamplingParams, compute, invalid_handle,
    state::StateCharge,
};
use std::sync::Arc;
use wasmtime::component::Resource;

fn sampling_words(params: SamplingParams) -> [u32; 5] {
    let seed = params.seed.to_le_bytes();
    [
        params.temperature.to_bits(),
        params.top_k,
        params.top_p.to_bits(),
        u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]),
        u32::from_le_bytes([seed[4], seed[5], seed[6], seed[7]]),
    ]
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn write_state_batch_inputs(
        &self,
        pool: u32,
        input: &EngineBatch,
    ) -> Result<(), compute::Error> {
        let pool = self
            .table
            .get(&Resource::<crate::PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?;
        let Some(slot) = pool.slot else {
            return Ok(());
        };
        let slot = self
            .table
            .get(&Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        let Some(inputs) = &slot.inputs else {
            return Ok(());
        };
        let layout = slot
            .input_layout
            .ok_or_else(|| invalid_handle("state input layout is missing"))?;
        let bank = &inputs.banks[slot.active_bank];
        let lanes = u32::try_from(slot.binding.lanes.len()).map_err(invalid_handle)?;
        let mask_words = layout.vocab.div_ceil(32);
        let mask_count = u64::from(lanes)
            .checked_mul(u64::from(mask_words))
            .ok_or_else(|| invalid_handle("input mask size overflow"))?;
        let words = u64::from(lanes)
            .checked_mul(7)
            .and_then(|n| n.checked_add(mask_count))
            .and_then(|n| n.checked_add(u64::from(layout.prompt_tokens)))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid_handle("input staging bytes overflow"))?;
        let _charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            words,
        )?;
        let lanes = usize::try_from(lanes).map_err(invalid_handle)?;
        let mut sampling = vec![sampling_words(SamplingParams::default()); lanes];
        let mut tokens = vec![0; lanes];
        let mut explicit = vec![0; lanes];
        let mut prompt = vec![0; usize::try_from(layout.prompt_tokens).map_err(invalid_handle)?];
        let mut allow = vec![0; usize::try_from(mask_count).map_err(invalid_handle)?];
        for row in allow.chunks_exact_mut(usize::try_from(mask_words).map_err(invalid_handle)?) {
            row[0] = 1;
        }
        let mut prompt_lanes = 0;
        for lane in input.lanes.iter().flatten() {
            let index = usize::try_from(lane.slot).map_err(invalid_handle)?;
            *sampling
                .get_mut(index)
                .ok_or_else(|| invalid_handle("sampling lane exceeds slot"))? =
                sampling_words(lane.sampling);
            if let Some(values) = &lane.tokens {
                *explicit
                    .get_mut(index)
                    .ok_or_else(|| invalid_handle("token lane exceeds slot"))? = 1;
                *tokens
                    .get_mut(index)
                    .ok_or_else(|| invalid_handle("token lane exceeds slot"))? = *values
                    .first()
                    .ok_or_else(|| invalid_handle("explicit tokens are empty"))?;
                if input.shape.phase == BatchPhase::Prompt {
                    prompt_lanes += 1;
                    prompt
                        .get_mut(..values.len())
                        .ok_or_else(|| invalid_handle("prompt exceeds slot input budget"))?
                        .copy_from_slice(values);
                }
            } else if input.shape.phase == BatchPhase::Prompt {
                return Err(invalid_handle("prompt input tokens are missing"));
            }
        }
        if input.shape.phase == BatchPhase::Prompt && prompt_lanes != 1 {
            return Err(invalid_handle("prompt requires exactly one input lane"));
        }
        self.write_state_input_words(
            bank,
            sampling.into_iter().flatten(),
            tokens,
            explicit,
            prompt,
            allow,
        )
    }

    fn write_state_input_words(
        &self,
        bank: &StateInputBank,
        sampling: impl Iterator<Item = u32>,
        tokens: Vec<u32>,
        explicit: Vec<u32>,
        prompt: Vec<u32>,
        allow: Vec<u32>,
    ) -> Result<(), compute::Error> {
        self.write_input_words(bank.sampling, sampling)?;
        for (handle, words) in [
            (bank.tokens, tokens),
            (bank.explicit, explicit),
            (bank.prompt, prompt),
            (bank.allow, allow),
        ] {
            self.write_input_words(handle, words.into_iter())?;
        }
        Ok(())
    }

    fn write_input_words(
        &self,
        handle: u32,
        words: impl Iterator<Item = u32>,
    ) -> Result<(), compute::Error> {
        let bytes = words.flat_map(u32::to_ne_bytes).collect::<Vec<_>>();
        self.write(&Resource::new_borrow(handle), &bytes)
    }
}
