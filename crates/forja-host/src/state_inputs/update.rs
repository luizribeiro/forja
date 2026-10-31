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

fn allow_vocabulary(allow: &mut [u32], lane: usize, vocab: u32) -> Result<(), compute::Error> {
    let row = allow
        .chunks_exact_mut(usize::try_from(vocab.div_ceil(32)).map_err(invalid_handle)?)
        .nth(lane)
        .ok_or_else(|| invalid_handle("mask lane exceeds slot"))?;
    row.fill(u32::MAX);
    if !vocab.is_multiple_of(32) {
        *row.last_mut()
            .ok_or_else(|| invalid_handle("vocabulary mask is empty"))? =
            (1_u32 << (vocab % 32)) - 1;
    }
    Ok(())
}

struct InputLane<'a> {
    slot: u32,
    tokens: Option<&'a [u32]>,
    sampling: SamplingParams,
}

enum AllowedTokens<'a> {
    Ids(&'a [u32]),
    Mask(&'a [u32]),
}
impl AllowedTokens<'_> {
    fn write(self, row: &mut [u32], vocab: u32) -> Result<(), compute::Error> {
        match self {
            Self::Ids(allow) => {
                if allow.is_empty()
                    || allow.windows(2).any(|pair| pair[0] >= pair[1])
                    || allow.iter().any(|token| *token >= vocab)
                {
                    return Err(invalid_handle("selection allow list is invalid"));
                }
                row.fill(0);
                for &token in allow {
                    row[usize::try_from(token / 32).map_err(invalid_handle)?] |= 1 << (token % 32);
                }
            }
            Self::Mask(mask) => {
                if mask.len() != row.len()
                    || mask.iter().all(|word| *word == 0)
                    || (!vocab.is_multiple_of(32)
                        && mask.last().is_some_and(|word| *word >> (vocab % 32) != 0))
                {
                    return Err(invalid_handle("selection mask has invalid words"));
                }
                row.copy_from_slice(mask);
            }
        }
        Ok(())
    }
}

struct SelectionLane<'a> {
    slot: usize,
    start: u32,
    sampling: SamplingParams,
    allow: AllowedTokens<'a>,
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn write_state_selection_inputs(
        &self,
        pool: u32,
        input: &crate::EngineBatchSelect,
    ) -> Result<(), compute::Error> {
        self.write_selection_inputs(
            pool,
            input.lanes.iter().enumerate().filter_map(|(slot, lane)| {
                lane.as_ref().map(|lane| SelectionLane {
                    slot,
                    start: lane.start_pos,
                    sampling: lane.sampling,
                    allow: AllowedTokens::Ids(&lane.allow),
                })
            }),
        )
    }

    pub(crate) fn write_state_single_selection_inputs(
        &self,
        pool: u32,
        start: u32,
        sampling: SamplingParams,
        allow: &[u32],
    ) -> Result<(), compute::Error> {
        self.write_selection_inputs(
            pool,
            std::iter::once(SelectionLane {
                slot: 0,
                start,
                sampling,
                allow: AllowedTokens::Mask(allow),
            }),
        )
    }

    pub(super) fn input_slot(
        &self,
        pool: u32,
    ) -> Result<Option<(u32, &crate::BatchSlotEntry, u32)>, compute::Error> {
        let pool = self
            .table
            .get(&Resource::<crate::PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?;
        let Some(rep) = pool.slot else {
            return Ok(None);
        };
        let slot = self
            .table
            .get(&Resource::<crate::BatchSlotEntry>::new_borrow(rep))
            .map_err(invalid_handle)?;
        Ok(Some((rep, slot, pool.max_context)))
    }

    fn write_selection_inputs<'a>(
        &self,
        pool: u32,
        selected: impl Iterator<Item = SelectionLane<'a>>,
    ) -> Result<(), compute::Error> {
        let Some((_, slot, max_context)) = self.input_slot(pool)? else {
            return Ok(());
        };
        let Some(inputs) = &slot.inputs else {
            return Ok(());
        };
        let layout = slot
            .input_layout
            .ok_or_else(|| invalid_handle("state input layout is missing"))?;
        let width = usize::try_from(layout.vocab.div_ceil(32)).map_err(invalid_handle)?;
        let lanes = slot.binding.lanes.len();
        let words = u64::from(layout.vocab.div_ceil(32))
            .checked_add(6)
            .and_then(|words| words.checked_mul(u64::try_from(lanes).ok()?))
            .and_then(|words| words.checked_mul(8))
            .ok_or_else(|| invalid_handle("selection staging bytes overflowed"))?;
        let _charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            words,
        )?;
        let count = lanes
            .checked_mul(width)
            .ok_or_else(|| invalid_handle("selection mask count overflowed"))?;
        let mut masks = vec![0_u32; count];
        for row in masks.chunks_exact_mut(width) {
            row[0] = 1;
        }
        let mut sampling = vec![sampling_words(SamplingParams::default()); lanes];
        let mut positions = vec![0; lanes];
        for lane in selected {
            if lane.start >= max_context {
                return Err(invalid_handle(
                    "selection position or allow list is invalid",
                ));
            }
            *sampling
                .get_mut(lane.slot)
                .ok_or_else(|| invalid_handle("selection lane exceeds slot"))? =
                sampling_words(lane.sampling);
            positions[lane.slot] = lane.start;
            let row = masks
                .chunks_exact_mut(width)
                .nth(lane.slot)
                .ok_or_else(|| invalid_handle("selection lane exceeds mask"))?;
            lane.allow.write(row, layout.vocab)?;
        }
        let bank = &inputs.banks[slot.active_bank];
        let positions_handle = slot.metadata[slot.active_bank][2];
        self.recover_state_input_handles([bank.sampling, bank.allow, positions_handle])?;
        self.write_input_words(bank.sampling, sampling.into_iter().flatten())?;
        self.write_input_words(bank.allow, masks.into_iter())?;
        self.write_input_words(positions_handle, positions.into_iter())
    }

    pub(crate) fn write_state_batch_inputs(
        &self,
        pool: u32,
        input: &EngineBatch,
    ) -> Result<(), compute::Error> {
        self.write_state_inputs(
            pool,
            input.shape.phase,
            input.lanes.iter().flatten().map(|lane| InputLane {
                slot: lane.slot,
                tokens: lane.tokens.as_deref(),
                sampling: lane.sampling,
            }),
        )
    }

    pub(crate) fn write_state_single_inputs(
        &self,
        pool: u32,
        phase: BatchPhase,
        tokens: Option<&[u32]>,
        sampling: SamplingParams,
    ) -> Result<(), compute::Error> {
        self.write_state_inputs(
            pool,
            phase,
            std::iter::once(InputLane {
                slot: 0,
                tokens,
                sampling,
            }),
        )
    }

    fn write_state_inputs<'a>(
        &self,
        pool: u32,
        phase: BatchPhase,
        source_lanes: impl Iterator<Item = InputLane<'a>>,
    ) -> Result<(), compute::Error> {
        let Some((slot_rep, slot, _)) = self.input_slot(pool)? else {
            return Ok(());
        };
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
        let lane_count = usize::try_from(lanes).map_err(invalid_handle)?;
        let mut sampling = vec![sampling_words(SamplingParams::default()); lane_count];
        let mut tokens = vec![0; lane_count];
        let mut explicit = vec![0; lane_count];
        let mut prompt = vec![0; usize::try_from(layout.prompt_tokens).map_err(invalid_handle)?];
        let mut allow = vec![0; usize::try_from(mask_count).map_err(invalid_handle)?];
        for row in allow.chunks_exact_mut(usize::try_from(mask_words).map_err(invalid_handle)?) {
            row[0] = 1;
        }
        let mut prompt_lanes = 0;
        for lane in source_lanes {
            let index = usize::try_from(lane.slot).map_err(invalid_handle)?;
            *sampling
                .get_mut(index)
                .ok_or_else(|| invalid_handle("sampling lane exceeds slot"))? =
                sampling_words(lane.sampling);
            allow_vocabulary(&mut allow, index, layout.vocab)?;
            if let Some(values) = lane.tokens {
                *explicit
                    .get_mut(index)
                    .ok_or_else(|| invalid_handle("token lane exceeds slot"))? = 1;
                *tokens
                    .get_mut(index)
                    .ok_or_else(|| invalid_handle("token lane exceeds slot"))? = *values
                    .first()
                    .ok_or_else(|| invalid_handle("explicit tokens are empty"))?;
                if phase == BatchPhase::Prompt {
                    prompt_lanes += 1;
                    prompt
                        .get_mut(..values.len())
                        .ok_or_else(|| invalid_handle("prompt exceeds slot input budget"))?
                        .copy_from_slice(values);
                }
            } else if phase == BatchPhase::Prompt {
                return Err(invalid_handle("prompt input tokens are missing"));
            }
            if let Some(state) = self.state_slots.get(&slot_rep) {
                state.select_feedback_input(slot.active_bank, index, lane.tokens.is_some())?;
            }
        }
        if phase == BatchPhase::Prompt && prompt_lanes != 1 {
            return Err(invalid_handle("prompt requires exactly one input lane"));
        }
        self.recover_state_input_bank(bank)?;
        self.write_state_input_words(
            bank,
            sampling.into_iter().flatten(),
            tokens,
            explicit,
            prompt,
            allow,
        )?;
        bank.prompt_phase.store(
            phase == BatchPhase::Prompt,
            std::sync::atomic::Ordering::Release,
        );
        Ok(())
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
