use super::{Backend, StateSlots};
use crate::{CanonicalPrompt, compute, invalid_handle};
use std::sync::Arc;

impl<B: Backend> StateSlots<B> {
    /// Validated geometry of every protected plane in this slot.
    #[must_use]
    pub fn prefix_layout(&self) -> &forja_core::StateLayout {
        self.storage.pool.layout()
    }
    pub(crate) fn validate_prompt_input(
        &self,
        lane: usize,
        phase: crate::BatchPhase,
        start: u32,
        tokens: Option<&[u32]>,
    ) -> Result<(), compute::Error> {
        let owner = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        let Some(prompt) = &owner.prompt else {
            return Ok(());
        };
        let length = u64::try_from(prompt.tokens().len()).map_err(invalid_handle)?;
        let start = u64::from(start);
        if phase == crate::BatchPhase::Decode {
            return if start >= length {
                Ok(())
            } else {
                Err(invalid_handle(
                    "decode cannot replace canonical prompt input",
                ))
            };
        }
        let tokens =
            tokens.ok_or_else(|| invalid_handle("canonical prompt requires explicit input"))?;
        let end = start
            .checked_add(u64::try_from(tokens.len()).map_err(invalid_handle)?)
            .ok_or_else(|| invalid_handle("canonical input extent overflow"))?;
        let block = u64::from(prompt.namespace().block().get());
        let range = usize::try_from(start).map_err(invalid_handle)?
            ..usize::try_from(end).map_err(invalid_handle)?;
        if tokens.is_empty()
            || !start.is_multiple_of(block)
            || (end < length && !end.is_multiple_of(block))
            || prompt.tokens().get(range) != Some(tokens)
        {
            return Err(invalid_handle(
                "input differs from the canonical prompt ledger",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_prompt(&self, prompt: &CanonicalPrompt) -> Result<(), compute::Error> {
        if !Arc::ptr_eq(&prompt.charge.live, &self.storage.charge.live)
            || !prompt.namespace().matches_state(self.prefix_layout())
        {
            return Err(invalid_handle(
                "canonical prompt belongs to another state domain",
            ));
        }
        Ok(())
    }
    /// Binds an exact prompt ledger before an admitted lane begins evaluating it.
    ///
    /// # Errors
    /// Refuses foreign identities, existing ledgers, used sequences or checkpoint readers.
    pub fn set_prompt(
        &mut self,
        lane: usize,
        prompt: Arc<CanonicalPrompt>,
    ) -> Result<(), compute::Error> {
        self.validate_prompt(&prompt)?;
        let owner = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        let sequence = owner
            .sequence
            .as_ref()
            .ok_or_else(|| invalid_handle("state lane is not admitted"))?;
        sequence.lock().ensure_fresh().map_err(invalid_handle)?;
        if owner.prompt.is_some() {
            return Err(invalid_handle("canonical prompt is already installed"));
        }
        owner.prompt = Some(prompt);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, LanePosition, Limits, StateAction};
    use forja_core::{
        DType, HybridPrefixNamespace, PagedLayout, PagedPlane, StateLayout, StatePlane,
    };
    use std::num::NonZeroU32;
    #[test]
    fn slot_prompt_provenance_ends_before_the_prompt_tail() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(131_072, 8, 4096, 32, 4096),
        );
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap();
        let pages = PagedLayout::new(
            64,
            vec![PagedPlane {
                dtype: DType::F32,
                elements_per_token: 7,
                alignment: 4,
                label: "key".into(),
            }],
        )
        .unwrap();
        let namespace = HybridPrefixNamespace::new(
            [0; 32],
            &layout,
            &pages,
            NonZeroU32::new(64).unwrap(),
            &["fixed"],
        )
        .unwrap();
        let prompt = host
            .canonical_prompt(namespace, (0..129).collect())
            .unwrap();
        let storage = host.create_state_storage(layout, 1).unwrap();
        let mut slots = StateSlots::new(storage, 1).unwrap();
        assert!(slots.stage_prefix(0, prompt.clone(), None, 64).is_err());
        assert!(slots.sequence_handle(0).unwrap().is_none());
        let staged = slots.stage_prefix(0, prompt.clone(), None, 257).unwrap();
        assert!(staged.attention().is_none());
        slots.install_prefix(0, staged).unwrap();
        assert!(slots.set_prompt(0, prompt).is_err());
        assert!(
            slots
                .validate_prompt_input(0, crate::BatchPhase::Decode, 0, Some(&[0]))
                .is_err()
        );
        assert!(
            slots
                .validate_prompt_input(0, crate::BatchPhase::Prompt, 0, None)
                .is_err()
        );
        assert!(
            slots
                .validate_prompt_input(0, crate::BatchPhase::Prompt, 0, Some(&[0; 64]))
                .is_err()
        );
        let tokens = (0..129).collect::<Vec<_>>();
        assert!(
            slots
                .validate_prompt_input(0, crate::BatchPhase::Prompt, 0, Some(&tokens[..63]))
                .is_err()
        );
        slots
            .validate_prompt_input(0, crate::BatchPhase::Prompt, 0, Some(&tokens[..64]))
            .unwrap();
        assert_eq!(slots.position(0).unwrap().tokens, 0);
        slots
            .validate_prompt_input(0, crate::BatchPhase::Decode, 129, None)
            .unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let mut end = 0;
        for rows in [64, 64, 1] {
            end += rows;
            let input = host.backend.alloc(DType::F32, &[rows, 7]).unwrap();
            let output = host.backend.alloc(DType::F32, &[rows, 7]).unwrap();
            slots
                .begin_action(
                    0,
                    0,
                    LanePosition {
                        tokens: u64::from(end),
                        reserved_tokens: 257,
                        ..LanePosition::default()
                    },
                    Some(StateAction::Step),
                )
                .unwrap();
            slots
                .recording(0, 0)
                .unwrap()
                .causal_convolution(0, &input, &weights, &output)
                .unwrap();
            slots.take_batch(0).unwrap().submit_and_finish().unwrap();
            let boundary = slots
                .sequence_handle(0)
                .unwrap()
                .unwrap()
                .canonical_token_count()
                .unwrap();
            assert_eq!(boundary, (end < 129).then_some(u64::from(end)));
        }
        slots.release(0).unwrap();
        assert!(slots.lanes[0].prompt.is_none());
    }
}
