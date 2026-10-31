use super::{Backend, StateSlots};
use crate::{CanonicalPrompt, compute, invalid_handle};
use std::sync::Arc;

impl<B: Backend> StateSlots<B> {
    /// Validated geometry of every protected plane in this slot.
    #[must_use]
    pub fn prefix_layout(&self) -> &forja_core::StateLayout {
        self.storage.pool.layout()
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
        if !Arc::ptr_eq(&prompt.charge.live, &self.storage.charge.live)
            || !prompt.namespace().matches_state(self.prefix_layout())
        {
            return Err(invalid_handle(
                "canonical prompt belongs to another state domain",
            ));
        }
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
        slots.admit_reserved(0, 257).unwrap();
        slots.set_prompt(0, prompt.clone()).unwrap();
        assert!(slots.set_prompt(0, prompt).is_err());
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
