use super::{Backend, StateSlots};
use crate::{
    CanonicalPrompt, HybridAttachment, HybridPrefix, PagedPrefix, StateSequenceHandle, compute,
    invalid_handle,
};
use std::sync::Arc;

/// Unpublished state and attention ownership prepared for one empty serving lane.
pub struct StagedPrefix<B: Backend> {
    lane: usize,
    sequence: StateSequenceHandle<B>,
    prompt: Arc<CanonicalPrompt>,
    attachment: Option<HybridAttachment<B>>,
}
impl<B: Backend> StagedPrefix<B> {
    /// Reserved attention capability for a hit; a cold admission has none.
    #[must_use]
    pub fn attention(&self) -> Option<&PagedPrefix> {
        self.attachment.as_ref().map(HybridAttachment::attention)
    }
}
impl<B: Backend> StateSlots<B> {
    /// Stages every state version and optional hybrid hit before publishing a lane.
    ///
    /// # Errors
    /// Refuses occupied lanes, foreign ledgers/images, inadequate reservations or quota.
    pub fn stage_prefix(
        &self,
        lane: usize,
        prompt: Arc<CanonicalPrompt>,
        hit: Option<&Arc<HybridPrefix<B>>>,
        reserved: u64,
    ) -> Result<StagedPrefix<B>, compute::Error> {
        let owner = self
            .lanes
            .get(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        if owner.sequence.is_some() {
            return Err(crate::quota(crate::PAGED_LANE_RELEASE_REQUIRED));
        }
        self.validate_prompt(&prompt)?;
        if u64::try_from(prompt.tokens().len()).map_err(invalid_handle)? > reserved {
            return Err(crate::quota("canonical prompt exceeds the reservation"));
        }
        if hit.is_some_and(|hit| {
            hit.namespace() != prompt.namespace()
                || hit.tokens().len() >= prompt.tokens().len()
                || !prompt.tokens().starts_with(hit.tokens())
        }) {
            return Err(invalid_handle(
                "hybrid hit does not match the complete prompt",
            ));
        }
        let sequence = self.storage.reserve()?;
        let attachment = hit.map(|hit| hit.attach(&sequence, reserved)).transpose()?;
        if let Some(feedback) = &self.feedback {
            sequence
                .lock()
                .attach_feedback(feedback.clone())
                .map_err(invalid_handle)?;
        }
        sequence.lock().position.reserved_tokens = reserved;
        Ok(StagedPrefix {
            lane,
            sequence,
            prompt,
            attachment,
        })
    }
    /// Publishes a previously staged lane after its attention tables have been admitted.
    ///
    /// # Errors
    /// Refuses a foreign slot, a different lane or an occupied destination without replacing it.
    pub fn install_prefix(
        &mut self,
        lane: usize,
        staged: StagedPrefix<B>,
    ) -> Result<(), compute::Error> {
        if staged.lane != lane || !Arc::ptr_eq(&self.storage, &staged.sequence.storage) {
            return Err(invalid_handle(
                "staged prefix belongs to another lane or storage",
            ));
        }
        let owner = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        if owner.sequence.is_some() {
            return Err(crate::quota(crate::PAGED_LANE_RELEASE_REQUIRED));
        }
        owner.sequence = Some(staged.sequence);
        owner.prompt = Some(staged.prompt);
        owner.attachment = staged.attachment;
        owner.serial = 0;
        Ok(())
    }
}
