use super::{
    CanonicalPosition, LanePosition, PagePoolError, SelectionPhase, StateJournal, StateTransaction,
};

impl StateJournal {
    pub(in crate::state) fn publish_accepted(
        &mut self,
        transaction: &StateTransaction,
        accepted: u32,
    ) -> Result<(), PagePoolError> {
        let acceptance = self
            .acceptance
            .as_ref()
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        if !acceptance.transaction.same_reservation(transaction) {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        let output = acceptance
            .outputs
            .get(usize::try_from(accepted).map_err(|_| PagePoolError::ArithmeticOverflow)?)
            .ok_or(PagePoolError::InvalidCheckpoint)?;
        let tokens = self
            .position
            .tokens
            .checked_add(u64::from(accepted))
            .and_then(|tokens| tokens.checked_add(1))
            .ok_or(PagePoolError::ArithmeticOverflow)?;
        let position = LanePosition {
            tokens,
            sampler: tokens
                .checked_sub(1)
                .ok_or(PagePoolError::ArithmeticOverflow)?,
            grammar: tokens,
            parser: tokens,
            reserved_tokens: acceptance.position.reserved_tokens,
        };
        let paged = self.retained_paged.as_ref();
        self.sequence.rewind_with(output, || {
            if let Some(paged) = paged {
                paged.truncate(acceptance.position.tokens, tokens)?;
            }
            Ok(())
        })?;
        self.position = position;
        self.phase = SelectionPhase::External;
        self.canonical = CanonicalPosition::Recurrent;
        if let Some(acceptance) = self.acceptance.take() {
            acceptance.outcome.finish(true);
        }
        Ok(())
    }
}
