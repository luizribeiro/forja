use crate::{
    Backend, BatchSlotEntry, CommandListEntry, Host, TensorEntry, compute, hybrid::StateUse,
    invalid_handle,
};
use forja_core::StateTransaction;
use wasmtime::component::Resource;

pub(crate) struct FeedbackWrite {
    pub(crate) usage: StateUse,
    transaction: StateTransaction,
}

impl std::fmt::Debug for FeedbackWrite {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FeedbackWrite")
            .field("usage", &self.usage)
            .finish_non_exhaustive()
    }
}

impl<B: Backend + 'static> Host<B> {
    /// Records a token into the current transaction's private feedback destination.
    ///
    /// # Errors
    /// Refuses invalid tokens, non-combined transactions, duplicate writes or exhausted quota.
    pub fn record_slot_feedback(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
        lane: u32,
        selected: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        let input = self.entry(selected)?;
        if input.symbolic.is_some()
            || input.tensor.layout().dtype() != forja_core::DType::U32
            || input.tensor.layout().shape() != [1]
        {
            return Err(invalid_handle("feedback requires one fixed token"));
        }
        let slot_entry = self.table.get(slot).map_err(invalid_handle)?;
        let bank = slot_entry.active_bank;
        let lane = usize::try_from(lane).map_err(invalid_handle)?;
        let list = self.table.get(commands).map_err(invalid_handle)?;
        let usage = *list
            .state_uses
            .iter()
            .find(|usage| {
                usage.slot == slot.rep() && usage.bank == bank && usage.lane == lane && usage.active
            })
            .ok_or_else(|| invalid_handle("feedback requires state updates in the same list"))?;
        let (transaction, index) = self
            .state_slots
            .get(&slot.rep())
            .ok_or_else(|| invalid_handle("slot has no protected state"))?
            .feedback_output(bank, lane)?
            .ok_or_else(|| invalid_handle("feedback requires a combined transaction"))?;
        if list
            .feedback_writes
            .iter()
            .any(|write| write.usage == usage)
        {
            return Err(invalid_handle("feedback is already recorded for this lane"));
        }
        let handle = *slot_entry
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("slot has no feedback storage"))?
            .feedback
            .get(usize::try_from(index).map_err(invalid_handle)?)
            .ok_or_else(|| invalid_handle("feedback destination is missing"))?;
        let output = self.lend_view(
            &Resource::new_borrow(handle),
            compute::ViewOp::Reshape(vec![1]),
            false,
        )?;
        let result = self.dispatch(
            commands,
            compute::Op::Copy,
            &[Resource::new_borrow(selected.rep())],
            &output,
        );
        self.drop_tensor(output)?;
        result?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry
            .feedback_writes
            .push(FeedbackWrite { usage, transaction });
        Ok(())
    }

    pub(crate) fn validate_feedback_writes(
        &self,
        usages: &[StateUse],
        writes: &[FeedbackWrite],
    ) -> Result<(), compute::Error> {
        for usage in usages.iter().filter(|usage| usage.active) {
            let state = self
                .state_slots
                .get(&usage.slot)
                .ok_or_else(|| invalid_handle("state slot is missing"))?;
            let expected = state.feedback_output(usage.bank, usage.lane)?;
            let recorded = writes.iter().find(|write| write.usage == *usage);
            match (expected, recorded) {
                (None, None) => {}
                (Some((expected, _)), Some(write))
                    if expected.same_reservation(&write.transaction) => {}
                _ => {
                    return Err(invalid_handle(
                        "feedback output differs from the current transaction",
                    ));
                }
            }
        }
        Ok(())
    }
}
