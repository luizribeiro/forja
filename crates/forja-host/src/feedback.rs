mod selection;
pub(crate) use selection::SelectionWrite;

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

#[derive(Clone, Debug)]
pub(crate) struct CapturedFeedback {
    usage: StateUse,
    destination: forja_core::Tensor,
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
        if self
            .table
            .get(commands)
            .map_err(invalid_handle)?
            .slot_capture
            .is_some()
        {
            return self.record_captured_feedback(commands, slot, lane, selected);
        }
        let slot_entry = self.table.get(slot).map_err(invalid_handle)?;
        let bank = slot_entry.active_bank;
        let lane = usize::try_from(lane).map_err(invalid_handle)?;
        let list = self.table.get(commands).map_err(invalid_handle)?;
        let usage = list
            .state_uses
            .iter()
            .find(|usage| {
                usage.slot == slot.rep() && usage.bank == bank && usage.lane == lane && usage.active
            })
            .copied();
        let Some(usage) = usage else {
            return self.record_selection_feedback(commands, slot.rep(), bank, lane, selected);
        };
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
        let output = self.feedback_destination(slot.rep(), index)?;
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

    fn record_captured_feedback(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
        lane: u32,
        selected: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        let lane = usize::try_from(lane).map_err(invalid_handle)?;
        let list = self.table.get(commands).map_err(invalid_handle)?;
        let usage = *list
            .state_uses
            .iter()
            .find(|usage| usage.slot == slot.rep() && usage.lane == lane)
            .ok_or_else(|| {
                invalid_handle("feedback capture requires state updates in the same list")
            })?;
        if list
            .captured_feedback
            .iter()
            .any(|site| site.usage == usage)
        {
            return Err(invalid_handle(
                "feedback capture is already recorded for this lane",
            ));
        }
        let charge = self.reserve_reference()?;
        let output = self.alloc(compute::Dtype::U32, &[1])?;
        self.table
            .get_mut(&output)
            .map_err(invalid_handle)?
            .reference_charge = Some(charge);
        let destination = self.entry(&output)?.tensor.clone();
        let result = self.dispatch(
            commands,
            compute::Op::Copy,
            &[Resource::new_borrow(selected.rep())],
            &output,
        );
        self.drop_tensor(output)?;
        result?;
        let list = self.table.get_mut(commands).map_err(invalid_handle)?;
        list.protected.insert(destination.buffer());
        list.captured_feedback
            .push(CapturedFeedback { usage, destination });
        Ok(())
    }

    pub(crate) fn bind_graph_feedback(
        &mut self,
        sites: &[CapturedFeedback],
        usages: &[StateUse],
        bindings: &mut forja_core::ReplayBindings,
        retained: &mut Vec<TensorEntry>,
    ) -> Result<Vec<FeedbackWrite>, compute::Error> {
        let mut writes = Vec::new();
        for site in sites {
            let usage = *usages
                .iter()
                .find(|usage| usage.slot == site.usage.slot && usage.lane == site.usage.lane)
                .ok_or_else(|| invalid_handle("feedback capture has no state lane"))?;
            if !usage.active {
                continue;
            }
            let Some((transaction, index)) = self
                .state_slots
                .get(&usage.slot)
                .ok_or_else(|| invalid_handle("state slot is missing"))?
                .feedback_output(usage.bank, usage.lane)?
            else {
                continue;
            };
            let loan = self.feedback_destination(usage.slot, index)?;
            let entry = self.entry(&loan).cloned();
            self.drop_tensor(loan)?;
            let entry = entry?;
            bindings
                .view(site.destination.clone(), entry.tensor.clone())
                .map_err(crate::guest_error)?;
            retained.push(entry);
            writes.push(FeedbackWrite { usage, transaction });
        }
        Ok(writes)
    }

    fn feedback_destination(
        &mut self,
        slot: u32,
        index: u32,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        let handle = *self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("slot has no feedback storage"))?
            .feedback
            .get(usize::try_from(index).map_err(invalid_handle)?)
            .ok_or_else(|| invalid_handle("feedback destination is missing"))?;
        self.lend_view(
            &Resource::new_borrow(handle),
            compute::ViewOp::Reshape(vec![1]),
            false,
        )
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
