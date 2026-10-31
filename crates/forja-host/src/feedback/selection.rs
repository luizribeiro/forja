use super::{
    Backend, CommandListEntry, FeedbackWrite, Host, Resource, StateUse, TensorEntry, compute,
    invalid_handle,
};
use std::sync::Weak;

#[derive(Debug)]
pub(crate) struct SelectionWrite {
    slot: u32,
    bank: usize,
    lane: usize,
    identity: Weak<()>,
}

impl<B: Backend + 'static> Host<B> {
    pub(super) fn record_selection_feedback(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: u32,
        bank: usize,
        lane: usize,
        selected: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        let list = self.table.get(commands).map_err(invalid_handle)?;
        if !list.state_uses.is_empty()
            || list
                .selection_writes
                .iter()
                .any(|write| write.slot == slot && write.bank == bank && write.lane == lane)
        {
            return Err(invalid_handle(
                "selection feedback conflicts with recorded lane work",
            ));
        }
        let (identity, index) = self
            .state_slots
            .get(&slot)
            .ok_or_else(|| invalid_handle("state slot is missing"))?
            .selection_reservation(bank, lane)?
            .ok_or_else(|| invalid_handle("feedback requires state work or a selection guard"))?;
        let output = self.feedback_destination(slot, index)?;
        let result = self.dispatch(
            commands,
            compute::Op::Copy,
            &[Resource::new_borrow(selected.rep())],
            &output,
        );
        self.drop_tensor(output)?;
        result?;
        self.table
            .get_mut(commands)
            .map_err(invalid_handle)?
            .selection_writes
            .push(SelectionWrite {
                slot,
                bank,
                lane,
                identity,
            });
        Ok(())
    }

    pub(crate) fn prepare_protected_work(
        &mut self,
        usages: &[StateUse],
        feedback: &[FeedbackWrite],
        selections: &[SelectionWrite],
    ) -> Result<
        (
            Option<crate::hybrid::ProtectedBatch<B>>,
            Vec<forja_core::CompletionResource>,
        ),
        compute::Error,
    > {
        let Some(first) = selections.first() else {
            return self
                .prepare_state_work(usages, feedback)
                .map(|(batch, owners)| (batch.map(Into::into), owners));
        };
        if !usages.is_empty() || !feedback.is_empty() {
            return Err(invalid_handle(
                "selection cannot publish a carry transaction",
            ));
        }
        let state = self
            .state_slots
            .get_mut(&first.slot)
            .ok_or_else(|| invalid_handle("state slot is missing"))?;
        if state.selection_count(first.bank)? != selections.len() {
            return Err(invalid_handle("selection must write every reserved lane"));
        }
        for (index, write) in selections.iter().enumerate() {
            if write.slot != first.slot
                || write.bank != first.bank
                || selections[..index]
                    .iter()
                    .any(|other| other.lane == write.lane)
                || state
                    .selection_reservation(write.bank, write.lane)?
                    .is_none_or(|(identity, _)| !identity.ptr_eq(&write.identity))
            {
                return Err(invalid_handle(
                    "selection output differs from its reserved boundary",
                ));
            }
        }
        Ok((Some(state.take_selections(first.bank)?.into()), Vec::new()))
    }
}
