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

#[derive(Clone, Debug)]
pub(crate) struct CapturedSelection {
    slot: u32,
    lane: usize,
    destination: forja_core::Tensor,
}

impl<B: Backend + 'static> Host<B> {
    pub(super) fn record_captured_selection(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<crate::BatchSlotEntry>,
        lane: usize,
        selected: &Resource<TensorEntry>,
    ) -> Result<(), compute::Error> {
        self.slot_capture_bank(commands, slot)?;
        let list = self.table.get(commands).map_err(invalid_handle)?;
        if lane
            >= self
                .table
                .get(slot)
                .map_err(invalid_handle)?
                .binding
                .lanes
                .len()
            || list
                .captured_selections
                .iter()
                .any(|site| site.slot == slot.rep() && site.lane == lane)
        {
            return Err(invalid_handle(
                "selection capture lane is invalid or already recorded",
            ));
        }
        let destination = self.record_feedback_placeholder(commands, selected)?;
        self.table
            .get_mut(commands)
            .map_err(invalid_handle)?
            .captured_selections
            .push(CapturedSelection {
                slot: slot.rep(),
                lane,
                destination,
            });
        Ok(())
    }

    pub(crate) fn bind_graph_selections(
        &mut self,
        sites: &[CapturedSelection],
        bindings: &mut forja_core::ReplayBindings,
        retained: &mut Vec<TensorEntry>,
    ) -> Result<Vec<SelectionWrite>, compute::Error> {
        let mut writes = Vec::new();
        for site in sites {
            let slot = self
                .table
                .get(&Resource::<crate::BatchSlotEntry>::new_borrow(site.slot))
                .map_err(invalid_handle)?;
            let bank = slot.active_bank;
            let Some((identity, index)) = self
                .state_slots
                .get(&site.slot)
                .ok_or_else(|| invalid_handle("state slot is missing"))?
                .selection_reservation(bank, site.lane)?
            else {
                continue;
            };
            let loan = self.feedback_destination(site.slot, index)?;
            let entry = self.entry(&loan).cloned();
            let entry = crate::recovery::finish(entry, self.drop_tensor(loan))
                .map_err(crate::recovery::RecoveryError::into_compute)?;
            bindings
                .view(site.destination.clone(), entry.tensor.clone())
                .map_err(crate::guest_error)?;
            retained.push(entry);
            writes.push(SelectionWrite {
                slot: site.slot,
                bank,
                lane: site.lane,
                identity,
            });
        }
        if !sites.is_empty() && writes.is_empty() {
            return Err(invalid_handle("selection capture has no reserved output"));
        }
        Ok(writes)
    }

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
        crate::recovery::finish(result, self.drop_tensor(output))
            .map_err(crate::recovery::RecoveryError::into_compute)?;
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
