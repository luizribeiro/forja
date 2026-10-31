use crate::{
    Backend, BatchSlotEntry, CommandListEntry, Host, SharedStateTrace, compute, invalid_handle,
    state::StateCharge,
};
use std::sync::Arc;
use wasmtime::component::Resource;

#[derive(Clone, Debug)]
pub(crate) struct SlotCapture {
    pub(crate) slot: u32,
    pub(crate) prefill: bool,
    pub(crate) trace: SharedStateTrace,
    _charge: Arc<StateCharge>,
}

impl<B: Backend + Send + Sync + 'static> Host<B> {
    /// Declares a graph-only slot recording with independent inactive state bindings.
    ///
    /// # Errors
    /// Refuses nonempty lists, duplicate declarations, missing state slots or exhausted quota.
    pub fn declare_slot_capture(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
        prefill: bool,
    ) -> Result<(), compute::Error> {
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if !entry.recorded.is_empty() || entry.slot_capture.is_some() {
            return Err(invalid_handle(
                "slot capture requires an empty undeclared command list",
            ));
        }
        let charge = self.reserve_reference()?;
        let trace = self
            .state_slots
            .get(&slot.rep())
            .ok_or_else(|| invalid_handle("capture slot has no state"))?
            .trace()?;
        self.table
            .get_mut(commands)
            .map_err(invalid_handle)?
            .slot_capture = Some(SlotCapture {
            slot: slot.rep(),
            prefill,
            trace: trace.share(),
            _charge: charge,
        });
        Ok(())
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn slot_capture_bank(
        &self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
    ) -> Result<Option<usize>, compute::Error> {
        match self
            .table
            .get(commands)
            .map_err(invalid_handle)?
            .slot_capture
            .as_ref()
        {
            None => Ok(None),
            Some(capture) if capture.slot == slot.rep() => Ok(Some(0)),
            Some(_) => Err(invalid_handle("paged work differs from its capture slot")),
        }
    }

    pub(crate) fn with_state_commands<T>(
        &mut self,
        usage: crate::hybrid::StateUse,
        capture: Option<&SlotCapture>,
        action: impl FnOnce(&mut crate::StateCommands<B>) -> Result<T, compute::Error>,
    ) -> Result<T, compute::Error> {
        if let Some(capture) = capture {
            if capture.slot != usage.slot
                || usage.active
                || usage.bank != 0
                || (!capture.prefill && usage.tokens != 1)
            {
                return Err(invalid_handle(
                    "state work differs from its capture declaration",
                ));
            }
            let mut trace = capture.trace.lock::<B>()?;
            return action(
                trace
                    .as_mut()
                    .ok_or_else(|| invalid_handle("state trace has retired"))?
                    .lane(usage.lane)?,
            );
        }
        let state = self
            .state_slots
            .get_mut(&usage.slot)
            .ok_or_else(|| invalid_handle("state slot is missing"))?;
        action(state.commands(usage.bank, usage.lane, usage.active)?)
    }
}
