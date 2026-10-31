mod rows;

use crate::{
    Backend, BatchSlotEntry, CommandListEntry, Host, PagedBinding, SharedStateTrace, compute,
    invalid_handle, state::StateCharge,
};
use std::sync::{Arc, Weak, atomic::Ordering};
use wasmtime::component::Resource;

#[derive(Clone, Debug)]
pub(crate) struct SlotCapture {
    pub(crate) slot: u32,
    pub(crate) prefill: bool,
    binding: Weak<PagedBinding>,
    pub(crate) trace: SharedStateTrace,
    _charge: Arc<StateCharge>,
}

#[derive(Clone, Copy)]
pub(crate) struct PromptReplayLane {
    slot: u32,
    lane: usize,
}
impl PromptReplayLane {
    pub(crate) fn map(remap: Option<Self>, slot: u32, lane: usize) -> usize {
        remap
            .filter(|remap| remap.slot == slot && lane == 0)
            .map_or(lane, |remap| remap.lane)
    }
}

impl SlotCapture {
    pub(crate) fn prompt_lane<B: Backend + 'static>(
        &self,
        host: &Host<B>,
        numerical: bool,
    ) -> Result<Option<PromptReplayLane>, compute::Error> {
        self.validate(host, numerical)?;
        if !self.prefill {
            return Ok(None);
        }
        let slot = host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
            .map_err(invalid_handle)?;
        let mut active = slot.binding.activity[slot.active_bank]
            .iter()
            .enumerate()
            .filter(|(_, active)| active.load(Ordering::Acquire));
        let lane = active.next().map(|(lane, _)| lane);
        if active.next().is_some() {
            return Err(invalid_handle(
                "prompt replay requires at most one active lane",
            ));
        }
        Ok(lane.map(|lane| PromptReplayLane {
            slot: self.slot,
            lane,
        }))
    }

    pub(crate) fn validate<B: Backend + 'static>(
        &self,
        host: &Host<B>,
        numerical: bool,
    ) -> Result<(), compute::Error> {
        let slot = host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
            .map_err(invalid_handle)?;
        let inputs = slot
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("capture slot has no state inputs"))?;
        if !self.binding.ptr_eq(&Arc::downgrade(&slot.binding))
            || (numerical
                && inputs.banks[slot.active_bank]
                    .prompt_phase
                    .load(Ordering::Acquire)
                    != self.prefill)
        {
            return Err(invalid_handle(
                "capture slot identity or numerical phase differs",
            ));
        }
        Ok(())
    }
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
        let binding = Arc::downgrade(&self.table.get(slot).map_err(invalid_handle)?.binding);
        let charge = self.reserve_reference()?;
        let slot_entry = self.table.get(slot).map_err(invalid_handle)?;
        let handles = slot_entry.metadata[0].into_iter().chain(
            slot_entry
                .inputs
                .iter()
                .flat_map(|inputs| inputs.banks[0].handles()),
        );
        let buffers = handles
            .map(|handle| {
                self.entry(&Resource::new_borrow(handle))
                    .map(|entry| entry.tensor.buffer())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let trace = self
            .state_slots
            .get(&slot.rep())
            .ok_or_else(|| invalid_handle("capture slot has no state"))?
            .trace()?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry.protected.extend(buffers);
        entry.slot_capture = Some(SlotCapture {
            slot: slot.rep(),
            prefill,
            binding,
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
                || (capture.prefill && usage.lane != 0)
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
