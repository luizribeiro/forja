use crate::{Host, PagePoolEntry, StateSlots, compute, invalid_handle, paged_storage};
use forja_core::{Backend, StateLayout, StatePlane};
use wasmtime::component::Resource;

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn open_state_slot(
        &mut self,
        pool: &Resource<PagePoolEntry>,
        planes: u32,
        descriptors: Vec<paged_storage::StatePlaneDescriptor>,
    ) -> Result<Resource<crate::BatchSlotEntry>, compute::Error> {
        let lanes = self.table.get(pool).map_err(invalid_handle)?.active_lanes;
        let layout = StateLayout::new(
            descriptors
                .into_iter()
                .map(|descriptor| {
                    StatePlane::new(crate::core_dtype(descriptor.dtype), descriptor.shape)
                })
                .collect::<Result<_, _>>()
                .map_err(invalid_handle)?,
        )
        .map_err(invalid_handle)?;
        let storage = self.create_state_storage(layout, lanes)?;
        let state = StateSlots::new(storage, lanes)?;
        let slot = self.open_paged_slot(pool, planes)?;
        self.state_slots.insert(slot.rep(), state);
        Ok(slot)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateUse {
    pub(crate) slot: u32,
    pub(crate) bank: usize,
    pub(crate) lane: usize,
}

impl<B: Backend + 'static> Host<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_slot_state(
        &mut self,
        commands: &Resource<crate::CommandListEntry>,
        slot: &Resource<crate::BatchSlotEntry>,
        lane: u32,
        plane: u32,
        operation: paged_storage::StateOperation,
        inputs: &[Resource<crate::TensorEntry>],
        output: &Resource<crate::TensorEntry>,
        algorithm: &str,
    ) -> Result<(), compute::Error> {
        let slot_entry = self.table.get(slot).map_err(invalid_handle)?;
        let bank = slot_entry.active_bank;
        let lane = usize::try_from(lane).map_err(invalid_handle)?;
        let active = slot_entry.binding.activity[bank]
            .get(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?
            .load(std::sync::atomic::Ordering::Acquire);
        let inputs = inputs
            .iter()
            .map(|input| self.entry(input).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.entry(output)?.clone();
        if inputs
            .iter()
            .chain([&output])
            .any(|entry| entry.symbolic.is_some())
        {
            return Err(invalid_handle("state operands require fixed layouts"));
        }
        if algorithm.len() > crate::MAX_VARIANT_NAME_BYTES {
            return Err(invalid_handle("state algorithm name exceeds its limit"));
        }
        if active {
            let window = slot_entry
                .binding
                .banks
                .append_window(bank, lane)
                .map_err(invalid_handle)?;
            let tokens = match operation {
                paged_storage::StateOperation::DeltaRule => 1,
                _ => inputs
                    .first()
                    .and_then(|entry| entry.tensor.layout().shape().first())
                    .copied()
                    .map_or(0, u64::from),
            };
            if tokens == 0 || window.end.checked_sub(window.start) != Some(tokens) {
                return Err(invalid_handle(
                    "state update differs from its admitted token span",
                ));
            }
        }
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if entry.recorded.len() >= self.limits.dispatches_per_list {
            return Err(crate::quota(
                "command list dispatch count exceeds the guest limit",
            ));
        }
        let state = self
            .state_slots
            .get_mut(&slot.rep())
            .ok_or_else(|| invalid_handle("slot has no protected state"))?;
        let body = state.commands(bank, lane, active)?;
        let tensors = inputs.iter().map(|entry| &entry.tensor).collect::<Vec<_>>();
        let plane = usize::try_from(plane).map_err(invalid_handle)?;
        match (operation, tensors.as_slice()) {
            (paged_storage::StateOperation::CausalConvolution, [input, weights])
                if algorithm.is_empty() =>
            {
                body.causal_convolution(plane, input, weights, &output.tensor)?;
            }
            (paged_storage::StateOperation::DeltaRule, [q, k, v, decay, beta])
                if algorithm.is_empty() =>
            {
                body.delta_rule(plane, [q, k, v, decay, beta], &output.tensor)?;
            }
            (paged_storage::StateOperation::DeltaRuleBlock, [q, k, v, decay, beta]) => {
                body.delta_rule_block(
                    plane,
                    [q, k, v, decay, beta],
                    &output.tensor,
                    Some(algorithm),
                )?;
            }
            _ => return Err(invalid_handle("invalid state operands or algorithm")),
        }
        let protected = body.protected_buffers().collect::<Vec<_>>();
        let dispatches = std::mem::take(&mut body.commands).into_dispatches();
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry.protected.extend(protected);
        entry.access.record(&inputs, std::slice::from_ref(&output));
        entry.retained.extend(inputs);
        entry.retained.push(output);
        for dispatch in dispatches {
            entry.commands.record_validated(dispatch.clone());
            entry
                .recorded
                .push(crate::RecordedDispatch::Static(Box::new(dispatch)));
        }
        let usage = StateUse {
            slot: slot.rep(),
            bank,
            lane,
        };
        if !entry.state_uses.contains(&usage) {
            entry.state_uses.push(usage);
        }
        Ok(())
    }
}
