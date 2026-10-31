use super::{Backend, BatchSlotEntry, Host, PromptReplayLane, Resource, compute, invalid_handle};
use forja_core::{BufferId, Layout, ParamValues, ReplayBindings};

fn row_roles(slot: &BatchSlotEntry) -> Result<[(u32, u32, u64); 10], compute::Error> {
    let inputs = slot
        .inputs
        .as_ref()
        .ok_or_else(|| invalid_handle("prompt inputs are missing"))?;
    let layout = slot
        .input_layout
        .ok_or_else(|| invalid_handle("prompt input layout is missing"))?;
    let (old, new) = (&inputs.banks[0], &inputs.banks[slot.active_bank]);
    let metadata = slot.metadata;
    let bank = slot.active_bank;
    Ok([
        (metadata[0][0], metadata[bank][0], 1),
        (metadata[0][1], metadata[bank][1], 1),
        (metadata[0][2], metadata[bank][2], 1),
        (metadata[0][3], metadata[bank][3], 1),
        (old.sampling, new.sampling, 5),
        (old.tokens, new.tokens, 1),
        (old.explicit, new.explicit, 1),
        (old.allow, new.allow, u64::from(layout.vocab.div_ceil(32))),
        (old.logits, new.logits, u64::from(layout.vocab)),
        (old.selected, new.selected, 1),
    ])
}

fn row_layout(
    layout: &Layout,
    buffer: BufferId,
    width: u64,
    lane: usize,
) -> Result<Layout, compute::Error> {
    let row_bytes = width
        .checked_mul(layout.dtype().byte_size())
        .ok_or_else(|| invalid_handle("prompt row bytes overflowed"))?;
    if layout.byte_span().end > row_bytes {
        return Err(invalid_handle(
            "prompt operand spans more than its logical row",
        ));
    }
    let offset = u64::try_from(lane)
        .ok()
        .and_then(|lane| lane.checked_mul(width))
        .and_then(|delta| layout.offset().checked_add(delta))
        .ok_or_else(|| invalid_handle("prompt row offset overflowed"))?;
    Layout::new(
        layout.dtype(),
        offset,
        layout.shape().to_vec(),
        layout.strides().to_vec(),
        buffer.byte_len(),
    )
    .map_err(crate::guest_error)
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn bind_prompt_rows(
        &self,
        remap: Option<PromptReplayLane>,
        entries: &[crate::TensorEntry],
        values: &ParamValues,
        bindings: &mut ReplayBindings,
    ) -> Result<Vec<crate::TensorEntry>, compute::Error> {
        let Some(remap) = remap.filter(|remap| remap.lane != 0) else {
            return Ok(Vec::new());
        };
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(remap.slot))
            .map_err(invalid_handle)?;
        let _roles = self.reserve_reference()?;
        let roles = row_roles(slot)?
            .into_iter()
            .map(|(old, new, width)| {
                self.entry(&Resource::new_borrow(old))
                    .map(|entry| (entry.tensor.buffer(), new, width))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = std::collections::HashSet::new();
        let mut retained = Vec::new();
        for entry in entries {
            if entry.slot_token.is_some() {
                continue;
            }
            let Some((_, new, width)) = roles
                .iter()
                .find(|(old, _, _)| *old == entry.tensor.buffer())
            else {
                continue;
            };
            let layout = entry.replay_layout(values)?;
            if !seen.insert((
                entry.tensor.buffer(),
                layout.offset(),
                layout.shape().to_vec(),
                layout.strides().to_vec(),
                entry.tensor.is_writable(),
            )) {
                continue;
            }
            let charge = self.reserve_reference()?;
            let mut fresh = self.entry(&Resource::new_borrow(*new))?.clone();
            let target = row_layout(&layout, fresh.tensor.buffer(), *width, remap.lane)?;
            fresh.tensor = self
                .backend
                .tensor(fresh.tensor.buffer(), target)
                .map_err(crate::guest_error)?;
            let mut old = self
                .backend
                .tensor(entry.tensor.buffer(), layout)
                .map_err(crate::guest_error)?;
            if !entry.tensor.is_writable() {
                fresh.tensor = fresh.tensor.read_only();
                old = old.read_only();
            }
            fresh.reference_charge = Some(charge);
            fresh.symbolic = None;
            bindings
                .view(old, fresh.tensor.clone())
                .map_err(crate::guest_error)?;
            retained.push(fresh);
        }
        Ok(retained)
    }
}
