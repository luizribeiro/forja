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
    pub(crate) active: bool,
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
            active,
        };
        if !entry.state_uses.contains(&usage) {
            entry.state_uses.push(usage);
        }
        Ok(())
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn prepare_state_work(
        &mut self,
        usages: &[StateUse],
    ) -> Result<
        (
            Option<crate::StateBatch<B>>,
            Vec<forja_core::CompletionResource>,
        ),
        compute::Error,
    > {
        let Some(first) = usages.first() else {
            return Ok((None, Vec::new()));
        };
        if usages
            .iter()
            .any(|usage| usage.slot != first.slot || usage.bank != first.bank)
        {
            return Err(invalid_handle("state commands mix slot banks"));
        }
        let state = self
            .state_slots
            .get_mut(&first.slot)
            .ok_or_else(|| invalid_handle("state slot is missing"))?;
        let mut owners = Vec::new();
        for usage in usages {
            owners.extend(
                state
                    .commands(usage.bank, usage.lane, usage.active)?
                    .retained_resources(),
            );
        }
        let batch = if usages.iter().any(|usage| usage.active) {
            Some(state.take_batch(first.bank)?)
        } else {
            None
        };
        Ok((batch, owners))
    }
}

pub(crate) enum SubmittedWork<B: Backend> {
    Plain(B::Submission),
    State(crate::StateBatchSubmission<B>),
}

impl<B: Backend> SubmittedWork<B> {
    pub(crate) fn queue(
        batch: Option<crate::StateBatch<B>>,
        mut owners: Vec<forja_core::CompletionResource>,
        submit: impl FnOnce(
            Vec<forja_core::CompletionResource>,
        ) -> Result<B::Submission, forja_core::BackendError>,
    ) -> Result<Self, forja_core::BackendError> {
        match batch {
            None => submit(owners).map(Self::Plain),
            Some(batch) => batch
                .submit_using(|retained| {
                    owners.extend(retained);
                    submit(owners)
                })
                .map(Self::State)
                .map_err(|_| forja_core::BackendError::ExecutionFailed),
        }
    }

    pub(crate) fn backend(&self) -> &B::Submission {
        match self {
            Self::Plain(submission) => submission,
            Self::State(batch) => batch.submission(),
        }
    }

    pub(crate) fn finish<T>(
        self,
        taint: crate::TaintFlight,
        result: Result<T, forja_core::BackendError>,
        timeout: std::time::Duration,
    ) -> Result<T, forja_core::BackendError> {
        let result = taint.check_dependencies(result, timeout);
        let execution = result.as_ref().map(|_| ()).map_err(|error| *error);
        let state = match self {
            Self::Plain(_) => execution,
            Self::State(batch) => batch.finish_after(execution).map_err(|_| {
                execution
                    .err()
                    .unwrap_or(forja_core::BackendError::ExecutionFailed)
            }),
        };
        taint.finish(
            result.and_then(|value| state.map(|()| value)),
            true,
            timeout,
        )
    }
}

impl<B: Backend + 'static> Host<B> {
    fn lane_state(
        &self,
        pool: u32,
        lane: u32,
    ) -> Result<Option<crate::StateSequenceHandle<B>>, compute::Error> {
        let slot = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?
            .slot;
        slot.and_then(|slot| self.state_slots.get(&slot))
            .map(|state| state.sequence_handle(usize::try_from(lane).map_err(invalid_handle)?))
            .transpose()
            .map(Option::flatten)
    }

    pub(crate) fn state_lane_checkpoint(
        &self,
        pool: u32,
        lane: u32,
    ) -> Result<Option<crate::StateLaneCheckpoint>, compute::Error> {
        self.lane_state(pool, lane)?
            .map(|state| state.checkpoint_latest().map_err(invalid_handle))
            .transpose()
    }

    pub(crate) fn restore_lane_checkpoint(
        &mut self,
        pool: u32,
        lane: u32,
        state: Option<&crate::StateLaneCheckpoint>,
        paged: forja_core::SequenceCheckpoint,
    ) -> Result<(), compute::Error> {
        let Some(sequence) = self.lane_state(pool, lane)? else {
            return self.rollback_paged_lane_append(pool, lane, paged);
        };
        let state = state
            .filter(|state| state.position().tokens == paged.logical_len())
            .ok_or_else(|| invalid_handle("paged and state checkpoints differ"))?;
        let prefix = self.paged_lane_prefix(pool, lane)?;
        sequence
            .restore_with(state, || {
                prefix
                    .sequence
                    .lock()
                    .map_err(|_| forja_core::PagePoolError::CorruptState)?
                    .rollback(paged)
            })
            .map_err(invalid_handle)?;
        self.mark_paged_tables_dirty(pool, lane)
    }

    fn mark_paged_tables_dirty(&self, pool: u32, lane: u32) -> Result<(), compute::Error> {
        let slot = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("paged slot is missing"))?;
        let binding = &self
            .table
            .get(&Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?
            .binding;
        for dirty in &binding
            .lane(usize::try_from(lane).map_err(invalid_handle)?)?
            .table_dirty
        {
            dirty.store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }
}
