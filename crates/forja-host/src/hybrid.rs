use crate::{Host, PagePoolEntry, StateSlots, compute, invalid_handle, paged_storage};
use forja_core::{Backend, StateLayout, StatePlane};
use wasmtime::component::Resource;

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn open_state_slot(
        &mut self,
        pool: &Resource<PagePoolEntry>,
        planes: u32,
        descriptors: Vec<paged_storage::StatePlaneDescriptor>,
        inputs: paged_storage::StateInputLayout,
    ) -> Result<Resource<crate::BatchSlotEntry>, compute::Error> {
        let entry = self.table.get(pool).map_err(invalid_handle)?;
        if inputs.vocab == 0
            || inputs.prompt_tokens == 0
            || inputs.prompt_tokens > entry.max_context
        {
            return Err(invalid_handle("invalid state input layout"));
        }
        let lanes = entry.active_lanes;
        self.limits.check_tensor_shape(&[lanes, inputs.vocab])?;
        self.limits.check_tensor_shape(&[inputs.prompt_tokens])?;
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
        let backing = self.allocate_state_inputs(lanes, inputs)?;
        let state = state.with_feedback(backing.feedback_versions.clone());
        let slot = match self.open_paged_slot(pool, planes) {
            Ok(slot) => slot,
            Err(error) => {
                self.drop_paged_handles(backing.handles().collect());
                return Err(error);
            }
        };
        let entry = self.table.get_mut(&slot).map_err(invalid_handle)?;
        entry.input_layout = Some(inputs);
        entry.inputs = Some(backing);
        self.state_slots.insert(slot.rep(), state);
        Ok(slot)
    }
}

impl crate::BatchSlotEntry {
    fn validate_state_span(
        &self,
        bank: usize,
        lane: usize,
        active: bool,
        tokens: u64,
    ) -> Result<(), compute::Error> {
        if tokens
            > self
                .input_layout
                .as_ref()
                .ok_or_else(|| invalid_handle("state input layout is missing"))?
                .prompt_tokens
                .into()
        {
            return Err(invalid_handle("state update exceeds its prompt budget"));
        }
        if active {
            let window = self
                .binding
                .banks
                .append_window(bank, lane)
                .map_err(invalid_handle)?;
            if tokens == 0 || window.end.checked_sub(window.start) != Some(tokens) {
                return Err(invalid_handle(
                    "state update differs from its admitted token span",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateUse {
    pub(crate) slot: u32,
    pub(crate) bank: usize,
    pub(crate) lane: usize,
    pub(crate) active: bool,
    pub(crate) tokens: u64,
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
        let tokens = match operation {
            paged_storage::StateOperation::DeltaRule => 1,
            _ => inputs
                .first()
                .and_then(|entry| entry.tensor.layout().shape().first())
                .copied()
                .map_or(0, u64::from),
        };
        slot_entry.validate_state_span(bank, lane, active, tokens)?;
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
        entry
            .access
            .record_conditional(&inputs, std::slice::from_ref(&output));
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
            tokens,
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
        let binding = std::sync::Arc::clone(
            &self
                .table
                .get(&Resource::<crate::BatchSlotEntry>::new_borrow(first.slot))
                .map_err(invalid_handle)?
                .binding,
        );
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
        for usage in usages.iter().filter(|usage| usage.active) {
            binding
                .lane(usage.lane)?
                .used
                .store(true, std::sync::atomic::Ordering::Release);
        }
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

    pub(crate) fn mark_paged_tables_dirty(
        &self,
        pool: u32,
        lane: u32,
    ) -> Result<(), compute::Error> {
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

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn drain_slot_state(
        &mut self,
        slot: u32,
        lane: usize,
    ) -> Result<(), compute::Error> {
        if let Some(state) = self.state_slots.get_mut(&slot) {
            for bank in 0..2 {
                state.cancel_lane(bank, lane)?;
            }
            if let Some(sequence) = state.sequence_handle(lane)? {
                sequence
                    .wait_drained(self.limits.submission_timeout)
                    .map_err(crate::guest_error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn rewind_slot_state(
        &self,
        slot: u32,
        lane: usize,
        tokens: u64,
    ) -> Result<(), compute::Error> {
        if let Some(state) = self.state_slots.get(&slot)
            && let Some(sequence) = state.sequence_handle(lane)?
        {
            let current = sequence.position().tokens;
            if current < tokens {
                return Err(invalid_handle(
                    "state rewind exceeds its committed boundary",
                ));
            }
            if current > tokens {
                sequence.rewind(tokens).map_err(invalid_handle)?;
            }
        }
        Ok(())
    }

    pub(crate) fn bind_slot_state(
        &mut self,
        slot: u32,
        lane: usize,
        release: bool,
        binding: &crate::PagedBinding,
        previous: [forja_core::PagedSlot; 2],
        reserved_tokens: u64,
    ) -> Result<(), compute::Error> {
        if let Some(state) = self.state_slots.get_mut(&slot) {
            let result = if release {
                state.release(lane)
            } else {
                (|| {
                    let position = state
                        .sequence_handle(lane)?
                        .map(|sequence| sequence.latest_position())
                        .transpose()
                        .map_err(invalid_handle)?;
                    if position.is_some_and(|position| {
                        position.tokens != 0 || position.reserved_tokens != reserved_tokens
                    }) {
                        Err(crate::quota(crate::PAGED_LANE_RELEASE_REQUIRED))
                    } else {
                        state.admit_reserved(lane, reserved_tokens)
                    }
                })()
            };
            if let Err(error) = result {
                for dirty in &binding.lane(lane)?.table_dirty {
                    dirty.store(true, std::sync::atomic::Ordering::Release);
                }
                binding
                    .banks
                    .restore_all(lane, previous)
                    .map_err(crate::paged_refusal)?;
                return Err(error);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedState {
    pub(crate) usage: StateUse,
    pub(crate) capture: crate::StateCapture,
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn capture_graph_state(
        &mut self,
        usages: &[StateUse],
    ) -> Result<Vec<CapturedState>, compute::Error>
    where
        B: Send + Sync,
    {
        usages
            .iter()
            .map(|usage| {
                let state = self
                    .state_slots
                    .get_mut(&usage.slot)
                    .ok_or_else(|| invalid_handle("state slot is missing"))?;
                let capture = state
                    .commands(usage.bank, usage.lane, usage.active)?
                    .capture()?;
                Ok(CapturedState {
                    usage: *usage,
                    capture,
                })
            })
            .collect()
    }

    pub(crate) fn release_graph_state_scratch(
        &mut self,
        captures: &mut [CapturedState],
    ) -> Result<(), compute::Error> {
        for captured in captures {
            let usage = captured.usage;
            self.state_slots
                .get_mut(&usage.slot)
                .ok_or_else(|| invalid_handle("state slot is missing"))?
                .commands(usage.bank, usage.lane, usage.active)?
                .release_graph_scratch();
            captured.capture.release_graph_scratch();
        }
        Ok(())
    }
}

struct BoundState {
    retained: Vec<crate::TensorEntry>,
    bindings: forja_core::ReplayBindings,
    usages: Vec<StateUse>,
    banks: Vec<(std::sync::Arc<crate::PagedBinding>, usize)>,
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn prepare_state_replay_with_flight(
        &mut self,
        resource: &Resource<crate::GraphEntry>,
        values: Vec<u32>,
        flight: crate::ReplayFlight,
    ) -> Result<crate::SubmitRequest<B>, compute::Error> {
        let entry = self.table.get(resource).map_err(invalid_handle)?;
        if entry.captured_state.is_empty() {
            return self.prepare_replay_with_flight(resource, values, flight);
        }
        let mut retained = entry.retained.clone();
        let (graph, access, captures, mut paged, mut checks) = (
            std::sync::Arc::clone(&entry.graph),
            entry.access.clone(),
            entry.captured_state.clone(),
            entry.paged.clone(),
            entry.paged_bounds.clone(),
        );
        let checked = graph
            .values(values.clone())
            .map_err(|error| crate::graph_error(&error))?;
        let bound = self.bind_graph_state(&captures)?;
        retained.extend(bound.retained);
        for usage in paged
            .iter_mut()
            .chain(checks.iter_mut().map(|check| &mut check.usage))
        {
            if let Some((_, bank)) = bound
                .banks
                .iter()
                .find(|(binding, _)| std::sync::Arc::ptr_eq(binding, &usage.binding))
            {
                usage.bank = *bank;
            }
        }
        crate::PagedCheck::validate_all(&checks, Some(&checked))?;
        let (batch, mut owners) = self.prepare_state_work(&bound.usages)?;
        owners.extend(
            captures
                .iter()
                .flat_map(|captured| captured.capture.retained_resources()),
        );
        let mut request = self.submission_request(
            crate::SubmissionWork::Replay {
                graph,
                values,
                bindings: bound.bindings,
            },
            access,
            retained,
            paged,
            Some(flight),
        );
        request.state_batch = batch;
        request.state_owners = owners;
        request.captured_state = captures
            .into_iter()
            .map(|captured| captured.capture)
            .collect();
        Ok(request)
    }

    fn bind_graph_state(
        &mut self,
        captures: &[CapturedState],
    ) -> Result<BoundState, compute::Error> {
        let mut bound = BoundState {
            retained: Vec::new(),
            bindings: forja_core::ReplayBindings::default(),
            usages: Vec::new(),
            banks: Vec::new(),
        };
        let mut slots = std::collections::HashSet::new();
        for captured in captures {
            let original = captured.usage;
            let slot = self
                .table
                .get(&Resource::<crate::BatchSlotEntry>::new_borrow(
                    original.slot,
                ))
                .map_err(invalid_handle)?;
            let bank = slot.active_bank;
            if slot
                .input_layout
                .as_ref()
                .is_none_or(|inputs| original.tokens > u64::from(inputs.prompt_tokens))
            {
                return Err(invalid_handle(
                    "captured state exceeds the slot prompt budget",
                ));
            }
            if slots.insert(original.slot) {
                if captures.iter().any(|capture| {
                    capture.usage.slot == original.slot && capture.usage.bank != original.bank
                }) || slot.binding.activity[bank]
                    .iter()
                    .enumerate()
                    .any(|(lane, active)| {
                        active.load(std::sync::atomic::Ordering::Acquire)
                            && !captures.iter().any(|capture| {
                                capture.usage.slot == original.slot && capture.usage.lane == lane
                            })
                    })
                {
                    return Err(invalid_handle(
                        "graph lanes or captured banks differ from the submission",
                    ));
                }
                bound
                    .banks
                    .push((std::sync::Arc::clone(&slot.binding), bank));
                self.bind_slot_bank_buffers(
                    original.slot,
                    original.bank,
                    bank,
                    &mut bound.bindings,
                    &mut bound.retained,
                )?;
            }
            let active = slot.binding.activity[bank][original.lane]
                .load(std::sync::atomic::Ordering::Acquire);
            if active {
                let window = slot
                    .binding
                    .banks
                    .append_window(bank, original.lane)
                    .map_err(invalid_handle)?;
                if window.end.checked_sub(window.start) != Some(original.tokens) {
                    return Err(invalid_handle(
                        "captured state token extent differs from the append",
                    ));
                }
            }
            let state = self
                .state_slots
                .get_mut(&original.slot)
                .ok_or_else(|| invalid_handle("state slot is missing"))?;
            let bindings = captured
                .capture
                .bind(state.commands(bank, original.lane, active)?)?;
            bound.bindings.extend(bindings).map_err(invalid_handle)?;
            bound.usages.push(StateUse {
                bank,
                active,
                ..original
            });
        }
        Ok(bound)
    }

    fn bind_slot_bank_buffers(
        &self,
        slot: u32,
        original: usize,
        bank: usize,
        bindings: &mut forja_core::ReplayBindings,
        retained: &mut Vec<crate::TensorEntry>,
    ) -> Result<(), compute::Error> {
        let slot = self
            .table
            .get(&Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        for (old, new) in slot.tables[original]
            .iter()
            .chain(&slot.metadata[original])
            .zip(slot.tables[bank].iter().chain(&slot.metadata[bank]))
        {
            let old = self.entry(&Resource::new_borrow(*old))?.tensor.buffer();
            let new = self.entry(&Resource::new_borrow(*new))?.clone();
            bindings
                .buffer(old, new.tensor.buffer())
                .map_err(invalid_handle)?;
            retained.push(new);
        }
        Ok(())
    }
}
