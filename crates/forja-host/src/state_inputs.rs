mod logits;
mod recovery;
mod update;

use crate::{Backend, Host, compute, invalid_handle, paged_storage};
use std::sync::atomic::AtomicBool;

// Table ownership and cloned layouts remain live until the scalar allocation retires.
const FEEDBACK_REFERENCE_BYTES: u64 = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SlotTokenSource {
    pub(crate) slot: u32,
    pub(crate) lane: u32,
}

pub(crate) struct StateInputBank {
    pub(crate) prompt_phase: AtomicBool,
    pub(crate) sampling: u32,
    pub(crate) tokens: u32,
    pub(crate) explicit: u32,
    pub(crate) prompt: u32,
    pub(crate) allow: u32,
    pub(crate) logits: u32,
    pub(crate) selected: u32,
    pub(crate) verify: Option<VerifyInputBank>,
}

pub(crate) struct VerifyInputBank {
    pub(crate) rows: u32,
    pub(crate) tokens: u32,
    pub(crate) logits: u32,
    pub(crate) selected: u32,
}

impl VerifyInputBank {
    fn handles(&self) -> [u32; 3] {
        [self.tokens, self.logits, self.selected]
    }
}

impl StateInputBank {
    fn lane_handles(&self) -> [u32; 7] {
        [
            self.sampling,
            self.tokens,
            self.explicit,
            self.prompt,
            self.allow,
            self.logits,
            self.selected,
        ]
    }

    pub(crate) fn handles(&self) -> impl Iterator<Item = u32> + '_ {
        self.lane_handles()
            .into_iter()
            .chain(self.verify.iter().flat_map(VerifyInputBank::handles))
    }
}

pub(crate) struct StateInputs {
    pub(crate) banks: [StateInputBank; 2],
    pub(crate) feedback: Vec<u32>,
    pub(crate) feedback_versions: crate::state::feedback::FeedbackBinding,
}

impl StateInputs {
    pub(crate) fn handles(&self) -> impl Iterator<Item = u32> + '_ {
        self.banks
            .iter()
            .flat_map(StateInputBank::handles)
            .chain(self.feedback.iter().copied())
    }
}

impl crate::TensorEntry {
    pub(crate) fn replay_layout(
        &self,
        values: &forja_core::ParamValues,
    ) -> Result<forja_core::Layout, compute::Error> {
        self.symbolic.as_ref().map_or_else(
            || Ok(self.tensor.layout().clone()),
            |layout| {
                layout
                    .instantiate(values)
                    .map_err(|error| crate::graph_error(&error))
            },
        )
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn validate_decode_output_roles(
        &self,
        slot: u32,
        dispatches: &[(usize, bool, forja_core::Dispatch)],
    ) -> Result<(), compute::Error> {
        let slot = self
            .table
            .get(&wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        let bank = &slot
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("decode reference inputs are missing"))?
            .banks[0];
        for handle in [bank.logits, bank.selected] {
            let role = &self
                .entry(&wasmtime::component::Resource::new_borrow(handle))?
                .tensor;
            let mut shape = role.layout().shape().to_vec();
            shape[0] = 1;
            let layout = forja_core::Layout::new(
                role.layout().dtype(),
                role.layout().offset(),
                shape,
                role.layout().strides().to_vec(),
                role.buffer().byte_len(),
            )
            .map_err(crate::guest_error)?;
            if !dispatches.iter().any(|(_, _, dispatch)| {
                dispatch
                    .outputs()
                    .iter()
                    .any(|output| output.buffer() == role.buffer() && output.layout() == &layout)
            }) {
                return Err(invalid_handle(
                    "decode reference must write the slot logits and selected output roles",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_slot_capture_roles(
        &self,
        commands: &crate::CommandListEntry,
    ) -> Result<(), compute::Error> {
        if commands
            .slot_capture
            .as_ref()
            .is_some_and(crate::slot_capture::SlotCapture::single_lane)
            && commands.paged.iter().any(|usage| usage.lane != 0)
        {
            return Err(invalid_handle(
                "single-lane capture requires logical paged lane zero",
            ));
        }
        for entry in &commands.retained {
            let Some(source) = entry.slot_token else {
                continue;
            };
            let scope = commands.slot_capture.as_ref().ok_or_else(|| {
                invalid_handle("token sources require a slot capture declaration")
            })?;
            let slot = self
                .table
                .get(
                    &wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(
                        source.slot,
                    ),
                )
                .map_err(invalid_handle)?;
            if source.slot != scope.slot
                || (scope.single_lane() && source.lane != 0)
                || usize::try_from(source.lane).map_err(invalid_handle)? >= slot.binding.lanes.len()
                || entry.tensor.layout().element_count() != 1
                || entry.tensor.is_writable()
            {
                return Err(invalid_handle(
                    "captured token source differs from its slot",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn bind_graph_token_sources(
        &mut self,
        entries: &[crate::TensorEntry],
        values: &forja_core::ParamValues,
        remap: Option<crate::slot_capture::PromptReplayLane>,
        bindings: &mut forja_core::ReplayBindings,
    ) -> Result<Vec<crate::TensorEntry>, compute::Error> {
        let mut seen = std::collections::HashSet::new();
        let mut retained = Vec::new();
        for entry in entries {
            let Some(source) = entry.slot_token else {
                continue;
            };
            let layout = entry.replay_layout(values)?;
            if !seen.insert((
                entry.tensor.buffer(),
                layout.offset(),
                layout.shape().to_vec(),
                layout.strides().to_vec(),
            )) {
                continue;
            }
            let old = self
                .backend
                .tensor(entry.tensor.buffer(), layout.clone())
                .map_err(crate::guest_error)?
                .read_only();
            let lane = crate::slot_capture::PromptReplayLane::map(
                remap,
                source.slot,
                usize::try_from(source.lane).map_err(invalid_handle)?,
            );
            let loan =
                self.state_input_token(source.slot, u32::try_from(lane).map_err(invalid_handle)?)?;
            let fresh = self.entry(&loan).cloned();
            self.drop_tensor(loan)?;
            let mut fresh = fresh?;
            let layout = forja_core::Layout::new(
                layout.dtype(),
                fresh.tensor.layout().offset(),
                layout.shape().to_vec(),
                layout.strides().to_vec(),
                fresh.tensor.buffer().byte_len(),
            )
            .map_err(crate::guest_error)?;
            fresh.tensor = self
                .backend
                .tensor(fresh.tensor.buffer(), layout)
                .map_err(crate::guest_error)?
                .read_only();
            fresh.symbolic = None;
            bindings
                .view_revalidated(old, fresh.tensor.clone())
                .map_err(crate::guest_error)?;
            retained.push(fresh);
        }
        Ok(retained)
    }

    pub(crate) fn state_input_token(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<wasmtime::component::Resource<crate::TensorEntry>, compute::Error> {
        self.state_input_token_in_bank(slot, lane, None)
    }

    pub(crate) fn state_input_token_in_bank(
        &mut self,
        slot: u32,
        lane: u32,
        bank: Option<usize>,
    ) -> Result<wasmtime::component::Resource<crate::TensorEntry>, compute::Error> {
        use wasmtime::component::Resource;
        let entry = self
            .table
            .get(&Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        if usize::try_from(lane).map_err(invalid_handle)? >= entry.binding.lanes.len() {
            return Err(invalid_handle("input token lane is out of range"));
        }
        let inputs = entry
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("slot has no state inputs"))?;
        let source = if bank.is_some() {
            None
        } else {
            self.state_slots
                .get(&slot)
                .ok_or_else(|| invalid_handle("slot has no protected state"))?
                .feedback_source(
                    entry.active_bank,
                    usize::try_from(lane).map_err(invalid_handle)?,
                )?
        };
        let (handle, start) = if let Some(index) = source {
            (
                *inputs
                    .feedback
                    .get(usize::try_from(index).map_err(invalid_handle)?)
                    .ok_or_else(|| invalid_handle("feedback source is missing"))?,
                0,
            )
        } else {
            (
                inputs
                    .banks
                    .get(bank.unwrap_or(entry.active_bank))
                    .ok_or_else(|| invalid_handle("token input bank is out of range"))?
                    .tokens,
                lane,
            )
        };
        let token = self.lend_view(
            &Resource::new_borrow(handle),
            compute::ViewOp::Slice(vec![compute::SliceSpec {
                start,
                len: 1,
                step: 1,
            }]),
            true,
        )?;
        self.table
            .get_mut(&token)
            .map_err(invalid_handle)?
            .slot_token = Some(SlotTokenSource { slot, lane });
        Ok(token)
    }

    pub(crate) fn alias_paged_tensors(
        &mut self,
        handles: &[u32],
    ) -> Result<Vec<wasmtime::component::Resource<crate::TensorEntry>>, compute::Error> {
        use wasmtime::component::Resource;
        let mut aliases = Vec::with_capacity(handles.len());
        for &handle in handles {
            let result = (|| {
                let resource = Resource::new_borrow(handle);
                let shape = self.entry(&resource)?.tensor.layout().shape().to_vec();
                self.lend_view(&resource, compute::ViewOp::Reshape(shape), false)
            })();
            match result {
                Ok(alias) => aliases.push(alias),
                Err(error) => {
                    for alias in aliases {
                        let _ = self.drop_tensor(alias);
                    }
                    return Err(error);
                }
            }
        }
        Ok(aliases)
    }

    pub(crate) fn slot_tensors(
        &mut self,
        slot: u32,
        bank: Option<usize>,
    ) -> Result<paged_storage::SlotTensors, compute::Error> {
        let resource = wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(slot);
        let (lanes, handles) = {
            let slot = self.table.get(&resource).map_err(invalid_handle)?;
            let selected = bank.unwrap_or(slot.active_bank);
            let [logical_lengths, query_starts, absolute_positions, _activity] = *slot
                .metadata
                .get(selected)
                .ok_or_else(|| invalid_handle("slot input bank is out of range"))?;
            (
                u32::try_from(slot.binding.lanes.len()).map_err(|_| {
                    compute::Error::Layout("paged lane count exceeds u32".to_owned())
                })?,
                [logical_lengths, query_starts, absolute_positions],
            )
        };
        let [logical_lengths, query_starts, absolute_positions] =
            self.alias_paged_array(handles)?;
        let inputs = match self.state_input_tensors(resource.rep(), bank) {
            Ok(inputs) => inputs,
            Err(error) => {
                for alias in [logical_lengths, query_starts, absolute_positions] {
                    let _ = self.drop_tensor(alias);
                }
                return Err(error);
            }
        };
        Ok(paged_storage::SlotTensors {
            lanes,
            logical_lengths,
            query_starts,
            absolute_positions,
            inputs,
        })
    }

    fn alias_paged_array<const N: usize>(
        &mut self,
        handles: [u32; N],
    ) -> Result<[wasmtime::component::Resource<crate::TensorEntry>; N], compute::Error> {
        let aliases = self.alias_paged_tensors(&handles)?;
        aliases.try_into().map_err(
            |aliases: Vec<wasmtime::component::Resource<crate::TensorEntry>>| {
                for alias in aliases {
                    let _ = self.drop_tensor(alias);
                }
                invalid_handle("paged alias count differs from its handles")
            },
        )
    }

    pub(crate) fn validate_verify_storage(
        &self,
        pool: u32,
        rows: u32,
    ) -> Result<(), compute::Error> {
        let missing = || invalid_handle("verify shapes require host row storage");
        let pool = self
            .table
            .get(&wasmtime::component::Resource::<crate::PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?;
        let slot = pool.slot.ok_or_else(missing)?;
        let slot = self
            .table
            .get(&wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        let inputs = slot.inputs.as_ref().ok_or_else(missing)?;
        if inputs.banks.iter().any(|bank| {
            bank.verify
                .as_ref()
                .is_none_or(|verify| verify.rows != rows)
        }) {
            return Err(missing());
        }
        Ok(())
    }

    pub(crate) fn state_input_tensors(
        &mut self,
        slot: u32,
        bank: Option<usize>,
    ) -> Result<Option<paged_storage::StateInputTensors>, compute::Error> {
        let entry = self
            .table
            .get(&wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        let Some(inputs) = &entry.inputs else {
            return Ok(None);
        };
        let bank = inputs
            .banks
            .get(bank.unwrap_or(entry.active_bank))
            .ok_or_else(|| invalid_handle("state input bank is out of range"))?;
        let ordinary = bank.lane_handles();
        let verify = bank
            .verify
            .as_ref()
            .map(|verify| (verify.rows, verify.handles()));
        let layout = entry
            .input_layout
            .as_ref()
            .ok_or_else(|| invalid_handle("state input layout is missing"))?;
        let (vocab, prompt_tokens) = (layout.vocab, layout.prompt_tokens);
        let (ordinary, verify) = match verify {
            None => (self.alias_paged_array(ordinary)?, None),
            Some((rows, [verify_tokens, verify_logits, verify_selected])) => {
                let [sampling, tokens, explicit, prompt, allow, logits, selected] = ordinary;
                let [
                    sampling,
                    tokens,
                    explicit,
                    prompt,
                    allow,
                    logits,
                    selected,
                    verify_tokens,
                    verify_logits,
                    verify_selected,
                ] = self.alias_paged_array([
                    sampling,
                    tokens,
                    explicit,
                    prompt,
                    allow,
                    logits,
                    selected,
                    verify_tokens,
                    verify_logits,
                    verify_selected,
                ])?;
                (
                    [sampling, tokens, explicit, prompt, allow, logits, selected],
                    Some(paged_storage::VerifyInputTensors {
                        rows,
                        tokens: verify_tokens,
                        logits: verify_logits,
                        selected: verify_selected,
                    }),
                )
            }
        };
        let [sampling, tokens, explicit, prompt, allow, logits, selected] = ordinary;
        Ok(Some(paged_storage::StateInputTensors {
            vocab,
            prompt_tokens,
            sampling,
            tokens,
            explicit,
            prompt,
            allow,
            logits,
            selected,
            verify,
        }))
    }

    #[cfg(test)]
    pub(crate) fn allocate_state_inputs(
        &mut self,
        lanes: u32,
        layout: paged_storage::StateInputLayout,
    ) -> Result<StateInputs, compute::Error> {
        self.allocate_state_inputs_with_spares(lanes, layout, 0)
    }

    pub(crate) fn allocate_state_inputs_with_spares(
        &mut self,
        lanes: u32,
        layout: paged_storage::StateInputLayout,
        spares: u32,
    ) -> Result<StateInputs, compute::Error> {
        let scalars =
            crate::state::feedback::FeedbackBinding::scalar_count_with_spares(lanes, spares)
                .map_err(invalid_handle)?;
        let available = self
            .limits
            .live_tensor_handles
            .checked_sub(self.live_handles)
            .ok_or_else(|| crate::quota("tensor handle quota is exhausted"))?;
        if usize::try_from(scalars).map_err(invalid_handle)? > available {
            return Err(crate::quota(
                "feedback scalar count exceeds the tensor handle quota",
            ));
        }
        let feedback_versions =
            crate::state::feedback::FeedbackBinding::with_spares(self, lanes, spares)?;
        let mut handles = Vec::new();
        let result = (|| {
            let mut alloc = |dtype, shape: &[u32]| {
                let handle = self.alloc_paged_tensor(dtype, shape)?;
                handles.push(handle);
                Ok::<_, compute::Error>(handle)
            };
            let mut bank = || {
                Ok::<_, compute::Error>(StateInputBank {
                    prompt_phase: AtomicBool::new(false),
                    sampling: alloc(compute::Dtype::U32, &[lanes, 5])?,
                    tokens: alloc(compute::Dtype::U32, &[lanes])?,
                    explicit: alloc(compute::Dtype::U32, &[lanes])?,
                    prompt: alloc(compute::Dtype::U32, &[layout.prompt_tokens])?,
                    allow: alloc(compute::Dtype::U32, &[lanes, layout.vocab.div_ceil(32)])?,
                    logits: alloc(compute::Dtype::F32, &[lanes, layout.vocab])?,
                    selected: alloc(compute::Dtype::U32, &[lanes])?,
                    verify: if layout.verify_rows == 0 {
                        None
                    } else {
                        Some(VerifyInputBank {
                            rows: layout.verify_rows,
                            tokens: alloc(compute::Dtype::U32, &[layout.verify_rows])?,
                            logits: alloc(
                                compute::Dtype::F32,
                                &[layout.verify_rows, layout.vocab],
                            )?,
                            selected: alloc(compute::Dtype::U32, &[layout.verify_rows])?,
                        })
                    },
                })
            };
            let banks = [bank()?, bank()?];
            let feedback = (0..scalars)
                .map(|_| {
                    let tensor = self.alloc_paged_tensor_charged(
                        compute::Dtype::U32,
                        &[1],
                        FEEDBACK_REFERENCE_BYTES,
                    )?;
                    handles.push(tensor);
                    Ok::<_, compute::Error>(tensor)
                })
                .collect::<Result<_, _>>()?;
            Ok(StateInputs {
                banks,
                feedback,
                feedback_versions,
            })
        })();
        if result.is_err() {
            self.drop_paged_handles(handles);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use forja_cpu::CpuBackend;
    use std::sync::atomic::Ordering;

    #[test]
    fn state_input_banks_refund_every_partial_allocation() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(65536, 8, 4096, 64, 4096));
        let layout = paged_storage::StateInputLayout {
            vocab: 33,
            prompt_tokens: 7,
            verify_rows: 3,
        };
        let inputs = host.allocate_state_inputs(2, layout).unwrap();
        let bytes = host.live_bytes.load(Ordering::Acquire);
        assert_eq!(
            inputs.feedback.len(),
            usize::try_from(crate::state::feedback::FeedbackBinding::scalar_count(2).unwrap())
                .unwrap()
        );
        assert_ne!(inputs.banks[0].logits, inputs.banks[1].logits);
        let mut buffers = std::collections::HashSet::new();
        for bank in &inputs.banks {
            let verify = bank.verify.as_ref().unwrap();
            for (handle, shape) in [
                (verify.tokens, vec![3]),
                (verify.logits, vec![3, 33]),
                (verify.selected, vec![3]),
                (bank.logits, vec![2, 33]),
                (bank.selected, vec![2]),
            ] {
                let tensor = &host
                    .entry(&wasmtime::component::Resource::new_borrow(handle))
                    .unwrap()
                    .tensor;
                assert_eq!(tensor.layout().shape(), shape);
                assert!(buffers.insert(tensor.buffer()));
            }
        }
        host.drop_paged_handles(inputs.handles().collect());
        drop(inputs);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        for limit in (0..bytes).step_by(4) {
            host.limits.live_bytes = limit;
            assert!(host.allocate_state_inputs(2, layout).is_err());
            assert_eq!(host.live_bytes.load(Ordering::Acquire), 0, "limit={limit}");
            assert_eq!(host.live_handles, 0, "limit={limit}");
        }
    }

    #[test]
    fn verify_allocation_failure_refunds_bytes_and_handles() {
        let layout = paged_storage::StateInputLayout {
            vocab: 33,
            prompt_tokens: 7,
            verify_rows: 3,
        };
        let mut host = Host::new(CpuBackend::new(), Limits::new(65536, 8, 4096, 64, 4096));
        let inputs = host.allocate_state_inputs(2, layout).unwrap();
        let required = host.live_bytes.load(Ordering::Acquire);
        host.drop_paged_handles(inputs.handles().collect());
        drop(inputs);
        host.limits.live_bytes = required - 1;
        assert!(matches!(
            host.allocate_state_inputs(2, layout),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        assert_eq!(host.live_handles, 0);
    }

    #[test]
    fn spare_feedback_handles_share_the_existing_handle_quota() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(65536, 8, 4096, 64, 4096));
        let layout = paged_storage::StateInputLayout {
            vocab: 33,
            prompt_tokens: 7,
            verify_rows: 0,
        };
        let inputs = host
            .allocate_state_inputs_with_spares(2, layout, 1)
            .unwrap();
        assert_eq!(inputs.feedback.len(), 11);
        host.drop_paged_handles(inputs.handles().collect());
        drop(inputs);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        host.limits.live_tensor_handles = 10;
        assert!(
            host.allocate_state_inputs_with_spares(2, layout, 1)
                .is_err()
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
    #[tokio::test]
    async fn failed_feedback_write_preserves_other_versions() {
        use crate::{BackendError, SubmissionAccess};
        use std::time::Duration;
        use wasmtime::component::Resource;
        let mut host = Host::new(CpuBackend::new(), Limits::new(65536, 8, 4096, 64, 4096));
        let inputs = host
            .allocate_state_inputs(
                2,
                paged_storage::StateInputLayout {
                    vocab: 33,
                    prompt_tokens: 7,
                    verify_rows: 0,
                },
            )
            .unwrap();
        let source = Resource::new_borrow(inputs.feedback[0]);
        let failed = Resource::new_borrow(inputs.feedback[1]);
        host.write(&source, &17_u32.to_ne_bytes()).unwrap();
        let failed_buffer = host.entry(&failed).unwrap().tensor.buffer();
        let mut access = SubmissionAccess::default();
        access.writes.insert(failed_buffer);
        let flight = host.taints.begin(&access).unwrap();
        assert!(
            flight
                .finish::<()>(Err(BackendError::ExecutionFailed), true, Duration::ZERO)
                .is_err()
        );
        assert!(host.prepare_read(&failed).unwrap().run().await.is_err());
        assert_eq!(
            host.prepare_read(&source).unwrap().run().await.unwrap(),
            17_u32.to_ne_bytes()
        );
        host.drop_paged_handles(inputs.handles().collect());
        drop(inputs);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
    #[test]
    fn excessive_feedback_reference_counts_refuse_before_allocating() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(65536, 8, 4096, 64, 4096));
        assert!(matches!(
            host.allocate_state_inputs(
                33,
                paged_storage::StateInputLayout {
                    vocab: 1,
                    prompt_tokens: 7,
                    verify_rows: 0,
                }
            ),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
}
