mod update;

use crate::{Backend, Host, compute, invalid_handle, paged_storage};

pub(crate) struct StateInputBank {
    pub(crate) sampling: u32,
    pub(crate) tokens: u32,
    pub(crate) explicit: u32,
    pub(crate) prompt: u32,
    pub(crate) allow: u32,
    pub(crate) logits: u32,
    pub(crate) selected: u32,
}

impl StateInputBank {
    pub(crate) const fn handles(&self) -> [u32; 7] {
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

impl<B: Backend + 'static> Host<B> {
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
                self.view(&resource, compute::ViewOp::Reshape(shape))
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

    pub(crate) fn state_input_tensors(
        &mut self,
        slot: u32,
    ) -> Result<Option<paged_storage::StateInputTensors>, compute::Error> {
        let entry = self
            .table
            .get(&wasmtime::component::Resource::<crate::BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        let Some(inputs) = &entry.inputs else {
            return Ok(None);
        };
        let handles = inputs.banks[entry.active_bank].handles();
        let layout = entry
            .input_layout
            .as_ref()
            .ok_or_else(|| invalid_handle("state input layout is missing"))?;
        let (vocab, prompt_tokens) = (layout.vocab, layout.prompt_tokens);
        let [sampling, tokens, explicit, prompt, allow, logits, selected] = self
            .alias_paged_tensors(&handles)?
            .try_into()
            .map_err(|_| invalid_handle("state input aliases are incomplete"))?;
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
        }))
    }

    pub(crate) fn allocate_state_inputs(
        &mut self,
        lanes: u32,
        layout: paged_storage::StateInputLayout,
    ) -> Result<StateInputs, compute::Error> {
        let scalars = lanes
            .checked_mul(5)
            .ok_or_else(|| invalid_handle("state feedback count overflowed"))?;
        let feedback_versions = crate::state::feedback::FeedbackBinding::new(self, lanes)?;
        let mut handles = Vec::new();
        let result = (|| {
            let mut alloc = |dtype, shape: &[u32]| {
                let handle = self.alloc_paged_tensor(dtype, shape)?;
                handles.push(handle);
                Ok::<_, compute::Error>(handle)
            };
            let mut bank = || {
                Ok::<_, compute::Error>(StateInputBank {
                    sampling: alloc(compute::Dtype::U32, &[lanes, 5])?,
                    tokens: alloc(compute::Dtype::U32, &[lanes])?,
                    explicit: alloc(compute::Dtype::U32, &[lanes])?,
                    prompt: alloc(compute::Dtype::U32, &[layout.prompt_tokens])?,
                    allow: alloc(compute::Dtype::U32, &[lanes, layout.vocab.div_ceil(32)])?,
                    logits: alloc(compute::Dtype::F32, &[lanes, layout.vocab])?,
                    selected: alloc(compute::Dtype::U32, &[lanes])?,
                })
            };
            let banks = [bank()?, bank()?];
            let feedback = (0..scalars)
                .map(|_| alloc(compute::Dtype::U32, &[1]))
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
        let mut host = Host::new(CpuBackend::new(), Limits::new(4096, 8, 4096, 64, 4096));
        let layout = paged_storage::StateInputLayout {
            vocab: 33,
            prompt_tokens: 7,
        };
        let inputs = host.allocate_state_inputs(2, layout).unwrap();
        let bytes = host.live_bytes.load(Ordering::Acquire);
        assert_ne!(inputs.banks[0].logits, inputs.banks[1].logits);
        host.drop_paged_handles(inputs.handles().collect());
        drop(inputs);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        for limit in (0..bytes).step_by(4) {
            host.limits.live_bytes = limit;
            assert!(host.allocate_state_inputs(2, layout).is_err());
            assert_eq!(host.live_bytes.load(Ordering::Acquire), 0, "limit={limit}");
        }
    }
    #[tokio::test]
    async fn failed_feedback_write_preserves_other_versions() {
        use crate::{BackendError, SubmissionAccess};
        use std::time::Duration;
        use wasmtime::component::Resource;
        let mut host = Host::new(CpuBackend::new(), Limits::new(4096, 8, 4096, 64, 4096));
        let inputs = host
            .allocate_state_inputs(
                2,
                paged_storage::StateInputLayout {
                    vocab: 33,
                    prompt_tokens: 7,
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
}
