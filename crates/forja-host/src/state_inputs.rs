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
    pub(crate) feedback: u32,
}

impl StateInputs {
    pub(crate) fn handles(&self) -> impl Iterator<Item = u32> + '_ {
        self.banks
            .iter()
            .flat_map(StateInputBank::handles)
            .chain([self.feedback])
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn allocate_state_inputs(
        &mut self,
        lanes: u32,
        layout: paged_storage::StateInputLayout,
    ) -> Result<StateInputs, compute::Error> {
        let scalars = lanes
            .checked_mul(5)
            .ok_or_else(|| invalid_handle("state feedback count overflowed"))?;
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
            Ok(StateInputs {
                banks,
                feedback: alloc(compute::Dtype::U32, &[scalars])?,
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
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        for limit in (0..bytes).step_by(4) {
            host.limits.live_bytes = limit;
            assert!(host.allocate_state_inputs(2, layout).is_err());
            assert_eq!(host.live_bytes.load(Ordering::Acquire), 0, "limit={limit}");
        }
    }
}
