use super::StateInputBank;
use crate::{Backend, Host, compute, guest_error, invalid_handle, state::StateCharge};
use std::sync::Arc;
use wasmtime::component::Resource;

impl<B: Backend + 'static> Host<B> {
    pub(super) fn recover_state_input_bank(
        &self,
        bank: &StateInputBank,
    ) -> Result<(), compute::Error> {
        self.recover_state_input_handles(bank.handles())
    }

    pub(super) fn recover_state_input_handles(
        &self,
        handles: impl IntoIterator<Item = u32>,
    ) -> Result<(), compute::Error> {
        let _order = self
            .taints
            .queue
            .lock()
            .map_err(|_| invalid_handle("submission order is poisoned"))?;
        for handle in handles {
            let entry = self.entry(&Resource::new_borrow(handle))?;
            let tensor = &entry.tensor;
            let buffer = tensor.buffer();
            if self.taints.check(buffer).is_ok() {
                continue;
            }
            self.taints
                .drain_writes(buffer, self.limits.submission_timeout)
                .map_err(guest_error)?;
            let bytes = buffer.byte_len();
            let _charge = StateCharge::reserve(
                self.backend.as_ref(),
                Arc::clone(&self.live_bytes),
                self.limits.live_bytes,
                bytes,
            )?;
            self.backend
                .write(
                    tensor,
                    &vec![0; usize::try_from(bytes).map_err(invalid_handle)?],
                )
                .map_err(guest_error)?;
            self.taints.release(buffer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendError, Limits, SubmissionAccess, paged_storage};
    use std::time::Duration;

    #[tokio::test]
    async fn bank_recovery_preserves_other_banks_and_feedback() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 64, 4096),
        );
        let inputs = host
            .allocate_state_inputs(
                2,
                paged_storage::StateInputLayout {
                    vocab: 33,
                    prompt_tokens: 7,
                },
            )
            .unwrap();
        let peer = Resource::new_borrow(inputs.banks[1].tokens);
        host.write(
            &peer,
            &[17_u32, 23]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let failed = Resource::new_borrow(inputs.banks[0].tokens);
        let feedback = Resource::new_borrow(inputs.feedback[0]);
        let mut access = SubmissionAccess::default();
        for resource in [&failed, &feedback] {
            access
                .writes
                .insert(host.entry(resource).unwrap().tensor.buffer());
        }
        assert!(
            host.taints
                .begin(&access)
                .unwrap()
                .finish::<()>(Err(BackendError::ExecutionFailed), true, Duration::ZERO)
                .is_err()
        );
        host.recover_state_input_bank(&inputs.banks[0]).unwrap();
        assert_eq!(
            host.prepare_read(&failed).unwrap().run().await.unwrap(),
            [0; 8]
        );
        assert_eq!(
            host.prepare_read(&peer).unwrap().run().await.unwrap(),
            [17_u32, 23]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert!(host.prepare_read(&feedback).unwrap().run().await.is_err());
    }
}
