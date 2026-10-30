use super::{StateCell, StateCharge, StateRecording};
use crate::{Limits, compute, guest_error, invalid_handle, quota};
use forja_core::{Backend, DeltaBlockScratch, StateOp, Tensor};
use std::sync::Arc;

pub(super) struct StateScratch<B: Backend> {
    backend: Arc<B>,
    tensor: Tensor,
    charge: Arc<StateCharge>,
}

impl<B: Backend> StateScratch<B> {
    pub(super) fn retained_resource(&self) -> forja_core::CompletionResource {
        forja_core::CompletionResource::new(Arc::clone(&self.charge))
    }
}

impl<B: Backend> Drop for StateScratch<B> {
    fn drop(&mut self) {
        let _ = self.backend.release(&self.tensor);
    }
}

impl<B: Backend + 'static> StateRecording<B> {
    /// Records fixed-size delta blocks with charged, private intermediate storage.
    ///
    /// # Errors
    /// Refuses invalid geometry, picks, quotas or a duplicate plane update.
    pub fn delta_rule_block(
        &mut self,
        plane: usize,
        inputs: [&Tensor; 5],
        output: &Tensor,
        variant: Option<&str>,
    ) -> Result<(), compute::Error> {
        let descriptor = self
            .storage
            .pool
            .layout()
            .planes()
            .get(plane)
            .ok_or_else(|| invalid_handle("unknown state plane"))?;
        let [heads, keys, values]: [u32; 3] = descriptor
            .shape()
            .try_into()
            .map_err(|_| invalid_handle("delta block state must have rank three"))?;
        let tokens = *inputs[0]
            .layout()
            .shape()
            .first()
            .ok_or_else(|| invalid_handle("delta block query is scalar"))?;
        let layout = DeltaBlockScratch::new(tokens, heads, keys, values).map_err(guest_error)?;
        self.limits.check_tensor_shape(&[layout.elements()])?;
        let bytes = u64::from(layout.elements())
            .checked_mul(4)
            .ok_or_else(|| quota("state scratch size overflow"))?;
        let charge = StateCharge::reserve(
            self.storage.backend.as_ref(),
            Arc::clone(&self.storage.charge.live),
            self.storage.limit.min(self.limits.live_bytes),
            bytes,
        )?;
        let scratch = StateScratch {
            backend: Arc::clone(&self.storage.backend),
            tensor: self
                .storage
                .backend
                .alloc(forja_core::DType::F32, &[layout.elements()])
                .map_err(guest_error)?,
            charge: Arc::new(charge),
        };
        let mut dispatch = self.prepare_update(
            plane,
            StateOp::DeltaRuleBlock(self.access),
            &inputs,
            output,
            Some(&scratch.tensor),
        )?;
        #[cfg(target_os = "macos")]
        if let Some(backend) = (&*self.storage.backend as &dyn std::any::Any)
            .downcast_ref::<forja_metal::MetalBackend>()
        {
            let name = variant
                .ok_or_else(|| invalid_handle("delta block requires an explicit Metal variant"))?;
            let selected = backend
                .validate_variant(&dispatch, name)
                .map_err(|error| crate::variant_error(&error))?;
            dispatch.set_backend_data(forja_core::BackendDispatchData::new(selected));
        }
        #[cfg(not(target_os = "macos"))]
        let _ = variant;
        self.record_update(plane, dispatch)?;
        self.scratch.push(scratch);
        Ok(())
    }
}

impl<B: Backend + 'static> StateCell<B> {
    /// Executes canonical delta blocks and publishes only after successful completion.
    ///
    /// # Errors
    /// Returns validation, quota or execution failures without publishing failed work.
    pub fn delta_rule_block(
        &mut self,
        inputs: [&Tensor; 5],
        output: &Tensor,
        limits: Limits,
        variant: Option<&str>,
    ) -> Result<Option<std::time::Duration>, compute::Error> {
        self.update(limits, |recording| {
            recording.delta_rule_block(0, inputs, output, variant)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Host;
    use forja_core::{DType, StateLayout, StatePlane};
    use std::sync::atomic::Ordering;

    #[test]
    fn block_scratch_is_refunded_after_refusal_and_completion() {
        let limits = Limits::new(1_048_576, 8, 1_048_576, 128, 1_048_576);
        let host = Host::new(forja_cpu::CpuBackend::new(), limits);
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 1, 1]).unwrap()])
                    .unwrap(),
                1,
            )
            .unwrap();
        let mut cell = StateCell::new(storage).unwrap();
        let inputs = [&[2, 1, 1][..], &[2, 1, 1], &[2, 1, 1], &[2, 1], &[2, 1]]
            .map(|shape| host.backend.alloc(DType::F32, shape).unwrap());
        for (tensor, value) in inputs.iter().zip([1_f32, 0.5, 2.0, -0.5, 0.25]) {
            host.backend
                .write(tensor, &value.to_le_bytes().repeat(2))
                .unwrap();
        }
        let output = host.backend.alloc(DType::F32, &[2, 1, 1]).unwrap();
        let before = host.live_bytes.load(Ordering::Acquire);
        let mut refused = limits;
        refused.live_bytes = before + 1024;
        assert!(
            cell.delta_rule_block(inputs.each_ref(), &output, refused, None)
                .is_err()
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
        refused = limits;
        refused.work_per_dispatch = 0;
        assert!(
            cell.delta_rule_block(inputs.each_ref(), &output, refused, None)
                .is_err()
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
        cell.delta_rule_block(inputs.each_ref(), &output, limits, None)
            .unwrap();
        assert!(
            host.backend
                .read(&output)
                .unwrap()
                .iter()
                .any(|&byte| byte != 0)
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    }

    #[test]
    fn timeout_retains_backing_and_scratch_charges_until_physical_completion() {
        let (control, mut host) = super::super::lifecycle_tests::controlled_host();
        host.limits.tensor_elements = 65536;
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 1, 1]).unwrap()])
                    .unwrap(),
                1,
            )
            .unwrap();
        let mut cell = StateCell::new(Arc::clone(&storage)).unwrap();
        let inputs = [&[2, 1, 1][..], &[2, 1, 1], &[2, 1, 1], &[2, 1], &[2, 1]]
            .map(|shape| host.backend.alloc(DType::F32, shape).unwrap());
        let output = host.backend.alloc(DType::F32, &[2, 1, 1]).unwrap();
        let live = Arc::clone(&host.live_bytes);
        let before = live.load(Ordering::Acquire);
        control.failure.store(1, Ordering::Release);
        assert!(
            cell.delta_rule_block(inputs.each_ref(), &output, host.limits, None)
                .is_err()
        );
        assert!(live.load(Ordering::Acquire) > before);
        drop(cell);
        drop(storage);
        drop(host);
        assert!(live.load(Ordering::Acquire) > before);
        control.complete();
        assert_eq!(live.load(Ordering::Acquire), 0);
    }
}
