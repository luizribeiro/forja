use super::{StateCharge, StateSequenceHandle, StateStorage, pool_error};
use crate::{compute, guest_error, invalid_handle, quota};
use forja_core::{
    Backend, CommandList, DType, KernelCost, Op, StateAccess, StateOp, StateTransaction, Tensor,
};
use std::sync::Arc;

/// A transaction recording that never exposes state backing as ordinary tensors.
pub struct StateRecording<B: Backend> {
    pub(super) storage: Arc<StateStorage<B>>,
    pub(super) transaction: StateTransaction,
    pub(super) sequence: StateSequenceHandle<B>,
    pub(super) handed_off: bool,
    access: StateAccess,
    metadata: Vec<Tensor>,
    _charge: StateCharge,
    pub(super) commands: CommandList,
    pub(super) updated: Vec<bool>,
}

impl<B: Backend> StateStorage<B> {
    /// Binds immutable authentication metadata to one live sequence transaction.
    ///
    /// # Errors
    /// Refuses foreign capabilities, quota exhaustion or backend allocation failure.
    pub fn record(
        self: &Arc<Self>,
        sequence: &StateSequenceHandle<B>,
        transaction: StateTransaction,
    ) -> Result<StateRecording<B>, compute::Error> {
        self.check_usable()?;
        if !Arc::ptr_eq(self, &sequence.storage) {
            return Err(invalid_handle("state sequence belongs to another storage"));
        }
        let records = self.planes[0].layout().shape()[0];
        let bytes = u64::from(records)
            .checked_mul(8)
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| n.checked_add(u64::try_from(self.planes.len()).ok()?.checked_mul(2048)?))
            .ok_or_else(|| quota("state recording size overflow"))?;
        let charge = StateCharge::reserve(Arc::clone(&self.charge.live), self.limit, bytes)?;
        let slot = sequence.lock().bind(&transaction).map_err(pool_error)?;
        self.pool.validate_slot(&slot).map_err(pool_error)?;
        let access = StateAccess::from_slot(&slot).map_err(pool_error)?;
        let mut recording = StateRecording {
            storage: Arc::clone(self),
            transaction,
            sequence: sequence.clone(),
            handed_off: false,
            access,
            metadata: Vec::new(),
            _charge: charge,
            commands: CommandList::new(),
            updated: vec![false; self.planes.len()],
        };
        let table = slot
            .table()
            .entries()
            .iter()
            .flat_map(|entry| [entry.page(), entry.generation()])
            .collect::<Vec<_>>();
        for (shape, values) in [
            (vec![2, 2], table),
            (vec![records], slot.table().generations().to_vec()),
            (vec![1], vec![1]),
        ] {
            let tensor = self
                .backend
                .alloc(DType::U32, &shape)
                .map_err(guest_error)?;
            recording.metadata.push(tensor.clone());
            let bytes = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            self.backend.write(&tensor, &bytes).map_err(guest_error)?;
        }
        Ok(recording)
    }
}

impl<B: Backend> StateRecording<B> {
    /// Serial of the version whose publication this recording requests.
    #[must_use]
    pub fn serial(&self) -> u64 {
        self.transaction.serial()
    }

    /// Records one plane's delta-rule update from Q, K, V, decay and beta.
    ///
    /// # Errors
    /// Refuses invalid geometry, aliasing, work quota or a duplicate plane update.
    pub fn delta_rule(
        &mut self,
        plane: usize,
        inputs: [&Tensor; 5],
        output: &Tensor,
    ) -> Result<(), compute::Error> {
        self.update(plane, StateOp::DeltaRule(self.access), &inputs, output)
    }

    /// Records one plane's raw history carry and causal convolution.
    ///
    /// # Errors
    /// Refuses invalid geometry, aliasing, work quota or a duplicate plane update.
    pub fn causal_convolution(
        &mut self,
        plane: usize,
        input: &Tensor,
        weights: &Tensor,
        output: &Tensor,
    ) -> Result<(), compute::Error> {
        self.update(
            plane,
            StateOp::CausalDepthwiseConv(self.access),
            &[input, weights],
            output,
        )
    }

    fn update(
        &mut self,
        plane: usize,
        operation: StateOp,
        inputs: &[&Tensor],
        output: &Tensor,
    ) -> Result<(), compute::Error> {
        if self.updated.get(plane).copied() != Some(false) {
            return Err(invalid_handle("state plane is unknown or already updated"));
        }
        let previous = self.storage.version_view(plane, self.access.source())?;
        let next = self
            .storage
            .version_view(plane, self.access.destination())?;
        let mut operands = inputs.to_vec();
        operands.push(&previous);
        operands.extend(&self.metadata);
        for tensor in operands.iter().copied().chain([output, &next]) {
            if tensor.layout().shape().len() > self.storage.limits.tensor_rank
                || tensor.layout().element_count() > self.storage.limits.tensor_elements
            {
                return Err(quota("state operand exceeds tensor limits"));
            }
        }
        let mut candidate = CommandList::new();
        candidate
            .dispatch_many(Op::State(operation), &operands, &[output, &next])
            .map_err(guest_error)?;
        let dispatch = candidate
            .into_dispatches()
            .pop()
            .ok_or_else(|| invalid_handle("missing validated state operation"))?;
        let cost = KernelCost::of(&dispatch).ok_or_else(|| quota("state work overflow"))?;
        let work = cost
            .flops
            .checked_add(cost.scalar_ops)
            .and_then(|n| n.checked_add(cost.read_bytes))
            .and_then(|n| n.checked_add(cost.write_bytes))
            .ok_or_else(|| quota("state work overflow"))?;
        if work > self.storage.limits.work_per_dispatch
            || self.commands.len() >= self.storage.limits.dispatches_per_list
        {
            return Err(quota("state recording exceeds command work limits"));
        }
        self.commands.record_validated(dispatch);
        self.updated[plane] = true;
        Ok(())
    }
}

impl<B: Backend> Drop for StateRecording<B> {
    fn drop(&mut self) {
        if !self.handed_off {
            let mut sequence = self.sequence.lock();
            if sequence.cancel_unsubmitted(&self.transaction).is_err()
                && self
                    .storage
                    .poisoned
                    .load(std::sync::atomic::Ordering::Acquire)
            {
                let _ = sequence.abandon(&self.transaction);
            }
        }
        for tensor in &self.metadata {
            let _ = self.storage.backend.release(tensor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, Limits};
    use forja_core::{StateLayout, StatePlane};

    #[test]
    fn opaque_recording_authenticates_ownership_and_refuses_aliases() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap();
        let storage = host.create_state_storage(layout.clone(), 1).unwrap();
        let foreign = host.create_state_storage(layout, 1).unwrap();
        let sequence = storage.reserve().unwrap();
        let tx = sequence.begin(1).unwrap();
        assert!(foreign.record(&sequence, tx.clone()).is_err());
        let charge = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        let mut recording = storage.record(&sequence, tx).unwrap();
        assert_eq!(recording.serial(), 1);
        let x = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let y = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        assert!(recording.causal_convolution(0, &x, &weights, &x).is_err());
        assert!(recording.causal_convolution(1, &x, &weights, &y).is_err());
        recording.causal_convolution(0, &x, &weights, &y).unwrap();
        assert!(recording.causal_convolution(0, &x, &weights, &y).is_err());
        drop(recording);
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            charge
        );
    }

    fn failed_predecessor_taints_queued_descendant<B: Backend + 'static>(backend: B) {
        let host = Host::new(backend, Limits::new(65536, 8, 4096, 128, 4096));
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 1]).unwrap()]).unwrap();
        let storage = host.create_state_storage(layout, 1).unwrap();
        let sequence = storage.reserve().unwrap();
        let x = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[2, 1]).unwrap();
        let y = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
        let later_y = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
        host.backend.write(&x, &2_f32.to_le_bytes()).unwrap();
        host.backend
            .write(
                &weights,
                &[1_f32.to_le_bytes(), 1_f32.to_le_bytes()].concat(),
            )
            .unwrap();
        host.backend.write(&y, &9_f32.to_le_bytes()).unwrap();
        let first = sequence.begin(1).unwrap();
        let mut bad = storage.record(&sequence, first).unwrap();
        bad.causal_convolution(0, &x, &weights, &y).unwrap();
        host.backend.write(&bad.metadata[1], &[0; 12]).unwrap();
        let first = bad.submit().unwrap();
        let second = sequence.begin(2).unwrap();
        let mut descendant = storage.record(&sequence, second).unwrap();
        descendant
            .causal_convolution(0, &x, &weights, &later_y)
            .unwrap();
        let second = descendant.submit().unwrap();
        assert!(first.finish().is_err());
        assert!(sequence.begin(3).is_err());
        assert!(second.finish().is_err());
        assert_eq!(host.backend.read(&y).unwrap(), 9_f32.to_le_bytes());
        let retry = sequence.begin(3).unwrap();
        let mut retry = storage.record(&sequence, retry).unwrap();
        retry.causal_convolution(0, &x, &weights, &y).unwrap();
        retry.submit().unwrap().finish().unwrap();
        assert_eq!(host.backend.read(&y).unwrap(), 2_f32.to_le_bytes());
    }

    #[test]
    fn failed_state_submission_restores_checkpoint_after_descendant_drain() {
        failed_predecessor_taints_queued_descendant(forja_cpu::CpuBackend::new());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_failed_state_submission_restores_checkpoint_after_descendant_drain() {
        failed_predecessor_taints_queued_descendant(forja_metal::MetalBackend::new().unwrap());
    }
}
