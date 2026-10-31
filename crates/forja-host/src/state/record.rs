use super::{StateCharge, StateSequenceHandle, StateStorage, pool_error};
use crate::{Limits, compute, guest_error, invalid_handle, quota};
use forja_core::{
    Backend, CommandList, DType, KernelCost, Op, StateAccess, StateOp, StateTransaction, Tensor,
};
use std::sync::Arc;

pub(super) struct StateMetadata<B: Backend> {
    backend: Arc<B>,
    tensors: Vec<Tensor>,
    charge: Arc<StateCharge>,
}

impl<B: Backend> std::ops::Deref for StateMetadata<B> {
    type Target = [Tensor];
    fn deref(&self) -> &Self::Target {
        &self.tensors
    }
}

impl<B: Backend> Drop for StateMetadata<B> {
    fn drop(&mut self) {
        for tensor in &self.tensors {
            let _ = self.backend.release(tensor);
        }
    }
}

/// Authenticated plane commands without ownership of a request's transaction.
pub struct StateCommands<B: Backend> {
    pub(super) storage: Arc<StateStorage<B>>,
    pub(super) access: StateAccess,
    pub(super) scratch: Vec<Arc<super::block::StateScratch<B>>>,
    pub(super) captured_scratch: Vec<forja_core::CompletionResource>,
    pub(super) captured_scratch_charges: Vec<forja_core::CompletionResource>,
    pub(super) metadata: Arc<StateMetadata<B>>,
    pub(super) feedback: Option<super::feedback::FeedbackUse>,
    pub(crate) commands: CommandList,
    pub(super) updated: Vec<bool>,
    pub(super) limits: Limits,
}

/// A transaction recording that never exposes state backing as ordinary tensors.
pub struct StateRecording<B: Backend> {
    body: StateCommands<B>,
    pub(super) transaction: StateTransaction,
    pub(super) sequence: StateSequenceHandle<B>,
    pub(super) handed_off: bool,
    pub(super) predecessor: Option<Arc<super::outcome::StateOutcome>>,
    _drain: super::sequence::RecordingGuard,
}

impl<B: Backend> std::ops::Deref for StateRecording<B> {
    type Target = StateCommands<B>;
    fn deref(&self) -> &Self::Target {
        &self.body
    }
}
impl<B: Backend> std::ops::DerefMut for StateRecording<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.body
    }
}
impl<B: Backend> StateRecording<B> {
    /// Serial of the version whose publication this recording requests.
    #[must_use]
    pub fn serial(&self) -> u64 {
        self.transaction.serial()
    }
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
        self.record_with_limits(sequence, transaction, self.limits)
    }

    pub(super) fn record_with_limits(
        self: &Arc<Self>,
        sequence: &StateSequenceHandle<B>,
        transaction: StateTransaction,
        limits: Limits,
    ) -> Result<StateRecording<B>, compute::Error> {
        let charge = self.command_charge(limits)?;
        self.record_body(sequence, transaction, |access| {
            Ok(StateCommands::empty(self, access, charge, limits))
        })
    }

    pub(super) fn record_prepared(
        self: &Arc<Self>,
        sequence: &StateSequenceHandle<B>,
        transaction: StateTransaction,
        mut body: StateCommands<B>,
    ) -> Result<StateRecording<B>, compute::Error> {
        let result = self.record_body(sequence, transaction, |access| {
            if !Arc::ptr_eq(self, &body.storage) {
                return Err(invalid_handle("state metadata belongs to another storage"));
            }
            body.access = access;
            Ok(body)
        });
        if result.is_err() {
            self.poisoned
                .store(true, std::sync::atomic::Ordering::Release);
        }
        result
    }

    fn record_body(
        self: &Arc<Self>,
        sequence: &StateSequenceHandle<B>,
        transaction: StateTransaction,
        prepare: impl FnOnce(StateAccess) -> Result<StateCommands<B>, compute::Error>,
    ) -> Result<StateRecording<B>, compute::Error> {
        self.check_usable()?;
        if !Arc::ptr_eq(self, &sequence.storage) {
            return Err(invalid_handle("state sequence belongs to another storage"));
        }
        let (slot, predecessor) = {
            let sequence = sequence.lock();
            (
                sequence.bind(&transaction).map_err(pool_error)?,
                sequence.dependency(&transaction).map_err(pool_error)?,
            )
        };
        self.pool.validate_slot(&slot).map_err(pool_error)?;
        let access = StateAccess::from_slot(&slot).map_err(pool_error)?;
        let table = slot
            .table()
            .entries()
            .iter()
            .flat_map(|entry| [entry.page(), entry.generation()])
            .collect::<Vec<_>>();
        let mut body = prepare(access)?;
        body.feedback = sequence
            .lock()
            .feedback_use(
                &transaction,
                access.destination(),
                Arc::clone(&body.metadata.charge),
            )
            .map_err(pool_error)?;
        let mut recording = StateRecording {
            body,
            transaction,
            sequence: sequence.clone(),
            handed_off: false,
            predecessor,
            _drain: sequence.recording_guard().map_err(guest_error)?,
        };
        recording.initialize_metadata(table, slot.table().generations().to_vec(), true)?;
        Ok(recording)
    }
    pub(super) fn prepare_record_metadata(
        self: &Arc<Self>,
        access: StateAccess,
    ) -> Result<StateCommands<B>, compute::Error> {
        self.check_usable()?;
        let mut commands =
            StateCommands::empty(self, access, self.command_charge(self.limits)?, self.limits);
        commands.allocate_metadata()?;
        Ok(commands)
    }

    pub(super) fn inactive_commands(
        self: &Arc<Self>,
        access: StateAccess,
    ) -> Result<StateCommands<B>, compute::Error> {
        let mut commands = self.prepare_record_metadata(access.inactive())?;
        let records = usize::try_from(self.planes[0].layout().shape()[0])
            .map_err(|_| quota("state metadata exceeds addressable memory"))?;
        commands.initialize_metadata(vec![u32::MAX; 4], vec![u32::MAX; records], false)?;
        Ok(commands)
    }

    fn command_charge(&self, limits: Limits) -> Result<Arc<StateCharge>, compute::Error> {
        let records = self.planes[0].layout().shape()[0];
        let bytes = u64::from(records)
            .checked_mul(8)
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| n.checked_add(super::feedback::FEEDBACK_USE_BYTES))
            .and_then(|n| n.checked_add(u64::try_from(self.planes.len()).ok()?.checked_mul(2048)?))
            .ok_or_else(|| quota("state recording size overflow"))?;
        StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.charge.live),
            self.limit.min(limits.live_bytes),
            bytes,
        )
        .map(Arc::new)
    }
}

impl<B: Backend> StateCommands<B> {
    fn empty(
        storage: &Arc<StateStorage<B>>,
        access: StateAccess,
        charge: Arc<StateCharge>,
        limits: Limits,
    ) -> Self {
        Self {
            storage: Arc::clone(storage),
            access,
            scratch: Vec::new(),
            feedback: None,
            captured_scratch: Vec::new(),
            captured_scratch_charges: Vec::new(),
            metadata: Arc::new(StateMetadata {
                backend: Arc::clone(&storage.backend),
                tensors: Vec::new(),
                charge,
            }),
            commands: CommandList::new(),
            updated: vec![false; storage.planes.len()],
            limits,
        }
    }

    fn allocate_metadata(&mut self) -> Result<(), compute::Error> {
        if self.metadata.len() == 3 {
            return Ok(());
        }
        if !self.metadata.is_empty() {
            return Err(invalid_handle("incomplete state metadata"));
        }
        let records = self.storage.planes[0].layout().shape()[0];
        for shape in [vec![2, 2], vec![records], vec![1]] {
            let tensor = self
                .storage
                .backend
                .alloc(DType::U32, &shape)
                .map_err(guest_error)?;
            Arc::get_mut(&mut self.metadata)
                .ok_or_else(|| invalid_handle("state metadata was shared before construction"))?
                .tensors
                .push(tensor);
        }
        Ok(())
    }

    fn initialize_metadata(
        &mut self,
        table: Vec<u32>,
        generations: Vec<u32>,
        active: bool,
    ) -> Result<(), compute::Error> {
        let records = self.storage.planes[0].layout().shape()[0];
        if table.len() != 4 || u32::try_from(generations.len()).ok() != Some(records) {
            return Err(invalid_handle(
                "state authentication metadata has invalid extents",
            ));
        }
        self.allocate_metadata()?;
        for (tensor, values) in
            self.metadata
                .iter()
                .zip([table, generations, vec![u32::from(active)]])
        {
            let bytes = values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            self.storage
                .backend
                .write(tensor, &bytes)
                .map_err(guest_error)?;
        }
        Ok(())
    }

    pub(crate) fn release_graph_scratch(&mut self) {
        self.scratch.clear();
        self.captured_scratch.clear();
        self.captured_scratch_charges.clear();
    }

    pub(super) fn binding_resources(&self) -> Vec<forja_core::CompletionResource> {
        vec![
            forja_core::CompletionResource::new(Arc::clone(&self.storage.charge)),
            forja_core::CompletionResource::new(Arc::clone(&self.metadata.charge)),
        ]
    }

    pub(crate) fn retained_resources(&self) -> Vec<forja_core::CompletionResource> {
        let mut owners = self.binding_resources();
        owners.extend(
            self.scratch
                .iter()
                .map(|scratch| scratch.retained_resource()),
        );
        owners.extend(self.captured_scratch_charges.iter().cloned());
        owners.extend(
            self.feedback
                .iter()
                .flat_map(super::feedback::FeedbackUse::retained_resources),
        );
        owners
    }

    pub(crate) fn protected_buffers(&self) -> impl Iterator<Item = forja_core::BufferId> + '_ {
        self.storage
            .planes
            .iter()
            .chain(self.metadata.iter())
            .map(Tensor::buffer)
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
        let dispatch = self.prepare_update(plane, operation, inputs, output, None)?;
        self.record_update(plane, dispatch)
    }

    pub(super) fn prepare_update(
        &self,
        plane: usize,
        operation: StateOp,
        inputs: &[&Tensor],
        output: &Tensor,
        scratch: Option<&Tensor>,
    ) -> Result<forja_core::Dispatch, compute::Error> {
        if self.updated.get(plane).copied() != Some(false) {
            return Err(invalid_handle("state plane is unknown or already updated"));
        }
        let previous = self.storage.version_view(plane, self.access.source())?;
        let next = self
            .storage
            .version_view(plane, self.access.destination())?;
        let operands = operation
            .bind_inputs(inputs, &previous, &self.metadata)
            .map_err(guest_error)?;
        let outputs = [output, &next]
            .into_iter()
            .chain(scratch)
            .collect::<Vec<_>>();
        for tensor in operands.iter().copied().chain(outputs.iter().copied()) {
            if tensor.layout().shape().len() > self.limits.tensor_rank
                || tensor.layout().element_count() > self.limits.tensor_elements
            {
                return Err(quota("state operand exceeds tensor limits"));
            }
        }
        let mut candidate = CommandList::new();
        candidate
            .dispatch_many(Op::State(operation), &operands, &outputs)
            .map_err(guest_error)?;
        candidate
            .into_dispatches()
            .pop()
            .ok_or_else(|| invalid_handle("missing validated state operation"))
    }

    pub(super) fn record_update(
        &mut self,
        plane: usize,
        dispatch: forja_core::Dispatch,
    ) -> Result<(), compute::Error> {
        let cost = KernelCost::of(&dispatch).ok_or_else(|| quota("state work overflow"))?;
        let work = cost
            .flops
            .checked_add(cost.scalar_ops)
            .and_then(|n| n.checked_add(cost.read_bytes))
            .and_then(|n| n.checked_add(cost.write_bytes))
            .ok_or_else(|| quota("state work overflow"))?;
        if work > self.limits.work_per_dispatch
            || self.commands.len() >= self.limits.dispatches_per_list
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

    #[test]
    fn captured_metadata_retains_its_charge_without_pinning_a_sequence() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
                1,
            )
            .unwrap();
        let before = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        let sequence = storage.reserve().unwrap();
        let recording = storage
            .record(&sequence, sequence.begin(1).unwrap())
            .unwrap();
        let metadata = Arc::clone(&recording.metadata);
        let retained = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        assert!(retained > before);
        drop(recording);
        drop(sequence);
        let replacement = storage.reserve().unwrap();
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            retained
        );
        drop(metadata);
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            before
        );
        drop(replacement);
    }

    fn failed_predecessor_taints_queued_descendant<B: Backend + 'static>(
        backend: B,
        check_failed_read: impl FnOnce(Result<Vec<u8>, forja_core::BackendError>),
    ) {
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
        check_failed_read(host.backend.read(&y));
        let retry = sequence.begin(3).unwrap();
        let mut retry = storage.record(&sequence, retry).unwrap();
        retry.causal_convolution(0, &x, &weights, &y).unwrap();
        retry.submit().unwrap().finish().unwrap();
        assert_eq!(host.backend.read(&y).unwrap(), 2_f32.to_le_bytes());
    }

    #[test]
    fn failed_state_submission_restores_checkpoint_after_descendant_drain() {
        failed_predecessor_taints_queued_descendant(forja_cpu::CpuBackend::new(), |read| {
            assert_eq!(read.unwrap(), 9_f32.to_le_bytes());
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_failed_state_submission_restores_checkpoint_after_descendant_drain() {
        failed_predecessor_taints_queued_descendant(
            forja_metal::MetalBackend::new().unwrap(),
            |read| {
                assert!(matches!(
                    read,
                    Err(forja_core::BackendError::DeviceErrorFlag { .. })
                ));
            },
        );
    }
}
