use super::{
    Arc, Backend, CommandListEntry, Host, Resource, TensorEntry, compute, core_dtype,
    invalid_handle,
};

impl<B: Backend + Send + Sync + 'static> Host<B> {
    pub(super) fn allocate_command_scratch(
        &mut self,
        commands: &Resource<CommandListEntry>,
        dtype: compute::Dtype,
        shape: &[u32],
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        self.check_tensor_shape(shape)?;
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        let candidate = entry
            .retained
            .iter()
            .find(|retained| {
                let owner = &retained.buffer.owner;
                if retained.buffer.guest_reachable()
                    || owner.layout().dtype() != core_dtype(dtype)
                    || owner.layout().shape() != shape
                    || !owner.is_writable()
                    || entry.protected.contains(&owner.buffer())
                    || !entry
                        .scratch_allocations
                        .iter()
                        .any(|(buffer, _)| *buffer == owner.buffer())
                {
                    return false;
                }
                let owners = entry
                    .retained
                    .iter()
                    .filter(|other| Arc::ptr_eq(&other.buffer, &retained.buffer))
                    .count();
                Arc::strong_count(&retained.buffer) == owners
            })
            .map(|retained| Arc::clone(&retained.buffer));
        if let Some(buffer) = candidate {
            let resource = self
                .table
                .push(TensorEntry {
                    tensor: buffer.owner.clone(),
                    buffer,
                    symbolic: None,
                    reference_charge: None,
                    slot_token: None,
                })
                .map_err(invalid_handle)?;
            self.table
                .get(&resource)
                .map_err(invalid_handle)?
                .buffer
                .retain_guest_handle();
            self.live_handles += 1;
            return Ok(resource);
        }
        let charge = self.reserve_reference()?;
        let resource = self.alloc(dtype, shape)?;
        let buffer = self.entry(&resource)?.tensor.buffer();
        self.table
            .get_mut(commands)
            .map_err(invalid_handle)?
            .scratch_allocations
            .push((buffer, charge));
        Ok(resource)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use forja_core::BufferId;
    use std::sync::atomic::Ordering;

    fn host() -> Host<forja_cpu::CpuBackend> {
        Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(8192, 8, 4096, 128, 4096),
        )
    }
    fn id(host: &Host<forja_cpu::CpuBackend>, tensor: &Resource<TensorEntry>) -> BufferId {
        host.entry(tensor).unwrap().tensor.buffer()
    }
    #[tokio::test]
    async fn retired_outputs_reuse_storage_without_changing_recorded_values() {
        let mut host = host();
        let commands = host.command_list().unwrap();
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        host.write(&input, &2_f32.to_le_bytes().repeat(7)).unwrap();
        let first = host
            .allocate_command_scratch(&commands, compute::Dtype::F32, &[7])
            .unwrap();
        let allocation = id(&host, &first);
        let saved = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &first,
        )
        .unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(first.rep())],
            &saved,
        )
        .unwrap();
        host.drop_tensor(first).unwrap();
        let next = host
            .allocate_command_scratch(&commands, compute::Dtype::F32, &[7])
            .unwrap();
        assert_eq!(id(&host, &next), allocation);
        let replacement = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        host.write(&replacement, &4_f32.to_le_bytes().repeat(7))
            .unwrap();
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(replacement.rep())],
            &next,
        )
        .unwrap();
        host.prepare_submit(commands).unwrap().run().await.unwrap();
        assert_eq!(
            host.backend
                .read(&host.entry(&saved).unwrap().tensor)
                .unwrap(),
            2_f32.to_le_bytes().repeat(7)
        );
        assert_eq!(
            host.backend
                .read(&host.entry(&next).unwrap().tensor)
                .unwrap(),
            4_f32.to_le_bytes().repeat(7)
        );
    }
    #[test]
    fn live_views_other_lists_and_quota_refusals_keep_allocations_distinct() {
        let mut host = host();
        let commands = host.command_list().unwrap();
        let other = host.command_list().unwrap();
        let first = host
            .allocate_command_scratch(&commands, compute::Dtype::F32, &[7])
            .unwrap();
        let target = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let allocation = id(&host, &first);
        host.dispatch(
            &commands,
            compute::Op::Copy,
            &[Resource::new_borrow(first.rep())],
            &target,
        )
        .unwrap();
        let alias = host
            .view(&first, compute::ViewOp::Reshape(vec![7]))
            .unwrap();
        host.drop_tensor(first).unwrap();
        let live = host
            .allocate_command_scratch(&commands, compute::Dtype::F32, &[7])
            .unwrap();
        assert_ne!(id(&host, &live), allocation);
        host.dispatch(
            &other,
            compute::Op::Copy,
            &[Resource::new_borrow(alias.rep())],
            &target,
        )
        .unwrap();
        host.drop_tensor(alias).unwrap();
        let retained = host
            .allocate_command_scratch(&commands, compute::Dtype::F32, &[7])
            .unwrap();
        assert_ne!(id(&host, &retained), allocation);
        let before = host.live_bytes.load(Ordering::Acquire);
        assert!(
            host.allocate_command_scratch(&commands, compute::Dtype::F32, &[4097])
                .is_err()
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    }
}
