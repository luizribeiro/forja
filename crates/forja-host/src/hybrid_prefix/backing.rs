use crate::{
    Backend, BatchSlotEntry, BufferHandle, Host, PagePoolCharge, PagePoolEntry, TensorEntry,
    compute, invalid_handle, quota, release_buffers, state::StateCharge,
};
use std::sync::Arc;
use wasmtime::component::Resource;

pub(super) struct HybridBacking<B: Backend> {
    backend: Arc<B>,
    buffers: Vec<Arc<BufferHandle>>,
    _pool_charge: Arc<PagePoolCharge>,
    _charge: StateCharge,
    pub(super) bytes: u64,
}
impl<B: Backend> Drop for HybridBacking<B> {
    fn drop(&mut self) {
        let _ = release_buffers(
            self.backend.as_ref(),
            std::mem::take(&mut self.buffers),
            None,
        );
    }
}
impl<B: Backend> Host<B> {
    pub(super) fn hybrid_backing(
        &self,
        resource: &Resource<PagePoolEntry>,
    ) -> Result<Arc<HybridBacking<B>>, compute::Error> {
        let owner = self.table.get(resource).map_err(invalid_handle)?;
        let slot = owner
            .slot
            .ok_or_else(|| invalid_handle("hybrid attention backing is not open"))?;
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?;
        if !slot.binding.pool.same_pool(&owner.pool)
            || slot.planes.len() != owner.pool.layout().planes().len()
        {
            return Err(invalid_handle(
                "hybrid entries require every plane from the owning attention pool",
            ));
        }
        let bytes = u64::try_from(slot.planes.len())
            .ok()
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| quota("hybrid backing references overflow"))?;
        let charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            bytes,
        )?;
        let mut backing = HybridBacking {
            backend: Arc::clone(&self.backend),
            buffers: Vec::new(),
            _pool_charge: Arc::clone(&owner.charge),
            _charge: charge,
            bytes,
        };
        backing
            .buffers
            .try_reserve_exact(slot.planes.len())
            .map_err(|_| quota("hybrid backing references allocation failed"))?;
        for &plane in &slot.planes {
            let entry = self
                .table
                .get(&Resource::<TensorEntry>::new_borrow(plane))
                .map_err(invalid_handle)?;
            backing.buffers.push(Arc::clone(&entry.buffer));
        }
        Ok(Arc::new(backing))
    }
}
