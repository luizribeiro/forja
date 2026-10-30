use crate::{Host, PagePoolEntry, StateSlots, compute, invalid_handle, paged_storage};
use forja_core::{Backend, StateLayout, StatePlane};
use wasmtime::component::Resource;

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn open_state_slot(
        &mut self,
        pool: &Resource<PagePoolEntry>,
        planes: u32,
        descriptors: Vec<paged_storage::StatePlaneDescriptor>,
    ) -> Result<Resource<crate::BatchSlotEntry>, compute::Error> {
        let lanes = self.table.get(pool).map_err(invalid_handle)?.active_lanes;
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
        let slot = self.open_paged_slot(pool, planes)?;
        self.state_slots.insert(slot.rep(), state);
        Ok(slot)
    }
}
