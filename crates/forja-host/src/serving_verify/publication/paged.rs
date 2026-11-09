use super::{
    Backend, BatchSlotEntry, Host, PagePoolEntry, Resource, VerifyPublication, compute,
    invalid_handle,
};
use std::sync::atomic::Ordering;

impl<B: Backend + 'static> Host<B> {
    fn mark_verify_tables_dirty(&self, pool: u32, lane: u32) -> Result<(), compute::Error> {
        let slot = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("verify slot is unavailable"))?;
        let binding = &self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?
            .binding;
        for dirty in &binding
            .lane(usize::try_from(lane).map_err(invalid_handle)?)?
            .table_dirty
        {
            dirty.store(true, Ordering::Release);
        }
        Ok(())
    }
}

impl VerifyPublication {
    pub(super) fn invalidate_after_cut<B: Backend + 'static>(
        &self,
        host: &Host<B>,
        accepted: u32,
    ) -> Result<(), compute::Error> {
        if u64::from(accepted) + 1 < u64::from(self.rows) {
            host.mark_verify_tables_dirty(self.pool, self.lane)?;
        }
        Ok(())
    }

    pub(super) fn publish_kv<B: Backend + 'static>(
        &self,
        host: &mut Host<B>,
        accepted: u32,
    ) -> Result<(), compute::Error> {
        if accepted >= self.rows {
            return Err(invalid_handle("accepted verify row exceeds the batch"));
        }
        let prefix = host.paged_lane_prefix(self.pool, self.lane)?;
        let mut sequence = prefix.sequence.lock().map_err(invalid_handle)?;
        let end = self
            .start
            .checked_add(u64::from(self.rows))
            .ok_or_else(|| invalid_handle("verify end overflowed"))?;
        let kept = self
            .start
            .checked_add(u64::from(accepted))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid_handle("verify cut overflowed"))?;
        if sequence.logical_len() != end {
            return Err(invalid_handle("verify KV boundary differs from the batch"));
        }
        if kept < end {
            sequence.rewind_private(kept).map_err(invalid_handle)?;
        }
        drop(sequence);
        self.invalidate_after_cut(host, accepted)
    }
}
