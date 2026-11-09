use crate::{
    BatchPhase, BatchSlotEntry, EngineBatch, Host, PagePoolEntry, compute, invalid_handle,
};
use forja_core::{Backend, StateTransaction};
use wasmtime::component::Resource;
mod paged;
#[cfg(test)]
mod tests;

pub(crate) struct VerifyPublication {
    pool: u32,
    lane: u32,
    start: u64,
    rows: u32,
    state: Option<StateTransaction>,
}

impl std::fmt::Debug for VerifyPublication {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifyPublication")
            .field("lane", &self.lane)
            .field("start", &self.start)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn verify_publication(
        &self,
        pool: u32,
        input: &EngineBatch,
    ) -> Result<Option<VerifyPublication>, compute::Error> {
        if input.shape.phase != BatchPhase::Verify {
            return Ok(None);
        }
        let lane = input
            .lanes
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid_handle("verify lane is inactive"))?;
        let slot_rep = self
            .table
            .get(&Resource::<PagePoolEntry>::new_borrow(pool))
            .map_err(invalid_handle)?
            .slot
            .ok_or_else(|| invalid_handle("verify slot is unavailable"))?;
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
            .map_err(invalid_handle)?;
        let state = self
            .state_slots
            .get(&slot_rep)
            .map(|state| {
                state.transaction(
                    slot.active_bank,
                    usize::try_from(lane.slot).map_err(invalid_handle)?,
                )
            })
            .transpose()?;
        Ok(Some(VerifyPublication {
            pool,
            lane: lane.slot,
            start: u64::from(lane.start_pos),
            rows: input.shape.tokens,
            state,
        }))
    }
}
