use crate::{
    Backend, BatchSlotEntry, Host, PagePoolEntry, PagedBinding, StateSelectionBoundary,
    StateSelectionLease, compute, invalid_handle, state::StateCharge,
};
use std::sync::{Arc, Weak};
use wasmtime::component::Resource;

/// Cancellable selection ownership until the engine output passes host validation.
pub struct SelectionAttempt<B: Backend> {
    slot: u32,
    binding: Weak<PagedBinding>,
    previous: (usize, usize),
    leases: Vec<StateSelectionLease<B>>,
    boundaries: Vec<(StateSelectionBoundary<B>, u32)>,
    accepted: bool,
    _charge: Arc<StateCharge>,
}

impl<B: Backend> SelectionAttempt<B> {
    fn restore(&mut self) -> Result<(), compute::Error> {
        self.leases.clear();
        let mut failed = None;
        for (boundary, _) in self.boundaries.drain(..) {
            if let Err(error) = boundary.restore()
                && failed.is_none()
            {
                failed = Some(error);
            }
        }
        failed.map_or(Ok(()), Err)
    }
}

impl<B: Backend + 'static> SelectionAttempt<B> {
    fn validate_host(&self, host: &Host<B>) -> Result<(), compute::Error> {
        let slot = host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
            .map_err(invalid_handle)?;
        if !self.binding.ptr_eq(&Arc::downgrade(&slot.binding)) {
            return Err(invalid_handle("selection slot identity changed"));
        }
        Ok(())
    }

    /// Verifies every published scalar and advances the bank only after output acceptance.
    ///
    /// # Errors
    /// Rolls back incomplete, failed or refused selection work after draining owners.
    pub fn accept(mut self, host: &mut Host<B>) -> Result<(), compute::Error> {
        let result = (|| {
            self.validate_host(host)?;
            for lease in &self.leases {
                if lease.is_pending()? {
                    return Err(invalid_handle("selection output was not submitted"));
                }
            }
            for (boundary, scalar) in &self.boundaries {
                boundary.verify_selected(*scalar)?;
            }
            self.validate_host(host)?;
            host.advance_paged_batch_bank(self.slot)
        })();
        if let Err(error) = result {
            return Err(self.rollback(host).err().unwrap_or(error));
        }
        self.accepted = true;
        Ok(())
    }

    /// Drains and restores every selection boundary and the original bank choice.
    ///
    /// # Errors
    /// Reports the first restore failure or a replaced slot without changing a foreign slot.
    pub fn rollback(mut self, host: &mut Host<B>) -> Result<(), compute::Error> {
        let restored = self.restore();
        let banks = self.validate_host(host).and_then(|()| {
            let slot = host
                .table
                .get_mut(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
                .map_err(invalid_handle)?;
            (slot.active_bank, slot.next_bank) = self.previous;
            Ok(())
        });
        self.accepted = true;
        restored.and(banks)
    }
}

impl<B: Backend> Drop for SelectionAttempt<B> {
    fn drop(&mut self) {
        if !self.accepted {
            let _ = self.restore();
        }
    }
}

impl<B: Backend + 'static> Host<B> {
    /// Reserves scoped selection guards in the next input bank without appending KV or carry.
    ///
    /// # Errors
    /// Refuses invalid lanes, mismatched boundaries, occupied banks or exhausted quota.
    pub fn begin_slot_selection(
        &mut self,
        pool: &Resource<PagePoolEntry>,
        active: impl Iterator<Item = (usize, u32)>,
    ) -> Result<Option<SelectionAttempt<B>>, compute::Error> {
        let Some(slot_rep) = self.table.get(pool).map_err(invalid_handle)?.slot else {
            return Ok(None);
        };
        let Some(state) = self.state_slots.get(&slot_rep) else {
            return Ok(None);
        };
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot_rep))
            .map_err(invalid_handle)?;
        let capacity = slot.binding.lanes.len();
        if state.selection_count(slot.next_bank)? != 0 {
            return Err(invalid_handle("selection bank is occupied"));
        }
        let charge = self.reserve_references(capacity)?;
        let active = active
            .take(
                capacity
                    .checked_add(1)
                    .ok_or_else(|| invalid_handle("selection lane count overflowed"))?,
            )
            .collect::<Vec<_>>();
        if active.is_empty()
            || active.len() > capacity
            || active.iter().any(|&(lane, _)| lane >= capacity)
            || active.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(invalid_handle("selection lane geometry is invalid"));
        }
        let bank = slot.next_bank;
        let mut attempt = SelectionAttempt {
            slot: slot_rep,
            binding: Arc::downgrade(&slot.binding),
            previous: (slot.active_bank, slot.next_bank),
            leases: Vec::with_capacity(active.len()),
            boundaries: Vec::with_capacity(active.len()),
            accepted: false,
            _charge: charge,
        };
        if let Err(error) = self.select_paged_batch_bank(slot_rep) {
            return Err(attempt.rollback(self).err().unwrap_or(error));
        }
        let admitted = (|| {
            for (lane, start) in active {
                let lease = self
                    .state_slots
                    .get_mut(&slot_rep)
                    .ok_or_else(|| invalid_handle("state slot is missing"))?
                    .begin_shared_selection(bank, lane, u64::from(start))?;
                let boundary = lease.boundary()?;
                attempt.leases.push(lease);
                attempt.boundaries.push(boundary);
                let lane = u32::try_from(lane).map_err(invalid_handle)?;
                let kv = self.paged_lane_prefix(pool.rep(), lane)?;
                let length = kv
                    .sequence
                    .lock()
                    .map_err(|_| invalid_handle("selection KV boundary is poisoned"))?
                    .logical_len();
                if u64::from(start).checked_add(1) != Some(length) {
                    return Err(invalid_handle("selection KV and carry boundaries differ"));
                }
            }
            Ok(())
        })();
        if let Err(error) = admitted {
            return Err(attempt.rollback(self).err().unwrap_or(error));
        }
        Ok(Some(attempt))
    }
}
