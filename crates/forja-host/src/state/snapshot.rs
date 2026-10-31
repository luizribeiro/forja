use super::{CanonicalPrompt, LanePosition, StateCharge, StateSequenceHandle};
use crate::{SnapshotAllowance, compute, guest_error, invalid_handle, quota};
use forja_core::{Backend, StateLayout, Tensor};
use std::sync::Arc;

/// Immutable, byte-exact image of every plane at one completed canonical boundary.
pub struct StateSnapshot<B: Backend> {
    pub(super) backend: Arc<B>,
    pub(super) layout: StateLayout,
    pub(super) planes: Vec<Tensor>,
    pub(super) prompt: Arc<CanonicalPrompt>,
    pub(super) position: LanePosition,
    pub(super) charge: Arc<StateCharge>,
    count: usize,
}

impl<B: Backend> StateSnapshot<B> {
    pub(crate) fn retained_allocations(&self) -> [(usize, u64); 2] {
        [
            (Arc::as_ptr(&self.charge).addr(), self.charge.bytes),
            (
                Arc::as_ptr(&self.prompt.charge).addr(),
                self.prompt.charged_bytes(),
            ),
        ]
    }
    pub(crate) fn owned_by(
        &self,
        backend: &Arc<B>,
        live: &Arc<std::sync::atomic::AtomicU64>,
    ) -> bool {
        Arc::ptr_eq(&self.backend, backend) && Arc::ptr_eq(&self.charge.live, live)
    }

    /// Exact prefix ledger; an earlier image cannot be obtained by shortening it.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.prompt.tokens[..self.count]
    }
    /// Numerical and artifact identity of this image.
    #[must_use]
    pub fn namespace(&self) -> forja_core::HybridPrefixNamespace {
        self.prompt.namespace
    }
    /// Serving cursors at the copied boundary.
    #[must_use]
    pub const fn position(&self) -> LanePosition {
        self.position
    }
    /// Image backing and descriptors; the shared ledger is charged separately.
    #[must_use]
    pub fn charged_bytes(&self) -> u64 {
        self.charge.bytes
    }
}

impl<B: Backend> StateSequenceHandle<B> {
    /// Copies every plane without scalar conversion or exposure through ordinary guest views.
    ///
    /// # Errors
    /// Refuses incomplete/generated boundaries, disabled allowances, quota or backend failures.
    pub fn capture_snapshot(
        &self,
        allowance: SnapshotAllowance,
    ) -> Result<Arc<StateSnapshot<B>>, compute::Error> {
        self.storage.check_usable()?;
        let journal = self.lock();
        let (prompt, end) = journal
            .canonical_boundary()
            .map_err(invalid_handle)?
            .ok_or_else(|| invalid_handle("state is not a completed canonical prompt boundary"))?;
        let checkpoint = journal.checkpoint().map_err(invalid_handle)?;
        let record = journal
            .allocation_table()
            .map_err(invalid_handle)?
            .resolve(checkpoint.version())
            .map_err(invalid_handle)?;
        let position = journal.position;
        let attached = self.snapshot.lock().map_err(invalid_handle)?.clone();
        drop(journal);
        let count = usize::try_from(end).map_err(|_| quota("snapshot token count overflow"))?;
        let layout = self.storage.pool.layout();
        let descriptors = u64::try_from(layout.planes().len())
            .map_err(|_| quota("snapshot plane count overflow"))?
            .checked_mul(4096)
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| quota("snapshot metadata overflow"))?;
        let bytes = layout
            .version_bytes()
            .checked_add(descriptors)
            .ok_or_else(|| quota("snapshot byte count overflow"))?;
        if bytes
            .checked_add(prompt.charged_bytes())
            .is_none_or(|total| total > allowance.bytes())
        {
            return Err(quota("snapshot exceeds the declared allowance"));
        }
        let charge = StateCharge::reserve(
            self.storage.backend.as_ref(),
            Arc::clone(&self.storage.charge.live),
            self.storage.limit,
            bytes,
        )?;
        let _copy_charge = StateCharge::reserve(
            self.storage.backend.as_ref(),
            Arc::clone(&self.storage.charge.live),
            self.storage.limit,
            layout.version_bytes(),
        )?;
        let mut image = StateSnapshot {
            backend: Arc::clone(&self.storage.backend),
            layout: layout.clone(),
            planes: Vec::new(),
            prompt,
            position,
            charge: Arc::new(charge),
            count,
        };
        for (plane, descriptor) in image.layout.planes().iter().enumerate() {
            image.planes.push(
                image
                    .backend
                    .alloc(descriptor.dtype(), descriptor.shape())
                    .map_err(guest_error)?,
            );
            let source = match &attached {
                Some(image) => image.planes[plane].clone(),
                None => self.storage.version_view(plane, record)?,
            };
            let data = image
                .backend
                .read_replay_output(&source)
                .map_err(guest_error)?;
            image
                .backend
                .write(&image.planes[plane], &data)
                .map_err(guest_error)?;
            image.planes[plane] = image.planes[plane].read_only();
        }
        drop(checkpoint);
        Ok(Arc::new(image))
    }
}

impl<B: Backend> Drop for StateSnapshot<B> {
    fn drop(&mut self) {
        for plane in &self.planes {
            let _ = self.backend.release(plane);
        }
    }
}

#[cfg(test)]
mod tests;

mod attach;
