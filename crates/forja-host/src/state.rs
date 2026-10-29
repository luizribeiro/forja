//! Opaque state storage charged to the runner's allocation budget.

mod record;
pub use record::StateRecording;
#[cfg(test)]
mod quota_tests;
mod submit;
pub use submit::StateSubmission;

use crate::{Host, Limits, compute, guest_error, invalid_handle, quota};
use forja_core::{Backend, Layout, PagePoolError, StateLayout, StatePool, StateSequence, Tensor};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

const POOL_METADATA_RESERVE: u64 = 4096;
const RECORD_METADATA_RESERVE: u64 = 128;
const SEQUENCE_METADATA_RESERVE: u64 = 1024;

struct StateCharge {
    bytes: u64,
    live: Arc<AtomicU64>,
}

impl StateCharge {
    fn reserve(live: Arc<AtomicU64>, limit: u64, bytes: u64) -> Result<Self, compute::Error> {
        live.try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
            used.checked_add(bytes).filter(|&sum| sum <= limit)
        })
        .map_err(|_| quota("state storage exceeds the live byte quota"))?;
        Ok(Self { bytes, live })
    }
}

impl Drop for StateCharge {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Host-owned state backing; callers receive sequence capabilities, never plane tensors.
pub struct StateStorage<B: Backend> {
    backend: Arc<B>,
    pool: StatePool,
    planes: Vec<Tensor>,
    charge: StateCharge,
    limit: u64,
    limits: Limits,
    last_submission: Mutex<u64>,
    poisoned: AtomicBool,
}

impl<B: Backend + 'static> Host<B> {
    /// Creates an opaque pool within the existing runner limits.
    ///
    /// # Errors
    /// Refuses invalid geometry or quota, releasing every partial allocation on failure.
    pub fn create_state_storage(
        &self,
        layout: StateLayout,
        sequences: u32,
    ) -> Result<Arc<StateStorage<B>>, compute::Error> {
        let records = sequences
            .checked_mul(3)
            .ok_or_else(|| quota("state record count overflow"))?;
        if usize::try_from(records).map_or(true, |count| count > self.limits.table_elements) {
            return Err(quota("state records exceed the table limit"));
        }
        for plane in layout.planes() {
            self.check_tensor_shape(plane.shape())?;
            let elements = u32::try_from(plane.elements())
                .map_err(|_| quota("state plane exceeds the tensor limit"))?;
            self.check_tensor_shape(&[records, elements])?;
        }
        let descriptor_bytes = u64::try_from(
            size_of::<forja_core::StatePlane>()
                + forja_core::MAX_RANK * 4
                + 2 * (size_of::<forja_core::PagedPlane>() + 64),
        )
        .map_err(|_| quota("state descriptor size overflow"))?
        .checked_mul(
            u64::try_from(layout.planes().len())
                .map_err(|_| quota("state plane count overflow"))?,
        )
        .and_then(|n| n.checked_add(POOL_METADATA_RESERVE))
        .ok_or_else(|| quota("state metadata size overflow"))?;
        let bytes = layout
            .version_bytes()
            .checked_add(4)
            .and_then(|n| n.checked_mul(u64::from(records)))
            .and_then(|n| n.checked_add(u64::from(records) * RECORD_METADATA_RESERVE))
            .and_then(|n| n.checked_add(u64::from(sequences) * SEQUENCE_METADATA_RESERVE))
            .and_then(|n| n.checked_add(descriptor_bytes))
            .ok_or_else(|| quota("state byte count overflow"))?;
        let charge =
            StateCharge::reserve(Arc::clone(&self.live_bytes), self.limits.live_bytes, bytes)?;
        let pool = StatePool::new(layout, sequences, bytes).map_err(pool_error)?;
        let mut storage = StateStorage {
            backend: Arc::clone(&self.backend),
            pool,
            planes: Vec::new(),
            charge,
            limit: self.limits.live_bytes,
            limits: self.limits,
            last_submission: Mutex::new(0),
            poisoned: AtomicBool::new(false),
        };
        for plane in storage.pool.layout().planes() {
            let elements = u32::try_from(plane.elements())
                .map_err(|_| quota("state plane exceeds the tensor limit"))?;
            storage.planes.push(
                storage
                    .backend
                    .alloc(plane.dtype(), &[records, elements])
                    .map_err(guest_error)?,
            );
        }
        Ok(Arc::new(storage))
    }
}

impl<B: Backend> StateStorage<B> {
    /// Reserves all versions and clears every byte before assigning a new sequence owner.
    ///
    /// # Errors
    /// Returns quota or backend failure without exposing an incompletely initialized sequence.
    pub fn reserve(&self) -> Result<StateSequence, compute::Error> {
        self.check_usable()?;
        let sequence = self.pool.reserve().map_err(pool_error)?;
        let table = sequence.allocation_table().map_err(pool_error)?;
        for (plane, descriptor) in self.pool.layout().planes().iter().enumerate() {
            let size = descriptor
                .elements()
                .checked_mul(descriptor.dtype().byte_size())
                .ok_or_else(|| quota("state initialization size overflow"))?;
            let _initialization_charge =
                StateCharge::reserve(Arc::clone(&self.charge.live), self.limit, size)?;
            let size = usize::try_from(size)
                .map_err(|_| quota("state initialization exceeds addressable memory"))?;
            let mut zeros = Vec::new();
            zeros
                .try_reserve_exact(size)
                .map_err(|_| quota("state initialization allocation failed"))?;
            zeros.resize(size, 0);
            for entry in table.entries() {
                self.backend
                    .write(&self.version_view(plane, entry.page())?, &zeros)
                    .map_err(guest_error)?;
            }
        }
        Ok(sequence)
    }

    fn check_usable(&self) -> Result<(), compute::Error> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(invalid_handle(
                "state storage requires reload after an uncertain submission",
            ))
        } else {
            Ok(())
        }
    }

    fn version_view(&self, plane: usize, record: u32) -> Result<Tensor, compute::Error> {
        let descriptor = self
            .pool
            .layout()
            .planes()
            .get(plane)
            .ok_or_else(|| invalid_handle("unknown state plane"))?;
        let base = self
            .planes
            .get(plane)
            .ok_or_else(|| invalid_handle("missing state plane backing"))?;
        let offset = u64::from(record)
            .checked_mul(descriptor.elements())
            .ok_or_else(|| quota("state view offset overflow"))?;
        let layout = Layout::contiguous(
            descriptor.dtype(),
            offset,
            descriptor.shape().to_vec(),
            base.buffer().byte_len(),
        )
        .map_err(|error| invalid_handle(error.to_string()))?;
        self.backend
            .tensor(base.buffer(), layout)
            .map_err(guest_error)
    }

    /// Bytes retained by the backing and its conservatively reserved allocator metadata.
    #[must_use]
    pub const fn charged_bytes(&self) -> u64 {
        self.charge.bytes
    }

    /// Immutable byte ceiling inherited from the runner.
    #[must_use]
    pub const fn byte_limit(&self) -> u64 {
        self.limit
    }
}

impl<B: Backend> Drop for StateStorage<B> {
    fn drop(&mut self) {
        for plane in &self.planes {
            let _ = self.backend.release(plane);
        }
    }
}

fn pool_error(error: PagePoolError) -> compute::Error {
    if error == PagePoolError::QuotaExceeded {
        quota(&error.to_string())
    } else {
        invalid_handle(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use forja_core::{DType, StatePlane};

    #[test]
    fn private_state_backing_is_charged_and_cleared_before_reuse() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(8192, 8, 4096, 128, 4096),
        );
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
        let storage = host.create_state_storage(layout.clone(), 2).unwrap();
        let charged = storage.charged_bytes();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), charged);
        assert!(host.create_state_storage(layout, 2).is_err());
        assert_eq!(host.live_bytes.load(Ordering::Acquire), charged);
        let sequence = storage.reserve().unwrap();
        let table = sequence.allocation_table().unwrap();
        for entry in table.entries() {
            host.backend
                .write(&storage.version_view(0, entry.page()).unwrap(), &[0x3f; 28])
                .unwrap();
        }
        drop(sequence);
        let replacement = storage.reserve().unwrap();
        for entry in replacement.allocation_table().unwrap().entries() {
            assert_eq!(
                host.backend
                    .read(&storage.version_view(0, entry.page()).unwrap())
                    .unwrap(),
                [0; 28]
            );
        }
        drop(replacement);
        drop(storage);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
}
