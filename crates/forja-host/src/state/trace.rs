use super::{StateCharge, StateCommands, StateStorage};
use crate::{compute, invalid_handle};
use forja_core::{Backend, StateAccess};
use std::{
    any::Any,
    sync::{Arc, Mutex, MutexGuard},
};

/// Independent inactive lane recordings for one graph trace, without request leases.
pub struct StateTrace<B: Backend> {
    lanes: Vec<StateCommands<B>>,
    charge: Arc<StateCharge>,
}

/// Shared trace ownership suitable for a backend-independent component resource.
#[derive(Clone)]
pub struct SharedStateTrace {
    trace: Arc<dyn Any + Send + Sync>,
    _charge: Arc<StateCharge>,
}

impl std::fmt::Debug for SharedStateTrace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SharedStateTrace")
            .finish_non_exhaustive()
    }
}

impl<B: Backend + Send + Sync + 'static> StateTrace<B> {
    /// Shares this trace while retaining its metadata charge through the final reference.
    #[must_use]
    pub fn share(self) -> SharedStateTrace {
        let charge = Arc::clone(&self.charge);
        SharedStateTrace {
            trace: Arc::new(Mutex::new(Some(self))),
            _charge: charge,
        }
    }
}

impl SharedStateTrace {
    /// Borrows the original backend's trace, or its empty state after capture retirement.
    ///
    /// # Errors
    /// Refuses a different backend type or a poisoned lock.
    pub fn lock<B: Backend + 'static>(
        &self,
    ) -> Result<MutexGuard<'_, Option<StateTrace<B>>>, compute::Error> {
        self.trace
            .downcast_ref::<Mutex<Option<StateTrace<B>>>>()
            .ok_or_else(|| invalid_handle("state trace belongs to another backend"))?
            .lock()
            .map_err(|_| invalid_handle("state trace lock is poisoned"))
    }
}

impl<B: Backend> StateTrace<B> {
    pub(super) fn new(
        storage: &Arc<StateStorage<B>>,
        access: StateAccess,
        lanes: usize,
    ) -> Result<Self, compute::Error> {
        if lanes == 0 {
            return Err(invalid_handle("state trace requires lanes"));
        }
        let bytes = u64::try_from(lanes)
            .ok()
            .and_then(|lanes| lanes.checked_mul(size_of::<StateCommands<B>>() as u64))
            .and_then(|bytes| bytes.checked_add(1024))
            .ok_or_else(|| invalid_handle("state trace metadata overflow"))?;
        let charge = StateCharge::reserve(
            storage.backend.as_ref(),
            Arc::clone(&storage.charge.live),
            storage.limit,
            bytes,
        )?;
        let lanes = (0..lanes)
            .map(|_| storage.inactive_commands(access))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            lanes,
            charge: Arc::new(charge),
        })
    }

    /// Borrows one trace's poisoned lane bindings, independent of all active transactions.
    ///
    /// # Errors
    /// Refuses an out-of-range lane.
    pub fn lane(&mut self, lane: usize) -> Result<&mut StateCommands<B>, compute::Error> {
        self.lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state trace lane is out of range"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, LanePosition, Limits, StateBatch, StateSlots};
    use forja_core::{DType, StateLayout, StatePlane};

    #[test]
    fn shared_traces_keep_metadata_charged_after_typed_capture_retirement() {
        use std::sync::atomic::Ordering;
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(131_072, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                1,
            )
            .unwrap();
        let slots = StateSlots::new(storage, 1).unwrap();
        let before = host.live_bytes.load(Ordering::Acquire);
        let shared = slots.trace().unwrap().share();
        let last = shared.clone();
        drop(shared);
        #[cfg(target_os = "macos")]
        assert!(last.lock::<forja_metal::MetalBackend>().is_err());
        drop(last.lock::<forja_cpu::CpuBackend>().unwrap().take());
        assert!(last.lock::<forja_cpu::CpuBackend>().unwrap().is_none());
        assert!(host.live_bytes.load(Ordering::Acquire) > before);
        drop(last);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    }

    #[test]
    fn interleaved_traces_rebind_their_own_metadata_without_request_leases() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(131_072, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                1,
            )
            .unwrap();
        let mut slots = StateSlots::new(Arc::clone(&storage), 1).unwrap();
        let before = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        let mut first = slots.trace().unwrap();
        assert!(StateTrace::new(&storage, first.lane(0).unwrap().access, usize::MAX).is_err());
        let mut second = slots.trace().unwrap();
        let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        host.backend
            .write(&input, &2_f32.to_ne_bytes().repeat(7))
            .unwrap();
        host.backend
            .write(&weights, &1_f32.to_ne_bytes().repeat(28))
            .unwrap();
        first
            .lane(0)
            .unwrap()
            .causal_convolution(0, &input, &weights, &output)
            .unwrap();
        second
            .lane(0)
            .unwrap()
            .causal_convolution(0, &input, &weights, &output)
            .unwrap();
        assert!(first.lane(1).is_err());
        assert!(slots.sequence_handle(0).unwrap().is_none());
        let capture = first.lane(0).unwrap().capture().unwrap();
        slots
            .begin(
                0,
                0,
                LanePosition {
                    tokens: 1,
                    reserved_tokens: 7,
                    ..LanePosition::default()
                },
            )
            .unwrap();
        let bindings = capture.bind(slots.recording(0, 0).unwrap()).unwrap();
        let commands = bindings
            .apply(std::mem::take(&mut first.lane(0).unwrap().commands))
            .unwrap();
        let batch: StateBatch<_> = slots.take_batch(0).unwrap();
        batch
            .submit_using(|owners| host.backend.submit_retained(commands, owners))
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(
            host.backend.read(&output).unwrap(),
            2_f32.to_ne_bytes().repeat(7)
        );
        slots.release(0).unwrap();
        assert!(slots.sequence_handle(0).unwrap().is_none());
        drop(capture);
        drop(first);
        drop(second);
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            before
        );
    }
}
