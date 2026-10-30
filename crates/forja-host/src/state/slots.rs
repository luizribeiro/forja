use super::{
    LanePosition, StateBatch, StateRecording, StateSequenceHandle, StateStorage, pool_error,
};
use crate::{compute, invalid_handle};
use forja_core::Backend;
use std::sync::Arc;

struct Lane<B: Backend> {
    sequence: StateSequenceHandle<B>,
    serial: u64,
}

/// Two rebinding banks of lane transactions over one private state pool.
pub struct StateSlots<B: Backend> {
    storage: Arc<StateStorage<B>>,
    lanes: Vec<Lane<B>>,
    banks: [Vec<Option<StateRecording<B>>>; 2],
}

impl<B: Backend> StateSlots<B> {
    /// Reserves every lane's complete three-version state before serving admission.
    ///
    /// # Errors
    /// Refuses empty geometry, insufficient capacity or failed initialization.
    pub fn new(storage: Arc<StateStorage<B>>, lanes: u32) -> Result<Self, compute::Error> {
        if lanes == 0 {
            return Err(invalid_handle("state slots require active lane capacity"));
        }
        let lanes = (0..lanes)
            .map(|_| {
                Ok(Lane {
                    sequence: storage.reserve()?,
                    serial: 0,
                })
            })
            .collect::<Result<Vec<_>, compute::Error>>()?;
        let banks = std::array::from_fn(|_| (0..lanes.len()).map(|_| None).collect());
        Ok(Self {
            storage,
            lanes,
            banks,
        })
    }

    /// Reserves a fresh transaction in one free bank position.
    ///
    /// # Errors
    /// Refuses invalid slots, occupied banks, quota or unavailable state versions.
    pub fn begin(
        &mut self,
        bank: usize,
        lane: usize,
        position: LanePosition,
    ) -> Result<(), compute::Error> {
        let record = self
            .banks
            .get_mut(bank)
            .and_then(|bank| bank.get_mut(lane))
            .ok_or_else(|| invalid_handle("state bank or lane is out of range"))?;
        if record.is_some() {
            return Err(invalid_handle("state bank already has a lane transaction"));
        }
        let lane = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        let serial = lane
            .serial
            .checked_add(1)
            .ok_or_else(|| invalid_handle("state lane timeline exhausted"))?;
        let transaction = lane
            .sequence
            .begin_at(serial, position)
            .map_err(pool_error)?;
        lane.serial = serial;
        *record = Some(self.storage.record(&lane.sequence, transaction)?);
        Ok(())
    }

    /// Borrows an admitted transaction for authenticated plane recording or replay binding.
    ///
    /// # Errors
    /// Refuses a missing bank position or a lane that has not been admitted.
    pub fn recording(
        &mut self,
        bank: usize,
        lane: usize,
    ) -> Result<&mut StateRecording<B>, compute::Error> {
        self.banks
            .get_mut(bank)
            .and_then(|bank| bank.get_mut(lane))
            .and_then(Option::as_mut)
            .ok_or_else(|| invalid_handle("state lane is not admitted in this bank"))
    }

    /// Transfers complete lane transactions to one submission owner.
    ///
    /// # Errors
    /// Refuses invalid banks or incomplete updates without publishing any lane.
    pub fn take_batch(&mut self, bank: usize) -> Result<StateBatch<B>, compute::Error> {
        let bank = self
            .banks
            .get_mut(bank)
            .ok_or_else(|| invalid_handle("state bank is out of range"))?;
        StateBatch::new(bank.iter_mut().filter_map(Option::take).collect())
    }

    /// Releases unsubmitted reservations; submitted owners retain their own versions until drain.
    ///
    /// # Errors
    /// Refuses an invalid bank index.
    pub fn cancel_unsubmitted_bank(&mut self, bank: usize) -> Result<(), compute::Error> {
        let bank = self
            .banks
            .get_mut(bank)
            .ok_or_else(|| invalid_handle("state bank is out of range"))?;
        for record in bank {
            *record = None;
        }
        Ok(())
    }

    /// Returns the cursors committed with one lane's current state version.
    ///
    /// # Errors
    /// Refuses an invalid lane index.
    pub fn position(&self, lane: usize) -> Result<LanePosition, compute::Error> {
        self.lanes
            .get(lane)
            .map(|lane| lane.sequence.position())
            .ok_or_else(|| invalid_handle("state lane is out of range"))
    }

    /// Restores all cursors and state planes from a retained predecessor after drain.
    ///
    /// # Errors
    /// Refuses an invalid lane, missing predecessor or outstanding execution.
    pub fn rewind(&self, lane: usize, tokens: u64) -> Result<(), compute::Error> {
        self.lanes
            .get(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?
            .sequence
            .rewind(tokens)
            .map_err(pool_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, Limits};
    use forja_core::{DType, StateLayout, StatePlane};

    #[test]
    fn alternating_banks_keep_lane_predecessors_independent() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(131_072, 8, 4096, 128, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                2,
            )
            .unwrap();
        let bytes = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        assert!(StateSlots::new(Arc::clone(&storage), 3).is_err());
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            bytes
        );
        let mut slots = StateSlots::new(storage, 2).unwrap();
        let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let outputs = [
            host.backend.alloc(DType::F32, &[1, 7]).unwrap(),
            host.backend.alloc(DType::F32, &[1, 7]).unwrap(),
        ];
        host.backend
            .write(&input, &2_f32.to_le_bytes().repeat(7))
            .unwrap();
        host.backend
            .write(&weights, &1_f32.to_le_bytes().repeat(28))
            .unwrap();
        let position = |tokens| LanePosition {
            tokens,
            reserved_tokens: 128,
            ..LanePosition::default()
        };
        let mut submitted = Vec::new();
        for bank in 0..2 {
            for (lane, output) in outputs.iter().enumerate() {
                slots
                    .begin(bank, lane, position(u64::try_from(bank + 1).unwrap()))
                    .unwrap();
                slots
                    .recording(bank, lane)
                    .unwrap()
                    .causal_convolution(0, &input, &weights, output)
                    .unwrap();
            }
            submitted.push(slots.take_batch(bank).unwrap().submit().unwrap());
        }
        assert!(slots.begin(0, 0, position(3)).is_err());
        assert_eq!(slots.position(0).unwrap(), LanePosition::default());
        for submission in submitted {
            submission.finish().unwrap();
        }
        slots.rewind(0, 1).unwrap();
        assert_eq!(slots.position(0).unwrap(), position(1));
        assert_eq!(slots.position(1).unwrap(), position(2));
        slots.begin(0, 0, position(2)).unwrap();
        assert!(slots.begin(0, 0, position(3)).is_err());
        slots
            .recording(0, 0)
            .unwrap()
            .causal_convolution(0, &input, &weights, &outputs[0])
            .unwrap();
        slots.take_batch(0).unwrap().submit_and_finish().unwrap();
        assert_eq!(
            host.backend.read(&outputs[0]).unwrap(),
            4_f32.to_le_bytes().repeat(7)
        );
        assert!(slots.position(2).is_err());
        assert!(slots.recording(2, 0).is_err());
        slots.cancel_unsubmitted_bank(1).unwrap();
    }
}
