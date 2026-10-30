#[cfg(test)]
mod inactive_tests;

use super::{
    LanePosition, StateBatch, StateCommands, StateRecording, StateSequenceHandle, StateStorage,
    pool_error,
};
use crate::{compute, invalid_handle};
use forja_core::{Backend, StateAccess};
use std::sync::Arc;

struct Lane<B: Backend> {
    sequence: Option<StateSequenceHandle<B>>,
    serial: u64,
}

/// Two rebinding banks of lane transactions over one private state pool.
pub struct StateSlots<B: Backend> {
    storage: Arc<StateStorage<B>>,
    inactive_access: StateAccess,
    lanes: Vec<Lane<B>>,
    banks: [Vec<Option<StateRecording<B>>>; 2],
    padding: [Vec<Option<StateCommands<B>>>; 2],
}

impl<B: Backend> StateSlots<B> {
    /// Creates banks within the pool's available lane capacity.
    ///
    /// # Errors
    /// Refuses empty geometry, insufficient capacity or failed initialization.
    pub fn new(storage: Arc<StateStorage<B>>, lanes: u32) -> Result<Self, compute::Error> {
        if lanes == 0 {
            return Err(invalid_handle("state slots require active lane capacity"));
        }
        let mut reserved = (0..lanes)
            .map(|_| storage.pool.reserve().map_err(pool_error))
            .collect::<Result<Vec<_>, _>>()?;
        let transaction = reserved[0].begin(1).map_err(pool_error)?;
        let inactive_access =
            StateAccess::from_slot(&reserved[0].bind(&transaction).map_err(pool_error)?)
                .map_err(pool_error)?;
        reserved[0]
            .cancel_unsubmitted(&transaction)
            .map_err(pool_error)?;
        let lanes = reserved
            .into_iter()
            .map(|_reservation| Lane {
                sequence: None,
                serial: 0,
            })
            .collect::<Vec<_>>();
        let banks = std::array::from_fn(|_| (0..lanes.len()).map(|_| None).collect());
        let padding = std::array::from_fn(|_| (0..lanes.len()).map(|_| None).collect());
        Ok(Self {
            storage,
            inactive_access,
            lanes,
            banks,
            padding,
        })
    }

    /// Creates poisoned, inactive command bindings without reserving a request lease.
    ///
    /// # Errors
    /// Refuses quota or metadata allocation failure.
    pub fn inactive(&self) -> Result<StateCommands<B>, compute::Error> {
        self.storage.inactive_commands(self.inactive_access)
    }

    /// Reserves and zeroes all three versions before admitting a new request.
    ///
    /// # Errors
    /// Refuses invalid lanes, quota or failed initialization without replacing an active lease.
    pub fn admit(&mut self, lane: usize) -> Result<(), compute::Error> {
        let lane = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        if lane.sequence.is_none() {
            lane.sequence = Some(self.storage.reserve()?);
            lane.serial = 0;
        }
        Ok(())
    }

    /// Releases a request lease after its final reader/writer and speculative descendants drain.
    ///
    /// # Errors
    /// Refuses invalid lanes or pending reservations/execution; committed state stays unchanged.
    pub fn release(&mut self, lane: usize) -> Result<(), compute::Error> {
        if self
            .banks
            .iter()
            .any(|bank| bank.get(lane).is_some_and(Option::is_some))
        {
            return Err(invalid_handle("state lane has an unsubmitted transaction"));
        }
        let lane = self
            .lanes
            .get_mut(lane)
            .ok_or_else(|| invalid_handle("state lane is out of range"))?;
        if let Some(sequence) = &lane.sequence {
            let mut journal = sequence.lock();
            let checkpoint = journal.checkpoint().map_err(pool_error)?;
            forja_core::StateSequence::rewind(&mut journal, &checkpoint).map_err(pool_error)?;
        }
        lane.sequence = None;
        Ok(())
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
            .get(bank)
            .and_then(|bank| bank.get(lane))
            .ok_or_else(|| invalid_handle("state bank or lane is out of range"))?;
        if record.is_some() || position.tokens > position.reserved_tokens {
            return Err(invalid_handle(
                "state bank is occupied or its cursor exceeds capacity",
            ));
        }
        let fresh = self.lanes[lane].sequence.is_none();
        self.admit(lane)?;
        let owner = &mut self.lanes[lane];
        let sequence = owner
            .sequence
            .as_ref()
            .ok_or_else(|| invalid_handle("state lane has no lease"))?;
        let serial = owner
            .serial
            .checked_add(1)
            .ok_or_else(|| invalid_handle("state lane timeline exhausted"))?;
        let transaction = sequence.begin_at(serial, position).map_err(pool_error)?;
        owner.serial = serial;
        match self.storage.record(sequence, transaction) {
            Ok(record) => {
                self.padding[bank][lane] = None;
                self.banks[bank][lane] = Some(record);
            }
            Err(error) => {
                if fresh {
                    owner.sequence = None;
                }
                return Err(error);
            }
        }
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

    /// Borrows authenticated commands for an active lane or poisoned commands for padding.
    ///
    /// # Errors
    /// Refuses invalid geometry or an activity flag inconsistent with the admitted bank.
    pub fn commands(
        &mut self,
        bank: usize,
        lane: usize,
        active: bool,
    ) -> Result<&mut StateCommands<B>, compute::Error> {
        let admitted = self
            .banks
            .get(bank)
            .and_then(|bank| bank.get(lane))
            .ok_or_else(|| invalid_handle("state bank or lane is out of range"))?
            .is_some();
        if active != admitted {
            return Err(invalid_handle(
                "state activity differs from its transaction",
            ));
        }
        if active {
            return self.recording(bank, lane).map(|record| &mut **record);
        }
        if self.padding[bank][lane].is_none() {
            self.padding[bank][lane] = Some(self.inactive()?);
        }
        self.padding[bank][lane]
            .as_mut()
            .ok_or_else(|| invalid_handle("inactive state bindings are missing"))
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

    /// Returns committed cursors; an inactive lane has zero cursors.
    ///
    /// # Errors
    /// Refuses an invalid lane index.
    pub fn position(&self, lane: usize) -> Result<LanePosition, compute::Error> {
        self.lanes
            .get(lane)
            .map(|lane| {
                lane.sequence
                    .as_ref()
                    .map_or_else(LanePosition::default, StateSequenceHandle::position)
            })
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
            .as_ref()
            .ok_or_else(|| invalid_handle("state lane is not admitted"))?
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
        assert!(slots.commands(0, 0, true).is_err());
        assert!(slots.commands(2, 0, false).is_err());
        assert!(slots.commands(0, 2, false).is_err());
        slots.commands(0, 0, false).unwrap();
        slots.commands(1, 0, false).unwrap();
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
        slots.release(0).unwrap();
        slots.release(1).unwrap();
        assert_eq!(slots.position(0).unwrap(), LanePosition::default());
        slots.admit(0).unwrap();
        slots.begin(1, 0, position(1)).unwrap();
        assert!(slots.release(0).is_err());
        slots
            .recording(1, 0)
            .unwrap()
            .causal_convolution(0, &input, &weights, &outputs[0])
            .unwrap();
        let pending = slots.take_batch(1).unwrap().submit().unwrap();
        assert!(slots.release(0).is_err());
        pending.finish().unwrap();
        assert_eq!(
            host.backend.read(&outputs[0]).unwrap(),
            2_f32.to_le_bytes().repeat(7)
        );
        slots.release(0).unwrap();
    }
}
