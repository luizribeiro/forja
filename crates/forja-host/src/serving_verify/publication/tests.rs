use super::*;
use crate::{BatchSelection, BatchShape, EngineBatchLane, SamplingParams, StateAction};
use forja_core::SequenceCheckpoint;
use forja_cpu::CpuBackend;

struct Fixture {
    host: Host<CpuBackend>,
    pool: Resource<PagePoolEntry>,
    slot: Resource<BatchSlotEntry>,
    publication: VerifyPublication,
    checkpoint: SequenceCheckpoint,
}

impl Fixture {
    fn new(rows: u32, protected: bool) -> Self {
        Self::new_at(rows, protected, 0)
    }

    fn new_at(rows: u32, protected: bool, start: u32) -> Self {
        let (mut host, pool, slot) =
            crate::tests::verify_input_fixture_with_plane(CpuBackend::new(), rows, 2, vec![3, 7]);
        if !protected {
            host.state_slots.remove(&slot.rep());
        }
        host.bind_paged_lane(pool.rep(), 1, None, 33).unwrap();
        assert!(host.reserve_draft_lane(pool.rep(), 1).unwrap());
        if start != 0 {
            host.begin_paged_lane_update(pool.rep(), 1, 0, start, Some(StateAction::Step))
                .unwrap();
            let value = host
                .backend
                .alloc(forja_core::DType::F32, &[start, 7])
                .unwrap();
            let weights = host.backend.alloc(forja_core::DType::F32, &[4, 7]).unwrap();
            let output = host
                .backend
                .alloc(forja_core::DType::F32, &[start, 7])
                .unwrap();
            let state = host.state_slots.get_mut(&slot.rep()).unwrap();
            state
                .recording(0, 1)
                .unwrap()
                .causal_convolution(0, &value, &weights, &output)
                .unwrap();
            state.take_batch(0).unwrap().submit_and_finish().unwrap();
        }
        let input = EngineBatch {
            shape: BatchShape {
                phase: BatchPhase::Verify,
                selection: BatchSelection::Combined,
                bucket: 1,
                tokens: rows,
            },
            lanes: vec![Some(EngineBatchLane {
                slot: 1,
                tokens: Some(vec![1; usize::try_from(rows).unwrap()]),
                start_pos: start,
                sampling: SamplingParams::default(),
            })],
        };
        let (checkpoint, _) = host
            .begin_paged_lane_update(pool.rep(), 1, start, rows, Some(StateAction::Verify))
            .unwrap();
        let publication = host
            .verify_publication(pool.rep(), &input)
            .unwrap()
            .unwrap();
        Self {
            host,
            pool,
            slot,
            publication,
            checkpoint: checkpoint.unwrap(),
        }
    }

    fn finish(&mut self) {
        let value = self
            .host
            .backend
            .alloc(forja_core::DType::F32, &[1, 7])
            .unwrap();
        let weights = self
            .host
            .backend
            .alloc(forja_core::DType::F32, &[4, 7])
            .unwrap();
        let output = self
            .host
            .backend
            .alloc(forja_core::DType::F32, &[1, 7])
            .unwrap();
        let state = self.host.state_slots.get_mut(&self.slot.rep()).unwrap();
        for _ in 0..self.publication.rows {
            state
                .recording(0, 1)
                .unwrap()
                .causal_convolution(0, &value, &weights, &output)
                .unwrap();
        }
        state.take_batch(0).unwrap().submit_and_finish().unwrap();
    }

    fn logical_len(&self) -> u64 {
        self.host
            .paged_lane_prefix(self.pool.rep(), 1)
            .unwrap()
            .logical_len()
            .unwrap()
    }
    fn rollback(self) {
        let Self {
            mut host,
            pool,
            checkpoint,
            ..
        } = self;
        host.rollback_paged_lane_append(pool.rep(), 1, checkpoint)
            .unwrap();
        assert_eq!(
            host.paged_lane_prefix(pool.rep(), 1)
                .unwrap()
                .logical_len()
                .unwrap(),
            0
        );
    }
}

#[test]
fn constructors_bind_the_active_physical_lane_and_transaction() {
    for protected in [false, true] {
        let mut fixture = Fixture::new(3, protected);
        assert_eq!(fixture.publication.lane, 1);
        assert_eq!(fixture.publication.rows, 3);
        assert_eq!(fixture.publication.state.is_some(), protected);
        assert_eq!(
            fixture.host.state_slots.contains_key(&fixture.slot.rep()),
            protected
        );
        assert_eq!(fixture.logical_len(), 3);
        if protected {
            fixture.finish();
        }
        fixture.rollback();
    }
}
