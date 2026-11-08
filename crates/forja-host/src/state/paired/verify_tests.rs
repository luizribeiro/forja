use super::*;
use crate::{LanePosition, StateAction, state::journal::StateJournal};
use forja_core::{DType, PagePool, PagedLayout, PagedPlane, StateLayout, StatePlane, StatePool};

#[test]
fn unpublished_verify_restores_paired_state_and_paged_input() {
    for rows in [2_u32, 3] {
        for outcome in [0, 1, 2, 3, 5] {
            let pool = StatePool::with_spares(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
                1,
                rows - 2,
                65536,
            )
            .unwrap();
            let mut sequence = pool.reserve().unwrap();
            sequence.reserve_spares().unwrap();
            let mut journal = StateJournal::new(sequence);
            journal.position.reserved_tokens = 16;
            let input = journal.checkpoint_latest().unwrap();
            let kv_pool = PagePool::new(
                PagedLayout::new(
                    4,
                    vec![PagedPlane {
                        dtype: DType::F32,
                        elements_per_token: 7,
                        alignment: 4,
                        label: "kv".into(),
                    }],
                )
                .unwrap(),
                4,
                4096,
            )
            .unwrap();
            let kv = Arc::new(Mutex::new(kv_pool.reserve(16).unwrap()));
            let dirty = std::array::from_fn(|_| Arc::new(AtomicBool::new(false)));
            let checkpoint = kv.lock().unwrap().checkpoint().unwrap();
            let caller = kv.lock().unwrap().checkpoint().unwrap();
            let tx = journal
                .begin_action(
                    1,
                    Some(LanePosition {
                        tokens: u64::from(rows),
                        reserved_tokens: 16,
                        ..LanePosition::default()
                    }),
                    Some(StateAction::Verify),
                )
                .unwrap();
            journal
                .attach_paged(
                    1,
                    PagedInput {
                        sequence: Arc::clone(&kv),
                        checkpoint,
                        dirty: dirty.clone(),
                    },
                )
                .unwrap();
            kv.lock().unwrap().cow_tail().unwrap();
            kv.lock().unwrap().commit_write(u64::from(rows)).unwrap();
            if outcome == 0 {
                journal.cancel_unsubmitted(&tx).unwrap();
            } else {
                journal.mark_submitted(&tx).unwrap();
                if outcome == 1 {
                    journal.abandon(&tx).unwrap();
                    assert_eq!(kv.lock().unwrap().logical_len(), u64::from(rows));
                    assert!(journal.begin(2, None).is_err());
                    pool.observe_completion(1).unwrap();
                    let retry = journal.begin(2, None).unwrap();
                    journal.cancel_unsubmitted(&retry).unwrap();
                } else {
                    pool.observe_completion(1).unwrap();
                    journal.finish(&tx, outcome != 5).unwrap();
                    if outcome == 5 {
                        assert_eq!(kv.lock().unwrap().logical_len(), 0);
                    } else {
                        assert_eq!(kv.lock().unwrap().logical_len(), u64::from(rows));
                        match outcome {
                            3 => journal
                                .restore_with(&input, || kv.lock().unwrap().rollback(caller))
                                .unwrap(),
                            _ => journal.cancel_unsubmitted(&tx).unwrap(),
                        }
                    }
                }
            }
            assert_eq!(journal.position, input.position);
            assert_eq!(kv.lock().unwrap().logical_len(), 0);
            if outcome != 3 {
                assert!(dirty.iter().all(|flag| flag.load(Ordering::Acquire)));
            }
            assert_eq!(
                journal.checkpoint_latest().unwrap().version.version(),
                input.version.version()
            );
        }
    }
}
