use super::*;
use crate::{LanePosition, StateAction, state::journal::StateJournal};
use forja_core::{DType, PagePool, PagedLayout, PagedPlane, StateLayout, StatePlane, StatePool};

#[test]
fn unpublished_verify_restores_paired_state_and_paged_input() {
    paired_verify_cases(|rows| 0..5 + rows);
}

#[test]
fn shared_verify_tail_refuses_publication_without_changing_boundary() {
    paired_verify_cases(|rows| 5 + rows..6 + rows);
}

fn paired_verify_cases(outcomes: impl Fn(u32) -> std::ops::Range<u32>) {
    for rows in [2_u32, 3] {
        for (outcome, start) in
            outcomes(rows).flat_map(|outcome| [0_u64, 3, 4, 7, 33].map(|start| (outcome, start)))
        {
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
            journal.position = LanePosition {
                tokens: start,
                reserved_tokens: 64,
                ..LanePosition::default()
            };
            let input = journal.checkpoint_latest().unwrap();
            let kv_pool = paged_pool();
            let kv = Arc::new(Mutex::new(kv_pool.reserve(64).unwrap()));
            let dirty = std::array::from_fn(|_| Arc::new(AtomicBool::new(false)));
            kv.lock().unwrap().commit_write(start).unwrap();
            let checkpoint = kv.lock().unwrap().checkpoint().unwrap();
            let caller = kv.lock().unwrap().checkpoint().unwrap();
            let tx = journal
                .begin_action(
                    1,
                    Some(LanePosition {
                        tokens: start + u64::from(rows),
                        reserved_tokens: 64,
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
                    assert_eq!(kv.lock().unwrap().logical_len(), start + u64::from(rows));
                    assert!(journal.begin(2, None).is_err());
                    pool.observe_completion(1).unwrap();
                    let retry = journal.begin(2, None).unwrap();
                    journal.cancel_unsubmitted(&retry).unwrap();
                } else {
                    pool.observe_completion(1).unwrap();
                    journal.finish(&tx, outcome != 5).unwrap();
                    if outcome == 5 {
                        assert_eq!(kv.lock().unwrap().logical_len(), start);
                    } else {
                        assert_eq!(kv.lock().unwrap().logical_len(), start + u64::from(rows));
                        match outcome {
                            _ if outcome == 5 + rows => {
                                refuse_shared_paged_cut(&mut journal, &tx, &kv, &dirty);
                            }
                            2 => journal.cancel_unsubmitted(&tx).unwrap(),
                            3 => journal
                                .restore_with(&input, || kv.lock().unwrap().rollback(caller))
                                .unwrap(),
                            _ => {
                                let accepted = if outcome == 4 { rows - 1 } else { outcome - 6 };
                                publish_paged_cut(&mut journal, &tx, &kv, &dirty, rows, accepted);
                            }
                        }
                    }
                }
            }
            assert_eq!(journal.position, input.position);
            assert_eq!(kv.lock().unwrap().logical_len(), start);
            if outcome != 3 {
                assert!(dirty.iter().all(|flag| flag.load(Ordering::Acquire)));
            }
            assert!(journal.publish_accepted(&tx, 0).is_err());
            assert_eq!(
                journal.checkpoint_latest().unwrap().version.version(),
                input.version.version()
            );
        }
    }
}

fn paged_pool() -> PagePool {
    PagePool::new(
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
        16,
        4096,
    )
    .unwrap()
}

fn publish_paged_cut(
    journal: &mut StateJournal,
    tx: &forja_core::StateTransaction,
    kv: &Arc<Mutex<PagedSequence>>,
    dirty: &[Arc<AtomicBool>; 2],
    rows: u32,
    accepted: u32,
) {
    let input = journal.position;
    let start = input.tokens;
    let end = start + u64::from(rows);
    let committed = journal.checkpoint().unwrap().version();
    kv.lock().unwrap().rewind_private(end - 1).unwrap();
    assert_eq!(
        journal.publish_accepted(tx, accepted),
        Err(PagePoolError::InvalidCheckpoint)
    );
    assert_eq!(journal.checkpoint().unwrap().version(), committed);
    assert_eq!(journal.position, input);
    assert!(dirty.iter().all(|flag| !flag.load(Ordering::Acquire)));
    kv.lock().unwrap().commit_write(1).unwrap();
    assert!(journal.publish_accepted(tx, rows).is_err());
    journal.publish_accepted(tx, accepted).unwrap();
    let tokens = start + u64::from(accepted) + 1;
    assert_eq!(journal.position.tokens, tokens);
    assert_eq!(kv.lock().unwrap().logical_len(), tokens);
    assert_eq!(kv.lock().unwrap().reserved_tokens(), 64);
    assert!(
        dirty
            .iter()
            .all(|flag| flag.load(Ordering::Acquire) == (accepted + 1 < rows))
    );
    journal.rewind(start).unwrap();
}

fn refuse_shared_paged_cut(
    journal: &mut StateJournal,
    tx: &forja_core::StateTransaction,
    kv: &Arc<Mutex<PagedSequence>>,
    dirty: &[Arc<AtomicBool>; 2],
) {
    let committed = journal.checkpoint().unwrap().version();
    let position = journal.position;
    let tokens = kv.lock().unwrap().logical_len();
    let sibling = kv.lock().unwrap().fork(tokens).unwrap();
    dirty[0].store(true, Ordering::Release);
    let flags = dirty.each_ref().map(|flag| flag.load(Ordering::Acquire));
    assert_eq!(
        journal.publish_accepted(tx, 0),
        Err(PagePoolError::SharedTail)
    );
    assert_eq!(journal.checkpoint().unwrap().version(), committed);
    assert_eq!(journal.position, position);
    assert_eq!(
        dirty.each_ref().map(|flag| flag.load(Ordering::Acquire)),
        flags
    );
    assert_eq!(kv.lock().unwrap().logical_len(), tokens);
    assert_eq!(sibling.logical_len(), tokens);
    drop(sibling);
    journal.cancel_unsubmitted(tx).unwrap();
}

fn accepted_paged_append<B: Backend + Send + Sync + 'static>(backend: impl Fn() -> B) {
    for rows in [2, 3] {
        for accepted in 0..rows {
            let (mut host, pool, slot, _) = crate::tests::lifecycle_fixture_backend(backend());
            let sequence = host.state_slots[&slot.rep()]
                .sequence_handle(0)
                .unwrap()
                .unwrap();
            assert!(sequence.try_reserve_spares().unwrap());
            let binding = Arc::clone(&host.table.get(&slot).unwrap().binding);
            let kv = Arc::clone(
                binding
                    .lane(0)
                    .unwrap()
                    .sequence
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap(),
            );
            host.begin_paged_lane_update(pool.rep(), 0, 0, rows, Some(StateAction::Verify))
                .unwrap();
            assert_eq!(binding.banks.append_window(0, 0), Ok(0..u64::from(rows)));
            let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
            let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
            let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
            host.backend
                .write(&input, &2_f32.to_le_bytes().repeat(7))
                .unwrap();
            host.backend
                .write(&weights, &1_f32.to_le_bytes().repeat(28))
                .unwrap();
            let slots = host.state_slots.get_mut(&slot.rep()).unwrap();
            let recording = slots.recording(0, 0).unwrap();
            let transaction = recording.transaction.clone();
            for _ in 0..rows {
                recording
                    .causal_convolution(0, &input, &weights, &output)
                    .unwrap();
            }
            slots.take_batch(0).unwrap().submit_and_finish().unwrap();
            sequence.publish_accepted(&transaction, accepted).unwrap();
            let tokens = accepted + 1;
            assert_eq!(kv.lock().unwrap().logical_len(), u64::from(tokens));
            for bank in 0..2 {
                host.table.get_mut(&slot).unwrap().active_bank = bank;
                let (checkpoint, _) = host
                    .begin_paged_lane_update(pool.rep(), 0, tokens, 1, Some(StateAction::Combined))
                    .unwrap();
                assert_eq!(
                    binding.banks.append_window(bank, 0),
                    Ok(u64::from(tokens)..u64::from(tokens + 1))
                );
                assert_eq!(
                    binding.banks.slot(bank, 0).unwrap().logical_len(),
                    u64::from(tokens + 1)
                );
                host.rollback_paged_lane_append(pool.rep(), 0, checkpoint.unwrap())
                    .unwrap();
                assert_eq!(sequence.position().tokens, u64::from(tokens));
                assert_eq!(kv.lock().unwrap().logical_len(), u64::from(tokens));
            }
            sequence.rewind(0).unwrap();
            assert_eq!(kv.lock().unwrap().logical_len(), 0);
        }
    }
}

#[test]
fn accepted_verify_reopens_paged_windows_at_the_published_boundary() {
    accepted_paged_append(forja_cpu::CpuBackend::new);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_accepted_verify_reopens_paged_windows_at_the_published_boundary() {
    accepted_paged_append(|| forja_metal::MetalBackend::new().unwrap());
}
