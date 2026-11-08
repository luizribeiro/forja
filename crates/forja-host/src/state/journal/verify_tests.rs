use super::*;
use forja_core::{DType, StateLayout, StatePlane, StatePool};

fn fixture(rows: u32) -> (StatePool, StateJournal) {
    let pool = StatePool::with_spares(
        StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap(),
        2,
        rows - 2,
        65536,
    )
    .unwrap();
    let mut sequence = pool.reserve().unwrap();
    sequence.reserve_spares().unwrap();
    let mut journal = StateJournal::new(sequence);
    let plain = journal
        .begin_action(1, Some(position(33)), Some(StateAction::Combined))
        .unwrap();
    journal.mark_submitted(&plain).unwrap();
    pool.observe_completion(1).unwrap();
    journal.finish(&plain, true).unwrap();
    (pool, journal)
}

fn position(tokens: u64) -> LanePosition {
    LanePosition {
        tokens,
        reserved_tokens: 4097,
        ..LanePosition::default()
    }
}

fn verify(journal: &mut StateJournal, serial: u64, rows: u32) -> StateTransaction {
    journal
        .begin_action(
            serial,
            Some(position(journal.position.tokens + u64::from(rows))),
            Some(StateAction::Verify),
        )
        .unwrap()
}

#[test]
fn verify_requires_a_drained_selected_source_and_checked_row_count() {
    let (pool, mut journal) = fixture(2);
    let forward = journal
        .begin_action(2, Some(position(34)), Some(StateAction::Forward))
        .unwrap();
    assert!(
        journal
            .begin_action(3, Some(position(36)), Some(StateAction::Verify))
            .is_err()
    );
    journal.mark_submitted(&forward).unwrap();
    pool.observe_completion(2).unwrap();
    journal.finish(&forward, true).unwrap();
    assert!(
        journal
            .begin_action(3, Some(position(36)), Some(StateAction::Verify))
            .is_err()
    );
    journal.rewind(33).unwrap();
    assert_eq!(
        journal
            .begin_action(
                3,
                Some(LanePosition {
                    tokens: u64::MAX,
                    reserved_tokens: u64::MAX,
                    ..LanePosition::default()
                }),
                Some(StateAction::Verify)
            )
            .err(),
        Some(PagePoolError::ArithmeticOverflow)
    );
    let chain = verify(&mut journal, 3, 2);
    journal.cancel_unsubmitted(&chain).unwrap();
    assert_eq!(journal.position.tokens, 33);
}

#[test]
fn verify_admission_preserves_predecessor_on_refusal_and_cancellation() {
    for rows in [2, 3] {
        let (_pool, mut journal) = fixture(rows);
        let input = journal.checkpoint_latest().unwrap();
        for end in [0, 33, 34, 4098] {
            assert!(
                journal
                    .begin_action(2, Some(position(end)), Some(StateAction::Verify))
                    .is_err()
            );
        }
        let caller = journal.retained.clone();
        assert!(
            journal
                .begin_action(
                    2,
                    Some(position(33 + u64::from(rows))),
                    Some(StateAction::Verify)
                )
                .is_err()
        );
        assert!(journal.retained.is_some());
        drop(caller);
        let chain = verify(&mut journal, 2, rows);
        assert_eq!(
            chain.output_versions().count(),
            usize::try_from(rows).unwrap()
        );
        assert!(journal.checkpoint_latest().is_err());
        assert!(journal.begin(3, None).is_err());
        journal.cancel_unsubmitted(&chain).unwrap();
        assert_eq!(journal.position, input.position);
        assert_eq!(journal.retained.as_ref().unwrap().position.tokens, 0);
    }
}

#[test]
fn completed_verify_defers_publication_until_its_acceptance_is_settled() {
    for rows in [2, 3] {
        let (pool, mut journal) = fixture(rows);
        let input = journal.checkpoint_latest().unwrap();
        let chain = verify(&mut journal, 2, rows);
        journal.mark_submitted(&chain).unwrap();
        assert!(journal.finish(&chain, true).is_err());
        pool.observe_completion(2).unwrap();
        journal.finish(&chain, true).unwrap();
        assert_eq!(journal.position, input.position);
        assert_eq!(journal.phase, SelectionPhase::AwaitingAcceptance);
        assert!(journal.checkpoint_latest().is_err());
        assert!(journal.ensure_quiescent().is_err());
        assert!(journal.canonical_boundary().is_err());
        assert!(journal.reserve_selection(32, &Arc::new(())).is_err());
        assert!(journal.begin(3, None).is_err());
        journal.cancel_unsubmitted(&chain).unwrap();
        assert_eq!(journal.position, input.position);
        assert_eq!(journal.phase, input.phase);
    }
}

#[test]
fn verify_execution_failure_preserves_source_after_physical_drain() {
    for rows in [2, 3] {
        for abandoned in [false, true] {
            let (pool, mut journal) = fixture(rows);
            let input = journal.checkpoint_latest().unwrap();
            let chain = verify(&mut journal, 2, rows);
            journal.mark_submitted(&chain).unwrap();
            if abandoned {
                journal.abandon(&chain).unwrap();
                assert!(journal.ensure_quiescent().is_err());
                pool.observe_completion(2).unwrap();
            } else {
                pool.observe_completion(2).unwrap();
                journal.finish(&chain, false).unwrap();
            }
            assert_eq!(journal.position, input.position);
            assert!(journal.retained.is_none());
            assert_eq!(
                journal.checkpoint_latest().unwrap().version.version(),
                input.version.version()
            );
            let plain = journal.begin(3, None).unwrap();
            journal.cancel_unsubmitted(&plain).unwrap();
        }
    }
}
