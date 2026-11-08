use super::*;
use crate::{LanePosition, StateAction};

fn position(tokens: u64) -> LanePosition {
    LanePosition {
        tokens,
        reserved_tokens: 4097,
        ..LanePosition::default()
    }
}

fn accepted_carry<B: Backend + Send + Sync + 'static>(backend: B) {
    let f = Fixture::with_spares(backend, 1);
    for rows in [2_u32, 3] {
        for accepted in 0..rows {
            let lane = f.storage.reserve().unwrap();
            assert!(lane.try_reserve_spares().unwrap());
            let tx = lane
                .begin_action_at(1, position(u64::from(rows)), StateAction::Verify)
                .unwrap();
            let mut recording = f.storage.record(&lane, tx.clone()).unwrap();
            for row in 0..usize::try_from(rows).unwrap() {
                f.update(&mut recording, row);
            }
            recording.submit().unwrap().finish().unwrap();
            lane.wait_drained(f.host.limits.submission_timeout).unwrap();
            assert!(lane.checkpoint_latest().is_err());
            lane.publish_accepted(&tx, accepted).unwrap();
            let tokens = u64::from(accepted) + 2;
            let next = lane.begin_at(2, position(tokens)).unwrap();
            let mut recording = f.storage.record(&lane, next).unwrap();
            f.update(&mut recording, 0);
            recording.submit().unwrap().finish().unwrap();
            let actual = f.read_outputs()[0].clone();
            let reference = f.storage.reserve().unwrap();
            for step in 1..=tokens {
                let tx = reference.begin_at(step, position(step)).unwrap();
                let mut recording = f.storage.record(&reference, tx).unwrap();
                f.update(&mut recording, 0);
                recording.submit().unwrap().finish().unwrap();
            }
            assert_eq!(actual, f.read_outputs()[0]);
        }
    }
}

#[test]
fn accepted_verify_carry_matches_separate_plain_updates() {
    accepted_carry(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_accepted_verify_carry_matches_separate_plain_updates() {
    accepted_carry(forja_metal::MetalBackend::new().unwrap());
}

fn failed_verify<B: Backend + Send + Sync + 'static>(backend: B) {
    let f = Fixture::with_spares(backend, 1);
    for rows in [2_u32, 3] {
        for failed_row in 0..rows {
            let lane = f.storage.reserve().unwrap();
            assert!(lane.try_reserve_spares().unwrap());
            let input = lane.checkpoint_latest().unwrap();
            let tx = lane
                .begin_action_at(1, position(u64::from(rows)), StateAction::Verify)
                .unwrap();
            let mut recording = f.storage.record(&lane, tx.clone()).unwrap();
            for row in 0..usize::try_from(rows).unwrap() {
                f.update(&mut recording, row);
            }
            f.corrupt_word(
                &recording
                    .table_view(usize::try_from(failed_row).unwrap())
                    .unwrap(),
                1,
            );
            assert!(recording.submit().unwrap().finish().is_err());
            assert!(lane.publish_accepted(&tx, 0).is_err());
            assert_eq!(lane.position(), input.position());
            assert_eq!(
                lane.checkpoint_latest().unwrap().version.version(),
                input.version.version()
            );
        }
    }
}

#[test]
fn failed_verify_link_never_publishes_an_accepted_row() {
    failed_verify(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_failed_verify_link_never_publishes_an_accepted_row() {
    failed_verify(forja_metal::MetalBackend::new().unwrap());
}
#[test]
fn verify_feedback_requires_staged_tokens_until_plain_selection() {
    use crate::state::feedback::FeedbackBinding;
    let f = Fixture::with_spares(forja_cpu::CpuBackend::new(), 1);
    for rows in [2_u32, 3] {
        let lane = f.storage.reserve().unwrap();
        lane.lock()
            .attach_feedback(FeedbackBinding::with_spares(&f.host, 2, 1).unwrap())
            .unwrap();
        assert!(lane.try_reserve_spares().unwrap());
        for (serial, action, end) in [
            (1, StateAction::Combined, 1),
            (2, StateAction::Verify, 1 + rows),
            (3, StateAction::Combined, 3),
            (4, StateAction::Combined, 4),
        ] {
            let tx = lane
                .begin_action_at(serial, position(u64::from(end)), action)
                .unwrap();
            let mut recording = f.storage.record(&lane, tx.clone()).unwrap();
            let feedback = recording.feedback.as_ref().unwrap();
            assert_eq!(feedback.source.is_some(), serial == 4);
            assert_eq!(feedback.select_input(false).is_ok(), serial == 4);
            feedback.select_input(true).unwrap();
            assert_eq!(feedback.combined, serial != 2);
            for row in 0..if serial == 2 {
                usize::try_from(rows).unwrap()
            } else {
                1
            } {
                f.update(&mut recording, row);
            }
            recording.submit().unwrap().finish().unwrap();
            if serial == 2 {
                lane.publish_accepted(&tx, 0).unwrap();
                assert_eq!(lane.position().tokens, 2);
            }
        }
    }
}
