use super::*;
use crate::Limits;
use forja_core::{DType, StatePlane};

#[test]
fn initialization_scratch_uses_the_live_budget_and_failure_releases_versions() {
    let host = Host::new(
        forja_cpu::CpuBackend::new(),
        Limits::new(8192, 8, 4096, 128, 4096),
    );
    let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
    let storage = host.create_state_storage(layout, 1).unwrap();
    let retained = host.live_bytes.load(Ordering::Acquire);
    let guard = StateCharge::reserve(
        host.backend.as_ref(),
        Arc::clone(&host.live_bytes),
        8192,
        8192 - retained,
    )
    .unwrap();
    assert!(storage.reserve().is_err());
    assert_eq!(host.live_bytes.load(Ordering::Acquire), 8192);
    drop(guard);
    let sequence = storage.reserve().unwrap();
    assert_eq!(host.live_bytes.load(Ordering::Acquire), retained);
    drop(sequence);
    assert_eq!(host.live_bytes.load(Ordering::Acquire), retained);
}

#[test]
fn state_plane_rank_is_validated_before_backing_allocation() {
    let host = Host::new(
        forja_cpu::CpuBackend::new(),
        Limits::new(65536, 2, 4096, 128, 4096),
    );
    let layout =
        StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap()]).unwrap();
    assert!(host.create_state_storage(layout, 1).is_err());
    assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_state_reservation_reclaims_completed_charges_after_caller_timeout() {
    use forja_core::{BackendError, CommandList, CompletionResource, Op, Submission};
    use std::time::{Duration, Instant};
    let host = Host::new(
        forja_metal::MetalBackend::new().unwrap(),
        Limits::new(128 * 1024 * 1024, 8, 32 * 1024 * 1024, 128, 4096),
    );
    let input = host.backend.alloc(DType::F32, &[1024, 4097]).unwrap();
    let output = host.backend.alloc(DType::F32, &[1024, 4097]).unwrap();
    let charge = Arc::new(
        StateCharge::reserve(
            host.backend.as_ref(),
            Arc::clone(&host.live_bytes),
            host.limits.live_bytes,
            host.limits.live_bytes,
        )
        .unwrap(),
    );
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&input], &output).unwrap();
    let submission = host
        .backend
        .submit_retained(commands, vec![CompletionResource::new(charge)])
        .unwrap();
    assert_eq!(
        submission.wait_timeout(Duration::ZERO),
        Err(BackendError::Timeout)
    );
    drop(submission);
    let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7]).unwrap()]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(storage) = host.create_state_storage(layout.clone(), 1) {
            assert_eq!(
                host.live_bytes.load(Ordering::Acquire),
                storage.charged_bytes()
            );
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    host.backend.release(&input).unwrap();
    host.backend.release(&output).unwrap();
}
