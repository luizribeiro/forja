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
    let guard = StateCharge::reserve(Arc::clone(&host.live_bytes), 8192, 8192 - retained).unwrap();
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
