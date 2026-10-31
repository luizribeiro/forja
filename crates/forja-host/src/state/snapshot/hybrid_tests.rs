use super::{tests::prepared, *};
use crate::{Host, PagePoolEntry, PagedPrefix};
use forja_core::{DType, PagePool, PagedLayout, PagedPlane};
use forja_cpu::CpuBackend;
use std::sync::atomic::Ordering;
use wasmtime::component::Resource;

struct HybridFixture {
    host: Host<CpuBackend>,
    parent: StateSequenceHandle<CpuBackend>,
    image: Arc<StateSnapshot<CpuBackend>>,
    pool: PagePool,
    source: PagedPrefix,
    resource: Resource<PagePoolEntry>,
}

fn hybrid_fixture(limit: u64) -> HybridFixture {
    let (mut host, parent, _) = prepared(limit);
    let image = parent
        .capture_snapshot(SnapshotAllowance::new(65_536, None).unwrap())
        .unwrap();
    let layout = PagedLayout::new(
        64,
        vec![PagedPlane {
            dtype: DType::F16,
            elements_per_token: 33,
            alignment: 2,
            label: "key".into(),
        }],
    )
    .unwrap();
    let pool = forja_core::PagePool::new_fixed(layout.clone(), 16, 131_072).unwrap();
    let mut source = pool.reserve(129).unwrap();
    source.commit_write(128).unwrap();
    let source = crate::PagedPrefix::new(source);
    let resource = host
        .table
        .push(crate::PagePoolEntry {
            pool: pool.clone(),
            active_lanes: 1,
            max_context: 257,
            plane_count: 1,
            slot_banks: None,
            slot: None,
            charged_bytes: std::sync::atomic::AtomicU64::new(0),
            live_bytes: Arc::clone(&host.live_bytes),
        })
        .unwrap();
    HybridFixture {
        host,
        parent,
        image,
        pool,
        source,
        resource,
    }
}

#[test]
fn hybrid_entries_pair_exact_ledgers_and_reject_foreign_pages_atomically() {
    let HybridFixture {
        host,
        parent,
        image,
        pool,
        source,
        resource,
    } = hybrid_fixture(262_144);
    let layout = pool.layout().clone();
    let before = host.live_bytes.load(Ordering::Acquire);
    let pages = pool.memory_usage().unwrap().0;
    let mut changed = image.tokens().to_vec();
    changed[33] ^= 1;
    assert!(
        host.hybrid_prefix(&resource, &source, image.clone(), &changed)
            .is_err()
    );
    let foreign = crate::PagedPrefix::new(
        forja_core::PagePool::new_fixed(layout, 16, 131_072)
            .unwrap()
            .reserve(129)
            .unwrap(),
    );
    assert!(
        host.hybrid_prefix(&resource, &foreign, image.clone(), image.tokens())
            .is_err()
    );
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    assert_eq!(pool.memory_usage().unwrap().0, pages);
    let entry = host
        .hybrid_prefix(&resource, &source, image.clone(), image.tokens())
        .unwrap();
    assert_eq!(entry.page_count(), 1);
    assert_eq!(entry.tokens(), image.tokens());
    assert_eq!(
        entry.retained_bytes(),
        image.charged_bytes()
            + image.prompt.charged_bytes()
            + entry.reference_bytes()
            + pool.layout().total_page_bytes()
    );
    let child = parent.storage.reserve().unwrap();
    let before = pool.memory_usage().unwrap().0;
    assert!(entry.attach(&parent, 257).is_err());
    assert_eq!(pool.memory_usage().unwrap().0, before);
    assert!(child.has_pristine_state().unwrap());
    let available = host.limits.live_bytes - host.live_bytes.load(Ordering::Acquire);
    let blocker = StateCharge::reserve(
        host.backend.as_ref(),
        Arc::clone(&host.live_bytes),
        host.limits.live_bytes,
        available - 1,
    )
    .unwrap();
    let exhausted_bytes = host.live_bytes.load(Ordering::Acquire);
    assert!(entry.attach(&child, 257).is_err());
    assert_eq!(host.live_bytes.load(Ordering::Acquire), exhausted_bytes);
    assert_eq!(pool.memory_usage().unwrap().0, before);
    assert!(child.has_pristine_state().unwrap());
    drop(blocker);
    let attached = entry.attach(&child, 257).unwrap();
    assert_eq!(attached.attention().logical_len().unwrap(), 64);
    assert_eq!(attached.state().position().tokens, 64);
    assert_eq!(attached.tokens(), entry.tokens());
    assert_eq!(Arc::strong_count(&entry), 2);
    drop(attached);
    assert_eq!(Arc::strong_count(&entry), 1);
    assert_eq!(entry.attention_tokens().unwrap(), 64);
    assert_eq!(entry.attention_layout(), pool.layout());
}

#[test]
fn hybrid_cache_admission_is_cancelable_and_counts_shared_backing() {
    let fixture = hybrid_fixture(262_144);
    let HybridFixture {
        host,
        image,
        resource,
        source,
        ..
    } = fixture;
    let entry = host
        .hybrid_prefix(&resource, &source, image.clone(), image.tokens())
        .unwrap();
    let cache = host
        .create_hybrid_prefix_store(
            &resource,
            image.namespace(),
            SnapshotAllowance::new(65_536, None).unwrap(),
            65_536,
            2,
        )
        .unwrap();
    let empty = cache.usage().unwrap();
    let cancelled = std::sync::atomic::AtomicBool::new(true);
    assert!(!cache.publish(entry.clone(), &cancelled).unwrap());
    assert_eq!(cache.usage().unwrap(), empty);
    cancelled.store(false, Ordering::Release);
    assert!(cache.publish(entry.clone(), &cancelled).unwrap());
    let accepted = cache.usage().unwrap();
    assert_eq!(accepted.entries, 1);
    assert_eq!(accepted.bytes, empty.bytes + entry.retained_bytes());
    assert!(!cache.publish(entry.clone(), &cancelled).unwrap());
    assert_eq!(cache.usage().unwrap(), accepted);
}
