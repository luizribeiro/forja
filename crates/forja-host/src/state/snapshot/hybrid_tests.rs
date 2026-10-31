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
    let mut source = pool.reserve(257).unwrap();
    source.commit_write(256).unwrap();
    let source = crate::PagedPrefix::new(source);
    let backing_bytes = pool.memory_usage().unwrap().0.total().unwrap();
    host.live_bytes.fetch_add(backing_bytes, Ordering::AcqRel);
    let resource = host
        .table
        .push(crate::PagePoolEntry {
            pool: pool.clone(),
            active_lanes: 1,
            max_context: 257,
            plane_count: 1,
            slot_banks: None,
            slot: None,
            charged_bytes: std::sync::atomic::AtomicU64::new(backing_bytes),
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

impl HybridFixture {
    fn entry(&self) -> Arc<crate::HybridPrefix<CpuBackend>> {
        self.host
            .hybrid_prefix(
                &self.resource,
                &self.source,
                self.image.clone(),
                self.image.tokens(),
            )
            .unwrap()
    }
    fn advance(&self, end: u64) -> Arc<crate::HybridPrefix<CpuBackend>> {
        let prompt = if end < 129 {
            Arc::clone(&self.image.prompt)
        } else {
            self.host
                .canonical_prompt(self.image.namespace(), (0..257).collect())
                .unwrap()
        };
        let transaction = self
            .parent
            .begin_prefill(end, prompt, end, 257, crate::StateAction::Step)
            .unwrap();
        self.parent.lock().mark_submitted(&transaction).unwrap();
        self.parent.storage.pool.observe_completion(end).unwrap();
        self.parent.lock().finish(&transaction, true).unwrap();
        let image = self
            .parent
            .capture_snapshot(SnapshotAllowance::new(65_536, None).unwrap())
            .unwrap();
        self.host
            .hybrid_prefix(&self.resource, &self.source, image.clone(), image.tokens())
            .unwrap()
    }
    fn cache(
        &self,
        allowance: u64,
        bytes: u64,
        entries: usize,
    ) -> crate::HybridPrefixStore<CpuBackend> {
        self.host
            .create_hybrid_prefix_store(
                &self.resource,
                self.image.namespace(),
                SnapshotAllowance::new(allowance, None).unwrap(),
                bytes,
                entries,
            )
            .unwrap()
    }
}

#[test]
fn hybrid_lru_counts_shared_allocations_and_promotes_exact_hits() {
    let fixture = hybrid_fixture(262_144);
    let cache = fixture.cache(65_536, 65_536, 2);
    let first = fixture.entry();
    let first_bytes = first.retained_bytes();
    let first_weak = Arc::downgrade(&first);
    let empty = cache.usage().unwrap();
    let running = std::sync::atomic::AtomicBool::new(false);
    cache.publish(first, &running).unwrap();
    let second = fixture.advance(128);
    let second_bytes = second.retained_bytes();
    let second_weak = Arc::downgrade(&second);
    cache.publish(second, &running).unwrap();
    let usage = cache.usage().unwrap();
    assert_eq!(usage.entries, 2);
    assert_eq!(
        usage.bytes,
        empty.bytes + first_bytes + second_bytes
            - fixture.image.prompt.charged_bytes()
            - fixture.pool.layout().total_page_bytes()
    );
    assert!(
        cache
            .lookup(fixture.image.namespace(), &(0..64).collect::<Vec<_>>())
            .unwrap()
            .is_none()
    );
    let hit = cache
        .lookup(fixture.image.namespace(), &(0..65).collect::<Vec<_>>())
        .unwrap()
        .unwrap();
    assert_eq!(hit.tokens().len(), 64);
    drop(hit);
    cache.publish(fixture.advance(192), &running).unwrap();
    assert!(first_weak.upgrade().is_some());
    assert!(second_weak.upgrade().is_none());
    let hit = cache
        .lookup(fixture.image.namespace(), &(0..257).collect::<Vec<_>>())
        .unwrap()
        .unwrap();
    assert_eq!(hit.tokens().len(), 192);
    assert_eq!(cache.usage().unwrap().entries, 2);
}
#[test]
fn byte_pressure_evicts_only_unpinned_entries_and_refusals_are_atomic() {
    let fixture = hybrid_fixture(262_144);
    let first = fixture.entry();
    let next = fixture.advance(128);
    let cache = fixture.cache(65_536, next.retained_bytes() + 1280, 8);
    let running = std::sync::atomic::AtomicBool::new(false);
    cache.publish(first.clone(), &running).unwrap();
    let before = cache.usage().unwrap();
    let pages = fixture.pool.memory_usage().unwrap().0;
    let live = fixture.host.live_bytes.load(Ordering::Acquire);
    assert!(cache.publish(next.clone(), &running).is_err());
    assert_eq!(cache.usage().unwrap(), before);
    assert_eq!(fixture.pool.memory_usage().unwrap().0, pages);
    assert_eq!(fixture.host.live_bytes.load(Ordering::Acquire), live);
    drop(first);
    cache.publish(next, &running).unwrap();
    assert!(
        cache
            .lookup(fixture.image.namespace(), &(0..65).collect::<Vec<_>>())
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.usage().unwrap().entries, 1);
}

#[test]
fn an_oversized_snapshot_or_accounting_quota_never_evicts_existing_entries() {
    let fixture = hybrid_fixture(262_144);
    let first = fixture.entry();
    let next = fixture.advance(128);
    let allowance = first.reference_bytes()
        + fixture.image.charged_bytes()
        + fixture.image.prompt.charged_bytes();
    let cache = fixture.cache(allowance, 65_536, 8);
    let running = std::sync::atomic::AtomicBool::new(false);
    let old = Arc::downgrade(&first);
    cache.publish(first, &running).unwrap();
    let before = cache.usage().unwrap();
    assert!(cache.publish(next.clone(), &running).is_err());
    assert_eq!(cache.usage().unwrap(), before);
    assert!(old.upgrade().is_some());
    let host = &fixture.host;
    let remaining = host.limits.live_bytes - host.live_bytes.load(Ordering::Acquire);
    let blocker = StateCharge::reserve(
        host.backend.as_ref(),
        Arc::clone(&host.live_bytes),
        host.limits.live_bytes,
        remaining - 1,
    )
    .unwrap();
    let exhausted = host.live_bytes.load(Ordering::Acquire);
    assert!(cache.publish(next.clone(), &running).is_err());
    assert_eq!(host.live_bytes.load(Ordering::Acquire), exhausted);
    assert!(old.upgrade().is_some());
    drop(blocker);
    assert_eq!(cache.usage().unwrap(), before);
}
