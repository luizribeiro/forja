use super::*;
use crate::{Host, Limits, StateAction};
use forja_core::{DType, HybridPrefixNamespace, PagedLayout, PagedPlane, StateLayout, StatePlane};
use std::{num::NonZeroU32, sync::atomic::Ordering};

pub(super) fn prepared(
    limit: u64,
) -> (
    Host<forja_cpu::CpuBackend>,
    StateSequenceHandle<forja_cpu::CpuBackend>,
    Vec<Vec<u8>>,
) {
    let host = Host::new(
        forja_cpu::CpuBackend::new(),
        Limits::new(limit, 8, 4096, 32, 4096),
    );
    let layout = StateLayout::new(vec![
        StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap(),
        StatePlane::new(DType::BF16, vec![3, 7]).unwrap(),
    ])
    .unwrap();
    let pages = PagedLayout::new(
        64,
        vec![PagedPlane {
            dtype: DType::F16,
            elements_per_token: 33,
            alignment: 2,
            label: "key".into(),
        }],
    )
    .unwrap();
    let namespace = HybridPrefixNamespace::new(
        [3; 32],
        &layout,
        &pages,
        NonZeroU32::new(64).unwrap(),
        &["fixed-tree"],
    )
    .unwrap();
    let prompt = host
        .canonical_prompt(namespace, (0..129).collect())
        .unwrap();
    let storage = host.create_state_storage(layout, 2).unwrap();
    let sequence = storage.reserve().unwrap();
    let transaction = sequence
        .begin_prefill(1, prompt, 64, 257, StateAction::Step)
        .unwrap();
    sequence.lock().mark_submitted(&transaction).unwrap();
    storage.pool.observe_completion(1).unwrap();
    sequence.lock().finish(&transaction, true).unwrap();
    let record = {
        let journal = sequence.lock();
        journal
            .allocation_table()
            .unwrap()
            .resolve(journal.checkpoint().unwrap().version())
            .unwrap()
    };
    let values = vec![
        (0..231_u32)
            .flat_map(|i| (0x7f80_0001_u32.wrapping_add(i)).to_le_bytes())
            .collect::<Vec<_>>(),
        (0..21_u16)
            .flat_map(|i| (0x7f81_u16.wrapping_add(i)).to_le_bytes())
            .collect(),
    ];
    for (plane, data) in values.iter().enumerate() {
        host.backend
            .write(&storage.version_view(plane, record).unwrap(), data)
            .unwrap();
    }
    (host, sequence, values)
}

#[test]
fn immutable_images_preserve_all_float_payload_bits_and_charges() {
    let (host, sequence, expected) = prepared(131_072);
    let before = host.live_bytes.load(Ordering::Acquire);
    assert!(
        sequence
            .capture_snapshot(SnapshotAllowance::new(1, None).unwrap())
            .is_err()
    );
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    let image = sequence
        .capture_snapshot(SnapshotAllowance::new(65_536, None).unwrap())
        .unwrap();
    assert_eq!(image.tokens(), &(0..64).collect::<Vec<_>>());
    assert_eq!(image.position().tokens, 64);
    assert!(image.namespace().matches_state(&image.layout));
    for (tensor, bytes) in image.planes.iter().zip(&expected) {
        assert_eq!(host.backend.read(tensor).unwrap(), *bytes);
        assert!(host.backend.write(tensor, bytes).is_err());
    }
    assert_eq!(
        host.live_bytes.load(Ordering::Acquire),
        before + image.charged_bytes()
    );
    let bytes = image.charged_bytes();
    drop(image);
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    let maximum = 966;
    let (limited, sequence, _) = prepared(before + bytes + maximum - 1);
    let before = limited.live_bytes.load(Ordering::Acquire);
    assert!(
        sequence
            .capture_snapshot(SnapshotAllowance::new(65_536, None).unwrap())
            .is_err()
    );
    assert_eq!(limited.live_bytes.load(Ordering::Acquire), before);
}

#[test]
fn attachments_share_immutable_planes_until_the_first_update() {
    let (host, parent, expected) = prepared(262_144);
    let allowance = SnapshotAllowance::new(65_536, None).unwrap();
    let image = parent.capture_snapshot(allowance).unwrap();
    let storage = Arc::clone(&parent.storage);
    drop(parent);
    let children = [storage.reserve().unwrap(), storage.reserve().unwrap()];
    for child in &children {
        child
            .attach_snapshot(Arc::clone(&image), image.namespace(), 257)
            .unwrap();
        assert!(child.snapshot.lock().unwrap().is_some());
        let copy = child.capture_snapshot(allowance).unwrap();
        for (plane, bytes) in copy.planes.iter().zip(&expected) {
            assert_eq!(host.backend.read(plane).unwrap(), *bytes);
        }
    }
    let transaction = children[0].begin(2).unwrap();
    let recording = storage.record(&children[0], transaction).unwrap();
    assert!(children[0].snapshot.lock().unwrap().is_none());
    assert!(children[1].snapshot.lock().unwrap().is_some());
    let journal = children[0].lock();
    let record = journal
        .allocation_table()
        .unwrap()
        .resolve(journal.checkpoint().unwrap().version())
        .unwrap();
    drop(journal);
    for (plane, bytes) in expected.iter().enumerate() {
        let private = storage.version_view(plane, record).unwrap();
        assert_eq!(host.backend.read(&private).unwrap(), *bytes);
        host.backend.write(&private, &vec![0; bytes.len()]).unwrap();
        assert_eq!(host.backend.read(&image.planes[plane]).unwrap(), *bytes);
    }
    drop(recording);
    assert_eq!(children[0].position().tokens, 64);
    assert_eq!(children[1].position().tokens, 64);
    assert!(
        children[0]
            .attach_snapshot(image.clone(), image.namespace(), 257)
            .is_err()
    );
}

#[test]
fn attachment_refuses_preexisting_checkpoints_and_previously_used_sequences() {
    let (host, parent, _) = prepared(262_144);
    let image = parent
        .capture_snapshot(SnapshotAllowance::new(65_536, None).unwrap())
        .unwrap();
    let storage = Arc::clone(&parent.storage);
    drop(parent);
    let fresh = storage.reserve().unwrap();
    let used = storage.reserve().unwrap();
    let checkpoint = fresh.checkpoint_latest().unwrap();
    let before = host.live_bytes.load(Ordering::Acquire);
    assert!(
        fresh
            .attach_snapshot(image.clone(), image.namespace(), 257)
            .is_err()
    );
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
    assert_eq!(fresh.position().tokens, 0);
    assert!(fresh.snapshot.lock().unwrap().is_none());
    drop(checkpoint);
    fresh
        .attach_snapshot(image.clone(), image.namespace(), 257)
        .unwrap();
    assert_eq!(fresh.checkpoint_latest().unwrap().position().tokens, 64);
    let transaction = used.begin(1).unwrap();
    used.cancel(&transaction).unwrap();
    assert!(used.has_pristine_state().unwrap());
    assert!(
        used.attach_snapshot(image.clone(), image.namespace(), 257)
            .is_err()
    );
    assert_eq!(used.position().tokens, 0);
    assert!(used.snapshot.lock().unwrap().is_none());
}
