fn lifecycle_fixture() -> (
    Host<CpuBackend>,
    Resource<PagePoolEntry>,
    Resource<BatchSlotEntry>,
    Arc<super::state::StateStorage<CpuBackend>>,
) {
    let mut host = Host::new(CpuBackend::new(), GENEROUS);
    let pool = insert_test_pool(&mut host, 2, 8);
    let slot = host.open_paged_slot(&pool, 2).unwrap();
    let storage = host
        .create_state_storage(
            forja_core::StateLayout::new(vec![
                forja_core::StatePlane::new(DType::F32, vec![3, 7]).unwrap(),
            ])
            .unwrap(),
            1,
        )
        .unwrap();
    host.state_slots.insert(
        slot.rep(),
        super::state::StateSlots::new(Arc::clone(&storage), 1).unwrap(),
    );
    host.bind_paged_lane(pool.rep(), 0, None, 8).unwrap();
    (host, pool, slot, storage)
}

#[test]
fn state_admission_failure_restores_paged_append_banks() {
    let (mut host, pool, slot, storage) = lifecycle_fixture();
    host.state_slots
        .get_mut(&slot.rep())
        .unwrap()
        .release(0)
        .unwrap();
    let occupied = storage.reserve().unwrap();
    let binding = Arc::clone(&host.table.get(&slot).unwrap().binding);
    let previous = [
        binding.banks.slot(0, 0).unwrap(),
        binding.banks.slot(1, 0).unwrap(),
    ];
    assert!(host.begin_paged_lane_append(pool.rep(), 0, 0, 1).is_err());
    let lane = binding.lane(0).unwrap();
    assert_eq!(
        lane.sequence
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .logical_len(),
        0
    );
    assert_eq!(
        host.state_slots[&slot.rep()].position(0).unwrap(),
        super::state::LanePosition::default()
    );
    for (bank, previous) in previous.iter().enumerate() {
        assert_eq!(&binding.banks.slot(bank, 0).unwrap(), previous);
        assert!(lane.table_dirty[bank].load(Ordering::Acquire));
    }
    drop(occupied);
    host.begin_paged_lane_append(pool.rep(), 0, 0, 1).unwrap();
}

#[test]
fn force_detach_cancels_unsubmitted_state_bank() {
    let (mut host, pool, slot, storage) = lifecycle_fixture();
    let (checkpoint, _) = host.begin_paged_lane_append(pool.rep(), 0, 0, 1).unwrap();
    assert!(
        host.state_slots
            .get_mut(&slot.rep())
            .unwrap()
            .release(0)
            .is_err()
    );
    host.rollback_paged_lane_append(pool.rep(), 0, checkpoint.unwrap())
        .unwrap();
    assert!(
        host.state_slots
            .get_mut(&slot.rep())
            .unwrap()
            .recording(0, 0)
            .is_err()
    );
    host.bind_paged_lane(pool.rep(), 0, None, 0).unwrap();
    let state = &host.state_slots[&slot.rep()];
    assert!(state.sequence_handle(0).unwrap().is_none());
    assert_eq!(
        state.position(0).unwrap(),
        super::state::LanePosition::default()
    );
    assert!(storage.reserve().is_ok());
}

#[test]
fn bank_swap_replaces_only_selected_padding_metadata() {
    let (mut host, _pool, slot, _storage) = lifecycle_fixture();
    let ids = |host: &mut Host<CpuBackend>, bank| {
        host.state_slots
            .get_mut(&slot.rep())
            .unwrap()
            .commands(bank, 0, false)
            .unwrap()
            .protected_buffers()
            .collect::<Vec<_>>()
    };
    let before = [ids(&mut host, 0), ids(&mut host, 1)];
    host.table.get_mut(&slot).unwrap().next_bank = 1;
    host.select_paged_batch_bank(slot.rep()).unwrap();
    assert_eq!(ids(&mut host, 0), before[0]);
    assert_ne!(ids(&mut host, 1), before[1]);
}
