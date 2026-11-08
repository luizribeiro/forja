use super::{test_support::Fixture, *};

#[test]
fn metadata_retains_every_adjacent_pair_and_refuses_incomplete_tables() {
    let f = Fixture::new(forja_cpu::CpuBackend::new());
    let mut sequence = f.reserve();
    let tx = sequence.begin_chain(1, 2).unwrap();
    let slot = sequence.bind(&tx).unwrap();
    let mut commands = f
        .storage
        .prepare_record_metadata(StateAccess::from_slot(&slot).unwrap())
        .unwrap();
    assert!(
        commands
            .initialize_metadata(vec![0; 4], slot.table().generations().to_vec(), true)
            .is_err()
    );
    commands.initialize_slot(&slot).unwrap();
    let [source, middle, last] = slot.table().entries() else {
        unreachable!()
    };
    let pairs = [source, middle, middle, last]
        .into_iter()
        .flat_map(|entry| [entry.page(), entry.generation()])
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(commands.metadata[0].layout().shape(), [4, 2]);
    assert_eq!(f.host.backend.read(&commands.metadata[0]).unwrap(), pairs);
    assert_eq!(
        f.host.backend.read(&commands.metadata[2]).unwrap(),
        1_u32.to_le_bytes()
    );
}

#[test]
fn longer_chain_metadata_requires_a_larger_live_byte_allowance() {
    let f = Fixture::new(forja_cpu::CpuBackend::new());
    let one = f.storage.command_charge(f.host.limits, 1).unwrap().bytes;
    let two = f.storage.command_charge(f.host.limits, 2).unwrap().bytes;
    assert!(two > one);
    let used = f.host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
    let limits = Limits {
        live_bytes: used + one,
        ..f.host.limits
    };
    assert!(f.storage.command_charge(limits, 2).is_err());
    assert!(f.storage.command_charge(limits, 1).is_ok());
    let mut sequence = f.reserve();
    let commands = f.commands(&mut sequence, 1, 2);
    assert_eq!(commands.metadata[0].layout().shape(), [4, 2]);
}
