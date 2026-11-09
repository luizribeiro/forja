#[test]
fn failed_decode_summary_reservation_leaves_no_reference_or_graph() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let before = host.live_bytes.load(Ordering::Acquire);
    let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
    let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
    let commands = host.command_list().unwrap();
    host.declare_slot_capture(&commands, &slot, test_capture_shape(false))
        .unwrap();
    record_test_decode_roles(&mut host, &commands, &slot);
    host.dispatch(
        &commands,
        compute::Op::Copy,
        &[Resource::new_borrow(input.rep())],
        &output,
    )
    .unwrap();
    host.limits.live_bytes = host.live_bytes.load(Ordering::Acquire);
    assert!(matches!(
        host.create_graph(commands),
        Err(compute::Error::Quota(_))
    ));
    assert!(host.table.get(&slot).unwrap().decode_arithmetic.is_none());
    assert_eq!(host.engine_metrics().live_graphs, 0);
    host.drop_tensor(input).unwrap();
    host.drop_tensor(output).unwrap();
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
}

#[test]
fn decode_arithmetic_reference_authenticates_the_capture_slot_identity() {
    let mut source = state_slot_fixture(CpuBackend::new());
    let commands = source.host.command_list().unwrap();
    source
        .host
        .declare_slot_capture(&commands, &source.slot, test_capture_shape(false))
        .unwrap();
    let capture = source
        .host
        .table
        .get(&commands)
        .unwrap()
        .slot_capture
        .clone();
    let mut other = state_slot_fixture(CpuBackend::new());
    assert_eq!(source.slot.rep(), other.slot.rep());
    let commands = other.host.command_list().unwrap();
    other.host.table.get_mut(&commands).unwrap().slot_capture = capture;
    let error = other.host.create_graph(commands).unwrap_err();
    assert!(
        matches!(error, compute::Error::InvalidHandle(message) if message.contains("capture slot identity"))
    );
    assert!(
        other
            .host
            .table
            .get(&other.slot)
            .unwrap()
            .decode_arithmetic
            .is_none()
    );
    assert_eq!(other.host.engine_metrics().live_graphs, 0);
}
