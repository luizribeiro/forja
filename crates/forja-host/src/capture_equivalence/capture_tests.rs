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

#[tokio::test]
async fn captured_verify_replays_after_its_decode_reference_is_dropped() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let decode = test_capture_shape(false);
    let reference = capture_arithmetic_copy(&mut host, &slot, decode).unwrap();
    let verify = capture_arithmetic_copy(
        &mut host,
        &slot,
        BatchShape {
            phase: BatchPhase::Verify,
            tokens: 3,
            ..decode
        },
    )
    .unwrap();
    host.drop_graph(reference).unwrap();
    let flight = ReplayFlight::acquire(host.replay_state(&verify).unwrap()).unwrap();
    host.prepare_state_replay_with_flight(&verify, vec![], flight)
        .unwrap()
        .run()
        .await
        .unwrap();
    assert!(host.table.get(&slot).unwrap().decode_arithmetic.is_none());
}

#[test]
fn only_a_live_combined_decode_capture_supplies_the_reference() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let decode = test_capture_shape(false);
    let verify = BatchShape {
        phase: BatchPhase::Verify,
        tokens: 3,
        ..decode
    };
    for shape in [
        test_capture_shape(true),
        BatchShape {
            bucket: 2,
            ..decode
        },
        BatchShape {
            selection: BatchSelection::Forward,
            ..decode
        },
        BatchShape {
            selection: BatchSelection::Masked,
            ..decode
        },
    ] {
        let graph = capture_arithmetic_copy(&mut host, &slot, shape).unwrap();
        host.drop_graph(graph).unwrap();
        assert!(capture_arithmetic_copy(&mut host, &slot, verify).is_err());
    }
    let graph = capture_arithmetic_copy(&mut host, &slot, decode).unwrap();
    assert!(capture_arithmetic_copy(&mut host, &slot, verify).is_ok());
    host.drop_graph(graph).unwrap();
    assert!(capture_arithmetic_copy(&mut host, &slot, verify).is_err());
}
