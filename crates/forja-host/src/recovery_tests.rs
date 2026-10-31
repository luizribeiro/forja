#[test]
fn paged_bank_restore_keeps_admission_error_when_cleanup_also_fails() {
    let StateSlotFixture { host, slot, .. } = state_slot_fixture(CpuBackend::new());
    let binding = &host.table.get(&slot).unwrap().binding;
    let previous = [
        binding.banks.slot(0, 0).unwrap(),
        binding.banks.slot(1, 0).unwrap(),
    ];
    let failure = super::restore_paged_banks::<()>(
        Err(compute::Error::Quota("append admission failed".into())),
        binding,
        usize::MAX,
        previous.clone(),
    )
    .unwrap_err();
    let compute::Error::Quota(message) = failure else {
        panic!("lost primary error kind")
    };
    assert!(message.contains("append admission failed"));
    assert!(message.contains("cleanup failed") && message.contains("paged pool error: "));
    for (bank, before) in previous.iter().enumerate() {
        assert_eq!(&binding.banks.slot(bank, 0).unwrap(), before);
    }
}

#[tokio::test]
async fn selection_trap_retains_the_trap_when_rollback_also_fails() {
    let StateSlotFixture {
        mut host,
        pool,
        slot,
        operands,
        output,
    } = state_slot_fixture_action(CpuBackend::new(), Some(StateAction::Forward));
    let commands = host.command_list().unwrap();
    host.record_slot_state(
        &commands,
        &slot,
        0,
        0,
        paged_storage::StateOperation::CausalConvolution,
        &operands,
        &output,
        "",
    )
    .unwrap();
    host.prepare_submit(commands).unwrap().run().await.unwrap();
    let attempt = host
        .begin_slot_selection(&pool, std::iter::once((0, 0)))
        .unwrap();
    host.table.delete(slot).unwrap();
    let weights = test_weight_file();
    let mut runner = EngineRunner::new(
        test_guests::engine_smoke(),
        CpuBackend::new(),
        GENEROUS,
        &weights,
    )
    .await
    .unwrap();
    *runner.store.data_mut() = host;
    let failure = runner
        .finish_selection_attempt::<()>(attempt, Err(wasmtime::Trap::UnreachableCodeReached.into()))
        .unwrap_err();
    assert_eq!(
        failure.downcast_ref::<wasmtime::Trap>(),
        Some(&wasmtime::Trap::UnreachableCodeReached)
    );
    let context = format!("{failure:#}");
    assert!(context.contains("unreachable") && context.contains("cleanup failed"));
    assert!(context.contains("InvalidHandle"));
    drop(runner);
    fs::remove_file(weights).unwrap();
}
