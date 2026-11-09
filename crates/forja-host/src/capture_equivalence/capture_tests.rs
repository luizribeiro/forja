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

pub(super) fn open_test_capture_weights(
    host: &mut Host<CpuBackend>,
) -> (std::path::PathBuf, [Resource<TensorEntry>; 3]) {
    let mut header = br#"{"left":{"dtype":"F32","shape":[7],"data_offsets":[0,28]},"right":{"dtype":"F32","shape":[7],"data_offsets":[28,56]},"unseen":{"dtype":"F32","shape":[7],"data_offsets":[56,84]}}"#.to_vec();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(1_f32.to_le_bytes().repeat(21));
    let path = std::env::temp_dir().join(format!(
        "forja-capture-weights-{}-{}",
        std::process::id(),
        NEXT_WEIGHT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, bytes).unwrap();
    host.grants = Grants::new().with_weights("capture", &path);
    let source = host.open_weights("capture").unwrap();
    (
        path,
        ["left", "right", "unseen"].map(|name| host.weight_tensor(&source, name).unwrap()),
    )
}

pub(super) fn record_test_weight_operations(
    host: &mut Host<CpuBackend>,
    commands: &Resource<super::CommandListEntry>,
    weights: &[u32],
    swap: bool,
) {
    for (index, weight) in weights.iter().enumerate() {
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let (op, inputs) = if (index == 0) ^ swap {
            (compute::Op::Copy, vec![Resource::new_borrow(*weight)])
        } else {
            (
                compute::Op::RmsNorm(1e-6),
                vec![Resource::new_borrow(*weight), Resource::new_borrow(*weight)],
            )
        };
        host.dispatch(commands, op, &inputs, &output).unwrap();
        host.drop_tensor(output).unwrap();
    }
}

#[test]
fn decode_reference_rejects_shifted_or_foreign_rule_parameters() {
    use forja_core::{BackendDispatchData, GraphLimits, GraphTemplate, ParamSpace, RuleKey};
    for (key, accepted) in [
        (
            RuleKey {
                parameter: 0,
                row_offset: 0,
            },
            true,
        ),
        (
            RuleKey {
                parameter: 0,
                row_offset: 1,
            },
            false,
        ),
        (
            RuleKey {
                parameter: 1,
                row_offset: 0,
            },
            false,
        ),
    ] {
        let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
        let commands = host.command_list().unwrap();
        host.declare_slot_capture(&commands, &slot, test_capture_shape(false))
            .unwrap();
        record_test_decode_roles(&mut host, &commands, &slot);
        let space = ParamSpace::new(vec![0..=33, 0..=33]).unwrap();
        host.table.get_mut(&commands).unwrap().space = Some(space.clone());
        let input = host.backend.alloc(DType::F32, &[7]).unwrap();
        let output = host.backend.alloc(DType::F32, &[7]).unwrap();
        let mut graph = GraphTemplate::new(space, GraphLimits::default());
        for recorded in &host.table.get(&commands).unwrap().recorded {
            super::record_graph_dispatch(&mut graph, recorded).unwrap();
        }
        graph
            .dispatch_variant_rule_many(
                Op::Copy,
                &[&input.into()],
                &[&output.into()],
                key,
                vec![(34, BackendDispatchData::new(0_u32))],
            )
            .unwrap();
        let result = host.capture_arithmetic(&graph, host.table.get(&commands).unwrap());
        assert_eq!(result.is_ok(), accepted);
    }
}

#[test]
fn decode_reference_requires_both_real_output_roles() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    for omit in [0, 1] {
        let commands = host.command_list().unwrap();
        host.declare_slot_capture(&commands, &slot, test_capture_shape(false))
            .unwrap();
        record_test_output_roles(&mut host, &commands, &slot, Some(omit));
        assert!(host.create_graph(commands).is_err());
        assert!(host.table.get(&slot).unwrap().decode_arithmetic.is_none());
    }
}

#[test]
fn decode_reference_requires_real_output_roles() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    assert!(
        capture_arithmetic_body(
            &mut host,
            &slot,
            test_capture_shape(false),
            Some(compute::Op::Copy),
            false
        )
        .is_err()
    );
}

#[test]
fn later_decode_capture_cannot_add_unmatched_arithmetic() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let shape = test_capture_shape(false);
    let reference = capture_arithmetic_copy(&mut host, &slot, shape).unwrap();
    let commands = host.command_list().unwrap();
    host.declare_slot_capture(&commands, &slot, shape).unwrap();
    record_test_decode_roles(&mut host, &commands, &slot);
    let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
    let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
    let kernel = host
        .create_kernel(doubling_program(2.0), unary_f32_signature(1))
        .unwrap();
    host.dispatch_kernel(&commands, &kernel, &[input], &[output])
        .unwrap();
    assert!(host.create_graph(commands).is_err());
    host.drop_graph(reference).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn metal_verify_capture_compares_every_rule_arm_at_its_key() {
    let (mut host, _pool, slot) =
        verify_input_fixture(forja_metal::MetalBackend::new().unwrap(), 3, 1);
    let (params, inputs, output) = metal_symbolic_matmul(&mut host);
    let decode = test_capture_shape(false);
    let steel = "matmul.steel-64x64x16-2x2";
    let portable = "matmul.portable";
    for (shape, picks, accepted) in [
        (decode, [steel, portable], true),
        (
            BatchShape {
                phase: BatchPhase::Verify,
                tokens: 3,
                ..decode
            },
            [steel, portable],
            true,
        ),
        (
            BatchShape {
                phase: BatchPhase::Verify,
                tokens: 3,
                ..decode
            },
            [portable, steel],
            false,
        ),
    ] {
        let commands = host.command_list().unwrap();
        host.declare_slot_capture(&commands, &slot, shape).unwrap();
        let arms = [2, 3]
            .into_iter()
            .zip(picks)
            .map(
                |(key, name)| super::metal_bindings::l9o::gpu::metal_variants::Arm {
                    lo: key,
                    hi: key,
                    name: name.to_owned(),
                },
            )
            .collect::<Vec<_>>();
        host.record_metal_variant_rule(
            &commands,
            &params,
            0,
            &arms,
            compute::Op::Matmul,
            &borrowed_tensors(&inputs),
            &[Resource::new_borrow(output.rep())],
        )
        .unwrap();
        if shape.phase == BatchPhase::Decode {
            record_test_decode_roles(&mut host, &commands, &slot);
        }
        let result = host.create_graph(commands);
        assert_eq!(result.is_ok(), accepted);
        if shape.phase != BatchPhase::Decode
            && let Ok(graph) = result
        {
            host.drop_graph(graph).unwrap();
        }
    }
}

fn capture_weight_operations(
    host: &mut Host<CpuBackend>,
    slot: &Resource<BatchSlotEntry>,
    shape: BatchShape,
    weights: &[u32],
    swap: bool,
) -> Result<Resource<super::GraphEntry>, compute::Error> {
    let commands = host.command_list().unwrap();
    host.declare_slot_capture(&commands, slot, shape).unwrap();
    record_test_weight_operations(host, &commands, weights, swap);
    if shape.phase == BatchPhase::Decode {
        record_test_decode_roles(host, &commands, slot);
    }
    host.create_graph(commands)
}

#[test]
fn verify_capture_matches_arithmetic_to_weight_regions_without_exemptions() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let (path, weights) = open_test_capture_weights(&mut host);
    assert_eq!(
        host.entry(&weights[0]).unwrap().tensor.buffer(),
        host.entry(&weights[1]).unwrap().tensor.buffer()
    );
    let alias = host
        .view(&weights[0], compute::ViewOp::Permute(vec![0]))
        .unwrap();
    let decode = test_capture_shape(false);
    let graph = capture_weight_operations(
        &mut host,
        &slot,
        decode,
        &[weights[0].rep(), weights[1].rep()],
        false,
    )
    .unwrap();
    let reference = graph;
    let verify = BatchShape {
        phase: BatchPhase::Verify,
        tokens: 3,
        ..decode
    };
    for (handles, swap, accepted) in [
        (vec![alias.rep(), weights[1].rep()], false, true),
        (vec![weights[0].rep(), weights[1].rep()], true, false),
        (vec![weights[2].rep()], false, false),
    ] {
        let result = capture_weight_operations(&mut host, &slot, verify, &handles, swap);
        assert_eq!(result.is_ok(), accepted);
        if let Ok(graph) = result {
            host.drop_graph(graph).unwrap();
        }
    }
    host.drop_graph(reference).unwrap();
    fs::remove_file(path).unwrap();
}
