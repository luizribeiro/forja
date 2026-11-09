fn record_test_decode_roles<B: Backend + Send + Sync + 'static>(
    host: &mut Host<B>,
    commands: &Resource<super::CommandListEntry>,
    slot: &Resource<BatchSlotEntry>,
) {
    record_test_output_roles(host, commands, slot, None);
}

fn record_test_output_roles<B: Backend + Send + Sync + 'static>(
    host: &mut Host<B>,
    commands: &Resource<super::CommandListEntry>,
    slot: &Resource<BatchSlotEntry>,
    omit: Option<usize>,
) {
    let bank = &host.table.get(slot).unwrap().inputs.as_ref().unwrap().banks[0];
    for (index, handle) in [bank.logits, bank.selected].into_iter().enumerate() {
        if omit == Some(index) {
            continue;
        }
        let entry = host.entry(&Resource::new_borrow(handle)).unwrap();
        let dtype = entry.tensor.layout().dtype();
        let slices = entry
            .tensor
            .layout()
            .shape()
            .iter()
            .enumerate()
            .map(|(axis, size)| compute::SliceSpec {
                start: 0,
                len: if axis == 0 { 1 } else { *size },
                step: 1,
            })
            .collect();
        let target = host
            .view(
                &Resource::new_borrow(handle),
                compute::ViewOp::Slice(slices),
            )
            .unwrap();
        let shape = host
            .entry(&target)
            .unwrap()
            .tensor
            .layout()
            .shape()
            .to_vec();
        let input = host
            .alloc(
                match dtype {
                    DType::F32 => compute::Dtype::F32,
                    DType::U32 => compute::Dtype::U32,
                    _ => unreachable!(),
                },
                &shape,
            )
            .unwrap();
        let size = host.entry(&input).unwrap().tensor.layout().byte_span().end;
        host.write(&input, &vec![0; usize::try_from(size).unwrap()])
            .unwrap();
        host.dispatch(
            commands,
            compute::Op::Copy,
            &[Resource::new_borrow(input.rep())],
            &target,
        )
        .unwrap();
        host.drop_tensor(input).unwrap();
        host.drop_tensor(target).unwrap();
    }
}

fn capture_arithmetic_body(
    host: &mut Host<CpuBackend>,
    slot: &Resource<BatchSlotEntry>,
    shape: BatchShape,
    operation: Option<compute::Op>,
    roles: bool,
) -> Result<Resource<super::GraphEntry>, compute::Error> {
    let commands = host.command_list().unwrap();
    host.declare_slot_capture(&commands, slot, shape).unwrap();
    if let Some(op) = operation {
        let input = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        host.write(&input, &1_f32.to_ne_bytes().repeat(7)).unwrap();
        let output = host.alloc(compute::Dtype::F32, &[7]).unwrap();
        let mut inputs = vec![Resource::new_borrow(input.rep())];
        if matches!(op, compute::Op::RmsNorm(_)) {
            inputs.push(Resource::new_borrow(input.rep()));
        }
        host.dispatch(&commands, op, &inputs, &output).unwrap();
        host.drop_tensor(input).unwrap();
        host.drop_tensor(output).unwrap();
    }
    if roles {
        record_test_decode_roles(host, &commands, slot);
    }
    host.create_graph(commands)
}

fn capture_arithmetic_copy(
    host: &mut Host<CpuBackend>,
    slot: &Resource<BatchSlotEntry>,
    shape: BatchShape,
) -> Result<Resource<super::GraphEntry>, compute::Error> {
    capture_arithmetic_body(host, slot, shape, Some(compute::Op::Copy), true)
}

#[test]
fn capture_fixtures_use_the_requested_slot_shape() {
    let (mut host, _pool, slot) = verify_input_fixture(CpuBackend::new(), 3, 1);
    let shape = BatchShape {
        selection: BatchSelection::Forward,
        ..test_capture_shape(false)
    };
    let graph = capture_arithmetic_copy(&mut host, &slot, shape).unwrap();
    assert_eq!(
        host.table
            .get(&graph)
            .unwrap()
            .slot_capture
            .as_ref()
            .unwrap()
            .shape,
        shape
    );
    host.drop_graph(graph).unwrap();
}
