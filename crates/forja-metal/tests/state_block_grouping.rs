use forja_core::{Backend, CommandList, DType, DeltaBlockScratch, Op, StateOp, Submission, Tensor};
use forja_testing::{DeterministicValues, generated_tensor_bytes, state_test_metadata};

fn floats(backend: &impl Backend, shape: &[u32], values: impl Iterator<Item = f32>) -> Tensor {
    let tensor = backend.alloc(DType::F32, shape).unwrap();
    backend
        .write(
            &tensor,
            &values.flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
        )
        .unwrap();
    tensor
}

fn grouped(
    backend: &(impl Backend + 'static),
    tokens: u32,
    grouping: u32,
    initial: f32,
    width: u32,
    variant: Option<&str>,
) -> (Vec<u8>, Vec<u8>) {
    let (access, specs) = state_test_metadata().unwrap();
    let metadata = specs.map(|spec| {
        let tensor = backend
            .alloc(spec.dtype(), spec.allocation_shape())
            .unwrap();
        backend
            .write(
                &tensor,
                &generated_tensor_bytes(&spec, &mut DeterministicValues::new(1)).unwrap(),
            )
            .unwrap();
        tensor
    });
    let mut previous = floats(
        backend,
        &[1, 7, width],
        std::iter::repeat_n(initial, usize::try_from(7 * width).unwrap()),
    );
    let mut output = Vec::new();
    for start in (0..tokens).step_by(usize::try_from(grouping).unwrap()) {
        let rows = grouping.min(tokens - start);
        let signal = |columns: u32, offset: u32| {
            (start * columns..(start + rows) * columns).map(move |index| {
                f32::from(u16::try_from((index + offset) % 17).unwrap()) / 128.0 - 0.0625
            })
        };
        let ordinary = [
            floats(backend, &[rows, 1, 7], signal(7, 0)),
            floats(backend, &[rows, 1, 7], signal(7, 5)),
            floats(backend, &[rows, 1, width], signal(width, 11)),
            floats(
                backend,
                &[rows, 1],
                (start..start + rows).map(|row| {
                    if row % 97 == 93 {
                        f32::NEG_INFINITY
                    } else {
                        -0.015
                    }
                }),
            ),
            floats(
                backend,
                &[rows, 1],
                std::iter::repeat_n(0.25, usize::try_from(rows).unwrap()),
            ),
        ];
        let y = backend.alloc(DType::F32, &[rows, 1, width]).unwrap();
        let next = backend.alloc(DType::F32, &[1, 7, width]).unwrap();
        let layout = DeltaBlockScratch::new(rows, 1, 7, width).unwrap();
        let scratch = backend.alloc(DType::F32, &[layout.elements()]).unwrap();
        let mut inputs = ordinary.iter().collect::<Vec<_>>();
        inputs.push(&previous);
        inputs.extend(&metadata);
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::State(StateOp::DeltaRuleBlock(access)),
                &inputs,
                &[&y, &next, &scratch],
            )
            .unwrap();
        if let Some(name) = variant {
            let metal = (backend as &dyn std::any::Any)
                .downcast_ref::<forja_metal::MetalBackend>()
                .unwrap();
            let pick = metal
                .validate_variant(commands.last_dispatch().unwrap(), name)
                .unwrap();
            assert!(commands.set_last_backend_data(forja_core::BackendDispatchData::new(pick)));
        }
        backend.submit(commands).unwrap().wait().unwrap();
        output.extend(backend.read(&y).unwrap());
        for tensor in ordinary.iter().chain([&previous, &y, &scratch]) {
            backend.release(tensor).unwrap();
        }
        previous = next;
    }
    let state = backend.read(&previous).unwrap();
    for tensor in metadata.iter().chain([&previous]) {
        backend.release(tensor).unwrap();
    }
    (output, state)
}

fn assert_grouping(backend: &(impl Backend + 'static)) {
    for tokens in [1, 7, 33, 63, 64, 65, 4097] {
        for initial in [0.0, 0.125] {
            let expected = grouped(backend, tokens, 64, initial, 7, None);
            for grouping in [512, 2048] {
                assert_eq!(
                    expected,
                    grouped(backend, tokens, grouping, initial, 7, None),
                    "tokens={tokens} grouping={grouping} initial={initial}"
                );
            }
        }
    }
}

#[test]
fn metal_mma_panels_preserve_bits_across_widths_and_submission_groups() {
    let backend = forja_metal::MetalBackend::new().unwrap();
    for tokens in [65, 4097] {
        for width in [33, 128] {
            for initial in [0.0, 0.125] {
                let expected = grouped(
                    &backend,
                    tokens,
                    64,
                    initial,
                    width,
                    Some("delta-rule-block.wy64-mma-v32"),
                );
                for name in [
                    "delta-rule-block.wy64-mma-v16",
                    "delta-rule-block.wy64-mma-v32",
                ] {
                    for grouping in [64, 512, 2048] {
                        assert_eq!(
                            expected,
                            grouped(&backend, tokens, grouping, initial, width, Some(name)),
                            "tokens={tokens} width={width} group={grouping} pick={name}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_delta_blocks_preserve_bits_across_submission_groupings() {
    assert_grouping(&forja_cpu::CpuBackend::new());
}

#[test]
fn metal_delta_blocks_preserve_bits_across_submission_groupings() {
    assert_grouping(&forja_metal::MetalBackend::new().unwrap());
}
