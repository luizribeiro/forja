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

fn grouped(backend: &impl Backend, tokens: u32, grouping: u32, initial: f32) -> (Vec<u8>, Vec<u8>) {
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
    let mut previous = floats(backend, &[1, 7, 7], std::iter::repeat_n(initial, 49));
    let mut output = Vec::new();
    for start in (0..tokens).step_by(usize::try_from(grouping).unwrap()) {
        let rows = grouping.min(tokens - start);
        let signal = |offset: u32| {
            (start * 7..(start + rows) * 7).map(move |index| {
                f32::from(u16::try_from((index + offset) % 17).unwrap()) / 128.0 - 0.0625
            })
        };
        let ordinary = [
            floats(backend, &[rows, 1, 7], signal(0)),
            floats(backend, &[rows, 1, 7], signal(5)),
            floats(backend, &[rows, 1, 7], signal(11)),
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
        let y = backend.alloc(DType::F32, &[rows, 1, 7]).unwrap();
        let next = backend.alloc(DType::F32, &[1, 7, 7]).unwrap();
        let layout = DeltaBlockScratch::new(rows, 1, 7, 7).unwrap();
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

fn assert_grouping(backend: &impl Backend) {
    for tokens in [1, 7, 33, 63, 64, 65, 4097] {
        for initial in [0.0, 0.125] {
            let expected = grouped(backend, tokens, 64, initial);
            for grouping in [512, 2048] {
                assert_eq!(
                    expected,
                    grouped(backend, tokens, grouping, initial),
                    "tokens={tokens} grouping={grouping} initial={initial}"
                );
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
