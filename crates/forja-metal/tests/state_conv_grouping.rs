use forja_core::{Backend, CommandList, DType, Op, StateOp, Submission, Tensor};
use forja_testing::{DeterministicValues, generated_tensor_bytes, state_test_metadata};

fn tensor(
    backend: &impl Backend,
    dtype: DType,
    shape: &[u32],
    values: impl Iterator<Item = f32>,
) -> Tensor {
    let output = backend.alloc(dtype, shape).unwrap();
    let bytes = values
        .flat_map(|value| match dtype {
            DType::F32 => value.to_le_bytes().to_vec(),
            DType::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    backend.write(&output, &bytes).unwrap();
    output
}

fn grouped(
    backend: &impl Backend,
    dtype: DType,
    tokens: u32,
    grouping: u32,
    initial: f32,
) -> (Vec<u8>, Vec<u8>) {
    let (access, specs) = state_test_metadata().unwrap();
    let metadata = specs.map(|spec| {
        let output = backend
            .alloc(spec.dtype(), spec.allocation_shape())
            .unwrap();
        backend
            .write(
                &output,
                &generated_tensor_bytes(&spec, &mut DeterministicValues::new(1)).unwrap(),
            )
            .unwrap();
        output
    });
    let weights = tensor(
        backend,
        dtype,
        &[4, 7],
        (0_u16..28).map(|index| f32::from(index / 7 + 1) / 64.0),
    );
    let mut history = tensor(backend, dtype, &[3, 7], std::iter::repeat_n(initial, 21));
    let mut output = Vec::new();
    for start in (0..tokens).step_by(usize::try_from(grouping).unwrap()) {
        let rows = grouping.min(tokens - start);
        let input = tensor(
            backend,
            dtype,
            &[rows, 7],
            (start * 7..(start + rows) * 7)
                .map(|index| f32::from(u16::try_from(index % 17).unwrap()) / 128.0 - 0.0625),
        );
        let y = backend.alloc(dtype, &[rows, 7]).unwrap();
        let next = backend.alloc(dtype, &[3, 7]).unwrap();
        let mut inputs = vec![&input, &weights, &history];
        inputs.extend(&metadata);
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::State(StateOp::CausalDepthwiseConv(access)),
                &inputs,
                &[&y, &next],
            )
            .unwrap();
        backend.submit(commands).unwrap().wait().unwrap();
        output.extend(backend.read(&y).unwrap());
        for old in [&input, &history, &y] {
            backend.release(old).unwrap();
        }
        history = next;
    }
    let carry = backend.read(&history).unwrap();
    for old in metadata.iter().chain([&weights, &history]) {
        backend.release(old).unwrap();
    }
    (output, carry)
}

fn assert_grouping(backend: &impl Backend) {
    for dtype in [DType::F32, DType::BF16] {
        for tokens in [1, 7, 33, 63, 64, 65, 4097] {
            for initial in [0.0, 0.125] {
                let expected = grouped(backend, dtype, tokens, 64, initial);
                for grouping in [512, 2048] {
                    assert_eq!(
                        expected,
                        grouped(backend, dtype, tokens, grouping, initial),
                        "{dtype:?} tokens={tokens} grouping={grouping}"
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_convolution_carries_boundary_history_without_changing_bits() {
    assert_grouping(&forja_cpu::CpuBackend::new());
}

#[test]
fn metal_convolution_carries_boundary_history_without_changing_bits() {
    assert_grouping(&forja_metal::MetalBackend::new().unwrap());
}
