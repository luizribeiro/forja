use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_sdk::Tensor;
use forja_testing::{DeterministicValues, F32_TOLERANCE, normwise_relative_error};

pub fn matmul_rope_embedding_match_cpu() {
    let left_values = values(7 * 1024, 1);
    let right_values = values(1024 * 2048, 2);
    let product = Tensor::from_slice(&left_values, &[7, 1024])
        .unwrap()
        .matmul(&Tensor::from_slice(&right_values, &[1024, 2048]).unwrap())
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = cpu_dispatch(
        Op::Matmul,
        &[
            Input::F32(&[7, 1024], &left_values),
            Input::F32(&[1024, 2048], &right_values),
        ],
        &[7, 2048],
    );
    assert_agrees(&expected, &product);

    let rope_values = values(7 * 16 * 128, 3);
    let positions = [0_u32, 1, 2, 3, 4, 5, 32];
    let rotated = Tensor::from_slice(&rope_values, &[7, 16, 128])
        .unwrap()
        .rope(&Tensor::from_slice(&positions, &[7]).unwrap(), 1_000_000.0)
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = cpu_dispatch(
        Op::Rope { theta: 1_000_000.0 },
        &[
            Input::F32(&[7, 16, 128], &rope_values),
            Input::U32(&[7], &positions),
        ],
        &[7, 16, 128],
    );
    assert_agrees(&expected, &rotated);

    let table_values = values(33 * 1024, 4);
    let ids = [0_u32, 32, 7, 1, 16, 8, 31];
    let embedded = Tensor::from_slice(&table_values, &[33, 1024])
        .unwrap()
        .embedding(&Tensor::from_slice(&ids, &[7]).unwrap())
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = cpu_dispatch(
        Op::Embed,
        &[
            Input::F32(&[33, 1024], &table_values),
            Input::U32(&[7], &ids),
        ],
        &[7, 1024],
    );
    assert_agrees(&expected, &embedded);
}

enum Input<'a> {
    F32(&'a [u32], &'a [f32]),
    U32(&'a [u32], &'a [u32]),
}

fn cpu_dispatch(operation: Op, inputs: &[Input<'_>], output_shape: &[u32]) -> Vec<f32> {
    let backend = forja_cpu::CpuBackend::new();
    let inputs = inputs
        .iter()
        .map(|input| match input {
            Input::F32(shape, values) => {
                let tensor = backend.alloc(DType::F32, shape).unwrap();
                backend.write(&tensor, &f32_bytes(values)).unwrap();
                tensor
            }
            Input::U32(shape, values) => {
                let tensor = backend.alloc(DType::U32, shape).unwrap();
                backend.write(&tensor, &u32_bytes(values)).unwrap();
                tensor
            }
        })
        .collect::<Vec<_>>();
    let output = backend.alloc(DType::F32, output_shape).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(operation, &inputs.iter().collect::<Vec<_>>(), &output)
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    let bytes = backend.read(&output).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

pub fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut values = DeterministicValues::new(seed);
    (0..len).map(|_| values.next_f32() * 0.1).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn assert_agrees(expected: &[f32], actual: &[f32]) {
    let error = normwise_relative_error(expected, actual);
    assert!(
        error <= F32_TOLERANCE,
        "relative error {error} exceeded {F32_TOLERANCE}"
    );
}
