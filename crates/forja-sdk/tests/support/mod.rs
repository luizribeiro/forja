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

pub fn attention_and_cache_copy_match_cpu() {
    let query_values = values(16 * 7 * 128, 7);
    let key_values = values(8 * 7 * 128, 8);
    let value_values = values(8 * 7 * 128, 9);
    let query = Tensor::from_slice(&query_values, &[16, 7, 128]).unwrap();
    let key = Tensor::from_slice(&key_values, &[8, 7, 128]).unwrap();
    let value = Tensor::from_slice(&value_values, &[8, 7, 128]).unwrap();
    let scale = 128.0_f32.sqrt().recip();
    let attended = forja_sdk::nn::ops::sdpa(&query, &key, &value, scale, true, 0)
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = cpu_dispatch(
        Op::Sdpa {
            scale,
            causal: true,
            q_start: 0,
        },
        &[
            Input::F32(&[16, 7, 128], &query_values),
            Input::F32(&[8, 7, 128], &key_values),
            Input::F32(&[8, 7, 128], &value_values),
        ],
        &[16, 7, 128],
    );
    assert_agrees(&expected, &attended);

    let source = Tensor::from_slice(&key_values, &[8, 7, 128]).unwrap();
    let cache = Tensor::from_slice(&vec![0.0_f32; 8 * 33 * 128], &[8, 33, 128]).unwrap();
    let mut destination = cache.narrow(1, 5, 7).unwrap();
    source.copy_into(&mut destination).unwrap();
    let copied = cache.to_vec().unwrap();
    let mut expected = vec![0.0_f32; 8 * 33 * 128];
    for head in 0..8 {
        let source_start = head * 7 * 128;
        let target_start = (head * 33 + 5) * 128;
        expected[target_start..target_start + 7 * 128]
            .copy_from_slice(&key_values[source_start..source_start + 7 * 128]);
    }
    assert_eq!(copied, expected);
}

pub fn neural_network_modules_match_cpu() {
    let input_values = values(7 * 1024, 10);
    let weight_values = values(2048 * 1024, 11);
    let input = Tensor::from_slice(&input_values, &[7, 1024]).unwrap();
    let linear =
        forja_sdk::nn::Linear::new(Tensor::from_slice(&weight_values, &[2048, 1024]).unwrap());
    let projected = linear.forward(&input).unwrap().to_vec().unwrap();
    let mut transposed_weight = vec![0.0; weight_values.len()];
    for output in 0..2048 {
        for input in 0..1024 {
            transposed_weight[input * 2048 + output] = weight_values[output * 1024 + input];
        }
    }
    let expected = cpu_dispatch(
        Op::Matmul,
        &[
            Input::F32(&[7, 1024], &input_values),
            Input::F32(&[1024, 2048], &transposed_weight),
        ],
        &[7, 2048],
    );
    assert_agrees(&expected, &projected);

    let norm_input = values(7 * 1024, 12);
    let norm_weight = values(1024, 13);
    let norm =
        forja_sdk::nn::RmsNorm::new(Tensor::from_slice(&norm_weight, &[1024]).unwrap(), 1.0e-6);
    let normalized = norm
        .forward(&Tensor::from_slice(&norm_input, &[7, 1024]).unwrap())
        .unwrap()
        .to_vec()
        .unwrap();
    let expected = cpu_dispatch(
        Op::RmsNorm { eps: 1.0e-6 },
        &[
            Input::F32(&[7, 1024], &norm_input),
            Input::F32(&[1024], &norm_weight),
        ],
        &[7, 1024],
    );
    assert_agrees(&expected, &normalized);

    let table_values = values(33 * 1024, 14);
    let ids = [0_u32, 32, 7, 1, 16, 8, 31];
    let embedding =
        forja_sdk::nn::Embedding::new(Tensor::from_slice(&table_values, &[33, 1024]).unwrap());
    let embedded = embedding
        .forward(&Tensor::from_slice(&ids, &[7]).unwrap())
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
