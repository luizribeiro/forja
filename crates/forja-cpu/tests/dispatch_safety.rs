//! Property tests for the validated-operation boundary.

use forja_core::{Backend, CommandList, DType, Op, Slice, Submission, Tensor, ViewOp};
use forja_cpu::CpuBackend;
use proptest::prelude::*;

#[derive(Clone, Debug)]
struct TensorCase {
    dtype: DType,
    shape: Vec<u32>,
    view: u8,
}

#[derive(Clone, Debug)]
struct DispatchCase {
    op: Op,
    inputs: Vec<TensorCase>,
    output: TensorCase,
}

fn dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::F32),
        Just(DType::F16),
        Just(DType::BF16),
        Just(DType::I32),
        Just(DType::U32),
    ]
}

fn tensor_case() -> impl Strategy<Value = TensorCase> {
    (dtype(), prop::collection::vec(0_u32..=9, 0..=4), 0_u8..4)
        .prop_map(|(dtype, shape, view)| TensorCase { dtype, shape, view })
}

fn arbitrary_case() -> impl Strategy<Value = DispatchCase> {
    (
        0_u8..9,
        prop::collection::vec(tensor_case(), 0..=4),
        tensor_case(),
        any::<f32>(),
        any::<f32>(),
        any::<f32>(),
        any::<bool>(),
        any::<u32>(),
    )
        .prop_map(
            |(kind, inputs, output, eps, theta, scale, causal, q_start)| DispatchCase {
                op: operation(kind, eps, theta, scale, causal, q_start),
                inputs,
                output,
            },
        )
}

#[allow(clippy::too_many_lines)]
fn valid_case() -> impl Strategy<Value = DispatchCase> {
    (
        0_u8..9,
        prop_oneof![Just(DType::F32), Just(DType::F16), Just(DType::BF16)],
        prop::collection::vec(1_u32..=9, 0..=4),
        prop::array::uniform8(1_u32..=9),
        prop::array::uniform4(0_u8..4),
        0.0_f32..1.0,
        0.01_f32..1_000_000.0,
        -2.0_f32..2.0,
        any::<bool>(),
        any::<u32>(),
    )
        .prop_map(
            |(kind, dtype, mut shape, dims, views, eps, theta, scale, causal, q_start)| {
                let input = |shape: Vec<u32>, index: usize| TensorCase {
                    dtype,
                    shape,
                    view: views[index],
                };
                let output = |shape: Vec<u32>| TensorCase {
                    dtype,
                    shape,
                    view: views[3] % 3,
                };
                let (op, inputs, output) = match kind {
                    0 => (Op::Copy, vec![input(shape.clone(), 0)], output(shape)),
                    1 => (
                        Op::Add,
                        vec![input(shape.clone(), 0), input(shape.clone(), 1)],
                        output(shape),
                    ),
                    2 => (
                        Op::SiluMul,
                        vec![input(shape.clone(), 0), input(shape.clone(), 1)],
                        output(shape),
                    ),
                    3 => {
                        if shape.is_empty() {
                            shape.push(dims[0]);
                        }
                        let width = *shape.last().unwrap();
                        (
                            Op::RmsNorm { eps },
                            vec![input(shape.clone(), 0), input(vec![width], 1)],
                            output(shape),
                        )
                    }
                    4 => {
                        if shape.is_empty() {
                            shape.push(dims[0]);
                        }
                        (Op::Softmax, vec![input(shape.clone(), 0)], output(shape))
                    }
                    5 => {
                        let rope_shape = vec![dims[0], dims[1], 2 * (1 + dims[2] % 4)];
                        (
                            Op::Rope { theta },
                            vec![
                                input(rope_shape.clone(), 0),
                                TensorCase {
                                    dtype: DType::U32,
                                    shape: vec![dims[0]],
                                    view: views[1],
                                },
                            ],
                            output(rope_shape),
                        )
                    }
                    6 => (
                        Op::Embed,
                        vec![
                            input(vec![dims[0], dims[1]], 0),
                            TensorCase {
                                dtype: DType::U32,
                                shape: vec![dims[2]],
                                view: views[1],
                            },
                        ],
                        output(vec![dims[2], dims[1]]),
                    ),
                    7 => {
                        let batch = (dims[3] % 2 == 0).then_some(dims[0]);
                        let mut left = batch.into_iter().collect::<Vec<_>>();
                        left.extend([dims[1], dims[2]]);
                        let mut right = batch.into_iter().collect::<Vec<_>>();
                        right.extend([dims[2], dims[3]]);
                        let mut result = batch.into_iter().collect::<Vec<_>>();
                        result.extend([dims[1], dims[3]]);
                        (
                            Op::Matmul,
                            vec![input(left, 0), input(right, 1)],
                            output(result),
                        )
                    }
                    _ => {
                        let kv_heads = 1 + dims[0] % 3;
                        let q_heads = kv_heads * (1 + dims[1] % 3);
                        let q_len = dims[2];
                        let kv_len = q_len.max(dims[3]);
                        let width = dims[4];
                        let value_width = dims[5];
                        (
                            Op::Sdpa {
                                scale,
                                causal,
                                q_start: if causal { 0 } else { q_start },
                            },
                            vec![
                                input(vec![q_heads, q_len, width], 0),
                                input(vec![kv_heads, kv_len, width], 1),
                                input(vec![kv_heads, kv_len, value_width], 2),
                            ],
                            output(vec![q_heads, q_len, value_width]),
                        )
                    }
                };
                DispatchCase { op, inputs, output }
            },
        )
}

fn operation(kind: u8, eps: f32, theta: f32, scale: f32, causal: bool, q_start: u32) -> Op {
    match kind {
        0 => Op::Copy,
        1 => Op::Add,
        2 => Op::SiluMul,
        3 => Op::RmsNorm { eps },
        4 => Op::Softmax,
        5 => Op::Rope { theta },
        6 => Op::Embed,
        7 => Op::Matmul,
        _ => Op::Sdpa {
            scale,
            causal,
            q_start,
        },
    }
}

fn materialize(backend: &CpuBackend, case: &TensorCase) -> Tensor {
    let rank = case.shape.len();
    match case.view {
        1 if rank > 1 => {
            let mut base = case.shape.clone();
            base.reverse();
            let tensor = backend.alloc(case.dtype, &base).unwrap();
            let axes = (0..rank)
                .rev()
                .map(|axis| u8::try_from(axis).unwrap())
                .collect();
            backend.view(&tensor, ViewOp::Permute(axes)).unwrap()
        }
        2 if rank > 0 && case.shape[0] > 0 && case.shape[0] < 9 => {
            let mut base = case.shape.clone();
            base[0] += 1;
            let tensor = backend.alloc(case.dtype, &base).unwrap();
            let slices = case
                .shape
                .iter()
                .enumerate()
                .map(|(axis, &len)| Slice::new(u32::from(axis == 0), len, 1).unwrap())
                .collect();
            backend.view(&tensor, ViewOp::Slice(slices)).unwrap()
        }
        3 if rank > 0 && case.shape[0] > 1 => {
            let mut base = case.shape.clone();
            base[0] = 1;
            let tensor = backend.alloc(case.dtype, &base).unwrap();
            backend
                .view(&tensor, ViewOp::Broadcast(case.shape.clone()))
                .unwrap()
        }
        _ => backend.alloc(case.dtype, &case.shape).unwrap(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn accepted_dispatches_never_panic(
        case in prop_oneof![4 => valid_case(), 1 => arbitrary_case()],
    ) {
        let backend = CpuBackend::new();
        let inputs = case.inputs.iter().map(|case| materialize(&backend, case)).collect::<Vec<_>>();
        let input_refs = inputs.iter().collect::<Vec<_>>();
        let output = materialize(&backend, &case.output);
        let mut commands = CommandList::new();
        if commands.dispatch(case.op, &input_refs, &output).is_ok() {
            let _result = match backend.submit(commands) {
                Ok(submission) => submission.wait(),
                Err(error) => Err(error),
            };
        }
    }
}
