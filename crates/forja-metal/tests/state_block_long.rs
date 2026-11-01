#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use super::*;

const KEY_HEADS: u32 = 16;
const VALUE_HEADS: u32 = 48;
const WIDTH: u32 = 128;
const TOKENS: u32 = 64;

fn target(head: u32, column: u32) -> f32 {
    ((head * 7 + column * 3) % 97) as f32 / 128.0 + 0.125
}

fn key(mode: u32, column: u32) -> f32 {
    if column >= 64 {
        0.0
    } else if (mode & column).count_ones().is_multiple_of(2) {
        0.125
    } else {
        -0.125
    }
}

// Each Hadamard mode is visited once per 64 tokens, giving a geometric series.
fn reference(tokens: u32, log_decay: f32, beta: f32) -> Vec<f32> {
    let g = f64::from(log_decay);
    let b = f64::from(beta);
    let ratio = (64.0 * g).exp() * (1.0 - b);
    let initial = 0.125 * (f64::from(tokens) * g).exp();
    let coefficients = (0..64)
        .map(|column| {
            (0..64)
                .map(|mode| {
                    if tokens <= mode {
                        return 0.0;
                    }
                    let visits = (tokens - 1 - mode) / 64 + 1;
                    let age = (tokens - 1 - mode) % 64;
                    f64::from(key(mode, column))
                        * b
                        * (f64::from(age) * g).exp()
                        * (1.0 - ratio.powf(f64::from(visits)))
                        / (1.0 - ratio)
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let initial_mode = initial * (1.0 - b).powf(f64::from(tokens.div_ceil(64)));
    (0..VALUE_HEADS)
        .flat_map(|head| {
            let coefficients = &coefficients;
            (0..WIDTH).flat_map(move |column| {
                (0..WIDTH).map(move |value| {
                    if column < 64 {
                        (initial_mode
                            + coefficients[usize::try_from(column).unwrap()]
                                * f64::from(target(head, value))) as f32
                    } else {
                        initial as f32
                    }
                })
            })
        })
        .collect()
}

fn write(backend: &forja_metal::MetalBackend, shape: &[u32], values: &[f32]) -> forja_core::Tensor {
    let tensor = backend.alloc(DType::F32, shape).unwrap();
    backend
        .write(
            &tensor,
            &values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    tensor
}

fn inputs(
    backend: &forja_metal::MetalBackend,
    log_decay: f32,
    beta: f32,
) -> Vec<forja_core::Tensor> {
    let keys = (0..TOKENS)
        .flat_map(|token| {
            (0..KEY_HEADS).flat_map(move |_| (0..WIDTH).map(move |column| key(token, column)))
        })
        .collect::<Vec<_>>();
    let values = (0..TOKENS)
        .flat_map(|_| {
            (0..VALUE_HEADS).flat_map(|head| (0..WIDTH).map(move |column| target(head, column)))
        })
        .collect::<Vec<_>>();
    vec![
        write(backend, &[TOKENS, KEY_HEADS, WIDTH], &keys),
        write(backend, &[TOKENS, KEY_HEADS, WIDTH], &keys),
        write(backend, &[TOKENS, VALUE_HEADS, WIDTH], &values),
        write(
            backend,
            &[TOKENS, VALUE_HEADS],
            &vec![log_decay; usize::try_from(TOKENS * VALUE_HEADS).unwrap()],
        ),
        write(
            backend,
            &[TOKENS, VALUE_HEADS],
            &vec![beta; usize::try_from(TOKENS * VALUE_HEADS).unwrap()],
        ),
    ]
}

fn measure(log_decay: f32, beta: f32) -> Vec<f64> {
    let backend = forja_metal::MetalBackend::new().unwrap();
    let mut inputs = inputs(&backend, log_decay, beta);
    let state_elements = usize::try_from(VALUE_HEADS * WIDTH * WIDTH).unwrap();
    let states = [
        write(
            &backend,
            &[VALUE_HEADS, WIDTH, WIDTH],
            &vec![0.125; state_elements],
        ),
        write(
            &backend,
            &[VALUE_HEADS, WIDTH, WIDTH],
            &vec![0.0; state_elements],
        ),
    ];
    let (access, metadata) = state_test_metadata().unwrap();
    let mut rng = DeterministicValues::new(73);
    for spec in metadata {
        let tensor = backend
            .alloc(spec.dtype(), spec.allocation_shape())
            .unwrap();
        backend
            .write(&tensor, &generated_tensor_bytes(&spec, &mut rng).unwrap())
            .unwrap();
        inputs.push(tensor);
    }
    let output = backend
        .alloc(DType::F32, &[TOKENS, VALUE_HEADS, WIDTH])
        .unwrap();
    let scratch = backend
        .alloc(
            DType::F32,
            &[DeltaBlockScratch::new(TOKENS, VALUE_HEADS, WIDTH, WIDTH)
                .unwrap()
                .elements()],
        )
        .unwrap();
    let mut errors = Vec::new();
    let mut completed = 0_u32;
    for checkpoint in [64, 32_768, 131_072, 262_144] {
        while completed < checkpoint {
            let end = checkpoint.min(completed + 2048);
            let mut commands = CommandList::new();
            while completed < end {
                let current = usize::try_from((completed / TOKENS) % 2).unwrap();
                let mut operands = inputs[..5].iter().collect::<Vec<_>>();
                operands.push(&states[current]);
                operands.extend(inputs[5..].iter());
                commands
                    .dispatch_many(
                        Op::State(StateOp::DeltaRuleBlock(access)),
                        &operands,
                        &[&output, &states[1 - current], &scratch],
                    )
                    .unwrap();
                let pick = backend
                    .validate_variant(
                        commands.last_dispatch().unwrap(),
                        "delta-rule-block.wy64-mma-v32",
                    )
                    .unwrap();
                assert!(commands.set_last_backend_data(BackendDispatchData::new(pick)));
                completed += TOKENS;
            }
            backend.submit(commands).unwrap().wait().unwrap();
        }
        let current = usize::try_from((completed / TOKENS) % 2).unwrap();
        let bytes = backend.read(&states[current]).unwrap();
        let (words, remainder) = bytes.as_chunks::<4>();
        assert_eq!(remainder.len(), 0);
        let actual = words
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect::<Vec<_>>();
        let error =
            forja_testing::normwise_relative_error(&reference(completed, log_decay, beta), &actual);
        eprintln!(
            "correctness diagnostic: recurrent tokens={completed} log_decay={log_decay} beta={beta} normwise_f64_reference_error={error}"
        );
        assert!(error.is_finite());
        errors.push(error);
    }
    errors
}

fn compare(log_decay: f32, beta: f32) {
    for error in measure(log_decay, beta) {
        assert!(
            error <= forja_testing::F32_TOLERANCE,
            "long-state drift {error}"
        );
    }
}

#[test]
fn metal_near_identity_recurrent_decay_reports_global_drift() {
    let errors = measure(-1.0e-7, 0.25);
    assert!(errors[0] <= forja_testing::F32_TOLERANCE);
    eprintln!(
        "correctness diagnostic only: near-identity global errors={errors:?}; long-state qualification remains separate from per-block operator tolerance"
    );
}

#[test]
fn metal_recurrent_state_drift_at_long_context() {
    compare(-0.015, 0.25);
}
#[test]
fn metal_recurrent_state_drift_with_slow_decay() {
    compare(-0.0001, 0.25);
}
#[test]
fn metal_recurrent_state_drift_with_decay_underflow() {
    compare(-1000.0, 0.25);
}

#[test]
fn orthogonal_closed_form_matches_token_reference() {
    for tokens in [1_u32, 7, 33, 65] {
        let count = usize::try_from(tokens).unwrap();
        let key = (0..tokens)
            .flat_map(|token| (0..WIDTH).map(move |column| key(token % 64, column)))
            .collect::<Vec<_>>();
        let values = (0..tokens)
            .flat_map(|_| (0..WIDTH).map(|column| target(0, column)))
            .collect::<Vec<_>>();
        let (_, expected) = forja_testing::delta_rule::recurrent(
            &key,
            &key,
            &values,
            &vec![-0.0001; count],
            &vec![0.25; count],
            &vec![0.125; usize::try_from(WIDTH * WIDTH).unwrap()],
            [128, 128],
        );
        let closed = reference(tokens, -0.0001, 0.25);
        assert!(
            forja_testing::normwise_relative_error(&expected, &closed[..expected.len()])
                <= forja_testing::F32_TOLERANCE
        );
    }
}
