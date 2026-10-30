use super::{block::StageTest, *};
use forja_core::{Backend, BackendDispatchData, CommandList, DType, DeltaBlockScratch, Submission};
use forja_testing::{
    DeterministicValues, assert_outputs_agree, generated_tensor_bytes, state_test_metadata,
};
use std::ops::Range;

fn written_ranges(
    stage: usize,
    layout: DeltaBlockScratch,
    rows: usize,
    keys: usize,
) -> Vec<Range<usize>> {
    let offsets = layout
        .offsets()
        .map(|offset| usize::try_from(offset).unwrap());
    match stage {
        0 => vec![
            offsets[0]..offsets[2],
            offsets[2]..offsets[2] + rows,
            offsets[3]..offsets[3] + rows,
        ],
        1 => vec![
            offsets[4]..offsets[4] + 64 * 33,
            offsets[5]..offsets[5] + 64 * keys,
        ],
        2 => vec![
            offsets[6]..offsets[6] + rows * 33,
            offsets[7]..offsets[7] + rows * 33,
        ],
        _ => vec![],
    }
}

fn check(
    stage: usize,
    kernel: &'static str,
    panel: Option<u32>,
    keys: u32,
) -> Result<(), BackendError> {
    for rows in [7, 33, 64] {
        for underflow in [false, true] {
            run(stage, kernel, panel, keys, rows, underflow)?;
        }
    }
    Ok(())
}

fn run(
    stage: usize,
    kernel: &'static str,
    panel: Option<u32>,
    keys: u32,
    rows: u32,
    underflow: bool,
) -> Result<(), BackendError> {
    let backend = MetalBackend::new().unwrap();
    let layout = DeltaBlockScratch::new(rows, 1, keys, 33).unwrap();
    let n = usize::try_from(rows).unwrap();
    let k = usize::try_from(keys).unwrap();
    let mut rng = DeterministicValues::new(73);
    let query = (0..n * k)
        .map(|_| rng.next_f32() * 0.03)
        .collect::<Vec<_>>();
    let key = (0..n * k)
        .map(|_| rng.next_f32() * 0.03)
        .collect::<Vec<_>>();
    let value = (0..n * 33).map(|_| rng.next_f32()).collect::<Vec<_>>();
    let decay = (0..n)
        .map(|row| {
            if underflow && row == 3 {
                f32::NEG_INFINITY
            } else {
                -0.015
            }
        })
        .collect::<Vec<_>>();
    let beta = vec![0.25; n];
    let initial = vec![0.125; k * 33];
    let [expected_scratch, expected_output, expected_state] =
        forja_cpu::block_reference([&query, &key, &value, &decay, &beta], &initial, [keys, 33])
            .unwrap();
    let mut scratch = expected_scratch.clone();
    let ranges = written_ranges(stage, layout, n, k);
    for range in &ranges {
        scratch[range.clone()].fill(f32::NAN);
    }
    let mut tensors = Vec::new();
    for (shape, values) in [
        (&[rows, 1, keys][..], &query),
        (&[rows, 1, keys], &key),
        (&[rows, 1, 33], &value),
        (&[rows, 1], &decay),
        (&[rows, 1], &beta),
        (&[1, keys, 33], &initial),
    ] {
        tensors.push(floats(&backend, shape, values));
    }
    let (access, metadata) = state_test_metadata().unwrap();
    for spec in metadata {
        let tensor = backend
            .alloc(spec.dtype(), spec.allocation_shape())
            .unwrap();
        backend
            .write(&tensor, &generated_tensor_bytes(&spec, &mut rng).unwrap())
            .unwrap();
        tensors.push(tensor);
    }
    let output = floats(&backend, &[rows, 1, 33], &vec![f32::NAN; n * 33]);
    let next = floats(&backend, &[1, keys, 33], &vec![f32::NAN; k * 33]);
    let scratch = floats(&backend, &[layout.elements()], &scratch);
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::State(StateOp::DeltaRuleBlock(access)),
            &tensors.iter().collect::<Vec<_>>(),
            &[&output, &next, &scratch],
        )
        .unwrap();
    assert!(
        commands.set_last_backend_data(BackendDispatchData::new(StageTest {
            stage,
            kernel,
            panel
        }))
    );
    backend.submit(commands)?.wait()?;
    let actual = backend.read(&scratch).unwrap();
    for range in ranges {
        agree(
            &expected_scratch[range.clone()],
            &actual[range.start * 4..range.end * 4],
        );
    }
    for (compare, tensor, expected) in [
        (stage == 2, &next, &expected_state),
        (stage == 3, &output, &expected_output),
    ] {
        if compare {
            agree(expected, &backend.read(tensor).unwrap());
        }
    }
    Ok(())
}

fn floats(backend: &MetalBackend, shape: &[u32], values: &[f32]) -> Tensor {
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

fn agree(expected: &[f32], actual: &[u8]) {
    let expected = expected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_outputs_agree(DType::F32, &expected, actual).unwrap();
}

#[test]
fn block_stage_harness_refuses_keys_above_the_backend_limit() {
    assert_eq!(
        check(0, "state_block_gram", None, 257),
        Err(BackendError::UnsupportedOperation)
    );
}

#[test]
fn scalar_gram_matches_cpu_stage_oracle() {
    check(0, "state_block_gram", None, 128).unwrap();
}

#[test]
fn scalar_solve_matches_cpu_stage_oracle() {
    check(1, "state_block_solve", None, 128).unwrap();
}

#[test]
fn scalar_carry_matches_cpu_stage_oracle() {
    check(2, "state_block_carry", None, 128).unwrap();
}
