//! Differential coverage for affine-quantized embedding lookup.

use forja_core::{Backend, CommandList, DType, Op, Slice, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{TensorSpec, assert_backends_agree};

#[test]
fn quantized_embedding_matches_cpu_for_formats_groups_and_views() {
    let reference = CpuBackend::new();
    let candidate = MetalBackend::new().unwrap();
    let ids = [0_u32, 32]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    for bits in [4, 8] {
        for group_size in [32, 64, 128] {
            for parameter_dtype in [DType::F16, DType::BF16] {
                let hidden = 256;
                let packed_width = hidden * u32::from(bits) / 32;
                let groups = hidden / group_size;
                assert_backends_agree(
                    &reference,
                    &candidate,
                    Op::QuantEmbed { bits, group_size },
                    &[
                        TensorSpec::contiguous(DType::U32, &[33, packed_width]),
                        TensorSpec::contiguous(parameter_dtype, &[33, groups]),
                        TensorSpec::contiguous(parameter_dtype, &[33, groups]),
                        TensorSpec::initialized(DType::U32, &[2], ids.clone()),
                    ],
                    &TensorSpec::contiguous(DType::F32, &[2, hidden]),
                )
                .unwrap();
            }
        }
    }

    assert_backends_agree(
        &reference,
        &candidate,
        Op::QuantEmbed {
            bits: 4,
            group_size: 64,
        },
        &[
            TensorSpec::sliced(
                DType::U32,
                &[33, 33],
                &[Slice::new(0, 33, 1).unwrap(), Slice::new(1, 32, 1).unwrap()],
            ),
            TensorSpec::sliced(
                DType::BF16,
                &[33, 5],
                &[Slice::new(0, 33, 1).unwrap(), Slice::new(1, 4, 1).unwrap()],
            ),
            TensorSpec::sliced(
                DType::BF16,
                &[33, 5],
                &[Slice::new(0, 33, 1).unwrap(), Slice::new(1, 4, 1).unwrap()],
            ),
            TensorSpec::initialized(DType::U32, &[2], ids),
        ],
        &TensorSpec::sliced(
            DType::F32,
            &[2, 257],
            &[Slice::new(0, 2, 1).unwrap(), Slice::new(1, 256, 1).unwrap()],
        ),
    )
    .unwrap();
}

#[test]
fn quantized_embedding_reports_a_gpu_produced_bad_id() {
    let backend = MetalBackend::new().unwrap();
    let logits = backend.alloc(DType::F32, &[1, 34]).unwrap();
    let ids = backend.alloc(DType::U32, &[1]).unwrap();
    let packed = backend.alloc(DType::U32, &[33, 32]).unwrap();
    let scales = backend.alloc(DType::BF16, &[33, 4]).unwrap();
    let biases = backend.alloc(DType::BF16, &[33, 4]).unwrap();
    let output = backend.alloc(DType::F32, &[1, 256]).unwrap();
    let mut values = vec![0.0_f32; 34];
    values[33] = 1.0;
    backend
        .write(
            &logits,
            &values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Argmax, &[&logits], &ids).unwrap();
    commands
        .dispatch(
            Op::QuantEmbed {
                bits: 4,
                group_size: 64,
            },
            &[&packed, &scales, &biases, &ids],
            &output,
        )
        .unwrap();

    assert_eq!(
        backend.submit(commands).unwrap().wait(),
        Err(forja_core::BackendError::IndexOutOfRange { index: 33 })
    );
}
