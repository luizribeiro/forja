use super::*;
use forja_core::BackendDispatchData;
use forja_cpu::CpuBackend;
use forja_testing::TensorSpec;

#[derive(Clone, Copy)]
struct Canary<'a> {
    load: &'a str,
    pointer: &'a str,
    alignment: u32,
    zero: &'a str,
}

const Q8_VECTOR_WORDS: Canary<'static> = Canary {
    load: "*(device const uint4 *)(packed + index * sizeof(uint))",
    pointer: "packed + index * sizeof(uint)",
    alignment: 16,
    zero: "uint4(0)",
};

fn assert_variant(
    name: Option<&str>,
    op: Op,
    inputs: &[TensorSpec],
    output: &TensorSpec,
    canary: Canary<'_>,
    skew_operand: Option<usize>,
) {
    let backend = MetalBackend::new().unwrap();
    let source = crate::storage::shader_source().replace(
        canary.load,
        &format!(
            "(ulong({}) % {} == 0 ? {} : {})",
            canary.pointer, canary.alignment, canary.load, canary.zero
        ),
    );
    assert_ne!(
        source,
        crate::storage::shader_source(),
        "{name:?}: missing canary site"
    );
    *backend.pipelines.lock().unwrap() = PipelineCache::new(&backend.device, &source).unwrap();
    forja_testing::assert_backends_agree_many_selected_with(
        &CpuBackend::new(),
        &backend,
        op,
        inputs,
        std::slice::from_ref(output),
        &[0],
        |commands| {
            let dispatch = commands.last_dispatch().unwrap();
            let pick = name.map(|name| backend.validate_variant(dispatch, name).unwrap());
            if let Some(index) = skew_operand {
                crate::storage::skew_read_origin_for_test(&backend, &dispatch.inputs()[index]);
            }
            if let Some(pick) = pick {
                assert!(commands.set_last_backend_data(BackendDispatchData::new(pick)));
            }
        },
    )
    .unwrap();
}

fn quant_inputs(
    rows: u32,
    columns: u32,
    bits: u8,
    dtype: DType,
    skew_input: bool,
    skew_words: bool,
) -> Vec<TensorSpec> {
    let inner = 1024;
    let mut inputs = vec![
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[columns, inner * u32::from(bits) / 32]),
        TensorSpec::contiguous(DType::F16, &[columns, 16]),
        TensorSpec::contiguous(DType::F16, &[columns, 16]),
    ];
    if skew_input {
        inputs[0] = TensorSpec::sliced(
            dtype,
            &[rows, inner + 4],
            &[
                forja_core::Slice::new(0, rows, 1).unwrap(),
                forja_core::Slice::new(1, inner, 1).unwrap(),
            ],
        );
    }
    if skew_words {
        let width = inner * u32::from(bits) / 32;
        inputs[1] = TensorSpec::sliced(
            DType::U32,
            &[columns, width + 4],
            &[
                forja_core::Slice::new(0, columns, 1).unwrap(),
                forja_core::Slice::new(1, width, 1).unwrap(),
            ],
        );
    }
    inputs
}

#[test]
fn q8_fast_packed_load_checks_the_absolute_address() {
    for (variant, rows, columns) in [
        ("quant-matmul.q8-fast-gemv", 1, 1),
        ("quant-matmul.q8-fast-gemv", 4, 9),
        ("quant-matmul.q8-fast-gemv-n2", 1, 9),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let mut inputs = quant_inputs(rows, columns, 8, dtype, false, false);
            inputs[0] = TensorSpec::sliced(
                dtype,
                &[rows * 1024 + 4],
                &[forja_core::Slice::new(4, rows * 1024, 1).unwrap()],
            )
            .reshaped(&[rows, 1024]);
            assert_variant(
                Some(variant),
                Op::QuantMatmul {
                    bits: 8,
                    group_size: 64,
                },
                &inputs,
                &TensorSpec::contiguous(dtype, &[rows, columns]),
                Canary {
                    load: "*reinterpret_cast<device const uint2 *>(packed + packed_index * sizeof(uint))",
                    pointer: "packed + packed_index * sizeof(uint)",
                    alignment: 8,
                    zero: "uint2(0)",
                },
                Some(1),
            );
        }
    }
}

#[test]
fn gather_vectors_check_absolute_weight_alignment() {
    let rows = 3;
    let indices = (0..rows * 2)
        .map(|i: u32| i % 7)
        .flat_map(u32::to_ne_bytes)
        .collect();
    assert_variant(
        None,
        Op::GatherMatmul,
        &[
            TensorSpec::contiguous(DType::F32, &[rows, 1024]),
            TensorSpec::sliced(
                DType::F32,
                &[7, 1024, 36],
                &[
                    forja_core::Slice::new(0, 7, 1).unwrap(),
                    forja_core::Slice::new(0, 1024, 1).unwrap(),
                    forja_core::Slice::new(1, 32, 1).unwrap(),
                ],
            ),
            TensorSpec::initialized(DType::U32, &[rows, 2], indices),
        ],
        &TensorSpec::contiguous(DType::F32, &[rows, 2, 32]),
        Canary {
            load: "*reinterpret_cast<device const float4 *>(buffer + index * 4)",
            pointer: "buffer + index * 4",
            alignment: 16,
            zero: "float4(0)",
        },
        None,
    );
}

fn assert_float_vector_origins(
    variant: &str,
    bits: u8,
    rows: u32,
    skew_input: bool,
    skew_operand: Option<usize>,
) {
    for (dtype, ty, bytes) in [
        (DType::F32, "float4", 4),
        (DType::F16, "half4", 2),
        (DType::BF16, "bfloat4", 2),
    ] {
        let load = format!("*reinterpret_cast<device const {ty} *>(buffer + index * {bytes})");
        let pointer = format!("buffer + index * {bytes}");
        let zero = format!("{ty}(0)");
        let mut inputs = quant_inputs(rows, 8, bits, dtype, skew_input, false);
        if skew_operand.is_some() {
            inputs[0] = TensorSpec::sliced(
                dtype,
                &[rows * 1024 + 4],
                &[forja_core::Slice::new(0, rows * 1024, 1).unwrap()],
            )
            .reshaped(&[rows, 1024]);
        }
        assert_variant(
            Some(variant),
            Op::QuantMatmul {
                bits,
                group_size: 64,
            },
            &inputs,
            &TensorSpec::contiguous(dtype, &[rows, 8]),
            Canary {
                load: &load,
                pointer: &pointer,
                alignment: 4 * bytes,
                zero: &zero,
            },
            skew_operand,
        );
    }
}

#[test]
fn contiguous_float_vectors_check_absolute_alignment() {
    for (variant, bits, skew_input, skew_operand) in [
        ("quant-matmul.gemv", 4, true, None),
        ("quant-matmul.gemv", 8, false, Some(0)),
        ("quant-matmul.q8-fast-gemv", 8, false, Some(0)),
        ("quant-matmul.q8-fast-gemv-n2", 8, false, Some(0)),
    ] {
        assert_float_vector_origins(variant, bits, 1, skew_input, skew_operand);
    }
}

#[test]
fn q8_fast_activation_vectors_check_multiple_row_origins() {
    assert_float_vector_origins("quant-matmul.q8-fast-gemv", 8, 4, false, Some(0));
}

#[test]
fn packed_vector_words_check_absolute_alignment() {
    for (variant, ty, alignment) in [
        ("quant-matmul.vector8", "uint2", 8),
        ("quant-matmul.vector16", "uint4", 16),
    ] {
        let load = format!("*reinterpret_cast<device const {ty} *>(row + index * 4)");
        let zero = format!("{ty}(0)");
        assert_variant(
            Some(variant),
            Op::QuantMatmul {
                bits: 4,
                group_size: 64,
            },
            &quant_inputs(1, 8, 4, DType::F32, false, true),
            &TensorSpec::contiguous(DType::F32, &[1, 8]),
            Canary {
                load: &load,
                pointer: "row + index * 4",
                alignment,
                zero: &zero,
            },
            None,
        );
    }
}

#[test]
fn q8_vector_words_check_absolute_alignment() {
    for (variant, rows, width, offset) in [
        ("quant-matmul.gemv", 1, 260, 1),
        ("quant-matmul.gemv", 1, 257, 0),
        ("quant-matmul.row-gemv", 4, 260, 1),
        ("quant-matmul.row-gemv", 4, 257, 0),
    ] {
        let mut inputs = quant_inputs(rows, 8, 8, DType::F32, false, false);
        inputs[1] = TensorSpec::sliced(
            DType::U32,
            &[8, width],
            &[
                forja_core::Slice::new(0, 8, 1).unwrap(),
                forja_core::Slice::new(offset, 256, 1).unwrap(),
            ],
        );
        assert_variant(
            Some(variant),
            Op::QuantMatmul {
                bits: 8,
                group_size: 64,
            },
            &inputs,
            &TensorSpec::contiguous(DType::F32, &[rows, 8]),
            Q8_VECTOR_WORDS,
            None,
        );
    }
}

#[test]
fn q8_packed_vectors_check_absolute_base() {
    for (variant, rows) in [("quant-matmul.gemv", 1), ("quant-matmul.row-gemv", 4)] {
        let mut inputs = quant_inputs(rows, 9, 8, DType::F32, false, false);
        inputs[1] = TensorSpec::sliced(
            DType::U32,
            &[9 * 256 + 4],
            &[forja_core::Slice::new(0, 9 * 256, 1).unwrap()],
        )
        .reshaped(&[9, 256]);
        assert_variant(
            Some(variant),
            Op::QuantMatmul {
                bits: 8,
                group_size: 64,
            },
            &inputs,
            &TensorSpec::contiguous(DType::F32, &[rows, 9]),
            Q8_VECTOR_WORDS,
            Some(1),
        );
    }
}

#[test]
fn grouped_input_vectors_check_absolute_alignment() {
    let mut inputs = quant_inputs(33, 32, 4, DType::F32, true, false);
    inputs[1] = TensorSpec::contiguous(DType::U32, &[7, 32, 128]);
    inputs[2] = TensorSpec::contiguous(DType::F16, &[7, 32, 16]);
    inputs[3] = TensorSpec::contiguous(DType::F16, &[7, 32, 16]);
    let indices = (0..66_u32)
        .map(|i| i % 7)
        .flat_map(u32::to_ne_bytes)
        .collect();
    inputs.push(TensorSpec::initialized(DType::U32, &[33, 2], indices));
    assert_variant(
        Some("gather-quant-matmul.grouped-32x32-2x2"),
        Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &TensorSpec::contiguous(DType::F32, &[33, 2, 32]),
        Canary {
            load: "*reinterpret_cast<device const float4 *>(source)",
            pointer: "source",
            alignment: 16,
            zero: "float4(0)",
        },
        None,
    );
}

#[test]
fn tiled_quant_input_checks_absolute_alignment() {
    let mut inputs = quant_inputs(33, 32, 4, DType::F32, false, false);
    inputs[0] = TensorSpec::sliced(
        DType::F32,
        &[33 * 1024 + 1],
        &[forja_core::Slice::new(1, 33 * 1024, 1).unwrap()],
    )
    .reshaped(&[33, 1024]);
    assert_variant(
        Some("quant-matmul.tiled"),
        Op::QuantMatmul {
            bits: 4,
            group_size: 64,
        },
        &inputs,
        &TensorSpec::contiguous(DType::F32, &[33, 32]),
        Canary {
            load: "*reinterpret_cast<device const QuantizedGemmRead16 *>(\n                    source + read * vector_elements)",
            pointer: "source + read * vector_elements",
            alignment: 16,
            zero: "QuantizedGemmRead16{}",
        },
        None,
    );
}
