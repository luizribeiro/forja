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

fn assert_variant(
    name: Option<&str>,
    op: Op,
    inputs: &[TensorSpec],
    output: &TensorSpec,
    canary: Canary<'_>,
    skew_packed: bool,
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
            if skew_packed {
                let packed = &dispatch.inputs()[1];
                crate::storage::skew_read_origin_for_test(&backend, packed);
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
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let mut inputs = quant_inputs(1, 1, 8, dtype, false, false);
        inputs[0] = TensorSpec::sliced(
            dtype,
            &[1, 1028],
            &[
                forja_core::Slice::new(0, 1, 1).unwrap(),
                forja_core::Slice::new(4, 1024, 1).unwrap(),
            ],
        );
        inputs[1] = TensorSpec::sliced(
            DType::U32,
            &[1, 257],
            &[
                forja_core::Slice::new(0, 1, 1).unwrap(),
                forja_core::Slice::new(0, 256, 1).unwrap(),
            ],
        );
        assert_variant(
            Some("quant-matmul.q8-fast-gemv"),
            Op::QuantMatmul {
                bits: 8,
                group_size: 64,
            },
            &inputs,
            &TensorSpec::contiguous(dtype, &[1, 1]),
            Canary {
                load: "*reinterpret_cast<device const uint2 *>(packed + packed_index * sizeof(uint))",
                pointer: "packed + packed_index * sizeof(uint)",
                alignment: 8,
                zero: "uint2(0)",
            },
            true,
        );
    }
}
