use super::*;

macro_rules! shared_rows {
    ($name:ident, $dtype:ident, $bits:literal) => {
        #[test]
        fn $name() {
            let backend = MetalBackend::new().unwrap();
            for (inner, columns) in [(5120, 17408), (17408, 5120), (4096, 4097)] {
                for parameters in [DType::F16, DType::BF16] {
                    assert_shared_rows(&backend, DType::$dtype, $bits, inner, columns, parameters);
                }
            }
        }
    };
}
shared_rows!(shared_q4_f32_preserves_single_row_bits, F32, 4);
shared_rows!(shared_q8_f32_preserves_single_row_bits, F32, 8);
shared_rows!(shared_q4_f16_preserves_single_row_bits, F16, 4);
shared_rows!(shared_q8_f16_preserves_single_row_bits, F16, 8);
shared_rows!(shared_q4_bf16_preserves_single_row_bits, BF16, 4);
shared_rows!(shared_q8_bf16_preserves_single_row_bits, BF16, 8);

fn assert_shared_rows(
    backend: &MetalBackend,
    dtype: DType,
    bits: u8,
    inner: u32,
    columns: u32,
    parameters: DType,
) {
    for rows in [1, 2, 4, 7, 8] {
        let mut specs = [
            TensorSpec::contiguous(dtype, &[rows, inner]),
            TensorSpec::contiguous(DType::U32, &[columns, inner * u32::from(bits) / 32]),
            TensorSpec::contiguous(parameters, &[columns, inner / 64]),
            TensorSpec::contiguous(parameters, &[columns, inner / 64]),
        ];
        if rows == 7 {
            for spec in &mut specs {
                let [height, width] = <[u32; 2]>::try_from(spec.allocation_shape()).unwrap();
                *spec = TensorSpec::sliced(
                    spec.dtype(),
                    &[height, width * 2 + 1],
                    &[
                        Slice::new(0, height, 1).unwrap(),
                        Slice::new(1, width, 2).unwrap(),
                    ],
                );
            }
        }
        let mut values = DeterministicValues::new(0xa54f_f53a_5f1d_36f1);
        let bytes = specs
            .iter()
            .map(|spec| generated_tensor_bytes(spec, &mut values).unwrap())
            .collect::<Vec<_>>();
        let inputs = allocate_inputs(backend, &specs, &bytes);
        let output = backend.alloc(dtype, &[rows, columns]).unwrap();
        let op = Op::QuantMatmul {
            bits,
            group_size: 64,
        };
        let variant = backend
            .validate_variant_for_op(
                op,
                &inputs.iter().collect::<Vec<_>>(),
                &[&output],
                "quant-matmul.vector8-shared",
            )
            .unwrap();
        submit_quant(
            backend,
            op,
            &inputs,
            &output,
            Some(BackendDispatchData::new(variant)),
        );
        let actual = backend.read(&output).unwrap();
        let mut expected = Vec::new();
        for row in 0..rows {
            let mut single = inputs.clone();
            single[0] = backend
                .view(
                    &inputs[0],
                    ViewOp::Slice(vec![
                        Slice::new(row, 1, 1).unwrap(),
                        Slice::new(0, inner, 1).unwrap(),
                    ]),
                )
                .unwrap();
            let result = backend.alloc(dtype, &[1, columns]).unwrap();
            let variant = backend
                .validate_variant_for_op(
                    op,
                    &single.iter().collect::<Vec<_>>(),
                    &[&result],
                    "quant-matmul.vector8-2g",
                )
                .unwrap();
            submit_quant(
                backend,
                op,
                &single,
                &result,
                Some(BackendDispatchData::new(variant)),
            );
            expected.extend(backend.read(&result).unwrap());
        }
        assert!(
            actual == expected,
            "rows={rows} inner={inner} columns={columns} bits={bits} dtype={dtype:?}; first mismatch={:?}",
            actual.iter().zip(&expected).position(|(a, b)| a != b)
        );
    }
}
