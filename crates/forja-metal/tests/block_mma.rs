//! Bitwise regression for both dense operand-fragment layouts.

use super::*;
use forja_core::{BackendDispatchData, Op, Submission};

fn dense_result(source: &str) -> Vec<u8> {
    let backend = MetalBackend::new().unwrap();
    *backend.pipelines.lock().unwrap() = PipelineCache::new(&backend.device, source).unwrap();
    let input = backend.alloc(DType::F32, &[64, 96]).unwrap();
    let weights = backend.alloc(DType::F32, &[96, 32]).unwrap();
    for (tensor, modulus) in [(&input, 251_u16), (&weights, 127)] {
        let bytes = (0..tensor.layout().element_count())
            .flat_map(|i| {
                let value = f32::from(u16::try_from(i % u64::from(modulus)).unwrap())
                    / f32::from(modulus)
                    - 0.5;
                value.to_le_bytes()
            })
            .collect::<Vec<_>>();
        backend.write(tensor, &bytes).unwrap();
    }
    let output = backend.alloc(DType::F32, &[64, 32]).unwrap();
    let inputs = [&input, &weights];
    let variant = backend
        .validate_variant_for_op(Op::Matmul, &inputs, &[&output], "matmul.steel-64x32x32-2x2")
        .unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Matmul, &inputs, &output).unwrap();
    assert!(commands.set_last_backend_data(BackendDispatchData::new(variant)));
    backend.submit(commands).unwrap().wait().unwrap();
    backend.read(&output).unwrap()
}

#[test]
fn metal_block_mma_preserves_dense_bits() {
    let production = shader_source();
    let start = production
        .find("    void multiply(\n        threadgroup const T *a,")
        .unwrap();
    let end = start + production[start..].find("    void store(").unwrap();
    let mut reference = production.to_owned();
    reference.replace_range(
        start..end,
        include_str!("fixtures/block_mma_reference.metal"),
    );
    let entry = "DEFINE_STEEL_GEMM(steel_gemm_64_32_32_2_2, 64, 32, 32, 2, 2)";
    let mut results = Vec::new();
    for transposed in [false, true] {
        let kernel = include_str!("fixtures/block_mma_dense.metal")
            .replace("TEST_TRANSPOSED", if transposed { "true" } else { "false" });
        let run = |source: &str| dense_result(&format!("{}\n{kernel}", source.replace(entry, "")));
        let expected = run(&reference);
        let actual = run(production);
        assert_eq!(actual, expected, "right_transposed={transposed}");
        results.push(actual);
    }
    assert_eq!(results[0], results[1]);
}
