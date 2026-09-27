//! Property tests for Metal's validated-operation boundary.

use forja_core::Op;
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::assert_outputs_agree;
use proptest::prelude::*;

#[path = "../../../support/forja-testing/test-support/dispatch_safety.rs"]
mod common;

fn supported(op: Op) -> bool {
    matches!(
        op,
        Op::Copy
            | Op::Add
            | Op::SiluMul
            | Op::RmsNorm { .. }
            | Op::Softmax
            | Op::Rope { .. }
            | Op::Embed
            | Op::Matmul
            | Op::Sdpa { .. }
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn accepted_dispatches_are_safe_and_supported_ops_agree(
        case in common::valid_case(),
    ) {
        let reference = common::run(&CpuBackend::new(), &case).unwrap();
        let candidate = common::run(&MetalBackend::new().unwrap(), &case).unwrap();
        if supported(case.op) {
            prop_assert_eq!(candidate.result, reference.result);
            let expected = reference.output.unwrap();
            let actual = candidate.output.unwrap();
            prop_assert!(
                assert_outputs_agree(reference.dtype, &expected, &actual).is_ok(),
                "Metal output differed for {:?}",
                case.op,
            );
        }
    }
}
