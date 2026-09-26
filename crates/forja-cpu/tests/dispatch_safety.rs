//! Property tests for the validated-operation boundary.

use forja_cpu::CpuBackend;
use proptest::prelude::*;

#[path = "../../../support/forja-testing/test-support/dispatch_safety.rs"]
mod common;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn accepted_dispatches_never_panic(
        case in prop_oneof![4 => common::valid_case(), 1 => common::arbitrary_case()],
    ) {
        let _result = common::run(&CpuBackend::new(), &case);
    }
}
