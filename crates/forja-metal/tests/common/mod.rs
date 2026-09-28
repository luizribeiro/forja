use std::time::Duration;

use forja_core::{Backend, CommandList, Submission};
use forja_metal::MetalBackend;
use forja_testing::PredicateReport;

pub fn report_predicates(report: PredicateReport) {
    eprintln!(
        "predicate-oracle ambiguous={} alternate={} excluded={} total={}",
        report.ambiguous_predicates,
        report.alternate_elements,
        report.excluded_elements,
        report.total_elements,
    );
}

pub fn median_gpu_time(
    backend: &MetalBackend,
    mut commands: impl FnMut() -> CommandList,
) -> Duration {
    for _ in 0..5 {
        backend.submit_profiled(commands()).unwrap().wait().unwrap();
    }
    let mut samples = (0..31)
        .map(|_| {
            let submission = backend.submit_profiled(commands()).unwrap();
            submission.wait().unwrap();
            submission.gpu_time().unwrap()
        })
        .collect::<Vec<_>>();
    samples.sort_unstable();
    samples[samples.len() / 2]
}
