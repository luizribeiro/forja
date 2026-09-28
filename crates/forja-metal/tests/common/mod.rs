use std::time::Duration;

use forja_core::{Backend, CommandList, Submission};
use forja_metal::MetalBackend;
use forja_testing::IntervalReport;

pub fn report_intervals(report: IntervalReport) {
    eprintln!(
        "interval-oracle ambiguous={} selects={} reductions={} vacuous={}/{} width-median={:.3e} width-p90={:.3e} width-max={:.3e}",
        report.ambiguous_predicates,
        report.ambiguous_selects,
        report.reduction_steps,
        report.vacuous_elements,
        report.total_elements,
        report.median_relative_width,
        report.p90_relative_width,
        report.max_relative_width,
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
