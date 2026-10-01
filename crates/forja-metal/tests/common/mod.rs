use std::time::Duration;

use forja_core::{Backend, CommandList, Submission};
use forja_metal::MetalBackend;
use forja_testing::IntervalReport;

pub const GROSS_REGRESSION_RATIO: f64 = 1.5;

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

pub fn comparative_gpu_times(
    backend: &MetalBackend,
    mut trusted: impl FnMut() -> CommandList,
    mut candidate: impl FnMut() -> CommandList,
) -> (Duration, Duration) {
    for iteration in 0..5 {
        if iteration % 2 == 0 {
            gpu_time(backend, trusted());
            gpu_time(backend, candidate());
        } else {
            gpu_time(backend, candidate());
            gpu_time(backend, trusted());
        }
    }

    let mut trusted_samples = Vec::with_capacity(64);
    let mut candidate_samples = Vec::with_capacity(64);
    for iteration in 0..64 {
        let (trusted_time, candidate_time) = if iteration % 2 == 0 {
            (gpu_time(backend, trusted()), gpu_time(backend, candidate()))
        } else {
            let candidate_time = gpu_time(backend, candidate());
            (gpu_time(backend, trusted()), candidate_time)
        };
        trusted_samples.push(trusted_time);
        candidate_samples.push(candidate_time);
    }

    (
        low_percentile(trusted_samples),
        low_percentile(candidate_samples),
    )
}

fn gpu_time(backend: &MetalBackend, commands: CommandList) -> Duration {
    let submission = backend.submit_profiled(commands).unwrap();
    submission.wait().unwrap();
    submission.gpu_time().unwrap()
}

fn low_percentile(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 10]
}
