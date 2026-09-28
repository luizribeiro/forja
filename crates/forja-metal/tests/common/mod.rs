use std::time::Duration;

use forja_core::{Backend, CommandList, Submission};
use forja_metal::MetalBackend;

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
