//! Measures end-to-end latency for representative Metal submission patterns.

use std::time::{Duration, Instant};

use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_metal::MetalBackend;

const WARM_UP: usize = 20;
const SAMPLES: usize = 200;

fn percentiles(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort_unstable();
    (samples[SAMPLES / 2], samples[SAMPLES * 9 / 10 - 1])
}

fn measure(mut run: impl FnMut()) -> (Duration, Duration) {
    for _ in 0..WARM_UP {
        run();
    }
    let samples = (0..SAMPLES)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect::<Vec<_>>();
    percentiles(samples)
}

fn print_result(name: &str, result: (Duration, Duration)) {
    println!(
        "{name}: median {:.3} us, p90 {:.3} us",
        result.0.as_secs_f64() * 1e6,
        result.1.as_secs_f64() * 1e6,
    );
}

fn print_completion_timing(backend: &MetalBackend) {
    for _ in 0..WARM_UP {
        backend.submit(CommandList::new()).unwrap().wait().unwrap();
    }
    let mut feedback = Vec::with_capacity(SAMPLES);
    let mut event = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let submission = backend.submit(CommandList::new()).unwrap();
        submission.wait().unwrap();
        let timing = submission.completion_timing().unwrap();
        feedback.push(timing.0);
        event.push(timing.1);
    }
    print_result("commit to feedback callback", percentiles(feedback));
    print_result("commit to event callback", percentiles(event));
}

fn main() {
    let backend = MetalBackend::new().unwrap();
    let gate = backend.alloc(DType::F32, &[1]).unwrap();
    let up = backend.alloc(DType::F32, &[1]).unwrap();
    let output = backend.alloc(DType::F32, &[1]).unwrap();

    print_result(
        "empty",
        measure(|| backend.submit(CommandList::new()).unwrap().wait().unwrap()),
    );
    print_result(
        "tiny SiluMul",
        measure(|| {
            let mut commands = CommandList::new();
            commands
                .dispatch(Op::SiluMul, &[&gate, &up], &output)
                .unwrap();
            backend.submit(commands).unwrap().wait().unwrap();
        }),
    );
    print_result(
        "8 submissions",
        measure(|| {
            let submissions = (0..8)
                .map(|_| backend.submit(CommandList::new()).unwrap())
                .collect::<Vec<_>>();
            for submission in submissions {
                submission.wait().unwrap();
            }
        }),
    );
    print_completion_timing(&backend);
}
