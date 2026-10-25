//! Measures a dependent chain with unprofiled, Relaxed, and Precise timestamps.

use std::time::{Duration, Instant};

use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_metal::{MetalBackend, ProfileTimestampGranularity};

const DISPATCHES: usize = 512;
const SAMPLES: usize = 5;

#[derive(Clone, Copy)]
enum Mode {
    Unprofiled,
    Relaxed,
    Precise,
}

struct Sample {
    wall: Duration,
    gpu: Duration,
    dispatch_gpu: Option<Duration>,
}

fn commands(backend: &MetalBackend) -> CommandList {
    let first = backend.alloc(DType::F32, &[1]).unwrap();
    let second = backend.alloc(DType::F32, &[1]).unwrap();
    let mut commands = CommandList::new();
    for index in 0..DISPATCHES {
        let (input, output) = if index % 2 == 0 {
            (&first, &second)
        } else {
            (&second, &first)
        };
        commands.dispatch(Op::Copy, &[input], output).unwrap();
    }
    commands
}

fn run(backend: &MetalBackend, mode: Mode) -> Sample {
    let commands = commands(backend);
    let started = Instant::now();
    let submission = match mode {
        Mode::Unprofiled => backend.submit(commands).unwrap(),
        Mode::Relaxed => backend
            .submit_profiled_with_granularity(commands, ProfileTimestampGranularity::Relaxed)
            .unwrap(),
        Mode::Precise => backend
            .submit_profiled_with_granularity(commands, ProfileTimestampGranularity::Precise)
            .unwrap(),
    };
    submission.wait().unwrap();
    let wall = started.elapsed();
    let gpu = submission.gpu_time().unwrap();
    let dispatch_gpu = submission.profile().map(|profile| {
        profile
            .per_dispatch
            .iter()
            .map(|dispatch| dispatch.gpu_time)
            .sum()
    });
    Sample {
        wall,
        gpu,
        dispatch_gpu,
    }
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn duration_values(samples: &[Sample], field: impl Fn(&Sample) -> Duration) -> Vec<u128> {
    samples
        .iter()
        .map(|sample| field(sample).as_nanos())
        .collect()
}

fn print_values(label: &str, values: &[u128]) {
    let values = values
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  {label}: {values}");
}

fn print_mode(name: &str, samples: &[Sample]) {
    let wall = duration_values(samples, |sample| sample.wall);
    let gpu = duration_values(samples, |sample| sample.gpu);
    let dispatch_gpu = samples
        .iter()
        .filter_map(|sample| sample.dispatch_gpu)
        .map(|duration| duration.as_nanos())
        .collect::<Vec<_>>();
    let median_wall = median(samples.iter().map(|sample| sample.wall).collect());
    let median_gpu = median(samples.iter().map(|sample| sample.gpu).collect());
    let median_dispatch_gpu = (!dispatch_gpu.is_empty()).then(|| {
        median(
            samples
                .iter()
                .filter_map(|sample| sample.dispatch_gpu)
                .collect(),
        )
    });
    println!("{name}:");
    print_values("wall ns", &wall);
    print_values("GPU ns", &gpu);
    print_values("summed dispatch interval ns", &dispatch_gpu);
    println!("  median wall ns: {}", median_wall.as_nanos());
    println!("  median GPU ns: {}", median_gpu.as_nanos());
    if let Some(value) = median_dispatch_gpu {
        println!("  median summed dispatch interval ns: {}", value.as_nanos());
    }
}

fn main() {
    let backend = MetalBackend::new().unwrap();
    for mode in [Mode::Unprofiled, Mode::Relaxed, Mode::Precise] {
        run(&backend, mode);
    }
    let mut unprofiled = Vec::with_capacity(SAMPLES);
    let mut relaxed = Vec::with_capacity(SAMPLES);
    let mut precise = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        unprofiled.push(run(&backend, Mode::Unprofiled));
        relaxed.push(run(&backend, Mode::Relaxed));
        precise.push(run(&backend, Mode::Precise));
    }
    println!("dependent dispatches: {DISPATCHES}");
    println!("samples: {SAMPLES}");
    print_mode("unprofiled", &unprofiled);
    print_mode("relaxed", &relaxed);
    print_mode("precise", &precise);
}
