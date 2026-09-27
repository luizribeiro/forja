//! Measures decode matrix-vector throughput for Qwen3-0.6B projection shapes.

use std::time::{Duration, Instant};

use forja_core::{Backend, CommandList, DType, Op, Submission, ViewOp};
use forja_metal::MetalBackend;

const WARM_UP: usize = 5;
const SAMPLES: usize = 30;
const OPERATIONS_PER_SAMPLE: u32 = 30;
const PEAK_GB_PER_SECOND: f64 = 819.2;

struct Shape {
    name: &'static str,
    inner: u32,
    columns: u32,
}

struct Timing {
    gpu: Duration,
    wall: Duration,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "q",
        inner: 1024,
        columns: 2048,
    },
    Shape {
        name: "k/v",
        inner: 1024,
        columns: 1024,
    },
    Shape {
        name: "o",
        inner: 2048,
        columns: 1024,
    },
    Shape {
        name: "gate/up",
        inner: 1024,
        columns: 3072,
    },
    Shape {
        name: "down",
        inner: 3072,
        columns: 1024,
    },
    Shape {
        name: "logits",
        inner: 1024,
        columns: 151_936,
    },
];

fn run(
    backend: &MetalBackend,
    left: &forja_core::Tensor,
    rights: &[forja_core::Tensor],
    outputs: &[forja_core::Tensor],
) -> Timing {
    let mut commands = CommandList::new();
    for (right, output) in rights.iter().zip(outputs) {
        commands
            .dispatch(Op::Matmul, &[left, right], output)
            .unwrap();
    }
    let started = Instant::now();
    let submission = backend.submit(commands).unwrap();
    submission.wait().unwrap();
    let wall = started.elapsed() / OPERATIONS_PER_SAMPLE;
    Timing {
        gpu: submission.gpu_time().unwrap() / OPERATIONS_PER_SAMPLE,
        wall,
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[SAMPLES / 2]
}

fn main() {
    let backend = MetalBackend::new().unwrap();
    println!("shape\tForja GPU ms\tForja wall ms\tGPU GB/s\twall GB/s\tGPU peak\twall peak");
    for shape in SHAPES {
        let left = backend.alloc(DType::BF16, &[1, shape.inner]).unwrap();
        let weights = (0..OPERATIONS_PER_SAMPLE)
            .map(|_| {
                backend
                    .alloc(DType::BF16, &[shape.columns, shape.inner])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let rights = weights
            .iter()
            .map(|weight| backend.view(weight, ViewOp::Permute(vec![1, 0])).unwrap())
            .collect::<Vec<_>>();
        let outputs = (0..OPERATIONS_PER_SAMPLE)
            .map(|_| backend.alloc(DType::BF16, &[1, shape.columns]).unwrap())
            .collect::<Vec<_>>();
        for _ in 0..WARM_UP {
            run(&backend, &left, &rights, &outputs);
        }
        let samples = (0..SAMPLES)
            .map(|_| run(&backend, &left, &rights, &outputs))
            .collect::<Vec<_>>();
        let gpu = median(samples.iter().map(|timing| timing.gpu).collect());
        let wall = median(samples.iter().map(|timing| timing.wall).collect());
        let elements = f64::from(shape.inner)
            + f64::from(shape.inner) * f64::from(shape.columns)
            + f64::from(shape.columns);
        let gpu_throughput = elements * 2.0 / gpu.as_secs_f64() / 1e9;
        let wall_throughput = elements * 2.0 / wall.as_secs_f64() / 1e9;
        println!(
            "{}\t{:.3}\t{:.3}\t{gpu_throughput:.1}\t{wall_throughput:.1}\t{:.1}%\t{:.1}%",
            shape.name,
            gpu.as_secs_f64() * 1e3,
            wall.as_secs_f64() * 1e3,
            gpu_throughput / PEAK_GB_PER_SECOND * 100.0,
            wall_throughput / PEAK_GB_PER_SECOND * 100.0,
        );
        backend.release(&left).unwrap();
        for weight in weights {
            backend.release(&weight).unwrap();
        }
        for output in outputs {
            backend.release(&output).unwrap();
        }
    }
}
