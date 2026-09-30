//! Measures decode affine-quantized matrix-vector throughput at model shapes.

use std::time::{Duration, Instant};

use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_metal::MetalBackend;

const COPY_BYTES: u64 = 512 * 1024 * 1024;
const WARM_UP: usize = 5;
const SAMPLES: usize = 30;
const OPERATIONS_PER_SAMPLE: u32 = 20;

struct Shape {
    name: &'static str,
    inner: u32,
    columns: u32,
    bits: u8,
}

struct Timing {
    gpu: Duration,
    wall: Duration,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "q",
        inner: 2048,
        columns: 4096,
        bits: 4,
    },
    Shape {
        name: "k/v",
        inner: 2048,
        columns: 512,
        bits: 4,
    },
    Shape {
        name: "o",
        inner: 4096,
        columns: 2048,
        bits: 4,
    },
    Shape {
        name: "expert gate/up",
        inner: 2048,
        columns: 768,
        bits: 4,
    },
    Shape {
        name: "expert down",
        inner: 768,
        columns: 2048,
        bits: 4,
    },
    Shape {
        name: "router",
        inner: 2048,
        columns: 128,
        bits: 8,
    },
];

fn commands(
    input: &forja_core::Tensor,
    weights: &[[forja_core::Tensor; 3]],
    outputs: &[forja_core::Tensor],
    bits: u8,
) -> CommandList {
    let mut commands = CommandList::new();
    for ([packed, scales, biases], output) in weights.iter().zip(outputs) {
        commands
            .dispatch(
                Op::QuantMatmul {
                    bits,
                    group_size: 64,
                },
                &[input, packed, scales, biases],
                output,
            )
            .unwrap();
    }
    commands
}

fn run(
    backend: &MetalBackend,
    input: &forja_core::Tensor,
    weights: &[[forja_core::Tensor; 3]],
    outputs: &[forja_core::Tensor],
    bits: u8,
) -> Timing {
    let commands = commands(input, weights, outputs, bits);
    let started = Instant::now();
    let submission = backend.submit(commands).unwrap();
    submission.wait().unwrap();
    Timing {
        gpu: submission.gpu_time().unwrap() / OPERATIONS_PER_SAMPLE,
        wall: started.elapsed() / OPERATIONS_PER_SAMPLE,
    }
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[SAMPLES / 2]
}

#[allow(clippy::cast_precision_loss)]
fn copy_peak(backend: &MetalBackend) -> f64 {
    let elements = u32::try_from(COPY_BYTES / DType::U32.byte_size()).unwrap();
    let input = backend.alloc(DType::U32, &[elements]).unwrap();
    let output = backend.alloc(DType::U32, &[elements]).unwrap();
    let measure = || {
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        let submission = backend.submit(commands).unwrap();
        submission.wait().unwrap();
        submission.gpu_time().unwrap()
    };
    for _ in 0..WARM_UP {
        measure();
    }
    let elapsed = median((0..SAMPLES).map(|_| measure()).collect());
    backend.release(&input).unwrap();
    backend.release(&output).unwrap();
    COPY_BYTES as f64 * 2.0 / elapsed.as_secs_f64() / 1e9
}

fn main() {
    let backend = MetalBackend::new().unwrap();
    let peak = copy_peak(&backend);
    println!("copy peak\t{peak:.1} GB/s");
    println!("shape\tGPU ms\twall ms\tGPU GB/s\twall GB/s\tcopy peak");
    for shape in SHAPES {
        let packed_width = shape.inner * u32::from(shape.bits) / 32;
        let groups = shape.inner / 64;
        let input = backend.alloc(DType::BF16, &[1, shape.inner]).unwrap();
        let weights = (0..OPERATIONS_PER_SAMPLE)
            .map(|_| {
                [
                    backend
                        .alloc(DType::U32, &[shape.columns, packed_width])
                        .unwrap(),
                    backend
                        .alloc(DType::BF16, &[shape.columns, groups])
                        .unwrap(),
                    backend
                        .alloc(DType::BF16, &[shape.columns, groups])
                        .unwrap(),
                ]
            })
            .collect::<Vec<_>>();
        let outputs = (0..OPERATIONS_PER_SAMPLE)
            .map(|_| backend.alloc(DType::BF16, &[1, shape.columns]).unwrap())
            .collect::<Vec<_>>();
        for _ in 0..WARM_UP {
            run(&backend, &input, &weights, &outputs, shape.bits);
        }
        let timings = (0..SAMPLES)
            .map(|_| run(&backend, &input, &weights, &outputs, shape.bits))
            .collect::<Vec<_>>();
        let gpu = median(timings.iter().map(|timing| timing.gpu).collect());
        let wall = median(timings.iter().map(|timing| timing.wall).collect());
        let bytes = f64::from(shape.inner) * 2.0
            + f64::from(shape.columns * packed_width) * 4.0
            + f64::from(shape.columns * groups) * 4.0
            + f64::from(shape.columns) * 2.0;
        let gpu_throughput = bytes / gpu.as_secs_f64() / 1e9;
        let wall_throughput = bytes / wall.as_secs_f64() / 1e9;
        println!(
            "{}\t{:.3}\t{:.3}\t{gpu_throughput:.1}\t{wall_throughput:.1}\t{:.1}%",
            shape.name,
            gpu.as_secs_f64() * 1e3,
            wall.as_secs_f64() * 1e3,
            gpu_throughput / peak * 100.0,
        );
        backend.release(&input).unwrap();
        for tensors in weights {
            for tensor in tensors {
                backend.release(&tensor).unwrap();
            }
        }
        for output in outputs {
            backend.release(&output).unwrap();
        }
    }
}
