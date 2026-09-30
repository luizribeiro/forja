#!/usr/bin/env python3
"""Measure synchronized MLX affine-quantized decode matrix multiplication."""

import statistics
import time

WARMUPS = 5
REPETITIONS = 30
OPERATIONS_PER_SAMPLE = 20
SHAPES = [
    ("q", 2048, 4096, 4),
    ("k/v", 2048, 512, 4),
    ("o", 4096, 2048, 4),
    ("expert gate/up", 2048, 768, 4),
    ("expert down", 768, 2048, 4),
    ("router", 2048, 128, 8),
]


def run(mx, inputs: list[tuple]) -> float:
    """Return synchronized wall time with queue latency amortized."""
    outputs = [
        mx.quantized_matmul(
            left,
            packed,
            scales=scales,
            biases=biases,
            transpose=True,
            group_size=64,
            bits=bits,
        )
        for left, packed, scales, biases, bits in inputs
    ]
    started = time.perf_counter()
    mx.eval(*outputs)
    mx.synchronize()
    return (time.perf_counter() - started) / OPERATIONS_PER_SAMPLE


def profile_shapes() -> list[dict[str, object]]:
    """Return synchronized timings for target decode shapes."""
    import mlx.core as mx

    results = []
    for name, inner, columns, bits in SHAPES:
        packed_width = inner * bits // 32
        groups = inner // 64
        inputs = [
            (
                mx.zeros((1, inner), dtype=mx.bfloat16),
                mx.zeros((columns, packed_width), dtype=mx.uint32),
                mx.ones((columns, groups), dtype=mx.bfloat16),
                mx.zeros((columns, groups), dtype=mx.bfloat16),
                bits,
            )
            for _ in range(OPERATIONS_PER_SAMPLE)
        ]
        mx.eval(*[value for operation in inputs for value in operation[:4]])
        for _ in range(WARMUPS):
            run(mx, inputs)
        elapsed = statistics.median(run(mx, inputs) for _ in range(REPETITIONS))
        byte_count = (
            inner * 2
            + columns * packed_width * 4
            + columns * groups * 4
            + columns * 2
        )
        results.append(
            {
                "name": name,
                "shape": [inner, columns],
                "bits": bits,
                "synchronized_seconds": elapsed,
                "gigabytes_per_second": byte_count / elapsed / 1e9,
            }
        )
    return results


def main() -> None:
    """Print the synchronized shape comparison."""
    print("shape\tMLX synchronized ms\tlogical GB/s")
    for result in profile_shapes():
        print(
            f"{result['name']}\t{result['synchronized_seconds'] * 1e3:.3f}\t"
            f"{result['gigabytes_per_second']:.1f}"
        )


if __name__ == "__main__":
    main()
