#!/usr/bin/env python3
"""Measure MLX decode matrix-vector throughput for Qwen3-0.6B shapes."""

import statistics
import time

import mlx.core as mx

WARMUPS = 5
REPETITIONS = 30
OPERATIONS_PER_SAMPLE = 30
PEAK_GB_PER_SECOND = 819.2
SHAPES = [
    ("q", 1024, 2048),
    ("k/v", 1024, 1024),
    ("o", 2048, 1024),
    ("gate/up", 1024, 3072),
    ("down", 3072, 1024),
    ("logits", 1024, 151_936),
]


def run(left: mx.array, weights: list[mx.array]) -> float:
    """Return synchronized wall time with queue latency amortized."""
    outputs = [mx.matmul(left, weight.T) for weight in weights]
    started = time.perf_counter()
    mx.eval(*outputs)
    mx.synchronize()
    return (time.perf_counter() - started) / OPERATIONS_PER_SAMPLE


def main() -> None:
    """Print median throughput for each production decode shape."""
    print("shape\tMLX wall ms\tMLX wall GB/s\tMLX wall peak")
    for name, inner, columns in SHAPES:
        left = mx.zeros((1, inner), dtype=mx.bfloat16)
        weights = [
            mx.zeros((columns, inner), dtype=mx.bfloat16)
            for _ in range(OPERATIONS_PER_SAMPLE)
        ]
        mx.eval(left, *weights)
        for _ in range(WARMUPS):
            run(left, weights)
        elapsed = statistics.median(
            run(left, weights) for _ in range(REPETITIONS)
        )
        elements = inner + inner * columns + columns
        throughput = elements * 2 / elapsed / 1e9
        print(
            f"{name}\t{elapsed * 1e3:.3f}\t{throughput:.1f}\t"
            f"{throughput / PEAK_GB_PER_SECOND * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
