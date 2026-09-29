#!/usr/bin/env python3
"""Measure synchronized MLX work for Qwen3-0.6B decode matmul classes."""

import argparse
import json
import statistics
import time
from pathlib import Path

WARMUPS = 5
REPETITIONS = 30
OPERATIONS_PER_SAMPLE = 30
SHAPES = [
    ("q", 1024, 2048, 28),
    ("k/v", 1024, 1024, 56),
    ("o", 2048, 1024, 28),
    ("gate/up", 1024, 3072, 56),
    ("down", 3072, 1024, 28),
    ("logits", 1024, 151_936, 1),
]


def arguments() -> argparse.Namespace:
    """Parse output options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def run(mx, left, weights: list) -> float:
    """Return synchronized wall time with queue latency amortized."""
    outputs = [mx.matmul(left, weight.T) for weight in weights]
    started = time.perf_counter()
    mx.eval(*outputs)
    mx.synchronize()
    return (time.perf_counter() - started) / OPERATIONS_PER_SAMPLE


def profile_shapes() -> dict[str, object]:
    """Return synchronized timings for every production decode matmul class."""
    import mlx.core as mx

    shapes = []
    for name, inner, columns, count in SHAPES:
        left = mx.zeros((1, inner), dtype=mx.bfloat16)
        weights = [
            mx.zeros((columns, inner), dtype=mx.bfloat16)
            for _ in range(OPERATIONS_PER_SAMPLE)
        ]
        mx.eval(left, *weights)
        for _ in range(WARMUPS):
            run(mx, left, weights)
        elapsed = statistics.median(
            run(mx, left, weights) for _ in range(REPETITIONS)
        )
        elements = inner + inner * columns + columns
        throughput = elements * 2 / elapsed / 1e9
        shapes.append(
            {
                "name": name,
                "shape": [inner, columns],
                "dispatches_per_token": count,
                "synchronized_seconds_per_dispatch": elapsed,
                "gigabytes_per_second": throughput,
                "estimated_seconds_per_token": elapsed * count,
            }
        )
    return {
        "gpu_time_seconds": None,
        "gpu_time_note": (
            "MLX 0.32 exposes Metal capture but no device-timestamp API; its binary "
            ".gputrace is not reliably machine-readable here. Timings synchronize the "
            "MLX stream and therefore include queue and synchronization overhead."
        ),
        "shape_classes": shapes,
        "estimated_matmul_seconds_per_token": sum(
            shape["estimated_seconds_per_token"] for shape in shapes
        ),
    }


def main() -> None:
    """Print and optionally record the synchronized shape-class comparison."""
    args = arguments()
    report = profile_shapes()
    print("shape\tdispatches/token\tMLX synchronized ms\tMLX synchronized GB/s")
    for shape in report["shape_classes"]:
        print(
            f"{shape['name']}\t{shape['dispatches_per_token']}\t"
            f"{shape['synchronized_seconds_per_dispatch'] * 1e3:.3f}\t"
            f"{shape['gigabytes_per_second']:.1f}"
        )
    print(
        "estimated matmul total\t"
        f"{report['estimated_matmul_seconds_per_token'] * 1e3:.3f} ms/token"
    )
    print(report["gpu_time_note"])
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
