#!/usr/bin/env python3
"""Evaluate or benchmark the real Qwen layer-zero sparse MoE block."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HIDDEN = 2048
TOP_K = 8
WARMUPS = 5
REPETITIONS = 30
BLOCKS_PER_SAMPLE = 20


def fixed_input() -> mx.array:
    """Return the deterministic activation shared with the Forja test."""
    values = [(index % 31) / 16.0 - 1.0 for index in range(HIDDEN)]
    return mx.array(values, dtype=mx.bfloat16).reshape(1, 1, HIDDEN)


def load_block(model_path: Path):
    """Load the model lazily and return its first sparse MoE block."""
    model, _ = load(str(model_path), lazy=True)
    return model.model.layers[0].mlp


def reference(model_path: Path) -> None:
    """Write sorted expert indices and float32 block output to stdout."""
    block = load_block(model_path)
    activation = fixed_input()
    logits = block.gate(activation)
    probabilities = mx.softmax(logits, axis=-1, precise=True)
    indices = mx.argsort(probabilities, axis=-1)[..., -TOP_K:][..., ::-1]
    output = block(activation)
    mx.eval(indices, output)
    sys.stdout.buffer.write(np.asarray(indices, dtype="<u4").tobytes())
    sys.stdout.buffer.write(
        np.asarray(output.astype(mx.float32), dtype="<f4").tobytes()
    )


def run(block, activation: mx.array) -> float:
    """Return synchronized wall time with queue latency amortized."""
    outputs = [block(activation) for _ in range(BLOCKS_PER_SAMPLE)]
    started = time.perf_counter()
    mx.eval(*outputs)
    mx.synchronize()
    return (time.perf_counter() - started) / BLOCKS_PER_SAMPLE


def benchmark(model_path: Path) -> None:
    """Print synchronized median microseconds per block."""
    block = load_block(model_path)
    activation = fixed_input()
    for _ in range(WARMUPS):
        run(block, activation)
    elapsed = statistics.median(
        run(block, activation) for _ in range(REPETITIONS)
    )
    print(f"{elapsed * 1_000_000.0:.3f}")


def main() -> None:
    """Run the selected operation for a local model directory."""
    if len(sys.argv) != 3 or sys.argv[1] not in {"reference", "benchmark"}:
        raise SystemExit(f"usage: {sys.argv[0]} reference|benchmark MODEL")
    operation = sys.argv[1]
    model_path = Path(sys.argv[2])
    if operation == "reference":
        reference(model_path)
    else:
        benchmark(model_path)


if __name__ == "__main__":
    main()
