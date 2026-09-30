#!/usr/bin/env python3
"""Evaluate or benchmark the real Qwen layer-zero sparse MoE block."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HIDDEN = 2048
TOP_K = 8


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


def main() -> None:
    """Run the reference operation for a local model directory."""
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} MODEL")
    reference(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
