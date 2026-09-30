#!/usr/bin/env python3
"""Emit MLX outputs for fixed real-weight affine quantized matmuls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

CASES = [
    ("model.layers.0.self_attn.q_proj", 4),
    ("model.layers.0.mlp.gate", 8),
]


def fixed_input() -> mx.array:
    """Return the deterministic activation shared with the Forja test."""
    values = [(index % 31) / 16.0 - 1.0 for index in range(2048)]
    return mx.array(values, dtype=mx.float32).reshape(1, 2048)


def outputs(model: Path) -> list[mx.array]:
    """Evaluate the real q4 projection and q8 router weights."""
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = {}
    values = []
    activation = fixed_input()
    for prefix, bits in CASES:
        names = [f"{prefix}.{suffix}" for suffix in ("weight", "scales", "biases")]
        shard_names = {weight_map[name] for name in names}
        if len(shard_names) != 1:
            raise ValueError(f"{prefix} tensors span multiple shards")
        shard_name = shard_names.pop()
        weights = shards.setdefault(shard_name, mx.load(str(model / shard_name)))
        values.append(
            mx.quantized_matmul(
                activation,
                weights[names[0]],
                scales=weights[names[1]],
                biases=weights[names[2]],
                transpose=True,
                group_size=64,
                bits=bits,
            )
        )
    mx.eval(*values)
    return values


def main() -> None:
    """Write little-endian float32 output bytes to stdout."""
    model = Path(sys.argv[1])
    for value in outputs(model):
        sys.stdout.buffer.write(np.asarray(value, dtype="<f4").tobytes())


if __name__ == "__main__":
    main()
