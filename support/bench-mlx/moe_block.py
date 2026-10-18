#!/usr/bin/env python3
"""Evaluate or benchmark real Qwen layer-zero decode blocks."""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

HIDDEN = 2048
TOP_K = 8
WARMUPS = 5
REPETITIONS = 30
BLOCKS_PER_SAMPLE = 20
QUERY_HEADS = 32
KEY_VALUE_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 768
EXPERTS = 128
VOCAB = 151_936
GROUP_SIZE = 64
CONTEXTS = (512, 4_000)


def fixed_input() -> mx.array:
    """Return the deterministic activation shared with the Forja test."""
    values = [(index % 31) / 16.0 - 1.0 for index in range(HIDDEN)]
    return mx.array(values, dtype=mx.bfloat16).reshape(1, 1, HIDDEN)


def load_block(model_path: Path):
    """Load the model lazily and return its first sparse MoE block."""
    model, _ = load(str(model_path), lazy=True)
    return model.model.layers[0].mlp


def leaves(value) -> list[mx.array]:
    """Flatten array tuples returned by a measured operation."""
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in leaves(child)]
    return [value]


def run_operation(operation: Callable[[], object]) -> float:
    """Return synchronized execution time with queue latency amortized."""
    outputs = [operation() for _ in range(BLOCKS_PER_SAMPLE)]
    started = time.perf_counter()
    mx.eval(*[leaf for output in outputs for leaf in leaves(output)])
    mx.synchronize()
    return (time.perf_counter() - started) / BLOCKS_PER_SAMPLE


def measure(name: str, byte_count: int, operation: Callable[[], object]) -> dict:
    """Measure one operation and report its logical bandwidth."""
    for _ in range(WARMUPS):
        run_operation(operation)
    elapsed = statistics.median(
        run_operation(operation) for _ in range(REPETITIONS)
    )
    return {
        "name": name,
        "synchronized_seconds": elapsed,
        "bytes_moved": byte_count,
        "gigabytes_per_second": byte_count / elapsed / 1e9,
    }


def tensor_bytes(elements: int) -> int:
    """Return bytes for a bf16 tensor."""
    return elements * 2


def norm_bytes(width: int) -> int:
    """Return logical traffic for bf16 RMS normalization."""
    return tensor_bytes(width * 3)


def gather_quantized_bytes(inner: int, output: int, routes: int) -> int:
    """Return logical traffic for selected affine-q4 expert matrices."""
    weights = routes * output * inner // 2
    metadata = routes * output * (inner // GROUP_SIZE) * 4
    activations = tensor_bytes(inner + routes * output)
    indices = routes * 4
    return weights + metadata + activations + indices


def quantized_bytes(inner: int, output: int, bits: int) -> int:
    """Return logical traffic for one affine-quantized matrix-vector product."""
    packed = output * inner * bits // 8
    metadata = output * (inner // GROUP_SIZE) * 4
    return packed + metadata + tensor_bytes(inner + output)


def block_profile(model_path: Path) -> list[dict]:
    """Profile attention, sparse MoE, and output-tail operations."""
    model, _ = load(str(model_path), lazy=True)
    layer = model.model.layers[0]
    attention = layer.self_attn
    block = layer.mlp
    activation = fixed_input()
    mx.eval(activation)

    attention_input = layer.input_layernorm(activation)
    query = attention.q_proj(attention_input).reshape(
        1, 1, QUERY_HEADS, HEAD_DIM
    )
    key = attention.k_proj(attention_input).reshape(
        1, 1, KEY_VALUE_HEADS, HEAD_DIM
    )
    value = attention.v_proj(attention_input).reshape(
        1, 1, KEY_VALUE_HEADS, HEAD_DIM
    )
    query_norm = attention.q_norm(query)
    key_norm = attention.k_norm(key)
    mx.eval(attention_input, query, key, value, query_norm, key_norm)

    results = [
        measure(
            "attention.norm",
            norm_bytes(HIDDEN),
            lambda: layer.input_layernorm(activation),
        ),
        measure(
            "attention.q_projection",
            quantized_bytes(HIDDEN, QUERY_HEADS * HEAD_DIM, 4),
            lambda: attention.q_proj(attention_input),
        ),
        measure(
            "attention.k_projection",
            quantized_bytes(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM, 4),
            lambda: attention.k_proj(attention_input),
        ),
        measure(
            "attention.v_projection",
            quantized_bytes(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM, 4),
            lambda: attention.v_proj(attention_input),
        ),
        measure(
            "attention.q_norm",
            tensor_bytes(QUERY_HEADS * HEAD_DIM * 2 + HEAD_DIM),
            lambda: attention.q_norm(query),
        ),
        measure(
            "attention.k_norm",
            tensor_bytes(KEY_VALUE_HEADS * HEAD_DIM * 2 + HEAD_DIM),
            lambda: attention.k_norm(key),
        ),
        measure(
            "attention.rope",
            tensor_bytes((QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM * 2),
            lambda: (
                attention.rope(query_norm.transpose(0, 2, 1, 3), offset=511),
                attention.rope(key_norm.transpose(0, 2, 1, 3), offset=511),
            ),
        ),
    ]

    write_key = key_norm.transpose(0, 2, 1, 3)
    write_value = value.transpose(0, 2, 1, 3)
    key_slot = mx.zeros_like(write_key)
    value_slot = mx.zeros_like(write_value)

    def cache_write():
        key_slot[...] = write_key
        value_slot[...] = write_value
        return key_slot, value_slot

    results.append(
        measure(
            "attention.cache_write",
            tensor_bytes(KEY_VALUE_HEADS * HEAD_DIM * 4),
            cache_write,
        )
    )
    for context in CONTEXTS:
        rope_query = attention.rope(
            query_norm.transpose(0, 2, 1, 3), offset=context - 1
        )
        cache_keys = mx.zeros(
            (1, KEY_VALUE_HEADS, context, HEAD_DIM), dtype=mx.bfloat16
        )
        cache_values = mx.zeros_like(cache_keys)
        mx.eval(rope_query, cache_keys, cache_values)
        results.append(
            measure(
                f"attention.sdpa@{context}",
                tensor_bytes(QUERY_HEADS * HEAD_DIM * 2)
                + tensor_bytes(context * KEY_VALUE_HEADS * HEAD_DIM * 2),
                lambda q=rope_query, k=cache_keys, v=cache_values: (
                    scaled_dot_product_attention(
                        q, k, v, cache=None, scale=attention.scale, mask=None
                    )
                ),
            )
        )

    attention_output = mx.zeros((1, 1, QUERY_HEADS * HEAD_DIM), dtype=mx.bfloat16)
    moe_input = layer.post_attention_layernorm(activation)
    router_logits = block.gate(moe_input)

    def top_k():
        probabilities = mx.softmax(router_logits, axis=-1, precise=True)
        indices = mx.argpartition(probabilities, kth=-TOP_K, axis=-1)[..., -TOP_K:]
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        return scores / mx.sum(scores, axis=-1, keepdims=True), indices

    scores, indices = top_k()
    routed_input = mx.expand_dims(moe_input, (-2, -3))
    up = block.switch_mlp.up_proj(routed_input, indices)
    gate = block.switch_mlp.gate_proj(routed_input, indices)
    activated = block.switch_mlp.activation(up, gate)
    down = block.switch_mlp.down_proj(activated, indices).squeeze(-2)
    mx.eval(
        attention_output,
        moe_input,
        router_logits,
        scores,
        indices,
        routed_input,
        up,
        gate,
        activated,
        down,
    )
    results.extend(
        [
            measure(
                "attention.o_projection",
                quantized_bytes(QUERY_HEADS * HEAD_DIM, HIDDEN, 4),
                lambda: attention.o_proj(attention_output),
            ),
            measure(
                "moe.norm",
                norm_bytes(HIDDEN),
                lambda: layer.post_attention_layernorm(activation),
            ),
            measure(
                "moe.router",
                quantized_bytes(HIDDEN, EXPERTS, 8),
                lambda: block.gate(moe_input),
            ),
            measure(
                "moe.top_k",
                tensor_bytes(EXPERTS + TOP_K) + TOP_K * 4,
                top_k,
            ),
            measure(
                "moe.gate_up_gather",
                2 * gather_quantized_bytes(HIDDEN, INTERMEDIATE, TOP_K),
                lambda: (
                    block.switch_mlp.up_proj(routed_input, indices),
                    block.switch_mlp.gate_proj(routed_input, indices),
                ),
            ),
            measure(
                "moe.activation",
                tensor_bytes(TOP_K * INTERMEDIATE * 3),
                lambda: block.switch_mlp.activation(up, gate),
            ),
            measure(
                "moe.down_gather",
                gather_quantized_bytes(INTERMEDIATE, HIDDEN, TOP_K),
                lambda: block.switch_mlp.down_proj(activated, indices),
            ),
            measure(
                "moe.weighted_combine",
                tensor_bytes(TOP_K * HIDDEN + TOP_K + HIDDEN),
                lambda: (down * scores[..., None]).sum(axis=-2),
            ),
        ]
    )

    final_input = model.model.norm(activation)
    logits = model.lm_head(final_input)
    mx.eval(final_input, logits)
    results.extend(
        [
            measure(
                "tail.final_norm",
                norm_bytes(HIDDEN),
                lambda: model.model.norm(activation),
            ),
            measure(
                "tail.lm_head",
                quantized_bytes(HIDDEN, VOCAB, 4),
                lambda: model.lm_head(final_input),
            ),
            measure(
                "tail.sampling",
                tensor_bytes(VOCAB) + 4,
                lambda: mx.argmax(logits, axis=-1),
            ),
        ]
    )
    return results


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
    if len(sys.argv) != 3 or sys.argv[1] not in {
        "reference",
        "benchmark",
        "profile",
    }:
        raise SystemExit(
            f"usage: {sys.argv[0]} reference|benchmark|profile MODEL"
        )
    operation = sys.argv[1]
    model_path = Path(sys.argv[2])
    if operation == "reference":
        reference(model_path)
    elif operation == "benchmark":
        benchmark(model_path)
    else:
        print(json.dumps(block_profile(model_path), indent=2))


if __name__ == "__main__":
    main()
