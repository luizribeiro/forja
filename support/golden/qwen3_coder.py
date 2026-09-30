#!/usr/bin/env python3
"""Generate Qwen3-Coder-30B-A3B-Instruct reference fixtures."""

import argparse
import json
import re
import time
from contextlib import ExitStack
from pathlib import Path

from fixture_library import GenerationSettings, generate

PROMPTS = (
    ("python", "def checked_sum(values: list[int]) -> int:\n"),
    ("rust", "fn checked_product(values: &[u64]) -> Option<u64> {\n"),
)

TENSOR_DESCRIPTIONS = {
    "prompt_token_ids": "Tokenized raw prompt, shape [prompt_tokens], I64.",
    "hidden_state_0": "Token embeddings before decoder layer 1, shape [prompt_tokens, 2048], F32.",
    "hidden_state_i": "For 1 <= i < 4, decoder layer i output before the final norm, shape [prompt_tokens, 2048], F32.",
    "hidden_state_4": "Decoder layer 4 output after the final model RMS norm, shape [prompt_tokens, 2048], F32.",
    "router_logits": "Prompt router scores before top-k, shape [4, prompt_tokens, 128], F32.",
    "prompt_last_logits": "Truncated-model logits at the last prompt position, shape [151936], F32.",
    "greedy_token_ids": "The truncated model's 32 argmax token ids, shape [32], I64.",
    "greedy_step_logits": "Truncated-model token logits, shape [32, 151936], F32.",
}

FULL_TENSOR_DESCRIPTIONS = {
    **TENSOR_DESCRIPTIONS,
    "hidden_state_i": "For 1 <= i < 48, decoder layer i output before the final norm, shape [prompt_tokens, 2048], F32.",
    "hidden_state_48": "Decoder layer 48 output after the final model RMS norm, shape [prompt_tokens, 2048], F32.",
    "router_logits": "Prompt router scores before top-k, shape [48, prompt_tokens, 128], F32.",
    "prompt_last_logits": "Model logits at the last prompt position, shape [151936], F32.",
    "greedy_token_ids": "The model's 32 argmax token ids, shape [32], I64.",
    "greedy_step_logits": "Model token logits, shape [32, 151936], F32.",
}
del FULL_TENSOR_DESCRIPTIONS["hidden_state_4"]

EXPERT_WEIGHT = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def dequantize_affine(weight, scales, biases, bits: int, group_size: int):
    """Dequantize MLX affine rows, whose first value occupies each word's low bits."""
    import torch

    values_per_word = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    values = ((weight.to(torch.int64).unsqueeze(-1) >> shifts) & ((1 << bits) - 1))
    values = values.reshape(*weight.shape[:-1], weight.shape[-1] * values_per_word)
    return (
        values.float() * scales.float().repeat_interleave(group_size, dim=-1)
        + biases.float().repeat_interleave(group_size, dim=-1)
    )


def mlx_state_dict(model_path: Path, config) -> dict[str, object]:
    """Build the truncated Transformers state dict from the checkpoint's exact bytes."""
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    import torch

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    targets = model.state_dict()
    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    sources: dict[str, list[tuple[str, int | None]]] = {}
    for target in targets:
        match = EXPERT_WEIGHT.match(target)
        if match:
            prefix, expert, projection = match.groups()
            source = f"{prefix}.switch_mlp.{projection}.weight"
            sources.setdefault(source, []).append((target, int(expert)))
        else:
            sources.setdefault(target, []).append((target, None))

    state: dict[str, object] = {}
    by_shard: dict[str, list[str]] = {}
    for source in sources:
        by_shard.setdefault(index[source], []).append(source)
    quantization = config.quantization_config
    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(
                safe_open(model_path / name, framework="pt", device="cpu")
            )
            for name in sorted(set(index.values()))
        }
        for shard_name, names in sorted(by_shard.items()):
            shard = shards[shard_name]
            for source in names:
                weight = shard.get_tensor(source)
                scale_name = source.removesuffix("weight") + "scales"
                if scale_name in index:
                    base = source.removesuffix(".weight")
                    override = quantization.get(base, {})
                    bits = override.get("bits", quantization["bits"])
                    group_size = override.get("group_size", quantization["group_size"])
                    weight = dequantize_affine(
                        weight,
                        shards[index[f"{base}.scales"]].get_tensor(f"{base}.scales"),
                        shards[index[f"{base}.biases"]].get_tensor(f"{base}.biases"),
                        bits,
                        group_size,
                    )
                else:
                    weight = weight.float()
                for target, expert in sources[source]:
                    state[target] = weight if expert is None else weight[expert].contiguous()
    return state


def initialize_model(model) -> None:
    """Materialize Qwen's derived, non-persistent RoPE buffer on CPU."""
    import torch

    rotary = model.model.rotary_emb
    inv_freq, scaling = rotary.rope_init_fn(rotary.config, torch.device("cpu"))
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq
    rotary.attention_scaling = scaling


def main() -> None:
    """Parse command-line arguments and generate fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--same-bytes",
        action="store_true",
        help="dequantize an MLX affine checkpoint before running Transformers",
    )
    args = parser.parse_args()
    started = time.monotonic()
    generate(
        args.model.resolve(),
        args.out.resolve(),
        PROMPTS,
        FULL_TENSOR_DESCRIPTIONS if args.same_bytes else TENSOR_DESCRIPTIONS,
        settings=GenerationSettings(
            num_hidden_layers=None if args.same_bytes else 4,
            reference_weights=(
                "mlx-affine-dequantized" if args.same_bytes else "native-float"
            ),
        ),
        state_dict_loader=mlx_state_dict if args.same_bytes else None,
        model_initializer=initialize_model if args.same_bytes else None,
    )
    print(f"completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
