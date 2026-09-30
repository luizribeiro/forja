#!/usr/bin/env python3
"""Generate four-layer Qwen3-Coder-30B-A3B-Instruct reference fixtures."""

import argparse
import time
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


def main() -> None:
    """Parse command-line arguments and generate fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    generate(
        args.model.resolve(),
        args.out.resolve(),
        PROMPTS,
        TENSOR_DESCRIPTIONS,
        settings=GenerationSettings(num_hidden_layers=4),
    )
    print(f"completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
